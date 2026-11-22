#!/usr/bin/env python3
"""CPU-only checks with the real weights (no GPU is touched; CUDA_VISIBLE_DEVICES must be empty).

  check_strict_load.py --src SOURCE --weights WEIGHTS --report NEW.json
                       [--input-root INPUTS --split dev --roundtrip WINDOW] [--pose-probe WINDOW ...]

1. Strict loads into the released classes: TrainTransformers (models/model.py) with world_model.pth (every key and shape,
   strict=True), and the released Tokenizer class (modules/tokenizers/model_tokenizer.py) with vqvae.pt (its own strict
   load), on the CPU device. The released modules call .cuda() at construction; this CPU check maps it to a no-op.
2. --roundtrip: the released conditioning encode of a DEV window's 15 conditioning frames (+1 zero frame), decoded with
   z_to_clips_all and, for the newest frame, z_to_clips; PSNR against the input frames; encode repeated for determinism.
3. --pose-probe (diagnostic only, DEV history): the model's next-state yaw / x / y distributions after the window's
   15 conditioning frames (eval_first_second once, float32), against the last history step. No image token is sampled.
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
import types

sys.dont_write_bytecode = True
import drivingworld_common as dc  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', type=Path, required=True)
    ap.add_argument('--weights', type=Path, required=True)
    ap.add_argument('--report', type=Path, required=True)
    ap.add_argument('--input-root', type=Path)
    ap.add_argument('--split', default='dev', choices=['dev'])
    ap.add_argument('--roundtrip')
    ap.add_argument('--pose-probe', nargs='*', default=[])
    a = ap.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU check: run with CUDA_VISIBLE_DEVICES=""')
    if a.report.exists():
        raise FileExistsError('fresh report required')
    report = dict(GPU_started=False, pins_sha256=dc.sha(dc.PINS_PATH), source=dc.verify_source(a.src))
    t = time.monotonic()
    report['weights'] = dc.verify_weights(a.weights, full=True)
    report['weights_hash_seconds'] = time.monotonic() - t
    report['pickle_globals'] = {k: dc.check_pickle_globals(a.weights / k, k) for k in dc.PINS['weights']['pickle_globals_allowed']}
    import run_drivingworld_bench as rb
    report['locked_packages'] = rb.check_environment()
    sys.path.insert(0, str(a.src.resolve()))
    import numpy as np
    import torch
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '2')))
    torch.Tensor.cuda = lambda self, *x, **k: self
    torch.nn.Module.cuda = lambda self, *x, **k: self
    from models.model import TrainTransformers
    from modules.tokenizers.model_tokenizer import Tokenizer
    cfg = dc.official_config(a.src)
    cfg['vq_ckpt'] = str(a.weights / 'vqvae.pt')
    args = types.SimpleNamespace(**cfg)
    t = time.monotonic()
    tw = TrainTransformers(args, local_rank=0, condition_frames=args.condition_frames).eval()
    state = torch.load(str(a.weights / 'world_model.pth'), map_location='cpu', weights_only=True)
    container = 'model_state_dict' if 'model_state_dict' in state else 'flat'
    if container != 'flat':
        state = state['model_state_dict']
    own = tw.model.state_dict()
    missing, unexpected = sorted(set(own) - set(state)), sorted(set(state) - set(own))
    shapes = [k for k in own if k in state and tuple(own[k].shape) != tuple(state[k].shape)]
    if missing or unexpected or shapes:
        raise ValueError(f'world model mismatch: missing {missing[:8]} unexpected {unexpected[:8]} shapes {shapes[:8]}')
    res = tw.model.load_state_dict(state, strict=True)
    dtypes = sorted({str(v.dtype) for v in state.values()})
    n_params = sum(v.numel() for v in state.values())
    del state
    heads = {k: list(v.shape) for k, v in tw.model.state_dict().items() if k.endswith(('head.4.weight', 'head.3.weight', 'time_emb', 'sos_emb'))}
    report['world_model'] = dict(class_='models.model.TrainTransformers -> models.ar.GPT', container=container, keys=len(own),
                                 parameters=n_params, dtypes=dtypes, missing_keys=list(res.missing_keys),
                                 unexpected_keys=list(res.unexpected_keys), head_shapes=heads, seconds=time.monotonic() - t)
    t = time.monotonic()
    tok = Tokenizer(args, 'cpu')
    tok.eval()
    report['tokenizer'] = dict(class_='modules.tokenizers.model_tokenizer.Tokenizer (VideoVQ-16)', loader='released, strict=True',
                               keys=len(tok.vq_model.state_dict()),
                               parameters=sum(v.numel() for v in tok.vq_model.state_dict().values()), seconds=time.monotonic() - t)
    print(json.dumps(dict(strict_loads='PASS', world_model_parameters=n_params)), flush=True)

    def history(window):
        root = Path(a.input_root) / 'windows' / a.split / window
        rec = json.loads((root / 'inputs.json').read_text())
        if dc.sha(root / 'history10.mp4') != rec['files']['history10.mp4']['sha256']:
            raise ValueError('input hash mismatch')
        frames = dc.decode(root / 'history10.mp4', dc.MODEL_WH)
        poses = json.loads((root / 'past_poses.json').read_text())
        x = torch.from_numpy(frames.copy()).permute(0, 3, 1, 2).float() / 255.0
        return frames, (x - 0.5) * 2, poses

    with torch.inference_mode():
        if a.roundtrip:
            frames, x, _ = history(a.roundtrip)
            t = time.monotonic()
            imgs = torch.cat([x[None, 6:21], torch.zeros_like(x[None, :1])], dim=1)
            tokens, _ = tok.encode_to_z(imgs)
            t_enc = time.monotonic() - t
            again, _ = tok.encode_to_z(imgs)
            from einops import rearrange
            t = time.monotonic()
            dec = tok.z_to_clips_all(rearrange(tokens, 'b F (h w) -> (b F) h w', h=16))
            t_dec = time.monotonic() - t
            dec_u8 = (dec.permute(0, 2, 3, 1).numpy() * 255).astype('uint8')
            last = tok.z_to_clips(rearrange(tokens, 'b F (h w) -> (b F) h w', h=16))
            last_u8 = (last[0].permute(1, 2, 0).numpy() * 255).astype('uint8')

            def psnr(u, v):
                mse = float(np.mean((u.astype(np.float64) - v.astype(np.float64)) ** 2))
                return 99.0 if mse == 0 else 10 * math.log10(255 ** 2 / mse)
            ps = [psnr(dec_u8[i], frames[6 + i]) for i in range(15)]
            report['roundtrip'] = dict(window=a.roundtrip, frames='ticks -14..0 + zero frame', tokens_shape=list(tokens.shape),
                                       deterministic=bool(torch.equal(tokens, again)), unique_tokens=int(tokens.unique().numel()),
                                       psnr_db=[round(p, 2) for p in ps], psnr_median_db=float(np.median(ps)),
                                       newest_frame_z_to_clips_vs_all_max_abs=int(np.abs(last_u8.astype(int) - dec_u8[15].astype(int)).max()),
                                       encode_seconds=t_enc, decode_all_seconds=t_dec, threads=torch.get_num_threads())
            print(json.dumps(dict(roundtrip=a.roundtrip, psnr_median_db=report['roundtrip']['psnr_median_db'])), flush=True)
        probes = []
        if a.pose_probe:
            from modules.tokenizers.pose_tokenizer import poses_to_indices, yaws_to_indices, indices_to_pose, indices_to_yaws
            import drivingworld_rollout as dr
            gpt = tw.model
            for window in a.pose_probe:
                frames, x, poses = history(window)
                imgs = torch.cat([x[None, 6:21], torch.zeros_like(x[None, :1])], dim=1)
                tokens, _ = tok.encode_to_z(imgs)
                tokens = tokens[:, :15]
                xy = torch.tensor(poses['condition']['xy_m'], dtype=torch.float32)[None]
                yaw = torch.tensor(poses['condition']['yaw_deg'], dtype=torch.float32)[None]
                xy[:, 0] = 0
                yaw[:, 0] = 0
                flag = torch.isinf(xy[:, 0, 0])
                t = time.monotonic()
                intra = gpt.eval_first_second(tok.vq_model.quantize.embedding(tokens), poses_to_indices(xy, 128, 128),
                                              yaws_to_indices(yaw, 512), flag)
                t_fs = time.monotonic() - t
                p_yaw = torch.softmax(gpt.eval_third(intra[:, :1], drop_flag=flag)['yaw_logits'][0, 0].float(), -1)
                yaw_hat = p_yaw.argmax()[None, None]
                p_x = torch.softmax(gpt.eval_third(intra[:, :2], yaws=yaw_hat[:, :, None], drop_flag=flag)['pose_x_logits'][0, 0].float(), -1)
                x_hat = p_x.argmax()[None, None]
                p_y = torch.softmax(dr.pose_y_logits(gpt, intra[:, :3], yaw_hat[:, :, None], x_hat[:, :, None], flag)[0, 0].float(), -1)
                y_rel = torch.softmax(gpt.eval_third(intra[:, :3], yaws=yaw_hat[:, :, None], poses=x_hat[:, :, None], drop_flag=flag)['pose_y_logits'][0, 0].float(), -1)
                bins_x, bins_y = indices_to_pose(torch.arange(128), torch.arange(128))
                bins_yaw = indices_to_yaws(torch.arange(512))

                def summary(p, bins):
                    top = torch.topk(p, 3)
                    return dict(argmax_value=float(bins[p.argmax()]), mean=float((p * bins).sum()),
                                top3=[[float(bins[i]), round(float(v), 4)] for v, i in zip(top.values, top.indices)])
                ix, iy, iyaw = dc.pose_tokens_np(poses['condition']['xy_m'][-1:], [poses['condition']['yaw_deg'][-1][0]])
                probes.append(dict(window=window, eval_first_second_seconds=t_fs,
                                   history_last_step=dict(x_m=poses['condition']['xy_m'][-1][0], y_m=poses['condition']['xy_m'][-1][1],
                                                          yaw_deg=poses['condition']['yaw_deg'][-1][0], tokens_x_y_yaw=[int(ix[0]), int(iy[0]), int(iyaw[0])],
                                                          speed_mps=poses['speed_last_step_mps']),
                                   history_x_steps_m=[round(v[0], 4) for v in poses['condition']['xy_m']],
                                   predicted=dict(yaw_deg=summary(p_yaw, bins_yaw), x_m_given_argmax_yaw=summary(p_x, bins_x),
                                                  y_m_given_argmax_yaw_x_pose_y_head=summary(p_y, bins_y),
                                                  y_m_released_eval_third_pose_x_head=summary(y_rel, bins_y))))
                print(json.dumps(dict(probe=window, x_mean=probes[-1]['predicted']['x_m_given_argmax_yaw']['mean'],
                                      history_x=probes[-1]['history_last_step']['x_m'], seconds=round(t_fs, 1))), flush=True)
        report['pose_probe'] = probes
    report['torch'] = torch.__version__
    dc.new_json(a.report, report)
    print(json.dumps(dict(PASS=True, report=str(a.report))))


if __name__ == '__main__':
    main()
