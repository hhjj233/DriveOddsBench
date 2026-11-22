#!/usr/bin/env python3
"""DrivingWorld bench runner: load the released model once, then run every spec, one batch per window.

  run_drivingworld_bench.py --specs SPEC... --bench BENCH --input-root INPUTS --output-root OUT --src SOURCE
                            --weights WEIGHTS [--frozen-file RECORD] [--equivalence-check] [--dry-run]

Per window: the 15 conditioning frames (ticks -14..0) are tokenised once with the released Tokenizer.encode_to_z
(15 frames + 1 zero frame, as the demos); the window's continuation specs run as one batch through
drivingworld_rollout.rollout (60 next states, each yaw -> x -> y -> 512 image tokens, per-sample generators); the
reference spec round-trips the real future through the same tokenizer. Outputs, never overwritten:
<OUT>/<run>/native/<window>-s<k>/ (or -recon/): vision.mp4 (81 x 512x288, 10 fps, lossless), native10.mp4 (81 x 512x256),
tokens.npz, pred_poses.json (continuations), sample_outputs.json. A failure leaves failure.json and every partial file.
--equivalence-check first runs one batch-1 frame both through the released generate_gt_pose_gt_yaw and through the kit
loop (given poses, same seed) and requires identical tokens (written to <OUT>/<run>/equivalence-check-<window>.json).
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import types

sys.dont_write_bytecode = True
import drivingworld_common as dc  # noqa: E402
import make_drivingworld_specs as ms  # noqa: E402


def local_seed(window, k):
    return int.from_bytes(hashlib.sha256(f'bench-v1|{window}|{k}'.encode()).digest()[:4], 'big') & 0x7fffffff


def validate_spec(spec, bench, input_root, frozen_file=None):
    if spec.get('schema') != 'wm-calib-bench.drivingworld.spec.v1' or spec.get('model') != 'drivingworld':
        raise ValueError('DrivingWorld spec required')
    split, mode = spec['split'], spec['mode']
    if mode not in ('continue', 'recon'):
        raise ValueError('unregistered mode')
    if split == 'test':
        expected = dc.B['test_generation_freeze'] if mode == 'continue' else dc.B['test_measurement_freeze']
        if not frozen_file or Path(frozen_file).name != expected:
            raise ValueError('correct TEST generation/measurement record required')
        dc.verify_freeze(frozen_file, required=[Path(__file__), dc.PINS_PATH, dc.HERE / 'drivingworld_rollout.py'])
    diagnostic = spec.get('diagnostic', False)
    if diagnostic and (split != 'dev' or not spec['run'].startswith('drivingworld-dev-diag-')):
        raise ValueError('diagnostic specs are DEV only and run as drivingworld-dev-diag-*')
    dc.guard_names(bench, split, [spec['window']], diagnostic)
    if spec['settings'] != dc.settings() or spec['band'] != dc.band(spec['settings']['band']):
        raise ValueError('spec settings/band changed')
    if spec['generator'] != ms.kit_hashes():
        raise ValueError('spec code provenance changed')
    if mode == 'continue' and (spec['seed'] != local_seed(spec['window'], spec['sample']) or spec['sample'] not in range(16)):
        raise ValueError('seed rule changed')
    if mode == 'recon' and 'real10.mp4' not in spec['inputs']:
        raise ValueError('reference needs real10.mp4')
    for row in spec['inputs'].values():
        if dc.sha(dc.safe_relative(input_root, row['path'])) != row['sha256']:
            raise ValueError('input hash mismatch')
    return spec


def check_environment():
    pins = re.findall(r'^([A-Za-z0-9_.-]+)==([^\s\\]+)', (dc.HERE / 'requirements-drivingworld.lock.txt').read_text(), re.M)
    for package, version in pins:
        if importlib.metadata.version(package) != version:
            raise ValueError('inference environment differs from the exact lock: ' + package)
    return len(pins)


def gpu_record(torch):
    p = torch.cuda.get_device_properties(0)
    rec = dict(name=p.name, total_MiB=p.total_memory / 2**20, capability=f'{p.major}.{p.minor}',
               CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'), torch_cuda=torch.version.cuda,
               cudnn=torch.backends.cudnn.version())
    visible = (os.environ.get('CUDA_VISIBLE_DEVICES') or '').split(',')[0].strip()
    try:
        query = ['nvidia-smi', '--query-gpu=uuid,name,driver_version,pci.bus_id', '--format=csv,noheader']
        rec['nvidia_smi_device0'] = subprocess.run(query + (['--id=' + visible] if visible else []), capture_output=True,
                                                   text=True, timeout=30).stdout.strip().splitlines() if visible else None
    except (OSError, subprocess.SubprocessError):
        rec['nvidia_smi_device0'] = None
    return rec


def load_official(src, weights):
    """Released classes, released config values, strict state-dict loads; returns (torch, mods, tw, tokenizer, record)."""
    dc.verify_source(src)
    dc.verify_weights(weights, full=True)
    globals_used = {k: dc.check_pickle_globals(Path(weights) / k, k) for k in dc.PINS['weights']['pickle_globals_allowed']}
    locked = check_environment()
    if any(m in sys.modules for m in ('models', 'modules', 'utils')):
        raise RuntimeError('clean process required')
    sys.path.insert(0, str(Path(src).resolve()))
    import torch
    from models.model import TrainTransformers
    from modules.tokenizers.model_tokenizer import Tokenizer
    import modules.tokenizers.pose_tokenizer as pose_tokenizer
    import utils.generat_utils as generat_utils
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    cfg = dc.official_config(src)
    cfg['vq_ckpt'] = str(Path(weights) / 'vqvae.pt')
    args = types.SimpleNamespace(**cfg)
    tw = TrainTransformers(args, local_rank=0, condition_frames=args.condition_frames)
    # models.ar is imported by TrainTransformers itself (importlib, 'models.' + gpt_type), so check provenance after it.
    for name, mod in [('models.model', sys.modules['models.model']), ('models.ar', sys.modules['models.ar']),
                      ('modules.tokenizers.model_tokenizer', sys.modules['modules.tokenizers.model_tokenizer'])]:
        rel = name.replace('.', '/') + '.py'
        if Path(mod.__file__).resolve() != (Path(src) / rel).resolve():
            raise RuntimeError('official module imported from the wrong place: ' + name)
    state = torch.load(str(Path(weights) / 'world_model.pth'), map_location='cpu', weights_only=True)
    if 'model_state_dict' in state:
        state = state['model_state_dict']
    own = tw.model.state_dict()
    missing, unexpected = sorted(set(own) - set(state)), sorted(set(state) - set(own))
    shapes = [k for k in own if k in state and tuple(own[k].shape) != tuple(state[k].shape)]
    if missing or unexpected or shapes:
        raise ValueError(f'world model state dict mismatch: missing {missing[:5]} unexpected {unexpected[:5]} shapes {shapes[:5]}')
    loaded = tw.model.load_state_dict(state, strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise ValueError('strict load failed')
    n_params = sum(v.numel() for v in state.values())
    del state
    tw.eval().cuda()
    tokenizer = Tokenizer(args, 0)        # released: VideoVQ-16, .to(0), torch.load + load_state_dict(strict=True)
    tokenizer.eval()
    record = dict(strict_state_dict_loads=[dict(component='world_model', keys=len(own), parameters=n_params, missing_keys=[], unexpected_keys=[]),
                                           dict(component='vqvae', keys=len(tokenizer.vq_model.state_dict()), loader='released Tokenizer (strict=True)')],
                  pickle_globals=globals_used, locked_packages=locked, config=cfg)
    mods = dict(pose_tokenizer=pose_tokenizer, generat_utils=generat_utils)
    return torch, mods, tw, tokenizer, record


def frames_to_tensor(torch, frames, device):
    x = torch.from_numpy(frames.copy()).permute(0, 3, 1, 2).float() / 255.0
    return ((x - 0.5) * 2).to(device)


def to_uint8(images):
    """The demos' rule on Tokenizer outputs (already clip(x/2+0.5, 0, 1)): (x * 255).astype(uint8), truncation."""
    return (images.permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')


def encode_window(torch, tokenizer, real15):
    """Released conditioning encode: 15 frames + one zero frame -> tokens (1, 16, 512)."""
    imgs = torch.cat([real15[None], torch.zeros_like(real15[None, :1])], dim=1)
    tokens, _ = tokenizer.encode_to_z(imgs)
    return tokens


def prefix_frames(torch, tokenizer, hist):
    """Shared canvas prefix, ticks -20..0 (21 frames), and the conditioning tokens (1, 15, 512)."""
    from einops import rearrange
    tokens_a = encode_window(torch, tokenizer, hist[6:21])   # ticks -14..0 (the model's conditioning frames)
    tokens_b = encode_window(torch, tokenizer, hist[0:15])   # ticks -20..-6 (codec only; the model never reads them)
    dec_a = to_uint8(tokenizer.z_to_clips_all(rearrange(tokens_a, 'b F (h w) -> (b F) h w', h=16)))
    dec_b = to_uint8(tokenizer.z_to_clips_all(rearrange(tokens_b, 'b F (h w) -> (b F) h w', h=16)))
    import numpy as np
    prefix = np.concatenate([dec_b[0:6], dec_a[0:15]], axis=0)
    return prefix, tokens_a, tokens_b


def write_sample(dest, spec, specfile, native, geometry, extra_files, record):
    files = {}
    files['native10.mp4'] = dc.write_checked(native, dest / 'native10.mp4', 10)
    files['vision.mp4'] = dc.write_checked(dc.to_canvas(native, geometry), dest / 'vision.mp4', 10)
    for name in extra_files:
        files[name] = dict(sha256=dc.sha(dest / name), bytes=(dest / name).stat().st_size)
    record.update(files=files, video_sha256=files['vision.mp4']['sha256'], spec_sha256=dc.sha(specfile))
    dc.new_json(dest / 'sample_outputs.json', record)


def run_window(torch, mods, tw, tokenizer, items, a, common_record, equivalence):
    import numpy as np
    from einops import rearrange
    from drivingworld_rollout import rollout, decode_rows, integrate, sample_next_state
    device = torch.device('cuda')
    window = items[0][1]['window']
    split = items[0][1]['split']
    spec0 = items[0][1]
    geometry = spec0['band']
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    hist_u8 = dc.decode(dc.safe_relative(a.input_root, spec0['inputs']['history10.mp4']['path']), dc.MODEL_WH)
    poses = json.loads(dc.safe_relative(a.input_root, spec0['inputs']['past_poses.json']['path']).read_text())
    if len(hist_u8) != 21 or poses['condition']['ticks'] != dc.CONDITION_K:
        raise ValueError('history inputs malformed')
    cond_xy = np.array(poses['condition']['xy_m'], np.float32)
    cond_yaw = np.array(poses['condition']['yaw_deg'], np.float32)
    with torch.inference_mode():
        hist = frames_to_tensor(torch, hist_u8, device)
        prefix, tokens_a, tokens_b = prefix_frames(torch, tokenizer, hist)
        cond_tokens = tokens_a[:, :15]
        prefix_sha = hashlib.sha256(prefix.tobytes()).hexdigest()
        t_prefix = time.monotonic() - started
        if equivalence:
            path = Path(a.output_root) / spec0['run'] / f'equivalence-check-{window}.json'
            seed = local_seed(window, 'equivalence')
            last = (np.array(dc.pose_tokens_np(cond_xy[-1:], cond_yaw[-1:, 0])).T)[0]    # (x, y, yaw) tokens
            from modules.tokenizers.pose_tokenizer import indices_to_pose, indices_to_yaws
            gx, gy = indices_to_pose(torch.tensor([[int(last[0])]], device=device), torch.tensor([[int(last[1])]], device=device))
            gyaw = indices_to_yaws(torch.tensor([[int(last[2])]], device=device))
            pose_win = torch.as_tensor(cond_xy, device=device)[None].clone()
            yaw_win = torch.as_tensor(cond_yaw, device=device)[None].clone()
            pose_win[:, 0] = 0
            yaw_win[:, 0] = 0
            codebook = tokenizer.vq_model.quantize.embedding
            feature = codebook(cond_tokens)
            t1 = time.monotonic()
            torch.manual_seed(seed)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                released = tw.generate_gt_pose_gt_yaw(cond_tokens, feature, pose_win, yaw_win, torch.stack([gx, gy], -1).float(),
                                                      gyaw[:, :, None].float(), codebook, sampling_mtd='top_k', temperature_k=1.0,
                                                      top_k=30, temperature_p=1.0, top_p=0.8)
            torch.cuda.synchronize()
            t2 = time.monotonic()
            gen = [torch.Generator(device=device).manual_seed(seed)]
            given = (torch.tensor([[int(last[2])]], device=device), torch.tensor([[int(last[0])]], device=device),
                     torch.tensor([[int(last[1])]], device=device))
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                kit = sample_next_state(tw, feature, pose_win, yaw_win, codebook, gen, spec0['settings']['pose_sampler'],
                                        spec0['settings']['image_sampler'], given=given)[3]
            torch.cuda.synchronize()
            t3 = time.monotonic()
            same = bool(torch.equal(released.cpu(), kit.cpu()))
            dc.new_json(path, dict(window=window, seed=seed, given_tokens_xyyaw=[int(v) for v in last], identical_tokens=same,
                                   released_seconds=t2 - t1, kit_seconds=t3 - t2, gpu=gpu_record(torch)))
            if not same:
                raise ValueError('kit image-token loop differs from the released generate_gt_pose_gt_yaw')
        cont = sorted([x for x in items if x[1]['mode'] == 'continue'], key=lambda x: x[1]['sample'])
        recon = [x for x in items if x[1]['mode'] == 'recon']
        dests = {}
        for _, spec in cont + recon:
            dest = dc.safe_relative(a.output_root, spec['output_dir'])
            if dest.exists():
                raise FileExistsError('generated sample already exists; no overwrite or automatic retry')
            dest.mkdir(parents=True, exist_ok=False)
            dests[spec['name']] = dest
        if cont:
            seeds = [s['seed'] for _, s in cont]
            stamps = []

            def on_frame(t):
                torch.cuda.synchronize()
                stamps.append(time.monotonic())

            t_gen = time.monotonic()
            mode = spec0['settings']['pose_mode']
            if mode == 'released_long_term_last_step':      # released test_long_term_demo.py: last history step repeated
                from modules.tokenizers.pose_tokenizer import poses_to_indices, yaws_to_indices   # released tokenizers
                pi = poses_to_indices(torch.as_tensor(cond_xy[-1:], device=device)[None], tw.pose_x_vocab_size, tw.pose_y_vocab_size)
                yi = yaws_to_indices(torch.as_tensor(cond_yaw[-1:], device=device)[None], tw.yaw_vocab_size)
                lx, ly, lyaw = int(pi[0, 0, 0]), int(pi[0, 0, 1]), int(yi[0, 0, 0])
                rep = lambda v: torch.full((len(seeds), len(dc.FUTURE_K)), v, dtype=torch.long, device=device)
                given_poses = (rep(lyaw), rep(lx), rep(ly))
            elif mode == 'own':
                given_poses = None
            else:
                raise ValueError('unregistered pose mode ' + str(mode))
            out = rollout(tw, tokenizer, cond_tokens, cond_xy, cond_yaw, seeds, len(dc.FUTURE_K),
                          spec0['settings']['pose_sampler'], spec0['settings']['image_sampler'], on_frame=on_frame,
                          given_poses=given_poses)
            gen_seconds = time.monotonic() - t_gen
            steps = np.diff([t_gen] + stamps)
            peak = dict(peak_allocated_MiB=torch.cuda.max_memory_allocated() / 2**20, peak_reserved_MiB=torch.cuda.max_memory_reserved() / 2**20)
            for row, (specfile, spec) in enumerate(cont):
                dest = dests[spec['name']]
                native = np.concatenate([prefix, out['frames'][row]], axis=0)
                np.savez_compressed(dest / 'tokens.npz', cond_tokens_a=tokens_a[0].cpu().numpy().astype(np.int16),
                                    cond_tokens_b=tokens_b[0].cpu().numpy().astype(np.int16),
                                    generated_tokens=out['tokens'][row].astype(np.int16),
                                    pose_tokens_yaw_x_y=out['pose_tokens'][row].astype(np.int16))
                traj = integrate(out['pose_values'][row])
                dc.new_json(dest / 'pred_poses.json', dict(
                    frame='ego frame at t0 (x forward m, y left m, heading deg, left positive); step t = motion from frame t-1 to t',
                    rate_hz=10, ticks=dc.FUTURE_K, tokens_yaw_x_y=out['pose_tokens'][row].tolist(),
                    rel_yaw_deg_x_m_y_m=out['pose_values'][row].tolist(), integrated_x_y_heading=traj.tolist(),
                    condition_xy_m=cond_xy.tolist(), condition_yaw_deg=cond_yaw.tolist(),
                    decoder='released indices_to_pose / indices_to_yaws (bin centres)'))
                record = dict(common_record, schema='wm-calib-bench.drivingworld.output.v1', status='success', mode='continue',
                              run=spec['run'], name=spec['name'], window=window, split=split, sample=spec['sample'], seed=spec['seed'],
                              settings=spec['settings'], band=geometry, inputs=spec['inputs'], frames=81, native_frames=81,
                              prefix_rule=dc.B['prefix_rule'], prefix_frames_sha256=prefix_sha, timeline='canvas frame i = tick i-20 = native10 frame i',
                              batch=dict(window=window, size=len(cont), row=row, members=[s['name'] for _, s in cont]),
                              seconds=dict(window_prefix=t_prefix, batch_generation=gen_seconds, per_sample_share=gen_seconds / len(cont),
                                           frame_step_median=float(np.median(steps)), frame_step_max=float(np.max(steps))),
                              gpu_memory=peak, pred_traj_end_xy_m=traj[-1, :2].tolist())
                write_sample(dest, spec, specfile, native, geometry, ['tokens.npz', 'pred_poses.json'], record)
                print(json.dumps(dict(name=spec['name'], status='success')), flush=True)
        for specfile, spec in recon:
            dest = dests[spec['name']]
            t_ref = time.monotonic()
            real = dc.decode(dc.safe_relative(a.input_root, spec['inputs']['real10.mp4']['path']), dc.MODEL_WH)
            if len(real) != 81 or not np.array_equal(real[:21], hist_u8):
                raise ValueError('reference input does not extend the history')
            real_t = frames_to_tensor(torch, real, device)
            ref_tokens = []
            for t in dc.FUTURE_K:                              # tick t is real index t + 20; window ticks t-14..t
                tok = encode_window(torch, tokenizer, real_t[t + 6:t + 21])
                ref_tokens.append(tok[:, 14])
            seq = torch.cat([cond_tokens, torch.stack(ref_tokens, dim=1)], dim=1)    # ticks -14..60, tick k at index k + 14
            frames = []
            for t in dc.FUTURE_K:
                new = rearrange(seq[:, t + 14], 'b (h w) -> b h w', h=16)
                frames.append(decode_rows(tokenizer, seq[:, t - 1:t + 14], new)[0])
            native = np.concatenate([prefix, np.stack(frames)], axis=0)
            np.savez_compressed(dest / 'tokens.npz', cond_tokens_a=tokens_a[0].cpu().numpy().astype(np.int16),
                                cond_tokens_b=tokens_b[0].cpu().numpy().astype(np.int16),
                                reference_tokens=seq[0, 15:].cpu().numpy().astype(np.int16))
            record = dict(common_record, schema='wm-calib-bench.drivingworld.output.v1', status='success', mode='recon',
                          run=spec['run'], name=spec['name'], window=window, split=split, sample=None, seed=None,
                          settings=spec['settings'], band=geometry, inputs=spec['inputs'], frames=81, native_frames=81,
                          prefix_rule=dc.B['prefix_rule'], reference_rule=dc.B['reference_rule'], prefix_frames_sha256=prefix_sha,
                          timeline='canvas frame i = tick i-20 = native10 frame i',
                          seconds=dict(window_prefix=t_prefix, reference=time.monotonic() - t_ref),
                          gpu_memory=dict(peak_allocated_MiB=torch.cuda.max_memory_allocated() / 2**20,
                                          peak_reserved_MiB=torch.cuda.max_memory_reserved() / 2**20))
            write_sample(dest, spec, specfile, native, geometry, ['tokens.npz'], record)
            print(json.dumps(dict(name=spec['name'], status='success')), flush=True)
    return time.monotonic() - started


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--specs', type=Path, nargs='+', required=True)
    ap.add_argument('--bench', type=Path, required=True)
    ap.add_argument('--input-root', type=Path, required=True)
    ap.add_argument('--output-root', type=Path, required=True)
    ap.add_argument('--src', type=Path, required=True)
    ap.add_argument('--weights', type=Path, required=True)
    ap.add_argument('--frozen-file', type=Path)
    ap.add_argument('--equivalence-check', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    groups = OrderedDict()
    for path in a.specs:
        spec = validate_spec(json.loads(path.read_text()), a.bench, a.input_root, a.frozen_file)
        groups.setdefault((spec['split'], spec['window']), []).append((path, spec))
    for key, items in groups.items():
        if len({s['name'] for _, s in items}) != len(items) or sum(s['mode'] == 'continue' for _, s in items) > 16:
            raise ValueError('duplicate specs or more than 16 samples in one window batch')
        if len({json.dumps(s['inputs']['history10.mp4']) for _, s in items}) != 1:
            raise ValueError('one window, one input build')
    dc.verify_source(a.src)
    if a.dry_run:
        print(json.dumps(dict(CPU_spec_and_source_hashes='PASS', specs=len(a.specs), windows=len(groups), GPU_started=False)))
        return
    if os.environ.get('CUDA_VISIBLE_DEVICES') is None:
        raise ValueError('launch through the approved GPU wrapper (CUDA_VISIBLE_DEVICES must be set)')
    if not (a.weights / 'download-verified.json').is_file():
        raise ValueError('owner-approved pinned weight download proof required')
    t_load = time.monotonic()
    torch, mods, tw, tokenizer, load_record = load_official(a.src, a.weights)
    common = dict(code_commit=dc.PINS['code']['commit'], weight_revision=dc.PINS['weights']['revision'],
                  model_load_seconds=time.monotonic() - t_load, gpu=gpu_record(torch),
                  software=dict(torch=torch.__version__, python=sys.version.split()[0]),
                  attention_backend=dict(flash=torch.backends.cuda.flash_sdp_enabled(), mem_efficient=torch.backends.cuda.mem_efficient_sdp_enabled(),
                                         math=torch.backends.cuda.math_sdp_enabled(), cudnn_deterministic=torch.backends.cudnn.deterministic,
                                         cudnn_benchmark=torch.backends.cudnn.benchmark, cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                                         matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32),
                  strict_state_dict_loads=load_record['strict_state_dict_loads'], pickle_globals=load_record['pickle_globals'],
                  runner_sha256=dc.sha(Path(__file__)), rollout_sha256=dc.sha(dc.HERE / 'drivingworld_rollout.py'))
    print(json.dumps(dict(loaded=True, seconds=common['model_load_seconds'], gpu=common['gpu']['name'])), flush=True)
    for (split, window), items in groups.items():
        try:
            seconds = run_window(torch, mods, tw, tokenizer, items, a, common, a.equivalence_check)
        except BaseException as error:
            for _, spec in items:
                dest = dc.safe_relative(a.output_root, spec['output_dir'])
                if dest.is_dir() and not (dest / 'sample_outputs.json').exists() and not (dest / 'failure.json').exists():
                    dc.new_json(dest / 'failure.json', dict(error_type=type(error).__name__, error=str(error)[:2000], failed=True,
                                                            no_retry=True, all_partial_outputs_retained=True))
            raise
        print(json.dumps(dict(window=window, specs=len(items), seconds=round(seconds, 1))), flush=True)


if __name__ == '__main__':
    main()
