#!/usr/bin/env python3
"""GPU memory of each phase of Vista's released sampler at 576 x 1024 (low_vram mode), measured on small batches that fit
a capped share of a shared GPU, then extrapolated linearly in the number of frames to the released batch sizes.

Phases (sample_utils.do_sample with --low_vram: only the active sub-model is on the GPU):
  cond_encoder   get_condition: the conditioner's VideoPredictionEmbedderWithEncoder (VAE encoder, fp32) on the 25 copies
                 of the condition frame that get_batch makes (n_samples = num_frames), plus the OpenCLIP image tower
  encode         encode_first_stage(images): VAE encoder, fp32, chunks of 14 frames
  unet           one denoiser call of the sampler: 25 frames x 2 (CFG) at 72 x 128 latents, UNet in fp16 under autocast
  decode         decode_first_stage: the temporal VAE decoder, fp32, windows of 14 latents
Each measurement: peak allocated (torch.cuda.max_memory_allocated) minus what was allocated before the call, plus the
sub-model's own weights on the GPU. It runs under the allocator cap of run_vista_bench.gpu_guard_setup.

  probe_vista_memory.py --vista-src DIR --ckpt FILE --out FILE.json [--leave-free-mib 6144] [--expect-gpu-uuid UUID]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_vista_bench as rvb  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--vista-src', type=Path, required=True)
    ap.add_argument('--ckpt', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--leave-free-mib', type=float, default=6144)
    ap.add_argument('--min-free-mib', type=float, default=7000)
    ap.add_argument('--max-gpu-mib', type=float, default=None)
    ap.add_argument('--expect-gpu-uuid', default=None)
    args = ap.parse_args()
    import torch
    torch.set_num_threads(2)
    guard = rvb.gpu_guard_setup(torch, args)
    vm = rvb.VistaModel(args.vista_src, args.ckpt, 'size', low_vram=True)
    model, su = vm.model, vm.su
    from torch import autocast
    res = dict(guard=guard, model_load=vm.load_record, phases={})
    GiB = 2 ** 30

    def measure(name, fn, module):
        torch.cuda.empty_cache()
        su.load_model(module)
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        t = time.time()
        try:
            with torch.no_grad(), autocast('cuda'):
                out = fn()
            torch.cuda.synchronize()
            rec = dict(ok=True, weights_gib=round(base / GiB, 3), extra_peak_gib=round((torch.cuda.max_memory_allocated() - base) / GiB, 3),
                       seconds=round(time.time() - t, 2))
            del out
        except torch.cuda.OutOfMemoryError as exc:
            rec = dict(ok=False, error='OOM', weights_gib=round(base / GiB, 3), peak_before_oom_gib=round(torch.cuda.max_memory_allocated() / GiB, 3),
                       detail=str(exc)[:200])
        su.unload_model(module)
        torch.cuda.empty_cache()
        res['phases'][name] = rec
        print(name, json.dumps(rec), flush=True)
        return rec

    H, W = rvb.MH, rvb.MW
    x1 = torch.zeros(1, 3, H, W, device='cuda').uniform_(-1, 1)
    enc = next(e for e in model.conditioner.embedders if type(e).__name__ == 'VideoPredictionEmbedderWithEncoder')
    clip = next(e for e in model.conditioner.embedders if type(e).__name__ == 'FrozenOpenCLIPImagePredictionEmbedder')
    for n in (1, 2, 3):
        measure(f'cond_encoder_n{n}', lambda n=n: enc(x1.repeat(n, 1, 1, 1)), enc)
    measure('cond_clip_n25', lambda: clip(x1.repeat(25, 1, 1, 1)), clip)
    for n in (1, 2, 3):
        measure(f'encode_n{n}', lambda n=n: model.encode_first_stage(x1.repeat(n, 1, 1, 1)), model.first_stage_model)
    z = torch.randn(6, 4, H // 8, W // 8, device='cuda')
    for n in (4, 5, 6):  # decode_first_stage (fp32 inside, one window of n <= 14 latents)
        measure(f'decode_n{n}', lambda n=n: model.decode_first_stage(z[:n]), model.first_stage_model)
    # one denoiser call with the real shapes (CFG doubles the batch): the c / uc dicts as get_condition makes them
    value_dict = su.init_embedder_options(vm.unique_keys)
    value_dict['cond_frames_without_noise'] = x1
    value_dict['cond_aug'] = 0.0
    value_dict['cond_frames'] = x1
    probe_c = None
    try:
        torch.cuda.empty_cache()
        with torch.no_grad(), autocast('cuda'):
            # conditioning on 2 copies only (memory); the UNet sees c / uc repeated to 25 frames exactly as get_condition returns them
            c, uc = su.get_condition(model, value_dict, 2, rvb.UC_KEYS, 'cuda')
        probe_c = {k: (v.shape if hasattr(v, 'shape') else v) for k, v in c.items()}
        for k in c:
            if hasattr(c[k], 'shape') and c[k].shape[0] == 2:
                c[k] = c[k][:1].repeat(25, *([1] * (c[k].dim() - 1)))
                uc[k] = uc[k][:1].repeat(25, *([1] * (uc[k].dim() - 1)))
        sampler = su.init_sampling(guider='TrianglePredictionGuider', steps=50, cfg_scale=2.5, num_frames=25)
        x = torch.randn(25, 4, H // 8, W // 8, device='cuda')
        mask = torch.zeros(25, device='cuda')
        mask[0] = 1
        sigma = torch.full((25,), 700.0, device='cuda')

        def call():
            def denoiser(xx, s, cond, cm):
                return model.denoiser(model.model, xx, s, cond, cm)
            return sampler.denoise(x, denoiser, sigma, c, mask, uc)
        def call_nocfg():  # one branch only (25 frames): the UNet activations of half a CFG call
            return model.denoiser(model.model, x, sigma, c, mask)
        su.load_model(model.denoiser)
        measure('unet_nocfg_25', call_nocfg, model.model)
        measure('unet_cfg_25x2', call, model.model)
    except Exception as exc:  # noqa: BLE001
        res['phases']['unet_cfg_25x2'] = dict(ok=False, error=repr(exc)[:300])
    res['condition_shapes'] = {k: str(v) for k, v in (probe_c or {}).items()}

    def fit(prefix, ns, target):
        pts = [(n, res['phases'][f'{prefix}_n{n}']['extra_peak_gib']) for n in ns if res['phases'].get(f'{prefix}_n{n}', {}).get('ok')]
        if len(pts) < 2:
            return None
        (n0, p0), (n1, p1) = pts[0], pts[-1]
        b = (p1 - p0) / (n1 - n0)
        a = p0 - b * n0
        w = res['phases'][f'{prefix}_n{pts[0][0]}']['weights_gib']
        return dict(points=pts, per_frame_gib=round(b, 3), intercept_gib=round(a, 3), n=target, predicted_extra_gib=round(a + b * target, 2),
                    weights_gib=w, predicted_total_gib=round(a + b * target + w, 2))
    res['extrapolated'] = dict(cond_encoder_25=fit('cond_encoder', (1, 2, 3), 25), encode_14=fit('encode', (1, 2, 3), 14),
                               decode_14=fit('decode', (4, 5, 6), 14))
    tot = [v['predicted_total_gib'] for v in res['extrapolated'].values() if v]
    u, u1 = res['phases'].get('unet_cfg_25x2', {}), res['phases'].get('unet_nocfg_25', {})
    if u.get('ok'):
        tot.append(u['weights_gib'] + u['extra_peak_gib'])
    elif u1.get('ok'):
        res['extrapolated']['unet_cfg_25x2'] = dict(from_='unet_nocfg_25 x 2', predicted_total_gib=round(u1['weights_gib'] + 2 * u1['extra_peak_gib'], 2))
        tot.append(res['extrapolated']['unet_cfg_25x2']['predicted_total_gib'])
    res['predicted_peak_gib'] = round(max(tot), 2) if tot else None
    res['note'] = ('allocator peaks; nvidia-smi shows about 0.5-0.8 GiB more (CUDA context, cuDNN / cuBLAS handles). The phases run one at '
                   'a time under --low_vram, so the process peak is the largest phase plus the few tensors kept across phases.')
    rvb.write_json_atomic(args.out, res)
    print(json.dumps(dict(extrapolated=res['extrapolated'], predicted_peak_gib=res['predicted_peak_gib']), indent=1))


if __name__ == '__main__':
    main()
