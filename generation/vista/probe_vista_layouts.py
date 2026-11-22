#!/usr/bin/env python3
"""Probe of Vista's official memory layout and of batching (one GPU process; research server or any GPU).

A. Official layout mechanics: the model built without --low_vram (every sub-model resident in fp32, the UNet under
   autocast) and the released conditioner (25 condition copies at once), the adapter's rollout (3 rounds) and decode at a
   reduced size (--mech-height x --mech-width, default 320 x 576, Vista's action-training size) with --mech-steps sampling
   steps, then the official sample_utils.do_sample with the same seed: the latents must be identical. Peak memory per phase
   is recorded; the size scales the activations (not the 9.4 GiB of resident weights).
B. Batching: one CFG denoiser call (both branches) at the native 576 x 1024 for b = 1, 2, 3 samples (25 b frames per
   branch) with the official layout's resident fp32 weights under autocast; median time of --reps calls after a warm-up, and
   peak memory. A batch of b samples per call would cost t(b) / b per sample-step instead of t(1).
C. Generation size (--size, default canvas 288 x 512): one CFG denoiser call (both branches, 25 frames, b = 1) of the
   released UNet without the size adaptation (expected to fail at a latent 36 x 64: the skip concatenation), then with
   run_vista_bench.configure_size (the one upsampling to its skip size): output shape, median time of --reps calls, peak
   memory; and the conditioner's 25-copy encode of the anchor at that size (peak memory).

  probe_vista_layouts.py --vista-src DIR --ckpt FILE --input-root DIR --window W --out FILE.json [--parts A,B,C] [--size canvas]
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
    ap.add_argument('--input-root', type=Path, required=True)
    ap.add_argument('--window', default='2eeaccb0-t2.5')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--parts', default='A,B')
    ap.add_argument('--mech-height', type=int, default=320)
    ap.add_argument('--mech-width', type=int, default=576)
    ap.add_argument('--mech-steps', type=int, default=2)
    ap.add_argument('--batches', default='1,2,3')
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--layout', choices=('official', 'lowvram'), default='official',
                    help='lowvram: the UNet alone on the GPU in fp16 (part B without memory pressure on a 48 GB GPU; the compute is fp16 in both)')
    ap.add_argument('--leave-free-mib', type=float, default=2048)
    ap.add_argument('--min-free-mib', type=float, default=30000)
    ap.add_argument('--max-gpu-mib', type=float, default=None)
    ap.add_argument('--expect-gpu-uuid', default=None)
    ap.add_argument('--size', choices=tuple(rvb.SIZES), default=rvb.B['default_size'], help='part C (and the input file read)')
    a = ap.parse_args()
    import numpy as np
    import torch
    from torch import autocast
    torch.set_num_threads(2)
    res = dict(guard=rvb.gpu_guard_setup(torch, a), software=rvb.software(None))
    vm = rvb.VistaModel(a.vista_src, a.ckpt, 'size', low_vram=a.layout == 'lowvram', cond_encode_chunk=None)
    res['layout'] = a.layout
    res['model_load'] = vm.load_record
    GiB = 2 ** 30
    size = rvb.Size(a.size)
    frames, _ = rvb.decode_rgb(a.input_root / 'windows' / 'dev' / a.window / size.history)
    res['size'] = size.record()
    parts = a.parts.split(',')
    if 'A' in parts:
        from PIL import Image  # the Vista environment has no OpenCV; any resize serves a mechanics check
        anchor = np.asarray(Image.fromarray(frames[20]).resize((a.mech_width, a.mech_height), Image.BOX))
        gen = dict(n_steps=a.mech_steps, n_rounds=rvb.N_ROUNDS, cfg_scale=rvb.D['cfg_scale'], cond_aug=rvb.D['cond_aug'])
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t = time.time()
        z_ad, z0, timer, settings = vm.rollout(anchor, 12345, gen, None)
        torch.cuda.synchronize(); roll = dict(seconds=round(time.time() - t, 1), peak_allocated_gib=round(torch.cuda.max_memory_allocated() / GiB, 2),
                                              peak_reserved_gib=round(torch.cuda.max_memory_reserved() / GiB, 2), timer=timer)
        torch.cuda.reset_peak_memory_stats(); t = time.time()
        x = vm.decode_latents(z_ad[:rvb.N_DEC])
        torch.cuda.synchronize(); dec = dict(seconds=round(time.time() - t, 1), peak_allocated_gib=round(torch.cuda.max_memory_allocated() / GiB, 2))
        images, value_dict, sampler, _ = vm.prepare(anchor, 12345, gen['n_steps'], gen['n_rounds'], gen['cfg_scale'], gen['cond_aug'])
        samples, samples_z, _ = vm.su.do_sample(images, vm.model, sampler, value_dict, num_rounds=gen['n_rounds'], num_frames=rvb.N_FRAMES,
                                                force_uc_zero_embeddings=rvb.UC_KEYS, initial_cond_indices=[0])
        res['A_official_layout_mechanics'] = dict(size_hw=[a.mech_height, a.mech_width], steps=a.mech_steps, rollout=roll, decode=dec,
                                                  latents_identical_to_do_sample=bool(torch.equal(z_ad, samples_z)), decoded_shape=list(x.shape),
                                                  settings=settings, unet_dtype=vm.load_record['unet_dtype'])
        print(json.dumps(res['A_official_layout_mechanics'])[:600], flush=True)
        del z_ad, z0, x, samples, samples_z, images
        torch.cuda.empty_cache()
    if 'B' in parts:
        H, W = rvb.SIZES['native']['hw'][0] // 8, rvb.SIZES['native']['hw'][1] // 8
        if frames.shape[1:3] != (H * 8, W * 8):
            raise SystemExit('part B needs the native-size inputs (history21_1024.mp4)')
        x1 = vm.to_input([frames[20]])[0][None]
        value_dict = vm.su.init_embedder_options(vm.unique_keys)
        value_dict.update(cond_frames_without_noise=x1, cond_aug=0.0, cond_frames=x1)
        with torch.no_grad(), autocast('cuda'):
            c1, uc1 = vm.su.get_condition(vm.model, value_dict, 1, rvb.UC_KEYS, 'cuda')   # one copy (memory); repeated below

        def rep(d, n):
            return {k: (v[:1].repeat(n, *([1] * (v.dim() - 1))) if hasattr(v, 'shape') else v) for k, v in d.items()}
        sampler = vm.su.init_sampling(guider='TrianglePredictionGuider', steps=50, cfg_scale=2.5, num_frames=rvb.N_FRAMES)
        out = {}
        for b in [int(v) for v in a.batches.split(',')]:
            n = 25 * b
            c, uc = rep(c1, n), rep(uc1, n)
            x = torch.randn(n, 4, H, W, device='cuda')
            mask = torch.zeros(n, device='cuda'); mask[::25] = 1
            sigma = torch.full((n,), 30.0, device='cuda')

            def denoiser(xx, s, cond, cm):
                return vm.model.denoiser(vm.model.model, xx, s, cond, cm)
            vm.su.load_model(vm.model.model)   # (a no-op in the official layout: everything is resident)
            try:
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                times = []
                with torch.no_grad(), autocast('cuda'):
                    for r in range(a.reps + 1):
                        t = time.time()
                        _ = sampler.denoise(x, denoiser, sigma, c, mask, uc)
                        torch.cuda.synchronize()
                        if r:
                            times.append(time.time() - t)
                med = float(np.median(times))
                out[b] = dict(frames_per_call=2 * n, seconds_per_call=round(med, 3), seconds_per_sample_step=round(med / b, 3),
                              peak_allocated_gib=round(torch.cuda.max_memory_allocated() / GiB, 2), peak_reserved_gib=round(torch.cuda.max_memory_reserved() / GiB, 2))
            except torch.cuda.OutOfMemoryError as exc:
                out[b] = dict(error='OOM', detail=str(exc)[:200])
            print(b, json.dumps(out[b]), flush=True)
            del x, c, uc
            torch.cuda.empty_cache()
        if 1 in out and 'seconds_per_call' in out[1]:
            for b, v in out.items():
                if 'seconds_per_call' in v:
                    v['speedup_per_sample_vs_b1'] = round(out[1]['seconds_per_call'] / v['seconds_per_sample_step'], 3)
        res['B_batching_unet_cfg_call_576x1024'] = out
    if 'C' in parts:
        H, W = size.h // 8, size.w // 8
        x1 = vm.to_input([frames[20]])[0][None]
        value_dict = vm.su.init_embedder_options(vm.unique_keys)
        value_dict.update(cond_frames_without_noise=x1, cond_aug=0.0, cond_frames=x1)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t = time.time()
        with torch.no_grad(), autocast('cuda'):
            c, uc = vm.su.get_condition(vm.model, value_dict, rvb.N_FRAMES, rvb.UC_KEYS, 'cuda')   # the released 25-copy encode
        torch.cuda.synchronize()
        cond = dict(seconds=round(time.time() - t, 2), peak_allocated_gib=round(torch.cuda.max_memory_allocated() / GiB, 2),
                    peak_reserved_gib=round(torch.cuda.max_memory_reserved() / GiB, 2), concat_shape=list(c['concat'].shape) if 'concat' in c else None)
        sampler = vm.su.init_sampling(guider='TrianglePredictionGuider', steps=50, cfg_scale=2.5, num_frames=rvb.N_FRAMES)
        x = torch.randn(rvb.N_FRAMES, 4, H, W, device='cuda')
        mask = torch.zeros(rvb.N_FRAMES, device='cuda'); mask[0] = 1
        sigma = torch.full((rvb.N_FRAMES,), 30.0, device='cuda')

        def denoiser(xx, s_, cond_, cm):
            return vm.model.denoiser(vm.model.model, xx, s_, cond_, cm)
        out = dict(conditioner_25_copies=cond, latent_hw=[H, W], plan=str(rvb.unet_size_plan(H, W)))
        try:   # the released UNet, no size adaptation
            with torch.no_grad(), autocast('cuda'):
                y = sampler.denoise(x, denoiser, sigma, c, mask, uc)
            out['released_unet'] = dict(ok=True, shape=list(y.shape))
        except Exception as exc:  # noqa: BLE001
            out['released_unet'] = dict(ok=False, error=f'{type(exc).__name__}: {str(exc)[:300]}')
        torch.cuda.empty_cache()
        out['size_adaptation'] = vm.configure_size(size)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        times = []
        with torch.no_grad(), autocast('cuda'):
            for r in range(a.reps + 1):
                t = time.time()
                y = sampler.denoise(x, denoiser, sigma, c, mask, uc)
                torch.cuda.synchronize()
                if r:
                    times.append(time.time() - t)
        out['adapted_unet'] = dict(ok=bool(torch.isfinite(y).all()), shape=list(y.shape), seconds_per_cfg_call=round(float(np.median(times)), 3),
                                   times=[round(v, 3) for v in times], peak_allocated_gib=round(torch.cuda.max_memory_allocated() / GiB, 2),
                                   peak_reserved_gib=round(torch.cuda.max_memory_reserved() / GiB, 2))
        res[f'C_size_{size.name}'] = out
        print(json.dumps(out)[:1500], flush=True)
    rvb.write_json_atomic(a.out, res)
    print(json.dumps({k: v for k, v in res.items() if k.startswith(('A_', 'B_', 'C_'))}, indent=1)[:3000])


if __name__ == '__main__':
    main()
