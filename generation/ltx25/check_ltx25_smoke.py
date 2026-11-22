#!/usr/bin/env python3
"""Checks of the LTX-2.5 DEV smoke outputs (CPU only; DEV windows only). Prints a PASS/FAIL list and writes JSON.

Reads every sample_outputs.json under --output-root (the smoke runs and the same-seed rerun run) and checks:
  format       vision.mp4 81 frames 10 fps 512x288 (lossless RGB, gbrp as the bench clips); with native20.mp4: 161 frames 20 fps
               and vision = native frames 0, 2, .., 160; with full1024.mp4: 161 frames 1024x576 and vision = INTER_AREA(full)[0::2];
               latents.safetensors present with the recorded sha256
  sampler      denoiser calls per continuation (distilled 8 + 3, dev 30); the history latents unchanged after denoising
               (distilled: in stage 1 and stage 2; max |diff| = 0)
  encoder      the reference's 41-frame history latents equal the first 6 latents of its 161-frame encode
  prefix       continuation frames 0..40 vs the reference's (DiffVAE noise, reported); history latents identical
  determinism  the rerun (fresh process, same seed, other transformer residency) reproduces the clip bit for bit
  memory       peak GPU memory, GPU free memory after each sample >= --keep-free-mib, host peak RSS
  timing       seconds per clip and per phase, and the TEST estimate on this GPU: 200 windows x (16 continuations + 1 reference)
  time scale   a crude indicator, not a measurement: mean |frame difference| over the lower half of the canvas per 0.1 s, for
               the last 1 s of history and the first 1 s and 1-4 s of the future, per clip, with the raw real clip and the
               reference as yardsticks. A slow-motion future shows a lower future/history ratio.

  check_ltx25_smoke.py --output-root DIR --bench BENCH_V1_DIR [--rerun-run PREFIX] [--keep-free-mib 6144] [--json OUT]
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

CW, CH = 512, 288
EXPECT_CALLS = {'distilled': [8, 3], 'dev': [30]}


def decode(path: Path, wh=(CW, CH)) -> np.ndarray:
    raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-threads', '2', '-i', str(path), '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'])
    return np.frombuffer(raw, np.uint8).reshape(-1, wh[1], wh[0], 3)


def motion(v10: np.ndarray) -> dict:
    """Mean |frame difference| over canvas rows 144..287 per 0.1 s step; vision frame 20 is t0."""
    d = np.abs(v10[1:, 144:].astype(np.int16) - v10[:-1, 144:].astype(np.int16)).mean(axis=(1, 2, 3))  # d[i] = |f[i+1] - f[i]|
    hist, fut1, fut14 = float(d[10:20].mean()), float(d[20:30].mean()), float(d[30:60].mean())
    return dict(hist_last_1s=round(hist, 3), fut_0_1s=round(fut1, 3), fut_1_4s=round(fut14, 3), ratio_fut1_hist=round(fut1 / hist, 3) if hist > 0 else None,
                ratio_fut14_hist=round(fut14 / hist, 3) if hist > 0 else None, step_19_20=round(float(d[19]), 3), step_20_21=round(float(d[20]), 3))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--output-root', type=Path, required=True)
    ap.add_argument('--bench', type=Path, required=True, help='bench-v1 directory (raw DEV canvas clips as yardstick)')
    ap.add_argument('--rerun-run', help='run-name prefix of the rerun (default ltx25d-dev-localsmoke-rerun / ltx25-dev-smoke-rerun)')
    ap.add_argument('--keep-free-mib', type=int, default=6144)
    ap.add_argument('--test-windows', type=int, default=200)
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--json', type=Path)
    args = ap.parse_args()

    recs = []
    for f in sorted(args.output_root.rglob('sample_outputs.json')):
        r = json.loads(f.read_text())
        r['_dir'] = f.parent
        recs.append(r)
    if not recs:
        raise SystemExit(f'no sample_outputs.json under {args.output_root}')
    if any(r['split'] != 'dev' for r in recs):
        raise SystemExit('this check is for DEV outputs only')
    variants = {r.get('variant', 'dev') for r in recs}
    if len(variants) != 1:
        raise SystemExit(f'mixed variants under {args.output_root}: {variants}')
    variant = variants.pop()
    rerun_prefix = args.rerun_run or ('ltx25d-dev-localsmoke-rerun' if variant == 'distilled' else 'ltx25-dev-smoke-rerun')
    checks, notes = [], dict(variant=variant)

    def check(name, ok, detail):
        checks.append(dict(check=name, ok=bool(ok), detail=detail))

    is_rerun = lambda r: r.get('run', '').startswith(rerun_prefix)  # noqa: E731
    label = lambda r: r['name'] + (' (rerun)' if is_rerun(r) else '')  # noqa: E731
    main_recs = [r for r in recs if not is_rerun(r)]
    rerun_recs = [r for r in recs if is_rerun(r)]
    check('all samples succeeded', all(r['status'] == 'success' for r in recs), {label(r): r['status'] for r in recs})
    ok_recs = [r for r in recs if r['status'] == 'success']

    import hashlib
    fmt, clips = {}, {}
    for r in ok_recs:
        vis = decode(r['_dir'] / 'vision.mp4')
        clips[(r['name'], is_rerun(r))] = vis
        f = r['files']
        good = vis.shape == (81, CH, CW, 3) and f['vision.mp4']['fps'] == 10.0 and f['vision.mp4']['pix_fmt'] == 'gbrp'
        lat = r['_dir'] / 'latents.safetensors'
        good &= 'latents.safetensors' in f and lat.is_file() and hashlib.sha256(lat.read_bytes()).hexdigest() == f['latents.safetensors']['sha256']
        if 'native20.mp4' in f:
            nat = decode(r['_dir'] / 'native20.mp4')
            good &= nat.shape == (161, CH, CW, 3) and f['native20.mp4']['fps'] == 20.0 and np.array_equal(vis, nat[0::2])
        if 'full1024.mp4' in f:
            import cv2
            full = decode(r['_dir'] / 'full1024.mp4', (1024, 576))
            good &= full.shape == (161, 576, 1024, 3) and np.array_equal(np.stack([cv2.resize(x, (CW, CH), interpolation=cv2.INTER_AREA) for x in full[0::2]]), vis)
        fmt[label(r)] = bool(good)
    check('format (vision 81 @ 10 fps at 512x288 lossless; latents present; native20 = 20 fps canvas and full1024 -> INTER_AREA = vision where kept)',
          all(fmt.values()), fmt)

    cont = [r for r in ok_recs if r['mode'] == 'continue']
    recon = [r for r in ok_recs if r['mode'] == 'recon']
    check(f'denoiser calls per continuation = {EXPECT_CALLS[variant]}', all(r['model']['denoiser_calls'] == EXPECT_CALLS[variant] for r in cont),
          {label(r): r['model']['denoiser_calls'] for r in cont})
    pin_keys = ['history_latent_pinned_max_abs'] + (['history_latent_pinned_max_abs_stage_1'] if variant == 'distilled' else [])
    check('history latents pinned through denoising (max |diff| 0' + (', both stages)' if variant == 'distilled' else ')'),
          all(r['model'][k] == 0 for r in cont for k in pin_keys), {label(r): {k: r['model'][k] for k in pin_keys} for r in cont})
    if variant == 'distilled':
        notes['upsampled_history_vs_encoded_hr_max_abs'] = {label(r): r['model']['upsampled_history_vs_encoded_hr_max_abs'] for r in cont}
    check('causal encoder: 41-frame latents = first 6 of the 161-frame encode', all(r['model']['causal_encode_hist_vs_full_max_abs'] == 0 for r in recon),
          {label(r): r['model']['causal_encode_hist_vs_full_max_abs'] for r in recon})
    pref = {r['name']: r['diagnostics'].get('prefix_vs_recon_in_process') for r in cont if not is_rerun(r)}
    check('continuation history latents identical to the reference\'s (same process)',
          bool(pref) and all(p and p['history_latent_identical'] for p in pref.values()), {k: (v or {}).get('history_latent_identical') for k, v in pref.items()})
    notes['prefix_vs_recon'] = pref
    notes['prefix_vs_input'] = {label(r): {k: v for k, v in r['diagnostics'].items() if k.startswith('prefix_vs_input')} for r in ok_recs}

    det = {}
    for rr in rerun_recs:
        orig = next((r for r in main_recs if r['name'] == rr['name']), None)
        if orig is None or rr['status'] != 'success' or orig['status'] != 'success':
            det[rr['name']] = None
            continue
        det[rr['name']] = dict(identical=orig['full_res_frames_sha256'] == rr['full_res_frames_sha256'] and
                               orig['files']['vision.mp4']['frames_sha256'] == rr['files']['vision.mp4']['frames_sha256'] and
                               orig['files']['latents.safetensors']['video_latent_sha256'] == rr['files']['latents.safetensors']['video_latent_sha256'],
                               same_seed=rr['seed'] == orig['seed'],
                               residency=[orig['settings_read_back'].get('transformer_residency'), rr['settings_read_back'].get('transformer_residency')])
        if not det[rr['name']]['identical']:
            a, b = clips[(rr['name'], False)], clips[(rr['name'], True)]
            dd = np.abs(a.astype(np.int16) - b.astype(np.int16))
            det[rr['name']].update(max_abs=int(dd.max()), mean_abs=float(dd.mean()), first_differing_vision_frame=int(np.argmax(dd.reshape(81, -1).max(1) > 0)))
    check('same-seed rerun reproduces the clip bit for bit (fresh process, other residency)',
          bool(det) and all(v and v['identical'] and v['same_seed'] for v in det.values()), det)

    mem = {label(r): dict(gpu_max_reserved_gib=r['gpu_memory_gib']['max_reserved'], gpu_max_allocated_gib=r['gpu_memory_gib']['max_allocated'],
                          gpu_free_mib_after=r.get('gpu_free_mib_after'), host_peak_rss_gib=r['host_memory_gib'].get('peak_rss'),
                          host_vmhwm_gib=r['host_memory_gib'].get('vmhwm'), host_mem_available_gib=r['host_memory_gib'].get('host_mem_available'))
           for r in ok_recs}
    check(f'GPU free memory after every sample >= {args.keep_free_mib} MiB', all((m['gpu_free_mib_after'] or 0) >= args.keep_free_mib for m in mem.values()), mem)

    main_cont = [r for r in cont if not is_rerun(r)]
    main_recon = [r for r in recon if not is_rerun(r)]
    est = None
    if main_cont and main_recon:
        c, rc = float(np.mean([r['timing_s']['total'] for r in main_cont])), float(np.mean([r['timing_s']['total'] for r in main_recon]))
        phases = sorted({k for r in main_cont for k in r['timing_s'] if k != 'total'})
        est = dict(seconds_per_continuation=round(c, 1), seconds_per_reference=round(rc, 1),
                   continuation_phases_mean={k: round(float(np.mean([r['timing_s'].get(k, 0) for r in main_cont])), 1) for k in phases},
                   residency=main_cont[0]['settings_read_back'].get('transformer_residency'), test_clips=args.test_windows * (args.k + 1),
                   test_gpu_hours_this_gpu=round((args.test_windows * args.k * c + args.test_windows * rc) / 3600, 1),
                   gpu=ok_recs[0]['gpu'].get('name'), note='excludes the one-time model construction per process (adapter-runs/*.json construct_s) and '
                   'the one-time prompt encoding; the first sample of a process may include warm-up')
    notes['timing'] = est

    ts = {}
    for r in ok_recs:
        if is_rerun(r):
            continue
        ts.setdefault(r['window'], {})[r['name'].split('-')[-1]] = motion(clips[(r['name'], False)])
    for w in ts:
        ts[w]['raw_real'] = motion(decode(args.bench / 'windows-dev' / w / 'canvas' / 'real.mp4'))
    notes['time_scale_indicator'] = ts

    for c in checks:
        print(('PASS ' if c['ok'] else 'FAIL ') + c['check'])
    print(json.dumps(dict(variant=variant, timing=est, time_scale_indicator=ts), indent=1))
    out = dict(checks=checks, all_pass=all(c['ok'] for c in checks), notes=notes)
    (args.json or args.output_root / 'smoke-check.json').write_text(json.dumps(out, indent=1, default=str) + '\n')


if __name__ == '__main__':
    main()
