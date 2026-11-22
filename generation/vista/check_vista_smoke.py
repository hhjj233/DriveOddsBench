#!/usr/bin/env python3
"""Checks of the local Vista DEV smoke (CPU): format, prefix identity, settings, seeding, determinism, cost; contact sheet.

  check_vista_smoke.py --out-root SMOKE_OUT --rerun-root RERUN_OUT --specs SPEC_DIR [--rerun-specs DIR] --result FILE.json
                       [--sheet FILE.png] [--extra-root DIR ...]

PASS / FAIL items:
  1 complete       every spec has status success
  2 format         vision.mp4: 81 frames, 512 x 288, 10 fps, lossless rgb (libx264rgb); prefix.mp4 20 frames
  3 prefix         frames 0-19 of every clip of a window equal its prefix.mp4 (frame hashes)
  4 settings       read back: EulerEDMSampler 50 steps, EDM sigma_max 700 / sigma_min 0.002 / rho 7, TrianglePredictionGuider
                   1.0 -> 2.5 -> 1.0, 3 rounds, 69 latents, 61 decoded, low_vram (UNet fp16)
  5 seeding        h0 and t2 sample k of a window share seed_of(w, k) and the anchor latent (same posterior draw); different
                   k differ
  6 actions        t2 samples carry 3 rounds of 8 numbers; an h0 / t2 pair with the same seed differs after t0
  7 determinism    the rerun (a second process) reproduces prefix, recon, h0 s0 and t2 s0 bit for bit (frames and latents)
  8 cost           seconds per continuation / reference, peak GPU memory per phase, host peak (reported; no threshold)
A contact sheet (local only: AV2 RGB without face / plate blurring) shows, per window, rows recon / h0 s0 / h0 s1 / t2 s0 /
t2 s1 at t = -1.0, 0.0, +0.5, +1, +2, +3, +4, +6 s, and the boundary-step statistic of the coordinator's continuation-break
check (|frame 21 - frame 20| over the median history step).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

FRAMES_T = (-1.0, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0)


def ffprobe(path: Path) -> dict:
    out = subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames', '-show_entries',
                                   'stream=width,height,nb_read_frames,r_frame_rate,pix_fmt,codec_name', '-of', 'json', str(path)], text=True)
    s = json.loads(out)['streams'][0]
    num, den = s['r_frame_rate'].split('/')
    return dict(width=int(s['width']), height=int(s['height']), frames=int(s['nb_read_frames']), fps=float(num) / float(den), pix_fmt=s.get('pix_fmt'),
                codec=s.get('codec_name'))


def decode(path: Path):
    info = ffprobe(path)
    raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-threads', '2', '-i', str(path), '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'])
    return np.frombuffer(raw, np.uint8).reshape(-1, info['height'], info['width'], 3)


def specs_in(d: Path) -> list:
    out = []
    for f in sorted(Path(d).rglob('*.json')):
        s = json.loads(f.read_text())
        if s.get('schema') == 'wm-calib-bench.vista.spec.v1':
            out.append(s)
    return out


def rec_of(root: Path, s: dict) -> dict | None:
    f = root / s['output_dir'] / 'sample_outputs.json'
    return json.loads(f.read_text()) if f.is_file() else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out-root', type=Path, required=True)
    ap.add_argument('--rerun-root', type=Path, default=None)
    ap.add_argument('--specs', type=Path, required=True)
    ap.add_argument('--rerun-specs', type=Path, default=None)
    ap.add_argument('--result', type=Path, required=True)
    ap.add_argument('--sheet', type=Path, default=None)
    args = ap.parse_args()
    specs = specs_in(args.specs)
    items, info = {}, {}
    recs = {(s['run'], s['name']): rec_of(args.out_root, s) for s in specs}
    missing = [k for k, r in recs.items() if not r or r.get('status') != 'success']
    items['1 complete'] = dict(ok=not missing, detail=dict(specs=len(specs), not_success=missing[:6]))
    fmt_bad, pre_bad = [], []
    clips = {}
    for s in specs:
        r = recs[(s['run'], s['name'])]
        if not r or r.get('status') != 'success':
            continue
        d = args.out_root / s['output_dir']
        if s['mode'] == 'prefix':
            p = ffprobe(d / 'prefix.mp4')
            if (p['frames'], p['width'], p['height']) != (20, 512, 288):
                fmt_bad.append((s['name'], p))
            clips[(s['window'], 'prefix')] = decode(d / 'prefix.mp4')
            continue
        p = ffprobe(d / 'vision.mp4')
        if (p['frames'], p['width'], p['height'], p['fps'], p['pix_fmt']) != (81, 512, 288, 10.0, 'gbrp') or p['codec'] != 'h264':
            fmt_bad.append((s['name'], p))
        tag = 'recon' if s['mode'] == 'recon' else f"{s['track']}-s{s['k']}" + ('ta' if s['generation'].get('action_variant') == 'time_aligned' else '')
        clips[(s['window'], tag)] = decode(d / 'vision.mp4')
    items['2 format'] = dict(ok=not fmt_bad, detail=fmt_bad[:4])
    for (w, tag), fr in clips.items():
        if tag == 'prefix':
            continue
        pre = clips.get((w, 'prefix'))
        if pre is None or not np.array_equal(fr[:20], pre):
            pre_bad.append((w, tag))
    items['3 prefix'] = dict(ok=not pre_bad, detail=pre_bad)
    set_bad = []
    for s in specs:
        r = recs[(s['run'], s['name'])]
        if s['mode'] != 'continue' or not r or r.get('status') != 'success':
            continue
        rb = r['settings_read_back']
        ok = (rb['sampler'] == 'EulerEDMSampler' and rb['num_steps'] == s['generation']['n_steps'] and rb['sigma_max'] == 700.0 and rb['sigma_min'] == 0.002
              and rb['rho'] == 7.0 and rb['guider'] == 'TrianglePredictionGuider' and rb['max_scale'] == 2.5 and rb['min_scale'] == 1.0
              and rb['scale_per_frame'][0] == 1.0 and rb['scale_per_frame'][12] == 2.5 and rb['scale_per_frame'][24] == 1.0 and r['latents'] == 69
              and r['decoded_latents'] == 61)
        if not ok:
            set_bad.append(s['name'])
    items['4 settings'] = dict(ok=not set_bad, detail=set_bad)
    seed_bad, act_bad = [], []
    by = {}
    for s in specs:
        r = recs[(s['run'], s['name'])]
        if s['mode'] == 'continue' and r and r.get('status') == 'success' and s['generation'].get('action_variant', 'frame_index') == 'frame_index':
            by[(s['window'], s['track'], s['k'])] = (s, r)
    for (w, tr, k), (s, r) in by.items():
        if tr != 't2':
            continue
        h = by.get((w, 'h0', k))
        if h is None:
            continue
        if h[0]['seed'] != s['seed'] or h[1]['anchor_latent_sha256'] != r['anchor_latent_sha256']:
            seed_bad.append((w, k, 'h0/t2 seed or anchor latent differ'))
        if not r.get('actions') or len(r['actions']['per_round']) != 3 or any(len(x['trajectory']) != 8 for x in r['actions']['per_round']):
            act_bad.append((w, k, 'actions'))
        a, b = clips.get((w, f'h0-s{k}')), clips.get((w, f't2-s{k}'))
        if a is not None and b is not None and np.array_equal(a[21:], b[21:]):
            act_bad.append((w, k, 'h0 and t2 identical after t0'))
    ks = {}
    for (w, tr, k), (s, r) in by.items():
        ks.setdefault((w, tr), []).append(r['latents_sha256'])
    for key, v in ks.items():
        if len(set(v)) != len(v):
            seed_bad.append((key, 'two samples identical'))
    items['5 seeding'] = dict(ok=not seed_bad, detail=seed_bad)
    items['6 actions'] = dict(ok=not act_bad, detail=act_bad)
    if args.rerun_root and args.rerun_specs:
        det = []
        key = lambda x: (x['window'], x['mode'], x.get('track'), x.get('k'), x['generation'].get('action_variant', 'frame_index'))  # noqa: E731
        main_by_key = {key(x): x for x in specs}
        for s in specs_in(args.rerun_specs):
            m = main_by_key.get(key(s))
            a, b = (rec_of(args.out_root, m) if m else None), rec_of(args.rerun_root, s)
            same = bool(a and b and a.get('status') == b.get('status') == 'success' and a['frames_sha256'] == b['frames_sha256']
                        and a.get('latents_sha256') == b.get('latents_sha256'))
            det.append(dict(name=s['name'], run=s['run'], identical=same))
        items['7 determinism'] = dict(ok=bool(det) and all(x['identical'] for x in det), detail=det)
    cont = [r for (s, r) in by.values()]
    recon = [recs[(s['run'], s['name'])] for s in specs if s['mode'] == 'recon' and recs[(s['run'], s['name'])]]
    if cont:
        info['seconds_per_continuation'] = dict(median=float(np.median([r['timings']['total_s'] for r in cont])), min=min(r['timings']['total_s'] for r in cont),
                                                max=max(r['timings']['total_s'] for r in cont))
        info['round_seconds'] = [r['timings']['round_s'] for r in cont]
        info['decode_seconds'] = [r['timings']['decode_s'] for r in cont]
        info['gpu_peak_allocated_gib'] = max(v['peak_allocated_gib'] for r in cont for v in r['gpu_memory'].values())
        info['gpu_peak_reserved_gib'] = max(v['peak_reserved_gib'] for r in cont for v in r['gpu_memory'].values())
        info['host_peak_rss_gib'] = max(r['host']['peak_rss_gib'] for r in cont)
        info['gpu'] = cont[0]['gpu']
    if recon:
        info['seconds_per_reference'] = float(np.median([r['timings']['total_s'] for r in recon]))
    items['8 cost'] = dict(ok=True, detail=info)
    # boundary step (continuation-break statistic) and contact sheet
    jumps = {}
    for (w, tag), fr in clips.items():
        if tag == 'prefix':
            continue
        f = fr.astype(np.int16)
        hist = np.median([np.abs(f[i + 1] - f[i]).mean() for i in range(0, 20)])
        jumps[f'{w} {tag}'] = dict(step_19_20=round(float(np.abs(f[20] - f[19]).mean()), 2), step_20_21=round(float(np.abs(f[21] - f[20]).mean()), 2),
                                   median_history_step=round(float(hist), 2), ratio_20_21=round(float(np.abs(f[21] - f[20]).mean() / max(hist, 1e-6)), 2))
    out = dict(items={k: v for k, v in items.items()}, passed=sum(v['ok'] for v in items.values()), total=len(items), boundary_steps=jumps)
    args.result.write_text(json.dumps(out, indent=1, default=str) + '\n')
    if args.sheet:
        from PIL import Image, ImageDraw
        windows = sorted({w for (w, t) in clips if t != 'prefix'})
        tags = ['recon', 'h0-s0', 'h0-s1', 't2-s0', 't2-s1', 't2-s0ta']
        tw, th = 256, 144
        rows = [(w, t) for w in windows for t in tags if (w, t) in clips]
        sheet = Image.new('RGB', (110 + tw * len(FRAMES_T), 18 + th * len(rows)), (20, 20, 20))
        dr = ImageDraw.Draw(sheet)
        for j, t in enumerate(FRAMES_T):
            dr.text((110 + j * tw + 4, 3), f't = {t:+.1f} s', fill=(230, 230, 230))
        for i, (w, tag) in enumerate(rows):
            fr = clips[(w, tag)]
            dr.text((4, 18 + i * th + 4), f'{w[:8]}\n{tag}', fill=(230, 230, 230))
            for j, t in enumerate(FRAMES_T):
                im = Image.fromarray(fr[20 + int(round(t * 10))]).resize((tw, th), Image.BILINEAR)
                sheet.paste(im, (110 + j * tw, 18 + i * th))
        sheet.save(args.sheet)
    for k, v in items.items():
        print(('PASS ' if v['ok'] else 'FAIL ') + k, json.dumps(v['detail'], default=str)[:300])
    print(f"{out['passed']} / {out['total']}")


if __name__ == '__main__':
    main()
