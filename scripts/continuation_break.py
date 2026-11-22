#!/usr/bin/env python3
"""Continuation-break rate of generated clips (descriptive; defined 2026-11-03, exploratory for Nano and Edge).

A continuation should pick up the scene where the history ends. For every generated clip and its window's real
reference (both on the same frame grid), with grey frames f:
- boundary jump J = mean|f[t0+1] - f[t0]| / mean over the last second of history of mean|f[i] - f[i-1]|
  (the first generated step relative to the clip's own typical history step);
- first-second change C = mean|f[t0+1 s] - f[t0]| of the clip / the same quantity of the window's real reference.
A clip is a "boundary break" if J > 3 (also reported: J > 5) and a "fast change" if C > 2. The real references give
the floor for J. No instrument reading is used; this is a pixel statistic of the videos only.

  continuation_break.py --out DIR --fps 10 --t0-frame 20 --ref REF_RUN_DIR \
      --run Nano=RUN_DIR[,RUN_DIR...] --run Cosmos3-Edge=RUN_DIR [--workers 4]

A run directory holds native/<window>-s<k>/vision.mp4; the reference run holds native/<window>-recon/vision.mp4.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np


def grey_frames(path: Path, n: int):
    cap = cv2.VideoCapture(str(path)); out = []
    while len(out) < n:
        ok, f = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32))
    cap.release()
    return out


def clip_stats(args):
    path, t0, fps = args
    f = grey_frames(Path(path), t0 + fps + 1)
    if len(f) < t0 + fps + 1:
        return path, None
    hist = np.mean([np.abs(f[i] - f[i - 1]).mean() for i in range(t0 - fps + 1, t0 + 1)])
    jump = np.abs(f[t0 + 1] - f[t0]).mean()
    change = np.abs(f[t0 + fps] - f[t0]).mean()
    return path, dict(J=float(jump / max(hist, 1e-3)), change=float(change))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True); ap.add_argument('--fps', type=int, required=True)
    ap.add_argument('--t0-frame', type=int, required=True); ap.add_argument('--ref', type=Path, required=True)
    ap.add_argument('--run', action='append', required=True, help='LABEL=DIR[,DIR...]')
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    refs = {p.parent.name[:-len('-recon')]: p for p in sorted((a.ref / 'native').glob('*-recon/vision.mp4'))}
    jobs = [(str(p), a.t0_frame, a.fps) for p in refs.values()]
    runs = {}
    for spec in a.run:
        label, dirs = spec.split('=', 1)
        clips = []
        for d in dirs.split(','):
            clips += sorted(Path(d).glob('native/*-s*/vision.mp4'))
        runs[label] = clips
        jobs += [(str(p), a.t0_frame, a.fps) for p in clips]
    with ProcessPoolExecutor(a.workers) as ex:
        stats = dict(ex.map(clip_stats, jobs, chunksize=16))
    ref_stats = {w: stats[str(p)] for w, p in refs.items() if stats.get(str(p))}
    summary = dict(definition=dict(J='mean|f[t0+1]-f[t0]| / mean last-second history step', C='first-second change / the real reference\'s',
                                   boundary_break='J > 3', strong_break='J > 5', fast_change='C > 2', fps=a.fps, t0_frame=a.t0_frame),
                   script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), groups={})
    def describe(js, cs=None, windows=None):
        js = np.array(js); d = dict(n=int(len(js)), J_median=float(np.median(js)), boundary_break=float(np.mean(js > 3)),
                                    strong_break=float(np.mean(js > 5)))
        if cs is not None:
            cs = np.array(cs); d.update(C_median=float(np.median(cs)), fast_change=float(np.mean(cs > 2)))
        if windows is not None:
            d['windows_with_any_boundary_break'] = float(np.mean([any(v > 3 for v in vs) for vs in windows.values()]))
        return d
    summary['groups']['real reference'] = describe([s['J'] for s in ref_stats.values()])
    per_clip = {}
    for label, clips in runs.items():
        js, cs, by_w = [], [], {}
        for p in clips:
            s = stats.get(str(p)); w = p.parent.name.rsplit('-s', 1)[0]
            if not s or w not in ref_stats:
                continue
            c = s['change'] / max(ref_stats[w]['change'], 1e-3)
            js.append(s['J']); cs.append(c); by_w.setdefault(w, []).append(s['J'])
            per_clip[f'{label}/{p.parent.name}'] = dict(J=round(s['J'], 4), C=round(c, 4))
        summary['groups'][label] = describe(js, cs, by_w)
    (a.out / 'continuation-break.json').write_text(json.dumps(dict(summary=summary, per_clip=per_clip), indent=1) + '\n')
    lines = ['# Continuation-break rate (descriptive)', '',
             'J: first generated step / typical last-second history step; C: first-second change relative to the real reference.', '',
             '| group | clips | J median | J > 3 | J > 5 | C median | C > 2 | windows with any J > 3 |', '|---|---|---|---|---|---|---|---|']
    for label, d in summary['groups'].items():
        lines.append(f"| {label} | {d['n']} | {d['J_median']:.2f} | {100*d['boundary_break']:.1f} % | {100*d['strong_break']:.1f} % | "
                     f"{d.get('C_median', float('nan')):.2f} | {100*d.get('fast_change', float('nan')):.1f} % | "
                     f"{100*d.get('windows_with_any_boundary_break', float('nan')):.1f} % |")
    (a.out / 'continuation-break.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
