#!/usr/bin/env python3
"""Continuation-break rate of VaViM on its own 2 Hz grid (exploratory).

The frozen continuation_break.py compares the first 10 Hz step after t0 with the clip's typical 10 Hz history step. VaViM's
10 Hz canvas repeats each 2 Hz model frame (frames 21-22 still show the t0 frame), so that test does not apply. Here the
same rule is applied to VaViM's own 0.5 s step, with the frozen frame reader (grey_frames, imported unchanged):
- boundary jump J = mean|f[t0+5] - f[t0]| / mean over the last second of history of mean|f[i] - f[i-5]| (i = t0-5, t0),
  i.e. the first model frame (tick 5) against t0, relative to the clip's own 0.5 s history steps;
- first-second change C = mean|f[t0+10] - f[t0]| of the clip / the same quantity of the window's real reference;
- boundary break J > 3, strong break J > 5, fast change C > 2 (the frozen thresholds). Real references give the floor.
Usage: continuation_break_2hz.py --out DIR --ref REF_RUN_DIR --run VaViM=RUN_DIR [--workers 4]"""
import argparse, hashlib, importlib.util, json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

FROZEN = Path(__file__).resolve().parent / 'continuation_break.py'
_s = importlib.util.spec_from_file_location('cb_frozen', FROZEN); cb = importlib.util.module_from_spec(_s); _s.loader.exec_module(cb)
T0, STEP, FPS = 20, 5, 10


def clip_stats(path):
    f = cb.grey_frames(Path(path), T0 + FPS + 1)
    if len(f) < T0 + FPS + 1:
        return path, None
    hist = np.mean([np.abs(f[i] - f[i - STEP]).mean() for i in (T0 - STEP, T0)])
    jump = np.abs(f[T0 + STEP] - f[T0]).mean(); change = np.abs(f[T0 + FPS] - f[T0]).mean()
    return path, dict(J=float(jump / max(hist, 1e-3)), change=float(change))


def describe(js, cs=None):
    js = np.array(js); d = dict(n=int(len(js)), J_median=float(np.median(js)), boundary_break=float(np.mean(js > 3)), strong_break=float(np.mean(js > 5)))
    if cs is not None:
        cs = np.array(cs); d.update(C_median=float(np.median(cs)), fast_change=float(np.mean(cs > 2)))
    return d


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', type=Path, required=True); ap.add_argument('--ref', type=Path, required=True)
    ap.add_argument('--run', required=True); ap.add_argument('--workers', type=int, default=4); a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True); label, rdir = a.run.split('=', 1)
    refs = {p.parent.name[:-len('-recon')]: p for p in sorted((a.ref / 'native').glob('*-recon/vision.mp4'))}
    clips = sorted(Path(rdir).glob('native/*-s*/vision.mp4'))
    with ProcessPoolExecutor(a.workers) as ex:
        stats = dict(ex.map(clip_stats, [str(p) for p in refs.values()] + [str(p) for p in clips], chunksize=16))
    ref = {w: stats[str(p)] for w, p in refs.items() if stats.get(str(p))}
    js, cs, per = [], [], {}
    for p in clips:
        s = stats.get(str(p)); w = p.parent.name.rsplit('-s', 1)[0]
        if not s or w not in ref:
            continue
        c = s['change'] / max(ref[w]['change'], 1e-3); js.append(s['J']); cs.append(c); per[p.parent.name] = dict(J=s['J'], C=c)
    out = dict(summary=dict(definition=dict(J='mean|f[t0+5]-f[t0]| / mean last-second 0.5 s history step (VaViM 2 Hz grid)', C="first-second change / the real reference's",
                                             boundary_break='J > 3', strong_break='J > 5', fast_change='C > 2', fps=FPS, t0_frame=T0, step=STEP, frozen_reader=str(FROZEN)),
                            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                            groups={'real reference': describe([s['J'] for s in ref.values()]), label: describe(js, cs)}), per_clip=per)
    (a.out / 'continuation-break.json').write_text(json.dumps(out, indent=1) + '\n'); print(json.dumps(out['summary']['groups']))


if __name__ == '__main__':
    main()
