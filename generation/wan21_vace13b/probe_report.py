#!/usr/bin/env python3
"""Size-probe summary of Wan2.1-VACE-1.3B DEV clips (research server, DEV only; descriptive, no test).

For each run directory (native/<window>-{prefix,recon,s<k>}/ as written by run_vace_bench.py or a local dev-smoke chunk)
and each continuation:
  - the VACE prefix regeneration per round (latents vs the context, pixels vs the kept frames) and the regenerated t0 / 33
    history frames against the window's shared prefix (all from sample_outputs.json);
  - the seams: mean grey step at t0 (frame 20 -> 21) and at the round boundary (frame 50 -> 51, t0 + 3.0 -> 3.1 s) over the
    clip's mean last-second history step, and the same for the window's reference;
  - sharpness (variance of the Laplacian of the grey canvas frame) relative to the window's reference at t = 0, 1, ..., 6 s,
    and the colour drift (mean absolute RGB difference of the frame mean to the reference's at the same tick);
  - the scene distance D(t) (mean |grey difference| to the reference at the same tick) with its scale, the reference's own
    distance to its t0 frame, and the largest future step over the reference's median step (with its tick);
  - timing (rounds, sample) and GPU memory.
Writes <out>/probe-report.json and prints one line per clip and a summary per run label.

  probe_report.py --run LABEL=RUN_DIR [--run LABEL=RUN_DIR ...] --out DIR
Run with the bench instrument's Python (OpenCV).
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from pathlib import Path

import cv2
import numpy as np

W, H = 512, 288
TICKS_S = (0, 1, 2, 3, 4, 5, 6)


def decode(p: Path) -> np.ndarray:
    raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-threads', '2', '-i', str(p), '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'])
    return np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3)


def grey(f):
    return cv2.cvtColor(f, cv2.COLOR_RGB2GRAY).astype(np.float32)


def sharp(f) -> float:
    return float(cv2.Laplacian(grey(f), cv2.CV_32F).var())


def steps(v) -> np.ndarray:
    g = np.stack([grey(f) for f in v])
    return np.abs(np.diff(g, axis=0)).mean(axis=(1, 2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', action='append', required=True, help='LABEL=RUN_DIR (holds native/)')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rows, summary = [], {}
    for spec in a.run:
        label, d = spec.split('=', 1)
        nat = Path(d) / 'native'
        recs = {p.parent.name: json.loads(p.read_text()) for p in nat.glob('*/sample_outputs.json')}
        refs = {r['window']: n for n, r in recs.items() if r['mode'] == 'recon'}
        ref_v = {w: decode(nat / n / 'vision.mp4') for w, n in refs.items()}
        ref_s = {w: steps(v) for w, v in ref_v.items()}
        lab = []
        for n, r in sorted(recs.items()):
            if r['mode'] != 'continue':
                continue
            w = r['window']
            v = decode(nat / n / 'vision.mp4')
            st = steps(v)
            hist = float(st[10:20].mean())
            row = dict(label=label, sample=n, window=w, size=r['size']['wh'],
                       rounds=[dict(round=x['round'], seconds=x['seconds'], latent_mean_abs=x['prefix_latent_vs_context']['mean_abs'],
                                    latent_rel_l2=x['prefix_latent_vs_context']['rel_l2'], kept_psnr_db=x['prefix_pixels_vs_kept']['psnr_db'],
                                    kept_mean_abs=x['prefix_pixels_vs_kept']['mean_abs'], last_kept_psnr_db=x['last_kept_pixels']['psnr_db'])
                               for x in r['generation']['rounds']],
                       regen_prefix_vs_prefix=r['generation'].get('regenerated_prefix_vs_prefix'),
                       regen_t0_vs_prefix=r['generation']['regenerated_t0_vs_prefix_t0'],
                       J_t0=round(float(st[20]) / max(hist, 1e-3), 3), J_round=round(float(st[50]) / max(hist, 1e-3), 3),
                       seconds=r['timing']['generate_s'], peak_reserved_gib=r['memory']['peak_reserved_gib'])
            if w in ref_v:
                rs = ref_s[w]
                rh = float(rs[10:20].mean())
                row['ref_J_t0'] = round(float(rs[20]) / max(rh, 1e-3), 3)
                row['ref_J_round'] = round(float(rs[50]) / max(rh, 1e-3), 3)
                row['sharpness_vs_ref'] = {f'{t}s': round(sharp(v[20 + 10 * t]) / max(sharp(ref_v[w][20 + 10 * t]), 1e-6), 3) for t in TICKS_S}
                row['colour_drift_vs_ref'] = {f'{t}s': round(float(np.abs(v[20 + 10 * t].reshape(-1, 3).mean(0) - ref_v[w][20 + 10 * t].reshape(-1, 3).mean(0)).mean()), 2)
                                              for t in TICKS_S}
                # scene distance: mean |difference| of the grey frames to the reference's at the same tick (a coherent
                # continuation keeps the static scene; a replaced scene is far from it)
                row['mad_vs_ref'] = {f'{t}s': round(float(np.abs(grey(v[20 + 10 * t]) - grey(ref_v[w][20 + 10 * t])).mean()), 2) for t in TICKS_S}
                # the scale of D(t): how far the real scene itself moves from its t0 frame (the reference against its own t0)
                row['ref_vs_own_t0'] = {f'{t}s': round(float(np.abs(grey(ref_v[w][20 + 10 * t]) - grey(ref_v[w][20])).mean()), 2) for t in TICKS_S}
                fut = st[20:]
                row['max_step_over_ref_median'] = round(float(fut.max()) / max(float(np.median(rs[20:])), 1e-3), 2)
                row['max_step_tick'] = int(np.argmax(fut)) + 1
            rows.append(row)
            lab.append(row)
            print(json.dumps({k: row[k] for k in ('label', 'sample', 'J_t0', 'J_round', 'seconds')} | dict(
                regen_t0_psnr=row['regen_t0_vs_prefix']['psnr_db'], kept_psnr=[x['kept_psnr_db'] for x in row['rounds']],
                mad=row.get('mad_vs_ref'), max_step=row.get('max_step_over_ref_median'), at_tick=row.get('max_step_tick'))), flush=True)
        if lab:
            med = lambda xs: round(statistics.median(xs), 3)
            summary[label] = dict(n=len(lab), seconds_median=med([x['seconds'] for x in lab]),
                                  J_t0_median=med([x['J_t0'] for x in lab]), J_round_median=med([x['J_round'] for x in lab]),
                                  regen_t0_psnr_median=med([x['regen_t0_vs_prefix']['psnr_db'] for x in lab]),
                                  kept_psnr_round1_median=med([x['rounds'][0]['kept_psnr_db'] for x in lab]),
                                  kept_psnr_round2_median=med([x['rounds'][1]['kept_psnr_db'] for x in lab]),
                                  sharpness_6s_median=med([x['sharpness_vs_ref']['6s'] for x in lab if 'sharpness_vs_ref' in x]) if any('sharpness_vs_ref' in x for x in lab) else None,
                                  sharpness_min_over_ticks=min(min(x['sharpness_vs_ref'].values()) for x in lab if 'sharpness_vs_ref' in x) if any('sharpness_vs_ref' in x for x in lab) else None,
                                  mad_vs_ref_6s_median=med([x['mad_vs_ref']['6s'] for x in lab if 'mad_vs_ref' in x]) if any('mad_vs_ref' in x for x in lab) else None,
                                  peak_reserved_gib_max=max(x['peak_reserved_gib'] for x in lab))
            withref = [x for x in lab if 'mad_vs_ref' in x]
            if withref:
                one = {x['window']: x for x in withref}.values()       # the reference's own values once per window
                summary[label].update(
                    D_median={f'{t}s': med([x['mad_vs_ref'][f'{t}s'] for x in withref]) for t in (1, 2, 4, 6)},
                    D_range={f'{t}s': [min(x['mad_vs_ref'][f'{t}s'] for x in withref), max(x['mad_vs_ref'][f'{t}s'] for x in withref)] for t in (1, 2, 4, 6)},
                    ref_vs_own_t0_median={f'{t}s': med([x['ref_vs_own_t0'][f'{t}s'] for x in one]) for t in (1, 2, 4, 6)},
                    R_values=[x['J_round'] for x in withref], ref_R_values=[x['ref_J_round'] for x in one],
                    max_step_median=med([x['max_step_over_ref_median'] for x in withref]),
                    max_step_ticks=[x['max_step_tick'] for x in withref])
    (a.out / 'probe-report.json').write_text(json.dumps(dict(rows=rows, summary=summary), indent=1) + '\n')
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
