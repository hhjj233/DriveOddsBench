#!/usr/bin/env python3
"""Checks of the local Epona DEV smoke (smoke_dev_local.sh): PASS/FAIL list, INFO numbers and a contact sheet.

  check_epona_smoke.py --smoke DIR --bench-root BENCH [--out DIR]

DEV only. Reads the smoke's clips and sample_outputs.json, the bench's DEV canvas/real.mp4 (raw reference row of the
contact sheet) and, for the time-scale check, the DEV gt.json ego poses (real speed before and after t0).
Writes check_smoke.json and contact-sheet.png (AV2 RGB without face/plate blurring: local viewing only).
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

CW, CH, MW, MH, N, HIST = 512, 288, 1024, 512, 41, 11
TICKS = list(range(-20, 61, 2))
SHEET_K = [-20, -10, 0, 10, 20, 30, 40, 50, 60]


def decode(path: Path, w: int, h: int):
    raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-threads', '2', '-i', str(path), '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'])
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)


def fps_of(path: Path) -> float:
    out = subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=r_frame_rate', '-of', 'csv=p=0', str(path)], text=True)
    a, b = out.strip().split('/')
    return float(a) / float(b)


def samples(root: Path) -> dict:
    out = {}
    for f in sorted(root.glob('*/native/*/sample_outputs.json')):
        r = json.loads(f.read_text())
        r['_dir'] = f.parent
        out[r['name']] = r
    return out


def step_mad(clip, lo, hi):
    return float(np.mean([np.abs(clip[i + 1].astype(np.int16) - clip[i]).mean() for i in range(lo, hi)]))


def real_speeds(gt_path: Path) -> dict:
    """DEV only: ego speed from gt.json ego poses at 5 Hz, before t0 (-1.0..0 s) and after (0..2, 0..6 s), m/s."""
    g = json.loads(gt_path.read_text())
    xyz = {e['k']: np.array(e['xyz_m']) for e in g['ego']}

    def sp(a, b):
        ks = list(range(a, b + 1, 2))
        d = sum(np.linalg.norm(xyz[k2][:2] - xyz[k1][:2]) for k1, k2 in zip(ks[:-1], ks[1:]))
        return round(float(d / ((b - a) * 0.1)), 3)
    return dict(past_1s=sp(-10, 0), fut_0_2s=sp(0, 20), fut_0_6s=sp(0, 60))


def label(img, text):
    import cv2
    img = img.copy()
    cv2.rectangle(img, (0, 0), (8 + 9 * len(text), 20), (0, 0, 0), -1)
    cv2.putText(img, text, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--smoke', type=Path, required=True)
    ap.add_argument('--bench-root', type=Path, required=True)
    ap.add_argument('--out', type=Path)
    args = ap.parse_args()
    out = args.out or args.smoke
    main_s, rerun_s = samples(args.smoke), samples(args.smoke / 'rerun')
    checks, info = {}, {}

    ok = all(r['status'] == 'success' for r in list(main_s.values()) + list(rerun_s.values())) and len(main_s) == 6 and len(rerun_s) == 2
    checks['all_success'] = dict(ok=ok, main=len(main_s), rerun=len(rerun_s), failed=[n for n, r in {**main_s, **rerun_s}.items() if r['status'] != 'success'])

    clips, natives, fmt_bad = {}, {}, []
    for tag, group in (('main', main_s), ('rerun', rerun_s)):
        for n, r in group.items():
            v = r['_dir'] / 'vision.mp4'
            c = decode(v, CW, CH)
            clips[(tag, n)] = c
            if c.shape != (N, CH, CW, 3) or abs(fps_of(v) - 5) > 1e-6:
                fmt_bad.append((tag, n, c.shape))
            nv = r['_dir'] / 'native.mp4'
            if nv.is_file():
                natives[(tag, n)] = decode(nv, MW, MH)
                if natives[(tag, n)].shape != (N, MH, MW, 3):
                    fmt_bad.append((tag, n, 'native', natives[(tag, n)].shape))
    checks['format_41x512x288_5fps'] = dict(ok=not fmt_bad, bad=fmt_bad)

    windows = sorted({r['window'] for r in main_s.values()})
    pref = {}
    for w in windows:
        names = [n for n, r in main_s.items() if r['window'] == w]
        ref = clips[('main', f'{w}-recon')][:HIST]
        same = {n: bool(np.array_equal(clips[('main', n)][:HIST], ref)) for n in names}
        inproc = {n: main_s[n].get('prefix_identical_to_recon_in_process') for n in names if main_s[n]['mode'] == 'continue'}
        pref[w] = dict(decoded_identical=same, in_process=inproc)
    checks['prefix_11_identical_to_recon'] = dict(ok=all(all(p['decoded_identical'].values()) and all(p['in_process'].values()) for p in pref.values()), windows=pref)

    counts = {n: {k: r['rollout'][k] for k in ('stt_evals', 'traj_dit_evals', 'vis_dit_evals')} for n, r in main_s.items() if r['mode'] == 'continue'}
    checks['network_evals_30_3000_3000'] = dict(ok=all(c == dict(stt_evals=30, traj_dit_evals=3000, vis_dit_evals=3000) for c in counts.values()), counts=counts)

    rr = {}
    for n in rerun_s:
        a, b = clips[('main', n)], clips[('rerun', n)]
        d = dict(vision_identical=bool(np.array_equal(a, b)), max_abs=int(np.abs(a.astype(np.int16) - b).max()))
        if ('main', n) in natives and ('rerun', n) in natives:
            d['native_identical'] = bool(np.array_equal(natives[('main', n)], natives[('rerun', n)]))
        if main_s[n]['mode'] == 'continue':
            d['pred_traj_identical'] = bool(np.array_equal(np.load(main_s[n]['_dir'] / 'pred_traj.npy'), np.load(rerun_s[n]['_dir'] / 'pred_traj.npy')))
        rr[n] = d
    checks['same_seed_rerun_identical'] = dict(ok=bool(rr) and all(all(v for k, v in d.items() if k != 'max_abs') for d in rr.values()), samples=rr)

    sett = {json.dumps(r['settings_read_back'], sort_keys=True) for r in {**main_s, **rerun_s}.values()}
    checks['settings_constant'] = dict(ok=len(sett) == 1, distinct=len(sett))
    need = ('name', 'uuid', 'capability', 'driver', 'cuda_driver', 'cuda_runtime', 'cudnn', 'torch')
    hw = {n: [k for k in need if not r['gpu'].get(k)] for n, r in {**main_s, **rerun_s}.items()}
    checks['hardware_record_complete'] = dict(ok=all(not v for v in hw.values()), missing=hw, gpu={k: next(iter(main_s.values()))['gpu'].get(k) for k in need})

    # INFO --------------------------------------------------------------------------------------------------------------
    info['timing_s'] = {n: {k: v for k, v in r['timing_s'].items() if not isinstance(v, list)} for n, r in {**main_s, **rerun_s}.items()}
    info['gpu_memory_gib'] = {n: r['gpu_memory_gib'] for n, r in {**main_s, **rerun_s}.items()}
    info['host_memory_gib'] = {n: r['host_memory_gib'] for n, r in {**main_s, **rerun_s}.items()}
    info['gpu_smi_per_sample'] = {n: r.get('gpu_smi') for n, r in main_s.items()}
    runs = sorted((args.smoke / 'adapter-runs').glob('*.json')) + sorted((args.smoke / 'rerun' / 'adapter-runs').glob('*.json'))
    info['adapter_runs'] = [dict(file=p.name, **{k: json.loads(p.read_text()).get(k) for k in ('model_load_s', 'gpu_free_at_start_mib', 'gpu_cap_mib',
                                 'host_memory_after_load', 'gpu_memory_after_load_gib', 'gpu_smi_after_load', 'gpu_memory_peak_gib', 'failures')}) for p in runs]
    guards = {p.name: json.loads(p.read_text()) for p in sorted((args.smoke / 'logs').glob('*-guard.json'))}
    info['guards'] = {k: {kk: v.get(kk) for kk in ('host_avail_start_gib', 'min_host_avail_gib', 'gpu_free_start_mib', 'min_gpu_free_mib', 'child_peak_rss_gib',
                                                    'stopped_by_guard', 'returncode')} for k, v in guards.items()}
    info['model_context_vs_prefix'] = {n: r['diagnostics']['model_context_vs_prefix_native'] for n, r in main_s.items() if r['mode'] == 'continue'}
    info['recon_diagnostics'] = {n: r['diagnostics'] for n, r in main_s.items() if r['mode'] == 'recon'}
    info['boundary_steps'] = {n: r['diagnostics']['step_mean_abs_canvas'] for n, r in main_s.items() if r['mode'] == 'continue'}
    mot = {}
    for w in windows:
        rec = clips[('main', f'{w}-recon')]
        rf = step_mad(rec, HIST - 1, N - 1)
        row = dict(recon_future_step_mad=round(rf, 3), recon_history_step_mad=round(step_mad(rec, 0, HIST - 1), 3))
        for n, r in main_s.items():
            if r['window'] == w and r['mode'] == 'continue':
                g = step_mad(clips[('main', n)], HIST - 1, N - 1)
                row[n] = dict(future_step_mad=round(g, 3), ratio_to_recon=round(g / rf, 3) if rf > 0 else None,
                              pred_speed_mps=r['rollout']['pred_speed_mps'])
        row['real_speed_mps_dev_gt'] = real_speeds(args.bench_root / 'windows-dev' / w / 'gt.json')
        mot[w] = row
    info['time_scale'] = mot

    # contact sheet -----------------------------------------------------------------------------------------------------
    import cv2
    rows = []
    s = 0.5
    for w in windows:
        raw = decode(args.bench_root / 'windows-dev' / w / 'canvas/real.mp4', CW, CH)
        for tag, src in [('raw', None), ('recon', f'{w}-recon'), ('s0', f'{w}-s0'), ('s1', f'{w}-s1')]:
            cells = []
            for k in SHEET_K:
                f = raw[k + 20] if src is None else clips[('main', src)][TICKS.index(k)]
                f = cv2.resize(f, (int(CW * s), int(CH * s)), interpolation=cv2.INTER_AREA)
                cells.append(label(f, f'{w[:8]} {tag} {k / 10:+.0f}s'))
            rows.append(np.concatenate(cells, axis=1))
        rows.append(np.zeros((6, rows[-1].shape[1], 3), np.uint8))
    sheet = np.concatenate(rows, axis=0)
    cv2.imwrite(str(out / 'contact-sheet.png'), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    # a second sheet at full canvas resolution: t0, +2 s, +4 s, +6 s for recon and the two samples
    rows = []
    for w in windows:
        for tag in ('recon', 's0', 's1'):
            rows.append(np.concatenate([label(clips[('main', f'{w}-{tag}')][TICKS.index(k)], f'{w[:8]} {tag} {k / 10:+.0f}s') for k in (0, 20, 40, 60)], axis=1))
    cv2.imwrite(str(out / 'contact-sheet-full.png'), cv2.cvtColor(np.concatenate(rows, axis=0), cv2.COLOR_RGB2BGR))

    res = dict(schema='wm-calib-bench.epona.check_smoke.v1', all_pass=all(c['ok'] for c in checks.values()),
               checks={k: ('PASS' if v['ok'] else 'FAIL') for k, v in checks.items()}, details=checks, info=info,
               sheets=['contact-sheet.png', 'contact-sheet-full.png'])
    (out / 'check_smoke.json').write_text(json.dumps(res, indent=1, default=str) + '\n')
    print(json.dumps(dict(all_pass=res['all_pass'], checks=res['checks']), indent=1))


if __name__ == '__main__':
    main()
