#!/usr/bin/env python3
"""Regression checks of the canvas instrument against the frozen instrument v2 (DEV only, CPU).

  geometry   K_band(797) == window.json K_canvas; band affine of box_source == stored box_canvas; band_flags with the
             bench band == stored in_band / height_ok / points_ok / visible (all DEV windows, every annotated tick)
  measure    ci_core.measure_clip with the 10 Hz bench spec == measure_bench_v2.measure_clip (frozen) on the cached
             canvas-raw detections of every DEV window: JSON-identical outputs; the effective config of the 10 Hz bench
             spec == config-v2.json
  validate   validate_canvas.evaluate_source / evaluate_history with the 10 Hz bench spec == validate_bench_v2's on the
             same measure output (JSON-identical records)

  regress_canvas.py geometry|measure|validate --bench B [--names-file F] [--out OUT.json]
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import ci_core as ci  # noqa: E402
import measure_bench_v2 as mb  # noqa: E402


def canon(x):
    return json.loads(json.dumps(x))


def cmd_geometry(args, names):
    spec = ci.load_spec('orbis2'); out = dict(windows=0, ticks=0, K_max_abs=0.0, box_max_abs=0.0, flag_mismatch=0, t0_box_max_abs=0.0)
    for w in names:
        wd = args.bench / 'windows-dev' / w
        win = json.loads((wd / 'window.json').read_text()); gt = json.loads((wd / 'gt.json').read_text())
        out['K_max_abs'] = max(out['K_max_abs'], float(np.abs(ci.K_band(win, 797) - np.asarray(win['intrinsics']['K_canvas'])).max()))
        A = ci.band_affine(797)
        for fam, t in win['targets'].items():
            if isinstance(t, dict) and t.get('box_source') is not None:
                out['t0_box_max_abs'] = max(out['t0_box_max_abs'], float(np.abs(np.subtract(ci.map_box(t['box_source'], A), t['box_canvas'])).max()))
        for fam, t in gt['targets'].items():
            for r in t.get('per_tick', []):
                if not r.get('annotated'):
                    continue
                out['ticks'] += 1
                if r.get('box_source') is not None:
                    out['box_max_abs'] = max(out['box_max_abs'], float(np.abs(np.subtract(ci.map_box(r['box_source'], A), r['box_canvas'])).max()))
                f = ci.band_flags(r.get('box_source'), int(r.get('lidar_points', 0)), spec['valid_source_rows'])
                vis = f['in_band'] and f['height_ok'] and f['points_ok']
                if any(f[k] != r[k] for k in f) or vis != r['visible']:
                    out['flag_mismatch'] += 1
        out['windows'] += 1
    out['pass'] = out['K_max_abs'] < 1e-9 and out['box_max_abs'] < 1e-6 and out['t0_box_max_abs'] < 1e-6 and out['flag_mismatch'] == 0
    return out


def cmd_measure(args, names):
    frozen = json.loads((args.bench / 'instrument' / 'config-v2.json').read_text())
    spec = ci.load_spec('orbis2'); eff = ci.effective_config(frozen, spec)
    cfg_same = canon({k: v for k, v in eff.items() if k != 'canvas_instrument'}) == canon(frozen)
    out = dict(windows=0, families=0, identical=0, differ=[], config_identical=cfg_same)
    for w in names:
        win = json.loads((args.bench / 'windows-dev' / w / 'window.json').read_text())
        dets = json.loads((args.bench / 'detections' / 'canvas-raw' / f'{w}.json').read_text())
        a = canon(mb.measure_clip(win, dets, frozen)); b = canon(ci.measure_clip(win, dets, eff, spec))
        out['windows'] += 1; out['families'] += len(a)
        if a == b:
            out['identical'] += len(a)
        else:
            out['differ'].append(w)
    out['pass'] = cfg_same and not out['differ']
    return out


def cmd_validate(args, names):
    import validate_bench_v2 as vb
    import validate_canvas as vc
    spec = ci.load_spec('orbis2')
    frozen = json.loads((args.bench / 'instrument' / 'config-v2.json').read_text())
    meas = json.loads(args.measure.read_text())
    a = canon(vb.evaluate_source(args.bench, names, meas, frozen['common']))
    b = canon(vc.evaluate_source(args.bench, names, meas, frozen['common'], spec))
    ha = canon(vb.evaluate_history(args.bench, names, meas, frozen['common']))
    hb = canon(vc.evaluate_history(args.bench, names, meas, frozen['common'], spec))
    out = dict(records=len(a), identical=a == b, history_records=len(ha), history_identical=ha == hb)
    if a != b:
        out['first_diff'] = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
    out['pass'] = out['identical'] and out['history_identical']
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=('geometry', 'measure', 'validate'))
    ap.add_argument('--bench', type=Path, required=True); ap.add_argument('--names-file', type=Path, default=None)
    ap.add_argument('--measure', type=Path, default=None, help='validate: a measure JSON of the 10 Hz bench spec')
    ap.add_argument('--out', type=Path, default=None)
    args = ap.parse_args()
    names = ci.names_from(args.names_file or args.bench / 'names-dev.txt'); ci.dev_only(args.bench, names)
    res = {'geometry': cmd_geometry, 'measure': cmd_measure, 'validate': cmd_validate}[args.cmd](args, names)
    res.update(check=args.cmd, script_sha256=ci.sha256_file(__file__), ci_core_sha256=ci.sha256_file(ci.__file__))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(res, indent=1) + '\n')
    print(json.dumps({k: v for k, v in res.items() if k != 'differ'} | {'n_differ': len(res.get('differ', []))}))


if __name__ == '__main__':
    main()
