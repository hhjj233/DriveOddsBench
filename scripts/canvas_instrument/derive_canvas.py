#!/usr/bin/env python3
"""bench-v1 v1.2 physical quantities (derive_bench_quantities.py, imported read only; definitions and reliability rule
of instrument/v1.2/QUANTITIES-PREREG.md) on the canvas instrument's outputs. derive: DEV, or TEST with --allow-test once the
model's frozen protocol file exists (never --gt on TEST); validate: DEV only.

Parameterised, not changed:
  - odometry records from this instrument's band odometry (ego_vo_band.py, VO_DIR/clips/<source>/...);
  - gt quantities with the spec's band and valid rows (ci_core.gt_on_band; the gt exit uses the effective edges);
  - 5 Hz: the pre-registered minimum numbers of readings of an LS slope are counts of 10 Hz frames (>= 6 of the 11
    frames within +-0.5 s; >= 6 of the 11 frames of the 1.0 s interval of lat_speed_1_2s). On the 5 Hz grid the same
    windows hold 5 and 6 frames, so the minimum is taken as the same share rounded up: 3 of 5 (+-0.5 s windows) and 4 of
    6 (1.0 s intervals). This is a frame-rate adaptation of section 2 of the pre-registration, reported as such.
Everything else (quantities, caps, censoring, ensemble, verdict rule) is the frozen code.

  derive_canvas.py derive   --bench B --model M --measure DIR --vo-dir DIR --sources RAW REF ... --out D [--gt] [--vo-alias SRC=VO_SRC]
  derive_canvas.py validate --bench B --derived D --raw-source RAW --recon-run REF --out D
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import ci_core as ci  # noqa: E402
import derive_bench_quantities as dq  # noqa: E402  (frozen v1.2, read only)


def configure(spec, vo_dir: Path, cfg_common, vo_alias=None):
    alias = dict(a.split('=', 1) for a in (vo_alias or []))

    def vo_record(bench, source, name, sample):
        f = vo_dir / 'clips' / alias.get(source, source) / f'{name}__{sample}.json'
        return json.loads(f.read_text()) if f.is_file() else None
    dq.vo_record = vo_record
    if spec['step'] == 2:
        orig = dq.ls_slope

        def ls_slope(pts, T, key, half=0.5, need=None, lo=None, hi=None):
            if need is None:
                need = 4 if lo is not None else 3
            return orig(pts, T, key, half=half, need=need, lo=lo, hi=hi)
        dq.ls_slope = ls_slope
    elif spec['step'] != 1:
        raise SystemExit('10 or 5 Hz only')

    def gt_record(bench, name):
        import validate_bench_v2 as vb
        wdir = bench / 'windows-dev' / name
        window = json.loads((wdir / 'window.json').read_text()); gt = ci.gt_on_band(json.loads((wdir / 'gt.json').read_text()), spec)
        ego = dq.ego_from_gt(gt)
        rec = dict(families={}, ego=dq.ego_quantities(ego))
        for fam in window['targets']:
            if fam not in dq.LEAD_FAMS and fam not in ('cutin', 'oncoming'):
                continue
            gs = vb.gt_series(window, gt, fam, cfg_common)
            pts = [(dq.r1(g['t']), g['x_near'], g['y_c']) for g in gs if g is not None]
            x4 = next((p[1] for p in pts if abs(p[0] - 4.0) < 1e-6), None)
            exits = None
            if fam == 'oncoming':
                ge = vb.gt_exit(gs)
                exits = {k: (None if ge is None else dict(t=ge)) for k in dq.TRACKERS}
            rec['families'][fam] = dq.family_quantities(fam, {k: pts for k in dq.TRACKERS}, {k: x4 for k in dq.TRACKERS}, ego, exits)
        return rec
    dq.gt_record = gt_record


def cmd_derive(args):
    spec = ci.load_spec(args.model)
    cfg = json.loads((args.measure / f'effective-config-{spec["model"]}.json').read_text())
    configure(spec, args.vo_dir, cfg['common'], args.vo_alias)
    names = set()
    for s in args.sources:
        names |= set(json.loads((args.measure / f'measure-{s}.json').read_text())['windows'])
    split = ci.split_of(args.bench, sorted(names), spec, args.allow_test)
    if split == 'test' and args.gt:
        raise SystemExit('refusing: --gt on TEST')
    dq.cmd_derive(argparse.Namespace(bench=args.bench, measure=args.measure, sources=args.sources, out=args.out, gt=args.gt))
    (args.out / 'derive-canvas.json').write_text(json.dumps(dict(model=spec['model'], fps=spec['fps'], vo_dir=str(args.vo_dir), sources=args.sources,
                                                                 ls_slope_need='6 (frozen)' if spec['step'] == 1 else '3 within +-0.5 s, 4 in 1.0 s intervals (5 Hz)',
                                                                 script_sha256=ci.sha256_file(__file__), frozen_sha256=ci.sha256_file(dq.__file__)), indent=1) + '\n')


def cmd_validate(args):
    """derive_bench_quantities.cmd_validate with the model's raw source and reference (QUANTITIES-PREREG.md section 6)."""
    D = args.derived
    raw = json.loads((D / f'derived-{args.raw_source}.json').read_text())
    rec = json.loads((D / f'derived-{args.recon_run}.json').read_text()) if args.recon_run else {'windows': {}}
    ci.dev_only(args.bench, sorted(raw['windows']))
    gtv = {n: s['gt'] for n, s in raw['windows'].items() if 'gt' in s}
    plan = [(f, q) for f in dq.LEAD_FAMS for q in dq.Q_LEAD] + [('cutin', q) for q in dq.Q_CUTIN] + [('oncoming', q) for q in dq.Q_ONC] + [('ego', q) for q in dq.Q_EGO]
    rows = []
    for fam, q in plan:
        r = dq.values(raw, 'raw', fam, q); c = dq.values(rec, 'real', fam, q)
        g = {n: ((gtv[n]['ego'] if fam == 'ego' else gtv[n]['families'].get(fam, {})).get(q, {}).get('value'), None) for n in gtv
             if (fam == 'ego' or fam in gtv[n]['families'])}
        both = [n for n in r if r[n][0] is not None and g.get(n, (None,))[0] is not None]
        a = [r[n][0] for n in both]; b = [g[n][0] for n in both]
        d = dict(family=fam, quantity=q, binary=q in dq.BINARY, n_raw=sum(v[0] is not None for v in r.values()), n_raw_gt=len(both),
                 raw_missing=float(np.mean([v[0] is None for v in r.values()])) if r else None,
                 raw_censored=float(np.mean(['censored' in v[1] for v in r.values()])) if r else None,
                 gt_censored=float(np.mean([g[n][0] is not None and q in dq.CAPS and g[n][0] >= dq.CAPS[q] - 1e-9 for n in g])) if g else None)
        if both:
            if q in dq.BINARY:
                d['agree_raw_gt'] = float(np.mean([(x > .5) == (y > .5) for x, y in zip(a, b)])); d['kappa_raw_gt'] = dq.kappa(a, b)
                d['gt_prevalence'] = float(np.mean(np.asarray(b) > .5)); d['raw_rate'] = float(np.mean(np.asarray(a) > .5))
            else:
                e = np.asarray(a) - np.asarray(b)
                d.update(spearman_raw_gt=dq.spearman(a, b), err_median_abs=float(np.median(np.abs(e))), err_p90_abs=float(np.percentile(np.abs(e), 90)),
                         bias=float(np.median(e)))
        rr = [n for n in c if c[n][0] is not None and n in r and r[n][0] is not None]
        d['n_raw_recon'] = len(rr)
        if rr:
            a2 = [r[n][0] for n in rr]; b2 = [c[n][0] for n in rr]
            if q in dq.BINARY:
                d['agree_raw_recon'] = float(np.mean([(x > .5) == (y > .5) for x, y in zip(a2, b2)]))
            else:
                d['spearman_raw_recon'] = dq.spearman(a2, b2)
        d['recon_missing'] = float(np.mean([v[0] is None for v in c.values()])) if c else None
        d['recon_censored'] = float(np.mean(['censored' in v[1] for v in c.values()])) if c else None
        d['verdict'], d['reasons'] = dq.verdict(d)
        if not args.recon_run:
            d['reasons'] = d['reasons'] + ['PROVISIONAL: no reference yet (criterion 2 not evaluated)']
        rows.append(d)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f'validation-v1.2-{args.raw_source}.json').write_text(json.dumps(rows, indent=1) + '\n')
    (args.out / f'verdicts-v1.2-{args.raw_source}.json').write_text(json.dumps({f"{d['family']}|{d['quantity']}": d['verdict'] for d in rows}, indent=1) + '\n')
    f = lambda v, n=2: '-' if v is None else (f'{v:.{n}f}' if isinstance(v, float) else str(v))
    print(f'| family | quantity | n raw | Spearman raw-gt / agree (kappa) | err median / p90 / bias | raw-recon (n) | censored raw / gt | verdict |')
    print('|---|---|---|---|---|---|---|---|')
    for d in rows:
        valid = f"agree {f(d.get('agree_raw_gt'))} ({f(d.get('kappa_raw_gt'))}), prev {f(d.get('gt_prevalence'))}" if d['binary'] else f(d.get('spearman_raw_gt'))
        err = '' if d['binary'] else f"{f(d.get('err_median_abs'))} / {f(d.get('err_p90_abs'))} / {f(d.get('bias'))}"
        rr = f(d.get('agree_raw_recon') if d['binary'] else d.get('spearman_raw_recon'))
        print(f"| {d['family']} | {d['quantity']} | {d['n_raw']} | {valid} | {err} | {rr} ({d['n_raw_recon']}) | {f(d.get('raw_censored'))} / {f(d.get('gt_censored'))} | "
              f"{d['verdict']}{(' (' + '; '.join(d['reasons']) + ')') if d['reasons'] else ''} |")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('derive')
    p.add_argument('--bench', type=Path, required=True); p.add_argument('--model', required=True); p.add_argument('--measure', type=Path, required=True)
    p.add_argument('--vo-dir', type=Path, required=True); p.add_argument('--sources', nargs='+', required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--gt', action='store_true')
    p.add_argument('--vo-alias', nargs='*', default=[], help='SRC=VO_SRC: read the odometry records of VO_SRC for SRC (checks only)')
    p.add_argument('--allow-test', action='store_true', help="TEST windows (no --gt); refused unless the spec's test_freeze_file exists in the bench")
    p = sub.add_parser('validate')
    p.add_argument('--bench', type=Path, required=True); p.add_argument('--derived', type=Path, required=True)
    p.add_argument('--raw-source', required=True); p.add_argument('--recon-run', default=None); p.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    {'derive': cmd_derive, 'validate': cmd_validate}[args.cmd](args)


if __name__ == '__main__':
    main()
