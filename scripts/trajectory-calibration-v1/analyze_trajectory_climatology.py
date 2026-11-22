"""Trajectory-level calibration v1, addendum 1 (PLAN-ADDENDUM-1.md): energy-score skill against climatology.

  python analyze_trajectory_climatology.py --bench BENCH_V1_ROOT --out RUN_DIR
Refuses to run unless frozen-addendum-1.txt lists the sha256 of PLAN-ADDENDUM-1.md and of this script, and unless the
frozen PLAN.md / analyze_trajectories.py still match frozen.txt.
"""
from __future__ import annotations
import argparse, csv, importlib.util, json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('at', HERE / 'analyze_trajectories.py')
at = importlib.util.module_from_spec(spec); spec.loader.exec_module(at)


def guard():
    at.guard()
    f = HERE / 'frozen-addendum-1.txt'
    if not f.is_file():
        sys.exit('frozen-addendum-1.txt missing')
    pins = dict(reversed(l.split()[:2]) for l in f.read_text().splitlines() if l and not l.startswith('#'))
    for name in ('PLAN-ADDENDUM-1.md', 'analyze_trajectory_climatology.py'):
        if pins.get(name) != at.sha(HERE / name):
            sys.exit(f'{name} changed since the freeze')


def fair_es(X, y):
    k = len(X)
    return float(np.linalg.norm(X - y, axis=1).mean() - np.linalg.norm(X[:, None] - X[None], axis=2).sum() / (2 * k * (k - 1)))


def climatology(bench):
    vals = {}
    for r in csv.DictReader(open(bench / 'instrument/outcomes/outcomes-dev-raw-climatology.csv')):
        f = r['family']
        if f in at.FAMS and r['kind'] == 'primary' and r['outcome'] == at.FAMS[f][2] and r['scorable'] == '1' and r['value'] != '':
            vals[(r['window'], f)] = float(r['value'])
    meas = json.loads((bench / 'instrument/measure-dev/measure-nano-raw.json').read_text())['windows']
    out, mismatch = {f: [] for f in at.FAMS}, []
    for (w, f), v in sorted(vals.items()):
        grid = at.FAMS[f][1]
        if f == 'ego_inter':
            c = at.ego_curve(bench / 'instrument/ego-vo/clips/nano-raw' / f'{w}__raw.json', grid)
            pv = None if c is None else c[grid.index(at.FAMS[f][3])]
        else:
            rec = meas[w]['raw']
            c = [at.series_value(rec, f, t) for t in grid]
            pv = at.series_value(rec, f, at.FAMS[f][3])
        if c is None or any(x is None for x in c):
            continue
        if abs(pv - v) > 1e-3:
            mismatch.append((w, f, pv, v))
        out[f].append(c)
    return {f: np.asarray(v, float) for f, v in out.items()}, mismatch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bench', type=Path, required=True); ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    guard()
    clim, mismatch = climatology(a.bench)
    if mismatch:
        sys.exit(f'{len(mismatch)} DEV curves disagree with the climatology CSV: {mismatch[:3]}')
    rep = dict(addendum_sha256=at.sha(HERE / 'PLAN-ADDENDUM-1.md'), script_sha256=at.sha(__file__),
               started=time.strftime('%Y-%m-%dT%H:%M:%S%z'), climatology_n={f: len(v) for f, v in clim.items()}, models={})
    for name, spec_ in at.MODELS.items():
        cases, _ = at.load_cases(a.bench, spec_[1])
        curv, dropped, mm = at.curves(a.bench, spec_, cases)
        if mm:
            sys.exit(f'{name}: {len(mm)} curves disagree with the frozen CSV')
        rng = np.random.default_rng(at.SEED)
        res = {}
        for f in at.FAMS:
            keys = sorted(k for k in curv if k[1] == f)
            em = np.array([fair_es(curv[k][1:], curv[k][0]) for k in keys])
            ec = np.array([fair_es(clim[f], curv[k][0]) for k in keys])
            bs = []
            for _ in range(at.B_BOOT):
                j = rng.integers(0, len(keys), len(keys)); bs.append(1 - em[j].sum() / ec[j].sum())
            res[f] = dict(n=len(keys), es_model=float(em.mean()), es_clim=float(ec.mean()), skill=float(1 - em.sum() / ec.sum()),
                          ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
        rep['models'][name] = res
        print(name, {f: round(v['skill'], 3) for f, v in res.items()}, flush=True)
    rep['finished'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    (a.out / 'addendum-1-climatology.json').write_text(json.dumps(rep, indent=1))
    lines = ['# Addendum 1: energy-score skill of whole trajectories against climatology (TEST, exploratory)', '',
             f"Climatology curves (DEV raw): {rep['climatology_n']}", '', '| model | ' + ' | '.join(at.FAMS) + ' |', '|---' * (len(at.FAMS) + 1) + '|']
    for m, r in rep['models'].items():
        lines.append(f'| {m} | ' + ' | '.join(f"{r[f]['skill']:+.2f} [{r[f]['ci95'][0]:+.2f}, {r[f]['ci95'][1]:+.2f}]" for f in at.FAMS) + ' |')
    (a.out / 'addendum-1-climatology.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
