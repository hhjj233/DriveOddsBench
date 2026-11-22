"""Trajectory-level calibration v1 (PLAN.md in this directory). Reads frozen TEST measurement files only.

  python analyze_trajectories.py --bench BENCH_V1_ROOT --out RUN_DIR [--models "Orbis 2,..."]
Refuses to run unless frozen.txt (this directory) lists the sha256 of PLAN.md and of this script.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, sys, time
from pathlib import Path
import numpy as np
from scipy.stats import binomtest

HERE = Path(__file__).resolve().parent
SEED, B_BOOT, B_NULL, K = 2026, 2000, 100000, 16
G04 = [round(0.4 * i, 1) for i in range(1, 11)]   # 0.4 s grid: shared by the 10 Hz models and Epona (5 Hz)
FAMS = {  # family: (reading, grid, primary outcome, primary horizon)
    'lead': ('x', G04, 'x_4s', 4.0), 'lead_stop': ('x', G04, 'x_4s', 4.0), 'cutin': ('absy', G04, 'absy_2s', 2.0),
    'oncoming': ('x', [0.2, 0.4, 0.6, 0.8, 1.0], 'x_1s', 1.0),
    'ego_inter': ('ego', [round(0.4 * i, 1) for i in range(1, 16)], 'ego_dx_0_4s', 4.0)}
MODELS = {  # name: frozen v1 analysis dir, outcome CSVs, generated measure JSONs, recon JSON, generated VO dirs, recon VO dir
    'Orbis 2': ('orbis2/analysis-test-v1', ['orbis2/instrument/outcomes/test/outcomes-orbis2-test-k16.csv'],
                ['orbis2/instrument/measure-test/measure-orbis2-test-k16.json'], 'orbis2/instrument/measure-test/measure-orbis2-test-recon.json',
                ['orbis2/ego-vo/clips/orbis2-test-k16'], 'orbis2/ego-vo/clips/orbis2-test-recon'),
    'Cosmos3-Nano': ('analysis-test-v1', ['instrument/outcomes/test/outcomes-test-k0-10.csv', 'instrument/outcomes/test/outcomes-test-k11-15.csv'],
                     ['instrument/measure-test/measure-test-k0-10.json', 'instrument/measure-test/measure-test-k11-15.json'],
                     'instrument/measure-test/measure-test-recon.json',
                     ['instrument/ego-vo/clips/test-k0-10', 'instrument/ego-vo/clips/test-k11-15'], 'instrument/ego-vo/clips/test-recon'),
    'Cosmos3-Edge': ('edge/analysis-test-v1', ['edge/instrument/outcomes/test/outcomes-edge-test-k0-15.csv'],
                     ['edge/instrument/measure-test/measure-edge-test-k0-15.json'], 'edge/instrument/measure-test/measure-edge-test-recon.json',
                     ['edge/instrument/ego-vo/clips/edge-test-k0-15'], 'edge/instrument/ego-vo/clips/edge-test-recon'),
    'Epona': ('epona/analysis-test-v1', ['epona/instrument/outcomes/test/outcomes-epona-test-b16-k16.csv'],
              ['epona/instrument/measure-test/measure-epona-test-b16-k16.json'], 'epona/instrument/measure-test/measure-epona-test-local-recon.json',
              ['epona/ego-vo/clips/epona-test-b16-k16'], 'epona/ego-vo/clips/epona-test-local-recon'),
    'LTX-2.5': ('ltx25d-2s/test/analysis-test-v1', ['ltx25d-2s/test/instrument/outcomes/test/outcomes-ltx25d-test-k16.csv'],
                ['ltx25d-2s/test/instrument/measure-test/measure-ltx25d-test-k16.json'], 'ltx25d-2s/test/instrument/measure-test/measure-ltx25d-test-recon.json',
                ['ltx25d-2s/ego-vo/clips/ltx25d-test-k16'], 'ltx25d-2s/ego-vo/clips/ltx25d-test-recon'),
    'Vista': ('vista/analysis-test-v1', ['vista/instrument/outcomes/test/outcomes-vista-test-h0-k16.csv'],
              ['vista/instrument/measure-test/measure-vista-test-h0-k16.json'], 'vista/instrument/measure-test/measure-vista-test-local-recon.json',
              ['vista/ego-vo/clips/vista-test-h0-k16'], 'vista/ego-vo/clips/vista-test-local-recon')}
SAMPLES = [str(k) for k in range(K)]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def guard():
    f = HERE / 'frozen.txt'
    if not f.is_file():
        sys.exit('frozen.txt missing: freeze PLAN.md and this script first')
    pins = dict(reversed(l.split()[:2]) for l in f.read_text().splitlines() if l and not l.startswith('#'))
    for name in ('PLAN.md', 'analyze_trajectories.py'):
        if pins.get(name) != sha(HERE / name):
            sys.exit(f'{name} changed since the freeze')


def load_cases(bench, csvs):
    prim = {f: FAMS[f][2] for f in FAMS}
    rows, hist = {}, {}
    for c in csvs:
        for r in csv.DictReader(open(bench / c)):
            f = r['family']
            if f not in FAMS:
                continue
            key = (r['window'], f)
            if r['outcome'] == prim[f] and r['kind'] == 'primary':
                rows.setdefault(key, {})[r['sample']] = (int(r['scorable']), None if r['value'] == '' else float(r['value']))
            elif r['kind'] == 'history' and r['sample'] == 'real' and r['scorable'] == '1' and r['value'] != '':
                hist.setdefault(key, {})[r['outcome']] = float(r['value'])
    cases = {k: v for k, v in rows.items() if all(s in v and v[s][0] == 1 for s in SAMPLES + ['real'])}
    return cases, hist


def series_value(rec, fam, t):
    """x: the ensemble series (median over trackers of x, each carrying its last reading). abs(y): median over trackers
    of each tracker's abs(y), carried the same way, which is the rule of the scalar absy outcomes (median of abs, not abs
    of median)."""
    ser = rec['families'][fam]['ensemble']['series']
    j = [i for i, s in enumerate(ser) if abs(s['t'] - t) < 1e-6]
    assert len(j) == 1, (fam, t)
    j = j[0]
    if FAMS[fam][0] == 'x':
        return ser[j]['x']
    vals = []
    for tr in rec['families'][fam]['trackers'].values():
        last = None
        for r in tr['track'][:j + 1]:
            if 'x' in r and r['state'] in ('followed', 'anchor_box'):
                last = abs(r['y'])
        assert abs(tr['track'][j]['t'] - t) < 1e-6
        if last is not None:
            vals.append(last)
    return float(np.median(vals)) if len(vals) >= 2 else None


def ego_curve(path, grid):
    if not path.is_file():
        return None
    tr = {round(p['t'], 1): p['x'] for p in json.loads(path.read_text())['planar']['trajectory']}
    return [tr.get(t) for t in grid]


def curves(bench, spec, cases):
    gen = {}
    for p in spec[2]:
        for w, d in json.loads((bench / p).read_text())['windows'].items():
            gen.setdefault(w, {}).update(d)
    real = json.loads((bench / spec[3]).read_text())['windows']
    out, dropped, mismatch = {}, [], []
    for (w, f), csvv in sorted(cases.items()):
        grid = FAMS[f][1]
        V = []
        for s in ['real'] + SAMPLES:
            if f == 'ego_inter':
                dirs = [spec[5]] if s == 'real' else spec[4]
                c = None
                for d in dirs:
                    c = ego_curve(bench / d / f'{w}__{s}.json', grid)
                    if c is not None:
                        break
                prim_val = None if c is None else c[grid.index(FAMS[f][3])]
            else:
                rec = real[w]['real'] if s == 'real' else gen[w][s]
                c = [series_value(rec, f, t) for t in grid]
                prim_val = series_value(rec, f, FAMS[f][3])
            if c is None or any(v is None for v in c):
                c = None
            elif abs(prim_val - csvv[s][1]) > 1e-3:
                mismatch.append((w, f, s, prim_val, csvv[s][1]))
            V.append(c)
        if any(c is None for c in V):
            dropped.append((w, f)); continue
        out[(w, f)] = np.asarray(V, float)
    return out, dropped, mismatch


def ranks_of(V, rng):
    """Per-time ranks 1..n with random tie tags, then band-depth and average-rank pre-ranks (integer sums)."""
    n, L = V.shape
    r = np.empty((n, L), int)
    for t in range(L):
        o = np.lexsort((rng.random(n), V[:, t])); r[o, t] = np.arange(1, n + 1)
    return r, (r * (n + 1 - r)).sum(1), r.sum(1)


def rank_among(pre, j, tags):
    """Rank of curve j's pre-rank among the other curves (ties by tags)."""
    others = np.delete(np.arange(len(pre)), j)
    return int(np.sum(pre[others] < pre[j]) + np.sum((pre[others] == pre[j]) & (tags[others] < tags[j])))


def analyse(curv, hist, n_obs_first=0, rng=None):
    """Statistics for one model. n_obs_first=0: row 0 (real) is the observation. Positive control: rows 1.. only."""
    res = {f: dict(per_t=[], R_bd=[], R_avg=[], R_bd_all=[], R_avg_all=[], windows=[], es=[], es_cv=[]) for f in FAMS}
    for (w, f), V in sorted(curv.items()):
        if n_obs_first:
            V = V[1:]
        n = V.shape[0]
        r, bd, av = ranks_of(V, rng)
        tb, ta = rng.random(n), rng.random(n)
        d = res[f]
        d['windows'].append(w)
        d['per_t'].append((r[0] - 1).tolist())
        d['R_bd'].append(rank_among(bd, 0, tb)); d['R_avg'].append(rank_among(av, 0, ta))
        d['R_bd_all'].append([rank_among(bd, j, tb) for j in range(n)]); d['R_avg_all'].append([rank_among(av, j, ta) for j in range(n)])
        if not n_obs_first:
            X, Y = V[1:], V[0]; k = len(X)
            es = np.linalg.norm(X - Y, axis=1).mean() - np.linalg.norm(X[:, None] - X[None], axis=2).sum() / (2 * k * (k - 1))
            cv = None; grid = np.asarray(FAMS[f][1]); h = hist.get((w, f), {})
            if f in ('lead', 'lead_stop') and {'hist_ls_x0', 'hist_ls_vx'} <= set(h):
                cv = h['hist_ls_x0'] + h['hist_ls_vx'] * grid
            elif f == 'cutin' and {'hist_ls_y0', 'hist_ls_vy'} <= set(h):
                cv = np.abs(h['hist_ls_y0'] + h['hist_ls_vy'] * grid)
            elif f == 'ego_inter' and 'ego_hist_ls_speed' in h:
                cv = h['ego_hist_ls_speed'] * grid
            d['es'].append(float(es)); d['es_cv'].append(None if cv is None else float(np.linalg.norm(cv - Y)))
    return res


def window_stat(res, event, rng, n):
    """H2-type statistic: mean over windows of the share of the window's families with the event; exact null by
    drawing which of the n clips is the observation in each window; window bootstrap interval."""
    per_w = {}
    for f, d in res.items():
        for w, Rall in zip(d['windows'], d[event[0] + '_all']):
            per_w.setdefault(w, []).append([event[1](x) for x in Rall])
    ws = sorted(per_w)
    M = np.array([np.mean(np.array(per_w[w], float), axis=0) for w in ws])   # (W, n): share if clip j were the observation
    T = float(M[:, 0].mean())
    null = np.empty(B_NULL)
    for i in range(0, B_NULL, 10000):
        J = rng.integers(0, n, size=(min(10000, B_NULL - i), len(ws)))
        null[i:i + len(J)] = M[np.arange(len(ws)), J].mean(1)
    p = float((1 + np.sum(null >= T)) / (1 + B_NULL)) if T >= null.mean() else float((1 + np.sum(null <= T)) / (1 + B_NULL))
    bs = [M[rng.integers(0, len(ws), len(ws)), 0].mean() for _ in range(B_BOOT)]
    return dict(T=T, null_mean=float(null.mean()), p_one_sided=p, ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], n_windows=len(ws))


def bh(ps):
    ps = np.asarray(ps, float); m = len(ps); o = np.argsort(ps)
    q = np.minimum.accumulate((ps[o] * m / np.arange(1, m + 1))[::-1])[::-1]
    out = np.empty(m); out[o] = np.minimum(q, 1); return out.tolist()


def summarise(res, rng, n, control=False):
    out = {}
    for f, d in res.items():
        if not d['windows']:
            continue
        P = np.asarray(d['per_t']); m = len(P)
        pit = (P + rng.random(P.shape)) / n - 0.5
        boot = [pit[rng.integers(0, m, m)].mean(0) for _ in range(B_BOOT)]
        o = dict(n=m, grid=FAMS[f][1], mean_pit_t=pit.mean(0).tolist(),
                 mean_pit_t_ci95=[np.percentile(boot, 2.5, axis=0).tolist(), np.percentile(boot, 97.5, axis=0).tolist()],
                 outside_t=((P == 0) | (P == n - 1)).mean(0).tolist())
        Rb, Ra = np.asarray(d['R_bd']), np.asarray(d['R_avg'])
        o['bd_hist'] = np.bincount(Rb, minlength=n).tolist(); o['avg_hist'] = np.bincount(Ra, minlength=n).tolist()
        o['bd_least_central'] = float((Rb == 0).mean()); o['bd_p'] = binomtest(int((Rb == 0).sum()), m, 1 / n).pvalue
        o['avg_outside'] = float(((Ra == 0) | (Ra == n - 1)).mean()); o['avg_p'] = binomtest(int(((Ra == 0) | (Ra == n - 1)).sum()), m, 2 / n).pvalue
        o['avg_mean_pit'] = float(((Ra + rng.random(m)) / n - 0.5).mean())
        if not control and any(v is not None for v in d['es_cv']):
            idx = [i for i, v in enumerate(d['es_cv']) if v is not None]
            es, ec = np.asarray(d['es'])[idx], np.asarray([d['es_cv'][i] for i in idx])
            bs = []
            for _ in range(B_BOOT):
                j = rng.integers(0, len(idx), len(idx)); bs.append(1 - es[j].sum() / ec[j].sum())
            o['es_skill_vs_cv'] = float(1 - es.sum() / ec.sum()); o['es_skill_ci95'] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
            o['es_n'] = len(idx)
        out[f] = o
    pooled = dict(bd_least_central=window_stat(res, ('R_bd', lambda x: x == 0), rng, n),
                  avg_outside=window_stat(res, ('R_avg', lambda x: x in (0, n - 1)), rng, n))
    return dict(families=out, pooled=pooled)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bench', type=Path, required=True); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--models', default=','.join(MODELS))
    a = ap.parse_args()
    guard()
    a.out.mkdir(parents=True, exist_ok=True)
    report = dict(plan_sha256=sha(HERE / 'PLAN.md'), script_sha256=sha(__file__), started=time.strftime('%Y-%m-%dT%H:%M:%S%z'), models={})
    for name in a.models.split(','):
        spec = MODELS[name]
        frozen = json.loads((a.bench / spec[0] / 'results.json').read_text())
        cases, hist = load_cases(a.bench, spec[1])
        for f in FAMS:
            n_frozen = frozen['families_results'][f]['stats']['n']
            n_here = sum(1 for k in cases if k[1] == f)
            if n_frozen != n_here:
                sys.exit(f'{name} {f}: {n_here} cases, frozen analysis has {n_frozen}')
        curv, dropped, mismatch = curves(a.bench, spec, cases)
        if mismatch:
            json.dump(mismatch[:50], open(a.out / f'mismatch-{name}.json', 'w')); sys.exit(f'{name}: {len(mismatch)} curves disagree with the frozen CSV')
        rng = np.random.default_rng(SEED)
        res = analyse(curv, hist, 0, rng)
        main_s = summarise(res, rng, K + 1)
        ctrl = summarise(analyse(curv, hist, 1, rng), rng, K, control=True)
        report['models'][name] = dict(cases={f: sum(1 for k in curv if k[1] == f) for f in FAMS}, dropped=dropped, result=main_s,
                                      positive_control=dict(families={f: {k: v[k] for k in ('bd_least_central', 'avg_outside', 'mean_pit_t')} for f, v in ctrl['families'].items()},
                                                            pooled={k: v['T'] for k, v in ctrl['pooled'].items()}))
        print(name, report['models'][name]['cases'], 'dropped', len(dropped), flush=True)
    for stat, pk in (('bd', 'bd_p'), ('avg', 'avg_p')):
        keys = [(m, f) for m, r in report['models'].items() for f in r['result']['families']]
        qs = bh([report['models'][m]['result']['families'][f][pk] for m, f in keys])
        for (m, f), q in zip(keys, qs):
            report['models'][m]['result']['families'][f][stat + '_q'] = q
    report['finished'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    (a.out / 'results.json').write_text(json.dumps(report, indent=1))
    lines = ['# Trajectory-level calibration v1 (TEST, exploratory; PLAN.md)', '',
             '| model | pooled: real curve least central (1/17 = 5.9%) | pooled: average-rank outside (2/17 = 11.8%) | positive control (bd / avg) |', '|---|---|---|---|']
    for m, r in report['models'].items():
        pb, pa = r['result']['pooled']['bd_least_central'], r['result']['pooled']['avg_outside']
        lines.append(f"| {m} | {100*pb['T']:.1f} [{100*pb['ci95'][0]:.1f}, {100*pb['ci95'][1]:.1f}] p {pb['p_one_sided']:.1e} | "
                     f"{100*pa['T']:.1f} [{100*pa['ci95'][0]:.1f}, {100*pa['ci95'][1]:.1f}] p {pa['p_one_sided']:.1e} | "
                     f"{100*r['positive_control']['pooled']['bd_least_central']:.1f} / {100*r['positive_control']['pooled']['avg_outside']:.1f} |")
    lines += ['', '| model | family | n | least central % (q) | avg outside % (q) | ES skill vs CV [CI] | mean PIT-1/2 at grid times |', '|---|---|---|---|---|---|---|']
    for m, r in report['models'].items():
        for f, o in r['result']['families'].items():
            sk = f"{o['es_skill_vs_cv']:+.2f} [{o['es_skill_ci95'][0]:+.2f}, {o['es_skill_ci95'][1]:+.2f}]" if 'es_skill_vs_cv' in o else '-'
            lines.append(f"| {m} | {f} | {o['n']} | {100*o['bd_least_central']:.1f} ({o['bd_q']:.3f}) | {100*o['avg_outside']:.1f} ({o['avg_q']:.3f}) | {sk} | "
                         + ' '.join(f'{v:+.2f}' for v in o['mean_pit_t']) + ' |')
    (a.out / 'results.md').write_text('\n'.join(lines) + '\n')
    print('done', a.out)


if __name__ == '__main__':
    main()
