#!/usr/bin/env python3
"""Checks of analyze_bench_v12.py (2026-11-02): a simulated K = 16 input in the v1.2 CSV format with known answers,
cross-checks against the validation-v1 implementation, and the TEST gates. Never reads bench-v1 TEST.

  check_analyze_bench_v12.py simulate --out DIR [--seed 21]
      DIR/windows/<name>/window.json (200 TEST-shaped simv12-* windows and 220 DEV-shaped simv12dev-* windows: log ids),
      DIR/v12-sim-k0-10.csv, DIR/v12-sim-k11-15.csv (samples 0-10 / 11-15, both with the real rows),
      DIR/derived-sim-raw.json (the climatology in the derived-JSON format) and DIR/sim-raw-flags.csv (raw t0 flags),
      DIR/truth.json (expected values by Monte Carlo of the same generator)
  check_analyze_bench_v12.py crosscheck --results DIR --runs CSV... [--truth truth.json]
  check_analyze_bench_v12.py gates --scratch DIR

Generator, per (window, family, quantity): Gaussian location-scale; real = m + s e, generated = m + s (-delta + lambda e);
caps censor at the cap (flag `censored`; ties); THW also censored at 10 s for every clip of a window whose ego is
stopped (structural ties); lat_speed_1_2s missing with probability 0.12 per clip; binary ego_stop_4s and
lead_hard_brake_0_4s Bernoulli with a generated rate offset; 3 % of window-families real_t0_not_found; the real
verdicts file decides which pairs carry `quantity_not_reliable`.
"""
from __future__ import annotations

import os

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_v, '2')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

import argparse, csv, json, math, subprocess, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_bench_v12 as a12  # noqa: E402
import analyze_bench_v1 as ab  # noqa: E402
import validate_calib_common as vc  # noqa: E402

K = 16
COUNTS = {'lead': 65, 'lead_stop': 65, 'startup': 10, 'cutin': 59, 'oncoming': 69, 'ego': 178}
Q = {'lead': ['closing_speed_2s', 'closing_speed_4s', 'min_ttc_0p5_4s', 'thw_4s', 'pet_follow', 'pet_t0', 'lead_speed_4s', 'lead_hard_brake_0_4s'],
     'cutin': ['lat_speed_1_2s', 'entry_time_absy_lt_1', 'cutin_min_ttc'], 'oncoming': ['exit_time'],
     'ego': ['ego_speed_2s', 'ego_speed_4s', 'ego_max_decel_0_4s', 'ego_stop_4s', 'ego_dx_0_2s', 'ego_dx_0_4s', 'ego_dx_0_6s']}
Q['lead_stop'] = Q['startup'] = Q['lead']
CAPS = {'min_ttc_0p5_4s': 20.0, 'cutin_min_ttc': 20.0, 'thw_4s': 10.0, 'pet_follow': 6.0, 'pet_t0': 6.0, 'entry_time_absy_lt_1': 6.0, 'exit_time': 6.0}
# (location range, scale) of each quantity, and the known miscalibration (delta, lambda); binaries: (rate range, generated offset)
MODEL = {'thw_4s': ((0.5, 6.0), 1.2), 'pet_follow': ((0.3, 5.0), 0.8), 'pet_t0': ((0.3, 5.0), 0.6), 'closing_speed_2s': ((-2, 4), 1.0), 'closing_speed_4s': ((-2, 4), 1.4),
         'min_ttc_0p5_4s': ((1, 15), 3.0), 'lead_speed_4s': ((0, 12), 1.5), 'lat_speed_1_2s': ((-0.3, 0.8), 0.25), 'entry_time_absy_lt_1': ((3, 9), 1.0),
         'cutin_min_ttc': ((10, 30), 4.0), 'exit_time': ((0.2, 2.5), 0.3), 'ego_speed_2s': ((0, 14), 0.9), 'ego_speed_4s': ((0, 14), 1.3), 'ego_max_decel_0_4s': ((0, 2.5), 0.5),
         'ego_dx_0_2s': ((0, 28), 1.0), 'ego_dx_0_4s': ((0, 56), 2.0), 'ego_dx_0_6s': ((0, 84), 3.0)}
MISCAL = {('lead', 'thw_4s'): (0.4, 0.75), ('lead', 'pet_follow'): (0.3, 1.0), ('lead_stop', 'thw_4s'): (0.25, 1.0), ('oncoming', 'exit_time'): (0.0, 0.6),
          ('ego', 'ego_speed_2s'): (0.3, 1.0), ('ego', 'ego_speed_4s'): (0.4, 1.0), ('ego', 'ego_dx_0_2s'): (0.3, 1.0), ('ego', 'ego_dx_0_4s'): (0.3, 1.0),
          ('ego', 'ego_dx_0_6s'): (0.3, 1.0)}
BIN = {'ego_stop_4s': ((0.05, 0.6), 0.10), 'lead_hard_brake_0_4s': ((0.0, 0.1), 0.0)}
FIELDS = ['window', 'family', 'sample', 'outcome', 'kind', 'value', 'scorable', 'flags', 'value_v3', 'value_v5', 'value_v8']


def window_family(fam, rng, stopped):
    """Real and K generated values {quantity: (value or None, censored, carried)} of one window-family."""
    real, gens = {}, [dict() for _ in range(K)]
    for q in Q[fam]:
        if q in BIN:
            (lo, hi), off = BIN[q]; p = rng.uniform(lo, hi)
            real[q] = (float(rng.random() < p), False, False)
            for g in gens:
                g[q] = (float(rng.random() < min(1.0, p + off)), False, False)
            continue
        (lo, hi), s = MODEL[q]; m = rng.uniform(lo, hi); d, lam = MISCAL.get((fam, q), (0.0, 1.0))
        vals = [m + s * rng.standard_normal()] + [m + s * (-d + lam * rng.standard_normal()) for _ in range(K)]
        for i, v in enumerate(vals):
            cen = False; carried = q.startswith('closing') and rng.random() < 0.3
            if q in CAPS and (v >= CAPS[q] or (q == 'thw_4s' and stopped)):
                v, cen = CAPS[q], True
            if q.startswith('ego_'):
                v = max(v, 0.0) if q != 'ego_max_decel_0_4s' else v
            if q == 'lat_speed_1_2s' and rng.random() < 0.12:
                v = None
            (real if i == 0 else gens[i - 1])[q] = (v, cen, carried)
    return real, gens


def expected(n_mc=20000, seed=5):
    rng = np.random.default_rng(seed); out = {}
    for fam in COUNTS:
        for q in Q[fam]:
            pits, ab_, be, rr, gr = [], [], [], [], []
            for _ in range(n_mc // 10 if fam == 'startup' else n_mc // 4):
                real, gens = window_family(fam, rng, stopped=rng.random() < 0.12)
                y = real[q][0]; Y = np.array([g[q][0] for g in gens if g[q][0] is not None], float)
                if y is None or len(Y) < 12:
                    continue
                if q in BIN:
                    rr.append(y); gr.append(Y.mean()); continue
                r = int((Y < y).sum()) + int(rng.integers(0, int((Y == y).sum()) + 1)); pits.append((r + rng.random()) / (len(Y) + 1))
                ab_.append(r == len(Y)); be.append(r == 0)
            out[f'{fam}|{q}'] = (dict(real_rate=float(np.mean(rr)), generated_rate=float(np.mean(gr))) if q in BIN else
                                 dict(theta=float(np.mean(pits) - 0.5), p_above=float(np.mean(ab_)), p_below=float(np.mean(be)), mc=len(pits)))
    return out


def cmd_simulate(a):
    rng = np.random.default_rng(a.seed); out = a.out; (out / 'windows').mkdir(parents=True, exist_ok=True)
    verdicts = json.loads((ab.BENCH / 'instrument/v1.2/verdicts-v1.2.json').read_text()); rows = []

    def make(prefix, n, counts):
        names = [f'{prefix}-{i:04d}-t2.5' for i in range(n)]; roles = defaultdict(list)
        for fam, c in counts.items():
            for i in rng.choice(n, size=c, replace=False):
                roles[names[i]].append(fam)
        for wi, w in enumerate(names):
            (out / 'windows' / w).mkdir(parents=True, exist_ok=True)
            (out / 'windows' / w / 'window.json').write_text(json.dumps(dict(name=w, log=f'simlog-{prefix}-{wi:04d}', roles=roles[w])) + '\n')
        return names, roles

    test_names, test_roles = make('simv12', 200, COUNTS)
    for w in test_names:
        stopped = rng.random() < 0.12
        for fam in test_roles[w]:
            real, gens = window_family(fam, rng, stopped); rt0 = rng.random() < 0.03
            for q in Q[fam]:
                rel = verdicts.get(f'{fam}|{q}') == 'reliable'
                for s, (v, cen, car) in [('real', real[q])] + [(str(k), gens[k][q]) for k in range(K)]:
                    fl = (['censored'] if cen else []) + (['carried'] if car else []) + ([] if rel else ['quantity_not_reliable'])
                    if rt0 and fam != 'ego':
                        fl += ['real_t0_not_found'] + (['t0_not_found'] if s == 'real' else [])
                    if v is None:
                        fl.append('no_value')
                    if fam == 'ego' and rng.random() < 0.05:
                        fl.append('vo_low_support')
                    hard = set(fl) & a12.HARD12
                    each = ('', '', '') if fam == 'ego' or v is None else tuple(round(v + (0 if cen else rng.normal(0, 0.02)), 4) for _ in range(3))
                    rows.append(dict(window=w, family=fam, sample=s, outcome=q, kind='v1.2-descriptive' if fam == 'startup' else 'v1.2-secondary',
                                     value='' if v is None else round(v, 4), scorable=int(not hard), flags=';'.join(sorted(set(fl))),
                                     value_v3=each[0], value_v5=each[1], value_v8=each[2]))
    for fn, keep in (('v12-sim-k0-10.csv', lambda s: s == 'real' or int(s) <= 10), ('v12-sim-k11-15.csv', lambda s: s == 'real' or int(s) >= 11)):
        with (out / fn).open('w', newline='') as fh:
            wr = csv.DictWriter(fh, fieldnames=FIELDS); wr.writeheader(); wr.writerows([r for r in rows if keep(r['sample'])])
    dev_names, dev_roles = make('simv12dev', 220, {f: int(round(c * 1.1)) for f, c in COUNTS.items()})
    der = dict(source='check_analyze_bench_v12.py simulate (DEV-shaped raw climatology)', windows={}); frows = []
    for w in dev_names:
        stopped = rng.random() < 0.12; rec = dict(families={}, ego=None)
        for fam in dev_roles[w]:
            real, _ = window_family(fam, rng, stopped)
            d = {q: dict(value=real[q][0], flags=(['censored'] if real[q][1] else []) + (['no_value'] if real[q][0] is None else [])) for q in Q[fam]}
            if fam == 'ego':
                rec['ego'] = d
            else:
                rec['families'][fam] = d
                frows.append(dict(window=w, family=fam, sample='raw', outcome='x_4s', kind='primary', value='', scorable=1,
                                  flags='t0_not_found' if rng.random() < 0.03 else '', value_v3='', value_v5='', value_v8=''))
        der['windows'][w] = dict(raw=rec)
    (out / 'derived-sim-raw.json').write_text(json.dumps(der) + '\n')
    with (out / 'sim-raw-flags.csv').open('w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=FIELDS); wr.writeheader(); wr.writerows(frows)
    exp = expected()
    (out / 'truth.json').write_text(json.dumps(dict(seed=a.seed, K=K, miscalibration={f'{k[0]}|{k[1]}': v for k, v in MISCAL.items()}, binary=BIN, expected=exp), indent=1) + '\n')
    print(json.dumps(dict(rows=len(rows), out=str(out), expected={k: round(v.get('theta', v.get('generated_rate', 0) - v.get('real_rate', 0)), 4) for k, v in exp.items() if verdicts.get(k) == 'reliable'})))


def cmd_crosscheck(a):
    res = json.loads((a.results / 'results-v12.json').read_text()); data, _ = ab.load_runs(a.runs); gids = ab.generated_ids(data); Kr = len(gids)
    rep = dict(per_pair={}, worst={}); worst = defaultdict(float)
    for t in res['tested']:
        if t['binary']:
            rows, _ = a12.build12(data, t['family'], t['quantity'], gids, 'tested')
            y = np.array([r['y'] for r in rows]); p = np.array([r['Y'].mean() for r in rows])
            d = dict(brier=abs(float(((p - y) ** 2).mean()) - t['binary_stats']['brier']), real_rate=abs(float(y.mean()) - t['binary_stats']['real_rate']))
        else:
            rows, _ = a12.build12(data, t['family'], t['quantity'], gids, 'tested'); ab.rank_rows(rows)
            pit = np.array([r['pit'] for r in rows]); crps = np.array([vc.awc.fair_crps(r['Y'], r['y']) for r in rows])
            b = vc.stats_block(ab.pit_bins(rows, Kr), pit, Kr, np.random.default_rng(1), crps=crps, mc=20000, boot=2000); s = t['stats']
            d = {k: abs(s[k] - b[k]) for k in ('mean_pit_minus_half', 'slope_z', 'convexity_z', 'chi2', 'mean_fair_crps')}
        rep['per_pair'][f"{t['family']}|{t['quantity']}"] = d
        for k, v in d.items():
            worst[k] = max(worst[k], v)
    rep['worst'] = dict(worst)
    if a.truth:
        tr = json.loads(Path(a.truth).read_text())['expected']; kn = {}
        for t in res['tested']:
            e = tr.get(f"{t['family']}|{t['quantity']}")
            if e is None:
                continue
            if t['binary']:
                b = t['binary_stats']; diff = e['real_rate'] - e['generated_rate']
                kn[f"{t['family']}|{t['quantity']}"] = dict(expected_difference=diff, estimate=b['rate_difference_real_minus_generated'], ci95=b['rate_difference_ci95'],
                                                            covers=bool(b['rate_difference_ci95'][0] <= diff <= b['rate_difference_ci95'][1]))
            else:
                s = t['stats']
                kn[f"{t['family']}|{t['quantity']}"] = dict(n=s['n'], expected_theta=e['theta'], estimate=s['mean_pit_minus_half'], ci95=s['mean_pit_ci95'],
                                                            covers=bool(s['mean_pit_ci95'][0] <= e['theta'] <= s['mean_pit_ci95'][1]), expected_above=e['p_above'], above=s['above_all']['share'])
        rep['known_answers'] = kn; rep['covered'] = f"{sum(v['covers'] for v in kn.values())} of {len(kn)}"
    (a.results / 'crosscheck-v12.json').write_text(json.dumps(rep, indent=1) + '\n')
    print(json.dumps(dict(worst=rep['worst'], covered=rep.get('covered')), indent=1))
    for k, v in (rep.get('known_answers') or {}).items():
        print(k, json.dumps({kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()}))


def cmd_gates(a):
    d = a.scratch; d.mkdir(parents=True, exist_ok=True); rng = np.random.default_rng(4); rows = []
    for w in ('1c98497e-t2.5', 'simgate-0001-t2.5', 'simgate-0002-t2.5', 'simgate-0003-t2.5'):   # a TEST window name: no TEST file is read
        for smp in ['real'] + [str(k) for k in range(4)]:
            rows.append(dict(window=w, family='ego', sample=smp, outcome='ego_speed_2s', kind='v1.2-secondary', value=round(5 + rng.normal(), 4), scorable=1, flags='',
                             value_v3='', value_v5='', value_v8=''))
    with (d / 'gate.csv').open('w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=FIELDS); wr.writeheader(); wr.writerows(rows)
    (d / 'done.json').write_text(json.dumps(dict(finished='x')) + '\n')
    bad = d / 'v12-frozen-bad.txt'; txt = (ab.BENCH / 'v12-frozen.txt').read_text().splitlines(); txt[1] = 'sha256 ' + '0' * 64; bad.write_text('\n'.join(txt) + '\n')
    base = [sys.executable, str(HERE / 'analyze_bench_v12.py'), '--runs', str(d / 'gate.csv'), '--windows-dir', str(d), '--label', 'GATE-CHECK', '--boot', '100', '--mc', '200', '--perm', '200']
    cases = [('no --allow-test', [], False), ('done file missing', ['--allow-test', '--done-file', str(d / 'missing.done')], False),
             ('freeze mismatch', ['--allow-test', '--done-file', str(d / 'done.json'), '--frozen', str(bad)], False),
             ('freeze mismatch + --force', ['--allow-test', '--done-file', str(d / 'done.json'), '--frozen', str(bad), '--force'], True),
             ('done exists, freeze matches', ['--allow-test', '--done-file', str(d / 'done.json')], True)]
    out = {}
    for i, (lab, extra, should) in enumerate(cases):
        od = d / f'out-{i}'; r = subprocess.run(base + extra + ['--out', str(od)], capture_output=True, text=True); ran = r.returncode == 0
        res = json.loads((od / 'results-v12.json').read_text()) if ran else None
        out[lab] = dict(expected_to_run=should, ran=ran, ok=ran == should, message=(r.stderr.strip().splitlines() or [''])[-1][:160],
                        gate=res and res['test_gate'], freeze_all_match=res and res['freeze']['all_match'])
    out['all_ok'] = all(v['ok'] for v in out.values() if isinstance(v, dict))
    (d / 'gates-v12.json').write_text(json.dumps(out, indent=1) + '\n'); print(json.dumps(out, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); sub = p.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('simulate'); s.add_argument('--out', required=True, type=Path); s.add_argument('--seed', type=int, default=21)
    c = sub.add_parser('crosscheck'); c.add_argument('--results', required=True, type=Path); c.add_argument('--runs', nargs='+', required=True, type=Path); c.add_argument('--truth')
    g = sub.add_parser('gates'); g.add_argument('--scratch', required=True, type=Path)
    a = p.parse_args(); dict(simulate=cmd_simulate, crosscheck=cmd_crosscheck, gates=cmd_gates)[a.cmd](a)


if __name__ == '__main__':
    main()
