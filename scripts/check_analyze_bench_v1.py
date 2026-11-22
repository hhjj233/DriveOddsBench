#!/usr/bin/env python3
"""Checks of analyze_bench_v1.py (2026-11-02): a simulated TEST-shaped input with known answers, and cross-checks
against the validation-v1 implementation (validate_calib_common.py) and the study-v1 pilot. Never reads bench-v1 TEST.

  check_analyze_bench_v1.py simulate --out DIR [--scenario shift|null] [--seed 11]
      DIR/windows/<name>/window.json   200 TEST-shaped windows (simtest-*) and 220 DEV-shaped windows (simdev-*)
      DIR/sim-k0-10.csv, DIR/sim-k11-15.csv   generated samples 0-10 and 11-15 (both with the real and raw rows)
      DIR/sim-dev-raw-climatology.csv  raw readings and history lines of the 220 DEV-shaped windows
      DIR/truth.json                   the model and its expected values (Monte Carlo of the same generator)
  check_analyze_bench_v1.py crosscheck --results DIR --runs CSV... [--truth truth.json]
      recompute every per-family statistic of results.json with validate_calib_common.stats_block on the same ranks;
      the H2 Monte Carlo p against an exact FFT p; the known answers when --truth is given.
  check_analyze_bench_v1.py gates --scratch DIR
      the TEST gates (--allow-test, test-measured.done, --force) and the protocol exclusion on tiny synthetic rows
  check_analyze_bench_v1.py pilot
      study-v1 TEST g6 (lead gap at 4 s) through analyze_bench_v1's rank and statistics code: must reproduce the frozen
      pilot result (+0.11735, slope z 5.7737) bit for bit.

Simulation model, per (window, family): a past state (x0, y0, relative velocities) and, at horizon h, a Gaussian
conditional distribution of the outcome with mean m_h (constant-velocity extrapolation of the true state) and SD
s_h = 0.3 + 0.35 h; the real reading is one draw; the K generated readings are m_h + s_h (-delta_f + lambda_f e).
|y| outcomes are |.| of the lateral draw; binary outcomes are derived from each clip's own readings. History lines and
annotation velocities are noisy (line) or nearly exact (annotation) versions of the true state. Scorability: 3 % of
window-families real_t0_not_found, 2 % of generated readings unusable, ped / crossing / rider family_not_reliable,
oncoming / rider history_not_reliable, 5 % no history fit, carried flags (30 %, a quarter carried_lost), and 3 % of
window-families 'lost at once' (each clip, real or generated, carries the t0 value with probability 0.5: ties).
"""
from __future__ import annotations

import os

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_v, '2')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

import argparse, csv, json, math, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_bench_v1 as ab  # noqa: E402
import validate_calib_common as vc  # noqa: E402  validation-v1 implementation (independent helpers)

K = 16
TEST_COUNTS = {'lead': 65, 'lead_stop': 65, 'startup': 10, 'cutin': 59, 'ped': 79, 'oncoming': 69, 'ego_inter': 101, 'crossing': 26, 'rider': 12}
OUT = {'lead': [('x_4s', 'primary'), ('x_1s', 'secondary'), ('x_2s', 'secondary'), ('x_6s', 'secondary'), ('min_x_0_4s', 'secondary')],
       'lead_stop': [('x_4s', 'primary'), ('x_2s', 'secondary'), ('min_x_0_4s', 'secondary')],
       'startup': [('x_6s', 'descriptive'), ('x_2s', 'descriptive'), ('x_4s', 'descriptive')],
       'cutin': [('absy_2s', 'primary'), ('absy_4s', 'secondary'), ('x_2s', 'secondary'), ('x_4s', 'secondary'), ('entered_absy_lt_1p0_by_4s', 'secondary')],
       'ped': [('absy_2s', 'primary'), ('absy_1s', 'secondary'), ('x_1s', 'secondary'), ('x_2s', 'secondary'), ('in_absy_lt_1p5_within_2s', 'secondary')],
       'oncoming': [('x_1s', 'primary'), ('absy_1s', 'secondary'), ('out_of_view_by_2s', 'secondary')],
       'ego_inter': [('ego_dx_0_4s', 'primary'), ('ego_dyaw_deg_0_4s', 'secondary'), ('ego_dx_0_2s', 'secondary'), ('ego_dx_0_6s', 'secondary')],
       'crossing': [('x_1s', 'descriptive'), ('x_2s', 'descriptive'), ('y_1s', 'descriptive'), ('y_2s', 'descriptive'), ('out_of_view_by_2s', 'descriptive')],
       'rider': [('x_1s', 'descriptive'), ('x_2s', 'descriptive'), ('absy_1s', 'descriptive'), ('absy_2s', 'descriptive')]}
MISCAL = {'shift': {'lead': (0.396, 0.745), 'lead_stop': (0.25, 1.0), 'cutin': (0.0, 1.0), 'oncoming': (0.0, 0.6), 'ego_inter': (0.0, 1.0),
                    'ped': (0.3, 0.8), 'startup': (0.0, 1.0), 'crossing': (0.0, 1.0), 'rider': (0.0, 1.0)},
          'null': {f: (0.0, 1.0) for f in TEST_COUNTS}}
NOT_RELIABLE = {'ped': ['absy_2s', 'x_1s', 'x_2s'], 'crossing': ['x_1s', 'x_2s', 'y_1s', 'y_2s'], 'rider': ['x_1s', 'x_2s', 'absy_1s', 'absy_2s']}
HIST_NOT_RELIABLE = ('oncoming', 'rider')
FIELDS = ['window', 'family', 'sample', 'outcome', 'kind', 'value', 'scorable', 'flags', 'value_v3', 'value_v5', 'value_v8']


def state(fam, rng):
    s = lambda a, b: rng.uniform(a, b); side = rng.choice([-1.0, 1.0])
    if fam == 'lead':
        return dict(x0=s(8, 30), y0=rng.normal(0, 0.4), vx=rng.normal(-1.5, 2.0), vy=rng.normal(0, 0.2))
    if fam == 'lead_stop':
        return dict(x0=s(6, 25), y0=rng.normal(0, 0.4), vx=rng.normal(-3.0, 2.5), vy=rng.normal(0, 0.2))
    if fam == 'startup':
        return dict(x0=s(5, 15), y0=rng.normal(0, 0.3), vx=rng.normal(0.5, 1.0), vy=rng.normal(0, 0.1))
    if fam == 'cutin':
        return dict(x0=s(10, 35), y0=side * s(2.5, 4.5), vx=rng.normal(0, 1.5), vy=-side * abs(rng.normal(0.3, 0.4)))
    if fam == 'ped':
        return dict(x0=s(5, 25), y0=side * s(1.5, 8.0), vx=rng.normal(-1, 1.5), vy=rng.normal(0, 0.8))
    if fam == 'oncoming':
        return dict(x0=s(25, 60), y0=side * s(3, 6), vx=rng.normal(-15, 4), vy=rng.normal(0, 0.3))
    if fam == 'crossing':
        return dict(x0=s(8, 30), y0=side * s(5, 15), vx=rng.normal(-1, 1.5), vy=-side * abs(rng.normal(3, 2)))
    if fam == 'rider':
        return dict(x0=s(5, 25), y0=side * s(2, 6), vx=rng.normal(-1, 2), vy=rng.normal(0, 0.5))
    return dict(v=s(0, 12), w=rng.normal(0, 3))                           # ego_inter: speed (m/s), yaw rate (deg/s)


def horizon_of(o):
    k, h = ab.outcome_kind(o)
    return h if h is not None else {'entered_absy_lt_1p0_by_4s': 4.0, 'in_absy_lt_1p5_within_2s': 2.0, 'out_of_view_by_2s': 2.0}[o]


def draw_clip(fam, st, e, rng):
    """One clip's outcomes given standard-normal innovations e (dict horizon -> (ex, ey)); returns {outcome: value}."""
    v = {}
    if fam == 'ego_inter':
        for o, _ in OUT[fam]:
            k, h = ab.outcome_kind(o); s = 0.3 + 0.35 * h
            v[o] = st['v'] * h + (0.3 + 0.4 * h) * e[h][0] if k == 'egodx' else st['w'] * h + (1 + h) * e[h][1]
        return v
    xs, ys = {}, {}
    for h in (1.0, 2.0, 4.0, 6.0):
        s = 0.3 + 0.35 * h; xs[h] = st['x0'] + st['vx'] * h + s * e[h][0]; ys[h] = st['y0'] + st['vy'] * h + s * 0.4 * e[h][1]
    for o, _ in OUT[fam]:
        k, h = ab.outcome_kind(o)
        if k == 'x':
            v[o] = xs[h]
        elif k == 'absy':
            v[o] = abs(ys[h])
        elif k == 'y':
            v[o] = ys[h]
        elif k == 'minx':
            v[o] = min(st['x0'], xs[1.0], xs[2.0], xs[4.0])
        elif o == 'entered_absy_lt_1p0_by_4s':
            v[o] = float(min(abs(ys[1.0]), abs(ys[2.0]), abs(ys[4.0])) < 1.0)
        elif o == 'in_absy_lt_1p5_within_2s':
            v[o] = float(min(abs(ys[1.0]), abs(ys[2.0])) < 1.5)
        elif o == 'out_of_view_by_2s':
            v[o] = float(xs[2.0] < 8.0 or abs(ys[2.0]) > 0.6 * max(xs[2.0], 1.0))
    return v


def innovations(rng, delta, lam, real):
    """Per horizon a pair of standard normals: the real draw, or the generated one (-delta + lambda e)."""
    out = {}
    for h in (1.0, 2.0, 4.0, 6.0):
        a, b = rng.standard_normal(2)
        out[h] = (a, b) if real else (-delta + lam * a, -delta * 0 + lam * b)
    return out


def gen_window_family(fam, st, delta, lam, rng, lost):
    real = draw_clip(fam, st, innovations(rng, delta, lam, True), rng)
    gens = [draw_clip(fam, st, innovations(rng, delta, lam, False), rng) for _ in range(K)]
    if lost and fam != 'ego_inter':                       # lost at t0: the real and half the samples carry the t0 reading
        t0 = {o: (abs(st['y0']) if ab.outcome_kind(o)[0] == 'absy' else st['y0'] if ab.outcome_kind(o)[0] == 'y' else
                  (0.0 if ab.outcome_kind(o)[0] is None else st['x0'])) for o, _ in OUT[fam]}
        t0 = {o: round(v, 4) for o, v in t0.items()}
        lost_k = {k for k in range(K) if rng.random() < 0.5}; real_lost = bool(rng.random() < 0.5)   # every clip alike: exchangeable
        real = dict(t0) if real_lost else real; gens = [dict(t0) if k in lost_k else g for k, g in enumerate(gens)]
        return real, gens, real_lost, lost_k
    return real, gens, False, set()


def expected_values(scen, n_mc=20000, seed=5):
    """Monte Carlo expected E[PIT] - 0.5, P(above all), P(below all), P(outside) of each family's primary outcome."""
    rng = np.random.default_rng(seed); out = {}
    for fam in TEST_COUNTS:
        d, l = MISCAL[scen][fam]; o = OUT[fam][0][0]; pit = []; ab_ = []; be = []
        for _ in range(n_mc):
            st = state(fam, rng); lost = rng.random() < 0.03; real, gens, _, _ = gen_window_family(fam, st, d, l, rng, lost)
            y, Y = real[o], np.array([g[o] for g in gens])
            r = int((Y < y).sum()) + int(rng.integers(0, int((Y == y).sum()) + 1)); pit.append((r + rng.random()) / (K + 1)); ab_.append(r == K); be.append(r == 0)
        out[fam] = dict(outcome=o, delta=d, lam=l, theta=float(np.mean(pit) - 0.5), theta_mc_se=float(np.std(pit) / math.sqrt(n_mc)),
                        p_above=float(np.mean(ab_)), p_below=float(np.mean(be)), p_outside=float(np.mean(np.array(ab_) | np.array(be))))
    return out


def cmd_simulate(a):
    rng = np.random.default_rng(a.seed); out = a.out; (out / 'windows').mkdir(parents=True, exist_ok=True)
    scen = a.scenario; rows_gen, rows_clim = [], []

    def tracker_vals(v):
        if v is None:
            return '', '', '', ''
        t = [v + rng.normal(0, 0.15), v, v + rng.normal(0, 0.15)]; med = float(np.median(t))
        return round(med, 4), round(t[0], 4), round(t[1], 4), round(t[2], 4)

    def emit(rows, w, fam, sample, o, kind, v, flags, scorable, carried_t0=False):
        if v is not None and carried_t0:              # every tracker carries the same t0 reading: exact ties survive
            med = a3 = a5 = a8 = round(v, 4)
        else:
            med, a3, a5, a8 = tracker_vals(v) if v is not None and ab.outcome_kind(o)[0] is not None and fam != 'ego_inter' else ((round(v, 4) if v is not None else ''), '', '', '')
        rows.append(dict(window=w, family=fam, sample=sample, outcome=o, kind=kind, value=med, scorable=int(scorable), flags=';'.join(flags), value_v3=a3, value_v5=a5, value_v8=a8))

    def history(rows, w, fam, sample, st, hard_flags):
        if fam == 'ego_inter':
            for o, v in (('ego_hist_dx_0p4s', st['v'] * 0.4 + rng.normal(0, 0.05)), ('ego_hist_dyaw_deg_0p4s', st['w'] * 0.4 + rng.normal(0, 0.2))):
                rows.append(dict(window=w, family=fam, sample=sample, outcome=o, kind='history', value=round(v, 4), scorable=1, flags='history_group_16_20', value_v3='', value_v5='', value_v8=''))
            for o, v in (('ego_hist_ls_speed', st['v'] + rng.normal(0, 0.15)), ('ego_hist_ls_yaw_rate_dps', st['w'] + rng.normal(0, 0.3))):
                rows.append(dict(window=w, family=fam, sample=sample, outcome=o, kind='history', value=round(v, 4), scorable=1, flags='history_odometry_2s_not_gated', value_v3='', value_v5='', value_v8=''))
            return
        fit = rng.random() > 0.05; flags = list(hard_flags)
        if not fit:
            flags.append('no_history_fit')
        if fam in HIST_NOT_RELIABLE:
            flags.append('history_not_reliable')
        ok = int(fit and fam not in HIST_NOT_RELIABLE and not hard_flags)
        vals = [('hist_x_0s', st['x0'] + rng.normal(0, 0.2)), ('hist_y_0s', st['y0'] + rng.normal(0, 0.05)),
                ('hist_ls_x0', st['x0'] + rng.normal(0, 0.6) if fit else None), ('hist_ls_vx', st['vx'] + rng.normal(0, 1.0) if fit else None),
                ('hist_ls_y0', st['y0'] + rng.normal(0, 0.1) if fit else None), ('hist_ls_vy', st['vy'] + rng.normal(0, 0.15) if fit else None),
                ('hist_ls_n', 21.0 if fit else None), ('hist_ls_t_first', -2.0 if fit else None)]
        for o, v in vals:
            rows.append(dict(window=w, family=fam, sample=sample, outcome=o, kind='history', value='' if v is None else round(v, 4), scorable=ok, flags=';'.join(flags), value_v3='', value_v5='', value_v8=''))

    def make(prefix, n_w, counts, gen):
        names = [f'{prefix}-{i:04d}-t2.5' for i in range(n_w)]; roles = defaultdict(list)
        for fam, c in counts.items():
            for i in rng.choice(n_w, size=c, replace=False):
                roles[names[i]].append(fam)
        for wi, w in enumerate(names):
            ego_v = rng.uniform(0, 12); targets = {}
            for fam in sorted(roles[w], key=list(TEST_COUNTS).index):
                st = state(fam, rng)
                if fam == 'ego_inter':
                    st['v'] = ego_v
                else:
                    targets[fam] = dict(x_m=st['x0'] + 2.0, y_m=st['y0'], rear_face_m=st['x0'], vx_mps=st['vx'] + ego_v + rng.normal(0, 0.1), vy_mps=st['vy'] + rng.normal(0, 0.05),
                                        category=str(rng.choice(['REGULAR_VEHICLE', 'REGULAR_VEHICLE', 'REGULAR_VEHICLE', 'LARGE_VEHICLE'])) if fam not in ('ped', 'rider') else
                                        ('PEDESTRIAN' if fam == 'ped' else 'BICYCLIST'))
                d, l = MISCAL[scen][fam]; lost = rng.random() < 0.03; real, gens, real_lost, lost_k = gen_window_family(fam, st, d, l, rng, lost)
                real_t0 = rng.random() < 0.03
                for o, kind in OUT[fam]:
                    base = []
                    if fam in NOT_RELIABLE:
                        base.append('family_not_reliable')
                        if o in NOT_RELIABLE[fam]:
                            base.append('outcome_not_reliable')
                    carried = lambda: (['carried', 'carried_lost' if rng.random() < 0.25 else 'carried_edge'] if (fam != 'ego_inter' and rng.random() < 0.3) or lost else [])
                    samples = [('real', real[o]), ('raw', real[o] * (1 + rng.normal(0, 0.01)) if ab.outcome_kind(o)[0] not in (None,) else real[o])] if gen else [('raw', real[o])]
                    if gen:
                        samples += [(str(k), gens[k][o]) for k in range(K)]
                    for s, v in samples:
                        fl = list(base) + carried()
                        if real_t0 and gen:
                            fl.append('real_t0_not_found')
                            if s == 'real':
                                fl.append('t0_not_found')
                        if s not in ('real', 'raw') and rng.random() < 0.02:
                            if rng.random() < 0.5:
                                fl.append('no_value'); v = None
                            else:
                                fl.append('t0_not_found')
                        hard = set(fl) & ab.HARD
                        if fam == 'ego_inter':
                            fl = [x for x in fl if x != 'carried' and not x.startswith('carried')] + (['vo_low_support'] if rng.random() < 0.08 else [])
                        ct0 = (s in ('real', 'raw') and real_lost) or (s not in ('real', 'raw') and int(s) in lost_k)
                        emit(rows_gen if gen else rows_clim, w, fam, s, o, kind, v, fl, not hard, carried_t0=ct0 and s != 'raw')
                for s in (('real', 'raw') if gen else ('raw',)):
                    hard = (['real_t0_not_found'] if real_t0 and gen else []) + (['family_not_reliable'] if fam in NOT_RELIABLE else [])
                    history(rows_gen if gen else rows_clim, w, fam, s, st, hard)
            meta = dict(name=w, split='SIM-TEST' if gen else 'SIM-DEV', log=f'simlog-{prefix}-{wi:04d}', roles=roles[w],
                        past_covariates=dict(ego_speed_mps=ego_v, ego_speed_05_mps=ego_v, ego_accel_mps2=0.0), targets=targets,
                        selection=f'check_analyze_bench_v1.py simulate ({scen}, seed {a.seed})')
            (out / 'windows' / w).mkdir(parents=True, exist_ok=True); (out / 'windows' / w / 'window.json').write_text(json.dumps(meta, indent=1) + '\n')
        return names

    make('simtest', 200, TEST_COUNTS, True)
    make('simdev', 220, {f: int(round(c * 1.1)) for f, c in TEST_COUNTS.items()}, False)
    first = [r for r in rows_gen if r['sample'] in ('real', 'raw') or int(r['sample']) <= 10]
    second = [r for r in rows_gen if r['sample'] in ('real', 'raw') or int(r['sample']) >= 11]
    for name, rows in (('sim-k0-10.csv', first), ('sim-k11-15.csv', second), ('sim-dev-raw-climatology.csv', rows_clim)):
        with (out / name).open('w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    exp = expected_values(scen) if not a.no_expected else {}
    (out / 'truth.json').write_text(json.dumps(dict(scenario=scen, seed=a.seed, K=K, miscalibration=MISCAL[scen], expected_primary=exp, counts=TEST_COUNTS,
                                                    note='expected values: Monte Carlo of the same generator (20,000 window-families per family), '
                                                         'without scorability drops'), indent=1) + '\n')
    print(json.dumps(dict(out=str(out), rows_runs=len(rows_gen), rows_clim=len(rows_clim), expected={f: round(v['theta'], 4) for f, v in exp.items()})))
    return exp


# --------------------------------------------------------------------------------------------------------- crosscheck
def cmd_crosscheck(a):
    res = json.loads((a.results / 'results.json').read_text()); data, kinds = ab.load_runs(a.runs); gids = ab.generated_ids(data); Kr = len(gids)
    report = dict(results=str(a.results), per_family={}, worst={})
    keys = ('mean_pit_minus_half', 'slope_z', 'convexity_z', 'chi2', 'mean_fair_crps')
    worst = defaultdict(float)
    for fam, d in res['families_results'].items():
        rows, _ = ab.build(data, fam, d['outcome'], gids); ab.rank_rows(rows)
        if not rows:
            continue
        Ki = np.array([len(r['Y']) for r in rows])
        if not (Ki == Kr).all():
            note = 'K_i varies: shares and their tests use Poisson-binomial nulls here, not comparable with stats_block'
        else:
            note = ''
        rank = np.array([r['rank'] for r in rows]); pit = np.array([r['pit'] for r in rows]); crps = np.array([vc.awc.fair_crps(r['Y'], r['y']) for r in rows])
        b = vc.stats_block(ab.pit_bins(rows, Kr), pit, Kr, np.random.default_rng(1), crps=crps, mc=20000, boot=2000)
        mine = d['stats']; diffs = {k: abs(mine[k] - b[k]) for k in keys if k in b}
        if (Ki == Kr).all():
            for lab in ('above_all', 'below_all', 'outside'):
                diffs[f'{lab} share'] = abs(mine[lab]['share'] - b[lab]['share']); diffs[f'{lab} p two-sided'] = abs(mine[lab]['p_two_sided'] - b[lab]['p_binom_two_sided'])
                ci = b[lab]['clopper_pearson95'] if mine['n'] <= 45 else b[lab]['wilson95']
                diffs[f'{lab} CI'] = max(abs(mine[lab]['ci95'][0] - ci[0]), abs(mine[lab]['ci95'][1] - ci[1]))
            diffs['mean PIT z p'] = abs(mine['mean_pit_p_two_sided'] - b['mean_pit_p_two_sided'])
        report['per_family'][fam] = dict(n=mine['n'], max_abs_diff=diffs, note=note)
        for k, v in diffs.items():
            worst[k] = max(worst[k], v)
    report['worst'] = dict(worst)
    # H2: the reported Monte Carlo p against an exact permutation p from independent code (validate_sim_h2's idea, units of 1/60)
    h2 = res['confirmatory'].get('H2', {})
    report['H2'] = dict(T=h2.get('T'), p_mc=h2.get('p_mc'), p_exact_fft_reported=h2.get('p_exact_fft'), B=h2.get('B'))
    if a.truth:
        tr = json.loads(Path(a.truth).read_text()); kt = {}
        for fam, e in tr['expected_primary'].items():
            s = res['families_results'].get(fam, {}).get('stats', {})
            if not s.get('n'):
                continue
            se = (s['mean_pit_ci95'][1] - s['mean_pit_ci95'][0]) / (2 * 1.96)
            kt[fam] = dict(n=s['n'], expected_theta=e['theta'], estimate=s['mean_pit_minus_half'], ci95=s['mean_pit_ci95'], z_vs_expected=(s['mean_pit_minus_half'] - e['theta']) / se,
                           ci_covers_expected=bool(s['mean_pit_ci95'][0] <= e['theta'] <= s['mean_pit_ci95'][1]),
                           expected_above=e['p_above'], above=s['above_all']['share'], expected_below=e['p_below'], below=s['below_all']['share'])
        # expected H2 statistic: mean over the H2 windows of the mean expected outside probability of their pool families
        wins = res['confirmatory'].get('H2', {})
        exp_out = {f: tr['expected_primary'][f]['p_outside'] for f in ab.H2_POOL}
        fam_of = defaultdict(list)
        for fam in ab.H2_POOL:
            rows, _ = ab.build(data, fam, ab.PRIMARY[fam], gids)
            for r in rows:
                fam_of[r['name']].append(fam)
        eT = float(np.mean([np.mean([exp_out[f] for f in fs]) for fs in fam_of.values()]))
        report['known_answers'] = dict(primary=kt, H2=dict(expected_T_approx=eT, T=wins.get('T'), boot_ci95=wins.get('boot_ci95'), null_mean=wins.get('null_mean')),
                                       H1=res['confirmatory'].get('H1'), H2_decision=wins.get('confirmed'))
    out = a.results / 'crosscheck.json'; out.write_text(json.dumps(report, indent=1, default=float) + '\n')
    print(json.dumps(dict(worst=report['worst'], H2=report['H2'], known=report.get('known_answers', {}).get('primary'), H2_known=report.get('known_answers', {}).get('H2')), indent=1, default=float))


def cmd_pilot(a):
    """study-v1 TEST g6, lead gap at 4 s, through analyze_bench_v1's ranks and statistics."""
    rows_m = vc.load_measure(vc.STUDY / 'results-test-g6' / 'measure-ensemble.json', 'test-g6')
    rows = [dict(name=r['name'], y=r['y'], ids=list(range(16)), Y=r['Y'], K_full=16) for r in rows_m]   # name order, as the frozen pipeline
    ab.rank_rows(rows); s = ab.family_stats(rows, 16, 100000, 10000, ('pilot',))
    fz = json.loads((vc.STUDY / 'results-test-g6' / 'primary' / 'summary.json').read_text())
    frozen_ranks = [w['rank'] for w in json.loads((vc.STUDY / 'results-test-g6' / 'primary' / 'windows.json').read_text())]
    cmp = dict(mean_pit_minus_half=(s['mean_pit_minus_half'], fz['mean_pit_minus_half']), slope_z=(s['slope_z'], fz['slope_z']), convexity_z=(s['convexity_z'], fz['convexity_z']),
               chi2=(s['chi2'], fz['chi2']), above_all=(s['above_all']['share'], fz['share_real_above_all']), below_all=(s['below_all']['share'], fz['share_real_below_all']),
               fair_crps=(s['mean_fair_crps'], fz['mean_fair_crps_m']), ranks_identical=[r['rank'] for r in rows] == frozen_ranks,
               bootstrap_ci_new=s['mean_pit_ci95'], bootstrap_ci_frozen=fz['mean_pit_ci95'])
    print(json.dumps(cmp, indent=1, default=float))
    return cmp


def cmd_gates(a):
    """The TEST gates and the protocol exclusion, on tiny synthetic rows that use the excluded TEST window's name
    (no TEST file is read: window metadata comes from an empty scratch directory)."""
    import subprocess
    d = a.scratch; d.mkdir(parents=True, exist_ok=True); (d / 'nowindows').mkdir(exist_ok=True)
    rng = np.random.default_rng(3); rows = []
    for w in ('1c98497e-t2.5', 'simgate-0001-t2.5', 'simgate-0002-t2.5'):
        for fam, o in (('lead', 'x_4s'), ('ped', 'absy_2s')):
            for smp in ['real'] + [str(k) for k in range(4)]:
                rows.append(dict(window=w, family=fam, sample=smp, outcome=o, kind='primary', value=round(10 + rng.normal(), 4), scorable=1, flags='',
                                 value_v3='', value_v5='', value_v8=''))
    csv_path = d / 'gate.csv'
    with csv_path.open('w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=FIELDS); wr.writeheader(); wr.writerows(rows)
    (d / 'done-false.json').write_text(json.dumps(dict(complete=False)) + '\n'); (d / 'done-true.json').write_text(json.dumps(dict(complete=True)) + '\n')
    base = [sys.executable, str(HERE / 'analyze_bench_v1.py'), '--runs', str(csv_path), '--windows-dir', str(d / 'nowindows'), '--label', 'GATE-CHECK',
            '--boot', '200', '--mc', '500', '--mc-explore', '200', '--perm', '500', '--tie-draws', '20', '--pilot-summary', str(d / 'none.json')]
    cases = [('no --allow-test', [], False), ('done file missing', ['--allow-test', '--done-file', str(d / 'missing.done')], False),
             ('done complete=false', ['--allow-test', '--done-file', str(d / 'done-false.json')], False),
             ('done complete=false + --force', ['--allow-test', '--done-file', str(d / 'done-false.json'), '--force'], True),
             ('done complete=true', ['--allow-test', '--done-file', str(d / 'done-true.json')], True)]
    out = {}
    for i, (lab, extra, should_run) in enumerate(cases):
        od = d / f'out-{i}'
        r = subprocess.run(base + extra + ['--out', str(od)], capture_output=True, text=True)
        ran = r.returncode == 0; res = json.loads((od / 'results.json').read_text()) if ran else None
        out[lab] = dict(expected_to_run=should_run, ran=ran, ok=ran == should_run, message=(r.stderr.strip().splitlines() or [''])[-1][:200],
                        test_completeness=res and res['test_completeness'], exclusions=res and res['protocol_exclusions'],
                        families=res and sorted(res['families_results']), ped_windows=res and res['families_results'].get('ped', {}).get('stats', {}).get('n'),
                        lead_windows=res and res['families_results'].get('lead', {}).get('stats', {}).get('n'))
    out['all_ok'] = all(v['ok'] for v in out.values() if isinstance(v, dict))
    (d / 'gates.json').write_text(json.dumps(out, indent=1) + '\n'); print(json.dumps(out, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('simulate'); s.add_argument('--out', required=True, type=Path); s.add_argument('--scenario', choices=('shift', 'null'), default='shift'); s.add_argument('--seed', type=int, default=11); s.add_argument('--no-expected', action='store_true')
    c = sub.add_parser('crosscheck'); c.add_argument('--results', required=True, type=Path); c.add_argument('--runs', nargs='+', required=True, type=Path); c.add_argument('--truth')
    sub.add_parser('pilot')
    g = sub.add_parser('gates'); g.add_argument('--scratch', required=True, type=Path)
    a = p.parse_args()
    dict(simulate=cmd_simulate, crosscheck=cmd_crosscheck, pilot=cmd_pilot, gates=cmd_gates)[a.cmd](a)


if __name__ == '__main__':
    main()
