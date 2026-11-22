"""Spread ratio of the primary quantities, and its stability under subsampling the number of samples K.

spread ratio = E|X - X'| / E|X - Y|: the mean distance between two imagined readings over the mean distance
from an imagined reading to the real one, as a ratio of sums over the cases of a family, then averaged over
the model's scored families. If the samples follow the real distribution, Y is one more draw and the ratio is
1 for any K, because both terms are estimated without bias from any K >= 2 samples (the two terms of the fair
CRPS). Below 1: the samples sit closer to each other than to reality.

Descriptive (not pre-registered). Reads the frozen outcomes under $DOB_ROOT/outputs/wm-calibration/bench-v1 and the
case pool $DOB_FIGURE_OUT/case_ranks.csv (written by scripts/export_supply.py). Writes spread_ratio.csv,
spread_ratio_family.csv and their .md files to $DOB_FIGURE_OUT (default figure_out).
"""
import argparse
import csv
import os
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np

BENCH = Path(os.environ.get('DOB_ROOT', '.'), 'outputs/wm-calibration/bench-v1')
DATA = Path(os.environ.get('DOB_FIGURE_OUT', 'figure_out'))
FILES = {
    'orbis2': ['orbis2/instrument/outcomes/test/outcomes-orbis2-test-k16.csv'],
    'cosmos3-nano': ['instrument/outcomes/test/outcomes-test-k0-10.csv', 'instrument/outcomes/test/outcomes-test-k11-15.csv'],
    'cosmos3-edge': ['edge/instrument/outcomes/test/outcomes-edge-test-k0-15.csv'],
    'reworld-video': ['reworld-video/stage2-p84/outcomes/test/outcomes-reworld-test-k16.csv'],
    'wan21-vace': ['wan21-vace13b/test-v1/instrument/outcomes/test/outcomes-vace832-test-k16.csv'],
    'drivingworld': ['drivingworld/stage2-p84/outcomes/test/outcomes-drivingworld-test-k16.csv'],
    'epona': ['epona/instrument/outcomes/test/outcomes-epona-test-b16-k16.csv'],
    'ltx25': ['ltx25d-2s/test/instrument/outcomes/test/outcomes-ltx25d-test-k16.csv'],
    'vavim': ['vavim/stage2-p84/outcomes/test/outcomes-vavim-test-k16.csv'],
    'drivelaw-video': ['drivelaw-video/stage2-p84/outcomes/test/outcomes-drivelaw-test-k16.csv'],
    'vista': ['vista/instrument/outcomes/test/outcomes-vista-test-h0-k16.csv'],
}
PRIMARY = {'lead': 'x_4s', 'lead_stop': 'x_4s', 'cutin': 'absy_2s', 'oncoming': 'x_1s', 'ego_inter': 'ego_dx_0_4s'}
KS = [2, 4, 8, 16]
REPEATS = 200
BOOT = 2000
SEED = 2026


def load(method, pool):
    vals = defaultdict(dict)
    for name in FILES[method]:
        for r in csv.DictReader(open(BENCH / name)):
            key = (r['window'], r['family'])
            if r['kind'] == 'primary' and r['outcome'] == PRIMARY.get(r['family']) and key in pool:
                vals[key][r['sample']] = float(r['value'])
    keys = sorted(pool)
    X = np.array([[vals[k][str(i)] for i in range(16)] for k in keys])
    y = np.array([vals[k]['real'] for k in keys])
    below = (X < y[:, None]).sum(1)
    assert (below == np.array([pool[k] for k in keys])).all(), method  # same readings as case_ranks.csv
    return keys, X, y


def terms(Xs, y):
    K = Xs.shape[1]
    spread = np.abs(Xs[:, :, None] - Xs[:, None, :]).sum((1, 2)) / (K * (K - 1))
    error = np.abs(Xs - y[:, None]).mean(1)
    return spread, error


def ratio(spread, error, fam_idx):
    return float(np.mean([spread[ix].sum() / error[ix].sum() for ix in fam_idx]))


def outside(Xs, y, win_idx, rng):
    K = Xs.shape[1]
    below = (Xs < y[:, None]).sum(1)
    ties = (Xs == y[:, None]).sum(1)
    rank = below + np.floor(rng.random(len(y)) * (ties + 1)).astype(int)
    out = (rank == 0) | (rank == K)
    return float(np.mean([out[ix].mean() for ix in win_idx]))


def subsets(n, k, rng, offset=0, pool=16):
    return offset + np.argsort(rng.random((n, pool)), 1)[:, :k]


def main():
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    rng = np.random.default_rng(SEED)
    pools = defaultdict(dict)
    for r in csv.DictReader(open(DATA / 'case_ranks.csv')):
        pools[r['method']][(r['window_id'], r['family'])] = int(r['n_below'])
    rows, fam_rows = [], []
    for method in FILES:
        keys, X, y = load(method, pools[method])
        fams = sorted({f for _, f in keys}, key=list(PRIMARY).index)
        fam_idx = [np.array([i for i, k in enumerate(keys) if k[1] == f]) for f in fams]
        windows = sorted({w for w, _ in keys})
        win_idx = [np.array([i for i, k in enumerate(keys) if k[0] == w]) for w in windows]
        for k in KS:
            reps = 1 if k == 16 else REPEATS
            sr, osh, ctrl = [], [], []
            for _ in range(reps):
                Xs = X if k == 16 else np.take_along_axis(X, subsets(len(y), k, rng), 1)
                s, e = terms(Xs, y)
                sr.append(ratio(s, e, fam_idx))
                osh.append(outside(Xs, y, win_idx, rng))
            kc = min(k, 15)  # positive control: imagined reading 0 as the observation, kc of the other 15
            for _ in range(1 if kc == 15 else REPEATS):
                Xc = np.take_along_axis(X, subsets(len(y), kc, rng, offset=1, pool=15), 1)
                s, e = terms(Xc, X[:, 0])
                ctrl.append(ratio(s, e, fam_idx))
            row = dict(method=method, k=k, spread_ratio=np.mean(sr), spread_ratio_ci_low='', spread_ratio_ci_high='',
                       outside_share=np.mean(osh), outside_excess=np.mean(osh) - 2 / (k + 1),
                       control_spread_ratio=np.mean(ctrl), control_k=kc, n_cases=len(y), n_windows=len(windows))
            if k == 16:
                s, e = terms(X, y)
                boot = []
                for _ in range(BOOT):  # resample histories, keep each history's cases together
                    pick = np.concatenate([win_idx[j] for j in rng.integers(0, len(win_idx), len(win_idx))])
                    fam_of = np.array([fams.index(keys[i][1]) for i in pick])
                    parts = [pick[fam_of == j] for j in range(len(fams)) if (fam_of == j).any()]
                    boot.append(ratio(s, e, parts))
                row['spread_ratio_ci_low'], row['spread_ratio_ci_high'] = np.percentile(boot, [2.5, 97.5])
                for f, ix in zip(fams, fam_idx):
                    fam_rows.append(dict(method=method, family=f, quantity=PRIMARY[f], n=len(ix),
                                         spread_ratio=s[ix].sum() / e[ix].sum(), mean_pair_distance_m=s[ix].mean(),
                                         mean_error_m=e[ix].mean()))
            rows.append(row)
    fmt = lambda v: f'{v:.6g}' if isinstance(v, float) else v
    with open(DATA / 'spread_ratio.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows({a: fmt(float(b)) if isinstance(b, (float, np.floating)) else b for a, b in r.items()} for r in rows)
    with open(DATA / 'spread_ratio_family.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(fam_rows[0]))
        w.writeheader()
        w.writerows({a: fmt(float(b)) if isinstance(b, (float, np.floating)) else b for a, b in r.items()} for r in fam_rows)
    today = date.today().isoformat()
    produced = (f'- **Produced by:** `python scripts/spread_ratio.py`, run on {today}, from the frozen TEST outcomes under '
                f'$DOB_ROOT/outputs/wm-calibration/bench-v1. Seed {SEED}; {REPEATS} random subsets of K samples per case for K < 16; '
                f'{BOOT} history-bootstrap draws for the interval.')
    (DATA / 'spread_ratio.md').write_text(f"""# spread_ratio.csv

One row is one model and one number of samples K: the spread ratio of the primary quantities, and the outside share, when only K of the 16 imagined readings are used. {len(rows)} rows.

| column | meaning | values or unit |
|---|---|---|
| method | the model | keys of `methods.csv` |
| k | number of imagined readings used (a random subset of the 16; all 16 when k = 16) | 2, 4, 8, 16 |
| spread_ratio | mean distance between two imagined readings over the mean distance from an imagined reading to the real one (ratio of sums over the cases of a family), averaged over the model's scored families; averaged over the random subsets | ratio; 1 if the samples follow the real distribution, for any k; below 1: samples too close together for how far reality lies |
| spread_ratio_ci_low | 95 % history-bootstrap interval, lower end (k = 16 only) | ratio; empty for k < 16 |
| spread_ratio_ci_high | 95 % history-bootstrap interval, upper end (k = 16 only) | ratio; empty for k < 16 |
| outside_share | share of cases with the real reading outside all k imagined readings, window-weighted as H2 (ties split at random) | share 0-1; 2/(k+1) if the samples follow the real distribution; at k = 16 it equals H2 within about 1 point |
| outside_excess | outside_share - 2/(k+1) | share; 0 if the samples follow the real distribution |
| control_spread_ratio | positive control: imagined reading 0 as the observation against control_k of the other 15 | ratio; reference 1 |
| control_k | the number of imagined readings in the control (15 when k = 16) | 2, 4, 8, 15 |
| n_cases | cases (the case pool of data/case_ranks.csv) | count |
| n_windows | TEST histories with at least one case | count |

{produced}
- **Filters:** the case pool of `data/case_ranks.csv` (the real and all 16 readings valid; five scored families; VaViM lead and lead_stop only; DrivingWorld without ego_inter).
- **Paired:** the rows of one model share its cases; k = 16 uses all samples, k < 16 random subsets of them.
- **Empty cells:** the interval columns for k < 16.
- **Note:** descriptive, not pre-registered. The spread ratio stays within about 0.01 across k for every model, while outside_share and outside_excess change with k.
""")
    (DATA / 'spread_ratio_family.md').write_text(f"""# spread_ratio_family.csv

One row is one model and one scored family: the spread ratio of the family's primary quantity with all 16 imagined readings. {len(fam_rows)} rows.

| column | meaning | values or unit |
|---|---|---|
| method | the model | keys of `methods.csv` |
| family | the scored family | lead, lead_stop, cutin, oncoming, ego_inter |
| quantity | the family's primary quantity | x_4s, absy_2s, x_1s, ego_dx_0_4s |
| n | cases | count |
| spread_ratio | sum over cases of the mean distance between two imagined readings, over the sum of the mean distance from an imagined reading to the real one | ratio; 1 if the samples follow the real distribution |
| mean_pair_distance_m | mean over cases of the mean distance between two imagined readings | m |
| mean_error_m | mean over cases of the mean distance from an imagined reading to the real one | m |

{produced}
- **Filters:** the case pool of `data/case_ranks.csv`.
- **Paired:** no.
- **Empty cells:** none.
- **Derived values:** spread_ratio in `data/spread_ratio.csv` (k = 16) = mean of this file's spread_ratio over the model's rows.
""")


if __name__ == '__main__':
    main()
