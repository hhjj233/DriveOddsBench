#!/usr/bin/env python3
"""Shared helpers of the validation-v1 experiments of the world-model calibration benchmark (2026-11-01; EXPLORATORY).

The statistics are those of bench-v1 protocol-v1.md section 5. The study's own functions are imported from
analyze_wm_calibration.py and used unchanged (ranks_of, jp_contrasts, jp_scores, chi2, fair_crps, bh, test_block).
Vectorised versions, needed for 10,000-replication simulations, are defined here and checked against the imported
functions by `self_check()` (identical numbers on random inputs; ranks identical in distribution, since the
tie-break draws differ in order).

Added statistics, as protocol section 5 marks them "new":
  - mean PIT - 0.5 with a z test using the exact null variance 1/(12 n), plus a window-bootstrap percentile CI;
  - chi-square on K+1 bins with a Monte Carlo p-value from multinomial draws under H0;
  - above-all / below-all shares against 1/(K+1) with an exact binomial test and Wilson / Clopper-Pearson / Wald
    intervals; the two-sided outside share against 2/(K+1);
  - sharpness (mean within-window SD of the generated readings).
Pilot data loaders read the frozen measure files of study-v1 (never written to).
"""
from __future__ import annotations

import os

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '2')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

import json, math, sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_wm_calibration as awc  # noqa: E402  (ranks_of, jp_contrasts, jp_scores, chi2, fair_crps, bh, test_block)

STUDY = Path(os.environ.get('DOB_ROOT', '.'), 'outputs/wm-calibration/study-v1')
OUT_ROOT = Path(os.environ.get('DOB_ROOT', '.'), 'outputs/wm-calibration/validation-v1')
Z975 = stats.norm.ppf(0.975)


# ----------------------------------------------------------------------------------------------- vectorised versions
def ranks_vec(y, Y, rng):
    """Rank of y among the last axis of Y with randomised ties, as awc.ranks_of: #{Y < y} + uniform on 0..#{Y == y}."""
    y = np.asarray(y, float); Y = np.asarray(Y, float)
    below = (Y < y[..., None]).sum(-1); tied = (Y == y[..., None]).sum(-1)
    return below + np.floor(rng.random(below.shape) * (tied + 1)).astype(np.int64)


def pit_of(rank, K, rng):
    """Randomised PIT (rank + U) / (K + 1); K may be an array (variable ensemble size)."""
    rank = np.asarray(rank); return (rank + rng.random(rank.shape)) / (np.asarray(K) + 1)


def counts_of(rank, K):
    """Rank histogram(s) on K + 1 bins; rank (..., n) -> (..., K + 1)."""
    rank = np.asarray(rank)
    if rank.ndim == 1:
        return np.bincount(rank, minlength=K + 1).astype(float)
    flat = rank + (K + 1) * np.arange(rank.shape[0])[:, None]
    return np.bincount(flat.ravel(), minlength=(K + 1) * rank.shape[0]).reshape(rank.shape[0], K + 1).astype(float)


def jp_vec(counts, K):
    """Jolliffe-Primo slope and convexity z for count rows, identical to awc.jp_scores."""
    c1, c2 = awc.jp_contrasts(K); counts = np.asarray(counts, float); N = counts.sum(-1); p = 1.0 / (K + 1)
    return counts @ c1 / np.sqrt(N * p * (c1 @ c1)), counts @ c2 / np.sqrt(N * p * (c2 @ c2))


def chi2_vec(counts, K):
    counts = np.asarray(counts, float); e = counts.sum(-1, keepdims=True) / (K + 1)
    return ((counts - e) ** 2 / e).sum(-1)


def fair_crps_vec(Y, y):
    """Fair (unbiased) ensemble CRPS (Ferro 2014) along the last axis, identical to awc.fair_crps."""
    Y = np.asarray(Y, float); y = np.asarray(y, float); K = Y.shape[-1]
    Ys = np.sort(Y, -1); w = 2 * np.arange(K) - K + 1
    return np.abs(Y - y[..., None]).mean(-1) - 2 * (Ys * w).sum(-1) / (2 * K * (K - 1))


def ens_crps_vec(Y, y):
    """Ordinary ensemble CRPS (CRPS of the empirical distribution of the K members), biased towards narrow ensembles."""
    Y = np.asarray(Y, float); y = np.asarray(y, float); K = Y.shape[-1]
    Ys = np.sort(Y, -1); w = 2 * np.arange(K) - K + 1
    return np.abs(Y - y[..., None]).mean(-1) - 2 * (Ys * w).sum(-1) / (2 * K * K)


def gauss_crps(mu, sigma, y):
    """Closed-form CRPS of N(mu, sigma^2) at y."""
    z = (np.asarray(y, float) - mu) / sigma
    return sigma * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / math.sqrt(math.pi))


# ------------------------------------------------------------------------------------------------ intervals and tests
def wilson(k, n, z=Z975):
    k = np.asarray(k, float); n = np.asarray(n, float); p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def wald(k, n, z=Z975):
    k = np.asarray(k, float); n = np.asarray(n, float); p = k / n; h = z * np.sqrt(p * (1 - p) / n)
    return p - h, p + h


def clopper_pearson(k, n, alpha=0.05):
    k = np.asarray(k, float); n = np.asarray(n, float)
    lo = np.where(k > 0, stats.beta.ppf(alpha / 2, np.maximum(k, 1e-12), n - k + 1), 0.0)
    hi = np.where(k < n, stats.beta.ppf(1 - alpha / 2, k + 1, np.maximum(n - k, 1e-12)), 1.0)
    return lo, hi


def binom_p_table(n, p):
    """Exact binomial p-values for every count 0..n: two-sided (scipy 'minlike'), greater, less."""
    two = np.array([stats.binomtest(k, n, p, alternative='two-sided').pvalue for k in range(n + 1)])
    ks = np.arange(n + 1); greater = stats.binom.sf(ks - 1, n, p); less = stats.binom.cdf(ks, n, p)
    return dict(two_sided=two, greater=greater, less=less)


def poisson_binomial_pmf(ps):
    """pmf of a sum of independent Bernoulli(p_i) (dynamic programming)."""
    pmf = np.zeros(len(ps) + 1); pmf[0] = 1.0
    for i, p in enumerate(ps):
        pmf[1:i + 2] = pmf[1:i + 2] * (1 - p) + pmf[0:i + 1] * p; pmf[0] *= (1 - p)
    return pmf


def poisson_binomial_tests(k, ps):
    """Exact tests of a count k against a sum of Bernoulli(p_i): greater, less, two-sided (minlike)."""
    pmf = poisson_binomial_pmf(ps); tol = 1e-12
    return dict(greater=float(pmf[k:].sum()), less=float(pmf[:k + 1].sum()), two_sided=float(min(1.0, pmf[pmf <= pmf[k] * (1 + tol)].sum())))


def mean_pit_tests(pit):
    """z test of mean PIT = 0.5 with the exact null variance 1/(12 n) (randomised PIT is exactly U(0, 1) under H0)."""
    n = len(pit); z = (np.mean(pit) - 0.5) / math.sqrt(1 / (12 * n))
    return dict(z=float(z), p_two_sided=float(2 * stats.norm.sf(abs(z))), p_one_sided_high=float(stats.norm.sf(z)))


def boot_ci(x, rng, boot=10000, fun=np.mean):
    x = np.asarray(x, float); idx = rng.integers(0, len(x), (boot, len(x)))
    b = fun(x[idx], axis=1) if fun in (np.mean, np.median) else np.array([fun(x[i]) for i in idx])
    return [float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))]


def stats_block(rank, pit, K, rng, crps=None, sd=None, mc=100000, boot=10000):
    """Every protocol-section-5 statistic for one set of windows with a common ensemble size K.

    Uses awc.test_block (chi-square, Jolliffe-Primo z with Monte Carlo p-values) unchanged, plus the "new" statistics.
    """
    rank = np.asarray(rank, int); pit = np.asarray(pit, float); n = len(rank)
    tb = awc.test_block(rank, K, rng, mc)
    out = dict(n=n, K=K, counts=tb['counts'])
    mp = mean_pit_tests(pit)
    out.update(mean_pit_minus_half=float(pit.mean() - 0.5), mean_pit_ci95=[v - 0.5 for v in boot_ci(pit, rng, boot)],
               mean_pit_z=mp['z'], mean_pit_p_two_sided=mp['p_two_sided'], mean_pit_p_one_sided_high=mp['p_one_sided_high'])
    out.update(slope_z=tb['slope_z'], slope_p_one_sided_mc=tb['slope_p_one_sided_mc'], slope_p_one_sided_normal=float(stats.norm.sf(tb['slope_z'])),
               slope_p_two_sided_normal=float(2 * stats.norm.sf(abs(tb['slope_z']))),
               convexity_z=tb['convexity_z'], convexity_p_two_sided_mc=tb['convexity_p_two_sided_mc'], convexity_p_two_sided_normal=float(2 * stats.norm.sf(abs(tb['convexity_z']))),
               chi2=tb['chi2'], chi2_p_mc=tb['chi2_p_mc'], chi2_p_asymptotic=float(stats.chi2.sf(tb['chi2'], K)))
    p1 = 1.0 / (K + 1)
    for label, k, p0 in (('above_all', int((rank == K).sum()), p1), ('below_all', int((rank == 0).sum()), p1), ('outside', int(((rank == K) | (rank == 0)).sum()), 2 * p1)):
        w = wilson(k, n); cp = clopper_pearson(k, n)
        bt = stats.binomtest(k, n, p0)
        out[label] = dict(count=k, share=k / n, expected=p0, ratio_to_expected=(k / n) / p0, p_binom_two_sided=float(bt.pvalue),
                          p_binom_greater=float(stats.binomtest(k, n, p0, alternative='greater').pvalue), wilson95=[float(w[0]), float(w[1])], clopper_pearson95=[float(cp[0]), float(cp[1])])
    if crps is not None:
        crps = np.asarray(crps, float); out.update(mean_fair_crps=float(crps.mean()), mean_fair_crps_ci95=boot_ci(crps, rng, boot))
    if sd is not None:
        out.update(sharpness_mean_sd=float(np.mean(sd)), sharpness_median_sd=float(np.median(sd)))
    return out


def fires(block, alpha=0.05):
    """Which tests of a stats_block reject at alpha (two-sided except where marked)."""
    return {'mean PIT z (2-sided)': block['mean_pit_p_two_sided'] < alpha,
            'mean PIT bootstrap CI excludes 0': not (block['mean_pit_ci95'][0] <= 0 <= block['mean_pit_ci95'][1]),
            'JP slope (1-sided, H1 direction, MC)': block['slope_p_one_sided_mc'] < alpha,
            'JP slope (2-sided, normal)': block['slope_p_two_sided_normal'] < alpha,
            'JP convexity (2-sided, MC)': block['convexity_p_two_sided_mc'] < alpha,
            'chi-square (MC)': block['chi2_p_mc'] < alpha,
            'above-all binomial (2-sided)': block['above_all']['p_binom_two_sided'] < alpha,
            'below-all binomial (2-sided)': block['below_all']['p_binom_two_sided'] < alpha,
            'outside binomial (2-sided)': block['outside']['p_binom_two_sided'] < alpha}


def null_tables(n, K, mc=100000, seed=2026):
    """Monte Carlo null of the rank statistics for n windows (multinomial counts under H0) and exact binomial tables."""
    rng = np.random.default_rng(seed); counts = rng.multinomial(n, np.ones(K + 1) / (K + 1), size=mc).astype(float)
    z1, z2 = jp_vec(counts, K); x2 = chi2_vec(counts, K)
    return dict(n=n, K=K, z1=np.sort(z1), absz2=np.sort(np.abs(z2)), x2=np.sort(x2), p1=binom_p_table(n, 1 / (K + 1)), p2=binom_p_table(n, 2 / (K + 1)))


def _p_ge(sorted_null, obs):
    return 1.0 - np.searchsorted(sorted_null, obs, side='left') / len(sorted_null)


def vector_rejections(rank, pit, K, nt, alpha=0.05):
    """Rejections of the main tests for many replicate data sets at once: rank, pit (D, n) -> dict of (D,) bool."""
    n = rank.shape[1]; counts = counts_of(rank, K); z1, z2 = jp_vec(counts, K); x2 = chi2_vec(counts, K)
    z = (pit.mean(1) - 0.5) / math.sqrt(1 / (12 * n)); above, below = (rank == K).sum(1), (rank == 0).sum(1)
    return {'mean PIT z (2-sided)': 2 * stats.norm.sf(np.abs(z)) < alpha, 'JP slope (1-sided, H1 direction, MC)': _p_ge(nt['z1'], z1) < alpha,
            'JP slope (2-sided, normal)': np.abs(z1) > Z975, 'JP convexity (2-sided, MC)': _p_ge(nt['absz2'], np.abs(z2)) < alpha,
            'chi-square (MC)': _p_ge(nt['x2'], x2) < alpha, 'above-all binomial (2-sided)': nt['p1']['two_sided'][above] < alpha,
            'below-all binomial (2-sided)': nt['p1']['two_sided'][below] < alpha, 'outside binomial (2-sided)': nt['p2']['two_sided'][above + below] < alpha,
            '_slope_z': z1, '_mean_pit': pit.mean(1) - 0.5}


# --------------------------------------------------------------------------------------------------- pilot data access
def seed_of(key):
    return int(key.rsplit('-s', 1)[1])


def load_measure(path, gen, real='real-recon', horizon='4.0', stat='value', names=None):
    """Windows of a frozen study-v1 measure file: real reading, generated readings in seed order, per-tracker states."""
    m = json.loads(Path(path).read_text()); rows = []
    for name in sorted(m['windows']):
        if names is not None and name not in names:
            continue
        clips = m['windows'][name]['clips']
        if real not in clips:
            continue
        keys = sorted((k for k in clips if k.startswith(gen + '-s')), key=seed_of)
        r = clips[real]['outcomes'][horizon]; g = [clips[k]['outcomes'][horizon] for k in keys]
        rows.append(dict(name=name, y=r[stat], Y=np.array([c[stat] for c in g], float), seeds=[seed_of(k) for k in keys],
                         real_each_state=r.get('each_state'), gen_each_state=[c.get('each_state') for c in g],
                         real_each_value=r.get('each_value'), gen_each_value=[c.get('each_value') for c in g],
                         real_state=r.get('state'), gen_state=[c.get('state') for c in g], cov=m['windows'][name]['covariates']))
    return rows


def names_of(path):
    return [l.strip() for l in Path(path).read_text().splitlines() if l.strip()]


def reproduce_frozen_primary(rows, K):
    """Re-run the frozen analysis' rank/PIT draws (seed 2026, window order, awc.ranks_of) to check the loader."""
    rng = np.random.default_rng(2026); rk, pit, crps = [], [], []
    for r in rows:
        Y = r['Y'][np.argsort([str(s) for s in r['seeds']], kind='stable')]  # analyze uses lexicographic clip order
        rank = awc.ranks_of(r['y'], Y, rng); rk.append(rank); pit.append((rank + rng.random()) / (K + 1)); crps.append(awc.fair_crps(Y, r['y']))
    return np.array(rk), np.array(pit), np.array(crps)


def write_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else (o.item() if isinstance(o, np.generic) else str(o))) + '\n')


# --------------------------------------------------------------------------------------------------------- self check
def self_check(seed=0):
    """Vectorised functions vs the imported ones (identical values); rank distribution with ties (chi-square)."""
    rng = np.random.default_rng(seed); out = {}
    Y = rng.normal(size=(500, 16)); y = rng.normal(size=500)
    out['fair_crps_max_abs_diff'] = float(np.max(np.abs(fair_crps_vec(Y, y) - np.array([awc.fair_crps(Y[i], y[i]) for i in range(500)]))))
    counts = rng.multinomial(60, np.ones(17) / 17, size=200).astype(float)
    z1, z2 = jp_vec(counts, 16); ref = np.array([awc.jp_scores(c, 16) for c in counts])
    out['jp_max_abs_diff'] = float(max(np.max(np.abs(z1 - ref[:, 0])), np.max(np.abs(z2 - ref[:, 1]))))
    out['chi2_max_abs_diff'] = float(np.max(np.abs(chi2_vec(counts, 16) - np.array([awc.chi2(c, 16) for c in counts]))))
    # ties: values on a coarse grid; compare the rank distributions of ranks_vec and awc.ranks_of
    Yt = rng.integers(0, 4, size=(40000, 16)).astype(float); yt = rng.integers(0, 4, size=40000).astype(float)
    rv = ranks_vec(yt, Yt, rng); ro = np.array([awc.ranks_of(yt[i], Yt[i], rng) for i in range(40000)])
    cv, co = np.bincount(rv, minlength=17), np.bincount(ro, minlength=17)
    out['ties_rank_hist_vec'] = cv.tolist(); out['ties_rank_hist_awc'] = co.tolist()
    out['ties_two_sample_chi2_p'] = float(stats.chi2_contingency(np.vstack([cv, co]))[1])
    out['ties_vec_uniform_chi2_p'] = float(stats.chisquare(cv).pvalue)
    out['ok'] = out['fair_crps_max_abs_diff'] < 1e-9 and out['jp_max_abs_diff'] < 1e-9 and out['chi2_max_abs_diff'] < 1e-9 and out['ties_two_sample_chi2_p'] > 1e-3
    return out


if __name__ == '__main__':
    print(json.dumps(self_check(), indent=1))
