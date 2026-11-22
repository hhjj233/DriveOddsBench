#!/usr/bin/env python3
"""bench-v1: the pre-registered analysis (protocol-v1.md section 5, frozen 2026-11-02; instrument addendum
protocol-v1.1-instrument.md sections 7, 9 and 10 and the coordinator decisions at its freezing). Written 2026-11-02.

Inputs
  --runs        one or more run CSVs of `measure_bench_v2.py export` (columns window, family, sample, outcome, kind,
                value, scorable, flags, value_v3, value_v5, value_v8). Several CSVs are combined per window: generated
                samples are unioned (e.g. test-k0-10 + test-k11-15 give samples 0..15); rows present in more than one CSV
                (the real and raw references) must be identical, else the run stops.
  --windows-dir window.json per analysed window: past-only covariates (strata) and the annotation velocities of the
                privileged constant-velocity baseline.
  --climatology the DEV raw readings (instrument/outcomes/outcomes-dev-raw-climatology.csv); also the DEV fit of the
                constant-velocity spread (--cv-fit, default the same file); --clim-windows-dir their window.json (log ids:
                windows from the analysed window's own log are left out, which only happens on DEV dry runs).
  --rules       instrument/scorability-rules.json (history_not_reliable families).
Refuses windows of the TEST lists unless --allow-test is given, and then also unless bench-v1/test-measured.done
exists with "complete": true (--force overrides; the override is recorded). On TEST pass all three exports to --runs:
outcomes-test-k0-10.csv and outcomes-test-k11-15.csv (samples 0-15 and the real reference, identical in both) and
outcomes-test-raw.csv (the raw reference); --extra-inputs logs the sha256 of the history CSVs, counts.json etc.
Protocol exclusions (EXCLUDED_ROLES, hard-coded from protocol-v1.md section 1): every row of an excluded
(window, family) is dropped at loading and the drop is reported.

What it computes (protocol 5)
  Per family and primary outcome:
    - the rank of the real reference among the window's scorable generated readings (K_i <= K), ties randomised with
      awc.ranks_of and a generator seeded 2026 (windows in name order), PIT = (rank + U) / (K_i + 1);
    - mean PIT - 0.5 with the exact-variance z test (1/(12 n)) and a percentile window bootstrap (10,000);
    - Jolliffe-Primo slope and convexity z and the chi-square on K + 1 bins with Monte Carlo p-values (100,000) by
      awc.test_block, unchanged. With K_i < K the bins are the K + 1 equal PIT bins (exactly uniform under H0; equal to
      the rank when K_i = K);
    - above-all and below-all shares (and the outside share) against 1/(K_i + 1) (2/(K_i + 1)): exact binomial tests
      (Poisson-binomial when K_i varies); intervals Clopper-Pearson for n <= 45, Wilson otherwise;
    - fair CRPS (awc.fair_crps), sharpness (mean within-window SD), CRPSS against climatology and constant velocity
      with paired window-bootstrap CIs. Constant velocity: the instrument's 2 s history least-squares line of the real
      reference (hist_ls_*); for families labelled history_not_reliable the privileged annotation version (the
      instrument's t0 reading + the LiDAR annotation velocity of window.json relative to the ego, ego speed for the ego
      family); the annotation version is also reported for every family. Gaussian spread per family and outcome =
      RMS of the DEV residuals of the same forecaster (--cv-fit).
  H1 (confirmatory): lead x_4s, one-sided JP slope p (normal, exact null variance); confirmed only if the Holm-adjusted
     p (with H2) <= 0.05 AND the bootstrap 95 % CI of mean PIT - 0.5 lies above 0.
  H2 (confirmatory): window-level exchangeability permutation test, B = 100,000, over the pool lead, lead_stop, cutin,
     oncoming, ego_inter (ped excluded by the freezing decision); per window only generated clips scorable on all the
     window's pool outcomes take part. p = (1 + #{T* >= T}) / (1 + B); the exact permutation p (FFT) is reported as a
     check.
  Secondary: BH q = 0.10 within each statistic (mean PIT, above-all, below-all, convexity) across the scored families.
  Pipeline checks: positive control (sample 0 as truth against the others), split-half (first and second half of the
     samples). Sensitivity (exploratory): carried_lost as missing, each tracker alone, raw reference, secondary
     outcomes, strata by past-only covariates, 2,000 tie-break draws. ped, startup, crossing and rider: descriptive.

  analyze_bench_v1.py --runs A.csv [B.csv ...] --windows-dir BENCH/windows-test --climatology CLIM.csv
      --clim-windows-dir BENCH/windows-dev --rules RULES.json --label TEST --allow-test --out DIR
"""
from __future__ import annotations

import os

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '2')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

import argparse, csv, datetime, hashlib, json, math, re, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_wm_calibration as awc  # noqa: E402  ranks_of, jp_contrasts, jp_scores, chi2, fair_crps, bh, test_block, simultaneous_band

BENCH = Path(os.environ.get('DOB_ROOT', '.'), 'outputs/wm-calibration/bench-v1')
FAMILY_ORDER = ['lead', 'lead_stop', 'cutin', 'oncoming', 'ego_inter', 'ped', 'startup', 'crossing', 'rider']
SCORED = ['lead', 'lead_stop', 'cutin', 'oncoming', 'ego_inter']          # confirmatory / secondary families
DESCRIPTIVE = ['ped', 'startup', 'crossing', 'rider']                      # ped: coordinator's freezing decision
H2_POOL = ['lead', 'lead_stop', 'cutin', 'oncoming', 'ego_inter']
PRIMARY = {'lead': 'x_4s', 'lead_stop': 'x_4s', 'cutin': 'absy_2s', 'oncoming': 'x_1s', 'ego_inter': 'ego_dx_0_4s',
           'ped': 'absy_2s', 'startup': 'x_6s', 'crossing': 'x_1s', 'rider': 'x_1s'}  # crossing, rider: first descriptive outcome
H1 = ('lead', 'x_4s')
HARD = {'t0_not_found', 'real_t0_not_found', 't0_box_at_edge', 'no_value', 'family_not_reliable', 'outcome_not_reliable'}
RELIABILITY = {'family_not_reliable', 'outcome_not_reliable'}
# protocol-v1.md section 1 ("Generation subsamples"): the TEST pedestrian target of window 1c98497e-t2.5 passed the t0
# detector rule on the first renders (IoU 0.52) but not on the final anti-aliased frames (IoU 0.24); the visibility rule
# is applied to the final frames, so that ped role is dropped and the window stays for its other roles. window.json still
# lists the target, so its ped rows are in the TEST exports: they are removed here.
EXCLUDED_ROLES = {('1c98497e-t2.5', 'ped'): 'protocol-v1.md section 1: t0 detector rule failed on the final anti-aliased frames (IoU 0.24)'}
SEED = 2026
STREAM = {'ties_h2': 2, 'perm': 3, 'boot': 4, 'mc': 5, 'cv_draws': 6, 'tiebreak_spread': 7, 'band': 8}
Z975 = stats.norm.ppf(0.975)


def rng_for(stream, *extra):
    """Independent, reproducible generators for everything except the pre-registered tie-break stream (seed 2026)."""
    key = [SEED, STREAM[stream]] + [int(hashlib.sha256(str(e).encode()).hexdigest()[:8], 16) for e in extra]
    return np.random.default_rng(key)


# ----------------------------------------------------------------------------------------------------------- input
class Row:
    __slots__ = ('value', 'scorable', 'flags', 'flagset', 'v3', 'v5', 'v8', 'kind', 'raw')

    def __init__(self, r):
        f = lambda s: None if s is None or s == '' else float(s)
        self.value, self.scorable, self.flags = f(r['value']), r['scorable'] == '1', r['flags']
        self.flagset = {t.split('=')[0] for t in r['flags'].split(';') if t}
        self.v3, self.v5, self.v8, self.kind = f(r.get('value_v3')), f(r.get('value_v5')), f(r.get('value_v8')), r['kind']
        self.raw = tuple(r[k] for k in ('value', 'scorable', 'flags', 'value_v3', 'value_v5', 'value_v8', 'kind'))

    def col(self, c):
        return {'value': self.value, 'value_v3': self.v3, 'value_v5': self.v5, 'value_v8': self.v8}[c]


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def load_runs(paths):
    """{(window, family, outcome): {sample: Row}}, kinds {(family, outcome): kind}; duplicates must agree."""
    data, kinds, origin, mismatches = defaultdict(dict), {}, {}, []
    for p in paths:
        with open(p, newline='') as fh:
            for r in csv.DictReader(fh):
                key, s = (r['window'], r['family'], r['outcome']), r['sample']
                row = Row(r)
                if s in data[key]:
                    if data[key][s].raw != row.raw:
                        mismatches.append((key, s, origin[(key, s)], str(p)))
                    continue
                data[key][s] = row; origin[(key, s)] = str(p); kinds[(r['family'], r['outcome'])] = r['kind']
    if mismatches:
        raise SystemExit(f'{len(mismatches)} rows differ between run CSVs, e.g. {mismatches[:3]}')
    return data, kinds


def apply_exclusions(data, protocol=None):
    """Drop every row of the protocol's excluded (window, family) roles; check that the protocol still names them."""
    text = protocol.read_text() if protocol is not None and protocol.is_file() else ''
    out = {}
    for (w, fam), why in EXCLUDED_ROLES.items():
        keys = [k for k in data if k[0] == w and k[1] == fam]
        out[f'{w} {fam}'] = dict(reason=why, rows_dropped=sum(len(data[k]) for k in keys), outcomes_dropped=len(keys),
                                 named_in_protocol=(w in text) if text else None)
        for k in keys:
            del data[k]
    return out


def generated_ids(data):
    ids = sorted({int(s) for v in data.values() for s in v if s not in ('real', 'raw')})
    return ids


def usable(row, rule, col='value', drop_lost=False):
    if row is None or row.col(col) is None:
        return False
    if drop_lost and 'carried_lost' in row.flagset:
        return False
    if rule == 'scorable':
        return row.scorable
    if rule == 'descriptive':                    # reliability labels ignored, every other hard rule kept
        return not ((row.flagset & HARD) - RELIABILITY)
    raise ValueError(rule)


def rule_of(fam):
    return 'descriptive' if fam in ('ped', 'crossing', 'rider') else 'scorable'


# ------------------------------------------------------------------------------------------------- per-family ranks
def build(data, fam, outcome, gen_ids, truth='real', col='value', rule=None, drop_lost=False, windows=None):
    """Windows (name order) with a usable truth and >= 2 usable generated readings; drop counts by reason."""
    rule = rule or rule_of(fam); rows, drops, gen_drops = [], Counter(), Counter()
    keys = sorted(k for k in data if k[1] == fam and k[2] == outcome and (windows is None or k[0] in windows))
    for key in keys:
        smp = data[key]; tr = smp.get(truth)
        if tr is None:
            drops['no truth row'] += 1; continue
        if not usable(tr, rule, col, drop_lost):
            why = sorted((tr.flagset & HARD) - (RELIABILITY if rule == 'descriptive' else set())) or (['carried_lost'] if drop_lost and 'carried_lost' in tr.flagset else ['no value'])
            drops['truth not usable: ' + '+'.join(why)] += 1; continue
        ids, Y = [], []
        for g in gen_ids:
            r = smp.get(str(g))
            if r is None:
                gen_drops['missing row'] += 1; continue
            if not usable(r, rule, col, drop_lost):
                gen_drops['+'.join(sorted((r.flagset & HARD) - (RELIABILITY if rule == 'descriptive' else set()))) or ('carried_lost' if drop_lost and 'carried_lost' in r.flagset else 'no value')] += 1
                continue
            ids.append(g); Y.append(r.col(col))
        if len(Y) < 2:
            drops['fewer than 2 usable generated'] += 1; continue
        rows.append(dict(name=key[0], y=tr.col(col), ids=ids, Y=np.array(Y, float), K_full=len(gen_ids),
                         truth_flags=sorted(tr.flagset), gen_flags={g: sorted(smp[str(g)].flagset) for g in ids}))
    return rows, dict(windows_with_family_outcome=len(keys), kept=len(rows), dropped=dict(drops), generated_not_usable=dict(gen_drops))


def rank_rows(rows, seed=SEED):
    """The pre-registered tie-break stream: one generator per analysis, windows in name order, awc.ranks_of then U."""
    rng = np.random.default_rng(seed)
    for r in rows:
        r['rank'] = awc.ranks_of(r['y'], r['Y'], rng); r['pit'] = (r['rank'] + rng.random()) / (len(r['Y']) + 1)
    return rows


# ------------------------------------------------------------------------------------------------------- statistics
def wilson(k, n, z=Z975):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


def clopper_pearson(k, n, a=0.05):
    return [float(stats.beta.ppf(a / 2, k, n - k + 1)) if k > 0 else 0.0, float(stats.beta.ppf(1 - a / 2, k + 1, n - k)) if k < n else 1.0]


def share_interval(k, n):
    return (clopper_pearson(k, n), 'Clopper-Pearson') if n <= 45 else (wilson(k, n), 'Wilson')


def poisson_binomial(ps):
    pmf = np.zeros(len(ps) + 1); pmf[0] = 1.0
    for i, p in enumerate(ps):
        pmf[1:i + 2] = pmf[1:i + 2] * (1 - p) + pmf[0:i + 1] * p; pmf[0] *= (1 - p)
    return pmf


def count_test(k, ps):
    """Exact tests of a count against independent Bernoulli(p_i): binomial when all p_i are equal."""
    ps = np.asarray(ps, float); n = len(ps)
    if np.allclose(ps, ps[0]):
        p0 = float(ps[0]); bt = lambda alt: float(stats.binomtest(k, n, p0, alternative=alt).pvalue)
        return dict(test='exact binomial', p_two_sided=bt('two-sided'), p_greater=bt('greater'), p_less=bt('less'))
    pmf = poisson_binomial(ps)
    return dict(test='exact Poisson-binomial', p_two_sided=float(min(1.0, pmf[pmf <= pmf[k] * (1 + 1e-12)].sum())), p_greater=float(pmf[k:].sum()), p_less=float(pmf[:k + 1].sum()))


def pit_bins(rows, K):
    return np.array([r['rank'] if len(r['Y']) == K else min(int(r['pit'] * (K + 1)), K) for r in rows], int)


def boot_ci(x, rng, B):
    x = np.asarray(x, float); idx = rng.integers(0, len(x), (B, len(x))); b = x[idx].mean(1)
    return [float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))]


def family_stats(rows, K, mc, boot, tag):
    n = len(rows)
    if n == 0:
        return dict(n=0)
    rank = np.array([r['rank'] for r in rows]); Ki = np.array([len(r['Y']) for r in rows]); pit = np.array([r['pit'] for r in rows])
    bins = pit_bins(rows, K)
    tb = awc.test_block(bins, K, rng_for('mc', tag), mc)
    z = (pit.mean() - 0.5) / math.sqrt(1 / (12 * n))
    out = dict(n=n, K=K, K_i=dict(min=int(Ki.min()), median=float(np.median(Ki)), mean=float(Ki.mean()), windows_below_K=int((Ki < K).sum())),
               counts=tb['counts'], bins='rank' if (Ki == K).all() else 'K+1 equal PIT bins (K_i varies)',
               mean_pit_minus_half=float(pit.mean() - 0.5), mean_pit_ci95=[v - 0.5 for v in boot_ci(pit, rng_for('boot', tag), boot)],
               mean_pit_z=float(z), mean_pit_p_two_sided=float(2 * stats.norm.sf(abs(z))), mean_pit_p_one_sided_high=float(stats.norm.sf(z)),
               slope_z=tb['slope_z'], slope_p_one_sided_normal=float(stats.norm.sf(tb['slope_z'])), slope_p_one_sided_mc=tb['slope_p_one_sided_mc'],
               slope_p_two_sided_normal=float(2 * stats.norm.sf(abs(tb['slope_z']))),
               convexity_z=tb['convexity_z'], convexity_p_two_sided_mc=tb['convexity_p_two_sided_mc'], convexity_p_two_sided_normal=float(2 * stats.norm.sf(abs(tb['convexity_z']))),
               chi2=tb['chi2'], chi2_p_mc=tb['chi2_p_mc'], mc_draws=mc)
    for lab, hit, ps in (('above_all', rank == Ki, 1 / (Ki + 1)), ('below_all', rank == 0, 1 / (Ki + 1)), ('outside', (rank == Ki) | (rank == 0), 2 / (Ki + 1))):
        k = int(hit.sum()); ci, meth = share_interval(k, n)
        out[lab] = dict(count=k, share=k / n, expected=float(ps.mean()), ratio=(k / n) / float(ps.mean()), ci95=ci, interval=meth, **count_test(k, ps))
    crps = np.array([awc.fair_crps(r['Y'], r['y']) for r in rows])
    out.update(mean_fair_crps=float(crps.mean()), mean_fair_crps_ci95=boot_ci(crps, rng_for('boot', tag, 'crps'), boot),
               sharpness_mean_sd=float(np.mean([np.std(r['Y'], ddof=1) for r in rows])), median_real=float(np.median([r['y'] for r in rows])),
               median_generated=float(np.median(np.concatenate([r['Y'] for r in rows]))))
    for r, c in zip(rows, crps):
        r['crps'] = float(c)
    return out


# ------------------------------------------------------------------------------------------------------- baselines
OUT_RE = [(re.compile(r'^x_(\d+(?:\.\d+)?)s$'), 'x'), (re.compile(r'^absy_(\d+(?:\.\d+)?)s$'), 'absy'), (re.compile(r'^y_(\d+(?:\.\d+)?)s$'), 'y'),
          (re.compile(r'^min_x_0_(\d+)s$'), 'minx'), (re.compile(r'^ego_dx_0_(\d+)s$'), 'egodx'), (re.compile(r'^ego_dyaw_deg_0_(\d+)s$'), 'egodyaw')]


def outcome_kind(outcome):
    for rx, k in OUT_RE:
        m = rx.match(outcome)
        if m:
            return k, float(m.group(1))
    return None, None                               # binary outcomes: no constant-velocity forecast


def history_of(data, w, fam, sample):
    """History rows of one (window, family, sample): {name: Row}."""
    names = ('hist_x_0s', 'hist_y_0s', 'hist_ls_x0', 'hist_ls_vx', 'hist_ls_y0', 'hist_ls_vy', 'hist_ls_n', 'hist_ls_t_first',
             'ego_hist_ls_speed', 'ego_hist_ls_yaw_rate_dps')
    return {n: data.get((w, fam, n), {}).get(sample) for n in names}


def cv_mean(fam, outcome, hist, meta, version):
    """Constant-velocity point forecast. 'line': the instrument's 2 s history least-squares line (protocol 5);
    'annotation': the instrument's t0 reading + the LiDAR annotation velocity relative to the ego (privileged)."""
    kind, h = outcome_kind(outcome)
    if kind is None:
        return None
    if fam == 'ego_inter':
        if version == 'line':
            r = hist['ego_hist_ls_speed'] if kind == 'egodx' else hist['ego_hist_ls_yaw_rate_dps']
            return None if r is None or not r.scorable or r.value is None else r.value * h
        if kind != 'egodx' or meta is None:
            return None
        v = (meta.get('past_covariates') or {}).get('ego_speed_mps')
        return None if v is None else v * h
    if version == 'line':
        need = ('hist_ls_x0', 'hist_ls_vx') if kind in ('x', 'minx') else ('hist_ls_y0', 'hist_ls_vy')
        if any(hist[k] is None or not hist[k].scorable or hist[k].value is None for k in need):
            return None
        a, v = hist[need[0]].value, hist[need[1]].value
    else:
        t0 = hist['hist_x_0s'] if kind in ('x', 'minx') else hist['hist_y_0s']
        tgt = ((meta or {}).get('targets') or {}).get(fam) or {}; ego_v = ((meta or {}).get('past_covariates') or {}).get('ego_speed_mps')
        if t0 is None or t0.value is None or (t0.flagset & HARD):
            return None
        if kind in ('x', 'minx'):
            if tgt.get('vx_mps') is None or ego_v is None:
                return None
            a, v = t0.value, tgt['vx_mps'] - ego_v
        else:
            if tgt.get('vy_mps') is None:
                return None
            a, v = t0.value, tgt['vy_mps']
    if kind == 'x':
        return a + v * h
    if kind == 'minx':
        return a + min(0.0, v * h)
    if kind == 'absy':
        return abs(a + v * h)
    if kind == 'y':
        return a + v * h
    return None


def gauss_crps(mu, sigma, y):
    z = (y - mu) / sigma
    return float(sigma * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / math.sqrt(math.pi)))


class Baselines:
    """Climatology (DEV raw readings) and the DEV fit of the constant-velocity spread; leaves out the analysed window's
    own log when it is in the DEV source (DEV dry runs only)."""

    def __init__(self, clim_data, clim_meta, fit_data, fit_meta):
        self.clim, self.fit = defaultdict(list), defaultdict(list)
        for (w, fam, o), smp in clim_data.items():
            r = smp.get('raw')
            if r is not None and r.kind != 'history' and usable(r, rule_of(fam)):
                self.clim[(fam, o)].append((log_of(clim_meta, w), r.value))
        for (w, fam, o), smp in fit_data.items():
            r = smp.get('raw')
            if r is None or not usable(r, rule_of(fam)) or outcome_kind(o)[0] is None:
                continue
            hist = history_of(fit_data, w, fam, 'raw')
            for version in ('line', 'annotation'):
                m = cv_mean(fam, o, hist, fit_meta.get(w), version)
                if m is not None:
                    self.fit[(fam, o, version)].append((log_of(fit_meta, w), r.value - m))

    def clim_values(self, fam, o, log):
        return np.array([v for lg, v in self.clim.get((fam, o), []) if lg != log or lg is None], float)

    def sigma(self, fam, o, version, log):
        res = np.array([e for lg, e in self.fit.get((fam, o, version), []) if lg != log or lg is None], float)
        return (float(np.sqrt(np.mean(res ** 2))), len(res)) if len(res) >= 5 else (None, len(res))


def log_of(meta, w):
    m = meta.get(w)
    return None if m is None else m.get('log')


def baseline_scores(rows, data, fam, o, meta, bases, truth_sample, boot, tag):
    """CRPS of climatology and of constant velocity (line, annotation) on the analysed windows; CRPSS with paired
    window-bootstrap CIs; the baselines' own PIT means (descriptive)."""
    hist_not_rel = fam in bases['history_not_reliable']
    cv_main = 'annotation' if hist_not_rel else 'line'
    per = {'climatology': [], 'cv (line)': [], 'cv (annotation, privileged)': []}; info = {}
    for r in rows:
        lg = log_of(meta, r['name']); cl = bases['obj'].clim_values(fam, o, lg)
        per['climatology'].append(awc.fair_crps(cl, r['y']) if len(cl) >= 2 else None)
        r['clim_pit'] = float((np.sum(cl < r['y']) + 0.5 * np.sum(cl == r['y'])) / len(cl)) if len(cl) else None
        hist = history_of(data, r['name'], fam, truth_sample)
        for version, lab in (('line', 'cv (line)'), ('annotation', 'cv (annotation, privileged)')):
            m = cv_mean(fam, o, hist, meta.get(r['name']), version)
            s, nfit = bases['obj'].sigma(fam, o, version, lg)
            info[f'{lab} sigma (DEV fit, n)'] = (s, nfit)
            if m is None or s is None or s <= 0:
                per[lab].append(None); continue
            per[lab].append(gauss_crps(m, s, r['y'])); r[f'cvpit_{version}'] = float(stats.norm.cdf((r['y'] - m) / s))
    model = np.array([r['crps'] for r in rows])
    out = dict(cv_baseline_used='cv (annotation, privileged)' if hist_not_rel else 'cv (line)', history_not_reliable=hist_not_rel,
               climatology_source_n=int(np.median([len(bases['obj'].clim_values(fam, o, log_of(meta, r['name']))) for r in rows])) if rows else 0,
               sigma_fit={k: dict(sigma=v[0], n=v[1]) for k, v in info.items()}, baselines={})
    for lab, vals in per.items():
        ok = np.array([v is not None for v in vals])
        if ok.sum() < 3:
            out['baselines'][lab] = dict(n=int(ok.sum()), note='fewer than 3 windows with this baseline'); continue
        b = np.array([v for v in vals if v is not None], float); m = model[ok]
        idx = rng_for('boot', tag, lab).integers(0, len(b), (boot, len(b))); bs = 1 - m[idx].mean(1) / b[idx].mean(1)
        pits = [r.get('clim_pit') if lab == 'climatology' else r.get('cvpit_' + ('line' if 'line' in lab else 'annotation')) for r, k in zip(rows, ok) if k]
        out['baselines'][lab] = dict(n=int(ok.sum()), mean_crps=float(b.mean()), model_mean_crps_same_windows=float(m.mean()),
                                     crpss=float(1 - m.mean() / b.mean()), crpss_ci95=[float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
                                     baseline_pit_mean_minus_half_descriptive=float(np.mean([p for p in pits if p is not None]) - 0.5) if any(p is not None for p in pits) else None)
    out['crpss_vs_cv'] = out['baselines'].get(out['cv_baseline_used'])
    return out


# --------------------------------------------------------------------------------------------------------------- H2
def h2_test(fam_rows, perm_B, boot, tag, exact=True):
    """Window-level exchangeability permutation test over the pool's primary outcomes. fam_rows: {family: rows with
    'rank' from that family's analysis}. Per window, the clips are the real reference and the generated samples usable
    on every pool outcome the window has; positions within ties follow the family rank of the real (the same
    randomisation) and random keys for generated-generated ties."""
    by_w = defaultdict(dict)
    for fam in H2_POOL:
        for r in fam_rows.get(fam, []):
            by_w[r['name']][fam] = r
    rng_keys = rng_for('ties_h2', tag); wins, drops = [], Counter()
    for w in sorted(by_w):
        outs = [by_w[w][f] for f in H2_POOL if f in by_w[w]]
        complete = set(outs[0]['ids'])
        for r in outs[1:]:
            complete &= set(r['ids'])
        complete = sorted(complete)
        if len(complete) < 2:
            drops['fewer than 2 generated clips usable on all pool outcomes'] += 1; continue
        clips = ['real'] + complete; outside = np.zeros(len(clips)); below_c, tied_c = [], []
        for r in outs:
            vals = dict(zip(r['ids'], r['Y'])); keys = rng_keys.random(len(r['ids']))
            order = [g for _, _, g in sorted(zip(r['Y'], keys, r['ids']))]
            order.insert(r['rank'], 'real')
            sub = [c for c in order if c == 'real' or c in complete]
            for c in (sub[0], sub[-1]):
                outside[clips.index(c)] += 1
            Yc = np.array([vals[g] for g in complete]); below_c.append(int((Yc < r['y']).sum())); tied_c.append(int((Yc == r['y']).sum()))
        out = outside / len(outs)
        wins.append(dict(name=w, families=[f for f in H2_POOL if f in by_w[w]], clips=len(clips), out=out, below=below_c, tied=tied_c))
    n = len(wins)
    if n == 0:
        return dict(n_windows=0, dropped=dict(drops))
    T = float(np.mean([x['out'][0] for x in wins])); null_mean = float(np.mean([x['out'].mean() for x in wins]))
    rng = rng_for('perm', tag); acc = 0; done = 0; chunk = 10000
    while done < perm_B:
        b = min(chunk, perm_B - done); tot = np.zeros(b)
        for x in wins:
            tot += x['out'][rng.integers(0, len(x['out']), b)]
        acc += int((tot / n >= T - 1e-12).sum()); done += b
    res = dict(n_windows=n, dropped=dict(drops), T=T, null_mean=null_mean, expected_2_over_K_plus_1=null_mean, B=perm_B, p_mc=(1 + acc) / (1 + perm_B),
               boot_ci95=boot_ci([x['out'][0] for x in wins], rng_for('boot', tag, 'h2'), boot),
               families_per_window=dict(Counter(len(x['families']) for x in wins)), clips_per_window=dict(Counter(x['clips'] for x in wins)))
    if exact:
        res['p_exact_fft'] = h2_exact_p(wins, T)
    res['_wins'] = wins
    return res


def h2_exact_p(wins, T):
    """Exact permutation p-value: out values are multiples of 1/m (m <= 5 outcomes) -> integers in units of 1/60."""
    U = 60; L = U * len(wins) + 1; nfft = 1 << int(math.ceil(math.log2(L))); logF = np.zeros(nfft // 2 + 1, complex)
    for x in wins:
        u = np.rint(x['out'] * U).astype(int); pmf = np.bincount(u, minlength=U + 1) / len(u)
        F = np.fft.rfft(pmf, nfft); logF += np.log(np.where(np.abs(F) < 1e-300, 1e-300, F))
    pmf = np.clip(np.fft.irfft(np.exp(logF), nfft)[:L], 0, None); pmf /= pmf.sum()
    s_obs = int(round(T * len(wins) * U))
    return float(pmf[s_obs:].sum())


def holm(p1, p2):
    ps = np.array([p1, p2]); order = np.argsort(ps); adj = np.empty(2)
    adj[order[0]] = min(1.0, 2 * ps[order[0]]); adj[order[1]] = min(1.0, max(adj[order[0]], ps[order[1]]))
    return float(adj[0]), float(adj[1])


# ------------------------------------------------------------------------------------------------- tie-break spread
def tie_spread(rows, K, draws, tag, h2_wins=None):
    """Statistics over `draws` tie-break draws (the confirmatory analysis uses the single draw of seed 2026)."""
    if not rows:
        return None
    rng = rng_for('tiebreak_spread', tag); y = [r['y'] for r in rows]
    below = np.array([int((r['Y'] < r['y']).sum()) for r in rows]); tied = np.array([int((r['Y'] == r['y']).sum()) for r in rows]); Ki = np.array([len(r['Y']) for r in rows])
    u = rng.random((draws, len(rows))); rank = below + np.floor(u * (tied + 1)).astype(int); pit = (rank + rng.random((draws, len(rows)))) / (Ki + 1)
    bins = np.where(Ki == K, rank, np.minimum((pit * (K + 1)).astype(int), K))
    c1, c2 = awc.jp_contrasts(K); n = len(rows); p = 1 / (K + 1)
    counts = np.stack([np.bincount(b, minlength=K + 1) for b in bins]).astype(float)
    z1 = counts @ c1 / np.sqrt(n * p * (c1 @ c1)); z2 = counts @ c2 / np.sqrt(n * p * (c2 @ c2))
    assert abs(z1[0] - awc.jp_scores(counts[0], K)[0]) < 1e-9
    summ = lambda v: dict(mean=float(np.mean(v)), sd=float(np.std(v)), min=float(np.min(v)), max=float(np.max(v)), q025=float(np.quantile(v, 0.025)), q975=float(np.quantile(v, 0.975)))
    out = dict(draws=draws, windows_with_ties=int((tied > 0).sum()), mean_pit_minus_half=summ(pit.mean(1) - 0.5), slope_z=summ(z1), convexity_z=summ(z2),
               above_all=summ((rank == Ki).mean(1)), below_all=summ((rank == 0).mean(1)))
    return out


def h2_tie_spread(wins, draws, tag):
    if not wins:
        return None
    rng = rng_for('tiebreak_spread', tag, 'h2'); tot = np.zeros(draws)
    for x in wins:
        K = x['clips'] - 1; acc = np.zeros(draws)
        for b, t in zip(x['below'], x['tied']):
            r = b + np.floor(rng.random(draws) * (t + 1)).astype(int); acc += ((r == 0) | (r == K))
        tot += acc / len(x['below'])
    T = tot / len(wins)
    return dict(draws=draws, mean=float(T.mean()), sd=float(T.std()), min=float(T.min()), max=float(T.max()))


# ------------------------------------------------------------------------------------------------------------ strata
def strata_of(rows, fam, meta):
    def cov(r, key):
        m = meta.get(r['name']) or {}
        if key == 'ego_speed':
            return (m.get('past_covariates') or {}).get('ego_speed_mps')
        t = (m.get('targets') or {}).get(fam) or {}
        if key == 'gap':
            return t.get('rear_face_m') if t.get('rear_face_m') is not None else t.get('x_m')
        return t.get('category')
    out = {}
    for key in (('ego_speed',) if fam == 'ego_inter' else ('gap', 'ego_speed', 'class')):
        vals = [cov(r, key) for r in rows]
        if key == 'class':
            for c in sorted({v for v in vals if v is not None}):
                out[f'target class {c}'] = [r for r, v in zip(rows, vals) if v == c]
            continue
        x = np.array([np.nan if v is None else v for v in vals], float)
        if np.isfinite(x).sum() < 6:
            continue
        t1, t2 = np.nanquantile(x, [1 / 3, 2 / 3]); lab = 'initial gap (m)' if key == 'gap' else 'ego speed (m/s)'
        out[f'{lab} < {t1:.1f}'] = [r for r, v in zip(rows, x) if v < t1]
        out[f'{lab} {t1:.1f}-{t2:.1f}'] = [r for r, v in zip(rows, x) if t1 <= v < t2]
        out[f'{lab} >= {t2:.1f}'] = [r for r, v in zip(rows, x) if v >= t2]
    return out


# ------------------------------------------------------------------------------------------------------------ report
def fmt_p(p):
    return 'n/a' if p is None else ('< 1e-5' if p < 1e-5 else (f'{p:.2g}' if p < 0.001 else f'{p:.3f}'))


def fam_line(fam, o, s, q=None, base=None):
    if not s or s.get('n', 0) == 0:
        return '| ' + ' | '.join([fam, o, '0'] + [''] * 10) + ' |'
    q = q or {}; qq = lambda k: '' if q.get(k) is None else f", q {q[k]:.3f}"
    cv = (base or {}).get('crpss_vs_cv') or {}; cl = ((base or {}).get('baselines') or {}).get('climatology') or {}
    crpss = lambda d: '' if not d or 'crpss' not in d else f"{d['crpss']:+.2f} [{d['crpss_ci95'][0]:+.2f}, {d['crpss_ci95'][1]:+.2f}]"
    return (f"| {fam} | {o} | {s['n']} (K_i {s['K_i']['min']}-{s['K']}) | {s['mean_pit_minus_half']:+.3f} [{s['mean_pit_ci95'][0]:+.3f}, {s['mean_pit_ci95'][1]:+.3f}] (p {fmt_p(s['mean_pit_p_two_sided'])}{qq('mean_pit')}) | "
            f"{s['slope_z']:+.2f} | {s['convexity_z']:+.2f} (p {fmt_p(s['convexity_p_two_sided_mc'])}{qq('convexity')}) | {fmt_p(s['chi2_p_mc'])} | "
            f"{s['above_all']['share']:.3f} [{s['above_all']['ci95'][0]:.3f}, {s['above_all']['ci95'][1]:.3f}] (p {fmt_p(s['above_all']['p_two_sided'])}{qq('above_all')}) | "
            f"{s['below_all']['share']:.3f} [{s['below_all']['ci95'][0]:.3f}, {s['below_all']['ci95'][1]:.3f}] (p {fmt_p(s['below_all']['p_two_sided'])}{qq('below_all')}) | "
            f"{s['above_all']['expected']:.3f} | {s['mean_fair_crps']:.2f} [{s['mean_fair_crps_ci95'][0]:.2f}, {s['mean_fair_crps_ci95'][1]:.2f}] / {s['sharpness_mean_sd']:.2f} | {crpss(cl)} | {crpss(cv)} |")


HEAD = ('| family | outcome | n (K_i) | mean PIT - 0.5 [95 % CI] | JP slope z | JP convexity z | chi-square p (MC) | above all [CI] | below all [CI] | '
        'expected share | fair CRPS [CI] / sharpness | CRPSS vs climatology | CRPSS vs constant velocity |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|')


def pit_figure(rows, K, title, path, tag):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    n = len(rows); bins = pit_bins(rows, K); pit = np.array([r['pit'] for r in rows])
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    counts = np.bincount(bins, minlength=K + 1); lo, hi = stats.binom.ppf(0.025, n, 1 / (K + 1)), stats.binom.ppf(0.975, n, 1 / (K + 1))
    ax[0].bar(np.arange(K + 1), counts, color='#4c72b0'); ax[0].axhline(n / (K + 1), color='k', lw=1); ax[0].axhspan(lo, hi, color='grey', alpha=0.2)
    ax[0].set_xlabel(f'rank of the real reference among K = {K} (0 = below all)'); ax[0].set_ylabel('windows'); ax[0].set_title(title, fontsize=9)
    grid = np.linspace(0.01, 0.99, 99); blo, bhi, _ = awc.simultaneous_band(n, grid, 0.95, rng_for('band', tag))
    ecdf = np.array([(pit <= z).mean() for z in grid])
    ax[1].fill_between(grid, blo - grid, bhi - grid, color='grey', alpha=0.25); ax[1].plot(grid, ecdf - grid, color='#c44e52'); ax[1].axhline(0, color='k', lw=1)
    ax[1].set_xlabel('PIT'); ax[1].set_ylabel('ECDF - PIT'); ax[1].set_title('ECDF difference, 95 % simultaneous band', fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items() if not str(k).startswith('_')}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    return o


# -------------------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', nargs='+', required=True, type=Path); ap.add_argument('--windows-dir', required=True, type=Path)
    ap.add_argument('--climatology', type=Path, default=BENCH / 'instrument/outcomes/outcomes-dev-raw-climatology.csv')
    ap.add_argument('--cv-fit', type=Path, help='DEV raw readings for the constant-velocity spread (default: --climatology)')
    ap.add_argument('--clim-windows-dir', type=Path, default=BENCH / 'windows-dev'); ap.add_argument('--rules', type=Path, default=BENCH / 'instrument/scorability-rules.json')
    ap.add_argument('--out', required=True, type=Path); ap.add_argument('--label', required=True, help='DEV / TEST / SIM: printed on every table')
    ap.add_argument('--K', type=int, help='expected K (checked)'); ap.add_argument('--allow-test', action='store_true')
    ap.add_argument('--done-file', type=Path, default=BENCH / 'test-measured.done', help='TEST: must hold "complete": true')
    ap.add_argument('--force', action='store_true', help='TEST: run although the done file is missing or not complete (recorded)')
    ap.add_argument('--extra-inputs', nargs='*', type=Path, default=[], help='further files whose sha256 is logged (history CSVs, counts.json)')
    ap.add_argument('--boot', type=int, default=10000); ap.add_argument('--mc', type=int, default=100000); ap.add_argument('--mc-explore', type=int, default=20000)
    ap.add_argument('--perm', type=int, default=100000); ap.add_argument('--tie-draws', type=int, default=2000)
    ap.add_argument('--pilot-summary', type=Path, default=Path(os.environ.get('DOB_ROOT', '.'), 'outputs/wm-calibration/study-v1/results-test-g6/primary/summary.json'))
    a = ap.parse_args()
    if a.out.exists() and any(a.out.iterdir()):
        raise SystemExit(f'{a.out} exists and is not empty: fresh output required')
    a.out.mkdir(parents=True, exist_ok=True); t_start = datetime.datetime.now().astimezone()
    data, kinds = load_runs(a.runs)
    test_names = set()
    for f in ('names-test.txt', 'names-test-gen.txt'):
        if (BENCH / f).is_file():
            test_names |= {l.strip() for l in (BENCH / f).read_text().splitlines() if l.strip()}
    is_test = bool({k[0] for k in data} & test_names) or 'windows-test' in str(a.windows_dir)
    done = dict(required=is_test, file=str(a.done_file), exists=a.done_file.is_file(), complete=None, forced=False)
    if is_test:
        if not a.allow_test:
            raise SystemExit('TEST windows: refused without --allow-test')
        if done['exists']:
            try:
                done['complete'] = json.loads(a.done_file.read_text()).get('complete')
            except (json.JSONDecodeError, AttributeError):
                m = re.search(r'complete["\s]*[:=]\s*(true|false)', a.done_file.read_text(), re.I)
                done['complete'] = None if m is None else m.group(1).lower() == 'true'
            done['sha256'] = sha256_file(a.done_file)
        if done['complete'] is not True:
            if not a.force:
                raise SystemExit(f'TEST measurement not complete ({a.done_file}: exists={done["exists"]}, complete={done["complete"]}); use --force to override')
            done['forced'] = True
    exclusions = apply_exclusions(data, BENCH / 'protocol-v1.md')
    windows = sorted({k[0] for k in data})
    gids = generated_ids(data); K = len(gids)
    if gids != list(range(K)):
        raise SystemExit(f'generated samples are not 0..K-1: {gids}')
    if a.K is not None and a.K != K:
        raise SystemExit(f'expected K = {a.K}, found {K}')
    meta = {}
    for w in windows:
        p = a.windows_dir / w / 'window.json'
        meta[w] = json.loads(p.read_text()) if p.is_file() else None
    missing_meta = [w for w in windows if meta[w] is None]
    rules = json.loads(a.rules.read_text()) if a.rules and a.rules.is_file() else {}
    clim_data, _ = load_runs([a.climatology]); fit_path = a.cv_fit or a.climatology
    fit_data = clim_data if fit_path == a.climatology else load_runs([fit_path])[0]
    clim_meta = {}
    for w in sorted({k[0] for k in clim_data} | {k[0] for k in fit_data}):
        p = a.clim_windows_dir / w / 'window.json'
        clim_meta[w] = json.loads(p.read_text()) if p.is_file() else None
    hist_flagged = sorted({k[1] for k, v in data.items() for r in v.values() if 'history_not_reliable' in r.flagset})
    bases = dict(obj=Baselines(clim_data, clim_meta, fit_data, clim_meta), history_not_reliable=set(rules.get('history_not_reliable', hist_flagged)))
    overlap_logs = sorted({log_of(meta, w) for w in windows} & {log_of(clim_meta, w) for w in clim_meta} - {None})
    families = [f for f in FAMILY_ORDER if any(k[1] == f for k in data)]
    L = a.label
    R = dict(label=L, K=K, windows=len(windows), generated_samples=gids, families=families, runs=[str(p) for p in a.runs],
             protocol='bench-v1 protocol-v1.md section 5 (frozen 2026-11-02) + protocol-v1.1-instrument.md (frozen 2026-11-02)',
             windows_without_window_json=missing_meta, climatology_logs_left_out_for_overlap=len(overlap_logs),
             history_not_reliable=sorted(bases['history_not_reliable']), history_not_reliable_in_flags=hist_flagged,
             protocol_exclusions=exclusions, test_completeness=done,
             settings=dict(seed_ties=SEED, boot=a.boot, mc=a.mc, mc_explore=a.mc_explore, perm=a.perm, tie_draws=a.tie_draws), families_results={})
    # ---------------------------------------------------------------- per family: primary outcome, main analysis
    main_rows = {}
    for fam in families:
        o = PRIMARY[fam]; rows, info = build(data, fam, o, gids)
        rank_rows(rows); s = family_stats(rows, K, a.mc, a.boot, ('main', fam, o)) if rows else dict(n=0)
        base = baseline_scores(rows, data, fam, o, meta, bases, 'real', a.boot, ('main', fam, o)) if rows else None
        main_rows[fam] = rows
        R['families_results'][fam] = dict(role='scored' if fam in SCORED else 'descriptive', outcome=o, kind=kinds.get((fam, o)), scorability=info, stats=s, baselines=base,
                                          rule=rule_of(fam))
        if rows:
            pit_figure(rows, K, f'{L} {fam} {o} ({"scored" if fam in SCORED else "descriptive"}), n = {len(rows)}, K = {K}', a.out / f'pit-{fam}.png', ('fig', fam))
    # ---------------------------------------------------------------- H1, H2, Holm
    h1s = R['families_results'].get('lead', {}).get('stats', {})
    h2 = h2_test({f: main_rows.get(f, []) for f in H2_POOL}, a.perm, a.boot, ('main',))
    conf = {}
    if h1s.get('n', 0) and h2.get('n_windows', 0):
        p1, p2 = h1s['slope_p_one_sided_normal'], h2['p_mc']; a1, a2 = holm(p1, p2)
        conf['H1'] = dict(family='lead', outcome='x_4s', n=h1s['n'], slope_z=h1s['slope_z'], p_one_sided=p1, p_one_sided_mc=h1s['slope_p_one_sided_mc'], p_holm=a1,
                          mean_pit_minus_half=h1s['mean_pit_minus_half'], mean_pit_ci95=h1s['mean_pit_ci95'],
                          confirmed=bool(a1 <= 0.05 and h1s['mean_pit_ci95'][0] > 0), ci_above_zero=bool(h1s['mean_pit_ci95'][0] > 0))
        conf['H2'] = dict({k: v for k, v in h2.items() if not k.startswith('_')}, p_holm=a2, confirmed=bool(a2 <= 0.05))
    else:
        conf['note'] = 'H1 or H2 has no data'
        conf['H2'] = {k: v for k, v in h2.items() if not k.startswith('_')}
    R['confirmatory'] = conf
    # ---------------------------------------------------------------- secondary: BH within each statistic across scored families
    sec = {}
    for lab, getp in (('mean_pit', lambda s: s['mean_pit_p_two_sided']), ('above_all', lambda s: s['above_all']['p_two_sided']),
                      ('below_all', lambda s: s['below_all']['p_two_sided']), ('convexity', lambda s: s['convexity_p_two_sided_mc'])):
        fams = [f for f in SCORED if R['families_results'].get(f, {}).get('stats', {}).get('n', 0)]
        ps = [getp(R['families_results'][f]['stats']) for f in fams]; qs = awc.bh(ps) if ps else []
        sec[lab] = {f: dict(p=p, q=q, significant_q_0_10=bool(q <= 0.10)) for f, p, q in zip(fams, ps, qs)}
    R['secondary_bh'] = sec
    # ---------------------------------------------------------------- pipeline checks
    pc, sh = {}, {}
    for fam in families:
        o = PRIMARY[fam]
        if K >= 3:
            rows, info = build(data, fam, o, [g for g in gids if g != 0], truth='0'); rank_rows(rows)
            st = family_stats(rows, K - 1, a.mc, a.boot, ('pc', fam)) if rows else dict(n=0)
            pc[fam] = dict(stats=st, scorability=info, expected_outside=2 / K)
        halves = {'first half (samples 0..%d)' % (K // 2 - 1): gids[:K // 2], 'second half (samples %d..%d)' % (K // 2, K - 1): gids[K // 2:]}
        sh[fam] = {}
        for hl, hg in halves.items():
            if len(hg) < 2:
                continue
            rows, info = build(data, fam, o, hg); rank_rows(rows)
            sh[fam][hl] = dict(stats=family_stats(rows, len(hg), a.mc, a.boot, ('sh', fam, hl)) if rows else dict(n=0), _rows=rows)
    sh_h12 = {}
    for hl in (next(iter(sh.values())) if sh else {}):
        frows = {f: sh[f][hl]['_rows'] for f in H2_POOL if f in sh and hl in sh[f]}
        hh = h2_test(frows, a.perm, a.boot, ('sh', hl), exact=False); lead = sh.get('lead', {}).get(hl, {}).get('stats', {})
        sh_h12[hl] = dict(H1_slope_z=lead.get('slope_z'), H1_mean_pit_minus_half=lead.get('mean_pit_minus_half'), H1_mean_pit_ci95=lead.get('mean_pit_ci95'),
                          H2={k: v for k, v in hh.items() if not k.startswith('_')})
    R['pipeline_checks'] = dict(positive_control=pc, split_half={f: {h: {'stats': v['stats']} for h, v in d.items()} for f, d in sh.items()}, split_half_H1_H2=sh_h12)
    # ---------------------------------------------------------------- sensitivity (exploratory)
    sens = dict(carried_rates={}, carried_lost_missing={}, trackers={}, raw_reference={}, secondary_outcomes={}, strata={}, tie_break_spread={})
    for fam in families:
        o = PRIMARY[fam]; rule = rule_of(fam)
        keys = [k for k in data if k[1] == fam and k[2] == o]
        real = [data[k].get('real') for k in keys if usable(data[k].get('real'), rule)]
        gen = [data[k].get(str(g)) for k in keys if usable(data[k].get('real'), rule) for g in gids if usable(data[k].get(str(g)), rule)]
        rate = lambda rs, fl: float(np.mean([fl in r.flagset for r in rs])) if rs else None
        sens['carried_rates'][fam] = dict(outcome=o, real_n=len(real), generated_n=len(gen), real_carried=rate(real, 'carried'), generated_carried=rate(gen, 'carried'),
                                          real_carried_lost=rate(real, 'carried_lost'), generated_carried_lost=rate(gen, 'carried_lost'),
                                          real_carried_edge=rate(real, 'carried_edge'), generated_carried_edge=rate(gen, 'carried_edge'))
        rows, info = build(data, fam, o, gids, drop_lost=True); rank_rows(rows)
        sens['carried_lost_missing'][fam] = dict(stats=family_stats(rows, K, a.mc_explore, a.boot, ('lost', fam)) if rows else dict(n=0), scorability=info)
        sens['trackers'][fam] = {}
        for col in ('value_v3', 'value_v5', 'value_v8'):
            rows, info = build(data, fam, o, gids, col=col); rank_rows(rows)
            sens['trackers'][fam][col] = dict(stats=family_stats(rows, K, a.mc_explore, a.boot, ('trk', fam, col)) if rows else dict(n=0), scorability=info)
        rows, info = build(data, fam, o, gids, truth='raw'); rank_rows(rows)
        st = family_stats(rows, K, a.mc_explore, a.boot, ('raw', fam)) if rows else dict(n=0)
        sens['raw_reference'][fam] = dict(stats=st, scorability=info, baselines=baseline_scores(rows, data, fam, o, meta, bases, 'raw', a.boot, ('raw', fam, o)) if rows else None)
        sens['secondary_outcomes'][fam] = {}
        for (f2, o2), kd in sorted(kinds.items()):
            if f2 != fam or o2 == o or kd not in ('secondary', 'descriptive'):
                continue
            rows, info = build(data, fam, o2, gids); rank_rows(rows)
            st = family_stats(rows, K, a.mc_explore, a.boot, ('sec', fam, o2)) if rows else dict(n=0)
            sens['secondary_outcomes'][fam][o2] = dict(kind=kd, stats=st, scorability=info,
                                                       baselines=baseline_scores(rows, data, fam, o2, meta, bases, 'real', a.boot, ('sec', fam, o2)) if rows else None)
        if fam in SCORED and main_rows[fam]:
            sens['strata'][fam] = {lab: (dict(n=len(rs), stats=family_stats(rs, K, a.mc_explore, a.boot, ('str', fam, lab))) if len(rs) >= 10 else dict(n=len(rs), note='n < 10: not tested'))
                                   for lab, rs in strata_of(main_rows[fam], fam, meta).items()}
        sens['tie_break_spread'][fam] = tie_spread(main_rows[fam], K, a.tie_draws, ('tie', fam)) if main_rows[fam] else None
    sens['tie_break_spread']['H2_T'] = h2_tie_spread(h2.get('_wins', []), a.tie_draws, ('tie', 'h2'))
    if a.pilot_summary and a.pilot_summary.is_file():
        ps = json.loads(a.pilot_summary.read_text())
        sens['pilot_study_v1'] = dict(source=str(a.pilot_summary), lead_x4_mean_pit_minus_half=ps.get('mean_pit_minus_half'), ci95=ps.get('mean_pit_ci95'),
                                      slope_z=ps.get('slope_z'), above_all=ps.get('share_real_above_all'), below_all=ps.get('share_real_below_all'), windows=ps.get('windows'), K=ps.get('K'),
                                      note='study-v1: portrait geometry, other prompt, seeds shared across windows, other instrument; comparison only')
    R['sensitivity'] = sens
    # ---------------------------------------------------------------- scorability summary
    R['scorability'] = {f: R['families_results'][f]['scorability'] for f in families}
    R['finished'] = datetime.datetime.now().astimezone().isoformat(timespec='seconds')
    (a.out / 'results.json').write_text(json.dumps(clean(R), indent=1) + '\n')
    write_md(R, a.out / 'results.md', families)
    with (a.out / 'inputs.log').open('w') as fh:
        fh.write(f'# analyze_bench_v1.py, label {L}, started {t_start.isoformat(timespec="seconds")}, finished {R["finished"]}\n')
        fh.write('command: ' + ' '.join(sys.argv) + '\n')
        for p in list(a.runs) + [a.climatology] + ([a.cv_fit] if a.cv_fit else []) + [a.rules] + list(a.extra_inputs) + ([a.done_file] if a.done_file.is_file() else []):
            fh.write(f'{sha256_file(p)}  {p}\n')
        fh.write(f'test completeness gate: {json.dumps(done)}\n')
        fh.write(f'protocol exclusions: {json.dumps(exclusions)}\n')
        for p in (Path(__file__).resolve(), HERE / 'analyze_wm_calibration.py', BENCH / 'protocol-v1.md', BENCH / 'protocol-v1.1-instrument.md'):
            if p.is_file():
                fh.write(f'{sha256_file(p)}  {p}\n')
        hw = hashlib.sha256()
        for w in windows:
            p = a.windows_dir / w / 'window.json'
            if p.is_file():
                hw.update(w.encode()); hw.update(p.read_bytes())
        fh.write(f'{hw.hexdigest()}  combined window.json of the {len(windows)} analysed windows ({a.windows_dir})\n')
        fh.write(f'numpy {np.__version__}, scipy {__import__("scipy").__version__}, python {sys.version.split()[0]}\n')
    print(json.dumps(dict(label=L, K=K, windows=len(windows), confirmatory={k: {kk: vv for kk, vv in v.items() if kk in ('confirmed', 'p_holm', 'slope_z', 'T', 'p_mc', 'mean_pit_minus_half', 'mean_pit_ci95')} for k, v in conf.items() if isinstance(v, dict)}, out=str(a.out))))


def write_md(R, path, families):
    L, K = R['label'], R['K']; fr = R['families_results']; sec = R['secondary_bh']
    q = lambda f: {k: sec[k][f]['q'] for k in sec if f in sec[k]}
    lines = [f'# bench-v1 analysis ({L}, K = {K}, {R["windows"]} windows)', '',
             f'Protocol: {R["protocol"]}. Runs: ' + ', '.join(f'`{Path(p).name}`' for p in R['runs']) + '. Input sha256s: `inputs.log`.', '']
    if L.upper() != 'TEST':
        lines += [f'**{L} RUN: NOT A CONFIRMATORY RESULT.** The labels below give the role each analysis has on TEST.', '']
    for k, v in R['protocol_exclusions'].items():
        lines.append(f"Protocol exclusion: {k} ({v['reason']}): {v['rows_dropped']} rows dropped" + ('' if v['rows_dropped'] else ' (not present in these inputs)') + '.')
    tc = R['test_completeness']
    if tc['required']:
        lines.append(f"TEST completeness: `{Path(tc['file']).name}` complete = {tc['complete']}" + (' — **run with --force: the measurement was NOT complete**' if tc['forced'] else '') + '.')
    lines.append('')
    c = R['confirmatory']
    lines += ['## Confirmatory (H1, H2; Holm across the two)', '']
    if 'H1' in c:
        h1, h2 = c['H1'], c['H2']
        lines += ['| hypothesis | n | statistic | one-sided p | Holm-adjusted p | extra condition | decision |', '|---|---|---|---|---|---|---|',
                  f"| H1: lead x_4s, generated too small | {h1['n']} | JP slope z = {h1['slope_z']:+.2f} (MC p {fmt_p(h1['p_one_sided_mc'])}) | {fmt_p(h1['p_one_sided'])} | {fmt_p(h1['p_holm'])} | "
                  f"bootstrap CI of mean PIT - 0.5 = {h1['mean_pit_minus_half']:+.3f} [{h1['mean_pit_ci95'][0]:+.3f}, {h1['mean_pit_ci95'][1]:+.3f}] above 0: {'yes' if h1['ci_above_zero'] else 'no'} | "
                  f"{'confirmed' if h1['confirmed'] else 'not confirmed'} |",
                  f"| H2: real outside the generated support more often than 2/(K+1) (pool {', '.join(H2_POOL)}) | {h2['n_windows']} windows | T = {h2['T']:.3f} (null mean {h2['null_mean']:.3f}; bootstrap CI [{h2['boot_ci95'][0]:.3f}, {h2['boot_ci95'][1]:.3f}]) | "
                  f"{fmt_p(h2['p_mc'])} (B = {h2['B']}; exact {fmt_p(h2.get('p_exact_fft'))}) | {fmt_p(h2['p_holm'])} | | {'confirmed' if h2['confirmed'] else 'not confirmed'} |", '',
                  f"H2 windows dropped: {h2.get('dropped')}; families per window: {h2.get('families_per_window')}; clips per window: {h2.get('clips_per_window')}.", '']
    else:
        lines += [f"No H1/H2 decision: {c.get('note')}", '']
    lines += ['## Secondary (scored families, primary outcomes; BH q = 0.10 within each statistic across the families)', '', HEAD]
    for f in [f for f in families if f in SCORED]:
        d = fr[f]; lines.append(fam_line(f, d['outcome'], d['stats'], q(f), d['baselines']))
    lines += ['', 'p-values: mean PIT z test (null variance 1/(12 n)), two-sided; convexity Monte Carlo, two-sided; above / below all exact binomial '
              '(Poisson-binomial when K_i varies), two-sided; intervals Clopper-Pearson for n <= 45, Wilson otherwise. The baselines\' own PIT statistics are '
              'descriptive (results.json).', '']
    sig = [f'{k}: ' + ', '.join(f for f, v in sec[k].items() if v['significant_q_0_10']) for k in sec if any(v['significant_q_0_10'] for v in sec[k].values())]
    lines += ['Significant at q <= 0.10: ' + ('; '.join(sig) if sig else 'none') + '.', '']
    lines += ['Constant-velocity baseline used: ' + '; '.join(f"{f}: {fr[f]['baselines']['cv_baseline_used']}" for f in families if f in SCORED and fr[f].get('baselines')) + '.', '']
    lines += ['## Descriptive families (ped by the freezing decision; startup, crossing, rider)', '', HEAD]
    for f in [f for f in families if f in DESCRIPTIVE]:
        d = fr[f]; lines.append(fam_line(f, d['outcome'] + (' (rule: reliability labels ignored)' if d['rule'] == 'descriptive' else ''), d['stats'], None, d['baselines']))
    for f in [f for f in families if f in DESCRIPTIVE]:
        for o2, v in R['sensitivity']['secondary_outcomes'].get(f, {}).items():
            lines.append(fam_line(f, o2, v['stats'], None, v['baselines']))
    lines += ['', '## Pipeline checks (descriptive)', '', f'Positive control: sample 0 as the truth against the other K - 1 = {K - 1} (expected outside share 2/K = {2 / K:.3f}).', '', HEAD]
    for f, v in R['pipeline_checks']['positive_control'].items():
        lines.append(fam_line(f, PRIMARY[f], v['stats']))
    lines += ['', 'Split-half:', '', HEAD]
    for f, d in R['pipeline_checks']['split_half'].items():
        for h, v in d.items():
            lines.append(fam_line(f, f'{PRIMARY[f]}, {h}', v['stats']))
    for h, v in R['pipeline_checks']['split_half_H1_H2'].items():
        h2 = v['H2']
        lines.append(f"- {h}: H1 slope z {v['H1_slope_z'] if v['H1_slope_z'] is None else format(v['H1_slope_z'], '+.2f')}; H2 T = "
                     f"{h2.get('T', float('nan')):.3f} (null mean {h2.get('null_mean', float('nan')):.3f}, p {fmt_p(h2.get('p_mc'))}, {h2.get('n_windows')} windows)")
    s = R['sensitivity']
    lines += ['', '## Exploratory sensitivity analyses', '', '### Carried readings at the primary outcome (shares of usable readings)', '',
              '| family | outcome | real n | real carried / carried_lost | generated n | generated carried / carried_lost |', '|---|---|---|---|---|---|']
    for f, v in s['carried_rates'].items():
        fm = lambda x: 'n/a' if x is None else f'{x:.3f}'
        lines.append(f"| {f} | {v['outcome']} | {v['real_n']} | {fm(v['real_carried'])} / {fm(v['real_carried_lost'])} | {v['generated_n']} | {fm(v['generated_carried'])} / {fm(v['generated_carried_lost'])} |")
    lines += ['', '### carried_lost readings treated as missing (K_i shrinks)', '', HEAD]
    for f, v in s['carried_lost_missing'].items():
        lines.append(fam_line(f, PRIMARY[f], v['stats']))
    lines += ['', '### Each tracker alone', '', HEAD]
    for f, d in s['trackers'].items():
        if not any(v['stats'].get('n', 0) for v in d.values()):
            lines.append('| ' + ' | '.join([f, PRIMARY[f], '0', 'no per-tracker values (ego odometry has one reading)' if f == 'ego_inter' else 'no data'] + [''] * 9) + ' |'); continue
        for col, v in d.items():
            lines.append(fam_line(f, f'{PRIMARY[f]} ({col})', v['stats']))
    lines += ['', '### Raw real clip as the reference', '', HEAD]
    for f, v in s['raw_reference'].items():
        lines.append(fam_line(f, PRIMARY[f], v['stats'], None, v['baselines']))
    lines += ['', '### Secondary outcomes (scored families)', '', HEAD]
    for f in [f for f in families if f in SCORED]:
        for o2, v in s['secondary_outcomes'].get(f, {}).items():
            lines.append(fam_line(f, o2, v['stats'], None, v['baselines']))
    lines += ['', '### Strata by past-only covariates (primary outcomes; tertiles within the family)', '', HEAD]
    for f, d in s['strata'].items():
        for lab, v in d.items():
            lines.append(fam_line(f, lab, v.get('stats')) if 'stats' in v else '| ' + ' | '.join([f, lab, str(v['n']), v['note']] + [''] * 9) + ' |')
    lines += ['', '### Spread over tie-break draws', '', '| family | windows with ties | mean PIT - 0.5: mean (sd) [min, max] | slope z: mean (sd) [min, max] | above all: [min, max] |', '|---|---|---|---|---|']
    for f, v in s['tie_break_spread'].items():
        if f == 'H2_T' or v is None:
            continue
        m, z, ab = v['mean_pit_minus_half'], v['slope_z'], v['above_all']
        lines.append(f"| {f} | {v['windows_with_ties']} | {m['mean']:+.3f} ({m['sd']:.3f}) [{m['min']:+.3f}, {m['max']:+.3f}] | {z['mean']:+.2f} ({z['sd']:.2f}) [{z['min']:+.2f}, {z['max']:+.2f}] | [{ab['min']:.3f}, {ab['max']:.3f}] |")
    if s['tie_break_spread'].get('H2_T'):
        t = s['tie_break_spread']['H2_T']; lines.append(f"\nH2's T over {t['draws']} tie-break draws: mean {t['mean']:.3f} (sd {t['sd']:.3f}) [{t['min']:.3f}, {t['max']:.3f}].")
    if 'pilot_study_v1' in s:
        p = s['pilot_study_v1']
        lines += ['', f"Pilot (study-v1, lead gap at 4 s, comparison only): mean PIT - 0.5 = {p['lead_x4_mean_pit_minus_half']:+.3f} [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}], "
                  f"slope z {p['slope_z']:.2f}, above / below all {p['above_all']:.3f} / {p['below_all']:.3f} (n = {p['windows']}, K = {p['K']})."]
    lines += ['', '## Scorability', '', '| family | outcome | windows with the family | kept | dropped (reason: count) | generated readings not usable (reason: count) |', '|---|---|---|---|---|---|']
    for f in families:
        d = fr[f]['scorability']
        lines.append(f"| {f} | {fr[f]['outcome']} | {d['windows_with_family_outcome']} | {d['kept']} | {'; '.join(f'{k}: {v}' for k, v in d['dropped'].items()) or '-'} | "
                     f"{'; '.join(f'{k}: {v}' for k, v in d['generated_not_usable'].items()) or '-'} |")
    lines += ['', f"History-not-reliable families (constant velocity from the annotation, privileged): {', '.join(R['history_not_reliable']) or 'none'}. "
              f"Climatology: DEV raw readings; windows of the analysed windows' own logs left out ({R['climatology_logs_left_out_for_overlap']} logs overlap).",
              f"Windows without window.json: {len(R['windows_without_window_json'])}."]
    path.write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
