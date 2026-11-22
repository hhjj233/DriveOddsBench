#!/usr/bin/env python3
"""bench-v1 amendment v1.2: analysis of the panel of physical quantities (protocol-v1.2-quantities.md section 5, frozen
2026-11-02; v12-frozen.txt). Written 2026-11-02. Everything here is secondary or exploratory; H1 and H2 of
protocol-v1.md are untouched.

Inputs
  --runs        v1.2 CSVs in the section-10 format (kind v1.2-secondary / v1.2-descriptive): DEV
                instrument/outcomes/v12-dev-k4.csv; TEST instrument/outcomes/test/v1.2/v12-test-k0-10.csv and
                v12-test-k11-15.csv (combined per window as analyze_bench_v1.py does; shared real rows must agree).
  --verdicts    instrument/v1.2/verdicts-v1.2.json: only `reliable` (family, quantity) pairs are tested.
  --clim-derived instrument/v1.2/derived-dev/derived-nano-raw.json: the DEV raw values of every quantity (220 windows),
                the climatology; --clim-flags the frozen DEV raw outcome CSV, whose rows give each raw (window, family)
                its frozen t0 flags; windows from an analysed window's own log are left out (DEV dry runs only).
TEST input is refused unless bench-v1/test-v1.2-derived.done exists and every sha256 in v12-frozen.txt matches
(--force overrides; recorded), and --allow-test is given.

Analysis (protocol-v1.2 section 5)
  - Tested: the (family, quantity) pairs with verdict `reliable` (startup and ped descriptive). The others, and startup,
    are reported descriptively (reliability labels ignored, every other hard flag kept).
  - Per (window, quantity): the real reference needs a usable value; generated samples without one are dropped, and the
    window is kept if K_i >= ceil(0.75 K) of K remain (12 of 16; DEV 3 of 4). Ranks with randomised ties (also at caps)
    by analyze_wm_calibration.ranks_of, generator seeded 2026 per analysis, windows in name order; PIT = (rank + U) /
    (K_i + 1).
  - Statistics (analyze_bench_v1.family_stats): mean PIT - 0.5 (exact-null z, window-bootstrap CI), Jolliffe-Primo slope
    and convexity, chi-square with a Monte Carlo p-value (100,000), above-all / below-all exact tests (binomial; Poisson-
    binomial when K_i varies) with Clopper-Pearson (n <= 45) or Wilson intervals, fair CRPS; CRPSS against climatology.
  - Binary ego_stop_4s: no PIT; the Brier score of the generated rate against the real indicator, its fair version, the
    Brier skill against the DEV raw stop rate, and an exact exchangeability test of the real stop count (given each
    window's K_i + 1 indicators, the real is a stop with probability S_i / (K_i + 1); Poisson-binomial).
  - BH q = 0.10 within each quantity across families, per statistic (mean PIT, above-all, below-all, convexity; the
    exchangeability test for ego_stop_4s). All labelled secondary. The ego family is reported separately and is never
    part of H2.
  - Exploratory: the calibration profile (one row per tested pair: mean PIT offset with CI, outside share against its
    null, CRPSS); coverage panels (window-level permutation of the real clip among the clips usable on all the window's
    tested continuous quantities; non-ego and ego separately); positive control (sample 0 as truth).

  analyze_bench_v12.py --runs A.csv [B.csv] --windows-dir BENCH/windows-test --label TEST --K 16 --allow-test --out DIR
"""
from __future__ import annotations

import os

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '2')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

import argparse, csv, datetime, hashlib, json, math, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_bench_v1 as ab  # noqa: E402  load_runs, rank_rows, family_stats, count_test, share_interval, boot_ci, rng_for, apply_exclusions, ...

awc = ab.awc
BENCH = ab.BENCH
HARD12 = {'t0_not_found', 'real_t0_not_found', 't0_box_at_edge', 'family_not_reliable', 'quantity_not_reliable', 'no_value', 'no_vo_record'}
RELIAB12 = {'family_not_reliable', 'quantity_not_reliable'}
BINARY = {'ego_stop_4s', 'lead_hard_brake_0_4s'}
FAMILY_ORDER = ['lead', 'lead_stop', 'cutin', 'oncoming', 'ego', 'startup', 'ped']
DESCRIPTIVE_FAMILIES = {'startup', 'ped'}
KMIN_FRAC = 0.75


def k_min(K):
    return int(math.ceil(KMIN_FRAC * K - 1e-9))


def usable12(row, rule):
    if row is None or row.value is None:
        return False
    if rule == 'tested':
        return row.scorable and not (row.flagset & HARD12)
    return not ((row.flagset & HARD12) - RELIAB12)            # descriptive: reliability labels ignored


def build12(data, fam, q, gen_ids, rule, truth='real'):
    rows, drops, gdrops = [], Counter(), Counter(); K = len(gen_ids); km = k_min(K)
    for key in sorted(k for k in data if k[1] == fam and k[2] == q):
        smp = data[key]; tr = smp.get(truth)
        if tr is None:
            drops['no real row'] += 1; continue
        if not usable12(tr, rule):
            drops['real not usable: ' + ('+'.join(sorted((tr.flagset & HARD12) - (RELIAB12 if rule != 'tested' else set()))) or 'no value')] += 1; continue
        ids, Y = [], []
        for g in gen_ids:
            r = smp.get(str(g))
            if r is None:
                gdrops['missing row'] += 1
            elif not usable12(r, rule):
                gdrops['+'.join(sorted((r.flagset & HARD12) - (RELIAB12 if rule != 'tested' else set()))) or 'no value'] += 1
            else:
                ids.append(g); Y.append(r.value)
        if len(Y) < km:
            drops[f'K_i < {km} of {K}'] += 1; continue
        rows.append(dict(name=key[0], y=tr.value, ids=ids, Y=np.array(Y, float), K_full=K, censored_real='censored' in tr.flagset,
                         censored_gen=sum('censored' in smp[str(g)].flagset for g in ids), carried_real='carried' in tr.flagset))
    return rows, dict(windows_with_quantity=len([k for k in data if k[1] == fam and k[2] == q]), kept=len(rows), dropped=dict(drops),
                      generated_not_usable=dict(gdrops), K_min=km)


# -------------------------------------------------------------------------------------------------- climatology
def load_climatology(derived_path, flags_csv, clim_meta):
    """{(family, quantity): [(log, value), ...]} from the DEV raw derived values; a raw (window, family) counts if its
    frozen raw rows carry no t0 flag (t0_not_found, real_t0_not_found, t0_box_at_edge); ego needs a value."""
    der = json.loads(Path(derived_path).read_text()); bad = set()
    with open(flags_csv, newline='') as fh:
        for r in csv.DictReader(fh):
            if r['sample'] == 'raw' and set(t.split('=')[0] for t in r['flags'].split(';') if t) & {'t0_not_found', 'real_t0_not_found', 't0_box_at_edge'}:
                bad.add((r['window'], r['family']))
    out = defaultdict(list); info = Counter()
    for w, samples in der['windows'].items():
        rec = samples.get('raw')
        if rec is None:
            continue
        log = (clim_meta.get(w) or {}).get('log')
        for fam, qs in list((rec.get('families') or {}).items()) + [('ego', rec.get('ego') or {})]:
            if fam != 'ego' and (w, fam) in bad:
                info[f'{fam}: t0 flag'] += 1; continue
            for q, d in qs.items():
                if d.get('value') is None or 'no_value' in (d.get('flags') or []) or 'no_vo_record' in (d.get('flags') or []):
                    continue
                out[(fam, q)].append((log, float(d['value'])))
    return out, dict(info), der.get('source')


def clim_values(clim, fam, q, log):
    return np.array([v for lg, v in clim.get((fam, q), []) if lg is None or lg != log], float)


def clim_crpss(rows, clim, fam, q, meta, boot, tag):
    model, base, keep = [], [], []
    for r in rows:
        cv = clim_values(clim, fam, q, (meta.get(r['name']) or {}).get('log'))
        if len(cv) < 5:
            continue
        model.append(r['crps']); base.append(awc.fair_crps(cv, r['y'])); keep.append(r)
    if len(model) < 3:
        return dict(n=len(model), note='fewer than 3 windows with a climatology')
    m, b = np.array(model), np.array(base)
    if b.mean() <= 0:
        return dict(n=len(m), note='climatology CRPS is 0: skill undefined')
    idx = ab.rng_for('boot', tag, 'clim').integers(0, len(m), (boot, len(m))); den = b[idx].mean(1)
    bs = np.where(den > 0, 1 - m[idx].mean(1) / np.where(den > 0, den, 1.0), np.nan)
    return dict(n=len(m), climatology_values_median=int(np.median([len(clim_values(clim, fam, q, (meta.get(r['name']) or {}).get('log'))) for r in keep])),
                mean_crps_model=float(m.mean()), mean_crps_climatology=float(b.mean()), crpss=float(1 - m.mean() / b.mean()),
                crpss_ci95=[float(np.nanquantile(bs, 0.025)), float(np.nanquantile(bs, 0.975))])


# ------------------------------------------------------------------------------------------------------- binary
def binary_stats(rows, clim, fam, q, meta, boot, tag):
    """Brier score of the generated rate against the real indicator; exact exchangeability test of the real count."""
    if not rows:
        return dict(n=0)
    y = np.array([r['y'] for r in rows], float); p = np.array([r['Y'].mean() for r in rows]); Ki = np.array([len(r['Y']) for r in rows])
    S = np.array([r['y'] + r['Y'].sum() for r in rows]); pi = S / (Ki + 1)
    k = int(y.sum()); t = ab.count_test(k, pi) if 0 < len(pi) else {}
    bs = (p - y) ** 2; fair = np.array([awc.fair_crps(r['Y'], r['y']) for r in rows]); rng = ab.rng_for('boot', tag, 'brier')
    out = dict(n=len(rows), K_i=dict(min=int(Ki.min()), median=float(np.median(Ki))), real_rate=float(y.mean()), generated_rate=float(p.mean()),
               rate_difference_real_minus_generated=float(y.mean() - p.mean()), rate_difference_ci95=ab.boot_ci(y - p, rng, boot),
               brier=float(bs.mean()), brier_ci95=ab.boot_ci(bs, rng, boot), fair_brier=float(fair.mean()),
               exchangeability_test=dict(statistic='real stop count against Poisson-binomial(S_i / (K_i + 1))', count=k, expected=float(pi.sum()), **t))
    cl = []
    for r in rows:
        cv = clim_values(clim, fam, q, (meta.get(r['name']) or {}).get('log'))
        cl.append(np.nan if len(cv) < 5 else (cv.mean() - r['y']) ** 2)
    cl = np.array(cl); ok = np.isfinite(cl)
    if ok.sum() >= 3 and cl[ok].mean() <= 0:
        out['brier_skill_vs_climatology'] = dict(n=int(ok.sum()), note='climatology Brier is 0: skill undefined')
    elif ok.sum() >= 3:
        idx = rng.integers(0, ok.sum(), (boot, ok.sum())); m, b = bs[ok], cl[ok]; den = b[idx].mean(1)
        sk = np.where(den > 0, 1 - m[idx].mean(1) / np.where(den > 0, den, 1.0), np.nan)
        out['brier_skill_vs_climatology'] = dict(n=int(ok.sum()), climatology_rate=float(np.mean([clim_values(clim, fam, q, None).mean()])),
                                                 brier_climatology=float(b.mean()), skill=float(1 - m.mean() / b.mean()),
                                                 skill_ci95=[float(np.nanquantile(sk, 0.025)), float(np.nanquantile(sk, 0.975))])
    return out


# --------------------------------------------------------------------------------------------- coverage panels
def coverage_panel(rows_by_key, perm_B, boot, tag):
    """Window-level permutation test over several (family, quantity) outcomes (exploratory): per window, the clips are
    the real reference and the generated samples usable on all its outcomes; out_c = share of the window's outcomes
    on which clip c lies above or below all others (ties: the real's rank draw; generated ties by a seeded stream)."""
    by_w = defaultdict(dict)
    for key, rows in rows_by_key.items():
        for r in rows:
            by_w[r['name']][key] = r
    rng_keys = ab.rng_for('ties_h2', tag); wins, drops = [], Counter()
    for w in sorted(by_w):
        outs = [by_w[w][k] for k in sorted(by_w[w])]
        complete = set(outs[0]['ids'])
        for r in outs[1:]:
            complete &= set(r['ids'])
        complete = sorted(complete)
        if len(complete) < 2:
            drops['fewer than 2 generated clips usable on all outcomes'] += 1; continue
        clips = ['real'] + complete; outside = np.zeros(len(clips))
        for r in outs:
            keys = rng_keys.random(len(r['ids'])); order = [g for _, _, g in sorted(zip(r['Y'], keys, r['ids']))]; order.insert(r['rank'], 'real')
            sub = [c for c in order if c == 'real' or c in complete]
            for c in (sub[0], sub[-1]):
                outside[clips.index(c)] += 1
        wins.append(dict(name=w, m=len(outs), out=outside / len(outs)))
    n = len(wins)
    if n == 0:
        return dict(n_windows=0, dropped=dict(drops))
    T = float(np.mean([x['out'][0] for x in wins])); null = float(np.mean([x['out'].mean() for x in wins]))
    rng = ab.rng_for('perm', tag); acc = done = 0
    while done < perm_B:
        b = min(10000, perm_B - done); tot = np.zeros(b)
        for x in wins:
            tot += x['out'][rng.integers(0, len(x['out']), b)]
        acc += int((tot / n >= T - 1e-12).sum()); done += b
    U = 1
    for x in wins:
        U = math.lcm(U, x['m'])
    L = U * n + 1; nfft = 1 << int(math.ceil(math.log2(L))); logF = np.zeros(nfft // 2 + 1, complex)
    for x in wins:
        u = np.rint(x['out'] * U).astype(int); F = np.fft.rfft(np.bincount(u, minlength=U + 1) / len(u), nfft)
        logF += np.log(np.where(np.abs(F) < 1e-300, 1e-300, F))
    pmf = np.clip(np.fft.irfft(np.exp(logF), nfft)[:L], 0, None); pmf /= pmf.sum(); s_obs = int(round(T * n * U))
    return dict(n_windows=n, dropped=dict(drops), outcomes=sorted(f'{k[0]}|{k[1]}' for k in rows_by_key), T=T, null_mean=null, B=perm_B,
                p_mc=(1 + acc) / (1 + perm_B), p_exact=float(pmf[s_obs:].sum()), boot_ci95=ab.boot_ci([x['out'][0] for x in wins], ab.rng_for('boot', tag, 'cov'), boot),
                outcomes_per_window=dict(Counter(x['m'] for x in wins)))


# ----------------------------------------------------------------------------------------------- freeze checks
def check_freeze(frozen):
    """Every sha256 recorded in v12-frozen.txt (the protocol on line 2, then '<sha>  <path>' lines)."""
    if not frozen.is_file():
        return dict(file=str(frozen), exists=False, all_match=False, checks=[])
    lines = frozen.read_text().splitlines(); checks = []
    first = lines[0].split()[0] if lines else ''
    for ln in lines[1:]:
        parts = ln.split()
        if len(parts) == 2 and parts[0] == 'sha256':
            p = BENCH / first; checks.append(dict(path=str(p), expected=parts[1]))
        elif len(parts) == 2 and len(parts[0]) == 64:
            p = Path(parts[1]); p = p if p.is_absolute() else BENCH / p; checks.append(dict(path=str(p), expected=parts[0]))
    for c in checks:
        c['actual'] = ab.sha256_file(c['path']) if Path(c['path']).is_file() else None; c['match'] = c['actual'] == c['expected']
    return dict(file=str(frozen), exists=True, all_match=bool(checks) and all(c['match'] for c in checks), checks=checks)


# ---------------------------------------------------------------------------------------------------- reporting
def stat_cells(s):
    if not s or not s.get('n'):
        return ['0'] + [''] * 7
    return [f"{s['n']} (K_i {s['K_i']['min']}-{s['K']})", f"{s['mean_pit_minus_half']:+.3f} [{s['mean_pit_ci95'][0]:+.3f}, {s['mean_pit_ci95'][1]:+.3f}] (p {ab.fmt_p(s['mean_pit_p_two_sided'])})",
            f"{s['slope_z']:+.2f}", f"{s['convexity_z']:+.2f} (p {ab.fmt_p(s['convexity_p_two_sided_mc'])})", ab.fmt_p(s['chi2_p_mc']),
            f"{s['above_all']['share']:.3f} [{s['above_all']['ci95'][0]:.3f}, {s['above_all']['ci95'][1]:.3f}] (p {ab.fmt_p(s['above_all']['p_two_sided'])})",
            f"{s['below_all']['share']:.3f} [{s['below_all']['ci95'][0]:.3f}, {s['below_all']['ci95'][1]:.3f}] (p {ab.fmt_p(s['below_all']['p_two_sided'])})",
            f"{s['above_all']['expected']:.3f}"]


def profile_figure(prof, path, label, K):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    rows = [p for p in prof]; n = len(rows)
    fig, ax = plt.subplots(1, 3, figsize=(13, 0.42 * n + 1.6), sharey=True)
    ys = np.arange(n)[::-1]
    for y, p in zip(ys, rows):
        col = '#55a868' if p['family'] == 'ego' else '#4c72b0'
        if p.get('mean_pit_minus_half') is not None:
            m, ci = p['mean_pit_minus_half'], p['mean_pit_ci95']; ax[0].errorbar(m, y, xerr=[[m - ci[0]], [ci[1] - m]], fmt='o', color=col)
        else:
            ax[0].text(0, y, 'binary: no PIT', ha='center', va='center', fontsize=7)
        if p.get('outside') is not None:
            o = p['outside']; ax[1].errorbar(o['share'], y, xerr=[[o['share'] - o['ci95'][0]], [o['ci95'][1] - o['share']]], fmt='o', color=col)
            ax[1].plot([o['expected']], [y], marker='|', color='k', ms=12)
        if p.get('crpss') is not None:
            c, ci = p['crpss'], p['crpss_ci95']; ax[2].errorbar(c, y, xerr=[[c - ci[0]], [ci[1] - c]], fmt='s' if p['binary'] else 'o', color=col)
    ax[0].axvline(0, color='k', lw=0.8); ax[2].axvline(0, color='k', lw=0.8)
    ax[0].set_yticks(ys); ax[0].set_yticklabels([f"{p['family']} | {p['quantity']} (n={p['n']})" for p in rows], fontsize=8)
    ax[0].set_xlabel('mean PIT - 0.5 (95 % CI); > 0: real above the generated'); ax[1].set_xlabel('outside share (95 % CI); | = calibrated null')
    ax[2].set_xlabel('CRPSS vs DEV raw climatology (square: Brier skill)')
    fig.suptitle(f'{label} (K = {K}) v1.2 calibration profile, reliable quantities (EXPLORATORY display; ego in green)', fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def write_md(R, path):
    L, K = R['label'], R['K']; lines = [f"# bench-v1 v1.2 quantity panel ({L}, K = {K}, {R['windows']} windows)", '',
                                        'Protocol: protocol-v1.2-quantities.md section 5 (frozen 2026-11-02). Everything below is **secondary** (tested pairs, BH within each '
                                        'quantity across families) or **descriptive / exploratory** as labelled; H1 and H2 of protocol-v1 are not touched, and the ego '
                                        'family never enters H2. Input sha256s: `inputs-v12.log`.', '']
    if L.upper() != 'TEST':
        lines += [f'**{L} RUN: NOT A TEST RESULT.**', '']
    fz = R['freeze']; lines += [f"Freeze check (`v12-frozen.txt`): {sum(c['match'] for c in fz['checks'])} of {len(fz['checks'])} sha256 match" + ('' if fz['all_match'] else ' — **MISMATCH**') + '.']
    if R['test_gate']['required']:
        lines.append(f"TEST gate: `test-v1.2-derived.done` exists = {R['test_gate']['done_exists']}" + (' — **run with --force**' if R['test_gate']['forced'] else '') + '.')
    lines += [f"Generated samples kept per window: K_i >= {R['K_min']} of {K}.", '']
    head = ('| family | quantity | n (K_i) | mean PIT - 0.5 [95 % CI] (p) | JP slope z | JP convexity z (p) | chi-square p | above all [CI] (p) | below all [CI] (p) | '
            'expected share | q (mean PIT / above / below / convexity) | fair CRPS | CRPSS vs climatology [CI] (n) | censored real / generated |\n'
            '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for section, fams in (('Secondary: tested (reliable) quantities, non-ego families', [f for f in FAMILY_ORDER if f != 'ego']),
                          ('Secondary: tested quantities, ego family (reported separately; not part of H2)', ['ego'])):
        lines += [f'## {section}', '', head]
        for t in R['tested']:
            if t['family'] not in fams or t['binary']:
                continue
            s, q, c = t['stats'], t['bh_q'], t['climatology']
            qq = ' / '.join('' if q.get(k) is None else f'{q[k]:.3f}' for k in ('mean_pit', 'above_all', 'below_all', 'convexity'))
            cl = '' if not c or 'crpss' not in c else f"{c['crpss']:+.2f} [{c['crpss_ci95'][0]:+.2f}, {c['crpss_ci95'][1]:+.2f}] ({c['n']})"
            cen = '' if not s.get('n') else f"{t['censored_real_share']:.2f} / {t['censored_gen_share']:.2f}"
            lines.append('| ' + ' | '.join([t['family'], t['quantity']] + stat_cells(s) + [qq, '' if not s.get('n') else f"{s['mean_fair_crps']:.3f}", cl, cen]) + ' |')
        for t in R['tested']:
            if t['family'] in fams and t['binary']:
                b = t['binary_stats']
                if not b.get('n'):
                    lines += ['', f"{t['family']} {t['quantity']} (binary): no windows."]; continue
                ex = b['exchangeability_test']; sk = b.get('brier_skill_vs_climatology') or {}
                lines += ['', f"**{t['family']} {t['quantity']} (binary; Brier instead of PIT)**: n = {b['n']}; real rate {b['real_rate']:.3f}, generated rate {b['generated_rate']:.3f} "
                          f"(difference {b['rate_difference_real_minus_generated']:+.3f} [{b['rate_difference_ci95'][0]:+.3f}, {b['rate_difference_ci95'][1]:+.3f}]); Brier {b['brier']:.3f} "
                          f"[{b['brier_ci95'][0]:.3f}, {b['brier_ci95'][1]:.3f}] (fair {b['fair_brier']:.3f}); exchangeability test: real stops {ex['count']} vs expected {ex['expected']:.1f}, "
                          f"p two-sided {ab.fmt_p(ex.get('p_two_sided'))} (q {t['bh_q'].get('exchangeability', float('nan')):.3f})"
                          + (f"; Brier skill vs DEV raw rate {sk['climatology_rate']:.3f}: {sk['skill']:+.2f} [{sk['skill_ci95'][0]:+.2f}, {sk['skill_ci95'][1]:+.2f}]" if sk.get('skill') is not None else
                             (f"; Brier skill: {sk['note']}" if sk else '')) + '.']
        lines.append('')
    sig = [f"{t['family']}|{t['quantity']}: " + ', '.join(k for k, v in t['bh_q'].items() if v is not None and v <= 0.10) for t in R['tested'] if any(v is not None and v <= 0.10 for v in t['bh_q'].values())]
    lines += ['Significant at q <= 0.10: ' + ('; '.join(sig) if sig else 'none') + '.', '']
    lines += ['## Exploratory: calibration profile (one row per tested pair; figure `calibration-profile.png`)', '',
              '| family | quantity | n | mean PIT - 0.5 [95 % CI] | outside share [CI] (null) | CRPSS vs climatology [CI] |', '|---|---|---|---|---|---|']
    for p in R['calibration_profile']:
        mp = 'binary: n/a' if p['mean_pit_minus_half'] is None else f"{p['mean_pit_minus_half']:+.3f} [{p['mean_pit_ci95'][0]:+.3f}, {p['mean_pit_ci95'][1]:+.3f}]"
        o = p.get('outside'); os_ = 'n/a' if o is None else f"{o['share']:.3f} [{o['ci95'][0]:.3f}, {o['ci95'][1]:.3f}] ({o['expected']:.3f})"
        cs = 'n/a' if p['crpss'] is None else f"{p['crpss']:+.2f} [{p['crpss_ci95'][0]:+.2f}, {p['crpss_ci95'][1]:+.2f}]" + (' (Brier skill)' if p['binary'] else '')
        lines.append(f"| {p['family']} | {p['quantity']} | {p['n']} | {mp} | {os_} | {cs} |")
    lines += ['', '## Exploratory: coverage panels (window-level permutation; one window contributes several correlated quantities)', '']
    for k, v in R['coverage_panels'].items():
        if v.get('n_windows'):
            lines.append(f"- {k}: {v['n_windows']} windows, T = {v['T']:.3f} (null mean {v['null_mean']:.3f}; bootstrap CI [{v['boot_ci95'][0]:.3f}, {v['boot_ci95'][1]:.3f}]), "
                         f"p {ab.fmt_p(v['p_mc'])} (B = {v['B']}; exact {ab.fmt_p(v['p_exact'])}); outcomes {', '.join(v['outcomes'])}.")
        else:
            lines.append(f'- {k}: no windows.')
    lines += ['', '## Descriptive (not reliable, not informative, insufficient; startup)', '', head]
    for t in R['descriptive']:
        if t['binary']:
            b = t['binary_stats']
            lines.append('| ' + ' | '.join([t['family'], f"{t['quantity']} ({t['verdict']}; binary)", str(b.get('n', 0)),
                                             '' if not b.get('n') else f"real rate {b['real_rate']:.3f} vs generated {b['generated_rate']:.3f}; Brier {b['brier']:.3f}"] + [''] * 10) + ' |')
            continue
        s = t['stats']; c = t['climatology']
        cl = '' if not c or 'crpss' not in c else f"{c['crpss']:+.2f} [{c['crpss_ci95'][0]:+.2f}, {c['crpss_ci95'][1]:+.2f}] ({c['n']})"
        lines.append('| ' + ' | '.join([t['family'], f"{t['quantity']} ({t['verdict']})"] + stat_cells(s) + ['', '' if not s.get('n') else f"{s['mean_fair_crps']:.3f}", cl,
                                                                                                         '' if not s.get('n') else f"{t['censored_real_share']:.2f} / {t['censored_gen_share']:.2f}"]) + ' |')
    lines += ['', '## Pipeline check (descriptive): positive control, generated sample 0 as the truth', '', head]
    for t in R['positive_control']:
        lines.append('| ' + ' | '.join([t['family'], t['quantity']] + stat_cells(t['stats']) + ['', '', '', '']) + ' |')
    lines += ['', '## Scorability', '', '| family | quantity | verdict | windows with the quantity | kept | dropped (reason: count) | generated not usable (reason: count) |', '|---|---|---|---|---|---|---|']
    for t in R['tested'] + R['descriptive']:
        d = t['scorability']
        lines.append(f"| {t['family']} | {t['quantity']} | {t['verdict']} | {d['windows_with_quantity']} | {d['kept']} | {'; '.join(f'{k}: {v}' for k, v in d['dropped'].items()) or '-'} | "
                     f"{'; '.join(f'{k}: {v}' for k, v in d['generated_not_usable'].items()) or '-'} |")
    lines += ['', f"Climatology: {R['climatology']['source']} (DEV raw), {R['climatology']['pairs']} (family, quantity) pairs; windows of the analysed windows' own logs left out "
              f"({R['climatology']['overlapping_logs']} logs overlap). Protocol exclusions: " + '; '.join(f"{k}: {v['rows_dropped']} rows" for k, v in R['protocol_exclusions'].items()) + '.']
    path.write_text('\n'.join(lines) + '\n')


# --------------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', nargs='+', required=True, type=Path); ap.add_argument('--windows-dir', type=Path, default=BENCH / 'windows-dev')
    ap.add_argument('--verdicts', type=Path, default=BENCH / 'instrument/v1.2/verdicts-v1.2.json')
    ap.add_argument('--clim-derived', type=Path, default=BENCH / 'instrument/v1.2/derived-dev/derived-nano-raw.json')
    ap.add_argument('--clim-flags', type=Path, default=BENCH / 'instrument/outcomes/outcomes-dev-raw-climatology.csv')
    ap.add_argument('--clim-windows-dir', type=Path, default=BENCH / 'windows-dev')
    ap.add_argument('--frozen', type=Path, default=BENCH / 'v12-frozen.txt'); ap.add_argument('--done-file', type=Path, default=BENCH / 'test-v1.2-derived.done')
    ap.add_argument('--out', required=True, type=Path); ap.add_argument('--label', required=True); ap.add_argument('--K', type=int)
    ap.add_argument('--allow-test', action='store_true'); ap.add_argument('--force', action='store_true')
    ap.add_argument('--boot', type=int, default=10000); ap.add_argument('--mc', type=int, default=100000); ap.add_argument('--perm', type=int, default=100000)
    ap.add_argument('--extra-inputs', nargs='*', type=Path, default=[])
    a = ap.parse_args()
    if a.out.exists() and any(a.out.iterdir()):
        raise SystemExit(f'{a.out} exists and is not empty: fresh output required')
    t0 = datetime.datetime.now().astimezone()
    data, kinds = ab.load_runs(a.runs)
    test_names = set()
    for f in ('names-test.txt', 'names-test-gen.txt'):
        if (BENCH / f).is_file():
            test_names |= {l.strip() for l in (BENCH / f).read_text().splitlines() if l.strip()}
    is_test = bool({k[0] for k in data} & test_names) or 'windows-test' in str(a.windows_dir) or any('/test/' in str(p) for p in a.runs)
    freeze = check_freeze(a.frozen); gate = dict(required=is_test, done_file=str(a.done_file), done_exists=a.done_file.is_file(), forced=False)
    if is_test:
        if not a.allow_test:
            raise SystemExit('TEST input: refused without --allow-test')
        problems = ([] if gate['done_exists'] else [f'{a.done_file} missing']) + ([] if freeze['all_match'] else ['v12-frozen.txt sha256 mismatch: ' + ', '.join(c['path'] for c in freeze['checks'] if not c['match'])])
        if gate['done_exists']:
            try:
                d = json.loads(a.done_file.read_text()); gate['done'] = d
                if d.get('complete') is False:
                    problems.append('done file says complete = false')
            except json.JSONDecodeError:
                gate['done'] = a.done_file.read_text()[:500]
        if problems:
            if not a.force:
                raise SystemExit('TEST refused: ' + '; '.join(problems) + ' (use --force to override)')
            gate['forced'] = True; gate['problems_overridden'] = problems
    a.out.mkdir(parents=True, exist_ok=True)
    exclusions = ab.apply_exclusions(data, BENCH / 'protocol-v1.md')
    gids = ab.generated_ids(data); K = len(gids)
    if gids != list(range(K)):
        raise SystemExit(f'generated samples are not 0..K-1: {gids}')
    if a.K is not None and a.K != K:
        raise SystemExit(f'expected K = {a.K}, found {K}')
    verdicts = json.loads(a.verdicts.read_text()); windows = sorted({k[0] for k in data})
    meta = {w: (json.loads((a.windows_dir / w / 'window.json').read_text()) if (a.windows_dir / w / 'window.json').is_file() else None) for w in windows}
    der_names = list(json.loads(a.clim_derived.read_text())['windows'])
    clim_meta = {w: (json.loads((a.clim_windows_dir / w / 'window.json').read_text()) if (a.clim_windows_dir / w / 'window.json').is_file() else None) for w in der_names}
    clim, clim_info, clim_src = load_climatology(a.clim_derived, a.clim_flags, clim_meta)
    overlap = len({(m or {}).get('log') for m in meta.values()} & {(m or {}).get('log') for m in clim_meta.values()} - {None})
    pairs = sorted({(k[1], k[2]) for k in data}, key=lambda p: (FAMILY_ORDER.index(p[0]) if p[0] in FAMILY_ORDER else 99, p[1]))
    R = dict(label=a.label, K=K, K_min=k_min(K), windows=len(windows), runs=[str(p) for p in a.runs], freeze=freeze, test_gate=gate, protocol_exclusions=exclusions,
             climatology=dict(source=str(a.clim_derived), flags=str(a.clim_flags), pairs=len(clim), overlapping_logs=overlap, excluded=clim_info),
             settings=dict(seed_ties=ab.SEED, boot=a.boot, mc=a.mc, perm=a.perm, k_min_fraction=KMIN_FRAC), tested=[], descriptive=[], positive_control=[])
    tested_rows = {}
    for fam, q in pairs:
        verdict = verdicts.get(f'{fam}|{q}', 'no verdict'); tested = verdict == 'reliable' and fam not in DESCRIPTIVE_FAMILIES
        rows, info = build12(data, fam, q, gids, 'tested' if tested else 'descriptive'); ab.rank_rows(rows)
        tag = ('v12', fam, q); entry = dict(family=fam, quantity=q, verdict=verdict, kind=kinds.get((fam, q)), binary=q in BINARY, scorability=info)
        if q in BINARY:
            entry['binary_stats'] = binary_stats(rows, clim, fam, q, meta, a.boot, tag)
        else:
            entry['stats'] = ab.family_stats(rows, K, a.mc, a.boot, tag) if rows else dict(n=0)
            entry['climatology'] = clim_crpss(rows, clim, fam, q, meta, a.boot, tag) if rows else None
            entry['censored_real_share'] = float(np.mean([r['censored_real'] for r in rows])) if rows else None
            entry['censored_gen_share'] = float(np.sum([r['censored_gen'] for r in rows]) / max(1, np.sum([len(r['Y']) for r in rows]))) if rows else None
        (R['tested'] if tested else R['descriptive']).append(entry)
        if tested:
            tested_rows[(fam, q)] = rows
            if rows and not entry['binary']:
                ab.pit_figure(rows, K, f"{a.label} {fam} {q} (secondary), n = {len(rows)}, K = {K}", a.out / f'pit-{fam}-{q}.png', ('fig', fam, q))
            if K >= 3:
                pr, pinfo = build12(data, fam, q, [g for g in gids if g != 0], 'tested', truth='0'); ab.rank_rows(pr)
                if not entry['binary']:
                    R['positive_control'].append(dict(family=fam, quantity=q, stats=ab.family_stats(pr, K - 1, a.mc, a.boot, ('pc', fam, q)) if pr else dict(n=0), scorability=pinfo))
    # BH within each quantity across families, per statistic
    by_q = defaultdict(list)
    for t in R['tested']:
        by_q[t['quantity']].append(t)
    for q, ts in by_q.items():
        getters = ({'exchangeability': lambda t: t['binary_stats'].get('exchangeability_test', {}).get('p_two_sided')} if q in BINARY else
                   {'mean_pit': lambda t: t['stats'].get('mean_pit_p_two_sided'), 'above_all': lambda t: (t['stats'].get('above_all') or {}).get('p_two_sided'),
                    'below_all': lambda t: (t['stats'].get('below_all') or {}).get('p_two_sided'), 'convexity': lambda t: t['stats'].get('convexity_p_two_sided_mc')})
        for t in ts:
            t['bh_q'] = {}
        for lab, get in getters.items():
            have = [t for t in ts if get(t) is not None]; qs = awc.bh([get(t) for t in have]) if have else []
            for t, qv in zip(have, qs):
                t['bh_q'][lab] = qv
            for t in ts:
                t['bh_q'].setdefault(lab, None)
        for t in ts:
            t['bh_family'] = [x['family'] for x in ts]
    # calibration profile (exploratory)
    prof = []
    for t in R['tested']:
        if t['binary']:
            b = t['binary_stats']; sk = (b or {}).get('brier_skill_vs_climatology') or {}   # skill None when undefined
            prof.append(dict(family=t['family'], quantity=t['quantity'], n=b.get('n', 0), binary=True, mean_pit_minus_half=None, mean_pit_ci95=None, outside=None,
                             crpss=sk.get('skill'), crpss_ci95=sk.get('skill_ci95'), real_rate=b.get('real_rate'), generated_rate=b.get('generated_rate')))
        else:
            s, c = t['stats'], t['climatology'] or {}
            if not s.get('n'):
                continue
            prof.append(dict(family=t['family'], quantity=t['quantity'], n=s['n'], binary=False, mean_pit_minus_half=s['mean_pit_minus_half'], mean_pit_ci95=s['mean_pit_ci95'],
                             outside=dict(share=s['outside']['share'], ci95=s['outside']['ci95'], expected=s['outside']['expected'], p_two_sided=s['outside']['p_two_sided']),
                             crpss=c.get('crpss'), crpss_ci95=c.get('crpss_ci95')))
    prof.sort(key=lambda p: (p['family'] == 'ego', FAMILY_ORDER.index(p['family']) if p['family'] in FAMILY_ORDER else 99, p['quantity']))
    R['calibration_profile'] = prof
    if prof:
        profile_figure(prof, a.out / 'calibration-profile.png', a.label, K)
    cont = {k: v for k, v in tested_rows.items() if k[1] not in BINARY}
    R['coverage_panels'] = {'non-ego tested quantities (exploratory)': coverage_panel({k: v for k, v in cont.items() if k[0] != 'ego'}, a.perm, a.boot, ('cov', 'nonego')),
                            'ego tested quantities (exploratory; never part of H2)': coverage_panel({k: v for k, v in cont.items() if k[0] == 'ego'}, a.perm, a.boot, ('cov', 'ego'))}
    R['finished'] = datetime.datetime.now().astimezone().isoformat(timespec='seconds')
    (a.out / 'results-v12.json').write_text(json.dumps(ab.clean(R), indent=1) + '\n')
    write_md(R, a.out / 'results-v12.md')
    with (a.out / 'inputs-v12.log').open('w') as fh:
        fh.write(f'# analyze_bench_v12.py, label {a.label}, started {t0.isoformat(timespec="seconds")}, finished {R["finished"]}\ncommand: ' + ' '.join(sys.argv) + '\n')
        for p in list(a.runs) + [a.verdicts, a.clim_derived, a.clim_flags] + list(a.extra_inputs) + ([a.done_file] if a.done_file.is_file() else []):
            fh.write(f'{ab.sha256_file(p)}  {p}\n')
        for p in (Path(__file__).resolve(), HERE / 'analyze_bench_v1.py', HERE / 'analyze_wm_calibration.py', BENCH / 'protocol-v1.2-quantities.md', a.frozen):
            if Path(p).is_file():
                fh.write(f'{ab.sha256_file(p)}  {p}\n')
        fh.write(f"freeze check: {sum(c['match'] for c in freeze['checks'])}/{len(freeze['checks'])} match\n")
        for c in freeze['checks']:
            fh.write(f"  {'ok  ' if c['match'] else 'FAIL'} {c['expected']}  {c['path']}\n")
        fh.write(f'TEST gate: {json.dumps(gate)}\nprotocol exclusions: {json.dumps(exclusions)}\n')
        hw = hashlib.sha256()
        for w in windows:
            p = a.windows_dir / w / 'window.json'
            if p.is_file():
                hw.update(w.encode()); hw.update(p.read_bytes())
        fh.write(f'{hw.hexdigest()}  combined window.json of the {len(windows)} analysed windows ({a.windows_dir})\n')
        fh.write(f'numpy {np.__version__}, scipy {__import__("scipy").__version__}, python {sys.version.split()[0]}\n')
    print(json.dumps(dict(label=a.label, K=K, windows=len(windows), tested=len(R['tested']), descriptive=len(R['descriptive']), freeze_all_match=freeze['all_match'], out=str(a.out))))


if __name__ == '__main__':
    main()
