#!/usr/bin/env python3
"""Rank (PIT) analysis of the world-model calibration study (protocol-v1.md, 2026-10-29).

For each window the real outcome y and the K generated outcomes Y_1..Y_K (from measure_lead_calibration.py) give the
rank r = #{Y_k < y} + a uniform draw over the tied positions (seed 2026), on {0..K}. Under the null (real and
generated futures exchangeable given the history) r is uniform and independent across windows. Reported:
  - chi-square on the K+1 rank bins (Monte Carlo p-value);
  - Jolliffe-Primo slope (one-sided: real ranks high = generated futures read smaller) and convexity components
    (two-sided; + = U shape = generated spread too narrow), as normal scores and with Monte Carlo p-values;
  - mean PIT - 0.5 with a window-bootstrap 95 % CI, and the shares of windows with the real reading above / below
    every sample (expected 1/(K+1) each);
  - fair CRPS (Ferro 2014) per window;
  - strata by anchor-time covariates only (protocol 5), with BH-FDR over the secondary tests;
  - figures: rank histogram with a 95 % pointwise band, PIT ECDF minus identity with a 95 % simultaneous band
    (Saeilynoja et al. 2022, by simulation).

  analyze_wm_calibration.py --measure M.json --real real-recon --gen RUN [--horizon 4] [--stat value|min] --output DIR
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np


def ranks_of(y, Y, rng):
    below, tied = int(np.sum(Y < y)), int(np.sum(Y == y))
    return below + int(rng.integers(0, tied + 1))


def jp_contrasts(K):
    j = np.arange(K + 1, dtype=float); c1 = j - j.mean(); c2 = c1 ** 2; c2 = c2 - c2.mean()
    c2 = c2 - c1 * (c2 @ c1) / (c1 @ c1)  # orthogonal to the linear contrast (already, by symmetry)
    return c1, c2


def jp_scores(counts, K):
    N = counts.sum(); p = 1.0 / (K + 1); c1, c2 = jp_contrasts(K)
    return float(c1 @ counts / np.sqrt(N * p * (c1 @ c1))), float(c2 @ counts / np.sqrt(N * p * (c2 @ c2)))


def chi2(counts, K):
    e = counts.sum() / (K + 1); return float(((counts - e) ** 2 / e).sum())


def fair_crps(Y, y):
    K = len(Y); return float(np.abs(Y - y).mean() - np.abs(Y[:, None] - Y[None, :]).sum() / (2 * K * (K - 1)))


def bh(pvalues):
    p = np.asarray(pvalues, float); order = np.argsort(p); q = np.empty_like(p); m = len(p); prev = 1.0
    for i in range(m - 1, -1, -1):
        prev = min(prev, p[order[i]] * m / (i + 1)); q[order[i]] = prev
    return q.tolist()


def simultaneous_band(N, grid, level, rng, sims=4000):
    """Pointwise level gamma such that the binomial band holds at all grid points with probability `level`."""
    from scipy.stats import binom
    u = np.sort(rng.random((sims, N)), axis=1); ecdf = np.stack([(u <= z).sum(1) for z in grid], 1)
    lo_g, hi_g = 1e-6, 1 - level
    for _ in range(40):
        g = (lo_g + hi_g) / 2; lo = binom.ppf(g / 2, N, grid); hi = binom.ppf(1 - g / 2, N, grid)
        inside = ((ecdf >= lo) & (ecdf <= hi)).all(1).mean()
        lo_g, hi_g = (g, hi_g) if inside > level else (lo_g, g)
    g = lo_g; return binom.ppf(g / 2, N, grid) / N, binom.ppf(1 - g / 2, N, grid) / N, g


def test_block(r, K, rng, mc):
    counts = np.bincount(r, minlength=K + 1).astype(float); N = len(r)
    sim = rng.integers(0, K + 1, size=(mc, N)); sim_counts = np.stack([np.bincount(s, minlength=K + 1) for s in sim]).astype(float)
    c1, c2 = jp_contrasts(K); p = 1.0 / (K + 1)
    s1 = sim_counts @ c1 / np.sqrt(N * p * (c1 @ c1)); s2 = sim_counts @ c2 / np.sqrt(N * p * (c2 @ c2))
    x2s = (((sim_counts - N * p) ** 2) / (N * p)).sum(1)
    z1, z2 = jp_scores(counts, K); x2 = chi2(counts, K)
    return dict(n=N, counts=counts.astype(int).tolist(), chi2=x2, chi2_p_mc=float((x2s >= x2).mean()), slope_z=z1, slope_p_one_sided_mc=float((s1 >= z1).mean()),
                convexity_z=z2, convexity_p_two_sided_mc=float((np.abs(s2) >= abs(z2)).mean()))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--measure', required=True, type=Path); parser.add_argument('--real', default='real-recon'); parser.add_argument('--gen', required=True)
    parser.add_argument('--horizon', default='4.0'); parser.add_argument('--stat', choices=('value', 'min'), default='value')
    parser.add_argument('--windows-info', type=Path, help='test-windows.json (adds the val/train source stratum)')
    parser.add_argument('--output', required=True, type=Path); parser.add_argument('--mc', type=int, default=20000); parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()
    m = json.loads(args.measure.read_text()); h = str(float(args.horizon)); rng = np.random.default_rng(args.seed)
    source_of = {w['name']: w['source'] for w in json.loads(args.windows_info.read_text())['test']} if args.windows_info else {}
    rows, skipped = [], []
    for name in sorted(m['windows']):
        clips = m['windows'][name]['clips']; real = clips.get(args.real)
        gen = [c for k, c in sorted(clips.items()) if k.startswith(args.gen + '-s')]
        if real is None or not gen:
            skipped.append((name, 'missing clips')); continue
        y = real['outcomes'][h][args.stat]; Y = np.array([c['outcomes'][h][args.stat] for c in gen if c['outcomes'][h][args.stat] is not None], float)
        if y is None or len(Y) < 2:
            skipped.append((name, 'no reading')); continue
        rows.append(dict(name=name, y=y, Y=Y, K=len(Y), cov=m['windows'][name]['covariates'], source=source_of.get(name), real_state=real['outcomes'][h]['state'],
                         gen_states=[c['outcomes'][h]['state'] for c in gen]))
    Ks = sorted({r['K'] for r in rows})
    if len(Ks) != 1:
        raise SystemExit(f'unequal K across windows: {Ks}')
    K = Ks[0]
    for r in rows:
        r['rank'] = ranks_of(r['y'], r['Y'], rng); r['pit'] = (r['rank'] + rng.random()) / (K + 1); r['crps'] = fair_crps(r['Y'], r['y'])
    rk = np.array([r['rank'] for r in rows]); pit = np.array([r['pit'] for r in rows]); N = len(rows)
    main_block = test_block(rk, K, rng, args.mc)
    boot = np.array([pit[rng.integers(0, N, N)].mean() for _ in range(10000)]) - 0.5
    states = lambda key: {s: sum(1 for r in rows for x in ([r['real_state']] if key == 'real' else r['gen_states']) if x == s) for s in ('followed', 'carried', 'none')}
    summary = dict(measure=str(args.measure), real=args.real, gen=args.gen, horizon_s=float(h), stat=args.stat, K=K, windows=N, skipped=skipped, **main_block,
                   mean_pit_minus_half=float(pit.mean() - 0.5), mean_pit_ci95=[float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
                   share_real_above_all=float((rk == K).mean()), share_real_below_all=float((rk == 0).mean()), expected_share=1 / (K + 1),
                   mean_fair_crps_m=float(np.mean([r['crps'] for r in rows])), median_real_m=float(np.median([r['y'] for r in rows])),
                   median_generated_m=float(np.median(np.concatenate([r['Y'] for r in rows]))), states_real=states('real'), states_generated=states('gen'))
    # strata by anchor-time covariates only
    cov = lambda key: np.array([np.nan if r['cov'].get(key) is None else r['cov'][key] for r in rows], float)
    speed, accel, gap, closing = cov('ego_speed_mps'), cov('ego_accel_mps2'), cov('rear_face_gap_m'), cov('lead_closing_speed_mps')
    t1, t2 = np.nanquantile(speed, [1 / 3, 2 / 3])
    strata = {f'ego speed < {t1:.1f} m/s': speed < t1, f'ego speed {t1:.1f}-{t2:.1f} m/s': (speed >= t1) & (speed < t2), f'ego speed >= {t2:.1f} m/s': speed >= t2,
              'ego accel < -0.5': accel < -0.5, 'ego accel -0.5..0.5': (accel >= -0.5) & (accel <= 0.5), 'ego accel > 0.5': accel > 0.5,
              'gap 5-12 m': gap < 12, 'gap 12-18 m': (gap >= 12) & (gap < 18), 'gap 18-25 m': gap >= 18,
              'closing <= 0': closing <= 0, 'closing > 0': closing > 0,
              'passenger car': np.array([r['cov']['lead_category'] == 'REGULAR_VEHICLE' for r in rows]), 'larger vehicle': np.array([r['cov']['lead_category'] != 'REGULAR_VEHICLE' for r in rows])}
    if source_of:
        strata |= {'source val': np.array([r['source'] == 'windows-test-val' for r in rows]), 'source train': np.array([r['source'] == 'windows-test-train' for r in rows])}
    out_strata = {}
    for label, mask in strata.items():
        if mask.sum() >= 10:
            b = test_block(rk[mask], K, rng, 4000); out_strata[label] = dict(n=int(mask.sum()), mean_pit_minus_half=float(pit[mask].mean() - 0.5), slope_z=b['slope_z'],
                                                                             slope_p_one_sided_mc=b['slope_p_one_sided_mc'], convexity_z=b['convexity_z'], chi2_p_mc=b['chi2_p_mc'])
    for label, q in zip(out_strata, bh([v['slope_p_one_sided_mc'] for v in out_strata.values()])):
        out_strata[label]['slope_q_bh_within_strata'] = q
    summary['strata'] = out_strata
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    (args.output / 'windows.json').write_text(json.dumps([dict(name=r['name'], real=r['y'], generated=r['Y'].tolist(), rank=r['rank'], pit=r['pit'], fair_crps=r['crps'],
                                                               real_state=r['real_state'], gen_states=r['gen_states'], covariates=r['cov'], source=r['source']) for r in rows], indent=1) + '\n')
    # figures
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from scipy.stats import binom
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    counts = np.bincount(rk, minlength=K + 1); lo, hi = binom.ppf(0.025, N, 1 / (K + 1)), binom.ppf(0.975, N, 1 / (K + 1))
    ax[0].bar(np.arange(K + 1), counts, color='#4c72b0'); ax[0].axhline(N / (K + 1), color='k', lw=1); ax[0].axhspan(lo, hi, color='grey', alpha=0.2)
    ax[0].set_xlabel(f'rank of the real reading among {K} generated (0 = below all)'); ax[0].set_ylabel('windows'); ax[0].set_title(f'{args.gen}, t = {h} s, {args.stat}, N = {N}')
    grid = np.linspace(0.01, 0.99, 99); blo, bhi, g = simultaneous_band(N, grid, 0.95, rng)
    ecdf = np.array([(pit <= z).mean() for z in grid])
    ax[1].fill_between(grid, blo - grid, bhi - grid, color='grey', alpha=0.25); ax[1].plot(grid, ecdf - grid, color='#c44e52'); ax[1].axhline(0, color='k', lw=1)
    ax[1].set_xlabel('PIT'); ax[1].set_ylabel('ECDF - PIT'); ax[1].set_title('ECDF difference, 95 % simultaneous band')
    fig.tight_layout(); fig.savefig(args.output / 'pit.png', dpi=130)
    print(json.dumps({k: summary[k] for k in ('windows', 'K', 'chi2', 'chi2_p_mc', 'slope_z', 'slope_p_one_sided_mc', 'convexity_z', 'convexity_p_two_sided_mc', 'mean_pit_minus_half',
                                               'mean_pit_ci95', 'share_real_above_all', 'share_real_below_all', 'median_real_m', 'median_generated_m', 'states_real', 'states_generated')}))


if __name__ == '__main__':
    main()
