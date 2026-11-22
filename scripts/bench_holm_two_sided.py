#!/usr/bin/env python3
"""Confirmatory decision with a TWO-SIDED H1 for a new bench-v1 model (2026-11-03; Cosmos3-Edge amendment, usable for
any later model whose amendment makes H1 two-sided). Reads only the output of the frozen analysis, never the data.

The frozen analyze_bench_v1.py applies Holm to H1's ONE-sided slope p (protocol-v1, Nano). For a model whose amendment
makes H1 two-sided (owner decision for new models), this script recomputes the confirmatory decision from that
script's results.json:
  - H1 (two-sided): lead family, primary outcome x_4s. p1 = 2 (1 - Phi(|z|)) with z the Jolliffe-Primo slope z (exact
    null variance) = families_results.lead.stats.slope_p_two_sided_normal (recomputed here from slope_z as a check).
  - H2 (one-sided, unchanged): p2 = confirmatory.H2.p_mc (window-level permutation test, B = 100,000).
  - Holm across the two at family-wise alpha 0.05: the smaller p is compared with 0.025, the other with 0.05; the
    adjusted p values are those of the frozen holm() (step-down, monotone).
  - H1 is confirmed iff its Holm-adjusted p <= 0.05 AND the window-bootstrap 95 % CI of mean PIT - 0.5
    (families_results.lead.stats.mean_pit_ci95) excludes 0 on the side of z: z > 0 needs CI low > 0 (generated gaps
    too small: the real reading lies above the generated ones), z < 0 needs CI high < 0 (generated gaps too large).
    The direction is part of the result.
  - H2 is confirmed iff its Holm-adjusted p <= 0.05.
The frozen script's own one-sided H1/Holm entries in results.json are reported unchanged, labelled as not the decision.

  bench_holm_two_sided.py --results ANALYSIS_DIR/results.json --model NAME --out OUT.json [--md OUT.md] [--expect-k 16]
"""
from __future__ import annotations

import argparse, hashlib, json, math, sys
from datetime import datetime
from pathlib import Path

ALPHA = 0.05


def holm(p1, p2):
    """Identical to analyze_bench_v1.holm (two hypotheses)."""
    ps = [p1, p2]; order = sorted(range(2), key=lambda i: ps[i]); adj = [0.0, 0.0]
    adj[order[0]] = min(1.0, 2 * ps[order[0]]); adj[order[1]] = min(1.0, max(adj[order[0]], ps[order[1]]))
    return adj[0], adj[1]


def two_sided_p(z):
    return math.erfc(abs(z) / math.sqrt(2))          # = 2 (1 - Phi(|z|))


def decide(R):
    s = R['families_results']['lead']['stats']; h2 = R['confirmatory']['H2']
    z, ci = float(s['slope_z']), [float(v) for v in s['mean_pit_ci95']]
    p1, p2 = float(s['slope_p_two_sided_normal']), float(h2['p_mc'])
    check = abs(p1 - two_sided_p(z)) <= 1e-9 * max(1.0, p1)
    a1, a2 = holm(p1, p2)
    same_side = (z > 0 and ci[0] > 0) or (z < 0 and ci[1] < 0)
    direction = ('generated gaps too small (real above the generated readings; z > 0)' if z > 0 else
                 'generated gaps too large (real below the generated readings; z < 0)' if z < 0 else 'none (z = 0)')
    return dict(
        H1=dict(family='lead', outcome='x_4s', sidedness='two-sided', n=s['n'], slope_z=z, p_two_sided=p1, p_two_sided_recomputed_ok=check,
                p_holm=a1, mean_pit_minus_half=float(s['mean_pit_minus_half']), mean_pit_ci95=ci, ci_excludes_zero_on_side_of_z=same_side,
                direction=direction, confirmed=bool(a1 <= ALPHA and same_side)),
        H2=dict(sidedness='one-sided', T=h2.get('T'), null_mean=h2.get('null_mean'), boot_ci95=h2.get('boot_ci95'), n_windows=h2.get('n_windows'),
                p_one_sided=p2, p_holm=a2, confirmed=bool(a2 <= ALPHA)),
        holm=dict(alpha=ALPHA, smaller_p=('H1' if p1 <= p2 else 'H2'), thresholds={'smaller': ALPHA / 2, 'other': ALPHA}),
        frozen_script_one_sided_entries_not_the_decision={k: R['confirmatory'][k] for k in ('H1', 'H2') if k in R['confirmatory']})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', type=Path, required=True); ap.add_argument('--model', required=True)
    ap.add_argument('--out', type=Path, required=True); ap.add_argument('--md', type=Path); ap.add_argument('--expect-k', type=int)
    a = ap.parse_args()
    raw = a.results.read_bytes(); R = json.loads(raw)
    if 'H2' not in R.get('confirmatory', {}) or 'lead' not in R.get('families_results', {}) or not R['families_results']['lead'].get('stats', {}).get('n'):
        sys.exit('results.json has no H1/H2 data (lead stats or H2 missing)')
    if a.expect_k is not None and R.get('K') != a.expect_k:
        sys.exit(f"expected K = {a.expect_k}, results.json has K = {R.get('K')}")
    me = Path(__file__).resolve()
    D = dict(model=a.model, label=R.get('label'), K=R.get('K'), windows=R.get('windows'), results=str(a.results),
             results_sha256=hashlib.sha256(raw).hexdigest(), script=str(me), script_sha256=hashlib.sha256(me.read_bytes()).hexdigest(),
             computed=datetime.now().astimezone().isoformat(timespec='seconds'), **decide(R))
    a.out.write_text(json.dumps(D, indent=1) + '\n')
    h1, h2 = D['H1'], D['H2']
    md = [f"# Confirmatory decision, {a.model} ({D['label']}, K = {D['K']}, {D['windows']} windows): H1 two-sided, H2 one-sided, Holm", '',
          '| hypothesis | statistic | p | Holm-adjusted p | extra condition | decision |', '|---|---|---|---|---|---|',
          f"| H1 (two-sided): lead x_4s | JP slope z = {h1['slope_z']:+.2f}; mean PIT - 0.5 = {h1['mean_pit_minus_half']:+.3f} "
          f"[{h1['mean_pit_ci95'][0]:+.3f}, {h1['mean_pit_ci95'][1]:+.3f}] | {h1['p_two_sided']:.3g} | {h1['p_holm']:.3g} | "
          f"CI excludes 0 on the side of z: {'yes' if h1['ci_excludes_zero_on_side_of_z'] else 'no'} | {'confirmed' if h1['confirmed'] else 'not confirmed'} |",
          f"| H2 (one-sided): outside share | T = {h2['T']:.3f} (null {h2['null_mean']:.3f}) | {h2['p_one_sided']:.3g} | {h2['p_holm']:.3g} | | "
          f"{'confirmed' if h2['confirmed'] else 'not confirmed'} |", '',
          f"Direction of H1's statistic: {h1['direction']}.", '',
          f"results.json sha256 {D['results_sha256']}; this script sha256 {D['script_sha256']}. The frozen script's own one-sided H1/Holm "
          'entries are kept in the JSON under frozen_script_one_sided_entries_not_the_decision.']
    if a.md:
        a.md.write_text('\n'.join(md) + '\n')
    print(json.dumps(dict(model=a.model, H1_confirmed=h1['confirmed'], H1_p_holm=h1['p_holm'], H2_confirmed=h2['confirmed'], H2_p_holm=h2['p_holm'])))


if __name__ == '__main__':
    main()
