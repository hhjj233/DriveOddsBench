#!/usr/bin/env python3
"""Markdown summary tables of a canvas-instrument DEV validation (validate_canvas.py report output, rules_canvas.py
labels, ego_vo_band.py gate). Read only; prints to stdout.

  tables_canvas.py --validation VALIDATION.json [--rules RULES.json] [--gate GATE.json] --raw RAW [--ref REF] [--compare SRC ...]
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import measure_bench_v2 as mb  # noqa: E402


def f(v, n=3):
    return '-' if v is None else (f'{v:.{n}f}' if isinstance(v, float) else str(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--validation', type=Path, required=True); ap.add_argument('--rules', type=Path, default=None); ap.add_argument('--gate', type=Path, default=None)
    ap.add_argument('--raw', required=True); ap.add_argument('--ref', default=None); ap.add_argument('--compare', nargs='*', default=[])
    a = ap.parse_args()
    val = json.loads(a.validation.read_text()); rules = json.loads(a.rules.read_text()) if a.rules else None
    srcs = [a.raw] + ([a.ref] if a.ref else []) + a.compare
    agr = val['agreement'].get(f'{a.raw} vs {a.ref}', {}) if a.ref else {}
    print(f"| family | primary | scorable {' / '.join(srcs)} | right target {' / '.join(srcs)} | error median / p90 ({a.raw}) | bias ({a.raw}) | raw vs ref | label |")
    print('|---|---|---|---|---|---|---|---|')
    for fam in mb.FAMILIES:
        prim = mb.OUTCOMES[fam][0][0]
        S = [(val['sources'].get(s) or {}).get(fam) for s in srcs]
        if S[0] is None:
            continue
        sc = ' / '.join('-' if s is None else f"{100 * s['scorable']:.0f}% ({s['clips']})" for s in S)
        o = [None if s is None else s['outcomes'].get(prim, {}) for s in S]
        rt = ' / '.join('-' if x is None or 'right_target' not in x else f"{x['right_target']:.3f}" for x in o)
        o0 = o[0] or {}
        err = f"{f(o0.get('err_read_median_abs'), 2)} / {f(o0.get('err_read_p90_abs'), 2)}" if 'err_read_median_abs' in o0 else (f"agree {f(o0.get('agree'))}" if 'agree' in o0 else '-')
        bias = f(o0.get('err_read_bias'), 2) if 'err_read_bias' in o0 else '-'
        ag = agr.get(f'{fam}|{prim}') or {}
        rv = (f"{ag['spearman']:.3f}" if ag.get('spearman') is not None else (f"agree {ag['agree']:.3f}" if ag.get('agree') is not None else '-')) + (f" (n {ag['n']})" if ag else '')
        lab = '-'
        if rules:
            L = rules['labels'].get(f'{fam}|{prim}', {})
            lab = ('reliable' if L.get('reliable') else 'NOT reliable') + (' (family)' if fam in rules['family_not_reliable'] else '')
            bad = [o_ for o_ in rules['outcome_not_reliable'].get(fam, []) if o_ != prim]
            if bad and fam not in rules['family_not_reliable']:
                lab += f"; not reliable: {', '.join(bad)}"
        print(f'| {fam} | {prim} | {sc} | {rt} | {err} | {bias} | {rv} | {lab} |')
    if a.gate:
        g = json.loads(a.gate.read_text())
        print('\n| odometry | n | median err / tol (<= 1) | median heading err deg (<= 3) | Spearman D4 (>= 0.90) | D4 abs err median / p90 m | accepted share | pass |')
        print('|---|---|---|---|---|---|---|---|')
        for k, r in g['results'].items():
            if r.get('n', 0) == 0:
                continue
            ok = r.get('criterion1_pass') and r.get('criterion2_pass') and r.get('criterion3_pass')
            print(f"| {k} | {r['n']} | {f(r['criterion1_median_err_over_tol'])} | {f(r['criterion2_median_abs_heading_err_deg'], 2)} | {f(r['criterion3_spearman'])} | "
                  f"{f(r['D_abs_err_median_m'], 2)} / {f(r['D_abs_err_p90_m'], 2)} | {f(r['accepted_share_median'], 2)} | {'yes' if ok else 'no'} |")
        print(f"\ngate passed: {g.get('gate_passed')} (raw {g.get('raw_passed')}, reference {g.get('recon_passed')}); secondary {g.get('secondary')}; raw vs reference D4 {g.get('raw_vs_recon_D4')}")


if __name__ == '__main__':
    main()
