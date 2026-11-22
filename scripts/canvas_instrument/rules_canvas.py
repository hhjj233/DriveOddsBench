#!/usr/bin/env python3
"""bench-v1/instrument/RELIABILITY-RULE.md (fixed 2026-11-01 20:41) and the ego gate (ego-vo/VO-GATE.md) applied to a
canvas-instrument DEV validation (validate_canvas.py report) with the model's own raw source and real reference in place
of nano-raw / dev-recon; the rule's thresholds are unchanged (right target >= 0.90 on raw AND on the reference; raw vs
reference Spearman >= 0.80, binary agreement >= 0.85, skipped below 8 windows; binary agreement with the annotation
truth >= 0.85 on raw). Writes the scorability rules file read by measure_canvas.py export --rules.

  rules_canvas.py --validation VALIDATION.json --raw-source RAW --recon-run REF --gate GATE.json --vo-dir REL_VO_DIR --out RULES.json
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import ci_core as ci  # noqa: E402
import measure_bench_v2 as mb  # noqa: E402

MIN_WINDOWS = 8


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--validation', type=Path, required=True); ap.add_argument('--raw-source', required=True)
    ap.add_argument('--recon-run', default=None, help='the real reference run (omit while it does not exist: raw-only labels, flagged)')
    ap.add_argument('--gate', type=Path, default=None); ap.add_argument('--vo-dir', default=None, help='VO dir relative to the bench (clips/, clips-hist/)')
    ap.add_argument('--ego-history-ok', type=int, default=None, help='1 / 0 from the ego history check (omit: not judged)')
    ap.add_argument('--history-not-reliable', default='oncoming,rider', help='bench-v1 VALIDATION.md 4b (kept)')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    val = json.loads(args.validation.read_text())
    raw = val['sources'][args.raw_source]; rec = val['sources'].get(args.recon_run, {}) if args.recon_run else {}
    agr = val['agreement'].get(f'{args.raw_source} vs {args.recon_run}', {}) if args.recon_run else {}
    labels, not_rel, fam_not_rel = {}, {}, []
    for fam in mb.FAMILIES:
        for oname, kind in mb.OUTCOMES[fam]:
            r_raw = (raw.get(fam) or {}).get('outcomes', {}).get(oname, {})
            r_rec = (rec.get(fam) or {}).get('outcomes', {}).get(oname, {})
            a = agr.get(f'{fam}|{oname}')
            n_rec = r_rec.get('n', 0)
            why = []
            if 'agree' in r_raw:
                if r_raw['agree'] < 0.85:
                    why.append(f"binary agreement with gt {r_raw['agree']:.3f} < 0.85 ({args.raw_source})")
                if a is not None and a['n'] >= MIN_WINDOWS and a.get('agree', 1.0) < 0.85:
                    why.append(f"raw vs recon agreement {a['agree']:.3f} < 0.85")
            else:
                if r_raw.get('right_target', 0.0) < 0.90:
                    why.append(f"right target {r_raw.get('right_target', float('nan')):.3f} < 0.90 ({args.raw_source})")
                if n_rec >= MIN_WINDOWS and r_rec.get('right_target', 0.0) < 0.90:
                    why.append(f"right target {r_rec.get('right_target', float('nan')):.3f} < 0.90 ({args.recon_run})")
                if a is not None and a['n'] >= MIN_WINDOWS and (a.get('spearman') is None or not a.get('spearman') >= 0.80):
                    why.append(f"raw vs recon Spearman {a.get('spearman')} < 0.80")
            if not args.recon_run:
                note = 'PROVISIONAL: no real reference yet, judged on raw only'
            else:
                note = '' if n_rec >= MIN_WINDOWS else f'judged on {args.raw_source} only ({n_rec} {args.recon_run} window-families)'
            labels[f'{fam}|{oname}'] = dict(kind=kind, reliable=not why, reasons=why, note=note,
                                            right_target_raw=r_raw.get('right_target'), right_target_recon=r_rec.get('right_target'),
                                            agree_raw=r_raw.get('agree'), raw_vs_recon=a, n_raw=r_raw.get('n'), n_recon=n_rec)
            if why:
                not_rel.setdefault(fam, []).append(oname)
        prim = mb.OUTCOMES[fam][0][0]
        if not labels[f'{fam}|{prim}']['reliable']:
            fam_not_rel.append(fam)
    gate = json.loads(args.gate.read_text()) if args.gate else {}
    ego_ok = bool(gate.get('gate_passed'))
    ego = dict(clips_dir=f'{args.vo_dir}/clips' if args.vo_dir else None, hist_dir=f'{args.vo_dir}/clips-hist' if args.vo_dir else None,
               windows_dir='windows-dev', gate_passed=ego_ok, raw_passed=gate.get('raw_passed'), recon_passed=gate.get('recon_passed'),
               history_ok=bool(args.ego_history_ok) if args.ego_history_ok is not None else False,
               scorable={'ego_dx_0_4s': ego_ok, 'ego_dyaw_deg_0_4s': ego_ok,
                         'ego_dx_0_2s': ego_ok and bool(gate.get('secondary', {}).get('D_2.0s')),
                         'ego_dx_0_6s': ego_ok and bool(gate.get('secondary', {}).get('D_6.0s'))})
    rules = dict(rule_file='instrument/RELIABILITY-RULE.md', gate_file='instrument/ego-vo/VO-GATE.md', model=val.get('model'),
                 raw_source=args.raw_source, recon_run=args.recon_run, provisional=not args.recon_run,
                 family_not_reliable=fam_not_rel, outcome_not_reliable=not_rel,
                 history_not_reliable=[f for f in args.history_not_reliable.split(',') if f],
                 history_note='bench-v1 VALIDATION.md section 4b labels kept (oncoming, rider)', ego=ego, labels=labels,
                 script_sha256=ci.sha256_file(__file__), validation_sha256=ci.sha256_file(args.validation),
                 gate_sha256=ci.sha256_file(args.gate) if args.gate else None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rules, indent=1) + '\n')
    print(f'| family | outcome | kind | reliable | right target {args.raw_source} / {args.recon_run} | raw vs recon | reasons |')
    print('|---|---|---|---|---|---|---|')
    for k, v in labels.items():
        fam, on = k.split('|'); a = v['raw_vs_recon'] or {}
        rv = (f"{a['spearman']:.3f}" if a.get('spearman') is not None else (f"agree {a['agree']:.3f}" if a.get('agree') is not None else '-')) + (f" (n {a['n']})" if a else '')
        rt = lambda x: '-' if x is None else f'{x:.3f}'
        print(f"| {fam} | {on} | {v['kind']} | {'yes' if v['reliable'] else 'NO'} | {rt(v['right_target_raw'])} / {rt(v['right_target_recon'])} | {rv} | "
              f"{'; '.join(v['reasons'])}{(' ' + v['note']) if v['note'] else ''} |")
    print(json.dumps(dict(family_not_reliable=fam_not_rel, outcome_not_reliable=not_rel, ego=ego['scorable'])))


if __name__ == '__main__':
    main()
