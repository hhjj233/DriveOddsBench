#!/usr/bin/env python3
"""Apply bench-v1/instrument/RELIABILITY-RULE.md (fixed 2026-11-01 20:41) and the ego gate (ego-vo/VO-GATE.md) to the
DEV validation, and write the scorability rules file read by `measure_bench_v2.py export --rules`.

  rules_bench_v2.py --validation VALIDATION.json --gate GATE.json --recon-run dev-recon --out RULES.json
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import measure_bench_v2 as mb  # noqa: E402

MIN_WINDOWS = 8


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--validation', type=Path, required=True); ap.add_argument('--gate', type=Path, required=True)
    ap.add_argument('--recon-run', default='dev-recon'); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--ego-history-ok', type=int, default=1, help='1 if the ego 2 s history odometry passed its DEV check (VALIDATION.md 5.1)')
    ap.add_argument('--history-not-reliable', default='oncoming,rider',
                    help='families whose 2 s history line is labelled not reliable (VALIDATION.md 4b; a post hoc DEV judgement)')
    args = ap.parse_args()
    val = json.loads(args.validation.read_text()); gate = json.loads(args.gate.read_text())
    raw, rec = val['sources']['nano-raw'], val['sources'].get(args.recon_run, {})
    agr = val['agreement'].get(f'nano-raw vs {args.recon_run}', {})
    labels, not_rel, fam_not_rel = {}, {}, []
    for fam in mb.FAMILIES:
        for oname, kind in mb.OUTCOMES[fam]:
            r_raw = (raw.get(fam) or {}).get('outcomes', {}).get(oname, {})
            r_rec = (rec.get(fam) or {}).get('outcomes', {}).get(oname, {})
            a = agr.get(f'{fam}|{oname}')
            n_rec = r_rec.get('n', 0)
            why = []
            if 'agree' in r_raw:                                    # binary outcome
                if r_raw['agree'] < 0.85:
                    why.append(f"binary agreement with gt {r_raw['agree']:.3f} < 0.85 (nano-raw)")
                if a is not None and a['n'] >= MIN_WINDOWS and a.get('agree', 1.0) < 0.85:
                    why.append(f"raw vs recon agreement {a['agree']:.3f} < 0.85")
            else:
                if r_raw.get('right_target', 0.0) < 0.90:
                    why.append(f"right target {r_raw.get('right_target', float('nan')):.3f} < 0.90 (nano-raw)")
                if n_rec >= MIN_WINDOWS and r_rec.get('right_target', 0.0) < 0.90:
                    why.append(f"right target {r_rec.get('right_target', float('nan')):.3f} < 0.90 ({args.recon_run})")
                if a is not None and a['n'] >= MIN_WINDOWS and (a.get('spearman') is None or not a.get('spearman') >= 0.80):
                    why.append(f"raw vs recon Spearman {a.get('spearman')} < 0.80")
            note = '' if n_rec >= MIN_WINDOWS else f'judged on nano-raw only ({n_rec} {args.recon_run} window-families)'
            labels[f'{fam}|{oname}'] = dict(kind=kind, reliable=not why, reasons=why, note=note,
                                            right_target_raw=r_raw.get('right_target'), right_target_recon=r_rec.get('right_target'),
                                            agree_raw=r_raw.get('agree'), raw_vs_recon=a)
            if why:
                not_rel.setdefault(fam, []).append(oname)
        prim = mb.OUTCOMES[fam][0][0]
        if not labels[f'{fam}|{prim}']['reliable']:
            fam_not_rel.append(fam)
    ego_ok = bool(gate.get('gate_passed'))
    ego = dict(clips_dir='instrument/ego-vo/clips', hist_dir='instrument/ego-vo/clips-hist', windows_dir='windows-dev', gate_passed=ego_ok,
               history_ok=bool(args.ego_history_ok),
               scorable={'ego_dx_0_4s': ego_ok, 'ego_dyaw_deg_0_4s': ego_ok,
                         'ego_dx_0_2s': ego_ok and bool(gate.get('secondary', {}).get('D_2.0s')),
                         'ego_dx_0_6s': ego_ok and bool(gate.get('secondary', {}).get('D_6.0s'))})
    hist_nr = [f for f in args.history_not_reliable.split(',') if f]
    rules = dict(rule_file='instrument/RELIABILITY-RULE.md', gate_file='instrument/ego-vo/VO-GATE.md',
                 family_not_reliable=fam_not_rel, outcome_not_reliable=not_rel, history_not_reliable=hist_nr,
                 history_note='VALIDATION.md section 4b; set after seeing the DEV history numbers (no pre-registered rule)',
                 ego=ego, labels=labels)
    args.out.write_text(json.dumps(rules, indent=1) + '\n')
    print('| family | outcome | kind | reliable | right target nano-raw / recon | raw vs recon | reasons |')
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
