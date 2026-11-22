#!/usr/bin/env python3
"""DEV action-scale check of Vista's T2 track (CPU): does the ego motion the generated video shows follow the given AV2
poses (units, signs, timing)? It reads band-odometry records (canvas instrument ego_vo_band.py, the frozen VO v2 port on
the 512 x 288 band canvas: planar D(T), Psi(T) at T = 1, 2, 4, 6 s, and the 0.4 s group trajectory) and compares them with
the commanded ego motion (actions_t2.json given_ego_t0_frame: the AV2 poses at the frame image times, rear axle, ego frame
of the t0 frame; gt.json is read as a cross-check).

The rule is the Cosmos3-Nano T2 rule (bench-v1/nano-t2/checks/ACTION-SCALE-RULE.md, sha256 9e23b5b5...3894), applied
unchanged to Vista with its own odometry (the band VO; Nano's is the Nano-frame VO), on the same DEV windows
(nano-t2/names-t2-scale-all.txt, 33 windows) and samples (s0, s1). Classes (by the commanded motion): moving
D_cmd(4) >= 5 m; turning |Psi_cmd(4)| >= 10 deg or |Psi_cmd(6)| >= 10 deg; stationary D_cmd(6) < 0.5 m; start-from-rest
D_cmd(1) < 0.5 m and D_cmd(6) >= 3 m.
  S1 scale    per moving window r4 = mean over its clips of D_gen(4) / D_cmd(4); median over moving windows in [0.85, 1.15]
              (the same ratio on the reference clips, the odometry floor, is reported next to it)
  S2 ranking  Spearman(D_gen(4), D_cmd(4)) over all generated clips >= 0.80
  S3 sign     turning clips (|Psi_cmd| >= 10 deg at 4 s, else 6 s): sign(Psi_gen) = sign(Psi_cmd) in >= 75 %
  S4 still    stationary windows' clips: |D_gen(6)| < 1.5 m in >= 75 %
  reported    D_gen(T) / D_cmd(T) at 1, 2, 4, 6 s; per-group advance correlation and best lag (-2..+2 groups); start-from-rest
              windows individually; speed dependence of r4
Decision (as the Nano rule): S1-S4 pass -> the mapping stands (KT = 1). S1 fails -> KT* = 1 / median r4 rounded to 0.05, one
confirmation run on the moving windows (s0, s1) at KT*, adopted if its median r4 lies in [0.9, 1.1] and S2-S4 still
pass; otherwise the owner decides (no further tuning on DEV). S3 failing alone -> the axis / sign convention is re-checked
(a sign error is a bug); S2 / S4 failing -> reported, the owner decides.
Vista-specific diagnostics (reported, not criteria): (a) the time-aligned trajectory variant (waypoints at frames
22n + 5i instead of 22n + 6i, i.e. 5/6 of the commanded motion per frame): a model that follows its trajectory reads about
0.83 x the frame-index ratio, one that ignores it reads the same; (b) the action-free H0 clips of the same windows and
seeds: the model's own ego motion from the anchor alone.

  action_scale_check.py --bench BENCH_V1 --inputs INPUT_ROOT --names FILE --source LABEL=VO_DIR:SOURCE:SAMPLES [...]
                        --out FILE.json [--md FILE.md]
LABEL: t2 (the clips the rule judges), ref (reference clips), t2ta, h0, raw, or any tag; SAMPLES: comma-separated record
sample keys (0,1 for s0, s1; real for a reference; raw for the raw clip).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

HORIZONS = ('1.0', '2.0', '4.0', '6.0')
RULE = dict(moving_D4_m=5.0, turning_deg=10.0, stationary_D6_m=0.5, rest_D1_m=0.5, rest_D6_m=3.0, S1=(0.85, 1.15), S2=0.80, S3=0.75, S4=0.75,
            S4_D6_m=1.5, confirm=(0.9, 1.1))


def commanded(inputs: Path, w: str) -> dict:
    a = json.loads((inputs / 'windows' / 'dev' / w / 'actions_t2.json').read_text())
    g = {e['k']: e for e in a['given_ego_t0_frame']}
    hz = {h: dict(D_m=g[int(round(float(h) * 10))]['x_m'], Psi_deg=g[int(round(float(h) * 10))]['yaw_deg']) for h in HORIZONS}
    pts = np.array([[g[k]['x_m'], g[k]['y_m']] for k in range(0, 61, 4)])
    return dict(horizons=hz, group_advance=np.linalg.norm(np.diff(pts, axis=0), axis=1))


def gt_of(bench: Path, w: str) -> dict:
    gt = json.loads((bench / 'windows-dev' / w / 'gt.json').read_text())
    e = {x['k']: x for x in gt['ego']}
    return {h: dict(D_m=e[int(round(float(h) * 10))]['xyz_m'][0], Psi_deg=math.degrees(e[int(round(float(h) * 10))]['yaw_rad'])) for h in HORIZONS}


def record(vo_dir: Path, source: str, w: str, sample: str):
    f = vo_dir / 'clips' / source / f'{w}__{sample}.json'
    if not f.is_file():
        return None
    r = json.loads(f.read_text())
    tr = r['planar']['trajectory']
    pts = np.array([[p['x'], p['y']] for p in tr])
    return dict(horizons={h: r['planar']['horizons'][h] for h in HORIZONS}, group_advance=np.linalg.norm(np.diff(pts, axis=0), axis=1),
                accepted_share=r['quality']['accepted_share'], file=str(f))


def classes(c: dict) -> dict:
    h = c['horizons']
    return dict(moving=h['4.0']['D_m'] >= RULE['moving_D4_m'],
                turning=abs(h['4.0']['Psi_deg']) >= RULE['turning_deg'] or abs(h['6.0']['Psi_deg']) >= RULE['turning_deg'],
                stationary=h['6.0']['D_m'] < RULE['stationary_D6_m'],
                start_from_rest=h['1.0']['D_m'] < RULE['rest_D1_m'] and h['6.0']['D_m'] >= RULE['rest_D6_m'])


def _corr(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and np.std(a) > 0 and np.std(b) > 0 else float('-inf')


def evaluate(clips: list, cmd: dict) -> dict:
    """clips: [(window, sample, record)]. The rule's statistics for one clip set."""
    from scipy.stats import spearmanr
    out = dict(n_clips=len(clips), n_windows=len({w for w, _, _ in clips}))
    if not clips:
        return out
    by_w = {}
    for w, s, r in clips:
        by_w.setdefault(w, []).append(r)
    mv = [w for w in by_w if classes(cmd[w])['moving']]
    r4 = {w: float(np.mean([r['horizons']['4.0']['D_m'] / cmd[w]['horizons']['4.0']['D_m'] for r in by_w[w]])) for w in mv}
    out['S1_median_r4'] = round(float(np.median(list(r4.values()))), 4) if r4 else None
    out['S1_n_moving'] = len(r4)
    out['r4_per_window'] = {w: round(v, 3) for w, v in sorted(r4.items())}
    rb = {}
    for h in HORIZONS:
        vals = [np.mean([r['horizons'][h]['D_m'] / cmd[w]['horizons'][h]['D_m'] for r in by_w[w]]) for w in mv if cmd[w]['horizons'][h]['D_m'] > 0.5]
        rb[h] = round(float(np.median(vals)), 4) if vals else None
    out['ratio_by_horizon'] = rb
    dg = [r['horizons']['4.0']['D_m'] for w, _, r in clips]
    dc = [cmd[w]['horizons']['4.0']['D_m'] for w, _, r in clips]
    out['S2_spearman_D4'] = round(float(spearmanr(dg, dc).correlation), 4) if len(clips) > 2 else None
    turn = []
    for w, s, r in clips:
        if not classes(cmd[w])['turning']:
            continue
        h = '4.0' if abs(cmd[w]['horizons']['4.0']['Psi_deg']) >= RULE['turning_deg'] else '6.0'
        turn.append(dict(clip=f'{w} s{s}', horizon=h, psi_cmd=round(cmd[w]['horizons'][h]['Psi_deg'], 1), psi_gen=round(r['horizons'][h]['Psi_deg'], 1),
                         agree=bool(np.sign(r['horizons'][h]['Psi_deg']) == np.sign(cmd[w]['horizons'][h]['Psi_deg']))))
    out['S3_sign_share'] = round(sum(t['agree'] for t in turn) / len(turn), 4) if turn else None
    out['S3_n'] = len(turn)
    out['turning_clips'] = turn
    st = [bool(abs(r['horizons']['6.0']['D_m']) < RULE['S4_D6_m']) for w, _, r in clips if classes(cmd[w])['stationary']]
    out['S4_still_share'] = round(sum(st) / len(st), 4) if st else None
    out['S4_n'] = len(st)
    corr, lags = [], {}
    for w, s, r in clips:
        if not classes(cmd[w])['moving']:
            continue
        a, b = r['group_advance'], cmd[w]['group_advance']
        c0 = _corr(a, b)
        if math.isfinite(c0):
            corr.append(c0)
        best = max(range(-2, 3), key=lambda L: _corr(a[max(0, L):len(a) + min(0, L)], b[max(0, -L):len(b) + min(0, -L)]))
        lags[str(best)] = lags.get(str(best), 0) + 1
    out['timing'] = dict(group_advance_corr_median=round(float(np.median(corr)), 3) if corr else None, best_lag_counts=lags)
    out['start_from_rest'] = [dict(clip=f'{w} s{s}', D6_cmd=round(cmd[w]['horizons']['6.0']['D_m'], 2), D6_gen=round(r['horizons']['6.0']['D_m'], 2))
                              for w, s, r in clips if classes(cmd[w])['start_from_rest']]
    if len(r4) > 2:
        speed = [cmd[w]['horizons']['4.0']['D_m'] / 4.0 for w in r4]
        out['speed_dependence_spearman'] = round(float(spearmanr(list(r4.values()), speed).correlation), 3)
    out['median_abs_err_D4_m'] = round(float(np.median([abs(r['horizons']['4.0']['D_m'] - cmd[w]['horizons']['4.0']['D_m']) for w, _, r in clips])), 3)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bench', type=Path, required=True)
    ap.add_argument('--inputs', type=Path, required=True)
    ap.add_argument('--names', type=Path, required=True)
    ap.add_argument('--source', action='append', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--md', type=Path, default=None)
    args = ap.parse_args()
    names = [l.strip() for l in args.names.read_text().splitlines() if l.strip()]
    cmd = {w: commanded(args.inputs, w) for w in names}
    gt = {w: gt_of(args.bench, w) for w in names}
    res = dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), rule=RULE, windows=len(names),
               names_sha256=hashlib.sha256(args.names.read_bytes()).hexdigest(),
               commanded_vs_gt_max_abs=dict(D_m=round(max(abs(cmd[w]['horizons'][h]['D_m'] - gt[w][h]['D_m']) for w in names for h in HORIZONS), 3),
                                            Psi_deg=round(max(abs(cmd[w]['horizons'][h]['Psi_deg'] - gt[w][h]['Psi_deg']) for w in names for h in HORIZONS), 3)),
               class_counts={k: int(sum(classes(cmd[w])[k] for w in names)) for k in ('moving', 'turning', 'stationary', 'start_from_rest')}, sets={})
    for spec in args.source:
        label, _, rest = spec.partition('=')
        vo_dir, source, samples = rest.rsplit(':', 2)
        clips = []
        for w in names:
            for s in samples.split(','):
                r = record(Path(vo_dir), source, w, s)
                if r is not None:
                    clips.append((w, s, r))
        res['sets'][label] = dict(vo_dir=vo_dir, source=source, samples=samples, **evaluate(clips, cmd))
    t2 = res['sets'].get('t2')
    if t2 and t2.get('n_clips'):
        v = dict(S1=t2['S1_median_r4'] is not None and RULE['S1'][0] <= t2['S1_median_r4'] <= RULE['S1'][1],
                 S2=t2['S2_spearman_D4'] is not None and t2['S2_spearman_D4'] >= RULE['S2'],
                 S3=t2['S3_sign_share'] is None or t2['S3_sign_share'] >= RULE['S3'],
                 S4=t2['S4_still_share'] is None or t2['S4_still_share'] >= RULE['S4'])
        v['all_pass'] = all(v.values())
        if not v['S1'] and t2['S1_median_r4']:
            v['KT_star'] = round(round((1.0 / t2['S1_median_r4']) / 0.05) * 0.05, 2)
        res['verdict'] = v
        ta = res['sets'].get('t2ta')
        if ta and ta.get('S1_median_r4') and t2['S1_median_r4']:
            res['dose_response'] = dict(ratio_time_aligned_over_frame_index=round(ta['S1_median_r4'] / t2['S1_median_r4'], 3),
                                        expected_if_followed=round(5 / 6, 3), expected_if_ignored=1.0)
    args.out.write_text(json.dumps(res, indent=1) + '\n')
    lines = ['# Vista T2 action-scale check (DEV; the Nano T2 rule)', '',
             f'windows {len(names)} (names sha256 {res["names_sha256"][:16]}); classes {res["class_counts"]}; '
             f'commanded vs gt.json max |dD| {res["commanded_vs_gt_max_abs"]["D_m"]} m', '',
             '| set | clips | S1 median r4 | S2 Spearman D4 | S3 sign | S4 still | D ratio 1 / 2 / 4 / 6 s | median abs err D4 (m) | timing corr / lags |',
             '|---|---|---|---|---|---|---|---|---|']
    for label, s in res['sets'].items():
        if not s.get('n_clips'):
            lines.append(f'| {label} | 0 | | | | | | | |')
            continue
        rb = s['ratio_by_horizon']
        lines.append(f'| {label} | {s["n_clips"]} | {s["S1_median_r4"]} (n {s["S1_n_moving"]}) | {s["S2_spearman_D4"]} | {s["S3_sign_share"]} (n {s["S3_n"]}) | '
                     f'{s["S4_still_share"]} (n {s["S4_n"]}) | {rb["1.0"]} / {rb["2.0"]} / {rb["4.0"]} / {rb["6.0"]} | {s["median_abs_err_D4_m"]} | '
                     f'{s["timing"]["group_advance_corr_median"]} / {s["timing"]["best_lag_counts"]} |')
    if 'verdict' in res:
        lines += ['', 'verdict (t2): ' + json.dumps(res['verdict'])]
    if 'dose_response' in res:
        lines += ['dose-response (time-aligned / frame-index median r4): ' + json.dumps(res['dose_response'])]
    if args.md:
        args.md.write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
