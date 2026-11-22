#!/usr/bin/env python3
"""DEV validation of the bench-v1 instrument v2 (measure_bench_v2.py) against AV2 ground truth (gt.json), 2026-11-01.

DEV only: gt.json exists for DEV windows only, and this script refuses any name outside names-dev.txt.

Ground truth per family target and tick (t0 = frame 20, t = 0..6 s), in the AV2 ego frame at the clip's time t:
  - the annotated cuboid (gt.json xyz_t0_m, yaw_t0_rad; size from window.json) moved into the ego frame at t with the
    gt ego pose (matrix_3x4 maps ego(t) to ego(t0));
  - x_near = forward coordinate of the nearest of its four bottom corners (what a box-bottom flat-road reading aims at:
    the nearest ground contact; for a lead this is the rear face), x_c / y_c = its centre;
  - the projected box (gt.json box_canvas, cuboid corners) for the right-target test (IoU >= 0.3) and the gt exit (the
    first tick whose box touches the instrument's edge limits or leaves the band).
Readings are compared as x - x_near and y - y_c.

  validate_bench_v2.py geometry --bench B --names-file F --config C.json --out OUT.json
  validate_bench_v2.py report   --bench B --names-file F --measure DIR --sources canvas-raw nano-raw RUN... --out DIR
"""
from __future__ import annotations

import argparse, json, math, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import measure_bench_v2 as mb  # noqa: E402

HORIZONS = (1.0, 2.0, 4.0, 6.0)


def load_names(bench: Path, names_file: Path):
    names = [l.strip() for l in names_file.read_text().splitlines() if l.strip()]
    dev = set(l.strip() for l in (bench / 'names-dev.txt').read_text().splitlines() if l.strip())
    if not set(names) <= dev:
        sys.exit('refusing: names outside names-dev.txt (validation is DEV only)')
    return names


def gt_series(window, gt, family, cfg=None, ks=range(0, 61)):
    """Per tick k in ks (default t = 0..6 s, 61 entries): ground truth of the family's target in the ego frame at t
    (or None). Negative k reach into the history (gt.json covers k = -20..60)."""
    tgt = window['targets'][family]; L, W, H = tgt['length_m'], tgt['width_m'], tgt['height_m']
    ego = {e['k']: e for e in gt['ego']}
    per = {r['k']: r for r in gt['targets'][family]['per_tick']}
    cfg = cfg or {'edge_x': [4.0, 508.0], 'edge_bottom': 284.0}
    out = []
    for k in ks:
        r = per.get(k); e = ego.get(k)
        if r is None or e is None or not r.get('annotated') or r.get('xyz_t0_m') is None:
            out.append(None); continue
        M = np.asarray(e['matrix_3x4'], float); R, t = M[:, :3], M[:, 3]
        c0 = np.asarray(r['xyz_t0_m'], float); yaw = float(r['yaw_t0_rad'])
        ax = np.array([math.cos(yaw), math.sin(yaw), 0.0]); ay = np.array([-math.sin(yaw), math.cos(yaw), 0.0]); az = np.array([0.0, 0.0, 1.0])
        corners0 = np.array([c0 + a * L / 2 * ax + b * W / 2 * ay - H / 2 * az for a in (-1, 1) for b in (-1, 1)])
        corners = (corners0 - t) @ R            # R^T (p - t) for row vectors
        c = (c0 - t) @ R
        box = r.get('box_canvas')
        touches = None
        if box is not None:
            touches = box[0] <= cfg['edge_x'][0] or box[2] >= cfg['edge_x'][1] or box[3] >= cfg['edge_bottom']
        corners8 = np.concatenate([corners, corners + (R.T @ (H * az))[None, :]], 0)   # bottom four, then top four
        out.append(dict(t=round(k / 10, 1), x_near=float(corners[:, 0].min()), x_c=float(c[0]), y_c=float(c[1]), corners8=corners8,
                        y_near=float(corners[np.argmin(np.abs(corners[:, 1])), 1]), z_bottom=float(c[2] - H / 2), box=box,
                        in_band=bool(r.get('in_band')), visible=bool(r.get('visible')), touches_edge=touches))
    return out


def gt_exit(series):
    """First tick t > 0 at which the gt box touches the edge limits or leaves the band (None if it never does)."""
    for g in series[1:]:
        if g is None:
            continue
        if g['touches_edge'] or not g['in_band']:
            return g['t']
    return None


# ------------------------------------------------------------------------------------------------- geometry check
def cmd_geometry(args):
    """Flat-road constants against gt.json on DEV raw frames: (a) the projected gt box's bottom-centre through the
    window's flat-road geometry (geometry alone), (b) the best-matching detector box (IoU >= 0.5, right class) through
    the same geometry (geometry + detector box bottom)."""
    names = load_names(args.bench, args.names_file)
    cfg = json.loads(args.config.read_text())
    recs = []
    for name in names:
        wdir = args.bench / 'windows-dev' / name
        window = json.loads((wdir / 'window.json').read_text()); gt = json.loads((wdir / 'gt.json').read_text())
        geo = mb.geometry_of(window, dict(cfg, dv=0.0))
        dets = {s: json.loads((args.bench / 'detections' / s / f'{name}.json').read_text()) for s in ('canvas-raw', 'nano-raw')
                if (args.bench / 'detections' / s / f'{name}.json').is_file()}
        seen = set()
        for fam, tgt in window['targets'].items():
            if fam not in mb.FAMILY_KIND or tgt['target_hash'] in seen:
                continue
            seen.add(tgt['target_hash'])
            classes = mb.CLASSES[mb.FAMILY_KIND[fam]]
            for g in gt_series(window, gt, fam, cfg['common']):
                if g is None or g['box'] is None or not g['in_band'] or g['touches_edge']:
                    continue
                b = g['box']
                p = geo.ground((b[0] + b[2]) / 2, b[3])
                uv, z = geo.project(g['corners8'])
                proj_err = float(np.max(np.abs(np.array([uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()]) - np.array(b)))) if (z > 0.1).all() else None
                rec = dict(name=name, family=fam, kind=mb.FAMILY_KIND[fam], t=g['t'], x_near=g['x_near'], x_c=g['x_c'], y_c=g['y_c'],
                           z_bottom=g['z_bottom'], visible=g['visible'], gtbox_x=None if p is None else p[0], gtbox_y=None if p is None else p[1],
                           box_h=b[3] - b[1], cam_h=geo.h, proj_err_px=proj_err)
                for s, d in dets.items():
                    cand = [x for x in d['detections'][20 + int(round(g['t'] * 10))] if int(x[5]) in classes and x[4] >= 0.4]
                    best = max(cand, key=lambda x: mb.iou(x[:4], b), default=None)
                    if best is not None and mb.iou(best[:4], b) >= 0.5:
                        q = geo.read_box(best)
                        rec[f'{s}_x'] = None if q is None else q[0]; rec[f'{s}_y'] = None if q is None else q[1]
                        rec[f'{s}_bottom_minus_gt_px'] = (best[3] - 0.5) - b[3]
                recs.append(rec)
    summary = geometry_summary(recs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(summary=summary, n=len(recs), records=recs)) + '\n')
    print(json.dumps(summary, indent=1))


def _stats(e):
    e = np.asarray([v for v in e if v is not None and np.isfinite(v)], float)
    if len(e) == 0:
        return dict(n=0)
    return dict(n=int(len(e)), median=round(float(np.median(e)), 4), p10=round(float(np.percentile(e, 10)), 4), p90=round(float(np.percentile(e, 90)), 4),
                median_abs=round(float(np.median(np.abs(e))), 4), p90_abs=round(float(np.percentile(np.abs(e), 90)), 4))


def geometry_summary(recs):
    out = {}
    bins = [(0, 10), (10, 20), (20, 40), (40, 80)]
    for kind in ('vehicle', 'ped', 'rider'):
        R = [r for r in recs if r['kind'] == kind and r['visible']]
        if not R:
            continue
        o = dict(n=len(R))
        o['gtbox_x_minus_x_near_rel'] = _stats([(r['gtbox_x'] - r['x_near']) / r['x_near'] for r in R if r['gtbox_x'] is not None])
        o['gtbox_y_minus_y_c'] = _stats([r['gtbox_y'] - r['y_c'] for r in R if r['gtbox_y'] is not None])
        o['z_bottom_of_cuboid'] = _stats([r['z_bottom'] for r in R])
        o['projection_contract_max_abs_px'] = _stats([r['proj_err_px'] for r in R if r.get('proj_err_px') is not None])
        for s in ('canvas-raw', 'nano-raw'):
            o[f'{s}_x_minus_x_near_m'] = _stats([r[f'{s}_x'] - r['x_near'] for r in R if r.get(f'{s}_x') is not None])
            o[f'{s}_x_minus_x_near_rel'] = _stats([(r[f'{s}_x'] - r['x_near']) / r['x_near'] for r in R if r.get(f'{s}_x') is not None])
            o[f'{s}_x_minus_x_c_rel'] = _stats([(r[f'{s}_x'] - r['x_c']) / r['x_c'] for r in R if r.get(f'{s}_x') is not None])
            o[f'{s}_y_minus_y_c'] = _stats([r[f'{s}_y'] - r['y_c'] for r in R if r.get(f'{s}_y') is not None])
            o[f'{s}_bottom_minus_gt_px'] = _stats([r[f'{s}_bottom_minus_gt_px'] for r in R if r.get(f'{s}_bottom_minus_gt_px') is not None])
            for lo, hi in bins:
                sel = [r for r in R if r.get(f'{s}_x') is not None and lo <= r['x_near'] < hi]
                o[f'{s}_x_minus_x_near_m|x_near {lo}-{hi}'] = _stats([r[f'{s}_x'] - r['x_near'] for r in sel])
        out[kind] = o
    return out



# ----------------------------------------------------------------------------------------------------------- report
def clip_box(b):
    return [min(max(b[0], 0.0), 512.0), min(max(b[1], 0.0), 288.0), min(max(b[2], 0.0), 512.0), min(max(b[3], 0.0), 288.0)]


def gt_iou(box, gtbox):
    """IoU of a tracked box with the projected gt box clipped to the canvas (a cuboid box reaching beyond the canvas
    would otherwise fail the right-target test while the tracker is on the right object)."""
    return mb.iou(box, clip_box(gtbox))


def median_tracker(rec, oname, outcomes=None):
    """Name of the tracker whose value is the ensemble's (the middle of three; lower middle if even)."""
    each = (outcomes or rec['ensemble']['outcomes'])[oname]['each']; names = sorted(rec['trackers'])
    vals = [(v, k) for v, k in zip(each, names) if v is not None]
    if not vals:
        return None
    vals.sort()
    return vals[(len(vals) - 1) // 2][1]


def reading_tick(track, h):
    """Tick (index) whose reading the tracker's outcome at horizon h uses: h if followed, else the last followed <= h."""
    j = int(round(h * 10))
    for i in range(j, -1, -1):
        r = track[i]
        if 'x' in r and r['state'] in ('followed', 'anchor_box'):
            return i
    return None


def gt_binary(fam, oname, gs, gexit):
    vis = [g for g in gs if g is not None]
    if oname == 'entered_absy_lt_1p0_by_4s':
        return float(any(abs(g['y_c']) < 1.0 for g in vis if 0 < g['t'] <= 4.0))
    if oname == 'in_absy_lt_1p5_within_2s':
        return float(any(abs(g['y_c']) < 1.5 and g['x_c'] > 0 for g in vis if 0 < g['t'] <= 2.0))
    if oname == 'out_of_view_by_2s':
        return float(gexit is not None and gexit <= 2.0)
    return None


def evaluate_source(bench, names, meas, cfg_common):
    """Per (window, family, sample) records of outcome-level checks against gt."""
    out = []
    for name in names:
        wdir = bench / 'windows-dev' / name
        if name not in meas['windows']:
            continue
        window = json.loads((wdir / 'window.json').read_text()); gt = json.loads((wdir / 'gt.json').read_text())
        for sample, clip in meas['windows'][name].items():
            for fam, rec in clip['families'].items():
                gs = gt_series(window, gt, fam, cfg_common); gex = gt_exit(gs)
                t0_ok = sum(rec['t0_found'].values()) >= 2
                # per-tick right-target rate of the median-x tracker box (ensemble series), followed ticks t > 0
                rt = [];
                for j, s in enumerate(rec['ensemble']['series'][1:], start=1):
                    if s['box'] is None or s['followed'] < 2 or gs[j] is None or gs[j]['box'] is None:
                        continue
                    rt.append(gt_iou(s['box'], gs[j]['box']) >= 0.3)
                per_tracker_rt = {}
                for k, tr in rec['trackers'].items():
                    v = [gt_iou(r['box'], gs[j]['box']) >= 0.3 for j, r in enumerate(tr['track']) if j > 0 and r['state'] == 'followed' and gs[j] is not None and gs[j]['box'] is not None]
                    per_tracker_rt[k] = (sum(v), len(v))
                ex = [e['t'] for e in rec['exits'].values() if e is not None]
                ens_exit = float(np.median([e['t'] if e else 99.0 for e in rec['exits'].values()]))
                orec = {}
                diag = mb.combine({k: mb.tracker_outcomes(tr['track'], '_diag')[0] for k, tr in rec['trackers'].items()}, '_diag')
                plan = [(on, kd, rec['ensemble']['outcomes'], on) for on, kd in mb.OUTCOMES[fam]] + [(on, kd, diag, 'diag:' + on) for on, kd in mb.OUTCOMES['_diag']]
                for oname, kind, outs, okey in plan:
                    o = outs[oname]
                    d = dict(value=o['value'], n_carried=o['n_carried'], kind=kind)
                    if o['value'] is not None and (oname.startswith(('x_', 'absy_', 'y_')) and oname.count('_') == 1):
                        key, h = oname.split('_'); h = float(h[:-1])
                        mt = median_tracker(rec, oname, outs); i = reading_tick(rec['trackers'][mt]['track'], h) if mt else None
                        if i is not None and gs[i] is not None:
                            g = gs[i]; box = rec['trackers'][mt]['track'][i]['box']
                            d['right_target'] = bool(g['box'] is not None and gt_iou(box, g['box']) >= 0.3)
                            d['read_t'] = i / 10
                            d['err_at_read'] = (o['value'] - g['x_near']) if key == 'x' else (o['value'] - abs(g['y_c'])) if key == 'absy' else (o['value'] - g['y_c'])
                            d['rel_err_at_read'] = d['err_at_read'] / g['x_near'] if key == 'x' and g['x_near'] > 1 else None
                        gh = gs[int(round(h * 10))]
                        if gh is not None:
                            d['err_at_h'] = (o['value'] - gh['x_near']) if key == 'x' else (o['value'] - abs(gh['y_c'])) if key == 'absy' else (o['value'] - gh['y_c'])
                            d['gt_at_h'] = gh['x_near'] if key == 'x' else abs(gh['y_c']) if key == 'absy' else gh['y_c']
                        d['followed_at_h'] = o['n_carried'] <= 1 and all(s != 'none' for s in o['states'])
                    elif o['value'] is not None and oname == 'min_x_0_4s':
                        vis = [g['x_near'] for g in gs[:41] if g is not None and g['visible']]
                        if vis:
                            d['err_at_h'] = o['value'] - min(vis)
                        # right target at the reading tick: the median tracker's followed tick of its minimum over 0-4 s
                        mt = median_tracker(rec, oname, outs)
                        if mt:
                            cand = [(r['x'], j) for j, r in enumerate(rec['trackers'][mt]['track'][:41]) if r['state'] in ('followed', 'anchor_box') and 'x' in r]
                            if cand:
                                xm, j = min(cand)
                                g = gs[j]
                                if g is not None and g['box'] is not None:
                                    d['right_target'] = bool(gt_iou(rec['trackers'][mt]['track'][j]['box'], g['box']) >= 0.3); d['read_t'] = j / 10
                                    d['err_at_read'] = xm - g['x_near']; d['rel_err_at_read'] = d['err_at_read'] / g['x_near'] if g['x_near'] > 1 else None
                    elif o['value'] is not None:
                        gb = gt_binary(fam, oname, gs, gex)
                        if gb is not None:
                            d['gt_binary'] = gb; d['agree'] = float(o['value'] >= 0.5) == gb
                    orec[okey] = d
                prim = mb.OUTCOMES[fam][0][0]
                scorable = bool(t0_ok and not rec.get('t0_box_at_edge') and rec['ensemble']['outcomes'][prim]['value'] is not None)
                out.append(dict(name=name, sample=sample, family=fam, t0_found=t0_ok, scorable=scorable, right_ticks=(sum(rt), len(rt)), per_tracker_right=per_tracker_rt,
                                exit_t=ens_exit if ens_exit < 99 else None, gt_exit=gex, outcomes=orec))
    return out


def _q(v, q):
    return float(np.percentile(v, q)) if len(v) else float('nan')


def summarise(recs, fam):
    R = [r for r in recs if r['family'] == fam]
    if not R:
        return None
    s = dict(clips=len(R), t0_found=float(np.mean([r['t0_found'] for r in R])), scorable=float(np.mean([r.get('scorable', r['t0_found']) for r in R])))
    a, b = sum(r['right_ticks'][0] for r in R), sum(r['right_ticks'][1] for r in R)
    s['right_target_ticks'] = (a / b if b else float('nan'), b)
    for k in ('v3', 'v5', 'v8'):
        a, b = sum(r['per_tracker_right'].get(k, (0, 0))[0] for r in R), sum(r['per_tracker_right'].get(k, (0, 0))[1] for r in R)
        s[f'right_target_ticks_{k}'] = (a / b if b else float('nan'), b)
    oc = {}
    for oname, kind in [(on, kd) for on, kd in mb.OUTCOMES[fam]] + [('diag:' + on, kd) for on, kd in mb.OUTCOMES['_diag']]:
        D = [r['outcomes'][oname] for r in R if r['t0_found'] and oname in r['outcomes']]
        o = dict(kind=kind, n=len(D), valued=sum(d['value'] is not None for d in D))
        if any('err_at_read' in d for d in D):
            E = [d for d in D if 'err_at_read' in d and 'right_target' in d]
            o['right_target'] = float(np.mean([d['right_target'] for d in E])) if E else float('nan')
            e = np.array([d['err_at_read'] for d in E if d['right_target']])
            o['err_read_median_abs'] = float(np.median(np.abs(e))) if len(e) else float('nan'); o['err_read_p90_abs'] = _q(np.abs(e), 90); o['err_read_bias'] = float(np.median(e)) if len(e) else float('nan')
            rel = np.array([d['rel_err_at_read'] for d in E if d['right_target'] and d.get('rel_err_at_read') is not None])
            o['rel_err_read_median_abs'] = float(np.median(np.abs(rel))) if len(rel) else None; o['rel_err_read_bias'] = float(np.median(rel)) if len(rel) else None
            o['rel_err_read_p90_abs'] = _q(np.abs(rel), 90) if len(rel) else None
            F = [d for d in D if d.get('followed_at_h') and 'err_at_h' in d]
            o['followed_at_h'] = len(F) / max(1, len(D))
            e2 = np.array([d['err_at_h'] for d in F])
            o['err_h_followed_median_abs'] = float(np.median(np.abs(e2))) if len(e2) else float('nan'); o['err_h_followed_p90_abs'] = _q(np.abs(e2), 90)
            o['err_h_followed_bias'] = float(np.median(e2)) if len(e2) else float('nan')
            C = [d for d in D if not d.get('followed_at_h') and 'err_at_h' in d]
            e3 = np.array([d['err_at_h'] for d in C])
            o['carried_share'] = len(C) / max(1, len(D)); o['err_h_carried_median_abs'] = float(np.median(np.abs(e3))) if len(e3) else float('nan')
        elif any('agree' in d for d in D):
            A = [d for d in D if 'agree' in d]
            o['agree'] = float(np.mean([d['agree'] for d in A])); o['gt_rate'] = float(np.mean([d['gt_binary'] for d in A])); o['read_rate'] = float(np.mean([d['value'] >= 0.5 for d in A]))
        elif any('err_at_h' in d for d in D):
            e = np.array([d['err_at_h'] for d in D if 'err_at_h' in d])
            o['err_median_abs'] = float(np.median(np.abs(e))); o['err_p90_abs'] = _q(np.abs(e), 90); o['err_bias'] = float(np.median(e))
        oc[oname] = o
    s['outcomes'] = oc
    ex = [(r['exit_t'], r['gt_exit']) for r in R if r['t0_found']]
    s['exit_by_2s_read_vs_gt'] = dict(read=float(np.mean([e[0] is not None and e[0] <= 2 for e in ex])), gt=float(np.mean([e[1] is not None and e[1] <= 2 for e in ex])),
                                     agree=float(np.mean([(e[0] is not None and e[0] <= 2) == (e[1] is not None and e[1] <= 2) for e in ex])))
    return s


def agreement(meas_a, meas_b, names, sample_a, sample_b):
    """Raw vs recon (or two sources): per family and outcome, Spearman over windows, median |diff|, state agreement."""
    from scipy.stats import spearmanr
    res = {}
    for fam in mb.FAMILIES:
        for oname, kind in mb.OUTCOMES[fam]:
            a, b, st = [], [], []
            for n in names:
                ra = meas_a['windows'].get(n, {}).get(sample_a, {}).get('families', {}).get(fam)
                rb = meas_b['windows'].get(n, {}).get(sample_b, {}).get('families', {}).get(fam)
                if ra is None or rb is None:
                    continue
                va, vb = ra['ensemble']['outcomes'][oname]['value'], rb['ensemble']['outcomes'][oname]['value']
                if va is None or vb is None:
                    continue
                a.append(va); b.append(vb)
                st.append((ra['ensemble']['outcomes'][oname]['n_carried'] >= 2) == (rb['ensemble']['outcomes'][oname]['n_carried'] >= 2))
            if len(a) < 3:
                continue
            a, b = np.array(a), np.array(b)
            d = dict(n=len(a), kind=kind)
            if set(np.unique(a)) <= {0.0, 1.0} and set(np.unique(b)) <= {0.0, 1.0}:
                d['agree'] = float(np.mean(a == b))
            else:
                d['spearman'] = float(spearmanr(a, b).correlation) if np.std(a) > 0 and np.std(b) > 0 else float('nan')
                d['median_abs_diff'] = float(np.median(np.abs(a - b)))
                if oname.startswith(('x_', 'min_x')):
                    d['median_rel_diff'] = float(np.median(b / a - 1)); d['median_abs_rel_diff'] = float(np.median(np.abs(b / a - 1)))
                d['state_agree'] = float(np.mean(st))
            res[f'{fam}|{oname}'] = d
    return res


def cmd_report(args):
    names = load_names(args.bench, args.names_file)
    cfg = json.loads(args.config.read_text())
    out = dict(sources={}, agreement={})
    meas = {}
    for src in args.sources:
        f = args.measure / f'measure-{src}.json'
        if not f.is_file():
            continue
        meas[src] = json.loads(f.read_text())
        recs = evaluate_source(args.bench, names, meas[src], cfg['common'])
        out['sources'][src] = {fam: summarise(recs, fam) for fam in mb.FAMILIES}
        (args.out / f'records-{src}.json').parent.mkdir(parents=True, exist_ok=True)
        (args.out / f'records-{src}.json').write_text(json.dumps(recs) + '\n')
    if args.recon_run and args.recon_run in meas:
        for raw in ('canvas-raw', 'nano-raw'):
            if raw in meas:
                out['agreement'][f'{raw} vs {args.recon_run}'] = agreement(meas[raw], meas[args.recon_run], names, 'raw', 'real')
    if 'canvas-raw' in meas and 'nano-raw' in meas:
        out['agreement']['canvas-raw vs nano-raw'] = agreement(meas['canvas-raw'], meas['nano-raw'], names, 'raw', 'raw')
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'validation.json').write_text(json.dumps(out, indent=1) + '\n')
    print_tables(out)


def print_tables(out):
    for src, fams in out['sources'].items():
        print(f'\n### {src}')
        print('| family | clips | t0 found | right target (ticks) | outcome | n | right target | err median abs | err p90 abs | bias | rel bias | followed at h | carried |')
        print('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
        for fam, s in fams.items():
            if s is None:
                continue
            for oname, o in s['outcomes'].items():
                if 'right_target' in o:
                    print(f"| {fam} | {s['clips']} | {s['t0_found']:.3f} | {s['right_target_ticks'][0]:.3f} | {oname} | {o['n']} | {o['right_target']:.3f} | "
                          f"{o['err_read_median_abs']:.2f} | {o['err_read_p90_abs']:.2f} | {o['err_read_bias']:+.2f} | "
                          f"{'' if o.get('rel_err_read_bias') is None else format(o['rel_err_read_bias'], '+.3f')} | {o['followed_at_h']:.2f} | {o['carried_share']:.2f} |")
                elif 'agree' in o:
                    print(f"| {fam} | {s['clips']} | {s['t0_found']:.3f} | {s['right_target_ticks'][0]:.3f} | {oname} | {o['n']} | agree {o['agree']:.3f} (gt {o['gt_rate']:.2f}, read {o['read_rate']:.2f}) | | | | | | |")
                elif 'err_median_abs' in o:
                    print(f"| {fam} | {s['clips']} | {s['t0_found']:.3f} | {s['right_target_ticks'][0]:.3f} | {oname} | {o['n']} | | {o['err_median_abs']:.2f} | {o['err_p90_abs']:.2f} | {o['err_bias']:+.2f} | | | |")
    for k, v in out['agreement'].items():
        print(f'\n### agreement {k}')
        for kk, d in v.items():
            print(kk, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in d.items()})


# ---------------------------------------------------------------------------------------------- history readings
def evaluate_history(bench, names, meas, cfg_common):
    """Backward history readings (t = 0 .. -2.0 s) against gt: per frame right target and x / y error, and the
    least-squares slopes against the gt slopes over the same frames and over the whole 2 s."""
    out = []
    for name in names:
        if name not in meas['windows']:
            continue
        wdir = bench / 'windows-dev' / name
        window = json.loads((wdir / 'window.json').read_text()); gt = json.loads((wdir / 'gt.json').read_text())
        for sample, clip in meas['windows'][name].items():
            if sample not in ('raw', 'real'):
                continue
            for fam, rec in clip['families'].items():
                if sum(rec['t0_found'].values()) < 2:
                    continue
                h = rec.get('history') or {}
                series = h.get('series') or []
                gs = gt_series(window, gt, fam, cfg_common, ks=range(0, -21, -1))      # aligned with the backward run
                frames = []
                for j, sr in enumerate(series):
                    g = gs[j] if j < len(gs) else None
                    if sr['x'] is None or g is None:
                        continue
                    frames.append(dict(t=sr['t'], right=bool(g['box'] is not None and sr['box'] is not None and gt_iou(sr['box'], g['box']) >= 0.3),
                                       ex=sr['x'] - g['x_near'], ey=sr['y'] - g['y_c'], gx=g['x_near'], gy=g['y_c']))
                fit = h.get('fit')
                d = dict(name=name, sample=sample, family=fam, frames=frames, fit=fit, n_series=sum(sr['x'] is not None for sr in series))
                if fit is not None:
                    used = [f for f in frames]
                    if len(used) >= 3:
                        T = np.array([f['t'] for f in used]); A = np.stack([np.ones_like(T), T], 1)
                        (gx0, gvx), *_ = np.linalg.lstsq(A, np.array([f['gx'] for f in used]), rcond=None)
                        (gy0, gvy), *_ = np.linalg.lstsq(A, np.array([f['gy'] for f in used]), rcond=None)
                        d['gt_same_frames'] = dict(x0=float(gx0), vx=float(gvx), y0=float(gy0), vy=float(gvy))
                    full = [g for g in gs if g is not None]
                    if len(full) >= 3:
                        T = np.array([g['t'] for g in full]); A = np.stack([np.ones_like(T), T], 1)
                        (gx0, gvx), *_ = np.linalg.lstsq(A, np.array([g['x_near'] for g in full]), rcond=None)
                        (gy0, gvy), *_ = np.linalg.lstsq(A, np.array([g['y_c'] for g in full]), rcond=None)
                        d['gt_full'] = dict(x0=float(gx0), vx=float(gvx), y0=float(gy0), vy=float(gvy), t_first=float(T.min()))
                # constant-velocity forecasts vs the instrument's own forward reading of the same real clip at h (what the
                # benchmark scores against): (A) the instrument's history line, (B) the instrument's t0 reading + the
                # annotations' 2 s velocity (privileged velocity), (C) the annotations' line (privileged, annotation space)
                ser = rec['ensemble']['series']
                d['cv'] = {}
                for hh in (1.0, 2.0, 4.0):
                    sr = ser[int(round(hh * 10))]
                    if sr['x'] is None or sr['followed'] < 2:
                        continue
                    c = dict(x_read=sr['x'], y_read=sr['y'])
                    if fit is not None:
                        c['A'] = (fit['x0'] + hh * fit['vx'] - sr['x'], fit['y0'] + hh * fit['vy'] - sr['y'])
                        if 'gt_full' in d and h.get('x_0s') is not None:
                            gf = d['gt_full']
                            c['B'] = (h['x_0s'] + hh * gf['vx'] - sr['x'], h['y_0s'] + hh * gf['vy'] - sr['y'])
                            c['C'] = (gf['x0'] + hh * gf['vx'] - sr['x'], gf['y0'] + hh * gf['vy'] - sr['y'])
                    d['cv'][str(hh)] = c
                out.append(d)
    return out


def summarise_history(recs, fam):
    R = [r for r in recs if r['family'] == fam]
    if not R:
        return None
    F = [f for r in R for f in r['frames'] if f['t'] < 0]
    s = dict(window_families=len(R), history_frames_read=len(F), frames_per_target_median=float(np.median([r['n_series'] for r in R])),
             right_target_frames=float(np.mean([f['right'] for f in F])) if F else float('nan'),
             targets_all_frames_right=float(np.mean([all(f['right'] for f in r['frames'] if f['t'] < 0) for r in R if any(f['t'] < 0 for f in r['frames'])])) if R else float('nan'),
             fit_available=float(np.mean([r['fit'] is not None for r in R])),
             full_span=float(np.mean([r['fit'] is not None and r['fit']['t_first'] <= -2.0 + 1e-6 for r in R])),
             t_first_median=float(np.median([r['fit']['t_first'] for r in R if r['fit'] is not None])) if any(r['fit'] for r in R) else float('nan'))
    G = [f for f in F if f['right']]
    for key, lab in (('ex', 'x'), ('ey', 'y')):
        e = np.array([f[key] for f in G])
        s[f'{lab}_err_median_abs'] = float(np.median(np.abs(e))) if len(e) else float('nan')
        s[f'{lab}_err_p90_abs'] = float(np.percentile(np.abs(e), 90)) if len(e) else float('nan')
        s[f'{lab}_bias'] = float(np.median(e)) if len(e) else float('nan')
        for tt in (-1.0, -2.0):
            e2 = np.array([f[key] for f in G if abs(f['t'] - tt) < 1e-6])
            s[f'{lab}_err_median_abs_at_{tt}'] = float(np.median(np.abs(e2))) if len(e2) else float('nan')
    for ref in ('gt_same_frames', 'gt_full'):
        W = [r for r in R if r['fit'] is not None and ref in r]
        for k in ('vx', 'vy'):
            e = np.array([r['fit'][k] - r[ref][k] for r in W])
            s[f'{k}_err_vs_{ref}_median_abs'] = float(np.median(np.abs(e))) if len(e) else float('nan')
            s[f'{k}_err_vs_{ref}_p90_abs'] = float(np.percentile(np.abs(e), 90)) if len(e) else float('nan')
            s[f'{k}_err_vs_{ref}_bias'] = float(np.median(e)) if len(e) else float('nan')
        # constant-velocity prediction at 2 s from the line, instrument vs gt line (x0 + 2 vx)
        e = np.array([(r['fit']['x0'] + 2 * r['fit']['vx']) - (r[ref]['x0'] + 2 * r[ref]['vx']) for r in W])
        s[f'cv_x_at_2s_err_vs_{ref}_median_abs'] = float(np.median(np.abs(e))) if len(e) else float('nan')
        s[f'cv_x_at_2s_err_vs_{ref}_p90_abs'] = float(np.percentile(np.abs(e), 90)) if len(e) else float('nan')
        e = np.array([(r['fit']['y0'] + 2 * r['fit']['vy']) - (r[ref]['y0'] + 2 * r[ref]['vy']) for r in W])
        s[f'cv_y_at_2s_err_vs_{ref}_median_abs'] = float(np.median(np.abs(e))) if len(e) else float('nan')
        s[f'cv_y_at_2s_err_vs_{ref}_p90_abs'] = float(np.percentile(np.abs(e), 90)) if len(e) else float('nan')
    # forecasts A / B / C against the instrument's forward reading (followed at h by >= 2 trackers), on the same clips
    for hh in ('1.0', '2.0', '4.0'):
        C = [r['cv'][hh] for r in R if hh in r.get('cv', {}) and all(k in r['cv'][hh] for k in ('A', 'B', 'C'))]
        s[f'cv_{hh}_n'] = len(C)
        for k in ('A', 'B', 'C'):
            ex = np.array([c[k][0] for c in C]); ey = np.array([c[k][1] for c in C])
            s[f'cv_{hh}_{k}_x_median_abs'] = float(np.median(np.abs(ex))) if len(ex) else float('nan')
            s[f'cv_{hh}_{k}_y_median_abs'] = float(np.median(np.abs(ey))) if len(ey) else float('nan')
    return s


def cmd_history(args):
    names = load_names(args.bench, args.names_file)
    cfg = json.loads(args.config.read_text())
    out = {}
    for src in args.sources:
        f = args.measure / f'measure-{src}.json'
        if not f.is_file():
            continue
        recs = evaluate_history(args.bench, names, json.loads(f.read_text()), cfg['common'])
        out[src] = {fam: summarise_history(recs, fam) for fam in mb.FAMILIES}
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / f'history-records-{src}.json').write_text(json.dumps(recs) + '\n')
    (args.out / 'history-validation.json').write_text(json.dumps(out, indent=1) + '\n')
    for src, fams in out.items():
        print(f'\n#### {src}: history readings (t = -2.0 .. -0.1 s, backward from t0)\n')
        print('| family | n | fit available | full 2 s span | frames read (median of 21) | right target, frames | x err median / p90 | y err median / p90 | x err at -1 s / -2 s | vx err median / p90 (vs gt, same frames) | vy err median / p90 | vx err vs gt full 2 s | CV x(2 s) err median / p90 | CV y(2 s) err median / p90 |')
        print('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
        for fam, h in fams.items():
            if h is None:
                continue
            print(f"| {fam} | {h['window_families']} | {100 * h['fit_available']:.0f}% | {100 * h['full_span']:.0f}% | {h['frames_per_target_median']:.0f} | {h['right_target_frames']:.3f} | "
                  f"{h['x_err_median_abs']:.2f} / {h['x_err_p90_abs']:.2f} | {h['y_err_median_abs']:.2f} / {h['y_err_p90_abs']:.2f} | "
                  f"{h['x_err_median_abs_at_-1.0']:.2f} / {h['x_err_median_abs_at_-2.0']:.2f} | "
                  f"{h['vx_err_vs_gt_same_frames_median_abs']:.2f} / {h['vx_err_vs_gt_same_frames_p90_abs']:.2f} | {h['vy_err_vs_gt_same_frames_median_abs']:.2f} / {h['vy_err_vs_gt_same_frames_p90_abs']:.2f} | "
                  f"{h['vx_err_vs_gt_full_median_abs']:.2f} | {h['cv_x_at_2s_err_vs_gt_full_median_abs']:.2f} / {h['cv_x_at_2s_err_vs_gt_full_p90_abs']:.2f} | "
                  f"{h['cv_y_at_2s_err_vs_gt_full_median_abs']:.2f} / {h['cv_y_at_2s_err_vs_gt_full_p90_abs']:.2f} |")
        print(f'\n#### {src}: constant-velocity forecast error against the instrument\'s own forward reading (followed at h); median |error| x / y (m)\n')
        print('| family | h | n | A: instrument 2 s line | B: instrument t0 + annotation velocity | C: annotation line |')
        print('|---|---|---|---|---|---|')
        for fam, h in fams.items():
            if h is None:
                continue
            for hh in ('1.0', '2.0', '4.0'):
                if not h.get(f'cv_{hh}_n'):
                    continue
                print(f"| {fam} | {hh} s | {h[f'cv_{hh}_n']} | " + ' | '.join(f"{h[f'cv_{hh}_{k}_x_median_abs']:.2f} / {h[f'cv_{hh}_{k}_y_median_abs']:.2f}" for k in ('A', 'B', 'C')) + ' |')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('geometry')
    p.add_argument('--bench', type=Path, required=True); p.add_argument('--names-file', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p = sub.add_parser('report')
    p.add_argument('--bench', type=Path, required=True); p.add_argument('--names-file', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True); p.add_argument('--measure', type=Path, required=True)
    p.add_argument('--sources', nargs='+', required=True); p.add_argument('--recon-run', default=None); p.add_argument('--out', type=Path, required=True)
    p = sub.add_parser('history')
    p.add_argument('--bench', type=Path, required=True); p.add_argument('--names-file', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True); p.add_argument('--measure', type=Path, required=True)
    p.add_argument('--sources', nargs='+', required=True); p.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    {'geometry': cmd_geometry, 'report': cmd_report, 'history': cmd_history}[args.cmd](args)


if __name__ == '__main__':
    main()
