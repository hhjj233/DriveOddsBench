#!/usr/bin/env python3
"""DEV validation of the shared canvas instrument against AV2 ground truth (validate_bench_v2.py on a model's canvas
geometry and frame rate; DEV only, gt.json exists for DEV windows only).

Ground truth as in validate_bench_v2 (gt_series, gt_exit, gt_binary, right target IoU >= 0.3 with the projected cuboid
box clipped to the canvas), with the projected boxes and the in-band flags of the spec's band and valid rows
(ci_core.gt_on_band), at the model's ticks only (5 Hz: k = 0, 2, ..., 60): outcomes, exits, binary truths and the
minimum over 0-4 s are compared with what the model's frames can show. With the 10 Hz bench spec every function returns
validate_bench_v2's records exactly (regress_canvas.py validate).

  validate_canvas.py report   --bench B --names-file F --model M --measure DIR --sources RAW REF ... --raw-source RAW
                              --recon-run REF --out DIR
  validate_canvas.py geometry --bench B --names-file F --model M [--det-root DIR... --sources SRC...] --out OUT.json
  validate_canvas.py t0       --bench B --names-file F --measure DIR --pairs REF:SAMPLE=CLIP:SAMPLE ... --out OUT.json
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import ci_core as ci  # noqa: E402
import measure_bench_v2 as mb  # noqa: E402  (frozen, read only)
import validate_bench_v2 as vb  # noqa: E402  (frozen, read only)


def load(bench, name, spec):
    wdir = bench / 'windows-dev' / name
    window = json.loads((wdir / 'window.json').read_text()); gt = json.loads((wdir / 'gt.json').read_text())
    return window, ci.gt_on_band(gt, spec)


def gt_model(window, gt, fam, cfg_common, spec, ks=None):
    """gt_series at the model's ticks (forward k = 0..60 by default)."""
    step = spec['step']
    ks = list(range(0, 61, step)) if ks is None else ks
    return vb.gt_series(window, gt, fam, cfg_common, ks=ks)


def reading_tick(track, h, fps):
    j = int(round(h * fps))
    for i in range(j, -1, -1):
        r = track[i]
        if 'x' in r and r['state'] in ('followed', 'anchor_box'):
            return i
    return None


def evaluate_source(bench, names, meas, cfg_common, spec):
    """validate_bench_v2.evaluate_source on the model's frame grid (index j = model frame after t0)."""
    fps = spec['fps']; n4 = 4 * fps + 1
    out = []
    for name in names:
        if name not in meas['windows']:
            continue
        window, gt = load(bench, name, spec)
        for sample, clip in meas['windows'][name].items():
            for fam, rec in clip['families'].items():
                gs = gt_model(window, gt, fam, cfg_common, spec); gex = vb.gt_exit(gs)
                t0_ok = sum(rec['t0_found'].values()) >= 2
                rt = []
                for j, s in enumerate(rec['ensemble']['series'][1:], start=1):
                    if s['box'] is None or s['followed'] < 2 or gs[j] is None or gs[j]['box'] is None:
                        continue
                    rt.append(vb.gt_iou(s['box'], gs[j]['box']) >= 0.3)
                per_tracker_rt = {}
                for k, tr in rec['trackers'].items():
                    v = [vb.gt_iou(r['box'], gs[j]['box']) >= 0.3 for j, r in enumerate(tr['track']) if j > 0 and r['state'] == 'followed' and gs[j] is not None and gs[j]['box'] is not None]
                    per_tracker_rt[k] = (sum(v), len(v))
                ens_exit = float(np.median([e['t'] if e else 99.0 for e in rec['exits'].values()]))
                orec = {}
                diag = mb.combine({k: mb.tracker_outcomes(tr['track'], '_diag')[0] for k, tr in rec['trackers'].items()}, '_diag')
                plan = [(on, kd, rec['ensemble']['outcomes'], on) for on, kd in mb.OUTCOMES[fam]] + [(on, kd, diag, 'diag:' + on) for on, kd in mb.OUTCOMES['_diag']]
                for oname, kind, outs, okey in plan:
                    o = outs[oname]
                    d = dict(value=o['value'], n_carried=o['n_carried'], kind=kind)
                    if o['value'] is not None and (oname.startswith(('x_', 'absy_', 'y_')) and oname.count('_') == 1):
                        key, h = oname.split('_'); h = float(h[:-1])
                        mt = vb.median_tracker(rec, oname, outs); i = reading_tick(rec['trackers'][mt]['track'], h, fps) if mt else None
                        if i is not None and gs[i] is not None:
                            g = gs[i]; box = rec['trackers'][mt]['track'][i]['box']
                            d['right_target'] = bool(g['box'] is not None and vb.gt_iou(box, g['box']) >= 0.3)
                            d['read_t'] = i / fps
                            d['err_at_read'] = (o['value'] - g['x_near']) if key == 'x' else (o['value'] - abs(g['y_c'])) if key == 'absy' else (o['value'] - g['y_c'])
                            d['rel_err_at_read'] = d['err_at_read'] / g['x_near'] if key == 'x' and g['x_near'] > 1 else None
                        gh = gs[int(round(h * fps))]
                        if gh is not None:
                            d['err_at_h'] = (o['value'] - gh['x_near']) if key == 'x' else (o['value'] - abs(gh['y_c'])) if key == 'absy' else (o['value'] - gh['y_c'])
                            d['gt_at_h'] = gh['x_near'] if key == 'x' else abs(gh['y_c']) if key == 'absy' else gh['y_c']
                        d['followed_at_h'] = o['n_carried'] <= 1 and all(s != 'none' for s in o['states'])
                    elif o['value'] is not None and oname == 'min_x_0_4s':
                        vis = [g['x_near'] for g in gs[:n4] if g is not None and g['visible']]
                        if vis:
                            d['err_at_h'] = o['value'] - min(vis)
                        mt = vb.median_tracker(rec, oname, outs)
                        if mt:
                            cand = [(r['x'], j) for j, r in enumerate(rec['trackers'][mt]['track'][:n4]) if r['state'] in ('followed', 'anchor_box') and 'x' in r]
                            if cand:
                                xm, j = min(cand)
                                g = gs[j]
                                if g is not None and g['box'] is not None:
                                    d['right_target'] = bool(vb.gt_iou(rec['trackers'][mt]['track'][j]['box'], g['box']) >= 0.3); d['read_t'] = j / fps
                                    d['err_at_read'] = xm - g['x_near']; d['rel_err_at_read'] = d['err_at_read'] / g['x_near'] if g['x_near'] > 1 else None
                    elif o['value'] is not None:
                        gb = vb.gt_binary(fam, oname, gs, gex)
                        if gb is not None:
                            d['gt_binary'] = gb; d['agree'] = float(o['value'] >= 0.5) == gb
                    orec[okey] = d
                prim = mb.OUTCOMES[fam][0][0]
                scorable = bool(t0_ok and not rec.get('t0_box_at_edge') and rec['ensemble']['outcomes'][prim]['value'] is not None)
                out.append(dict(name=name, sample=sample, family=fam, t0_found=t0_ok, scorable=scorable, right_ticks=(sum(rt), len(rt)), per_tracker_right=per_tracker_rt,
                                exit_t=ens_exit if ens_exit < 99 else None, gt_exit=gex, outcomes=orec))
    return out


def evaluate_history(bench, names, meas, cfg_common, spec):
    """validate_bench_v2.evaluate_history on the model's frame grid (backward run t = 0 .. -2.0 s at the model rate)."""
    step, fps = spec['step'], spec['fps']
    out = []
    for name in names:
        if name not in meas['windows']:
            continue
        window, gt = load(bench, name, spec)
        for sample, clip in meas['windows'][name].items():
            if sample not in ('raw', 'real'):
                continue
            for fam, rec in clip['families'].items():
                if sum(rec['t0_found'].values()) < 2:
                    continue
                h = rec.get('history') or {}
                series = h.get('series') or []
                gs = vb.gt_series(window, gt, fam, cfg_common, ks=range(0, -21, -step))
                frames = []
                for j, sr in enumerate(series):
                    g = gs[j] if j < len(gs) else None
                    if sr['x'] is None or g is None:
                        continue
                    frames.append(dict(t=sr['t'], right=bool(g['box'] is not None and sr['box'] is not None and vb.gt_iou(sr['box'], g['box']) >= 0.3),
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
                ser = rec['ensemble']['series']
                d['cv'] = {}
                for hh in (1.0, 2.0, 4.0):
                    sr = ser[int(round(hh * fps))]
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


def cmd_report(args):
    spec = ci.load_spec(args.model)
    names = ci.names_from(args.names_file); ci.dev_only(args.bench, names)
    cfg = json.loads((args.measure / f'effective-config-{spec["model"]}.json').read_text())
    out = dict(model=spec['model'], spec=spec, sources={}, agreement={}, history={}, script_sha256=dict(validate_canvas=ci.sha256_file(__file__), ci_core=ci.sha256_file(ci.__file__)))
    meas = {}
    args.out.mkdir(parents=True, exist_ok=True)
    for src in args.sources:
        f = args.measure / f'measure-{src}.json'
        if not f.is_file():
            continue
        meas[src] = json.loads(f.read_text())
        recs = evaluate_source(args.bench, names, meas[src], cfg['common'], spec)
        out['sources'][src] = {fam: vb.summarise(recs, fam) for fam in mb.FAMILIES}
        (args.out / f'records-{src}.json').write_text(json.dumps(recs) + '\n')
        hrecs = evaluate_history(args.bench, names, meas[src], cfg['common'], spec)
        out['history'][src] = {fam: vb.summarise_history(hrecs, fam) for fam in mb.FAMILIES}
    for pair in args.pairs:
        a, b = pair.split('=')
        (sa, ka), (sb, kb) = a.split(':'), b.split(':')
        if sa in meas and sb in meas:
            out['agreement'][f'{sa} vs {sb}'] = vb.agreement(meas[sa], meas[sb], names, ka, kb)
    (args.out / 'validation.json').write_text(json.dumps(out, indent=1) + '\n')
    vb.print_tables(out)


def cmd_geometry(args):
    """validate_bench_v2.cmd_geometry on the spec's band: (a) the projected gt box's bottom-centre through the band's
    flat-road geometry (geometry alone; band intrinsics K_band), projection contract (gt cuboid through K_band vs the
    band box), (b) optionally the best-matching detector box (IoU >= 0.5, right class, score >= 0.4) of each source."""
    spec = ci.load_spec(args.model)
    names = ci.names_from(args.names_file); ci.dev_only(args.bench, names)
    cfg = ci.effective_config(json.loads((args.bench / 'instrument' / 'config-v2.json').read_text()), spec)
    recs = []
    for name in names:
        window, gt = load(args.bench, name, spec)
        wb = ci.window_on_band(window, spec)
        geo = mb.geometry_of(wb, dict(cfg, dv=0.0))
        dets = {}
        for s in args.sources:
            for r in args.det_root:
                f = Path(r) / s / f'{name}.json'
                if f.is_file():
                    dets[s] = json.loads(f.read_text()); break
        seen = set()
        for fam, tgt in window['targets'].items():
            if fam not in mb.FAMILY_KIND or tgt['target_hash'] in seen:
                continue
            seen.add(tgt['target_hash'])
            classes = mb.CLASSES[mb.FAMILY_KIND[fam]]
            for g in gt_model(window, gt, fam, cfg['common'], spec):
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
                    cand = [x for x in d['detections'][ci.frame_of_tick(spec, 20 + int(round(g['t'] * 10)))] if int(x[5]) in classes and x[4] >= 0.4]
                    best = max(cand, key=lambda x: mb.iou(x[:4], b), default=None)
                    if best is not None and mb.iou(best[:4], b) >= 0.5:
                        q = geo.read_box(best)
                        rec[f'{s}_x'] = None if q is None else q[0]; rec[f'{s}_y'] = None if q is None else q[1]
                        rec[f'{s}_bottom_minus_gt_px'] = (best[3] - 0.5) - b[3]
                recs.append(rec)
    summary = {}
    for kind in ('vehicle', 'ped', 'rider'):
        R = [r for r in recs if r['kind'] == kind and r['visible']]
        if not R:
            continue
        o = dict(n=len(R))
        o['gtbox_x_minus_x_near_rel'] = vb._stats([(r['gtbox_x'] - r['x_near']) / r['x_near'] for r in R if r['gtbox_x'] is not None])
        o['gtbox_y_minus_y_c'] = vb._stats([r['gtbox_y'] - r['y_c'] for r in R if r['gtbox_y'] is not None])
        o['projection_contract_max_abs_px'] = vb._stats([r['proj_err_px'] for r in R if r.get('proj_err_px') is not None])
        o['x_near_m'] = vb._stats([r['x_near'] for r in R])
        for s in args.sources:
            o[f'{s}_x_minus_x_near_m'] = vb._stats([r[f'{s}_x'] - r['x_near'] for r in R if r.get(f'{s}_x') is not None])
            o[f'{s}_x_minus_x_near_rel'] = vb._stats([(r[f'{s}_x'] - r['x_near']) / r['x_near'] for r in R if r.get(f'{s}_x') is not None])
            o[f'{s}_y_minus_y_c'] = vb._stats([r[f'{s}_y'] - r['y_c'] for r in R if r.get(f'{s}_y') is not None])
            o[f'{s}_bottom_minus_gt_px'] = vb._stats([r[f'{s}_bottom_minus_gt_px'] for r in R if r.get(f'{s}_bottom_minus_gt_px') is not None])
            for lo, hi in [(0, 10), (10, 20), (20, 40), (40, 80)]:
                sel = [r for r in R if r.get(f'{s}_x') is not None and lo <= r['x_near'] < hi]
                o[f'{s}_x_minus_x_near_m|x_near {lo}-{hi}'] = vb._stats([r[f'{s}_x'] - r['x_near'] for r in sel])
        summary[kind] = o
    res = dict(model=spec['model'], band_rows=spec['band_rows'], valid_source_rows=spec['valid_source_rows'], fps=spec['fps'], summary=summary, n=len(recs),
               K_example=dict(window=names[0], K_band=ci.K_band(json.loads((args.bench / 'windows-dev' / names[0] / 'window.json').read_text()), spec['band_rows'][0]).tolist()),
               script_sha256=ci.sha256_file(__file__))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(res, records=recs)) + '\n')
    print(json.dumps(res, indent=1))


def cmd_t0(args):
    """Per-clip t0 association check (ci_core.t0_agreement) between a reference and a clip of the same window: share of
    window-families with a found reference (>= 2 trackers) where fewer than 2 trackers agree (IoU >= 0.5)."""
    names = ci.names_from(args.names_file); ci.dev_only(args.bench, names)
    out = {}
    for pair in args.pairs:
        a, b = pair.split('=')
        (sa, ka), (sb, kb) = a.split(':'), b.split(':')
        ma = json.loads((args.measure / f'measure-{sa}.json').read_text()); mbb = json.loads((args.measure / f'measure-{sb}.json').read_text())
        per_fam = {}
        for n in names:
            ra = ma['windows'].get(n, {}).get(ka, {}).get('families', {}); rb_ = mbb['windows'].get(n, {}).get(kb, {}).get('families', {})
            for fam, rec_a in ra.items():
                if fam not in rb_ or sum(rec_a['t0_found'].values()) < 2:
                    continue
                n_ok = ci.t0_agreement(rb_[fam], rec_a)
                per_fam.setdefault(fam, []).append(dict(window=n, agree=n_ok, clip_t0_found=sum(rb_[fam]['t0_found'].values())))
        out[pair] = {fam: dict(n=len(v), mismatch_share=float(np.mean([x['agree'] < 2 for x in v])),
                               clip_t0_not_found_share=float(np.mean([x['clip_t0_found'] < 2 for x in v])),
                               mismatch_windows=[x['window'] for x in v if x['agree'] < 2]) for fam, v in sorted(per_fam.items())}
        allv = [x for v in per_fam.values() for x in v]
        out[pair]['_all'] = dict(n=len(allv), mismatch_share=float(np.mean([x['agree'] < 2 for x in allv])) if allv else None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + '\n')
    for k, v in out.items():
        print(k, {f: (round(d['mismatch_share'], 3) if d.get('mismatch_share') is not None else None, d['n']) for f, d in v.items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('report')
    p.add_argument('--bench', type=Path, required=True); p.add_argument('--names-file', type=Path, required=True)
    p.add_argument('--model', required=True); p.add_argument('--measure', type=Path, required=True)
    p.add_argument('--sources', nargs='+', required=True)
    p.add_argument('--pairs', nargs='*', default=[], help='agreement pairs SRC_A:SAMPLE=SRC_B:SAMPLE (e.g. canvas-raw:raw=orbis2-dev-recon-tok:real)')
    p.add_argument('--out', type=Path, required=True)
    p = sub.add_parser('geometry')
    p.add_argument('--bench', type=Path, required=True); p.add_argument('--names-file', type=Path, required=True)
    p.add_argument('--model', required=True); p.add_argument('--det-root', nargs='*', type=Path, default=[]); p.add_argument('--sources', nargs='*', default=[])
    p.add_argument('--out', type=Path, required=True)
    p = sub.add_parser('t0')
    p.add_argument('--bench', type=Path, required=True); p.add_argument('--names-file', type=Path, required=True)
    p.add_argument('--measure', type=Path, required=True); p.add_argument('--pairs', nargs='+', required=True); p.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    {'report': cmd_report, 'geometry': cmd_geometry, 't0': cmd_t0}[args.cmd](args)


if __name__ == '__main__':
    main()
