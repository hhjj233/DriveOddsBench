#!/usr/bin/env python3
"""bench-v1 v1.2: physical quantities derived from the frozen instrument's outputs (2026-11-02).

Definitions, censoring and the reliability rule are pre-registered in
bench-v1/instrument/v1.2/QUANTITIES-PREREG.md (written before any quantity was computed). Nothing of the frozen
instrument (protocol-v1.1-instrument.md section 11) is changed or re-run here: the inputs are the per-tracker tracks of
`measure_bench_v2.py measure` (measure-<source>.json) and the planar trajectories of the gated odometry
(instrument/ego-vo/clips/<source>/<window>__<sample>.json). For DEV windows the same quantities are computed from gt.json.

  derive_bench_quantities.py derive   --bench B --measure M --sources nano-raw dev-recon dev-k4 --out D [--gt]
  derive_bench_quantities.py validate --bench B --derived D --recon-run dev-recon --gen-run dev-k4 --out D   (DEV only)
  derive_bench_quantities.py export   --bench B --derived D --measure M --run RUN --real-run RECON --verdicts V --out CSV

BLIND use on TEST: `derive` and `export` only print counts; `validate` refuses TEST windows.
"""
from __future__ import annotations

import argparse, csv, json, math, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import measure_bench_v2 as mb  # noqa: E402  (frozen; imported read-only)

F_FRONT = 3.80          # ego front bumper ahead of the rear axle (declared AV2 Fusion envelope; QUANTITIES-PREREG.md 0)
TTC_CAP, THW_CAP, PET_CAP, ENTRY_CAP, EXIT_CAP = 20.0, 10.0, 6.0, 6.0, 6.0
C_MIN = 0.3             # closing speed needed for a finite TTC (m/s)
PATH_LAT = 1.5          # lateral distance from the ego path for a PET point (m)
TRACKERS = ('v3', 'v5', 'v8')
LEAD_FAMS = ('lead', 'lead_stop', 'startup')
Q_LEAD = ['closing_speed_2s', 'closing_speed_4s', 'min_ttc_0p5_4s', 'thw_4s', 'pet_follow', 'pet_t0', 'lead_speed_4s', 'lead_hard_brake_0_4s']
Q_CUTIN = ['lat_speed_1_2s', 'entry_time_absy_lt_1', 'cutin_min_ttc']
Q_ONC = ['exit_time']
Q_EGO = ['ego_speed_2s', 'ego_speed_4s', 'ego_max_decel_0_4s', 'ego_stop_4s', 'ego_dx_0_2s', 'ego_dx_0_4s', 'ego_dx_0_6s']
BINARY = {'lead_hard_brake_0_4s', 'ego_stop_4s'}
CAPS = {'min_ttc_0p5_4s': TTC_CAP, 'cutin_min_ttc': TTC_CAP, 'thw_4s': THW_CAP, 'pet_follow': PET_CAP, 'pet_t0': PET_CAP,
        'entry_time_absy_lt_1': ENTRY_CAP, 'exit_time': EXIT_CAP}


def r1(t):
    return round(float(t), 1)


# ---------------------------------------------------------------------------------------------------------- ego
class Ego:
    """Ego trajectory at group ends t_j = 0.4 j (j = 0..15) in the t0 frame, and group advances a_j (j = 1..15)."""

    def __init__(self, T, X, Y, PSI, adv):
        self.T, self.X, self.Y, self.PSI = map(lambda a: np.asarray(a, float), (T, X, Y, PSI))
        self.adv = np.asarray(adv, float)                  # a_1 .. a_15
        self.u = self.adv / 0.4; self.mid = 0.4 * np.arange(1, len(self.adv) + 1) - 0.2
        seg = np.hypot(np.diff(self.X), np.diff(self.Y))
        self.S = np.concatenate([[0.0], np.cumsum(seg)])   # rear-axle arclength at group ends
        # path polyline for projection: group-end positions, extended 200 m along the last heading
        ex, ey = self.X[-1] + 200 * math.cos(self.PSI[-1]), self.Y[-1] + 200 * math.sin(self.PSI[-1])
        self.PX = np.concatenate([self.X, [ex]]); self.PY = np.concatenate([self.Y, [ey]])
        self.PS = np.concatenate([self.S, [self.S[-1] + 200.0]])

    def speed(self, t):
        return float(np.interp(t, self.mid, self.u))

    def pose(self, t):
        return float(np.interp(t, self.T, self.X)), float(np.interp(t, self.T, self.Y)), float(np.interp(t, self.T, self.PSI))

    def arclength_at(self, t):
        return float(np.interp(t, self.T, self.S))

    def time_at_arclength(self, s):
        """First time the rear axle's arclength reaches s (None if not within 0-6 s)."""
        if s <= self.S[0]:
            return 0.0
        idx = np.nonzero(self.S >= s)[0]
        if len(idx) == 0:
            return None
        j = int(idx[0])
        s0, s1, t0, t1 = self.S[j - 1], self.S[j], self.T[j - 1], self.T[j]
        return float(t0 + (t1 - t0) * (s - s0) / (s1 - s0)) if s1 > s0 else float(t1)

    def project(self, px, py):
        """(arclength, lateral distance) of a t0-frame point on the path polyline."""
        best = (None, 1e18)
        for i in range(len(self.PX) - 1):
            ax, ay, bx, by = self.PX[i], self.PY[i], self.PX[i + 1], self.PY[i + 1]
            dx, dy = bx - ax, by - ay; L2 = dx * dx + dy * dy
            f = 0.0 if L2 < 1e-12 else min(1.0, max(0.0, ((px - ax) * dx + (py - ay) * dy) / L2))
            qx, qy = ax + f * dx, ay + f * dy; d = math.hypot(px - qx, py - qy)
            if d < best[1]:
                best = (self.PS[i] + f * (self.PS[i + 1] - self.PS[i]), d)
        return best


def ego_from_vo(rec):
    tr = rec['planar']['trajectory']
    return Ego([p['t'] for p in tr], [p['x'] for p in tr], [p['y'] for p in tr], [math.radians(p['psi_deg']) for p in tr],
               [g['final_m'] for g in rec['groups'][1:]])


def ego_from_gt(gt):
    ego = {e['k']: e for e in gt['ego']}
    ks = list(range(0, 61, 4)); P = []
    for k in ks:
        M = np.asarray(ego[k]['matrix_3x4'], float); P.append((M[0, 3], M[1, 3], math.atan2(M[1, 0], M[0, 0]), M))
    adv = []
    for j in range(1, len(ks)):
        R0 = P[j - 1][3][:, :3]; d = np.array([P[j][0] - P[j - 1][0], P[j][1] - P[j - 1][1], P[j][3][2, 3] - P[j - 1][3][2, 3]])
        adv.append(float((R0.T @ d)[0]))               # forward component in the group's start frame
    psi = np.unwrap([p[2] for p in P])
    return Ego([k / 10 for k in ks], [p[0] for p in P], [p[1] for p in P], psi, adv)


def ego_quantities(ego: Ego, horizons=None):
    q = {'ego_speed_2s': ego.speed(2.0), 'ego_speed_4s': ego.speed(4.0)}
    w = [(ego.adv[2 * i] + ego.adv[2 * i + 1]) / 0.8 for i in range(5)]      # [0,0.8], ..., [3.2,4.0]
    q['ego_max_decel_0_4s'] = float(max((w[i] - w[i + 1]) / 0.8 for i in range(4)))
    q['ego_stop_4s'] = float(np.any(ego.u[:10] < 0.5))
    for h in (2.0, 4.0, 6.0):
        q[f'ego_dx_0_{int(h)}s'] = horizons[str(h)]['D_m'] if horizons is not None else float(np.interp(h, ego.T, ego.X))
    return {k: dict(value=v, flags=[]) for k, v in q.items()}


# ------------------------------------------------------------------------------------------------------- target
def ls_slope(pts, T, key, half=0.5, need=6, lo=None, hi=None):
    """LS slope of key ('x' or 'absy') over readings with |t - T| <= half (or lo <= t <= hi)."""
    if lo is None:
        sel = [p for p in pts if abs(p[0] - T) <= half + 1e-6]
    else:
        sel = [p for p in pts if lo - 1e-6 <= p[0] <= hi + 1e-6]
    if len(sel) < need:
        return None
    t = np.array([p[0] for p in sel]); v = np.array([p[1] if key == 'x' else abs(p[2]) for p in sel])
    if np.ptp(t) <= 0:
        return None
    return float(np.polyfit(t, v, 1)[0])


def closing(pts, T):
    """c(T) with the carried rule: T, T-0.1, ..., 0.5 -> (value, time used)."""
    for i in range(int(round((T - 0.5) / 0.1)) + 1):
        TT = r1(T - 0.1 * i)
        s = ls_slope(pts, TT, 'x')
        if s is not None:
            return -s, TT
    return None, None


def min_ttc(pts, t_from, t_to=4.0):
    best = None
    for t, x, y in pts:
        if t_from - 1e-6 <= t <= t_to + 1e-6 and t >= 0.5 - 1e-6:
            s = ls_slope(pts, t, 'x')
            if s is not None and -s > C_MIN:
                v = (x - F_FRONT) / (-s)
                best = v if best is None else min(best, v)
    if best is None or best > TTC_CAP:
        return TTC_CAP, True
    return max(best, 0.0), False


def pet(pts, ego: Ego, times):
    vals = []
    for t, x, y in pts:
        if not any(abs(t - tt) < 1e-6 for tt in times):
            continue
        ex, ey, ep = ego.pose(t)
        px, py = ex + math.cos(ep) * x - math.sin(ep) * y, ey + math.sin(ep) * x + math.cos(ep) * y
        sig, d = ego.project(px, py)
        if d > PATH_LAT:
            continue
        tau = ego.time_at_arclength(sig - F_FRONT)
        if tau is None or tau < t - 1e-9 or tau > 6.0 + 1e-9:
            continue
        vals.append(tau - t)
    if not vals:
        return PET_CAP, True
    return min(min(vals), PET_CAP), False


def lead_quantities(pts, x4, ego: Ego | None):
    q = {}
    for T in (2.0, 4.0):
        c, TT = closing(pts, T)
        q[f'closing_speed_{int(T)}s'] = dict(value=c, flags=(['carried'] if (c is not None and TT != T) else []) + ([] if c is not None else ['no_value']))
    v, cen = min_ttc(pts, 0.5)
    q['min_ttc_0p5_4s'] = dict(value=v, flags=['censored'] if cen else [])
    if ego is None:
        for k in ('thw_4s', 'pet_follow', 'pet_t0', 'lead_speed_4s', 'lead_hard_brake_0_4s'):
            q[k] = dict(value=None, flags=['no_vo_record'])
        return q
    ve = ego.speed(4.0)
    if x4 is None:
        q['thw_4s'] = dict(value=None, flags=['no_value'])
    elif ve < 0.5 or (x4 - F_FRONT) / ve > THW_CAP:
        q['thw_4s'] = dict(value=THW_CAP, flags=['censored'])
    else:
        q['thw_4s'] = dict(value=(x4 - F_FRONT) / ve, flags=[])
    v, cen = pet(pts, ego, [r1(0.1 * i) for i in range(41)])
    q['pet_follow'] = dict(value=v, flags=['censored'] if cen else [])
    v, cen = pet(pts, ego, [0.0])
    q['pet_t0'] = dict(value=v, flags=['censored'] if cen else [])
    c, TT = closing(pts, 4.0)
    q['lead_speed_4s'] = dict(value=None if c is None else ego.speed(TT) - c, flags=(['carried'] if (c is not None and TT != 4.0) else []) + ([] if c is not None else ['no_value']))
    vl = {}
    for t in (0.5, 1.5, 2.5, 3.5):
        s = ls_slope(pts, t, 'x')
        if s is not None:
            vl[t] = ego.speed(t) + s           # v_L = v_e - c = v_e + dx/dt
    pairs = [(t, t + 1.0) for t in (0.5, 1.5, 2.5) if t in vl and t + 1.0 in vl]
    q['lead_hard_brake_0_4s'] = dict(value=None, flags=['no_value']) if not pairs else dict(value=float(any(vl[a] - vl[b] > 3.0 for a, b in pairs)), flags=[])
    return q


def cutin_quantities(pts):
    q = {}
    s = ls_slope(pts, None, 'absy', lo=1.0, hi=2.0)
    flags = []
    if s is None:
        for i in range(1, 11):
            a = r1(1.0 - 0.1 * i)
            s = ls_slope(pts, None, 'absy', lo=a, hi=a + 1.0)
            if s is not None:
                flags = ['carried']; break
    q['lat_speed_1_2s'] = dict(value=None if s is None else -s, flags=flags if s is not None else ['no_value'])
    te = next((t for t, x, y in pts if t > 1e-6 and abs(y) < 1.0), None)
    q['entry_time_absy_lt_1'] = dict(value=te if te is not None else ENTRY_CAP, flags=[] if te is not None else ['censored'])
    if te is not None and te <= 4.0 + 1e-6:
        v, cen = min_ttc(pts, max(te, 0.5))
    else:
        v, cen = TTC_CAP, True
    q['cutin_min_ttc'] = dict(value=v, flags=['censored'] if cen else [])
    return q


def family_quantities(fam, per_tracker_pts, per_tracker_x4, ego, exits=None):
    """Per tracker, then the ensemble (median of >= 2 values; binary: majority)."""
    per = {}
    for k, pts in per_tracker_pts.items():
        if fam in LEAD_FAMS:
            per[k] = lead_quantities(pts, per_tracker_x4.get(k), ego)
        elif fam == 'cutin':
            per[k] = cutin_quantities(pts)
        elif fam == 'oncoming':
            e = (exits or {}).get(k)
            per[k] = {'exit_time': dict(value=EXIT_CAP if e is None else min(e['t'], EXIT_CAP), flags=['censored'] if e is None else [])}
    names = Q_LEAD if fam in LEAD_FAMS else Q_CUTIN if fam == 'cutin' else Q_ONC if fam == 'oncoming' else []
    out = {}
    for n in names:
        vals = {k: per[k][n]['value'] for k in per}
        good = [v for v in vals.values() if v is not None]
        if len(good) < 2:
            value = None
        elif n in BINARY:
            value = float(sum(good) * 2 > len(good))
        else:
            value = float(np.median(good))
        fl = set()
        for k in per:
            fl |= set(per[k][n]['flags'])
        if value is None:
            fl.add('no_value')
        if n in CAPS and value is not None and value >= CAPS[n] - 1e-9:
            fl.add('censored')
        elif 'censored' in fl and value is not None and value < CAPS.get(n, 1e18) - 1e-9:
            fl.discard('censored')
        out[n] = dict(value=value, each=vals, flags=sorted(fl))
    return out


# ------------------------------------------------------------------------------------------------------- derive
def tracker_pts(track):
    return [(r1(r['t']), r['x'], r['y']) for r in track if r['state'] in ('followed', 'anchor_box') and 'x' in r]


def vo_record(bench, source, name, sample):
    f = bench / 'instrument' / 'ego-vo' / 'clips' / source / f'{name}__{sample}.json'
    return json.loads(f.read_text()) if f.is_file() else None


def cmd_derive(args):
    for source in args.sources:
        meas = json.loads((args.measure / f'measure-{source}.json').read_text())
        out = dict(source=source, prereg='instrument/v1.2/QUANTITIES-PREREG.md', front_offset_m=F_FRONT, windows={})
        n_clips = 0
        for name, samples in meas['windows'].items():
            wdir = args.bench / ('windows-dev' if (args.bench / 'windows-dev' / name).is_dir() else 'windows-test')
            for sample, clip in samples.items():
                vrec = vo_record(args.bench, source, name, sample)
                ego = ego_from_vo(vrec) if vrec is not None else None
                rec = dict(families={}, ego=None, vo_accepted_share=None if vrec is None else vrec['quality']['accepted_share'])
                for fam, frec in clip['families'].items():
                    if fam not in LEAD_FAMS and fam not in ('cutin', 'oncoming'):
                        continue
                    pts = {k: tracker_pts(frec['trackers'][k]['track']) for k in frec['trackers']}
                    x4 = {k: frec['trackers'][k]['outcomes'].get('x_4s', {}).get('value') for k in frec['trackers']} if fam in LEAD_FAMS else {}
                    rec['families'][fam] = family_quantities(fam, pts, x4, ego, frec.get('exits'))
                if vrec is not None:
                    rec['ego'] = ego_quantities(ego, vrec['planar']['horizons'])
                out['windows'].setdefault(name, {})[sample] = rec
                n_clips += 1
                if args.gt and sample in ('raw', 'real') and wdir.name == 'windows-dev' and (wdir / name / 'gt.json').is_file() and 'gt' not in out['windows'][name]:
                    out['windows'][name]['gt'] = gt_record(args.bench, name)
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / f'derived-{source}.json').write_text(json.dumps(out) + '\n')
        print(json.dumps(dict(source=source, clips=n_clips, windows=len(out['windows']))), flush=True)


def gt_record(bench, name):
    import validate_bench_v2 as vb
    wdir = bench / 'windows-dev' / name
    window = json.loads((wdir / 'window.json').read_text()); gt = json.loads((wdir / 'gt.json').read_text())
    ego = ego_from_gt(gt)
    rec = dict(families={}, ego=ego_quantities(ego))
    for fam in window['targets']:
        if fam not in LEAD_FAMS and fam not in ('cutin', 'oncoming'):
            continue
        gs = vb.gt_series(window, gt, fam)
        pts = [(r1(g['t']), g['x_near'], g['y_c']) for g in gs if g is not None]
        x4 = next((p[1] for p in pts if abs(p[0] - 4.0) < 1e-6), None)
        exits = None
        if fam == 'oncoming':
            ge = vb.gt_exit(gs)
            exits = {k: (None if ge is None else dict(t=ge)) for k in TRACKERS}
        rec['families'][fam] = family_quantities(fam, {k: pts for k in TRACKERS}, {k: x4 for k in TRACKERS}, ego, exits)
    return rec


# ----------------------------------------------------------------------------------------------------- validate
def spearman(a, b):
    from scipy.stats import spearmanr
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(spearmanr(a, b).correlation)


def kappa(a, b):
    a, b = np.asarray(a) > 0.5, np.asarray(b) > 0.5
    po = np.mean(a == b); pe = np.mean(a) * np.mean(b) + np.mean(~a) * np.mean(~b)
    return None if pe >= 1 else float((po - pe) / (1 - pe))


def values(der, sample, fam, q):
    """{window: (value, flags)} for a family quantity (fam = 'ego' for ego quantities)."""
    out = {}
    for name, samples in der['windows'].items():
        rec = samples.get(sample)
        if rec is None:
            continue
        d = rec['ego'] if fam == 'ego' else rec['families'].get(fam)
        if d is None or q not in d:
            continue
        out[name] = (d[q]['value'], d[q]['flags'])
    return out


def cmd_validate(args):
    D = args.derived
    raw = json.loads((D / 'derived-nano-raw.json').read_text())
    rec = json.loads((D / f'derived-{args.recon_run}.json').read_text())
    gen = json.loads((D / f'derived-{args.gen_run}.json').read_text()) if (D / f'derived-{args.gen_run}.json').is_file() else None
    dev = set(l.strip() for l in (args.bench / 'names-dev.txt').read_text().splitlines() if l.strip())
    if not set(raw['windows']) <= dev:
        sys.exit('refusing: validation is DEV only')
    gtv = {n: s['gt'] for n, s in raw['windows'].items() if 'gt' in s}
    plan = [(f, q) for f in LEAD_FAMS for q in Q_LEAD] + [('cutin', q) for q in Q_CUTIN] + [('oncoming', q) for q in Q_ONC] + [('ego', q) for q in Q_EGO]
    rows = []
    for fam, q in plan:
        r = values(raw, 'raw', fam, q); c = values(rec, 'real', fam, q)
        g = {n: ((gtv[n]['ego'] if fam == 'ego' else gtv[n]['families'].get(fam, {})).get(q, {}).get('value'), None) for n in gtv
             if (fam == 'ego' or fam in gtv[n]['families'])}
        both = [n for n in r if r[n][0] is not None and g.get(n, (None,))[0] is not None]
        a = [r[n][0] for n in both]; b = [g[n][0] for n in both]
        d = dict(family=fam, quantity=q, binary=q in BINARY, n_raw=sum(v[0] is not None for v in r.values()), n_raw_gt=len(both),
                 raw_missing=float(np.mean([v[0] is None for v in r.values()])) if r else None,
                 raw_censored=float(np.mean(['censored' in v[1] for v in r.values()])) if r else None,
                 gt_censored=float(np.mean([g[n][0] is not None and q in CAPS and g[n][0] >= CAPS[q] - 1e-9 for n in g])) if g else None)
        if both:
            if q in BINARY:
                d['agree_raw_gt'] = float(np.mean([(x > .5) == (y > .5) for x, y in zip(a, b)])); d['kappa_raw_gt'] = kappa(a, b)
                d['gt_prevalence'] = float(np.mean(np.asarray(b) > .5)); d['raw_rate'] = float(np.mean(np.asarray(a) > .5))
            else:
                e = np.asarray(a) - np.asarray(b)
                d.update(spearman_raw_gt=spearman(a, b), err_median_abs=float(np.median(np.abs(e))), err_p90_abs=float(np.percentile(np.abs(e), 90)),
                         bias=float(np.median(e)))
        rr = [n for n in c if c[n][0] is not None and n in r and r[n][0] is not None]
        d['n_raw_recon'] = len(rr)
        if rr:
            a2 = [r[n][0] for n in rr]; b2 = [c[n][0] for n in rr]
            if q in BINARY:
                d['agree_raw_recon'] = float(np.mean([(x > .5) == (y > .5) for x, y in zip(a2, b2)]))
            else:
                d['spearman_raw_recon'] = spearman(a2, b2)
        rg = [n for n in c if c[n][0] is not None and g.get(n, (None,))[0] is not None]
        if rg and q not in BINARY:
            d['spearman_recon_gt'] = spearman([c[n][0] for n in rg], [g[n][0] for n in rg])
        d['recon_missing'] = float(np.mean([v[0] is None for v in c.values()])) if c else None
        d['recon_censored'] = float(np.mean(['censored' in v[1] for v in c.values()])) if c else None
        if gen is not None:
            gv = [v for s in ('0', '1', '2', '3') for v in values(gen, s, fam, q).values()]
            d['gen_missing'] = float(np.mean([v[0] is None for v in gv])) if gv else None
            d['gen_censored'] = float(np.mean(['censored' in v[1] for v in gv])) if gv else None
            d['gen_carried'] = float(np.mean(['carried' in v[1] for v in gv])) if gv else None
            d['recon_carried'] = float(np.mean(['carried' in v[1] for v in c.values()])) if c else None
            if q in BINARY:
                d['gen_rate'] = float(np.mean([v[0] > .5 for v in gv if v[0] is not None])) if gv else None
                d['recon_rate'] = float(np.mean([v[0] > .5 for v in c.values() if v[0] is not None])) if c else None
        d['verdict'], d['reasons'] = verdict(d)
        rows.append(d)
    (args.out / 'validation-v1.2.json').write_text(json.dumps(rows, indent=1) + '\n')
    (args.out / 'verdicts-v1.2.json').write_text(json.dumps({f"{d['family']}|{d['quantity']}": d['verdict'] for d in rows}, indent=1) + '\n')
    f = lambda v, n=2: '-' if v is None else (f'{v:.{n}f}' if isinstance(v, float) else str(v))
    print('| family | quantity | n raw | Spearman raw-gt / agree (kappa) | err median / p90 / bias | raw-recon (n) | recon-gt | censored raw / gt | censored recon / gen | missing recon / gen | carried recon / gen | verdict |')
    print('|---|---|---|---|---|---|---|---|---|---|---|---|')
    for d in rows:
        valid = f"agree {f(d.get('agree_raw_gt'))} ({f(d.get('kappa_raw_gt'))}), prev {f(d.get('gt_prevalence'))}" if d['binary'] else f(d.get('spearman_raw_gt'))
        err = '' if d['binary'] else f"{f(d.get('err_median_abs'))} / {f(d.get('err_p90_abs'))} / {f(d.get('bias'))}"
        rr = f(d.get('agree_raw_recon') if d['binary'] else d.get('spearman_raw_recon')) + f" ({d['n_raw_recon']})"
        print(f"| {d['family']} | {d['quantity']} | {d['n_raw']} | {valid} | {err} | {rr} | {f(d.get('spearman_recon_gt'))} | {f(d.get('raw_censored'))} / {f(d.get('gt_censored'))} | "
              f"{f(d.get('recon_censored'))} / {f(d.get('gen_censored'))} | {f(d.get('recon_missing'))} / {f(d.get('gen_missing'))} | {f(d.get('recon_carried'))} / {f(d.get('gen_carried'))} | "
              f"{d['verdict']}{(' (' + '; '.join(d['reasons']) + ')') if d['reasons'] else ''} |")


def verdict(d):
    """QUANTITIES-PREREG.md section 6."""
    reasons = []
    if d['n_raw'] < 15:
        return 'insufficient', [f"n raw {d['n_raw']} < 15"]
    if d.get('raw_censored') is not None and d['raw_censored'] >= 0.80:
        return 'not informative', [f"censored on raw {d['raw_censored']:.2f} >= 0.80"]
    if d['binary']:
        p = d.get('gt_prevalence')
        if p is None or p < 0.05 or p > 0.95:
            return 'not informative', [f"gt prevalence {p}"]
        if d.get('agree_raw_gt') is None or d['agree_raw_gt'] < 0.85:
            reasons.append(f"agreement with gt {d.get('agree_raw_gt')} < 0.85")
        if d.get('kappa_raw_gt') is None or d['kappa_raw_gt'] < 0.40:
            reasons.append(f"kappa {d.get('kappa_raw_gt')} < 0.40")
        if d['n_raw_recon'] >= 8 and (d.get('agree_raw_recon') is None or d['agree_raw_recon'] < 0.85):
            reasons.append(f"raw-recon agreement {d.get('agree_raw_recon')} < 0.85")
    else:
        if d.get('spearman_raw_gt') is None or d['spearman_raw_gt'] < 0.80:
            reasons.append(f"Spearman raw-gt {d.get('spearman_raw_gt')} < 0.80")
        if d['n_raw_recon'] >= 8 and (d.get('spearman_raw_recon') is None or d['spearman_raw_recon'] < 0.80):
            reasons.append(f"Spearman raw-recon {d.get('spearman_raw_recon')} < 0.80")
    return ('reliable' if not reasons else 'not reliable'), reasons


# ------------------------------------------------------------------------------------------------------- export
FIELDS = ['window', 'family', 'sample', 'outcome', 'kind', 'value', 'scorable', 'flags', 'value_v3', 'value_v5', 'value_v8']


def cmd_export(args):
    """Section-10 format; kind = v1.2-secondary (v1.2-descriptive for startup); scorable by the frozen family flags, the
    v1.2 verdicts and the value. Prints counts only."""
    verdicts = json.loads(args.verdicts.read_text())
    rules = json.loads((args.bench / 'instrument' / 'scorability-rules.json').read_text())
    gen = json.loads((args.derived / f'derived-{args.run}.json').read_text())
    real = json.loads((args.derived / f'derived-{args.real_run}.json').read_text())
    mg = json.loads((args.measure / f'measure-{args.run}.json').read_text())
    mr = json.loads((args.measure / f'measure-{args.real_run}.json').read_text())
    rows = []
    for name in sorted(set(gen['windows']) | set(real['windows'])):
        samples = {s: (gen['windows'][name][s], mg['windows'][name][s]) for s in gen['windows'].get(name, {}) if s != 'gt'}
        if 'real' in real['windows'].get(name, {}):
            samples['real'] = (real['windows'][name]['real'], mr['windows'][name]['real'])
        real_m = mr['windows'].get(name, {}).get('real', {}).get('families', {})
        for sample in sorted(samples, key=lambda x: (x == 'real', x.zfill(4))):
            drec, mrec = samples[sample]
            for fam, qs in drec['families'].items():
                fm = mrec['families'][fam]
                base = []
                if sum(fm['t0_found'].values()) < 2:
                    base.append('t0_not_found')
                if fam not in real_m or sum(real_m[fam]['t0_found'].values()) < 2:
                    base.append('real_t0_not_found')
                if fm.get('t0_box_at_edge'):
                    base.append('t0_box_at_edge')
                if fam in rules.get('family_not_reliable', []):
                    base.append('family_not_reliable')
                for q, d in qs.items():
                    fl = list(base) + list(d['flags'])
                    if verdicts.get(f'{fam}|{q}') != 'reliable':
                        fl.append('quantity_not_reliable')
                    hard = {'t0_not_found', 'real_t0_not_found', 't0_box_at_edge', 'family_not_reliable', 'quantity_not_reliable', 'no_value', 'no_vo_record'}
                    rows.append(dict(window=name, family=fam, sample=sample, outcome=q, kind='v1.2-descriptive' if fam == 'startup' else 'v1.2-secondary',
                                     value='' if d['value'] is None else round(d['value'], 4), scorable=int(not (set(fl) & hard) and d['value'] is not None),
                                     flags=';'.join(sorted(set(fl))),
                                     **{f'value_{k}': ('' if d.get('each', {}).get(k) is None else round(d['each'][k], 4)) for k in TRACKERS}))
            if drec.get('ego') is not None:
                for q, d in drec['ego'].items():
                    fl = list(d['flags'])
                    if (drec.get('vo_accepted_share') or 1.0) < 0.5:
                        fl.append('vo_low_support')
                    if verdicts.get(f'ego|{q}') != 'reliable':
                        fl.append('quantity_not_reliable')
                    rows.append(dict(window=name, family='ego', sample=sample, outcome=q, kind='v1.2-secondary', value=round(d['value'], 4),
                                     scorable=int('quantity_not_reliable' not in fl), flags=';'.join(sorted(set(fl))), value_v3='', value_v5='', value_v8=''))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    print(json.dumps(dict(rows=len(rows), scorable=sum(r['scorable'] for r in rows), out=str(args.out))))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('derive')
    p.add_argument('--bench', type=Path, required=True); p.add_argument('--measure', type=Path, required=True)
    p.add_argument('--sources', nargs='+', required=True); p.add_argument('--out', type=Path, required=True); p.add_argument('--gt', action='store_true')
    p = sub.add_parser('validate')
    p.add_argument('--bench', type=Path, required=True); p.add_argument('--derived', type=Path, required=True)
    p.add_argument('--recon-run', default='dev-recon'); p.add_argument('--gen-run', default='dev-k4'); p.add_argument('--out', type=Path, required=True)
    p = sub.add_parser('export')
    p.add_argument('--bench', type=Path, required=True); p.add_argument('--derived', type=Path, required=True); p.add_argument('--measure', type=Path, required=True)
    p.add_argument('--run', required=True); p.add_argument('--real-run', required=True); p.add_argument('--verdicts', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    {'derive': cmd_derive, 'validate': cmd_validate, 'export': cmd_export}[args.cmd](args)


if __name__ == '__main__':
    main()
