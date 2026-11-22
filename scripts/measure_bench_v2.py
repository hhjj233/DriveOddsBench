#!/usr/bin/env python3
"""bench-v1 instrument v2: target positions x(t), y(t) read from video on the 512x288 evaluation canvas (2026-11-01).

Generalises study-v1's frozen lead instrument (measure_lead_calibration.py v3 / v5 / v8 + combine_wm_measures.py, read
only, never modified) to every bench-v1 family. Detections come from detect_bench_v2.py (cached, per frame, canvas
coordinates). The same fixed function of the video is applied to every clip of a window (raw, reconstructed, generated).

Geometry (per window, from window.json; no fitted lead constants):
  - canvas camera: K_canvas (the source intrinsics through the band affine) and ego_SE3_cam (AV2 calibration);
  - flat road GROUND_BELOW_EGO_M = 0.33 m below the AV2 ego origin (the median LiDAR ground at the study-v1 DEV anchors,
    as in measure_ego_motion_raft.py), i.e. the camera height = ego_SE3_cam z + 0.33 (1.68-1.79 m on DEV);
  - reading point: the bottom-centre of the tracked box, (u, v) = ((x1 + x2) / 2 - 0.5, y2 - 0.5 + cfg.dv) in the
    pixel-centre convention of K (detector boxes are pixel edges); cfg.dv is the single global correction (canvas px);
  - its ray is intersected with the road: x = forward, y = left, in the AV2 ego frame at the clip's time t (origin at
    the rear axle). |y| is the lateral distance from the ego axis. Gates use the camera-forward range d = x - t_x(cam).

Trackers (study-v1 v3, v5, v8; every pixel threshold x CANVAS_PER_PILOT_PX = 512 / 234.7 = 2.18, metric gates kept,
edge stops at the canvas edges):
  v3  IoU association with flat-ground kinematic gates (range max(dist_abs, dist_rel d) and lateral per frame step);
  v5  constant-velocity predicted box, physical size gate, range dist_rel d + dist_per_frame k, lateral abs + per frame k;
  v8  bottom-centre (u, v) and log height in pixels, constant-velocity prediction, pixel gates.
Each family's target starts from its t0 box_canvas (window.json); if no detection of the family's classes matches it at
t0 the t0 box itself is the t0 observation (state anchor_box; flagged, see scorability). Following stops when the box
touches the canvas's side or bottom edge (state edge; exit recorded) or after more than max_misses missed frames (lost).
Family gate classes may override any tracker parameter (config 'family_overrides').

Outcome at horizon T (per tracker): the reading at T if followed then; otherwise the last reading before T (carried,
as in study-v1). Minima over 0..T use followed frames. Binary outcomes per tracker, then the ensemble = median over the
three trackers (majority for binaries). The exit time (first edge stop or the first miss of a final loss) is recorded.

  measure_bench_v2.py measure --bench B --names-file F --config C.json --sources canvas-raw nano-raw RUN... --out DIR
  measure_bench_v2.py export  --bench B --measure DIR --run RUN --real-run RECON_RUN --out OUT.csv
"""
from __future__ import annotations

import argparse, csv, hashlib, json, math, sys
from pathlib import Path

import numpy as np

CANVAS_W, CANVAS_H = 512, 288
PILOT_SPAN_PX = 244.9 - 10.2                 # the pilot's 256x320 frame spanned 234.7 px for the 1550 source columns
CANVAS_PER_PILOT_PX = CANVAS_W / PILOT_SPAN_PX  # 2.1815
GROUND_BELOW_EGO_M = 0.33
FPS = 10.0
T0_FRAME = 20
FAMILIES = ('lead', 'lead_stop', 'startup', 'cutin', 'ped', 'oncoming', 'crossing', 'rider')
CLASSES = {'vehicle': (3, 6, 8), 'ped': (1,), 'rider': (2, 4)}
FAMILY_KIND = dict(lead='vehicle', lead_stop='vehicle', startup='vehicle', cutin='vehicle', oncoming='vehicle', crossing='vehicle', ped='ped', rider='rider')

# outcomes per family (protocol-v1.md section 4); 'descriptive' families get a fixed descriptive set
OUTCOMES = {
    'lead': [('x_4s', 'primary'), ('x_1s', 'secondary'), ('x_2s', 'secondary'), ('x_6s', 'secondary'), ('min_x_0_4s', 'secondary')],
    'lead_stop': [('x_4s', 'primary'), ('x_2s', 'secondary'), ('min_x_0_4s', 'secondary')],
    'startup': [('x_6s', 'descriptive'), ('x_2s', 'descriptive'), ('x_4s', 'descriptive')],   # protocol: x(6) leads, family descriptive only
    'cutin': [('absy_2s', 'primary'), ('absy_4s', 'secondary'), ('x_2s', 'secondary'), ('x_4s', 'secondary'), ('entered_absy_lt_1p0_by_4s', 'secondary')],
    'ped': [('absy_2s', 'primary'), ('absy_1s', 'secondary'), ('x_1s', 'secondary'), ('x_2s', 'secondary'), ('in_absy_lt_1p5_within_2s', 'secondary')],
    'oncoming': [('x_1s', 'primary'), ('absy_1s', 'secondary'), ('out_of_view_by_2s', 'secondary')],
    'crossing': [('x_1s', 'descriptive'), ('x_2s', 'descriptive'), ('y_1s', 'descriptive'), ('y_2s', 'descriptive'), ('out_of_view_by_2s', 'descriptive')],
    'rider': [('x_1s', 'descriptive'), ('x_2s', 'descriptive'), ('absy_1s', 'descriptive'), ('absy_2s', 'descriptive')],
    # validation diagnostics only (every family at every horizon; never exported)
    '_diag': [(f'{k}_{h}s', 'diag') for k in ('x', 'absy') for h in (1, 2, 4, 6)],
}


# --------------------------------------------------------------------------------------------------------- geometry
class Geometry:
    """Flat-road back-projection of canvas pixels for one window (pixel-centre convention of K_canvas)."""

    def __init__(self, K_canvas, ego_SE3_cam, ground_below_ego=GROUND_BELOW_EGO_M, dv=0.0, max_range=250.0):
        K = np.asarray(K_canvas, float); T = np.asarray(ego_SE3_cam, float)
        self.fx, self.fy, self.cx, self.cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        self.R, self.t = T[:3, :3], T[:3, 3]
        self.h = float(self.t[2]) + ground_below_ego      # camera height above the road
        self.g = ground_below_ego; self.dv = dv; self.max_range = max_range

    def ground(self, u, v):
        """Road point (x, y) in the ego frame for canvas pixel (u, v) (centre convention), or None above the horizon."""
        r = self.R @ np.array([(u - self.cx) / self.fx, (v - self.cy) / self.fy, 1.0])
        if r[2] >= -1e-6:
            return None
        lam = -(self.t[2] + self.g) / r[2]
        if lam * np.linalg.norm(r) > self.max_range:
            return None
        p = self.t + lam * r
        return float(p[0]), float(p[1])

    def read_box(self, box):
        """(x, y) of a detector box (pixel edges) from its bottom-centre, or None."""
        return self.ground((box[0] + box[2]) / 2 - 0.5, box[3] - 0.5 + self.dv)

    def project(self, P):
        """Ego-frame points (N, 3) -> canvas pixels (centre convention) and camera depth."""
        Q = (np.asarray(P, float) - self.t) @ self.R          # camera coordinates (x right, y down, z forward)
        return np.stack([self.fx * Q[:, 0] / Q[:, 2] + self.cx, self.fy * Q[:, 1] / Q[:, 2] + self.cy], 1), Q[:, 2]


def geometry_of(window, cfg):
    return Geometry(window['intrinsics']['K_canvas'], window['intrinsics']['ego_SE3_cam'], cfg.get('ground_below_ego_m', GROUND_BELOW_EGO_M),
                    cfg.get('dv', 0.0))


# -------------------------------------------------------------------------------------------------------- utilities
def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1); u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def centre(b):
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def predicted(history, frame, n):
    """Constant-velocity box at `frame` from the last n followed (frame, box) pairs (centre, width, height). (pilot copy)"""
    if n < 2 or len(history) < 2:
        return list(history[-1][1])
    (f0, b0), (f1, b1) = history[-min(n, len(history))], history[-1]
    if f1 == f0:
        return list(b1)
    q0 = np.array([(b0[0] + b0[2]) / 2, (b0[1] + b0[3]) / 2, b0[2] - b0[0], b0[3] - b0[1]]); q1 = np.array([(b1[0] + b1[2]) / 2, (b1[1] + b1[3]) / 2, b1[2] - b1[0], b1[3] - b1[1]])
    q = q1 + (q1 - q0) / (f1 - f0) * (frame - f1); q[2:] = np.maximum(q[2:], 2.0)
    return [q[0] - q[2] / 2, q[1] - q[3] / 2, q[0] + q[2] / 2, q[1] + q[3] / 2]


def edge_of(box, cfg):
    if box[0] <= cfg['edge_x'][0]:
        return 'left'
    if box[2] >= cfg['edge_x'][1]:
        return 'right'
    if box[3] >= cfg['edge_bottom']:
        return 'bottom'
    return None


def plausible_box(d, cfg, classes):
    b, sc, lab = d[:4], d[4], int(d[5])
    w, h = b[2] - b[0], b[3] - b[1]
    return lab in classes and sc >= cfg['score'] and h > cfg['min_h'] and h <= cfg['max_h'] and 1 / cfg['max_aspect'] <= w / h <= cfg['max_aspect_w']


class Place:
    """(range d from the camera, lateral y, x in the ego frame) of a box; cached per box."""

    def __init__(self, geo: Geometry):
        self.geo = geo; self.cache = {}

    def __call__(self, b):
        k = tuple(round(v, 3) for v in b[:4])
        if k not in self.cache:
            p = self.geo.read_box(b)
            self.cache[k] = None if p is None else (p[0] - self.geo.t[0], p[1], p[0])
        return self.cache[k]


def row(t, state, box=None, score=None, p=None, **extra):
    r = dict(t=t, state=state)
    if box is not None:
        r['box'] = [round(v, 2) for v in box[:4]]
    if score is not None or box is not None:
        r['score'] = score
    if p is not None:
        r['x'] = round(p[2], 4); r['y'] = round(p[1], 4)
    r.update(extra)
    return r


# --------------------------------------------------------------------------------------------------------- trackers
def predicted_exit(boxes, frame, cfg):
    """With cfg.predict_exit: the constant-velocity box (last cfg.predict_exit followed boxes) at `frame` touches an
    edge limit -> that edge (the target is taken to have left the canvas while undetected), else None."""
    n = cfg.get('predict_exit', 0)
    if not n or len(boxes) < 2:
        return None, None
    pb = predicted(boxes, frame, n)
    return edge_of(pb, cfg), pb


def follow_iou(dets_per_tick, init_box, cfg, ticks, place: Place, classes, t0_ok=None, prior=None):
    """study-v1 'follow' (v3 / v5): IoU association + size and kinematic gates; copied, generalised (classes, metric
    gates on the flat-road range and lateral position of this window's camera)."""
    prev, misses, track = list(init_box), 0, []
    prev_ground, history = None, [(f, list(b)) for f, b in (prior or [])]
    for i, dets in enumerate(dets_per_tick):
        t = round((ticks[i] - T0_FRAME) / FPS, 1); widen = min(1 + misses, cfg.get('max_widen', 1e9)); steps = abs(ticks[i] - ticks[i - 1 - misses]) if i > misses else 1
        ref = predicted(history, ticks[i], cfg.get('predict', 0)) if i > 0 and history and cfg.get('predict') else prev
        if misses > cfg['max_misses'] or (track and track[-1]['state'] in ('edge', 'after_edge')):
            track.append(row(t, 'after_edge' if track and track[-1]['state'] in ('edge', 'after_edge') else 'lost')); continue
        gsteps = min(steps, cfg.get('gate_steps_cap', 1e9))   # cfg.gate_steps_cap: gates stop widening after that many frames
        kin_ref = prev_ground
        # cfg.kin_boot_yaw (rad per frame): while the tracker has no velocity estimate (fewer than 2 followed boxes), the
        # lateral gate also allows the bearing change of an ego turning at up to that rate (d x yaw x frames)
        boot_yaw = cfg.get('kin_boot_yaw', 0.0) if len(history) < 2 else 0.0
        if cfg.get('kin_on_prediction') and i > 0 and prev_ground is not None and len(history) >= 2:
            # cfg.kin_on_prediction: the kinematic gates are centred on the ground point of the constant-velocity box
            # (study-v1 v5 centred them on the last followed point, which fails when the ego turns or the target moves fast)
            gp = place(ref)
            kin_ref = (gp[0], gp[1]) if gp is not None else prev_ground
        ok = []
        for d in dets:
            if not plausible_box(d, cfg, classes):
                continue
            b, s = d[:4], d[4]; h = b[3] - b[1]
            if i == 0:
                scale = cfg['init_scale']
            elif 'scale_gate' in cfg:
                scale = math.exp(math.log(cfg['scale_gate']['base']) + cfg['scale_gate']['per_miss'] * misses)
            else:
                scale = cfg['max_scale'] ** widen
            if not (1 / scale <= h / max(ref[3] - ref[1], 1e-6) <= scale):
                continue
            if cfg.get('kin_gate2') and i > 0 and kin_ref is not None:
                g = place(b); kg = cfg['kin_gate2']
                if g is None or abs(g[0] - kin_ref[0]) > kg['dist_rel'] * kin_ref[0] + kg['dist_per_frame'] * gsteps:
                    continue
                if abs(g[1] - kin_ref[1]) > kg['lateral_abs'] + kg['lateral_per_frame'] * gsteps + boot_yaw * kin_ref[0] * gsteps:
                    continue
            if cfg.get('kin_gate') and i > 0 and kin_ref is not None:
                g = place(b); kg = cfg['kin_gate']
                if g is None or abs(g[0] - kin_ref[0]) > max(kg['dist_abs'], kg['dist_rel'] * kin_ref[0]) * gsteps:
                    continue
                if abs(g[1] - kin_ref[1]) > kg['lateral'] * gsteps:
                    continue
            ok.append((b, s))
        if i == 0 and t0_ok is not None:
            ok = [d for d in ok if t0_ok(d[0])]
        best = max(ok, key=lambda d: iou(d[0], ref), default=None)
        if best is not None and iou(best[0], ref) < cfg['iou_min']:
            cx_, cy_ = centre(ref); near = min(ok, key=lambda d: np.hypot(centre(d[0])[0] - cx_, centre(d[0])[1] - cy_))
            best = near if np.hypot(centre(near[0])[0] - cx_, centre(near[0])[1] - cy_) <= cfg['gate_px'] * widen else None
        if best is None and i == 0:
            best = (list(init_box), None)
        if best is None:
            pe, pb = predicted_exit(history, ticks[i], cfg)
            if pe:
                track.append(row(t, 'edge', pb, None, edge='pred-' + pe)); continue
            misses += 1; track.append(row(t, 'miss')); continue
        misses = 0; prev = list(best[0])
        if best[1] is not None:
            history.append((ticks[i], list(prev)))
        g = place(prev)
        prev_ground = (g[0], g[1]) if (g is not None and best[1] is not None) else None
        e = edge_of(prev, cfg) if i > 0 else None
        if e:
            track.append(row(t, 'edge', prev, best[1], edge=e)); continue
        track.append(row(t, 'followed' if best[1] is not None else 'anchor_box', prev, best[1], g))
    return track


def follow_pixel(dets_per_tick, init_box, cfg, ticks, place: Place, classes, t0_ok=None, prior=None):
    """study-v1 'follow_pixel' (v8): bottom-centre (u, v) and log height, constant-velocity prediction, pixel gates;
    copied, generalised (classes; reading = this window's flat-road point)."""
    pg = cfg['pixel']; track, misses = [], 0
    state = lambda b: ((b[0] + b[2]) / 2, b[3], math.log(max(b[3] - b[1], 1.0)))
    boxes = [(f, list(b)) for f, b in (prior or [])]; hist = [(f, state(b)) for f, b in boxes]
    for i, dets in enumerate(dets_per_tick):
        t = round((ticks[i] - T0_FRAME) / FPS, 1)
        if misses > cfg['max_misses'] or (track and track[-1]['state'] in ('edge', 'after_edge')):
            track.append(row(t, 'after_edge' if track and track[-1]['state'] in ('edge', 'after_edge') else 'lost')); continue
        ok = [(d[:4], d[4]) for d in dets if plausible_box(d, cfg, classes)]
        best = None
        if i == 0 and t0_ok is not None:
            ok = [d for d in ok if t0_ok(d[0])]
        if i == 0:
            m0 = max(ok, key=lambda d: iou(d[0], init_box), default=None)
            best = m0 if m0 is not None and iou(m0[0], init_box) >= cfg['iou_min'] and 1 / cfg['init_scale'] <= (m0[0][3] - m0[0][1]) / (init_box[3] - init_box[1]) <= cfg['init_scale'] else (list(init_box), None)
        else:
            f_last, s_last = hist[-1]; k = ticks[i] - f_last; recent = hist[-cfg['predict']:]
            if len(recent) >= 2:
                v = [float(np.clip(np.median([(b[1][j] - a[1][j]) / (b[0] - a[0]) for a, b in zip(recent, recent[1:])]), -lim, lim)) for j, lim in enumerate((pg['v_max'], pg['v_max'], pg['vlogh_max']))]
            else:
                v = list(pg.get('v_init', [0.0, 0.0, 0.0]))
            pred = [s_last[j] + v[j] * k for j in range(3)]
            ka = min(abs(k), cfg.get('gate_steps_cap', 1e9))
            # pixel.size_frac: the absolute gates are at least that fraction of the last followed box's width / height
            # (near objects move more pixels; the pilot's fixed 3 px suited leads of one size range)
            lb = boxes[-1][1] if boxes else init_box
            fx_, fy_ = pg.get('size_frac', 0.0) * (lb[2] - lb[0]), pg.get('size_frac', 0.0) * (lb[3] - lb[1])
            gx, gy, gh = max(pg['x_abs'], fx_) + pg['x_per_frame'] * ka, max(pg['y_abs'], fy_) + pg['y_per_frame'] * ka, math.log(pg['h_ratio'])
            scored = []
            for b, sc in ok:
                q = state(b); dx, dy, dh = q[0] - pred[0], q[1] - pred[1], q[2] - pred[2]
                if abs(dx) <= gx and abs(dy) <= gy and abs(dh) <= gh:
                    scored.append(((dx / gx) ** 2 + (dy / gy) ** 2 + (dh / gh) ** 2, -sc, b, sc))
            if scored:
                _, _, b, sc = min(scored, key=lambda z: (z[0], z[1])); best = (b, sc)
        if best is None:
            pe, pb = predicted_exit(boxes, ticks[i], cfg)
            if pe:
                track.append(row(t, 'edge', pb, None, edge='pred-' + pe)); continue
            misses += 1; track.append(row(t, 'miss')); continue
        misses = 0; box = list(best[0])
        if best[1] is not None or not hist:
            hist.append((ticks[i], state(box)))
        if best[1] is not None:
            boxes.append((ticks[i], list(box)))
        e = edge_of(box, cfg) if i > 0 else None
        if e:
            track.append(row(t, 'edge', box, best[1], edge=e)); continue
        track.append(row(t, 'followed' if best[1] is not None else 'anchor_box', box, best[1], place(box)))
    return track


TRACKERS = {'iou': follow_iou, 'pixel': follow_pixel}


def t0_check(tgt, place: Place, chk):
    """cfg.t0_check: a t0 detection must agree with the target's annotated t0 position (window.json: centre x_m, y_m,
    size and |heading|; t0 information only). Its flat-road reading x must lie in [x_near - x_abs - x_rel x_near,
    x_c + x_abs + x_rel x_c] and |y - y_c| <= y_abs + the box's half extent across + y_rel x_c."""
    L, W = tgt['length_m'], tgt['width_m']; psi = math.radians(tgt.get('heading_diff_deg') or 0.0)
    xc, yc = tgt['x_m'], tgt['y_m']
    hx = L / 2 * abs(math.cos(psi)) + W / 2 * abs(math.sin(psi)); hy = L / 2 * abs(math.sin(psi)) + W / 2 * abs(math.cos(psi))
    xn = xc - hx
    lo, hi = xn - chk['x_abs'] - chk['x_rel'] * xn, xc + chk['x_abs'] + chk['x_rel'] * xc
    ytol = chk['y_abs'] + hy + chk['y_rel'] * xc

    def ok(b):
        p = place(b)
        return p is not None and lo <= p[2] <= hi and abs(p[1] - yc) <= ytol
    return ok


# --------------------------------------------------------------------------------------------------------- outcomes
def exit_of(track):
    """Exit time and kind: first edge stop, or the first miss of the final run of misses that ended in a loss."""
    for j, r in enumerate(track):
        if r['state'] == 'edge':
            return dict(t=r['t'], kind='edge-' + r.get('edge', '?'))
        if r['state'] == 'lost':
            k = j
            while k > 0 and track[k - 1]['state'] == 'miss':
                k -= 1
            return dict(t=track[k]['t'], kind='lost')
    return None


def tracker_outcomes(track, family):
    """All outcome values of one tracker's track (None where no reading exists)."""
    reads = [r for r in track if 'x' in r and r['state'] in ('followed', 'anchor_box')]
    ex = exit_of(track)

    def at(h, key):
        r = next((r for r in track if abs(r['t'] - h) < 1e-6), None)
        if r is not None and key in r and r['state'] in ('followed', 'anchor_box'):
            return r[key], 'followed'
        upto = [r for r in reads if r['t'] <= h + 1e-6]
        return (upto[-1][key], 'carried') if upto else (None, 'none')

    out = {}
    for name, _ in OUTCOMES[family]:
        if name.startswith(('x_', 'y_', 'absy_')) and name.endswith('s') and name.count('_') == 1:
            key, h = name.split('_'); h = float(h[:-1])
            v, st = at(h, 'y' if key in ('y', 'absy') else 'x')
            out[name] = dict(value=None if v is None else (abs(v) if key == 'absy' else v), state=st)
        elif name == 'min_x_0_4s':
            vals = [r['x'] for r in reads if r['t'] <= 4.0 + 1e-6]
            out[name] = dict(value=min(vals) if vals else None, state='followed' if vals else 'none')
        elif name == 'entered_absy_lt_1p0_by_4s':
            vals = [abs(r['y']) for r in reads if 0 < r['t'] <= 4.0 + 1e-6]
            out[name] = dict(value=None if not reads else float(any(v < 1.0 for v in vals)), state='followed' if reads else 'none')
        elif name == 'in_absy_lt_1p5_within_2s':
            vals = [abs(r['y']) for r in reads if 0 < r['t'] <= 2.0 + 1e-6 and r['x'] > 0]
            out[name] = dict(value=None if not reads else float(any(v < 1.5 for v in vals)), state='followed' if reads else 'none')
        elif name == 'out_of_view_by_2s':
            out[name] = dict(value=None if not reads else float(ex is not None and ex['t'] <= 2.0 + 1e-6), state='followed' if reads else 'none')
        else:
            raise ValueError(name)
    return out, ex


def combine(per_tracker, family):
    """Ensemble = median over trackers of each outcome (binary: majority); the pilot's combine_wm_measures rule."""
    out = {}
    for name, _ in OUTCOMES[family]:
        vals = [per_tracker[k][name]['value'] for k in sorted(per_tracker)]
        states = [per_tracker[k][name]['state'] for k in sorted(per_tracker)]
        good = [v for v in vals if v is not None]
        value = float(np.median(good)) if len(good) >= 2 else None
        spread = None
        if name.startswith(('x_', 'min_x')) and len(good) >= 2 and min(good) > 0:
            spread = float(max(good) / min(good) - 1)
        elif name.startswith(('absy_', 'y_')) and len(good) >= 2:
            spread = float(max(good) - min(good))
        out[name] = dict(value=value, each=vals, states=states, spread=spread, n_carried=sum(s == 'carried' for s in states))
    return out


def ensemble_series(tracks):
    """Per frame: median over trackers of x, y (LOCF after a tracker stops), and the box of the median-x tracker."""
    n = len(next(iter(tracks.values())))
    series = []
    last = {k: None for k in tracks}
    for j in range(n):
        cur = {}
        for k, tr in tracks.items():
            r = tr[j]
            if 'x' in r and r['state'] in ('followed', 'anchor_box'):
                last[k] = (r['x'], r['y'], r['box']); cur[k] = (r['x'], r['y'], r['box'], 'followed')
            elif last[k] is not None:
                cur[k] = (*last[k], 'carried')
        if len(cur) >= 2:
            xs = sorted(cur.items(), key=lambda kv: kv[1][0])
            mid = xs[len(xs) // 2] if len(xs) % 2 else xs[len(xs) // 2 - 1]
            series.append(dict(t=tracks[next(iter(tracks))][j]['t'], x=float(np.median([c[0] for c in cur.values()])),
                               y=float(np.median([c[1] for c in cur.values()])), box=mid[1][2], box_tracker=mid[0],
                               followed=sum(c[3] == 'followed' for c in cur.values())))
        else:
            series.append(dict(t=tracks[next(iter(tracks))][j]['t'], x=None, y=None, box=None, followed=0))
    return series


# --------------------------------------------------------------------------------------------------------- measuring
def tracker_cfg(cfg, tracker, gate_class):
    base = dict(cfg['common']); base.update(cfg['trackers'][tracker])
    over = cfg.get('family_overrides', {}).get(gate_class, {})
    for k, v in over.get('common', {}).items():
        base[k] = v
    for k, v in over.get(tracker, {}).items():
        base[k] = (dict(base.get(k, {}), **v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v)
    return base


def measure_clip(window, dets, cfg, families=None):
    geo = geometry_of(window, cfg); place = Place(geo)
    hcfg = dict(HISTORY_DEFAULT, **cfg.get('history', {}))
    frames = list(range(T0_FRAME, 81))
    per_frame = [dets['detections'][f] for f in frames]
    out = {}
    for fam, tgt in window['targets'].items():
        if families and fam not in families or fam not in FAMILY_KIND:
            continue
        classes = CLASSES[FAMILY_KIND[fam]]
        gate_class = cfg.get('gate_class', {}).get(fam, fam)
        tracks, outs, exits, backs = {}, {}, {}, {}
        for name in cfg['trackers']:
            tc = tracker_cfg(cfg, name, gate_class)
            chk = t0_check(tgt, place, tc['t0_check']) if tc.get('t0_check') else None
            prior = None
            # (1) history warm-up (instrument v2, validated): the same tracker run backwards from t0 over the last
            # cfg.warmup history frames (20 -> 15; part of the conditioning, the same for every clip of a window), without
            # exit prediction; its followed boxes, in forward order, seed the forward run's velocity and exit prediction.
            if tc.get('warmup'):
                bf = list(range(T0_FRAME, T0_FRAME - tc['warmup'] - 1, -1))
                wb = TRACKERS[tc['kind']]([dets['detections'][f] for f in bf], tgt['box_canvas'], dict(tc, predict_exit=0), bf, place, classes, t0_ok=chk)
                if wb[0]['state'] == 'followed':
                    prior = [(f, r['box']) for f, r in zip(bf[1:], wb[1:]) if r['state'] == 'followed'][::-1]
            # (2) history readings for the constant-velocity baseline (protocol 5): a separate backward run over the whole
            # real history, frames 20 -> 0 (t = 0 .. -2.0 s), with cfg.history.predict_exit; it does not feed the forward run.
            hf = list(range(T0_FRAME, T0_FRAME - hcfg['frames'] - 1, -1))
            backs[name] = TRACKERS[tc['kind']]([dets['detections'][f] for f in hf], tgt['box_canvas'], dict(tc, predict_exit=hcfg.get('predict_exit', 0)),
                                               hf, place, classes, t0_ok=chk)
            tr = TRACKERS[tc['kind']](per_frame, tgt['box_canvas'], tc, frames, place, classes, t0_ok=chk, prior=prior)
            tracks[name] = tr
            outs[name], exits[name] = tracker_outcomes(tr, fam)
        t0_found = {k: tr[0]['state'] == 'followed' for k, tr in tracks.items()}
        hist = history_readings(backs, hcfg)
        out[fam] = dict(gate_class=gate_class, t0_found=t0_found, exits=exits, t0_box_at_edge=edge_of(tgt['box_canvas'], cfg['common']) is not None,
                        history=hist, trackers={k: dict(track=v, outcomes=outs[k], history_track=backs[k]) for k, v in tracks.items()},
                        ensemble=dict(outcomes=combine(outs, fam), series=ensemble_series(tracks)))
    return out


HISTORY_DEFAULT = dict(frames=20, predict_exit=0, min_trackers=2, min_frames=5)


def history_readings(backs, hcfg):
    """History readings of the backward run (t = 0, -0.1, ..., -2.0 s): per frame, the median over the trackers that
    follow the target there (at least hcfg.min_trackers), and the least-squares line x(t) = x0 + vx t, y(t) = y0 + vy t
    through those per-frame readings (protocol 5's constant-velocity baseline; needs >= hcfg.min_frames frames)."""
    names = sorted(backs)
    n = len(backs[names[0]])
    series = []
    for j in range(n):
        fol = [backs[k][j] for k in names if backs[k][j]['state'] == 'followed' and 'x' in backs[k][j]]
        t = backs[names[0]][j]['t']
        if len(fol) >= hcfg['min_trackers']:
            series.append(dict(t=t, x=float(np.median([r['x'] for r in fol])), y=float(np.median([r['y'] for r in fol])), n=len(fol),
                               box=fol[0]['box'] if len(fol) == 1 else sorted(fol, key=lambda r: r['x'])[(len(fol) - 1) // 2]['box']))
        else:
            series.append(dict(t=t, x=None, y=None, n=len(fol), box=None))
    pts = [s_ for s_ in series if s_['x'] is not None]
    fit = None
    if len(pts) >= hcfg['min_frames'] and series[0]['x'] is not None:
        T = np.array([p['t'] for p in pts]); X = np.array([p['x'] for p in pts]); Y = np.array([p['y'] for p in pts])
        A = np.stack([np.ones_like(T), T], 1)
        (x0, vx), *_ = np.linalg.lstsq(A, X, rcond=None); (y0, vy), *_ = np.linalg.lstsq(A, Y, rcond=None)
        fit = dict(x0=float(x0), vx=float(vx), y0=float(y0), vy=float(vy), n=len(pts), t_first=float(T.min()),
                   rms_x=float(np.sqrt(np.mean((X - x0 - vx * T) ** 2))), rms_y=float(np.sqrt(np.mean((Y - y0 - vy * T) ** 2))))
    return dict(series=series, fit=fit, x_0s=series[0]['x'], y_0s=series[0]['y'])


def clip_list(bench: Path, names, sources):
    det = bench / 'detections'
    for name in names:
        for source in sources:
            if source in ('canvas-raw', 'nano-raw'):
                yield name, source, 'raw', det / source / f'{name}.json'
            else:
                for f in sorted((det / source).glob(f'{name}-*.json')):
                    sample = f.stem[len(name) + 1:]
                    yield name, source, ('real' if sample == 'recon' else sample.lstrip('s')), f


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cmd_measure(args):
    cfg = json.loads(args.config.read_text())
    names = [l.strip() for l in args.names_file.read_text().splitlines() if l.strip()]
    dev = set(l.strip() for l in (args.bench / 'names-dev.txt').read_text().splitlines() if l.strip())
    if not set(names) <= dev and not (args.allow_test and (args.bench / 'instrument-frozen.txt').is_file()):
        sys.exit('refusing: names outside names-dev.txt (DEV only until protocol-v1.1 is frozen)')
    if set(names) & dev and not set(names) <= dev:
        sys.exit('refusing: a names file must not mix DEV and TEST windows')
    windows_dir = args.bench / ('windows-dev' if set(names) <= dev else 'windows-test')
    fams = set(args.families.split(',')) if args.families else None
    args.out.mkdir(parents=True, exist_ok=True)
    for source in args.sources:
        res = dict(config=cfg, config_file=str(args.config), config_sha256=sha256(args.config), script_sha256=sha256(__file__), source=source, windows={})
        n = 0
        for name, src, sample, f in clip_list(args.bench, names, [source]):
            if not f.is_file():
                continue
            window = json.loads((windows_dir / name / 'window.json').read_text())
            dets = json.loads(f.read_text())
            res['windows'].setdefault(name, {})[sample] = dict(detections=str(f), families=measure_clip(window, dets, cfg, fams))
            n += 1
        (args.out / f'measure-{source}.json').write_text(json.dumps(res) + '\n')
        print(json.dumps(dict(source=source, clips=n, windows=len(res['windows']))), flush=True)


# ------------------------------------------------------------------------------------------------------------ export
EGO_OUTCOMES = [('ego_dx_0_4s', 'primary', '4.0', 'D_m'), ('ego_dyaw_deg_0_4s', 'secondary', '4.0', 'Psi_deg'),
                ('ego_dx_0_2s', 'secondary', '2.0', 'D_m'), ('ego_dx_0_6s', 'secondary', '6.0', 'D_m')]
FIELDS = ['window', 'family', 'sample', 'outcome', 'kind', 'value', 'scorable', 'flags', 'value_v3', 'value_v5', 'value_v8']


def cmd_export(args):
    """One CSV row per (window, family, sample, outcome).
    sample: k for generated clips, 'real' for the recon reference, 'raw' for the raw real clip (--raw-source, sensitivity
    analysis only). kind: primary / secondary / descriptive, or 'history' (inputs of the constant-velocity baseline, real
    and raw samples only). value: the instrument's reading (median of the three trackers); value_v3/v5/v8: each tracker
    alone (protocol 5, sensitivity). scorable: 0/1 by the rules of protocol-v1.1-instrument.md. flags: ';'-separated."""
    gen = json.loads((args.measure / f'measure-{args.run}.json').read_text())
    # --real-run none: no reference run (e.g. exporting DEV raw readings for the climatology baseline); each clip is then
    # its own reference for the t0 rule
    real = {'windows': {}} if args.real_run == 'none' else json.loads((args.measure / f'measure-{args.real_run}.json').read_text())
    raw = json.loads((args.measure / f'measure-{args.raw_source}.json').read_text()) if args.raw_source else None
    rules = json.loads(args.rules.read_text()) if args.rules else {}
    ego = rules.get('ego')
    rows, hrows = [], []
    names = sorted(set(gen['windows']) | set(real['windows']))
    for name in names:
        samples = dict(gen['windows'].get(name, {}))
        if 'real' in real['windows'].get(name, {}):
            samples['real'] = real['windows'][name]['real']
        if raw is not None and 'raw' in raw['windows'].get(name, {}):
            samples['raw'] = raw['windows'][name]['raw']
        real_fams = real['windows'].get(name, {}).get('real', {}).get('families', {})
        order = sorted(samples, key=lambda x: (x in ('real', 'raw'), x == 'raw', x.zfill(4)))
        for sample in order:
            if args.real_run == 'none':
                real_fams = samples[sample]['families']
            for fam, rec in samples[sample]['families'].items():
                for oname, kind in OUTCOMES[fam]:
                    o = rec['ensemble']['outcomes'][oname]
                    flags, scorable = scorability(fam, rec, real_fams.get(fam), oname, rules)
                    each = dict(zip(sorted(rec['trackers']), o['each']))
                    rows.append(dict(window=name, family=fam, sample=sample, outcome=oname, kind=kind,
                                     value='' if o['value'] is None else round(o['value'], 4), scorable=int(scorable), flags=';'.join(flags),
                                     **{f'value_{k}': ('' if each.get(k) is None else round(each[k], 4)) for k in ('v3', 'v5', 'v8')}))
                if sample in ('real', 'raw'):
                    rows += history_rows(name, fam, sample, rec, real_fams.get(fam), rules)
                    hrows += history_frame_rows(name, fam, sample, rec)
        if ego:
            wdir = args.bench / ego.get('windows_dir', 'windows-dev') / name
            if 'ego_inter' in json.loads((wdir / 'window.json').read_text())['roles']:
                for sample in order:
                    src, key = {'real': (args.real_run, 'real'), 'raw': (args.raw_source or args.run, 'raw')}.get(sample, (args.run, sample))
                    rows += ego_rows(args.bench / ego['clips_dir'] / src / f'{name}__{key}.json', name, sample, ego,
                                     args.bench / ego.get('hist_dir', 'instrument/ego-vo/clips-hist') / src / f'{name}__{key}.json')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader(); w.writerows(rows)
    if args.history_out:
        with args.history_out.open('w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=['window', 'family', 'sample', 't', 'x', 'y', 'n_trackers'])
            w.writeheader(); w.writerows(hrows)
    print(json.dumps(dict(rows=len(rows), scorable=sum(r['scorable'] for r in rows), out=str(args.out), history_rows=len(hrows))))


def history_rows(name, fam, sample, rec, real_rec, rules):
    """Inputs of the constant-velocity baseline (protocol 5): the least-squares line through the instrument's
    per-frame readings of the target over the real history (t = -2.0 .. 0 s, frames 0-20, read backwards from t0 by
    the same three trackers; per frame the median of the trackers following there): intercept at t0 and slope (m/s) of
    x and y, the number of frames used and the earliest one; plus the t0 reading itself."""
    h = rec.get('history') or {}
    fit = h.get('fit')
    flags, scorable = scorability(fam, rec, real_rec, OUTCOMES[fam][0][0], rules)
    flags = [f for f in flags if not f.startswith('carried')]
    if fit is None:
        flags.append('no_history_fit')
    elif fit['t_first'] > -2.0 + 1e-6:
        flags.append(f"history_span={-fit['t_first']:.1f}s")
    if fam in rules.get('history_not_reliable', []):
        flags.append('history_not_reliable')
    ok = int(scorable and fit is not None and fam not in rules.get('history_not_reliable', []))
    vals = [('hist_x_0s', h.get('x_0s')), ('hist_y_0s', h.get('y_0s'))]
    vals += [(f'hist_ls_{k}', None if fit is None else fit[k]) for k in ('x0', 'vx', 'y0', 'vy', 'n', 't_first')]
    return [dict(window=name, family=fam, sample=sample, outcome=oname, kind='history', value='' if v is None else round(v, 4),
                 scorable=ok, flags=';'.join(flags), value_v3='', value_v5='', value_v8='') for oname, v in vals]


def history_frame_rows(name, fam, sample, rec):
    """Per-frame history readings (one row per frame t = 0 .. -2.0 s) for --history-out."""
    return [dict(window=name, family=fam, sample=sample, t=r['t'], x='' if r['x'] is None else round(r['x'], 4),
                 y='' if r['y'] is None else round(r['y'], 4), n_trackers=r['n']) for r in (rec.get('history') or {}).get('series', [])]


def ego_rows(f, name, sample, ego, fh=None):
    vrec = json.loads(f.read_text()) if f.is_file() else None
    out = []
    for oname, kind, hh, field in EGO_OUTCOMES:
        flags = []
        val = None if vrec is None else vrec['planar']['horizons'][hh][field]
        if vrec is None:
            flags.append('no_vo_record')
        elif (vrec.get('quality') or {}).get('accepted_share', 1.0) < 0.5:
            flags.append('vo_low_support')
        if not ego['scorable'].get(oname, False):
            flags.append('vo_gate_failed')
        out.append(dict(window=name, family='ego_inter', sample=sample, outcome=oname, kind=kind, value='' if val is None else round(val, 4),
                        scorable=int(val is not None and ego['scorable'].get(oname, False)), flags=';'.join(flags), value_v3='', value_v5='', value_v8=''))
    if sample in ('real', 'raw') and vrec is not None:
        g0 = vrec['groups'][0]                     # the history group, frames 16 -> 20 (0.4 s)
        th = g0['passes'][0]['theta'] if g0.get('accepted_m') is not None and g0['passes'][0]['theta'] is not None else g0['init_theta']
        import math as _m
        R_ec = np.asarray(vrec['camera']['ego_R_cam'], float)
        yaw = None
        if th is not None:
            from scipy.spatial.transform import Rotation
            Rc = Rotation.from_rotvec(np.asarray(th[3:6], float)).as_matrix().T
            Re = R_ec @ Rc @ R_ec.T; yaw = _m.degrees(_m.atan2(Re[1, 0], Re[0, 0]))
        for oname, v in (('ego_hist_dx_0p4s', g0['final_m']), ('ego_hist_dyaw_deg_0p4s', yaw)):
            out.append(dict(window=name, family='ego_inter', sample=sample, outcome=oname, kind='history', value='' if v is None else round(v, 4),
                            scorable=int(v is not None and ego['scorable'].get('ego_dx_0_4s', False)), flags='history_group_16_20',
                            value_v3='', value_v5='', value_v8=''))
    if sample in ('real', 'raw'):
        hrec = json.loads(fh.read_text()) if fh is not None and fh.is_file() else None
        for oname, key in (('ego_hist_ls_speed', 'ls_speed_mps'), ('ego_hist_ls_yaw_rate_dps', 'ls_yaw_rate_dps')):
            v = None if hrec is None else hrec['history'][key]
            out.append(dict(window=name, family='ego_inter', sample=sample, outcome=oname, kind='history', value='' if v is None else round(v, 4),
                            scorable=int(v is not None and ego.get('history_ok', False)),
                            flags='history_odometry_2s_not_gated' + ('' if hrec is not None else ';no_history_record'),
                            value_v3='', value_v5='', value_v8=''))
    return out


def scorability(fam, rec, real_rec, oname, rules):
    """Scorability of one (window, family, sample, outcome) (protocol-v1.1-instrument.md, section Scorability). Only t0
    information and the clip's own track enter, never the real future:
      t0_not_found       fewer than 2 of the 3 trackers matched a detection at t0 that passes the t0 position check;
      real_t0_not_found  the same for the real reference of the window (the whole window-family is unscorable);
      t0_box_at_edge     the target's t0 box touches the edge limits (it is leaving the canvas at t0);
      no_value           fewer than 2 trackers have any reading;
      family_not_reliable / outcome_not_reliable   DEV validation found the family / outcome unreliable;
      carried            (flag only) at least 2 trackers carry their last reading (target out of view or lost)."""
    flags = []
    if sum(rec['t0_found'].values()) < 2:
        flags.append('t0_not_found')
    if real_rec is None or sum(real_rec['t0_found'].values()) < 2:
        flags.append('real_t0_not_found')
    if rec.get('t0_box_at_edge'):
        flags.append('t0_box_at_edge')
    o = rec['ensemble']['outcomes'][oname]
    if o['value'] is None:
        flags.append('no_value')
    if o.get('n_carried', 0) >= 2:
        flags.append('carried')
        kinds = [(rec['exits'].get(k) or {}).get('kind', '') for k, st in zip(sorted(rec['trackers']), o['states']) if st == 'carried']
        flags.append('carried_lost' if sum(k == 'lost' for k in kinds) * 2 > len(kinds) else 'carried_edge')
    if fam in rules.get('family_not_reliable', []):
        flags.append('family_not_reliable')
    if oname in rules.get('outcome_not_reliable', {}).get(fam, []):
        flags.append('outcome_not_reliable')
    hard = {'t0_not_found', 'real_t0_not_found', 't0_box_at_edge', 'no_value', 'family_not_reliable', 'outcome_not_reliable'}
    return flags, not (set(flags) & hard)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('measure')
    p.add_argument('--bench', type=Path, required=True); p.add_argument('--names-file', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True); p.add_argument('--sources', nargs='+', required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--families', default='')
    p.add_argument('--allow-test', action='store_true', help='TEST windows; refused unless BENCH/instrument-frozen.txt exists (protocol-v1.1 frozen)')
    p = sub.add_parser('export')
    p.add_argument('--bench', type=Path, required=True); p.add_argument('--measure', type=Path, required=True)
    p.add_argument('--run', required=True); p.add_argument('--real-run', required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--rules', type=Path, default=None)
    p.add_argument('--raw-source', default=None, help="also export the raw real clip as sample 'raw' (e.g. nano-raw)")
    p.add_argument('--history-out', type=Path, default=None, help='CSV of the per-frame history readings (real / raw samples)')
    args = ap.parse_args()
    {'measure': cmd_measure, 'export': cmd_export}[args.cmd](args)


if __name__ == '__main__':
    main()
