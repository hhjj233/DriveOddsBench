#!/usr/bin/env python3
"""Shared canvas instrument (canvas-v1, 2026-11-03): bench-v1 instrument v2 on the canvas geometry and frame rate of a
non-Nano model (Orbis 2, LTX-2.5 distilled, Epona). New code only: the frozen scripts are imported, never modified.

What changes per model (models/<model>.json, the "spec"):
  - frame rate: fps 10 (81 frames, t0 = frame 20) or 5 (41 frames, t0 = frame 10). The frozen trackers are called with
    10 Hz tick numbers (tick = 20 + (frame - t0_frame) * 10 / fps), so their times t, their step-scaled gates (kinematic
    gates, per-frame pixel gates, velocities per tick) and their constant-velocity predictions are per second
    automatically. Parameters counted in frames (misses, warm-up, prediction depth, history length) and the per-step
    gates of v3 / v5 are converted explicitly (effective_config; the table is written into the effective config);
  - geometry: band_rows = the source rows [r0, r1) that the canvas covers at the bench-v1 scale s = 512 / 1550, so
    K_band = K_source through the band affine (x' = s x + 0.5 s - 0.5, y' = s (y - r0) + 0.5 s - 0.5; for the bench
    band [797, 1669) this is window.json K_canvas); valid_canvas_rows = the canvas rows that carry picture (Epona's
    adapter writes rows [216, 288) black); the bottom edge limit is the last valid row minus the frozen 4 px margin;
  - gt (DEV validation only): projected boxes and in-band flags recomputed for the band and its valid rows, with the
    frozen rules of make_benchmark_windows.band_flags (2 px margin, height >= 97 source px, >= 5 LiDAR points);
  - prefix: 'shared' (frames 0..t0 of every clip of a window are the same pixels as the reference's: Orbis 2, Epona)
    or 'per_clip' (LTX-2.5: every clip decodes its own history; each clip gets its own t0 association and warm-up, as
    the frozen measure always does, and the export adds the hard flag t0_mismatch, see measure_canvas.py).
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
import sys  # noqa: E402
sys.path.insert(0, str(SCRIPTS))
import measure_bench_v2 as mb  # noqa: E402  (frozen instrument v2, read only)

SRC_W, SRC_H = 1550, 2048
S = mb.CANVAS_W / SRC_W                      # 0.3303226: the canvas scale of every bench-v1 band
BENCH_BAND = (797, 1669)
EDGE_MARGIN_PX = 4.0                         # frozen: edge_bottom 284 = 288 - 4, edge_x [4, 508]
BAND_MARGIN_SRC_PX, MIN_H_SRC_PX, MIN_POINTS = 2.0, 97.0, 5   # make_benchmark_windows.band_flags (frozen)


def sha256_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------------------------------------- the spec
def load_spec(model_or_path) -> dict:
    """Model spec with derived fields: step (10 Hz ticks per frame), t0_frame, n_frames, ticks (per frame)."""
    p = Path(model_or_path)
    if not p.suffix:
        p = HERE / 'models' / f'{model_or_path}.json'
    spec = json.loads(p.read_text())
    fps = int(spec['fps'])
    if 10 % fps:
        raise ValueError(f'fps {fps}: must divide 10')
    step = 10 // fps
    first, last = spec['ticks']['first'], spec['ticks']['last']
    if spec['ticks']['step'] != step or first != -20 or last != 60:
        raise ValueError('ticks must be k = -20..60 at the model frame rate')
    n = (last - first) // step + 1
    if n != spec['clip_frames']:
        raise ValueError(f'clip_frames {spec["clip_frames"]} != {n}')
    band = list(spec.get('band_rows', BENCH_BAND))
    if band[1] - band[0] != 872:
        raise ValueError('a band is 872 source rows (the 512x288 canvas at s = 512/1550)')
    valid = list(spec.get('valid_canvas_rows', [0, mb.CANVAS_H]))
    vsrc = list(spec.get('valid_source_rows', band))
    spec.update(spec_file=str(p), spec_sha256=sha256_file(p), step=step, t0_frame=(0 - first) // step, n_frames=n,
                tick_of_frame=[mb.T0_FRAME + k for k in range(first, last + 1, step)], band_rows=band,
                valid_canvas_rows=valid, valid_source_rows=vsrc)
    return spec


def tick(spec, frame: int) -> int:
    """10 Hz tick number (20 + k) of a native frame index."""
    return mb.T0_FRAME + (frame - spec['t0_frame']) * spec['step']


def frame_of_tick(spec, t: int) -> int:
    f, r = divmod(t - mb.T0_FRAME, spec['step'])
    if r:
        raise ValueError(f'tick {t} is not on the {spec["fps"]} Hz grid')
    return spec['t0_frame'] + f


# ------------------------------------------------------------------------------------------------ effective config
def effective_config(frozen_cfg: dict, spec: dict) -> dict:
    """The frozen tracker config (instrument/config-v2.json) on the spec's frame rate and valid rows.

    10 Hz: unchanged except edge_bottom when the valid rows end above the canvas bottom. 5 Hz (one frame = two 10 Hz
    ticks; ticks keep the step-scaled gates per second):
      common.max_misses 10 -> 5            the same 1.0 s of misses
      common.max_widen 2 -> 1              at 10 Hz the v3 / v5 gates stop widening after 0.2 s (2 ticks); one 5 Hz
                                           frame is 0.2 s, so the 5 Hz gates (below) are already the capped ones
      v3/v5 gate_px 43.63 -> 87.26         nearest-centre gate per frame step: 2 ticks (exactly the 10 Hz gate at 0.2 s)
      v3 max_scale 1.25 -> 1.5625          height ratio per frame step: 1.25 per tick, compounded over 2 ticks
      v5 scale_gate base 1.15 -> 1.3225    the same compounding; per_miss 0.05 -> 0.10 (0.05 per missed tick)
      v5 / v8 predict 5 -> 3               constant-velocity fit over the same 0.4 s (5 frames at 10 Hz, 3 at 5 Hz)
      common.predict_exit 5 -> 3           the same 0.4 s
      common.warmup 5 -> 3                 0.5 s back from t0 at 10 Hz; 0.6 s at 5 Hz (0.5 s is not on the grid)
      history.frames 20 -> 10              the whole 2.0 s history
      history.min_frames 5 -> 3            the same 0.4 s minimum span of the line fit
    Kept: every absolute tolerance (scores, sizes, aspect, iou_min, init_scale, t0_check, kin_gate dist_abs / dist_rel /
    lateral per tick, kin_gate2 per-frame terms per tick, v8 x/y_abs, per-tick terms, h_ratio, v_max, vlogh_max,
    size_frac, kin_boot_yaw per tick)."""
    cfg = copy.deepcopy(frozen_cfg)
    conv = []
    vb = spec['valid_canvas_rows'][1]
    if vb < mb.CANVAS_H:
        old = cfg['common']['edge_bottom']
        cfg['common']['edge_bottom'] = float(vb) - EDGE_MARGIN_PX
        conv.append(f'common.edge_bottom {old} -> {cfg["common"]["edge_bottom"]} (valid canvas rows end at {vb})')
    if spec['step'] == 2:
        def setp(path, new):
            d = cfg
            for k in path[:-1]:
                d = d[k]
            old = d[path[-1]]; d[path[-1]] = new
            conv.append(f'{".".join(path)} {old} -> {new}')
        c, tr = cfg['common'], cfg['trackers']
        setp(['common', 'max_misses'], int(round(c['max_misses'] / 2)))
        setp(['common', 'max_widen'], 1)
        setp(['common', 'predict_exit'], 3)
        setp(['common', 'warmup'], 3)
        for k in ('v3', 'v5'):
            setp(['trackers', k, 'gate_px'], round(tr[k]['gate_px'] * 2, 4))
        setp(['trackers', 'v3', 'max_scale'], round(tr['v3']['max_scale'] ** 2, 6))
        setp(['trackers', 'v5', 'scale_gate', 'base'], round(tr['v5']['scale_gate']['base'] ** 2, 6))
        setp(['trackers', 'v5', 'scale_gate', 'per_miss'], round(tr['v5']['scale_gate']['per_miss'] * 2, 6))
        for k in ('v5', 'v8'):
            setp(['trackers', k, 'predict'], 3)
        h = cfg.setdefault('history', {})
        setp(['history', 'frames'], 10)
        setp(['history', 'min_frames'], 3)
        if cfg.get('family_overrides') or cfg.get('gate_class'):
            raise ValueError('family overrides would need their own 5 Hz conversion')
    elif spec['step'] != 1:
        raise ValueError('only 10 Hz and 5 Hz are defined')
    cfg['canvas_instrument'] = dict(model=spec['model'], fps=spec['fps'], band_rows=spec['band_rows'],
                                    valid_canvas_rows=spec['valid_canvas_rows'], valid_source_rows=spec['valid_source_rows'],
                                    conversions=conv, spec_sha256=spec['spec_sha256'])
    return cfg


# ------------------------------------------------------------------------------------------------------- geometry
def band_affine(r0: float):
    """2x3 affine source px -> band canvas px (pixel-centre convention, as window.json geometry.canvas.affine)."""
    return np.array([[S, 0.0, 0.5 * S - 0.5], [0.0, S, -S * r0 + 0.5 * S - 0.5]])


def map_box(box, A):
    """Pixel-edge box through the affine (make_benchmark_windows.map_box, frozen convention)."""
    if box is None:
        return None
    return [float(A[0, 0] * box[0] + A[0, 2]), float(A[1, 1] * box[1] + A[1, 2]), float(A[0, 0] * box[2] + A[0, 2]),
            float(A[1, 1] * box[3] + A[1, 2])]


def K_band(window: dict, r0: float):
    """Band intrinsics from K_source; for the bench band equal to window.json K_canvas (checked)."""
    K = np.asarray(window['intrinsics']['K_source'], float)
    A = band_affine(r0)
    Kb = np.array([[A[0, 0] * K[0, 0], 0.0, A[0, 0] * K[0, 2] + A[0, 2]], [0.0, A[1, 1] * K[1, 1], A[1, 1] * K[1, 2] + A[1, 2]], [0, 0, 1.0]])
    return Kb


def band_flags(box_src, points, vsrc):
    """make_benchmark_windows.band_flags with the spec's valid source rows (frozen margins and thresholds)."""
    if box_src is None:
        return dict(in_band=False, height_ok=False, points_ok=points >= MIN_POINTS)
    inside = (box_src[0] >= BAND_MARGIN_SRC_PX and box_src[2] <= SRC_W - BAND_MARGIN_SRC_PX and box_src[1] >= vsrc[0] + BAND_MARGIN_SRC_PX
              and box_src[3] <= vsrc[1] - BAND_MARGIN_SRC_PX)
    return dict(in_band=bool(inside), height_ok=bool(box_src[3] - box_src[1] >= MIN_H_SRC_PX), points_ok=bool(points >= MIN_POINTS))


def is_bench_geometry(spec) -> bool:
    return list(spec['band_rows']) == list(BENCH_BAND) and list(spec['valid_source_rows']) == list(BENCH_BAND)


def window_on_band(window: dict, spec: dict) -> dict:
    """window.json with K_canvas and the targets' t0 box_canvas of the spec's band (unchanged for the bench band)."""
    if list(spec['band_rows']) == list(BENCH_BAND):
        return window
    w = copy.deepcopy(window)
    A = band_affine(spec['band_rows'][0])
    w['intrinsics']['K_canvas'] = K_band(window, spec['band_rows'][0]).tolist()
    for fam, t in w['targets'].items():
        if isinstance(t, dict) and t.get('box_source') is not None:
            t['box_canvas'] = map_box(t['box_source'], A)
    w['canvas_instrument_band'] = spec['band_rows']
    return w


def gt_on_band(gt: dict, spec: dict) -> dict:
    """gt.json with box_canvas / in_band / visible for the spec's band and valid rows (unchanged for the bench band)."""
    if is_bench_geometry(spec):
        return gt
    g = copy.deepcopy(gt)
    A = band_affine(spec['band_rows'][0])
    for fam, t in g['targets'].items():
        for r in t.get('per_tick', []):
            if not r.get('annotated'):
                continue
            f = band_flags(r.get('box_source'), int(r.get('lidar_points', 0)), spec['valid_source_rows'])
            r.update(box_canvas=map_box(r.get('box_source'), A), **f, visible=bool(f['in_band'] and f['height_ok'] and f['points_ok']))
    g['canvas_instrument_band'] = dict(band_rows=spec['band_rows'], valid_source_rows=spec['valid_source_rows'])
    return g


# -------------------------------------------------------------------------------------------------------- measuring
def measure_clip(window: dict, dets: dict, cfg: dict, spec: dict, families=None) -> dict:
    """measure_bench_v2.measure_clip on the spec's frame grid (the same calls; frames as 10 Hz ticks). With the 10 Hz
    bench spec this is the frozen function call for call (regression-checked on DEV)."""
    n = spec['n_frames']
    if len(dets['detections']) != n:
        raise ValueError(f'{len(dets["detections"])} detection frames, the spec needs {n}')
    window = window_on_band(window, spec)
    geo = mb.geometry_of(window, cfg); place = mb.Place(geo)
    hcfg = dict(mb.HISTORY_DEFAULT, **cfg.get('history', {}))
    t0f = spec['t0_frame']
    D = lambda t: dets['detections'][frame_of_tick(spec, t)]
    frames = [tick(spec, f) for f in range(t0f, n)]
    per_frame = [D(t) for t in frames]
    out = {}
    for fam, tgt in window['targets'].items():
        if families and fam not in families or fam not in mb.FAMILY_KIND:
            continue
        classes = mb.CLASSES[mb.FAMILY_KIND[fam]]
        gate_class = cfg.get('gate_class', {}).get(fam, fam)
        tracks, outs, exits, backs = {}, {}, {}, {}
        for name in cfg['trackers']:
            tc = mb.tracker_cfg(cfg, name, gate_class)
            chk = mb.t0_check(tgt, place, tc['t0_check']) if tc.get('t0_check') else None
            prior = None
            if tc.get('warmup'):
                bf = [tick(spec, f) for f in range(t0f, t0f - tc['warmup'] - 1, -1)]
                wb = mb.TRACKERS[tc['kind']]([D(t) for t in bf], tgt['box_canvas'], dict(tc, predict_exit=0), bf, place, classes, t0_ok=chk)
                if wb[0]['state'] == 'followed':
                    prior = [(f, r['box']) for f, r in zip(bf[1:], wb[1:]) if r['state'] == 'followed'][::-1]
            hf = [tick(spec, f) for f in range(t0f, t0f - hcfg['frames'] - 1, -1)]
            backs[name] = mb.TRACKERS[tc['kind']]([D(t) for t in hf], tgt['box_canvas'], dict(tc, predict_exit=hcfg.get('predict_exit', 0)),
                                                  hf, place, classes, t0_ok=chk)
            tr = mb.TRACKERS[tc['kind']](per_frame, tgt['box_canvas'], tc, frames, place, classes, t0_ok=chk, prior=prior)
            tracks[name] = tr
            outs[name], exits[name] = mb.tracker_outcomes(tr, fam)
        t0_found = {k: tr[0]['state'] == 'followed' for k, tr in tracks.items()}
        hist = mb.history_readings(backs, hcfg)
        out[fam] = dict(gate_class=gate_class, t0_found=t0_found, exits=exits, t0_box_at_edge=mb.edge_of(tgt['box_canvas'], cfg['common']) is not None,
                        history=hist, trackers={k: dict(track=v, outcomes=outs[k], history_track=backs[k]) for k, v in tracks.items()},
                        ensemble=dict(outcomes=mb.combine(outs, fam), series=mb.ensemble_series(tracks)))
    return out


def t0_agreement(rec: dict, ref: dict, iou_min=0.5) -> int:
    """Number of trackers that matched a detection at t0 in both the clip and its reference with boxes of IoU >= iou_min
    (the per-clip t0 association check of per_clip-prefix models)."""
    n = 0
    for k in rec['trackers']:
        a, b = rec['trackers'][k]['track'][0], (ref.get('trackers', {}).get(k) or {}).get('track', [{}])[0]
        if a.get('state') == 'followed' and b.get('state') == 'followed' and mb.iou(a['box'], b['box']) >= iou_min:
            n += 1
    return n


def names_from(path) -> list:
    return [l.strip() for l in Path(path).read_text().splitlines() if l.strip()]


def dev_only(bench: Path, names) -> None:
    dev = set(names_from(bench / 'names-dev.txt'))
    if not set(names) <= dev:
        raise SystemExit('refusing: DEV windows only (validation tools read gt.json)')


def split_of(bench: Path, names, spec: dict, allow_test: bool = False) -> str:
    """'dev' or 'test' for a names list. TEST windows are refused unless --allow-test is given AND the model's frozen
    protocol file (spec 'test_freeze_file', relative to the bench) exists; DEV and TEST are never mixed."""
    dev = set(names_from(bench / 'names-dev.txt')); test = set(names_from(bench / 'names-test.txt')); s = set(names)
    if s <= dev:
        return 'dev'
    if s & dev:
        raise SystemExit('refusing: a names file must not mix DEV and TEST windows')
    if not s <= test:
        raise SystemExit('refusing: names outside names-dev.txt and names-test.txt')
    ff = spec.get('test_freeze_file')
    if not allow_test:
        raise SystemExit('refusing: TEST windows need --allow-test')
    if not ff or not (bench / ff).is_file():
        raise SystemExit(f'refusing: TEST windows need the frozen protocol file {ff!r} in {bench} (model {spec.get("model")})')
    return 'test'


def windows_dir(split: str) -> str:
    return {'dev': 'windows-dev', 'test': 'windows-test'}[split]
