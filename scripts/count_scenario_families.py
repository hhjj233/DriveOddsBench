#!/usr/bin/env python3
"""Feasibility count of candidate scenario families on the local AV2 Sensor logs, past-only.

Context: the world-model calibration benchmark continues real AV2 front-centre clips K times; a fixed instrument reads
one target agent's future position, and calibration is judged by the rank of the real outcome among the generated
ones. This script counts how many benchmark windows each candidate scenario family would give on the study's DEV and
TEST logs. It is a feasibility count for choosing families, not a selection of windows.

Past-only. Every family rule uses the annotation, the ego poses, the map and the anchor image at or before the
anchor. Data availability (log length after the anchor, camera frames present) is checked from timestamps only.
On TEST logs no target position after the anchor is read or computed: the annotation table is cut at the last
candidate anchor right after loading, every per-anchor lookup is restricted to times <= the anchor (asserted), and
the outcome code refuses to run. On DEV logs (exploratory only) the target's future positions are read to report how
often the outcome varies.

Reuses scripts/make_wm_calibration_windows.py by import (unchanged): AFFINE, VEHICLES, the cuboid projection, the
visibility gates, the lead rule and its COCO detector check. Official AV2 devkit APIs for the camera, poses and map.

  count_scenario_families.py --output DIR [--cache DIR] [--workers 12] [--limit-logs N]
"""
from __future__ import annotations

import argparse, csv, datetime, json, math, os, sys, time, traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_wm_calibration_windows as mw  # noqa: E402  read-only reuse of the lead-family rules

DATA = Path(os.environ.get('DOB_ROOT', '.'), 'datasets/av2_sensor')
STUDY = Path(os.environ.get('DOB_ROOT', '.'), 'outputs/wm-calibration/study-v1')
SMOKE_WINDOWS = Path(os.environ.get('DOB_ROOT', '.'), 'outputs/wm-calibration/smoke-v1/windows/windows.json')
RULES_VERSION = 'scenario-families-v0-r3'
FIRST_S, STEP_S, SEPARATION_S, MARGINS_S = 2.5, 0.5, 8.0, (6.0, 5.0, 4.0)
TICK_NS, TOL_NS, SEC_NS = 100_000_000, 50_000_000, 1_000_000_000
FULL_W, FULL_H = 1550, 2048
LAND_W, LAND_H = 512, 288
LAND_SCALE = LAND_W / FULL_W          # 0.3303 output px per source px
LAND_ROWS = LAND_H / LAND_SCALE       # 871.875 source rows
LAND_BAND = (760, 1632)               # full-width 16:9 band of the 1550x2048 image used for 512x288 (fixed 2026-11-01)
EDGE_PX = 2.0                         # frame margin of the existing rule, in output pixels
GROUND_DEPTH_M = 5.0                  # the crop must contain the ground at this camera depth (the minimum lead gap)
PEDESTRIANS, RIDERS = {'PEDESTRIAN'}, {'BICYCLIST', 'MOTORCYCLIST'}
ROAD_USERS = set(mw.VEHICLES) | PEDESTRIANS | RIDERS
HORIZONS_S = (2, 4, 6)

FAMILIES = {
    'F1': 'lead vehicle, existing geometric rule: the nearest vehicle with x > 0 and |y| < 1.8 m (box centre); it must have '
          'its rear face (camera depth of the box centre minus L/2) at 5-25 m, >= 5 lidar points, its 256x320 box inside the '
          'frame (2 px) and >= 20 px tall. No detector check.',
    'F1det': 'F1 + the existing detector check (COCO Faster R-CNN v2, car/bus/truck, score >= 0.4, IoU >= 0.3 with the '
             'projected box on the causal anchor frame): the full existing lead rule.',
    'F2a': 'adjacent vehicle ahead: vehicle category, x 5-40 m, 1.8 <= |y| <= 5.5 m, visible; no motion condition.',
    'F2b': 'F2a + mean lateral velocity toward the ego corridor >= 0.2 m/s over the last 1 s (world velocity in the axes '
           'of the anchor ego frame).',
    'F2b_lane': 'F2a + lateral velocity toward the ego >= 0.2 m/s over the last 1 s measured across the target\'s own map lane '
                '(nearest devkit lane centreline within 3 m whose direction is within 45 deg of the box heading); a past-only '
                'variant that road curvature and the ego\'s own turning do not affect.',
    'F2c': 'F2b restricted to targets moving in the ego\'s direction: speed > 1 m/s and heading difference < 30 deg (a strict '
           'cut-in / lane-change candidate; F2b also admits oncoming and crossing vehicles).',
    'F3': 'oncoming vehicle: |heading difference to the ego| > 150 deg (box yaw), x 5-60 m, |y| < 15 m, speed > 2 m/s, visible.',
    'F4': 'crossing or turning vehicle: heading difference 45-135 deg (box yaw), x 5-50 m, |y| < 20 m, speed > 1 m/s, visible.',
    'F5': 'pedestrian ahead: PEDESTRIAN, x 3-30 m, |y| < 10 m, visible.',
    'F5_moving': 'F5 + speed > 0.5 m/s.',
    'F5_road': 'F5 + box centre on the map drivable area (devkit raster) or inside a map pedestrian crossing.',
    'F5_moving_road': 'F5_moving and F5_road.',
    'F6': 'cyclist or motorcyclist ahead: BICYCLIST or MOTORCYCLIST (the rider box), x 3-40 m, |y| < 10 m, visible.',
    'F7': 'stopped or braking lead: F1 whose lead has speed < 0.5 m/s, or mean acceleration < -1 m/s^2 over the last 0.5 s.',
    'F7det': 'F7 on F1det.',
    'F8': 'ego approaching an intersection: the ego forward line (x 0.5-30 m, y = -1, 0, +1 m) meets a map lane segment '
          'flagged is_intersection while the ego origin is not on one; ego speed > 2 m/s over the last 1 s. No target.',
    'F8_target': 'F8 + a visible road user (vehicle, pedestrian or rider) with x 3-50 m and |y| < 20 m; target = the nearest.',
}
FAMILY_ORDER = list(FAMILIES)
POOL_FAMILIES = [f for f in FAMILY_ORDER if f not in ('F1det', 'F7det', 'F8')]  # no detector, needs a target

ROW_FIELDS = (['family', 'set', 'split', 'source', 'log', 'anchor_offset_s', 'anchor_timestamp_ns', 'avail_6s', 'avail_5s', 'avail_4s',
               'pick_6s', 'pick_5s', 'pick_4s', 'n_targets', 'track_uuid', 'category', 'x_m', 'y_m', 'dist_m', 'heading_diff_deg',
               'speed_mps', 'vx_mps', 'vy_mps', 'toward_mps', 'toward_lane_mps', 'accel_mps2', 'lidar_points',
               'rear_face_m', 'length_m', 'width_m', 'height_m', 'box256_x1', 'box256_y1', 'box256_x2', 'box256_y2', 'px_h_256x320',
               'boxfull_x1', 'boxfull_y1', 'boxfull_x2', 'boxfull_y2', 'px_h_512x288', 'in_view', 'portrait_inside', 'portrait_ok',
               'land_inside', 'land_bottom_inside', 'land_ok', 'any_portrait_ok', 'any_land_ok',
               'det_iou', 'on_drivable', 'on_crosswalk', 'ego_speed_mps', 'intersection_hit_x_m', 'ego_in_intersection']
              + ['d_path_0']
              + [f'{k}_{t}' for t in HORIZONS_S for k in ('obs', 'vis', 'inimg', 'x_anc', 'y_anc', 'x_cur', 'y_cur', 'd_path', 'ahead_path',
                                                          'enter_anc', 'enter_cur', 'enter_path')]
              + ['cvres_anc_4'])


def clean(v):
    """JSON-safe copy: numpy scalars to Python, non-finite floats to None."""
    if isinstance(v, dict):
        return {k: clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean(x) for x in v]
    if hasattr(v, 'item') and not isinstance(v, (str, bytes)):
        v = v.item()
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def visible(box, points):
    """The existing script's visibility rule: projected box inside the 256 x 320 frame (2 px), >= 20 px tall, >= 5 lidar points."""
    return (box is not None and box[0] >= 2 and box[1] >= 2 and box[2] <= 254 and box[3] <= 318
            and box[3] - box[1] >= mw.MIN_BOX_PX and points >= mw.MIN_POINTS)


def land_box(box_full):
    """Full-resolution box -> the 512x288 landscape band (rows LAND_BAND of the 1550x2048 image, full width)."""
    c = 0.5 * LAND_SCALE - 0.5
    return (box_full[0] * LAND_SCALE + c, (box_full[1] - LAND_BAND[0]) * LAND_SCALE + c,
            box_full[2] * LAND_SCALE + c, (box_full[3] - LAND_BAND[0]) * LAND_SCALE + c)


def frame_flags(box, box_full, points):
    """Per-object frame checks at the anchor: the existing portrait rule and its analogue in the landscape band."""
    if box is None or box_full is None:
        return dict(in_view=False, portrait_inside=False, portrait_ok=False, px_h_512x288=None, land_inside=False,
                    land_bottom_inside=False, land_ok=False)
    p_in = box[0] >= 2 and box[1] >= 2 and box[2] <= 254 and box[3] <= 318
    lb = land_box(box_full); l_h = lb[3] - lb[1]
    l_in = lb[0] >= EDGE_PX and lb[1] >= EDGE_PX and lb[2] <= LAND_W - EDGE_PX and lb[3] <= LAND_H - EDGE_PX
    cu, cv = (box_full[0] + box_full[2]) / 2, (box_full[1] + box_full[3]) / 2
    return dict(in_view=bool(0 <= cu < FULL_W and 0 <= cv < FULL_H), portrait_inside=bool(p_in),
                portrait_ok=bool(visible(box, points)), px_h_512x288=float(l_h), land_inside=bool(l_in),
                land_bottom_inside=bool(EDGE_PX <= lb[3] <= LAND_H - EDGE_PX),
                land_ok=bool(l_in and l_h >= mw.MIN_BOX_PX and points >= mw.MIN_POINTS))


def yaw_rad(row):
    """Box yaw in the ego frame of its sweep, wrapped to [-pi, pi) (devkit quaternion conversion)."""
    from av2.geometry.geometry import mat_to_xyz, quat_to_mat
    yaw = mat_to_xyz(quat_to_mat(np.array([[row.qw, row.qx, row.qy, row.qz]])))[0, 2]
    return float((yaw + math.pi) % (2 * math.pi) - math.pi)


def path_coords(path, cum, q):
    """Arc length along a polyline and signed lateral offset (left positive) of the point q."""
    a, ab = path[:-1], path[1:] - path[:-1]; l2 = np.maximum((ab ** 2).sum(1), 1e-12)
    t = np.clip(((q - a) * ab).sum(1) / l2, 0.0, 1.0); dist = np.linalg.norm(q - (a + t[:, None] * ab), axis=1)
    i = int(np.argmin(dist)); w = q - a[i]
    return float(cum[i] + t[i] * math.sqrt(l2[i])), float(math.copysign(dist[i], ab[i, 0] * w[1] - ab[i, 1] * w[0]))


class LogScan:
    """One log: past-only features per candidate anchor; DEV-only future summaries."""

    def __init__(self, log_dir: Path, split: str):
        import pandas as pd
        from av2.geometry.camera.pinhole_camera import PinholeCamera
        from av2.utils.io import read_city_SE3_ego
        self.dir, self.split, self.is_dev = log_dir, split, split == 'DEV'
        ann = pd.read_feather(log_dir / 'annotations.feather')
        self.sweeps = np.sort(ann.timestamp_ns.unique())
        self.duration = (self.sweeps[-1] - self.sweeps[0]) / 1e9
        offsets = np.arange(FIRST_S, self.duration - min(MARGINS_S) + 1e-6, STEP_S)
        self.anchors = [(round(float(o), 1), int(self.sweeps[np.argmin(np.abs(self.sweeps - (self.sweeps[0] + o * 1e9)))])) for o in offsets]
        cutoff = max((a for _, a in self.anchors), default=int(self.sweeps[0]))
        if not self.is_dev:  # TEST: nothing after the last candidate anchor is kept in memory
            ann = ann[ann.timestamp_ns <= cutoff]
        self.by_t = {int(t): g for t, g in ann.groupby('timestamp_ns')}
        self.tracks = {}
        for uuid, g in ann.groupby('track_uuid', sort=False):
            g = g.sort_values('timestamp_ns')
            self.tracks[uuid] = (g.timestamp_ns.to_numpy(), g[['tx_m', 'ty_m', 'tz_m']].to_numpy())
        poses = read_city_SE3_ego(log_dir)
        self.poses = {int(t): p for t, p in poses.items() if self.is_dev or t <= cutoff}
        self.poses_t = np.array(sorted(self.poses)); self.poses_xyz = np.array([self.poses[t].translation for t in self.poses_t])
        self.images = sorted((log_dir / 'sensors/cameras/ring_front_center').glob('*.jpg'))
        self.stamps = np.array([int(p.stem) for p in self.images])
        self.cam = PinholeCamera.from_feather(log_dir, 'ring_front_center')
        self.cam_T_ego = np.linalg.inv(self.cam.ego_SE3_cam.transform_matrix)
        self.K_full = self.cam.intrinsics.K
        self.K_aff = np.vstack([mw.AFFINE, [0, 0, 1]]) @ self.K_full
        self.map_error = None
        try:
            import shapely
            from shapely.geometry import Polygon
            from av2.map.map_api import ArgoverseStaticMap
            self.avm = ArgoverseStaticMap.from_map_dir(log_dir / 'map', build_raster=True)
            def union(polys):
                u = shapely.union_all([Polygon(p[:, :2]).buffer(0) for p in polys if len(p) >= 3])
                shapely.prepare(u); return u
            self.xwalks = union([pc.polygon for pc in self.avm.get_scenario_ped_crossings()])
            self.inter = union([ls.polygon_boundary for ls in self.avm.get_scenario_lane_segments() if ls.is_intersection])
            seg_a, seg_b = [], []  # devkit lane centrelines as segments (city frame), for the lane-relative lateral velocity
            for ls in self.avm.get_scenario_lane_segments():
                cl = self.avm.get_lane_segment_centerline(ls.id)[:, :2]
                seg_a.append(cl[:-1]); seg_b.append(cl[1:])
            self.lane_a, self.lane_b = np.vstack(seg_a), np.vstack(seg_b)
        except Exception as exc:  # reported, the map-based subsets are then unknown for this log
            self.avm = None; self.map_error = f'{type(exc).__name__}: {exc}'
        self.n_detector = 0

    # ---- past-only accessors -------------------------------------------------------------------------------------
    def _past(self, t, anchor):
        if not self.is_dev:
            assert t <= anchor, 'TEST: lookup after the anchor refused'
        return t

    def pose(self, t, anchor):
        return self.poses[self._past(int(t), anchor)]

    def track_index(self, uuid, t, anchor):
        """Index of the track's row nearest to t (within 50 ms), never after the anchor on TEST."""
        ts = self.tracks[uuid][0]; j = np.searchsorted(ts, t)
        best = None
        for i in (j - 1, j):
            if 0 <= i < len(ts) and abs(int(ts[i]) - int(t)) <= TOL_NS and (best is None or abs(int(ts[i]) - t) < abs(int(ts[best]) - t)):
                best = i
        if best is not None:
            self._past(int(ts[best]), anchor)
        return best

    def to_anchor(self, xyz, t, anchor):
        anchor_T_t = self.pose(anchor, anchor).inverse().compose(self.pose(t, anchor))
        return anchor_T_t.transform_point_cloud(np.atleast_2d(xyz))[0]

    def past_velocity(self, uuid, anchor):
        """Mean velocity over the last 1 s (world motion, in the axes of the anchor ego frame); >= 0.45 s of track needed."""
        ts, xyz = self.tracks[uuid]
        lo = np.searchsorted(ts, anchor - SEC_NS - TOL_NS); hi = np.searchsorted(ts, anchor - 450_000_000, side='right')
        i_now = self.track_index(uuid, anchor, anchor)
        if hi <= lo or i_now is None:
            return None
        s0 = int(ts[lo]); self._past(s0, anchor)
        p_prev = self.to_anchor(xyz[lo], s0, anchor)
        return (xyz[i_now, :2] - p_prev[:2]) / ((anchor - s0) / 1e9)

    def past_accel(self, uuid, anchor):
        """Mean acceleration over the last 0.5 s: speed over [-0.5, 0] s minus speed over [-1, -0.5] s, per 0.5 s."""
        idx = [self.track_index(uuid, anchor - k * 500_000_000, anchor) for k in (0, 1, 2)]
        if any(i is None for i in idx):
            return None
        ts, xyz = self.tracks[uuid]
        t = [int(ts[i]) for i in idx]; p = [self.to_anchor(xyz[i], ti, anchor) for i, ti in zip(idx, t)]
        s0 = np.linalg.norm(p[0][:2] - p[1][:2]) / ((t[0] - t[1]) / 1e9); s1 = np.linalg.norm(p[1][:2] - p[2][:2]) / ((t[1] - t[2]) / 1e9)
        return float((s0 - s1) / ((t[0] - t[2]) / 2e9))

    def camera_ok(self, anchor, margin_s):
        """Images within 50 ms of every tick: causal for the 2 s history, nearest for the future (timestamps only)."""
        ticks = anchor + np.arange(-mw.HISTORY, int(round(margin_s * 10)) + 1) * TICK_NS
        if ticks[-1] > self.sweeps[-1] + TOL_NS or ticks[0] < self.stamps[0]:
            return False
        hist, fut = ticks[:mw.HISTORY + 1], ticks[mw.HISTORY + 1:]
        i = np.searchsorted(self.stamps, hist, side='right') - 1
        if (i < 0).any() or (np.abs(self.stamps[i] - hist) > TOL_NS).any():
            return False
        j = np.searchsorted(self.stamps, fut); j0, j1 = np.clip(j - 1, 0, len(self.stamps) - 1), np.clip(j, 0, len(self.stamps) - 1)
        return bool((np.minimum(np.abs(self.stamps[j0] - fut), np.abs(self.stamps[j1] - fut)) <= TOL_NS).all())

    def anchor_frame_256(self, anchor):
        import cv2
        i = int(np.flatnonzero(self.stamps <= anchor)[-1])
        img = cv2.cvtColor(cv2.imread(str(self.images[i])), cv2.COLOR_BGR2RGB)
        return cv2.warpAffine(img, mw.AFFINE, (256, 320), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # ---- geometry of the camera (for the landscape crop) ----------------------------------------------------------
    def geometry(self):
        a0 = self.anchors[0][1] if self.anchors else int(self.sweeps[0])
        # map ground height under the ego minus the ego origin height, median over the sweeps up to the first candidate
        # anchor (in the past of every anchor of the log)
        z_ground = None
        if self.avm is not None and self.avm.raster_ground_height_layer is not None:
            xyz = np.array([self.pose(int(t), a0).translation for t in self.sweeps if t <= a0 and int(t) in self.poses])
            dz = self.avm.raster_ground_height_layer.get_ground_height_at_xy(xyz) - xyz[:, 2]
            if np.isfinite(dz).any():
                z_ground = float(np.median(dz[np.isfinite(dz)]))
        z_g = z_ground if z_ground is not None else -0.30
        c = self.cam.ego_SE3_cam.translation

        def depth_row(px):
            p = self.cam_T_ego @ np.array([px, c[1], z_g, 1.0]); uv = self.K_full @ p[:3]
            return float(p[2]), float(uv[1] / uv[2])
        d1, _ = depth_row(c[0] + 5.0); d2, _ = depth_row(c[0] + 10.0)
        px5 = c[0] + 5.0 + (GROUND_DEPTH_M - d1) * 5.0 / (d2 - d1)
        K = self.K_full
        return dict(fx=float(K[0, 0]), fy=float(K[1, 1]), cx=float(K[0, 2]), cy=float(K[1, 2]), camera_z_ego_m=float(c[2]),
                    ground_z_ego_m=z_ground, camera_height_m=float(c[2] - z_g), horizon_row=depth_row(c[0] + 1e4)[1],
                    ground_5m_row=depth_row(px5)[1])

    # ---- DEV-only outcome ---------------------------------------------------------------------------------------
    def ego_path(self, anchor):
        """DEV only: the ego's driven path (sweep-time poses from the anchor to the log end, anchor frame), extended 100 m."""
        assert self.is_dev, 'the ego future is read on DEV logs only'
        inv = self.poses[anchor].inverse()
        ts = [int(t) for t in self.sweeps if t >= anchor and int(t) in self.poses]
        pts = inv.transform_point_cloud(np.array([self.poses[t].translation for t in ts]))[:, :2]
        keep = [0]
        for i in range(1, len(pts)):
            if np.linalg.norm(pts[i] - pts[keep[-1]]) >= 0.05:
                keep.append(i)
        head = inv.rotation @ self.poses[ts[-1]].rotation[:, 0]; head = head[:2] / np.linalg.norm(head[:2])
        path = np.vstack([pts[keep], pts[keep][-1] + 100.0 * head])
        return path, np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]), inv

    def future_summary(self, uuid, anchor, p0_xy, v_xy, path):
        """DEV only: the target's future in three frames (anchor, current ego, ego's driven path)."""
        assert self.is_dev, 'future target positions are read on DEV logs only'
        ts, xyz = self.tracks[uuid]
        poly, cum, anchor_T_city = path
        anc = np.full((60, 2), np.nan); cur = np.full((60, 2), np.nan); pth = np.full((60, 2), np.nan)  # pth: lateral offset, ahead
        for k in range(1, 61):
            t = anchor + k * TICK_NS; j = np.searchsorted(ts, t)
            cand = [i for i in (j - 1, j) if 0 <= i < len(ts) and abs(int(ts[i]) - t) <= TOL_NS]
            if not cand or int(ts[cand[0]]) not in self.poses:
                continue
            i = cand[0]; pose_t = self.poses[int(ts[i])]
            cur[k - 1] = xyz[i, :2]
            anc[k - 1] = anchor_T_city.compose(pose_t).transform_point_cloud(xyz[i][None])[0, :2]
            s_q, d_q = path_coords(poly, cum, anc[k - 1])
            s_e, _ = path_coords(poly, cum, anchor_T_city.transform_point_cloud(pose_t.translation[None])[0, :2])
            pth[k - 1] = (d_q, s_q - s_e)
        out = {'d_path_0': path_coords(poly, cum, np.asarray(p0_xy, dtype=float))[1]}
        for T in HORIZONS_S:
            k = 10 * T - 1; obs = bool(np.isfinite(anc[k, 0]))
            out[f'obs_{T}'] = int(obs)
            vis = inimg = 0  # is the outcome still readable in the front camera at T? (not annotated -> 0)
            if obs:
                j = int(np.argmin(np.abs(ts - (anchor + 10 * T * TICK_NS)))); g = self.by_t[int(ts[j])]; r = g[g.track_uuid == uuid].iloc[0]
                _, box = mw.project(r, self.cam_T_ego, self.K_aff); _, bf = mw.project(r, self.cam_T_ego, self.K_full)
                vis = int(visible(box, r.num_interior_pts))
                inimg = int(bf is not None and bf[0] >= 0 and bf[1] >= 0 and bf[2] <= FULL_W and bf[3] <= FULL_H)
            out[f'vis_{T}'], out[f'inimg_{T}'] = vis, inimg
            for name, arr in (('anc', anc), ('cur', cur)):
                out[f'x_{name}_{T}'] = float(arr[k, 0]) if obs else None; out[f'y_{name}_{T}'] = float(arr[k, 1]) if obs else None
            out[f'd_path_{T}'] = float(pth[k, 0]) if obs else None; out[f'ahead_path_{T}'] = float(pth[k, 1]) if obs else None
            with np.errstate(invalid='ignore'):
                ent_a = bool((np.abs(anc[:10 * T, 1]) < mw.LATERAL_M).any())
                ent_c = bool(((np.abs(cur[:10 * T, 1]) < mw.LATERAL_M) & (cur[:10 * T, 0] > 0)).any())
                ent_p = bool(((np.abs(pth[:10 * T, 0]) < mw.LATERAL_M) & (pth[:10 * T, 1] > 0)).any())
            out[f'enter_anc_{T}'] = 1 if ent_a else (0 if obs else None)
            out[f'enter_cur_{T}'] = 1 if ent_c else (0 if obs else None)
            out[f'enter_path_{T}'] = 1 if ent_p else (0 if obs else None)
        out['cvres_anc_4'] = (float(np.linalg.norm(anc[39] - (np.asarray(p0_xy) + 4.0 * np.asarray(v_xy))))
                              if v_xy is not None and np.isfinite(anc[39, 0]) else None)
        return out

    # ---- one anchor --------------------------------------------------------------------------------------------
    def target(self, r, anchor, box=None, depth=None):
        """Past-only description of one annotated object at the anchor (ego frame at the anchor)."""
        if box is None:
            depth, box = mw.project(r, self.cam_T_ego, self.K_aff)
        _, box_full = mw.project(r, self.cam_T_ego, self.K_full)
        v = self.past_velocity(r.track_uuid, anchor)
        side, yaw = (1.0 if r.ty_m > 0 else -1.0), yaw_rad(r)
        o = dict(track_uuid=r.track_uuid, category=r.category, x_m=float(r.tx_m), y_m=float(r.ty_m), dist_m=float(math.hypot(r.tx_m, r.ty_m)),
                 yaw_rad=yaw, heading_diff_deg=abs(math.degrees(yaw)), speed_mps=float(np.linalg.norm(v)) if v is not None else None,
                 vx_mps=float(v[0]) if v is not None else None, vy_mps=float(v[1]) if v is not None else None,
                 toward_mps=float(-side * v[1]) if v is not None else None,
                 lidar_points=int(r.num_interior_pts), rear_face_m=float(depth - r.length_m / 2), length_m=float(r.length_m),
                 width_m=float(r.width_m), height_m=float(r.height_m), px_h_256x320=float(box[3] - box[1]) if box else None)
        for i, k in enumerate(('x1', 'y1', 'x2', 'y2')):
            o[f'box256_{k}'] = float(box[i]) if box else None; o[f'boxfull_{k}'] = float(box_full[i]) if box_full else None
        o.update(frame_flags(box, box_full, int(r.num_interior_pts)))
        return o

    def object_families(self, o, anchor):
        """Family conditions (category, position, motion) that one object meets at the anchor; no visibility gate here."""
        c, x, ay, sp, hd = o['category'], o['x_m'], abs(o['y_m']), o['speed_mps'], o['heading_diff_deg']
        is_veh, fams = c in mw.VEHICLES, []
        if x <= 50.0:
            fams.append('users')  # F8_target candidates
        if is_veh and 5.0 <= x <= 40.0 and 1.8 <= ay <= 5.5:
            fams.append('F2a')
            o['toward_lane_mps'] = self.lane_toward(o, o['yaw_rad'], anchor)
            if o['toward_mps'] is not None and o['toward_mps'] >= 0.2:
                fams.append('F2b')
                if sp > 1.0 and hd < 30.0:
                    fams.append('F2c')
            if o['toward_lane_mps'] is not None and o['toward_lane_mps'] >= 0.2:
                fams.append('F2b_lane')
        if is_veh and hd > 150.0 and 5.0 <= x <= 60.0 and ay < 15.0 and sp is not None and sp > 2.0:
            fams.append('F3')
        if is_veh and 45.0 <= hd <= 135.0 and 5.0 <= x <= 50.0 and ay < 20.0 and sp is not None and sp > 1.0:
            fams.append('F4')
        if c in PEDESTRIANS and 3.0 <= x <= 30.0 and ay < 10.0:
            o['on_drivable'], o['on_crosswalk'] = self.on_map(o, anchor)
            moving, road = sp is not None and sp > 0.5, bool(o['on_drivable'] or o['on_crosswalk'])
            fams.append('F5')
            if moving:
                fams.append('F5_moving')
            if road:
                fams.append('F5_road')
            if moving and road:
                fams.append('F5_moving_road')
        if c in RIDERS and 3.0 <= x <= 40.0 and ay < 10.0:
            fams.append('F6')
        return fams

    def on_map(self, o, anchor):
        if self.avm is None:
            return None, None
        from av2.map.map_api import RasterLayerType
        import shapely
        p = self.pose(anchor, anchor).transform_point_cloud(np.array([[o['x_m'], o['y_m'], 0.0]]))
        drivable = bool(self.avm.get_raster_layer_points_boolean(p, RasterLayerType.DRIVABLE_AREA)[0])
        crosswalk = bool(shapely.contains_xy(self.xwalks, p[0, 0], p[0, 1])) if not self.xwalks.is_empty else False
        return drivable, crosswalk

    def lane_toward(self, o, yaw, anchor):
        """Past-only lateral velocity of the target across its own map lane, positive toward the ego's side (m/s)."""
        if self.avm is None or o['vx_mps'] is None:
            return None
        pose = self.pose(anchor, anchor); R = pose.rotation[:2, :2]
        q = pose.transform_point_cloud(np.array([[o['x_m'], o['y_m'], 0.0]]))[0, :2]
        v, h = R @ np.array([o['vx_mps'], o['vy_mps']]), R @ np.array([math.cos(yaw), math.sin(yaw)])
        a, ab = self.lane_a, self.lane_b - self.lane_a; l2 = np.maximum((ab ** 2).sum(1), 1e-12)
        t = np.clip(((q - a) * ab).sum(1) / l2, 0.0, 1.0); dist = np.linalg.norm(q - (a + t[:, None] * ab), axis=1)
        tan = ab / np.sqrt(l2)[:, None]
        ok = (dist <= 3.0) & (tan @ h >= math.cos(math.radians(45.0)))
        if not ok.any():
            return None
        i = int(np.flatnonzero(ok)[np.argmin(dist[ok])]); u = tan[i]; e = pose.translation[:2] - q
        lateral = u[0] * v[1] - u[1] * v[0]; side = u[0] * e[1] - u[1] * e[0]
        return float(math.copysign(1.0, side) * lateral)

    def intersection_ahead(self, anchor):
        if self.avm is None or self.inter.is_empty:
            return (None, None) if self.avm is None else (None, False)
        import shapely
        xs = np.arange(0.5, 30.0 + 1e-6, 0.5); grid = np.array([[x, y, 0.0] for x in xs for y in (-1.0, 0.0, 1.0)] + [[0.0, 0.0, 0.0]])
        city = self.pose(anchor, anchor).transform_point_cloud(grid)
        inside = shapely.contains_xy(self.inter, city[:, 0], city[:, 1])
        hits = grid[:-1][inside[:-1], 0]
        return (float(hits.min()) if len(hits) else None), bool(inside[-1])

    def anchor_rows(self, offset, anchor):
        avail = {m: bool(offset <= self.duration - m + 1e-6 and self.camera_ok(anchor, m)) for m in MARGINS_S}
        if not avail[min(MARGINS_S)]:
            return []
        now = self.by_t[self._past(anchor, anchor)]
        base = dict(split=self.split, source=self.dir.parent.name, log=self.dir.name, anchor_offset_s=offset, anchor_timestamp_ns=anchor,
                    avail_6s=int(avail[6.0]), avail_5s=int(avail[5.0]), avail_4s=int(avail[4.0]),
                    ego_speed_mps=mw.speed_at(self.poses_t[self.poses_t <= anchor], self.poses_xyz[self.poses_t <= anchor], anchor - SEC_NS, anchor))
        # found: the family rules with the existing visibility gate (portrait 256x320, >= 20 px); pool: the same family
        # conditions without any size or frame gate (>= 5 lidar points, box centre inside the 1550x2048 image)
        found = {f: [] for f in FAMILY_ORDER + ['users']}; pool = {f: [] for f in FAMILY_ORDER + ['users']}
        # F1 / F7: the existing lead rule (nearest vehicle in the corridor; no fallback to a farther one)
        veh = now[now.category.isin(mw.VEHICLES)]
        corridor = veh[(veh.tx_m > 0) & (veh.ty_m.abs() < mw.LATERAL_M)]
        if not corridor.empty:
            lead = corridor.sort_values('tx_m').iloc[0]
            depth, box = mw.project(lead, self.cam_T_ego, self.K_aff); face = depth - lead.length_m / 2
            if mw.GAP_M[0] <= face <= mw.GAP_M[1] and lead.num_interior_pts >= mw.MIN_POINTS and box is not None:
                o = self.target(lead, anchor, box, depth)
                o['accel_mps2'] = self.past_accel(lead.track_uuid, anchor)
                stopped = (o['speed_mps'] is not None and o['speed_mps'] < 0.5) or (o['accel_mps2'] is not None and o['accel_mps2'] < -1.0)
                if o['in_view']:
                    pool['F1'].append(o)
                    if stopped:
                        pool['F7'].append(o)
                if o['portrait_ok']:  # = the existing gates: box inside 256x320 (2 px), >= 20 px tall (points checked above)
                    self.n_detector += 1
                    o['det_iou'] = float(max((mw.iou(b, box) for b, _ in mw.detections(self.anchor_frame_256(anchor))), default=0.0))
                    found['F1'].append(o)
                    if stopped:
                        found['F7'].append(o)
                    if o['det_iou'] >= mw.DETECT['iou']:
                        found['F1det'].append(o)
                        if stopped:
                            found['F7det'].append(o)
        # F2-F6, F8 targets: every road user near the ego with >= 5 lidar points and a projected box
        sel = now[now.category.isin(ROAD_USERS) & (now.tx_m >= 3.0) & (now.tx_m <= 60.0) & (now.ty_m.abs() < 20.0)
                  & (now.num_interior_pts >= mw.MIN_POINTS)]
        for r in sel.itertuples(index=False):
            depth, box = mw.project(r, self.cam_T_ego, self.K_aff)
            if box is None:
                continue
            o = self.target(r, anchor, box, depth)
            if not (o['portrait_ok'] or o['in_view']):
                continue
            for fam in self.object_families(o, anchor):
                if o['portrait_ok']:
                    found[fam].append(o)
                if o['in_view']:
                    pool[fam].append(o)
        hit_x, ego_inside = self.intersection_ahead(anchor)
        extra = dict(intersection_hit_x_m=hit_x, ego_in_intersection=None if ego_inside is None else int(ego_inside))
        f8 = hit_x is not None and ego_inside is False and base['ego_speed_mps'] is not None and base['ego_speed_mps'] > 2.0
        rows, futures, path = [], {}, (self.ego_path(anchor) if self.is_dev else None)
        for fam in POOL_FAMILIES:  # size-free pool: one row per family and anchor, the nearest pool target
            cands = pool['users'] if fam == 'F8_target' else pool[fam]
            if not cands or (fam == 'F8_target' and not f8):
                continue
            o = min(cands, key=lambda q: q['dist_m'])
            row = dict(base, family=fam, set='size_free_pool', n_targets=len(cands), any_portrait_ok=int(any(q['portrait_ok'] for q in cands)),
                       any_land_ok=int(any(q['land_ok'] for q in cands)), **extra)
            row.update({k: v for k, v in o.items() if k not in ('det_iou',)})
            rows.append(row)
        for fam in FAMILY_ORDER:
            if fam in ('F8', 'F8_target'):
                if not f8 or (fam == 'F8_target' and not found['users']):
                    continue
                cands = found['users'] if fam == 'F8_target' else []
            else:
                cands = found[fam]
                if not cands:
                    continue
            row = dict(base, family=fam, set='family', n_targets=len(cands), **extra)
            if cands:
                o = min(cands, key=lambda q: q['dist_m'])
                row.update(o)
                if self.is_dev:  # exploratory outcome summaries; never on TEST
                    if o['track_uuid'] not in futures:
                        v = (o['vx_mps'], o['vy_mps']) if o['vx_mps'] is not None else None
                        futures[o['track_uuid']] = self.future_summary(o['track_uuid'], anchor, (o['x_m'], o['y_m']), v, path)
                    row.update(futures[o['track_uuid']])
            rows.append(row)
        return rows


def init_worker():
    """Keep each worker at about 2 cores: OpenCV's pool (map rasterization, warps) and pyarrow otherwise use every core."""
    import cv2, pyarrow
    cv2.setNumThreads(1); pyarrow.set_cpu_count(2); pyarrow.set_io_thread_count(2)


def scan_log(job):
    log_dir, split = Path(job[0]), job[1]
    t0 = time.time()
    try:
        scan = LogScan(log_dir, split)
        rows = []
        for offset, anchor in scan.anchors:
            rows.extend(scan.anchor_rows(offset, anchor))
        return clean(dict(log=log_dir.name, split=split, source=log_dir.parent.name, duration_s=scan.duration, n_anchor_grid=len(scan.anchors),
                          geometry=scan.geometry(), map_error=scan.map_error, n_detector=scan.n_detector, seconds=time.time() - t0, rows=rows,
                          rules=RULES_VERSION, error=None))
    except Exception as exc:
        return dict(log=log_dir.name, split=split, source=log_dir.parent.name, rows=[], rules=RULES_VERSION,
                    error=f'{type(exc).__name__}: {exc}', trace=traceback.format_exc(), seconds=time.time() - t0)


# ---------------------------------------------------------------------------------------------------------------------
def split_logs():
    """DEV and TEST log lists from protocol-v1.md section 2, cross-checked against the study's windows.json and name lists."""
    train = sorted(p for p in (DATA / 'train').iterdir() if p.is_dir())
    train = [train[i] for i in np.random.default_rng(2026).permutation(len(train))]
    smoke = {w['log'] for w in json.loads(SMOKE_WINDOWS.read_text())['windows']}
    val = [p for p in sorted((DATA / 'val').iterdir()) if p.is_dir()]
    dev, test = train[:200], [p for p in val if p.name not in smoke] + train[200:450]

    def scanned(name):
        w = json.loads((STUDY / name / 'windows.json').read_text())
        return {x['log'] for x in w['windows']} | {x['log'] for x in w['rejected_logs']}
    assert scanned('windows-dev') == {p.name for p in dev}
    assert scanned('windows-test-train') | (scanned('windows-test-val') - smoke) == {p.name for p in test}
    names = {s: [l.strip() for l in (STUDY / f'names-{s.lower()}.txt').read_text().splitlines() if l.strip()] for s in ('DEV', 'TEST')}
    for s, logs in (('DEV', dev), ('TEST', test)):
        prefixes = {p.name[:8] for p in logs}
        assert all(n.split('-t')[0] in prefixes for n in names[s]), s
    return dict(DEV=dev, TEST=test), names, sorted(smoke)


def quantile(vals, q):
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    return float(np.quantile(vals, q)) if vals else None


def greedy(offsets):
    picked = []
    for o in sorted(offsets):
        if not picked or o - picked[-1] >= SEPARATION_S - 1e-6:
            picked.append(o)
    return set(picked)


def aggregate(results, names, smoke, out_dir, args):
    rows_all = [r for res in results for r in res['rows']]
    rows = [r for r in rows_all if r.get('set', 'family') == 'family']       # the family rules (existing visibility gate)
    pool_rows = [r for r in rows_all if r.get('set') == 'size_free_pool']   # the same conditions without size/frame gate
    logs = {s: [res for res in results if res['split'] == s] for s in ('DEV', 'TEST')}
    # greedy non-overlapping pick per family, log and margin
    by_key = {}
    for r in rows:
        by_key.setdefault((r['family'], r['log']), []).append(r)
    for group in by_key.values():
        for m in MARGINS_S:
            tag = f'{int(m)}s'
            keep = greedy([r['anchor_offset_s'] for r in group if r[f'avail_{tag}']])
            for r in group:
                r[f'pick_{tag}'] = int(bool(r[f'avail_{tag}']) and r['anchor_offset_s'] in keep)
    # landscape frame: the fixed full-width band LAND_BAND; for reference also the rows a DEV-fitted rule would choose
    # (the flat-ground point at 5 m camera depth inside with a 2-px margin)
    geo = {s: [res['geometry'] for res in logs[s] if res.get('geometry')] for s in logs}
    r1 = math.ceil(max(g['ground_5m_row'] for g in geo['DEV']) + EDGE_PX / LAND_SCALE); r0_rule = int(math.floor(r1 - LAND_ROWS))
    margin = EDGE_PX / LAND_SCALE

    def inside(rows_from, g):
        return rows_from + margin <= g['ground_5m_row'] <= rows_from + LAND_ROWS - margin
    crop = dict(rule=f'full-width 16:9 band, rows [{LAND_BAND[0]}, {LAND_BAND[1]}) of the 1550x2048 image, scaled by 512/1550 to 512x288 '
                     '(fixed on 2026-11-01)',
                full_res_rows=list(LAND_BAND), full_res_cols=[0, FULL_W], scale=LAND_SCALE,
                affine_512x288=[[LAND_SCALE, 0.0, 0.5 * LAND_SCALE - 0.5], [0.0, LAND_SCALE, 0.5 * LAND_SCALE - 0.5 - LAND_SCALE * LAND_BAND[0]]],
                horizon_row_in_512x288_median_dev=float(np.median([(g['horizon_row'] - LAND_BAND[0]) * LAND_SCALE for g in geo['DEV']])),
                ground_5m_row_in_512x288_median_dev=float(np.median([(g['ground_5m_row'] - LAND_BAND[0]) * LAND_SCALE for g in geo['DEV']])),
                logs_with_ground_5m_inside={s: sum(inside(LAND_BAND[0], g) for g in geo[s]) for s in geo},
                rule_derived_alternative=dict(rule=f'rows chosen on DEV logs so that the flat-ground point at {GROUND_DEPTH_M} m camera depth '
                                                   f'straight ahead is inside with a {EDGE_PX:.0f}-px margin at 512x288 (not used for the tables)',
                                              full_res_rows=[r0_rule, round(r0_rule + LAND_ROWS, 3)],
                                              logs_with_ground_5m_inside={s: sum(inside(r0_rule, g) for g in geo[s]) for s in geo}),
                ground_5m_row_full={s: dict(min=min(g['ground_5m_row'] for g in geo[s]), median=float(np.median([g['ground_5m_row'] for g in geo[s]])),
                                            max=max(g['ground_5m_row'] for g in geo[s])) for s in geo if geo[s]},
                horizon_row_full={s: dict(min=min(g['horizon_row'] for g in geo[s]), median=float(np.median([g['horizon_row'] for g in geo[s]])),
                                          max=max(g['horizon_row'] for g in geo[s])) for s in geo if geo[s]},
                camera_height_m={s: float(np.median([g['camera_height_m'] for g in geo[s]])) for s in geo if geo[s]},
                fy_full={s: [min(g['fy'] for g in geo[s]), max(g['fy'] for g in geo[s])] for s in geo if geo[s]},
                cy_full={s: [min(g['cy'] for g in geo[s]), max(g['cy'] for g in geo[s])] for s in geo if geo[s]},
                logs={s: len(geo[s]) for s in geo})
    # share of targets >= 20 px tall at t0, per frame, on the size-free pool (6-s-margin anchors; past-only, both splits)
    size_share = {}
    for fam in POOL_FAMILIES:
        size_share[fam] = {}
        for s in ('DEV', 'TEST'):
            sel = [r for r in pool_rows if r['family'] == fam and r['split'] == s and r['avail_6s']]
            mean = lambda f: float(np.mean([bool(f(r)) for r in sel])) if sel else None
            win = {}
            for frame, key in (('portrait', 'any_portrait_ok'), ('landscape', 'any_land_ok')):
                n = 0
                for lg in {r['log'] for r in sel}:
                    n += len(greedy([r['anchor_offset_s'] for r in sel if r['log'] == lg and r[key]]))
                win[frame] = n
            size_share[fam][s] = dict(anchors=len(sel),
                                      portrait_ge20=mean(lambda r: r['px_h_256x320'] is not None and r['px_h_256x320'] >= mw.MIN_BOX_PX),
                                      landscape_ge20=mean(lambda r: r['px_h_512x288'] is not None and r['px_h_512x288'] >= mw.MIN_BOX_PX),
                                      portrait_ge20_inside=mean(lambda r: r['portrait_ok']), landscape_ge20_inside=mean(lambda r: r['land_ok']),
                                      portrait_inside=mean(lambda r: r['portrait_inside']), landscape_inside=mean(lambda r: r['land_inside']),
                                      windows_portrait_rule=win['portrait'], windows_landscape_rule=win['landscape'],
                                      portrait_median_px=quantile([r['px_h_256x320'] for r in sel], 0.5),
                                      landscape_median_px=quantile([r['px_h_512x288'] for r in sel], 0.5))
    # counts
    counts = {}
    for m in MARGINS_S:
        tag = f'{int(m)}s'; counts[tag] = {}
        for fam in FAMILY_ORDER:
            counts[tag][fam] = {}
            for s in ('DEV', 'TEST'):
                sel = [r for r in rows if r['family'] == fam and r['split'] == s and r[f'avail_{tag}']]
                counts[tag][fam][s] = dict(logs=len({r['log'] for r in sel}), anchors=len(sel), windows=sum(r[f'pick_{tag}'] for r in sel),
                                           logs_with_2_windows=sum(1 for lg in {r['log'] for r in sel} if sum(r[f'pick_{tag}'] for r in sel if r['log'] == lg) >= 2))
    # pixel heights of the window targets
    heights = {}
    for tag in ('6s', '4s'):
        heights[tag] = {}
        for fam in FAMILY_ORDER:
            heights[tag][fam] = {}
            for s in ('DEV', 'TEST'):
                sel = [r for r in rows if r['family'] == fam and r['split'] == s and r[f'pick_{tag}'] and r.get('px_h_256x320') is not None]
                heights[tag][fam][s] = dict(n=len(sel), portrait_median=quantile([r['px_h_256x320'] for r in sel], 0.5),
                                            portrait_p10=quantile([r['px_h_256x320'] for r in sel], 0.1),
                                            landscape_median=quantile([r['px_h_512x288'] for r in sel], 0.5),
                                            landscape_p10=quantile([r['px_h_512x288'] for r in sel], 0.1),
                                            landscape_box_inside=float(np.mean([r['land_inside'] for r in sel])) if sel else None,
                                            landscape_bottom_inside=float(np.mean([r['land_bottom_inside'] for r in sel])) if sel else None,
                                            # the existing rule tests the 256-wide frame incl. its ~10-px black side pads, so a box may
                                            # reach ~54 source px beyond the image edge and still count as inside
                                            portrait_box_beyond_image=float(np.mean([r['boxfull_x1'] < 0 or r['boxfull_x2'] > FULL_W or r['boxfull_y1'] < 0
                                                                                     or r['boxfull_y2'] > FULL_H for r in sel])) if sel else None)
    # composition of the window targets (past-only quantities, both splits)
    composition = {}
    for fam in ('F1', 'F2a', 'F2b', 'F2b_lane', 'F2c', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8_target'):
        composition[fam] = {}
        for s in ('DEV', 'TEST'):
            sel = [r for r in rows if r['family'] == fam and r['split'] == s and r['pick_6s']]
            known = [r for r in sel if r.get('speed_mps') is not None]
            share = lambda f, pool: float(np.mean([bool(f(r)) for r in pool])) if pool else None
            composition[fam][s] = dict(n=len(sel), velocity_unknown=share(lambda r: r.get('speed_mps') is None, sel),
                                       static=share(lambda r: r['speed_mps'] < 0.5, known),
                                       same_direction_moving=share(lambda r: r['speed_mps'] >= 0.5 and r['heading_diff_deg'] < 30.0, known),
                                       opposite_moving=share(lambda r: r['speed_mps'] >= 0.5 and r['heading_diff_deg'] > 150.0, known),
                                       crossing_moving=share(lambda r: r['speed_mps'] >= 0.5 and 45.0 <= r['heading_diff_deg'] <= 135.0, known),
                                       passenger_car=share(lambda r: r['category'] == 'REGULAR_VEHICLE', sel),
                                       pedestrian=share(lambda r: r['category'] in PEDESTRIANS, sel),
                                       rider=share(lambda r: r['category'] in RIDERS, sel))
    # DEV outcome variability (exploratory; 6-s-margin candidates and windows)
    dev_out = {}

    def shares(sel, key):
        vals = [r.get(key) for r in sel]; known = [v for v in vals if v is not None]
        return dict(known=len(known), share=float(np.mean(known)) if known else None, unknown_share=1 - len(known) / len(vals) if vals else None)
    for fam in ('F2a', 'F2b', 'F2b_lane', 'F2c', 'F5', 'F5_moving', 'F5_road', 'F5_moving_road', 'F3', 'F4', 'F6'):
        dev_out[fam] = {}
        for unit, flag in (('candidates', 'avail_6s'), ('windows', 'pick_6s')):
            sel = [r for r in rows if r['family'] == fam and r['split'] == 'DEV' and r[flag]]
            d = dict(n=len(sel))
            if fam.startswith('F2'):
                known0 = [abs(r['d_path_0']) < mw.LATERAL_M for r in sel if r.get('d_path_0') is not None]
                d['in_path_at_0'] = float(np.mean(known0)) if known0 else None
                outside = [r for r in sel if r.get('d_path_0') is not None and abs(r['d_path_0']) >= mw.LATERAL_M]  # path frame: not in path at t=0
                for T in HORIZONS_S:
                    for frame, key, pool in (('anchor', 'anc', sel), ('current', 'cur', sel), ('path', 'path', outside)):
                        d[f'enter_{frame}_frame_{T}s'] = shares(pool, f'enter_{key}_{T}')
            if fam.startswith('F5'):
                d['start_within_2m'] = float(np.mean([abs(r['y_m']) < 2.0 for r in sel])) if sel else None
                known0 = [abs(r['d_path_0']) < 2.0 for r in sel if r.get('d_path_0') is not None]
                d['start_within_2m_path'] = float(np.mean(known0)) if known0 else None
                for T in HORIZONS_S:
                    for frame, xk, yk in (('anchor', None, f'y_anc_{T}'), ('current', f'x_cur_{T}', f'y_cur_{T}'), ('path', f'ahead_path_{T}', f'd_path_{T}')):
                        vals = [None if r.get(yk) is None else int(abs(r[yk]) < 2.0 and (xk is None or r[xk] > 0)) for r in sel]
                        known = [v for v in vals if v is not None]
                        d[f'end_within_2m_{frame}_frame_{T}s'] = dict(known=len(known), share=float(np.mean(known)) if known else None,
                                                                      unknown_share=1 - len(known) / len(vals) if vals else None)
            if fam in ('F3', 'F4', 'F6'):
                for key in ('x_anc_4', 'y_anc_4', 'x_cur_4', 'y_cur_4', 'cvres_anc_4'):
                    vals = [r.get(key) for r in sel]
                    d[key] = dict(known=sum(v is not None for v in vals), p10=quantile(vals, 0.1), p50=quantile(vals, 0.5), p90=quantile(vals, 0.9),
                                  sd=float(np.std([v for v in vals if v is not None])) if any(v is not None for v in vals) else None)
                disp = [None if r.get('x_anc_4') is None else math.hypot(r['x_anc_4'] - r['x_m'], r['y_anc_4'] - r['y_m']) for r in sel]
                d['displacement_4s'] = dict(p10=quantile(disp, 0.1), p50=quantile(disp, 0.5), p90=quantile(disp, 0.9))
            dev_out[fam][unit] = d
    # is the outcome still readable in the front camera? (DEV only; target annotated and passing the visibility rule at T)
    dev_visibility = {}
    for fam in FAMILY_ORDER:
        sel = [r for r in rows if r['family'] == fam and r['split'] == 'DEV' and r['avail_6s'] and r.get('track_uuid')]
        if sel:
            dev_visibility[fam] = dict(n=len(sel), **{f'{k}_{T}s': float(np.mean([r.get(f'{k}_{T}') == 1 for r in sel])) for T in HORIZONS_S
                                                      for k in ('obs', 'vis', 'inimg')})
    # the existing lead rule on its own grid (2.5 s, then every 1 s; first passing anchor per log) against the name lists
    repro = {}
    for s in ('DEV', 'TEST'):
        mine = []
        for res in logs[s]:
            hits = sorted(r['anchor_offset_s'] for r in res['rows'] if r.get('set', 'family') == 'family' and r['family'] == 'F1det' and r['avail_6s'] and abs((r['anchor_offset_s'] - FIRST_S) % 1.0) < 1e-6)
            if hits:
                mine.append(f"{res['log'][:8]}-t{hits[0]:.1f}")
        ref = set(names[s]) if args.limit_logs is None else {n for n in names[s] if n[:8] in {res['log'][:8] for res in logs[s]}}
        repro[s] = dict(windows=len(mine), reference=len(ref), identical=sorted(mine) == sorted(ref), only_here=sorted(set(mine) - ref), only_reference=sorted(ref - set(mine)))
    summary = dict(version=RULES_VERSION, created=datetime.datetime.now().isoformat(timespec='seconds'), script='scripts/count_scenario_families.py',
                   splits={s: dict(logs=len(logs[s]), sources={src: sum(res['source'] == src for res in logs[s]) for src in ('train', 'val')},
                                   errors=[dict(log=res['log'], error=res['error']) for res in logs[s] if res.get('error')],
                                   map_errors=[dict(log=res['log'], error=res['map_error']) for res in logs[s] if res.get('map_error')])
                           for s in logs},
                   split_rule='DEV = train logs at positions 0-199 of the seed-2026 permutation; TEST = val minus the 3 smoke-test logs, plus train '
                              'positions 200-449 (protocol-v1.md section 2; checked against windows-*/windows.json and names-*.txt)',
                   smoke_logs_excluded=smoke,
                   anchors=dict(first_s=FIRST_S, step_s=STEP_S, margins_s=list(MARGINS_S), separation_s=SEPARATION_S,
                                availability='offset <= log length - margin; images within 50 ms of every tick of the 2 s history (causal) and of the margin (nearest)'),
                   families=FAMILIES, counts=counts, pixel_height=heights, landscape_crop=crop, size_share_20px_6s=size_share, target_composition_6s=composition,
                   dev_outcomes=dev_out, dev_visibility_at_horizon=dev_visibility, f1_existing_rule_reproduction=repro,
                   detector_calls=sum(res.get('n_detector', 0) for res in results), compute_seconds=sum(res.get('seconds', 0) for res in results),
                   past_only='TEST: annotation table cut at the last candidate anchor at load; per-anchor lookups asserted <= anchor; no future target '
                             'position read or computed. DEV: future positions read for the exploratory outcome summaries only.')
    summary = clean(summary)
    (out_dir / 'counts.json').write_text(json.dumps(summary, indent=1, allow_nan=False) + '\n')
    rows_all = rows + pool_rows
    rows_all.sort(key=lambda r: (r.get('set') != 'family', FAMILY_ORDER.index(r['family']), r['split'], r['log'], r['anchor_offset_s']))
    with open(out_dir / 'candidates.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=ROW_FIELDS, extrasaction='ignore'); w.writeheader()
        for r in rows_all:
            w.writerow({k: (int(v) if isinstance(v, bool) else round(v, 4) if isinstance(v, float) else ('' if v is None else v))
                        for k, v in ((k, r.get(k)) for k in ROW_FIELDS)})
    write_note(summary, out_dir)
    return summary


def fmt(v, nd=1):
    return '-' if v is None else (f'{v:.{nd}f}' if isinstance(v, float) else str(v))


def pct(d):
    return '-' if d is None or d.get('share') is None else f"{100 * d['share']:.0f} % (n={d['known']}, unk {100 * d['unknown_share']:.0f} %)"


def pct_short(d):
    return '-' if d is None or d.get('share') is None else f"{100 * d['share']:.0f} %"


def write_note(s, out_dir):
    c, h, crop, dev = s['counts'], s['pixel_height'], s['landscape_crop'], s['dev_outcomes']
    L = [f"# Scenario-family feasibility counts v0 (AV2 Sensor, past-only), {s['created'][:10]}", '',
         'Feasibility count for choosing scenario families of the world-model calibration benchmark. Not a final selection.',
         f"Produced by `{s['script']}` (reuses `scripts/make_wm_calibration_windows.py` by import). Files: `counts.json`, "
         '`candidates.csv` (one row per family x log x candidate anchor with the chosen target; column `set`: `family` = the family rules, '
         '`size_free_pool` = the same conditions without size or frame gate, used only for the 20-px share table).', '',
         '## Logs', '',
         f"- DEV: {s['splits']['DEV']['logs']} logs ({s['splits']['DEV']['sources']}); TEST: {s['splits']['TEST']['logs']} logs "
         f"({s['splits']['TEST']['sources']}).",
         f"- {s['split_rule']}. Smoke-test logs excluded: {', '.join(x[:8] for x in s['smoke_logs_excluded'])}.", '',
         '## Anchors, windows, targets', '',
         '- Candidate anchors every 0.5 s from 2.5 s after the first annotated sweep to M s before the last one, M = 6 s (main), '
         '5 s and 4 s; the anchor is the nearest sweep.',
         f"- Available = {s['anchors']['availability']}.",
         '- Geometry in the ego frame at the anchor (x forward, y left, AV2 ego origin); positions are box centres. Velocity = mean '
         'over the last 1 s of the target track (positions moved into the anchor frame through the ego poses). Heading difference '
         '= |box yaw| in the anchor frame.',
         '- Visible = the existing rule: projected cuboid box inside the 256x320 frame (2 px margin), >= 20 px tall, >= 5 lidar points.',
         '- Per family and anchor, the target is the nearest qualifying object (F1/F7: the existing lead).',
         '- Windows = greedy pick per family and log: anchors in time order, kept if >= 8 s after the last kept one; one target per window.',
         '  591 of the 597 logs last 14.9-15.8 s, so with a 6-s margin their anchors span at most 7 s and they give at most one window per',
         '  family; the 6 logs of 19.5-19.6 s can give two. With a 4-s margin a 15.5-s log can give two (anchors 2.5-11.5 s).', '',
         '## Families', '', '| family | definition |', '|---|---|']
    L += [f"| {k} | {v.replace('|', chr(92) + '|')} |" for k, v in s['families'].items()]
    L += ['', '## Counts', '', 'Summary: windows (greedy, >= 8 s apart) at future margins 6 / 5 / 4 s.', '',
          '| family | DEV windows 6 / 5 / 4 s | TEST windows 6 / 5 / 4 s | DEV logs (6 s) | TEST logs (6 s) |', '|---|---|---|---|---|']
    for fam in FAMILY_ORDER:
        w = {sp: ' / '.join(str(c[tag][fam][sp]['windows']) for tag in ('6s', '5s', '4s')) for sp in ('DEV', 'TEST')}
        L.append(f"| {fam} | {w['DEV']} | {w['TEST']} | {c['6s'][fam]['DEV']['logs']} | {c['6s'][fam]['TEST']['logs']} |")
    L += ['', 'Full tables: logs with >= 1 candidate anchor / candidate anchors / windows.', '']
    for tag in ('6s', '5s', '4s'):
        L += [f'### Future margin {tag}', '', '| family | DEV logs | DEV anchors | DEV windows | TEST logs | TEST anchors | TEST windows |', '|---|---|---|---|---|---|---|']
        for fam in FAMILY_ORDER:
            d, t = c[tag][fam]['DEV'], c[tag][fam]['TEST']
            L.append(f"| {fam} | {d['logs']} | {d['anchors']} | {d['windows']} | {t['logs']} | {t['anchors']} | {t['windows']} |")
        L.append('')
    r = s['f1_existing_rule_reproduction']
    L += ['Check of the existing lead rule on its own grid (2.5 s then every 1 s, first passing anchor per log, 6-s margin): '
          f"DEV {r['DEV']['windows']} windows (reference {r['DEV']['reference']}, identical names: {r['DEV']['identical']}); "
          f"TEST {r['TEST']['windows']} (reference {r['TEST']['reference']}, identical: {r['TEST']['identical']}).", '']
    L += ['## Target pixel height at the anchor (windows, 6-s margin)', '',
          f"Portrait = the existing 256x320 frame (AFFINE). Landscape = 512x288 band of the full-width image, source rows "
          f"[{crop['full_res_rows'][0]}, {crop['full_res_rows'][1]}) (scale {crop['scale']:.4f}). Inside = box inside the band (2 px); "
          'bottom = box bottom edge inside the band. Beyond image = projected box partly outside the 1550x2048 image although it '
          'passes the existing rule (that rule tests the 256-wide frame including its black side pads).', '',
          '| family | split | n | portrait median | portrait p10 | landscape median | landscape p10 | landscape inside | landscape bottom inside | beyond image |',
          '|---|---|---|---|---|---|---|---|---|---|']
    for fam in FAMILY_ORDER:
        for sp in ('DEV', 'TEST'):
            d = h['6s'][fam][sp]
            if d['n']:
                L.append(f"| {fam} | {sp} | {d['n']} | {fmt(d['portrait_median'])} | {fmt(d['portrait_p10'])} | {fmt(d['landscape_median'])} | "
                         f"{fmt(d['landscape_p10'])} | {fmt(100 * d['landscape_box_inside'], 0)} % | {fmt(100 * d['landscape_bottom_inside'], 0)} % | "
                         f"{fmt(100 * d['portrait_box_beyond_image'], 0)} % |")
    L += ['', '## Composition of the window targets (6-s margin; past-only quantities)', '',
          'Motion classes use the last-1-s velocity (moving = >= 0.5 m/s) and the box heading difference to the ego.', '',
          '| family | split | n | static | same direction, moving | opposite, moving | crossing (45-135 deg), moving | velocity unknown | passenger car | pedestrian | rider |',
          '|---|---|---|---|---|---|---|---|---|---|---|']
    for fam, per in s['target_composition_6s'].items():
        for sp in ('DEV', 'TEST'):
            d = per[sp]
            if d['n']:
                cells = [fmt(None if d[k] is None else 100 * d[k], 0) + ' %' for k in ('static', 'same_direction_moving', 'opposite_moving', 'crossing_moving',
                                                                                   'velocity_unknown', 'passenger_car', 'pedestrian', 'rider')]
                L.append(f"| {fam} | {sp} | {d['n']} | " + ' | '.join(cells) + ' |')
    L += ['', '## Share of targets >= 20 px tall at t0, portrait 256x320 vs landscape band 512x288 (size-free pool)', '',
          'The family targets above all pass the existing 20-px rule in the portrait frame by construction, so the share is computed on a '
          'size-free pool: per family and 6-s-margin candidate anchor, the nearest object that meets the family conditions (category, '
          'position, motion; F1/F7: the existing lead choice and rear-face gap) with >= 5 lidar points and its box centre inside the '
          '1550x2048 image, with no size or frame gate. ">= 20 px" = projected box height; "+ inside" = also inside that frame (2 px; '
          'portrait = the existing rule). Windows = greedy windows if the family used that frame\'s rule (any qualifying object at the anchor).', '',
          '| family | split | pool anchors | portrait >= 20 px | band >= 20 px | portrait >= 20 px + inside | band >= 20 px + inside | windows portrait rule | windows band rule | median px portrait / band |',
          '|---|---|---|---|---|---|---|---|---|---|']
    for fam, per in s['size_share_20px_6s'].items():
        for sp in ('DEV', 'TEST'):
            d = per[sp]
            if d['anchors']:
                pc = lambda k: f"{100 * d[k]:.0f} %"
                L.append(f"| {fam} | {sp} | {d['anchors']} | {pc('portrait_ge20')} | {pc('landscape_ge20')} | {pc('portrait_ge20_inside')} | "
                         f"{pc('landscape_ge20_inside')} | {d['windows_portrait_rule']} | {d['windows_landscape_rule']} | "
                         f"{fmt(d['portrait_median_px'])} / {fmt(d['landscape_median_px'])} |")
    L += ['', '## Landscape band rows', '',
          f"- {crop['rule']}.",
          f"- Rows used (full resolution 1550x2048): [{crop['full_res_rows'][0]}, {crop['full_res_rows'][1]}) (872 rows), all 1550 columns; "
          f"affine to 512x288: {[[round(v, 6) for v in row] for row in crop['affine_512x288']]}.",
          f"- For reference, a DEV-fitted rule (5-m ground point inside with a 2-px margin) would choose rows "
          f"[{crop['rule_derived_alternative']['full_res_rows'][0]}, {crop['rule_derived_alternative']['full_res_rows'][1]:.0f}): ground at 5 m inside for "
          f"{crop['rule_derived_alternative']['logs_with_ground_5m_inside']['DEV']}/{crop['logs']['DEV']} DEV and "
          f"{crop['rule_derived_alternative']['logs_with_ground_5m_inside']['TEST']}/{crop['logs']['TEST']} TEST logs (not used for the tables).",
          f"- Ground at 5 m camera depth, row (min/median/max): DEV {fmt(crop['ground_5m_row_full']['DEV']['min'])}/"
          f"{fmt(crop['ground_5m_row_full']['DEV']['median'])}/{fmt(crop['ground_5m_row_full']['DEV']['max'])}, TEST "
          f"{fmt(crop['ground_5m_row_full']['TEST']['min'])}/{fmt(crop['ground_5m_row_full']['TEST']['median'])}/{fmt(crop['ground_5m_row_full']['TEST']['max'])}; "
          f"inside the band (2-px margin) for {crop['logs_with_ground_5m_inside']['DEV']}/{crop['logs']['DEV']} DEV and "
          f"{crop['logs_with_ground_5m_inside']['TEST']}/{crop['logs']['TEST']} TEST logs; in the 512x288 frame it sits at row "
          f"~{fmt(crop['ground_5m_row_in_512x288_median_dev'], 0)} (DEV median).",
          f"- Horizon row (min/median/max): DEV {fmt(crop['horizon_row_full']['DEV']['min'])}/{fmt(crop['horizon_row_full']['DEV']['median'])}/"
          f"{fmt(crop['horizon_row_full']['DEV']['max'])}; in the 512x288 frame the horizon sits at row ~{fmt(crop['horizon_row_in_512x288_median_dev'], 0)}.",
          f"- Camera height above the map ground (median): DEV {fmt(crop['camera_height_m']['DEV'], 2)} m, TEST {fmt(crop['camera_height_m']['TEST'], 2)} m; "
          f"fy {fmt(crop['fy_full']['DEV'][0])}-{fmt(crop['fy_full']['DEV'][1])} px, cy {fmt(crop['cy_full']['DEV'][0])}-{fmt(crop['cy_full']['DEV'][1])} px (DEV).", '',
          '## DEV outcome variability (exploratory; DEV logs only; 6-s-margin candidates, windows in brackets)', '',
          'Shares are among targets whose outcome is known (annotated at the horizon, or already inside before it); "unk" = share '
          'whose track ended earlier. Three frames: anchor = the fixed ego frame at the anchor (target world motion; road '
          'curvature moves targets in and out); current = the ego frame at the time of the reading, target ahead (x > 0) (what a '
          'camera-relative instrument reads; the ego\'s own turns and lane changes sweep the corridor across targets); path = '
          'lateral offset from the ego\'s actually driven path (anchor to log end, extended straight), target ahead of the ego '
          'along it (curvature- and turn-robust; an ego lane change into the target\'s lane counts). F2 in the path frame: share '
          'among targets not already within 1.8 m of the path at t = 0.', '',
          '| family | n cand. [win.] | measure | 2 s | 4 s | 6 s |', '|---|---|---|---|---|---|']
    for fam in ('F2a', 'F2b', 'F2b_lane', 'F2c'):
        cd, wd = dev[fam]['candidates'], dev[fam]['windows']
        for frame in ('anchor', 'current', 'path'):
            cells = [f"{pct(cd[f'enter_{frame}_frame_{T}s'])} [{pct_short(wd[f'enter_{frame}_frame_{T}s'])}]" for T in HORIZONS_S]
            note = f" (in path at t=0: {fmt(100 * cd['in_path_at_0'], 0)} %)" if frame == 'path' and cd['in_path_at_0'] is not None else ''
            L.append(f"| {fam} | {cd['n']} [{wd['n']}] | enters abs(y) < 1.8 m, {frame} frame{note} | " + ' | '.join(cells) + ' |')
    for fam in ('F5', 'F5_moving', 'F5_road', 'F5_moving_road'):
        cd, wd = dev[fam]['candidates'], dev[fam]['windows']
        for frame in ('anchor', 'current', 'path'):
            start = cd['start_within_2m_path'] if frame == 'path' else cd['start_within_2m']
            cells = [f"{pct(cd[f'end_within_2m_{frame}_frame_{T}s'])} [{pct_short(wd[f'end_within_2m_{frame}_frame_{T}s'])}]" for T in HORIZONS_S]
            L.append(f"| {fam} | {cd['n']} [{wd['n']}] | ends abs(y) < 2 m, {frame} frame (at t=0: {fmt(100 * start, 0) if start is not None else '-'} %) | "
                     + ' | '.join(cells) + ' |')
    L += ['', 'Position at 4 s (DEV candidates; p10 / p50 / p90, m). CV residual = distance from the constant-velocity extrapolation of the '
          'last 1 s (anchor frame).', '', '| family | n | known at 4 s | x anchor frame | y anchor frame | x current frame | y current frame | displacement | CV residual |',
          '|---|---|---|---|---|---|---|---|---|']
    for fam in ('F3', 'F4', 'F6'):
        d = dev[fam]['candidates']
        q = lambda k: f"{fmt(d[k]['p10'])} / {fmt(d[k]['p50'])} / {fmt(d[k]['p90'])}"
        L.append(f"| {fam} | {d['n']} | {d['x_anc_4']['known']} | {q('x_anc_4')} | {q('y_anc_4')} | {q('x_cur_4')} | {q('y_cur_4')} | "
                 f"{fmt(d['displacement_4s']['p10'])} / {fmt(d['displacement_4s']['p50'])} / {fmt(d['displacement_4s']['p90'])} | {q('cvres_anc_4')} |")
    L += ['', 'Is the outcome still readable in the front camera? Share of DEV candidates (6-s margin) whose target is annotated / passes '
          'the existing visibility rule (box in the 256x320 frame, >= 20 px, >= 5 lidar points) / has its box fully inside the 1550x2048 '
          'image, at 2 / 4 / 6 s.', '', '| family | n | annotated 2 / 4 / 6 s | visible 2 / 4 / 6 s | box in image 2 / 4 / 6 s |', '|---|---|---|---|---|']
    for fam, d in s['dev_visibility_at_horizon'].items():
        trip = lambda k: ' / '.join(f"{100 * d[f'{k}_{T}s']:.0f} %" for T in HORIZONS_S)
        L.append(f"| {fam} | {d['n']} | {trip('obs')} | {trip('vis')} | {trip('inimg')} |")
    L += ['', '## Past-only discipline', '', f"- {s['past_only']}",
          '- Selection on TEST uses: the annotation at the anchor and in the last 1 s, ego poses up to the anchor, the static map, the '
          'causal anchor image (detector, F1det/F7det only), and timestamps of the log end and of the camera frames.', '',
          '## Run', '', f"- Detector calls: {s['detector_calls']}; worker CPU time {s['compute_seconds'] / 3600:.2f} h.",
          f"- Errors: DEV {s['splits']['DEV']['errors'] or 'none'}; TEST {s['splits']['TEST']['errors'] or 'none'}.",
          f"- Map errors: DEV {s['splits']['DEV']['map_errors'] or 'none'}; TEST {s['splits']['TEST']['map_errors'] or 'none'}.", '']
    (out_dir / 'NOTE.md').write_text('\n'.join(L))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--cache', type=Path, help='per-log result cache (reused when present and of the same rules version)')
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--limit-logs', type=int, help='first N logs of each split (smoke test)')
    args = parser.parse_args()
    if args.workers > 12:
        raise SystemExit('at most 12 workers on the shared machine')
    if args.output.exists():
        raise SystemExit('Fresh output required')
    os.environ.setdefault('OMP_NUM_THREADS', '2'); os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
    splits, names, smoke = split_logs()
    jobs = [(str(p), s) for s in ('DEV', 'TEST') for p in (splits[s][:args.limit_logs] if args.limit_logs else splits[s])]
    results, todo = [], []
    for job in jobs:
        f = args.cache / job[1] / f'{Path(job[0]).name}.json' if args.cache else None
        if f is not None and f.exists():
            res = json.loads(f.read_text())
            if res.get('rules') == RULES_VERSION and not res.get('error'):
                results.append(res); continue
        todo.append(job)
    print(f'{len(jobs)} logs, {len(results)} cached, {len(todo)} to scan', flush=True)
    t0 = time.time()
    with Pool(args.workers, initializer=init_worker) as pool:
        for i, res in enumerate(pool.imap_unordered(scan_log, todo, chunksize=1)):
            results.append(res)
            if args.cache:
                (args.cache / res['split']).mkdir(parents=True, exist_ok=True)
                (args.cache / res['split'] / f"{res['log']}.json").write_text(json.dumps(clean(res)))
            if res.get('error'):
                print('ERROR', res['log'], res['error'], flush=True)
            if (i + 1) % 20 == 0 or i + 1 == len(todo):
                print(f'{i + 1}/{len(todo)} scanned, {time.time() - t0:.0f} s', flush=True)
    order = {Path(j[0]).name: k for k, j in enumerate(jobs)}
    results.sort(key=lambda res: order[res['log']])
    args.output.mkdir(parents=True)
    summary = aggregate(results, names, smoke, args.output, args)
    print(json.dumps(dict(reproduction=summary['f1_existing_rule_reproduction'], crop=summary['landscape_crop']['full_res_rows'],
                          windows_6s={f: (summary['counts']['6s'][f]['DEV']['windows'], summary['counts']['6s'][f]['TEST']['windows']) for f in FAMILY_ORDER})))


if __name__ == '__main__':
    main()
