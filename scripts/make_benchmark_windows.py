#!/usr/bin/env python3
"""Benchmark windows in the band geometry (bench-v1): selection, rendering, contact sheets.

Source: the 1550x2048 AV2 ring_front_center image. Geometry (fixed):
  - Nano frame: full width, rows [768, 1698) (930 rows, 5:3) -> 320x192, Cosmos3's 256-tier "16,9" bucket.
  - Measurement band (all models): full width, rows [797, 1669) (DEV-fitted, 16:9) -> the 512x288 canvas.
  - Resampling: the call of make_wm_calibration_windows.py, cv2.warpAffine(..., flags=cv2.INTER_AREA, BORDER_CONSTANT 0),
    with its pixel-centre convention (x' = s x + 0.5 s - 0.5). OpenCV's warpAffine treats INTER_AREA as INTER_LINEAR
    (checked bit-identical with cv2 5.0.0), so the effective filter is bilinear without prefiltering.
Timing: anchors t0 = LiDAR-annotation times on a 0.5 s grid from log start + 2.5 s to log end - 6.1 s; 81 ticks
t0 + k 0.1 s, k = -20..60, each taking the nearest 20 Hz camera frame (ties to the earlier frame, |offset| <= 50 ms);
the 20 Hz frames over [t0 - 2.0 s, t0 + 6.0 s] (+-50 ms) are listed too.

Families (roles at t0, past-only; rules of scripts/count_scenario_families.py where they exist), one target each:
  lead       the lead (nearest vehicle with box centre x > 0, |y| < 1.8 m; no fallback to a farther one), rear face
             (camera depth of the box centre - L/2) 5-40 m, heading difference < 30 deg, speed > 1 m/s
  lead_stop  the lead, rear face 5-40 m, speed < 0.5 m/s or (mean acceleration over the last 0.5 s < -1 m/s^2 and
             heading difference < 30 deg)
  startup    ego speed < 0.5 m/s and the lead at rear face 5-15 m with speed < 0.5 m/s
  cutin      nearest vehicle with heading difference < 30 deg, speed > 1 m/s, 1.8 <= |y| <= 5.5 m, -2 <= x <= 40 m;
             stratum cutin_onset: its lane-relative lateral velocity toward the ego >= 0.2 m/s (count rule F2b_lane)
  ped        count rule F5; strata ped_road (drivable area or crosswalk), ped_moving (> 0.5 m/s)
  oncoming   F3; crossing F4; rider F6
  ego_inter  target = the ego: F8 (stratum ego_approach) or ego_stop_inter (ego < 0.5 m/s with an intersection lane on
             its forward line within 15 m, or the ego on one). F8 needs > 2 m/s, so the stopped case cannot be a subset
             of F8 itself; the family is their union and both are strata.
  Ego speed = mean over the last 1 s of poses (the count rule). Velocities over the last 1 s of the target track.
Visibility of every non-ego target at t0: projected box inside the band and the image width with a 2 source-px margin,
height >= 97 source px (20 px in 320x192), >= 5 lidar points, and a COCO Faster R-CNN v2 detection (score >= 0.4) of the
right class (vehicle: car/bus/truck; ped: person; rider: bicycle/motorcycle) with IoU >= 0.3 on the t0 canvas frame.
Selection per log: first window = the anchor with the highest score (sum over roles present of 1/sqrt(n_role), n_role =
candidate anchors of that role in the split; roles = the 9 families and the strata cutin_onset, ped_road, ped_moving,
ego_stop_inter; ties to the earlier anchor); a second window only at an anchor >= 3.0 s away that holds a rare role
(lead_stop, startup, cutin_onset, ped_road, crossing, rider) absent from the first; at most 2 per log.

Past-only on TEST: annotation and pose tables are read with a timestamp filter <= the log's last candidate anchor (later
rows are never materialised), every lookup is asserted <= the anchor, and no target or ego position after t0 is read or
computed (TEST gets no gt.json; its render reads only camera images and calibration). DEV gets gt.json for instrument
validation.

Generation subsamples (initially about PAWBench's size): the selection is the pool; the draw writes
names-dev-gen.txt (seed 2027, >= 8 windows per target family, 12 for lead) and names-test-gen.txt (seed 2026, >= 30, 50
for lead), at most 1 window per log. Target families lead, lead_stop, startup, cutin, ped, ego_inter, oncoming are processed
from the rarest to the most common in the split's pool (ties in the listed order); for each, the name-sorted windows that
contain it are shuffled with one random.Random(seed) and added until the subsample holds the target, counting windows
added for other families. Crossing and rider stay recorded but are not draw targets. DEV renders every selected window;
TEST renders only names-test-gen.txt.

Render version 2 (adopted before any generation): clips are made by cropping the source rows and calling
cv2.resize(crop, (W, H), interpolation=cv2.INTER_AREA) (anti-aliased area averaging). Version 1 used cv2.warpAffine with
flags=cv2.INTER_AREA, which OpenCV treats as INTER_LINEAR (bilinear, no prefilter). The geometry and the box mapping are
unchanged; the selection (and its t0 detector check, run on version-1 canvas frames) is unchanged.
Extension: the generation lists grow to 60 DEV / 200 TEST windows, keeping the draws as prefixes; the rest is
added in the order of a fresh random.Random('<seed>|extend') shuffle of the name-sorted remaining pool, from logs not yet
used, at most 1 window per log.

  make_benchmark_windows.py --phase select [--workers 12] [--limit-logs N]
  make_benchmark_windows.py --phase draw
  make_benchmark_windows.py --phase extend
  make_benchmark_windows.py --phase render [--workers 12] [--split dev|test|both] [--names FILE] [--done FILE]
  make_benchmark_windows.py --phase sheets [--count 12] [--names FILE] [--sheets-dir DIR]
  make_benchmark_windows.py --phase verify [--workers 12]
"""
from __future__ import annotations

import argparse, csv, datetime, hashlib, json, math, os, subprocess, sys, time, traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_wm_calibration_windows as mw  # noqa: E402  read-only reuse (projection, gates, detector weights, encoder)
import count_scenario_families as cf  # noqa: E402  read-only reuse (splits, past-only accessors, family primitives)

ROOT = Path(os.environ.get('DOB_ROOT', '.'), 'outputs/wm-calibration/bench-v1')
SRC_W, SRC_H = 1550, 2048
NANO_ROWS, NANO_W, NANO_H = (768, 1698), 320, 192
BAND_ROWS, CANVAS_W, CANVAS_H = (797, 1669), 512, 288


def affine_for(width_out, row0):
    s = width_out / SRC_W
    return np.array([[s, 0.0, 0.5 * s - 0.5], [0.0, s, 0.5 * s - 0.5 - s * row0]])


NANO_AFFINE, CANVAS_AFFINE = affine_for(NANO_W, NANO_ROWS[0]), affine_for(CANVAS_W, BAND_ROWS[0])
WARP = dict(call='cv2.warpAffine(img_rgb, A, (W, H), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT, borderValue=0)',
            source='make_wm_calibration_windows.py image_at()', effective_filter='bilinear (OpenCV warpAffine maps INTER_AREA to INTER_LINEAR; '
            'checked bit-identical with cv2 5.0.0)', pixel_centres="x' = s x + 0.5 s - 0.5 (the convention of the existing AFFINE)")
RENDER_VERSION = 2
RESAMPLE_V2 = dict(render_version=2, method='crop the source rows, then cv2.resize(crop, (W, H), interpolation=cv2.INTER_AREA) (area averaging)',
                   nano='rows [768, 1698) of 1550x2048 (1550x930) -> 320x192: scale 0.2064516 on both axes',
                   canvas='rows [797, 1669) (1550x872) -> 512x288: scale 0.3303226 in x and 0.3302752 in y; boxes keep the uniform 0.3303226 '
                          'mapping, at most 0.05 canvas px off at the bottom row',
                   pixel_centres="x' = (x + 0.5) * s - 0.5 (cv2.resize convention, the same as the version-1 affine)",
                   reason='version 1 used cv2.warpAffine(..., flags=cv2.INTER_AREA); OpenCV treats INTER_AREA as INTER_LINEAR there '
                          '(bit-identical), i.e. bilinear sampling without anti-aliasing at the 4.8x (Nano) and 3.0x (canvas) downscales',
                   changed='2026-11-01, before any generation; clips re-rendered in place',
                   selection='unchanged: the selection, its boxes and its t0 detector check (run on version-1 canvas frames) are kept')
EXTEND = {'DEV': dict(total=60, seed=2027), 'TEST': dict(total=200, seed=2026)}
TICK_NS, TOL_NS, SEC_NS = 100_000_000, 50_000_000, 1_000_000_000
TICKS = np.arange(-20, 61)
HISTORY_IDX = 20  # index of k = 0
FIRST_S, STEP_S, END_S = 2.5, 0.5, 6.1
LIST20_PAD_NS = 50_000_000
MARGIN_PX, MIN_H_PX, MIN_POINTS = 2.0, 97.0, 5
DET_SCORE, DET_IOU = 0.4, 0.3
COCO = {'vehicle': (3, 6, 8), 'ped': (1,), 'rider': (2, 4)}
COCO_NAMES = {1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 6: 'bus', 8: 'truck'}
FAMILIES = ('lead', 'lead_stop', 'startup', 'cutin', 'ped', 'oncoming', 'crossing', 'rider', 'ego_inter')
STRATA = {'cutin': ('cutin_onset',), 'ped': ('ped_road', 'ped_moving'), 'ego_inter': ('ego_approach', 'ego_stop_inter')}
ALL_STRATA = tuple(s for v in STRATA.values() for s in v)
SCORE_ROLES = FAMILIES + ('cutin_onset', 'ped_road', 'ped_moving', 'ego_stop_inter')
RARE = ('lead_stop', 'startup', 'cutin_onset', 'ped_road', 'crossing', 'rider')
MIN_SEP_S, MAX_WINDOWS = 3.0, 2
DRAW_FAMILIES = ('lead', 'lead_stop', 'startup', 'cutin', 'ped', 'ego_inter', 'oncoming')
DRAW = {'DEV': dict(seed=2027, file='names-dev-gen.txt', targets=dict(lead=12, lead_stop=8, startup=8, cutin=8, ped=8, ego_inter=8, oncoming=8)),
        'TEST': dict(seed=2026, file='names-test-gen.txt', targets=dict(lead=50, lead_stop=30, startup=30, cutin=30, ped=30, ego_inter=30, oncoming=30))}
RULES_VERSION = 'bench-v1'
_model = None


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def window_name(log_id, offset):
    return f'{sha(log_id)[:8]}-t{offset:.1f}'


def map_box(box, A):
    if box is None:
        return None
    return [float(A[0, 0] * box[0] + A[0, 2]), float(A[1, 1] * box[1] + A[1, 2]), float(A[0, 0] * box[2] + A[0, 2]), float(A[1, 1] * box[3] + A[1, 2])]


def band_flags(box_src, points):
    """Visibility geometry in source pixels: inside the band and the image width (2-px margin), >= 97 px tall, >= 5 points."""
    if box_src is None:
        return dict(in_band=False, height_ok=False, points_ok=points >= MIN_POINTS)
    inside = (box_src[0] >= MARGIN_PX and box_src[2] <= SRC_W - MARGIN_PX and box_src[1] >= BAND_ROWS[0] + MARGIN_PX
              and box_src[3] <= BAND_ROWS[1] - MARGIN_PX)
    return dict(in_band=bool(inside), height_ok=bool(box_src[3] - box_src[1] >= MIN_H_PX), points_ok=bool(points >= MIN_POINTS))


def warp(img_rgb, A, w, h):
    import cv2
    return cv2.warpAffine(img_rgb, A, (w, h), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def resample_v2(img_rgb):
    """Render version 2: (Nano 320x192, canvas 512x288) by crop + cv2.resize INTER_AREA."""
    import cv2
    nano = cv2.resize(img_rgb[NANO_ROWS[0]:NANO_ROWS[1]], (NANO_W, NANO_H), interpolation=cv2.INTER_AREA)
    canvas = cv2.resize(img_rgb[BAND_ROWS[0]:BAND_ROWS[1]], (CANVAS_W, CANVAS_H), interpolation=cv2.INTER_AREA)
    return nano, canvas


def read_rgb(path):
    import cv2
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


def detect(frame):
    """COCO Faster R-CNN v2 (the instrument's local weights) on one frame: [(box, score, label)] for the classes used."""
    global _model
    import torch, torchvision
    if _model is None:
        torch.set_num_threads(2)
        _model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None)
        _model.load_state_dict(torch.load(mw.WEIGHTS, map_location='cpu', weights_only=True)); _model.eval()
    with torch.no_grad():
        out = _model([torch.as_tensor(np.array(frame)).permute(2, 0, 1).float() / 255.0])[0]
    keep = {l for v in COCO.values() for l in v}
    return [(b.tolist(), float(s), int(l)) for b, s, l in zip(out['boxes'], out['scores'], out['labels']) if int(l) in keep and float(s) >= DET_SCORE]


def kind_of(category):
    return 'vehicle' if category in mw.VEHICLES else 'ped' if category in cf.PEDESTRIANS else 'rider' if category in cf.RIDERS else None


def nearest_frames(stamps, times):
    """Index of the nearest camera frame for each time (ties to the earlier frame) and its offset in ns."""
    j = np.searchsorted(stamps, times); j0 = np.clip(j - 1, 0, len(stamps) - 1); j1 = np.clip(j, 0, len(stamps) - 1)
    pick = np.where(np.abs(stamps[j0] - times) <= np.abs(stamps[j1] - times), j0, j1)
    return pick, stamps[pick] - times


def city_of(log_dir):
    return next((p.name.split('____')[1].split('_')[0] for p in (log_dir / 'map').glob('log_map_archive_*.json')), None)


class BenchLog(cf.LogScan):
    """One log for the bench grid. Reuses the count script's past-only accessors and family primitives."""

    def __init__(self, log_dir: Path, split: str, with_map=True):  # noqa: super().__init__ is replaced on purpose (filtered reads)
        import pyarrow.dataset as ds, pyarrow.feather as pf
        from av2.geometry.camera.pinhole_camera import PinholeCamera
        from av2.geometry.geometry import quat_to_mat
        from av2.geometry.se3 import SE3
        self.dir, self.split, self.is_dev = log_dir, split, split == 'DEV'
        self.sweeps = np.unique(pf.read_table(log_dir / 'annotations.feather', columns=['timestamp_ns']).column(0).to_numpy())
        self.duration = (self.sweeps[-1] - self.sweeps[0]) / 1e9
        offsets = np.arange(FIRST_S, self.duration - END_S + 1e-6, STEP_S)
        self.anchors = [(round(float(o), 1), int(self.sweeps[np.argmin(np.abs(self.sweeps - (self.sweeps[0] + o * 1e9)))])) for o in offsets]
        self.cutoff = None if self.is_dev else max((a for _, a in self.anchors), default=int(self.sweeps[0]))
        flt = None if self.cutoff is None else ds.field('timestamp_ns') <= self.cutoff  # TEST: later rows never materialised
        ann = ds.dataset(str(log_dir / 'annotations.feather'), format='feather').to_table(filter=flt).to_pandas()
        poses = ds.dataset(str(log_dir / 'city_SE3_egovehicle.feather'), format='feather').to_table(filter=flt).to_pandas()
        self.by_t = {int(t): g for t, g in ann.groupby('timestamp_ns')}
        self.tracks = {}
        for uuid, g in ann.groupby('track_uuid', sort=False):
            g = g.sort_values('timestamp_ns')
            self.tracks[uuid] = (g.timestamp_ns.to_numpy(), g[['tx_m', 'ty_m', 'tz_m']].to_numpy())
        rot = quat_to_mat(poses[['qw', 'qx', 'qy', 'qz']].to_numpy()); trans = poses[['tx_m', 'ty_m', 'tz_m']].to_numpy()
        self.poses = {int(t): SE3(rotation=rot[i], translation=trans[i]) for i, t in enumerate(poses.timestamp_ns.to_numpy())}  # as read_city_SE3_ego
        self.poses_t = np.array(sorted(self.poses)); self.poses_xyz = np.array([self.poses[t].translation for t in self.poses_t])
        self.images = sorted((log_dir / 'sensors/cameras/ring_front_center').glob('*.jpg'))
        self.stamps = np.array([int(p.stem) for p in self.images])
        self.cam = PinholeCamera.from_feather(log_dir, 'ring_front_center')
        self.cam_T_ego = np.linalg.inv(self.cam.ego_SE3_cam.transform_matrix)
        self.K_full = self.cam.intrinsics.K
        self.K_aff = np.vstack([mw.AFFINE, [0, 0, 1]]) @ self.K_full
        self.map_error, self.avm, self.n_detector = None, None, 0
        if with_map:
            try:  # as count_scenario_families.LogScan
                import shapely
                from shapely.geometry import Polygon
                from av2.map.map_api import ArgoverseStaticMap
                self.avm = ArgoverseStaticMap.from_map_dir(log_dir / 'map', build_raster=True)

                def union(polys):
                    u = shapely.union_all([Polygon(p[:, :2]).buffer(0) for p in polys if len(p) >= 3]); shapely.prepare(u); return u
                self.xwalks = union([pc.polygon for pc in self.avm.get_scenario_ped_crossings()])
                self.inter = union([ls.polygon_boundary for ls in self.avm.get_scenario_lane_segments() if ls.is_intersection])
                seg_a, seg_b = [], []
                for ls in self.avm.get_scenario_lane_segments():
                    cl = self.avm.get_lane_segment_centerline(ls.id)[:, :2]; seg_a.append(cl[:-1]); seg_b.append(cl[1:])
                self.lane_a, self.lane_b = np.vstack(seg_a), np.vstack(seg_b)
            except Exception as exc:
                self.avm = None; self.map_error = f'{type(exc).__name__}: {exc}'

    def frames_ok(self, anchor):
        times = anchor + TICKS * TICK_NS
        if times[-1] > self.sweeps[-1] + TOL_NS:
            return False
        _, dt = nearest_frames(self.stamps, times)
        return bool((np.abs(dt) <= TOL_NS).all())

    def ego_speeds(self, anchor):
        m = self.poses_t <= anchor; t, x = self.poses_t[m], self.poses_xyz[m]
        v1 = mw.speed_at(t, x, anchor - SEC_NS, anchor); v05 = mw.speed_at(t, x, anchor - SEC_NS // 2, anchor)
        vprev = mw.speed_at(t, x, anchor - SEC_NS, anchor - SEC_NS // 2)
        return v1, v05, ((v05 - vprev) / 0.5 if v05 is not None and vprev is not None else None)

    def describe(self, r, anchor):
        """Past-only description of one object at t0 with its source, Nano and canvas boxes, or None if it has no box."""
        depth, box = mw.project(r, self.cam_T_ego, self.K_aff)
        if box is None:
            return None
        o = self.target(r, anchor, box, depth)
        src = [o['boxfull_x1'], o['boxfull_y1'], o['boxfull_x2'], o['boxfull_y2']]
        o.update(box_source=src, box_nano=map_box(src, NANO_AFFINE), box_canvas=map_box(src, CANVAS_AFFINE), kind=kind_of(o['category']),
                 box_h_src_px=src[3] - src[1], **band_flags(src, int(r.num_interior_pts)))
        o['geo_visible'] = o['in_band'] and o['height_ok'] and o['points_ok']
        for k in ('px_h_256x320', 'px_h_512x288', 'in_view', 'portrait_inside', 'portrait_ok', 'land_inside', 'land_bottom_inside', 'land_ok',
                  'box256_x1', 'box256_y1', 'box256_x2', 'box256_y2', 'boxfull_x1', 'boxfull_y1', 'boxfull_x2', 'boxfull_y2'):
            o.pop(k, None)
        return o

    def anchor_record(self, offset, anchor):
        rec = dict(offset_s=offset, anchor_timestamp_ns=anchor, anchor_offset_exact_s=(anchor - int(self.sweeps[0])) / 1e9, available=self.frames_ok(anchor))
        if not rec['available']:
            return rec
        now = self.by_t[self._past(anchor, anchor)]
        v1, v05, acc = self.ego_speeds(anchor)
        hit_x, ego_inside = self.intersection_ahead(anchor)
        rec.update(ego_speed_mps=v1, ego_speed_05_mps=v05, ego_accel_mps2=acc, intersection_hit_x_m=hit_x,
                   ego_in_intersection=None if ego_inside is None else bool(ego_inside))
        cand = {f: [] for f in FAMILIES if f != 'ego_inter'}
        # lead families: the nearest vehicle in the corridor is the lead (no fallback)
        veh = now[now.category.isin(mw.VEHICLES)]
        corridor = veh[(veh.tx_m > 0) & (veh.ty_m.abs() < mw.LATERAL_M)]
        if not corridor.empty:
            lead = corridor.sort_values('tx_m').iloc[0]
            o = self.describe(lead, anchor)
            if o is not None and o['geo_visible']:
                o['accel_mps2'] = self.past_accel(lead.track_uuid, anchor)
                sp, hd, rf, a = o['speed_mps'], o['heading_diff_deg'], o['rear_face_m'], o['accel_mps2']
                if 5.0 <= rf <= 40.0 and hd < 30.0 and sp is not None and sp > 1.0:
                    cand['lead'].append(o)
                if 5.0 <= rf <= 40.0 and ((sp is not None and sp < 0.5) or (a is not None and a < -1.0 and hd < 30.0)):
                    cand['lead_stop'].append(o)
                if v1 is not None and v1 < 0.5 and sp is not None and sp < 0.5 and 5.0 <= rf <= 15.0:
                    cand['startup'].append(o)
        # the other families: every geometrically visible road user near the ego
        sel = now[now.category.isin(cf.ROAD_USERS) & (now.tx_m >= -2.0) & (now.tx_m <= 60.0) & (now.ty_m.abs() < 20.0)
                  & (now.num_interior_pts >= MIN_POINTS)]
        for r in sel.itertuples(index=False):
            depth, box = mw.project(r, self.cam_T_ego, self.K_full)
            if not band_flags(box, int(r.num_interior_pts))['in_band'] or box[3] - box[1] < MIN_H_PX:
                continue
            o = self.describe(r, anchor)
            if o is None or not o['geo_visible']:
                continue
            c, x, ay, sp, hd = o['category'], o['x_m'], abs(o['y_m']), o['speed_mps'], o['heading_diff_deg']
            is_veh = c in mw.VEHICLES
            if is_veh and hd < 30.0 and sp is not None and sp > 1.0 and 1.8 <= ay <= 5.5 and -2.0 <= x <= 40.0:
                o['toward_lane_mps'] = self.lane_toward(o, o['yaw_rad'], anchor)  # signed box yaw, as the count script
                cand['cutin'].append(o)
            if c in cf.PEDESTRIANS and 3.0 <= x <= 30.0 and ay < 10.0:
                o['on_drivable'], o['on_crosswalk'] = self.on_map(o, anchor)
                cand['ped'].append(o)
            if is_veh and hd > 150.0 and 5.0 <= x <= 60.0 and ay < 15.0 and sp is not None and sp > 2.0:
                cand['oncoming'].append(o)
            if is_veh and 45.0 <= hd <= 135.0 and 5.0 <= x <= 50.0 and ay < 20.0 and sp is not None and sp > 1.0:
                cand['crossing'].append(o)
            if c in cf.RIDERS and 3.0 <= x <= 40.0 and ay < 10.0:
                cand['rider'].append(o)
        # detector check on the t0 canvas frame, run only if some candidate needs it
        pending = {id(o): o for v in cand.values() for o in v}
        if pending:
            pick, dt = nearest_frames(self.stamps, np.array([anchor]))
            self.n_detector += 1
            dets = detect(warp(read_rgb(self.images[int(pick[0])]), CANVAS_AFFINE, CANVAS_W, CANVAS_H))
            for o in pending.values():
                best = max(((mw.iou(b, o['box_canvas']), l) for b, s, l in dets if l in COCO[o['kind']]), default=(0.0, None))
                o['det_iou'], o['det_label'] = float(best[0]), (COCO_NAMES[best[1]] if best[1] is not None else None)
                o['visible'] = o['det_iou'] >= DET_IOU
            rec['detections_t0'] = len(dets)
        targets = {}
        for fam, objs in cand.items():
            vis = [o for o in objs if o.get('visible')]
            if vis:
                targets[fam] = min(vis, key=lambda q: q['dist_m'])
        # strata are flags of the family target
        if 'cutin' in targets:
            t = targets['cutin']; t['cutin_onset'] = bool(t.get('toward_lane_mps') is not None and t['toward_lane_mps'] >= 0.2)
        if 'ped' in targets:
            t = targets['ped']; t['ped_road'] = bool(t.get('on_drivable') or t.get('on_crosswalk'))
            t['ped_moving'] = bool(t['speed_mps'] is not None and t['speed_mps'] > 0.5)
        approach = hit_x is not None and ego_inside is False and v1 is not None and v1 > 2.0
        stop_inter = v1 is not None and v1 < 0.5 and (bool(ego_inside) or (hit_x is not None and hit_x <= 15.0))
        if approach or stop_inter:
            targets['ego_inter'] = dict(target='ego', ego_approach=bool(approach), ego_stop_inter=bool(stop_inter))
        roles = [f for f in FAMILIES if f in targets]
        roles += [s for f in ('cutin', 'ped', 'ego_inter') if f in targets for s in STRATA[f] if targets[f].get(s) and s in SCORE_ROLES]
        for t in targets.values():
            if 'track_uuid' in t:
                t['target_hash'] = sha(t['track_uuid'])[:12]
        rec.update(targets=targets, roles=roles)
        return rec


def scan_log(job):
    log_dir, split = Path(job[0]), job[1]
    t0 = time.time()
    try:
        scan = BenchLog(log_dir, split)
        recs = [scan.anchor_record(o, a) for o, a in scan.anchors]
        return cf.clean(dict(log=log_dir.name, split=split, source=log_dir.parent.name, path=str(log_dir), duration_s=scan.duration,
                             first_sweep_ns=int(scan.sweeps[0]), city=city_of(log_dir), anchors=recs, map_error=scan.map_error,
                             n_detector=scan.n_detector, seconds=time.time() - t0, rules=RULES_VERSION, error=None))
    except Exception as exc:
        return dict(log=log_dir.name, split=split, path=str(log_dir), anchors=[], rules=RULES_VERSION, error=f'{type(exc).__name__}: {exc}',
                    trace=traceback.format_exc(), seconds=time.time() - t0)


# ---------------------------------------------------------------------------------------------------------------------
def select(results):
    """Greedy per-log selection; returns windows per split and the role counts n_role."""
    out = {}
    for split in ('DEV', 'TEST'):
        logs = [r for r in results if r['split'] == split]
        n = {role: sum(role in a.get('roles', []) for r in logs for a in r['anchors'] if a.get('available')) for role in SCORE_ROLES}
        weight = {role: (1.0 / math.sqrt(n[role]) if n[role] else 0.0) for role in SCORE_ROLES}
        windows = []
        for r in logs:
            cands = [a for a in r['anchors'] if a.get('available') and a.get('roles')]
            if not cands:
                continue
            score = lambda a: sum(weight[x] for x in a['roles'])
            first = max(cands, key=lambda a: (score(a), -a['offset_s']))
            chosen = [first]
            missing = set(RARE) - set(first['roles'])
            second = [a for a in cands if abs(a['offset_s'] - first['offset_s']) >= MIN_SEP_S - 1e-6 and missing & set(a['roles'])]
            if second and MAX_WINDOWS >= 2:
                chosen.append(max(second, key=lambda a: (score(a), -a['offset_s'])))
            for rank, a in enumerate(chosen, 1):
                windows.append(dict(name=window_name(r['log'], a['offset_s']), split=split, log=r['log'], log_path=r['path'], city=r.get('city'),
                                    first_sweep_ns=r['first_sweep_ns'], offset_s=a['offset_s'], anchor_timestamp_ns=a['anchor_timestamp_ns'],
                                    rank=rank, score=score(a), roles=a['roles'], record=a))
        out[split] = dict(n_role=n, weight=weight, windows=windows, logs=len(logs),
                          logs_with_candidates=sum(any(a.get('available') and a.get('roles') for a in r['anchors']) for r in logs),
                          candidate_anchors=sum(1 for r in logs for a in r['anchors'] if a.get('available') and a.get('roles')),
                          available_anchors=sum(1 for r in logs for a in r['anchors'] if a.get('available')),
                          grid_anchors=sum(len(r['anchors']) for r in logs))
    return out


FAMILY_FIELDS = ['window', 'split', 'log_hash', 'anchor_offset_s', 'window_rank', 'window_score', 'family', *ALL_STRATA, 'target_hash', 'target_category',
                 'target_x_m', 'target_y_m', 'target_dist_m', 'rear_face_m', 'heading_diff_deg', 'speed_mps', 'vx_mps', 'vy_mps', 'toward_mps',
                 'toward_lane_mps', 'accel_mps2', 'lidar_points', 'box_h_src_px', 'box_h_nano_px', 'box_h_canvas_px', 'det_iou', 'det_label',
                 'ego_speed_mps', 'ego_speed_05_mps', 'ego_accel_mps2', 'intersection_hit_x_m', 'ego_in_intersection', 'city']


def family_rows(w):
    rec = w['record']; rows = []
    for fam in FAMILIES:
        t = rec['targets'].get(fam)
        if t is None:
            continue
        row = dict(window=w['name'], split=w['split'], log_hash=w['name'][:8], anchor_offset_s=w['offset_s'], window_rank=w['rank'], window_score=w['score'],
                   family=fam, ego_speed_mps=rec.get('ego_speed_mps'), ego_speed_05_mps=rec.get('ego_speed_05_mps'), ego_accel_mps2=rec.get('ego_accel_mps2'),
                   intersection_hit_x_m=rec.get('intersection_hit_x_m'), ego_in_intersection=rec.get('ego_in_intersection'), city=w.get('city'))
        for s in STRATA.get(fam, ()):
            row[s] = int(bool(t.get(s)))
        if fam == 'ego_inter':
            row['target_hash'] = 'ego'
        else:
            bs = t['box_source']
            row.update(target_hash=t['target_hash'], target_category=t['category'], target_x_m=t['x_m'], target_y_m=t['y_m'], target_dist_m=t['dist_m'],
                       rear_face_m=t['rear_face_m'], heading_diff_deg=t['heading_diff_deg'], speed_mps=t['speed_mps'], vx_mps=t['vx_mps'], vy_mps=t['vy_mps'],
                       toward_mps=t['toward_mps'], toward_lane_mps=t.get('toward_lane_mps'), accel_mps2=t.get('accel_mps2'), lidar_points=t['lidar_points'],
                       box_h_src_px=bs[3] - bs[1], box_h_nano_px=t['box_nano'][3] - t['box_nano'][1], box_h_canvas_px=t['box_canvas'][3] - t['box_canvas'][1],
                       det_iou=t['det_iou'], det_label=t['det_label'])
        rows.append(row)
    return rows


def write_csv(path, fields, rows):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore'); w.writeheader()
        for r in rows:
            w.writerow({k: (int(v) if isinstance(v, bool) else round(v, 4) if isinstance(v, float) else ('' if v is None else v)) for k, v in ((k, r.get(k)) for k in fields)})


def selection_counts(sel):
    counts = {}
    for split, s in sel.items():
        ws = s['windows']
        per_log = {}
        for w in ws:
            per_log[w['log']] = per_log.get(w['log'], 0) + 1
        counts[split] = dict(windows=len(ws), logs=s['logs'], logs_with_candidates=s['logs_with_candidates'],
                             windows_per_log={str(k): sum(1 for v in per_log.values() if v == k) for k in (1, 2)} | {'0': s['logs'] - len(per_log)},
                             second_windows=sum(w['rank'] == 2 for w in ws),
                             per_role={role: sum(role in w['roles'] for w in ws) for role in FAMILIES + ALL_STRATA},
                             per_role_first_window_only={role: sum(role in w['roles'] for w in ws if w['rank'] == 1) for role in FAMILIES + ALL_STRATA},
                             candidate_anchors_per_role=s['n_role'], candidate_anchors=s['candidate_anchors'], available_anchors=s['available_anchors'],
                             grid_anchors=s['grid_anchors'])
        # ego_approach is not a score role: count it from the targets
        for key in ('per_role', 'per_role_first_window_only'):
            counts[split][key]['ego_approach'] = sum(bool(w['record']['targets'].get('ego_inter', {}).get('ego_approach')) for w in ws
                                                     if key == 'per_role' or w['rank'] == 1)
        counts[split]['candidate_anchors_per_role']['ego_approach'] = None
    return counts


def draw_subsample(windows, targets, seed):
    """Stratified draw, at most one window per log; see the module docstring."""
    import random
    rng = random.Random(seed)
    pool_count = {f: sum(f in w['roles'] for w in windows) for f in DRAW_FAMILIES}
    order = sorted(DRAW_FAMILIES, key=lambda f: (pool_count[f], DRAW_FAMILIES.index(f)))
    chosen, logs, steps = [], set(), []
    for fam in order:
        pool = sorted((w for w in windows if fam in w['roles']), key=lambda w: w['name'])
        rng.shuffle(pool)
        added = 0
        for w in pool:
            if sum(fam in c['roles'] for c in chosen) >= targets[fam]:
                break
            if w['log'] in logs:
                continue
            chosen.append(w); logs.add(w['log']); added += 1
        steps.append(dict(family=fam, pool_windows=len(pool), pool_logs=len({w['log'] for w in pool}), added=added,
                          holds=sum(fam in c['roles'] for c in chosen), target=targets[fam]))
    return chosen, order, steps, pool_count


def per_family(windows):
    out = {r: sum(r in w['roles'] for w in windows) for r in SCORE_ROLES}
    out['ego_approach'] = out['ego_inter'] - out['ego_stop_inter']  # > 2 m/s and < 0.5 m/s exclude each other
    return out


def phase_draw(args):
    sel = json.loads((args.root / 'selection.json').read_text())
    block = dict(rule='the selection is the pool. Per split: target families processed from the rarest to the most common in the split pool '
                      '(ties in the listed order); for each, the windows containing it are sorted by name and shuffled with one '
                      'random.Random(seed), then added in that order, skipping logs already in the subsample, until the subsample holds '
                      'the target number of windows containing the family (windows added for other families count). At most 1 window per log. '
                      'Crossing and rider are recorded but are not draw targets.',
                 target_families=list(DRAW_FAMILIES), script_sha256_at_draw=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 created=datetime.datetime.now().isoformat(timespec='seconds'), splits={})
    for split, cfg in DRAW.items():
        chosen, order, steps, pool_count = draw_subsample(sel['windows'][split], cfg['targets'], cfg['seed'])
        names = [w['name'] for w in chosen]
        (args.root / cfg['file']).write_text('\n'.join(names) + '\n')
        block['splits'][split] = dict(seed=cfg['seed'], file=cfg['file'], targets=cfg['targets'], order=order, pool_windows_per_family=pool_count,
                                      steps=steps, total=len(chosen),
                                      per_family=per_family(chosen),
                                      shortfall={f: cfg['targets'][f] - per_family(chosen)[f] for f in DRAW_FAMILIES if per_family(chosen)[f] < cfg['targets'][f]},
                                      names=names)
    sel['generation_subsample'] = block
    (args.root / 'selection.json').write_text(json.dumps(cf.clean(sel), indent=1) + '\n')
    write_note(args.root)
    print(json.dumps({s: dict(total=b['total'], per_family={f: b['per_family'][f] for f in FAMILIES}, order=b['order'], shortfall=b['shortfall'])
                      for s, b in block['splits'].items()}))


def phase_extend(args):
    """Grow the generation lists to EXTEND totals; the draws stay as prefixes."""
    import random
    sel = json.loads((args.root / 'selection.json').read_text())
    gen = sel['generation_subsample']
    for split, cfg in EXTEND.items():
        part = gen['splits'][split]; f = args.root / part['file']
        current = f.read_text().split()
        prefix = part['names']
        assert current[:len(prefix)] == prefix, f'{f.name} does not start with the draw'
        current = prefix  # rebuild from the draw (idempotent)
        pool = {w['name']: w for w in sel['windows'][split]}
        used = {pool[n]['log'] for n in current}
        rng = random.Random(f"{cfg['seed']}|extend")
        order = sorted(n for n in pool if n not in set(current))
        rng.shuffle(order)
        added = []
        for n in order:
            if len(current) + len(added) >= cfg['total']:
                break
            if pool[n]['log'] in used:
                continue
            added.append(n); used.add(pool[n]['log'])
        final = current + added
        f.write_text('\n'.join(final) + '\n')
        part['extension'] = dict(total=cfg['total'], seed=cfg['seed'], rng=f"random.Random('{cfg['seed']}|extend'): a fresh stream named 'extend' "
                                 '(Python string seed); the remaining pool windows, sorted by name, shuffled once; added in that order from logs '
                                 'not yet used, at most 1 window per log', added=len(added), added_names=added,
                                 reached=len(final), shortfall=cfg['total'] - len(final))
        part['final_names'] = final; part['final_total'] = len(final)
        part['final_per_family'] = per_family([pool[n] for n in final])
        part['final_logs'] = len({pool[n]['log'] for n in final})
    gen['extension_created'] = datetime.datetime.now().isoformat(timespec='seconds')
    gen['script_sha256_at_extend'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (args.root / 'selection.json').write_text(json.dumps(cf.clean(sel), indent=1) + '\n')
    write_note(args.root)
    print(json.dumps({s: dict(total=gen['splits'][s]['final_total'], added=gen['splits'][s]['extension']['added'],
                              per_family={k: gen['splits'][s]['final_per_family'][k] for k in FAMILIES}) for s in EXTEND}))


def verify_window(job):
    """t0 detector on the version-2 canvas frame (k = 0; past-only) against the stored t0 target boxes."""
    name, split, targets = job
    d = ROOT_RUNTIME[0] / f'windows-{split.lower()}' / name
    raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(d / 'canvas/real.mp4'), '-frames:v', str(HISTORY_IDX + 1), '-f', 'rawvideo',
                                   '-pix_fmt', 'rgb24', '-'])  # history frames only (k <= 0)
    frame = np.frombuffer(raw, np.uint8).reshape(-1, CANVAS_H, CANVAS_W, 3)[HISTORY_IDX]
    dets = detect(frame)
    out = {}
    for fam, t in targets.items():
        best = max((mw.iou(b, t['box_canvas']) for b, sc, l in dets if l in COCO[t['kind']]), default=0.0)
        out[fam] = dict(iou_v2=float(best), iou_selection_v1=t['det_iou'])
    return dict(name=name, split=split, families=out)


def phase_verify(args):
    sel = json.loads((args.root / 'selection.json').read_text()); gen = sel['generation_subsample']['splits']
    jobs = []
    for split in ('DEV', 'TEST'):
        names = set((args.root / gen[split]['file']).read_text().split())
        for wv in sel['windows'][split]:
            if wv['name'] not in names:
                continue
            rec = json.loads((args.root / f'windows-{split.lower()}' / wv['name'] / 'window.json').read_text())
            assert rec.get('render_version') == RENDER_VERSION, wv['name']
            tg = {f: dict(box_canvas=t['box_canvas'], det_iou=t['det_iou'], kind=kind_of(t['category'])) for f, t in rec['targets'].items() if f != 'ego_inter'}
            jobs.append((wv['name'], split, tg))
    with Pool(args.workers, initializer=cf.init_worker) as pool:
        res = list(pool.imap_unordered(verify_window, jobs, chunksize=1))
    summary = {}
    for split in ('DEV', 'TEST'):
        rows = [(f, v) for r in res if r['split'] == split for f, v in r['families'].items()]
        per = {}
        for fam in FAMILIES:
            vals = [v for f, v in rows if f == fam]
            if vals:
                per[fam] = dict(targets=len(vals), pass_v2=sum(v['iou_v2'] >= DET_IOU for v in vals),
                                median_iou_v2=float(np.median([v['iou_v2'] for v in vals])), median_iou_v1=float(np.median([v['iou_selection_v1'] for v in vals])))
        summary[split] = dict(windows=sum(r['split'] == split for r in res), targets=len(rows), pass_v2=sum(v['iou_v2'] >= DET_IOU for _, v in rows),
                              failing=[dict(window=r['name'], family=f, iou_v2=v['iou_v2'], iou_v1=v['iou_selection_v1']) for r in res if r['split'] == split
                                       for f, v in r['families'].items() if v['iou_v2'] < DET_IOU], per_family=per)
    (args.root / 'verify-v2-t0.json').write_text(json.dumps(dict(rule=f'COCO detector on the render-v2 canvas frame at t0, IoU >= {DET_IOU} with the stored '
                                                                     't0 box of each family target (right class), as the selection check', summary=summary), indent=1) + '\n')
    write_note(args.root)
    print(json.dumps({s: {k: summary[s][k] for k in ('windows', 'targets', 'pass_v2')} for s in summary}))


def phase_select(args):
    root = args.root
    splits, _, smoke = cf.split_logs()
    jobs = [(str(p), s) for s in ('DEV', 'TEST') for p in (splits[s][:args.limit_logs] if args.limit_logs else splits[s])]
    cache = root / 'scan'
    results, todo = [], []
    for job in jobs:
        f = cache / job[1] / f'{Path(job[0]).name}.json'
        if f.exists():
            res = json.loads(f.read_text())
            if res.get('rules') == RULES_VERSION and not res.get('error'):
                results.append(res); continue
        todo.append(job)
    print(f'select: {len(jobs)} logs, {len(results)} cached, {len(todo)} to scan', flush=True)
    t0 = time.time()
    with Pool(args.workers, initializer=cf.init_worker) as pool:
        for i, res in enumerate(pool.imap_unordered(scan_log, todo, chunksize=1)):
            results.append(res)
            (cache / res['split']).mkdir(parents=True, exist_ok=True)
            (cache / res['split'] / f"{res['log']}.json").write_text(json.dumps(cf.clean(res)))
            if res.get('error'):
                print('ERROR', res['log'], res['error'], flush=True)
            if (i + 1) % 25 == 0 or i + 1 == len(todo):
                print(f'{i + 1}/{len(todo)} scanned, {time.time() - t0:.0f} s', flush=True)
    errors = [dict(log=r['log'], split=r['split'], error=r['error']) for r in results if r.get('error')]
    good = [r for r in results if not r.get('error')]
    order = {Path(j[0]).name: k for k, j in enumerate(jobs)}
    good.sort(key=lambda r: order[r['log']])
    sel = select(good)
    names = {s: [w['name'] for w in sel[s]['windows']] for s in sel}
    for s in sel:
        assert len(set(names[s])) == len(names[s]), f'{s}: window names collide'
    assert not set(names['DEV']) & set(names['TEST'])
    for s in ('DEV', 'TEST'):
        (root / f'names-{s.lower()}.txt').write_text('\n'.join(names[s]) + '\n')
        write_csv(root / f'families-{s.lower()}.csv', FAMILY_FIELDS, [row for w in sel[s]['windows'] for row in family_rows(w)])
    counts = selection_counts(sel)
    me = Path(__file__).read_bytes()
    selection = dict(version=RULES_VERSION, created=datetime.datetime.now().isoformat(timespec='seconds'),
                     script='scripts/make_benchmark_windows.py', script_sha256=hashlib.sha256(me).hexdigest(),
                     imports_sha256={p: hashlib.sha256((Path(__file__).resolve().parent / p).read_bytes()).hexdigest()
                                     for p in ('count_scenario_families.py', 'make_wm_calibration_windows.py')},
                     limit_logs=args.limit_logs,
                     splits=dict(DEV='train logs at positions 0-199 of the seed-2026 permutation',
                                 TEST='val minus the 3 smoke-test logs, plus train positions 200-449', smoke_logs_excluded=smoke),
                     geometry=geometry_block(), timing=timing_block(), rules=rules_block(), counts=counts, scan_errors=errors,
                     map_errors=[dict(log=r['log'], error=r['map_error']) for r in good if r.get('map_error')],
                     detector_calls=sum(r.get('n_detector', 0) for r in good), scan_cpu_seconds=sum(r.get('seconds', 0) for r in good),
                     windows={s: [{k: w[k] for k in ('name', 'log', 'offset_s', 'anchor_timestamp_ns', 'rank', 'score', 'roles')} for w in sel[s]['windows']]
                              for s in sel},
                     role_weights={s: sel[s]['weight'] for s in sel})
    (root / 'selection.json').write_text(json.dumps(cf.clean(selection), indent=1) + '\n')
    write_note(root)
    print(json.dumps(dict(windows={s: counts[s]['windows'] for s in counts}, errors=len(errors))))


def geometry_block(for_render=False):
    if for_render:
        g = geometry_block(); g['resampling'] = RESAMPLE_V2; g['resampling_of_selection_detector_frames'] = WARP; g['render_version'] = RENDER_VERSION
        return g
    return dict(source=dict(camera='ring_front_center', width=SRC_W, height=SRC_H),
                nano=dict(rows=list(NANO_ROWS), cols=[0, SRC_W], size=[NANO_W, NANO_H], scale=NANO_W / SRC_W, affine=NANO_AFFINE.tolist(),
                          note="Cosmos3 256-tier '16,9' bucket (VIDEO_RES_SIZE_INFO -> 320x192); other inputs are centre-cropped by the framework"),
                canvas=dict(rows=list(BAND_ROWS), cols=[0, SRC_W], size=[CANVAS_W, CANVAS_H], scale=CANVAS_W / SRC_W, affine=CANVAS_AFFINE.tolist(),
                            note='the measurement band, common to all models; the affine maps source rows 797..1668.875 onto the 288 canvas rows'),
                resampling=WARP)


def timing_block():
    return dict(anchor='nearest LiDAR-annotation sweep to log start + offset; offsets 2.5, 3.0, ... <= log end - 6.1 s (log start/end = first/last sweep)',
                ticks='t0 + k * 0.1 s, k = -20..60; each takes the nearest 20 Hz ring_front_center frame (ties to the earlier), |offset| <= 50 ms '
                      'required for every tick', list_20hz='every camera frame with timestamp in [t0 - 2.0 s - 50 ms, t0 + 6.0 s + 50 ms]',
                videos=dict(history='nano/history.mp4: ticks k = -20..0 (21 frames, 10 Hz)', nano='nano/real.mp4: k = -20..60 (81 frames)',
                            canvas='canvas/real.mp4: k = -20..60 (81 frames)', encoder='libx264rgb -crf 0 -preset medium, rgb24, 10 fps '
                            '(make_wm_calibration_windows.write_video, as pad_anchor_clips.py); decoded frames checked equal to the rendered ones'))


def rules_block():
    return dict(families={
        'lead': 'the lead (nearest vehicle with box centre x > 0 and |y| < 1.8 m, no fallback); rear face (camera depth of the box centre - L/2) '
                '5-40 m; heading difference < 30 deg; speed > 1 m/s',
        'lead_stop': 'the lead; rear face 5-40 m; speed < 0.5 m/s, or mean acceleration over the last 0.5 s < -1 m/s^2 with heading difference < 30 deg',
        'startup': 'ego speed < 0.5 m/s and the lead at rear face 5-15 m with speed < 0.5 m/s',
        'cutin': 'nearest vehicle with heading difference < 30 deg, speed > 1 m/s, 1.8 <= |y| <= 5.5 m, -2 <= x <= 40 m',
        'ped': 'nearest PEDESTRIAN with x 3-30 m, |y| < 10 m (count rule F5)',
        'oncoming': 'nearest vehicle with heading difference > 150 deg, x 5-60 m, |y| < 15 m, speed > 2 m/s (F3)',
        'crossing': 'nearest vehicle with heading difference 45-135 deg, x 5-50 m, |y| < 20 m, speed > 1 m/s (F4)',
        'rider': 'nearest BICYCLIST or MOTORCYCLIST with x 3-40 m, |y| < 10 m (F6)',
        'ego_inter': 'target = the ego; ego_approach (F8: forward line x 0.5-30 m, y = -1/0/+1 m meets an is_intersection lane segment, the ego '
                     'origin not on one, ego speed > 2 m/s) or ego_stop_inter (ego speed < 0.5 m/s, an intersection lane on the forward line '
                     'within 15 m or under the ego origin)'},
        strata={'cutin_onset': 'the cutin target moves toward the ego across its own map lane at >= 0.2 m/s over the last 1 s (F2b_lane rule)',
                'ped_road': 'the ped target is on the map drivable area (devkit raster) or inside a pedestrian crossing',
                'ped_moving': 'the ped target moves > 0.5 m/s', 'ego_approach': 'F8 as above', 'ego_stop_inter': 'as above'},
        common='ego frame at t0 (x forward, y left); positions = box centres; velocities = mean over the last 1 s of the track (world motion in '
               't0 axes); heading difference = |box yaw| in the t0 frame; ego speed = mean over the last 1 s of poses; strata are flags of the '
               'family target; families may share a target',
        visibility=dict(in_band=f'box inside columns [{MARGIN_PX}, {SRC_W - MARGIN_PX}] and rows [{BAND_ROWS[0] + MARGIN_PX}, {BAND_ROWS[1] - MARGIN_PX}] (source px)',
                        height=f'>= {MIN_H_PX} source px (20 px in 320x192)', lidar_points=f'>= {MIN_POINTS}',
                        detector=f'COCO Faster R-CNN v2 ({mw.WEIGHTS.name}), score >= {DET_SCORE}, IoU >= {DET_IOU} with the target box on the t0 canvas frame, '
                                 'class vehicle: car/bus/truck, ped: person, rider: bicycle/motorcycle'),
        selection=dict(score='sum over the roles present at the anchor of 1/sqrt(n_role); n_role = candidate anchors of the role in the split; '
                             f'roles = {list(SCORE_ROLES)}', first='highest score (ties: earlier anchor)',
                       second=f'highest score among anchors >= {MIN_SEP_S} s from the first that hold a rare role absent from the first; rare = {list(RARE)}',
                       max_windows_per_log=MAX_WINDOWS),
        past_only='TEST: annotation and pose tables read with a timestamp filter <= the last candidate anchor of the log; lookups asserted <= t0; '
                  'no target or ego position after t0 read or computed; no gt.json')


# ---------------------------------------------------------------------------------------------------------------------
def decode(path, w, h):
    raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(path), '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'])
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)


def corners_of(row):
    from scipy.spatial.transform import Rotation
    R = Rotation.from_quat([row.qx, row.qy, row.qz, row.qw]).as_matrix(); L, W, H = row.length_m, row.width_m, row.height_m
    return np.array([[sx * L / 2, sy * W / 2, sz * H / 2] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) @ R.T + np.array([row.tx_m, row.ty_m, row.tz_m])


def project_corners(corners_ego, cam_T_ego, K):
    """As make_wm_calibration_windows.project, for corners already in the ego frame of the image."""
    in_cam = (cam_T_ego @ np.hstack([corners_ego, np.ones((8, 1))]).T)[:3].T
    if not (in_cam[:, 2] > 0.5).all():
        return None
    uv = (K @ in_cam.T).T; uv = uv[:, :2] / uv[:, 2:3]
    return [float(uv[:, 0].min()), float(uv[:, 1].min()), float(uv[:, 0].max()), float(uv[:, 1].max())]


def dev_gt(log, rec, tick_frames):
    """DEV only: per family target, positions in the t0 ego frame and canvas boxes per tick; ego poses at the frame times."""
    assert log.is_dev, 'gt.json is written for DEV windows only'
    from av2.geometry.geometry import mat_to_xyz
    from scipy.spatial.transform import Rotation
    anchor = rec['anchor_timestamp_ns']; t0_T_city = log.poses[anchor].inverse()

    def pose_near(t):
        i = int(np.argmin(np.abs(log.poses_t - t))); return int(log.poses_t[i]), log.poses[int(log.poses_t[i])]
    ego = []
    for k, f_ns in tick_frames:
        pt, p = pose_near(f_ns); rel = t0_T_city.compose(p)
        ego.append(dict(k=k, frame_timestamp_ns=f_ns, pose_timestamp_ns=pt, xyz_m=rel.translation.tolist(),
                        yaw_rad=float(mat_to_xyz(rel.rotation[None])[0, 2]), matrix_3x4=np.hstack([rel.rotation, rel.translation[:, None]]).round(6).tolist()))
    targets = {}
    for fam, t in rec['targets'].items():
        if fam == 'ego_inter':
            targets[fam] = dict(target='ego', see='ego'); continue
        uuid = t['track_uuid']; ts = log.tracks[uuid][0]; per = []
        for k, f_ns in tick_frames:
            tick = anchor + k * TICK_NS; j = int(np.argmin(np.abs(ts - tick)))
            if abs(int(ts[j]) - tick) > TOL_NS:
                per.append(dict(k=k, annotated=False)); continue
            s = int(ts[j]); g = log.by_t[s]; r = g[g.track_uuid == uuid].iloc[0]
            t0_T_s = t0_T_city.compose(log.poses[s]); c = t0_T_s.transform_point_cloud(np.array([[r.tx_m, r.ty_m, r.tz_m]]))[0]
            yaw = float(mat_to_xyz((t0_T_s.rotation @ Rotation.from_quat([r.qx, r.qy, r.qz, r.qw]).as_matrix())[None])[0, 2])
            _, pose_f = pose_near(f_ns); f_T_s = pose_f.inverse().compose(log.poses[s])
            corners = f_T_s.transform_point_cloud(corners_of(r))  # ego motion between the sweep and the frame compensated
            src = project_corners(corners, log.cam_T_ego, log.K_full)
            flags = band_flags(src, int(r.num_interior_pts))
            per.append(dict(k=k, annotated=True, sweep_timestamp_ns=s, frame_timestamp_ns=f_ns, xyz_t0_m=c.round(4).tolist(), yaw_t0_rad=round(yaw, 5),
                            lidar_points=int(r.num_interior_pts), box_source=src, box_canvas=map_box(src, CANVAS_AFFINE), **flags,
                            visible=bool(flags['in_band'] and flags['height_ok'] and flags['points_ok'])))
        targets[fam] = dict(track_uuid=uuid, target_hash=t['target_hash'], category=t['category'], per_tick=per)
    return dict(frame='ego frame at t0 (AV2 egovehicle, x forward, y left, z up)', ticks=TICKS.tolist(), ego=ego, targets=targets,
                box_method='annotation cuboid at the nearest sweep (<= 50 ms from the tick), moved by the ego motion from the sweep time to the '
                           'frame time (nearest pose), projected with the camera at the frame time; visibility = in band + height + points (no detector)')


def render_window(job):
    w = job
    split = w['split']; out = ROOT_RUNTIME[0] / f"windows-{split.lower()}" / w['name']
    if (out / 'render.json').exists():
        done = json.loads((out / 'render.json').read_text())
        if done.get('render_version') == RENDER_VERSION:
            return done
    t_start = time.time()
    try:
        log_dir = Path(w['log_path'])
        images = sorted((log_dir / 'sensors/cameras/ring_front_center').glob('*.jpg')); stamps = np.array([int(p.stem) for p in images])
        anchor = w['anchor_timestamp_ns']
        pick, dt = nearest_frames(stamps, anchor + TICKS * TICK_NS)
        assert (np.abs(dt) <= TOL_NS).all(), 'camera gap'
        lo, hi = anchor - 2 * SEC_NS - LIST20_PAD_NS, anchor + 6 * SEC_NS + LIST20_PAD_NS
        list20 = [dict(timestamp_ns=int(s), path=str(images[i])) for i, s in enumerate(stamps) if lo <= s <= hi]
        cache, nano, canvas = {}, [], []
        for i in pick:
            if int(i) not in cache:
                cache[int(i)] = resample_v2(read_rgb(images[int(i)]))
            nano.append(cache[int(i)][0]); canvas.append(cache[int(i)][1])
        (out / 'nano').mkdir(parents=True, exist_ok=True); (out / 'canvas').mkdir(parents=True, exist_ok=True)
        clips = {'nano/history.mp4': (nano[:HISTORY_IDX + 1], NANO_W, NANO_H), 'nano/real.mp4': (nano, NANO_W, NANO_H), 'canvas/real.mp4': (canvas, CANVAS_W, CANVAS_H)}
        checks = {}
        for rel, (frames, fw, fh) in clips.items():
            tmp = out / (rel + '.tmp.mp4'); mw.write_video(frames, tmp); tmp.rename(out / rel)
            back = decode(out / rel, fw, fh)
            checks[rel] = dict(frames=len(frames), decoded=int(len(back)), identical=bool(len(back) == len(frames) and np.array_equal(back, np.stack(frames))),
                               bytes=(out / rel).stat().st_size)
            if not checks[rel]['identical']:
                raise RuntimeError(f'{rel}: decoded frames differ from the rendered ones')
        log = BenchLog(log_dir, split, with_map=False) if split == 'DEV' else None
        from av2.geometry.camera.pinhole_camera import PinholeCamera
        cam = PinholeCamera.from_feather(log_dir, 'ring_front_center')
        K = cam.intrinsics.K
        rec = w['record']
        tick_rows = [dict(k=int(k), tick_timestamp_ns=int(anchor + k * TICK_NS), frame_timestamp_ns=int(stamps[i]), offset_ms=round(float(d) / 1e6, 3),
                          path=str(images[i])) for k, i, d in zip(TICKS, pick, dt)]
        targets = {}
        for fam, t in rec['targets'].items():
            if fam == 'ego_inter':
                targets[fam] = dict(target='ego', **{s: t.get(s) for s in STRATA['ego_inter']}); continue
            targets[fam] = {k: t.get(k) for k in ('track_uuid', 'target_hash', 'category', 'x_m', 'y_m', 'dist_m', 'rear_face_m', 'heading_diff_deg', 'speed_mps',
                                                 'vx_mps', 'vy_mps', 'toward_mps', 'toward_lane_mps', 'accel_mps2', 'lidar_points', 'length_m', 'width_m', 'height_m',
                                                 'box_source', 'box_nano', 'box_canvas', 'det_iou', 'det_label', 'on_drivable', 'on_crosswalk', *STRATA.get(fam, ()))}
        window = dict(name=w['name'], split=split, log=w['log'], log_source=log_dir.parent.name, city=w.get('city'), rank=w['rank'], score=w['score'], roles=w['roles'],
                      anchor=dict(timestamp_ns=anchor, offset_s=w['offset_s'], offset_exact_s=rec['anchor_offset_exact_s'], log_first_sweep_ns=w['first_sweep_ns']),
                      ticks=tick_rows, frames_20hz=list20, geometry=geometry_block(for_render=True), render_version=RENDER_VERSION,
                      intrinsics=dict(K_source=K.tolist(), K_nano=(np.vstack([NANO_AFFINE, [0, 0, 1]]) @ K).tolist(),
                                      K_canvas=(np.vstack([CANVAS_AFFINE, [0, 0, 1]]) @ K).tolist(), ego_SE3_cam=cam.ego_SE3_cam.transform_matrix.tolist(),
                                      width_px=int(cam.intrinsics.width_px), height_px=int(cam.intrinsics.height_px)),
                      past_covariates={k: rec.get(k) for k in ('ego_speed_mps', 'ego_speed_05_mps', 'ego_accel_mps2', 'intersection_hit_x_m', 'ego_in_intersection')},
                      targets=targets, clips={k: v for k, v in checks.items()}, selection='make_benchmark_windows.py (past-only; see ../../selection.json)')
        (out / 'window.json').write_text(json.dumps(cf.clean(window), indent=1) + '\n')
        if split == 'DEV':
            gt = dev_gt(log, rec, [(int(k), int(stamps[i])) for k, i in zip(TICKS, pick)])
            (out / 'gt.json').write_text(json.dumps(cf.clean(gt)) + '\n')
        status = dict(name=w['name'], split=split, ok=True, render_version=RENDER_VERSION, checks=checks, repeated_frames=int(len(pick) - len(set(pick.tolist()))),
                      max_abs_offset_ms=float(np.abs(dt).max() / 1e6), history_frames_after_tick=int((dt[:HISTORY_IDX + 1] > 0).sum()),
                      t0_frame_offset_ms=float(dt[HISTORY_IDX] / 1e6), seconds=time.time() - t_start,
                      bytes=sum(p.stat().st_size for p in out.rglob('*') if p.is_file()))
        (out / 'render.json').write_text(json.dumps(status) + '\n')
        return status
    except Exception as exc:
        return dict(name=w['name'], split=split, ok=False, error=f'{type(exc).__name__}: {exc}', trace=traceback.format_exc())


ROOT_RUNTIME = [ROOT]


def phase_render(args):
    sel = json.loads((args.root / 'selection.json').read_text())
    splits = ['DEV', 'TEST'] if args.split == 'both' else [args.split.upper()]
    # full records come from the scan cache (past-only for TEST)
    jobs = []
    subset = set(args.names.read_text().split()) if args.names else None
    for split in splits:
        recs = {}
        for wv in sel['windows'][split]:
            if subset is not None and wv['name'] not in subset:
                continue
            if wv['log'] not in recs:
                recs[wv['log']] = json.loads((args.root / 'scan' / split / f"{wv['log']}.json").read_text())
            r = recs[wv['log']]
            a = next(x for x in r['anchors'] if x['offset_s'] == wv['offset_s'])
            jobs.append(dict(wv, split=split, log_path=r['path'], city=r.get('city'), first_sweep_ns=r['first_sweep_ns'], record=a))
    print(f'render: {len(jobs)} windows', flush=True)
    t0 = time.time(); results = []
    with Pool(args.workers, initializer=cf.init_worker) as pool:
        for i, res in enumerate(pool.imap_unordered(render_window, jobs, chunksize=1)):
            results.append(res)
            if not res.get('ok'):
                print('FAIL', res['name'], res.get('error'), flush=True)
            if (i + 1) % 50 == 0 or i + 1 == len(jobs):
                print(f'{i + 1}/{len(jobs)} rendered, {time.time() - t0:.0f} s', flush=True)
    prev = json.loads((args.root / 'render-status.json').read_text()) if (args.root / 'render-status.json').exists() else {}
    for split in splits:
        rs = [r for r in results if r['split'] == split]
        prev[split] = dict(names_file=str(args.names) if args.names else 'all selected windows', windows=len(rs), ok=sum(r.get('ok', False) for r in rs), failed=[dict(name=r['name'], error=r.get('error')) for r in rs if not r.get('ok')],
                           all_clips_identical=all(c['identical'] for r in rs if r.get('ok') for c in r['checks'].values()),
                           gt_json=sum((args.root / f'windows-{split.lower()}' / r['name'] / 'gt.json').exists() for r in rs),
                           bytes=sum(r.get('bytes', 0) for r in rs if r.get('ok')),
                           max_abs_frame_offset_ms=max((r['max_abs_offset_ms'] for r in rs if r.get('ok')), default=None),
                           windows_with_repeated_frames=sum(r.get('repeated_frames', 0) > 0 for r in rs if r.get('ok')),
                           history_frames_after_their_tick=sum(r.get('history_frames_after_tick', 0) for r in rs if r.get('ok')),
                           t0_frame_after_t0=sum(r.get('t0_frame_offset_ms', 0) > 0 for r in rs if r.get('ok')),
                           median_t0_frame_offset_ms=float(np.median([r['t0_frame_offset_ms'] for r in rs if r.get('ok')])) if rs else None)
    prev['updated'] = datetime.datetime.now().isoformat(timespec='seconds')
    prev['render_version'] = RENDER_VERSION; prev['resampling'] = RESAMPLE_V2
    for split in splits:
        prev[split]['render_version'] = RENDER_VERSION
        prev[split]['all_render_version_2'] = all(r.get('render_version') == RENDER_VERSION for r in results if r['split'] == split)
    (args.root / 'render-status.json').write_text(json.dumps(prev, indent=1) + '\n')
    sel_path = args.root / 'selection.json'; sel2 = json.loads(sel_path.read_text())
    if sel2.get('render') != RESAMPLE_V2:
        sel2['render'] = RESAMPLE_V2; sel_path.write_text(json.dumps(cf.clean(sel2), indent=1) + '\n')
    ok_all = bool(results) and all(r.get('ok') and r.get('render_version') == RENDER_VERSION and all(c['identical'] for c in r['checks'].values())
                                   for r in results)
    if args.done is not None:
        if ok_all:
            (args.root / args.done).write_text(json.dumps(dict(windows=len(results), render_version=RENDER_VERSION,
                                                               names_file=str(args.names) if args.names else 'all', finished=prev['updated'])) + '\n')
        else:
            print(f'NOT touching {args.done}: some windows failed', flush=True)
    write_note(args.root)
    print(json.dumps({s: {k: prev[s][k] for k in ('windows', 'ok', 'all_clips_identical', 'gt_json', 'bytes')} for s in splits}))


def phase_sheets(args):
    """A few PNG contact sheets of DEV windows (local only): canvas frames at 0/2/4/6 s with the gt boxes, and the Nano frame at t0."""
    import cv2
    sel = json.loads((args.root / 'selection.json').read_text())
    out = args.root / (args.sheets_dir or 'contact-sheets-dev'); out.mkdir(exist_ok=True)
    pool_dev = sel['windows']['DEV']
    if args.names:
        keep = set(args.names.read_text().split()); pool_dev = [w for w in pool_dev if w['name'] in keep]
    colours = {'lead': (255, 140, 0), 'lead_stop': (255, 0, 0), 'startup': (255, 0, 255), 'cutin': (0, 200, 255), 'ped': (0, 255, 0),
               'oncoming': (255, 255, 0), 'crossing': (160, 32, 240), 'rider': (255, 255, 255)}
    chosen, seen = [], set()
    for fam in FAMILIES:  # one window per family first, then fill
        for wv in pool_dev:
            if fam in wv['roles'] and wv['name'] not in seen:
                chosen.append(wv); seen.add(wv['name']); break
    for wv in pool_dev:
        if len(chosen) >= args.count:
            break
        if wv['name'] not in seen:
            chosen.append(wv); seen.add(wv['name'])
    per_sheet, sheets = 4, []
    for s0 in range(0, len(chosen), per_sheet):
        rows = []
        for wv in chosen[s0:s0 + per_sheet]:
            d = args.root / 'windows-dev' / wv['name']
            canvas = decode(d / 'canvas/real.mp4', CANVAS_W, CANVAS_H); nano = decode(d / 'nano/real.mp4', NANO_W, NANO_H)
            gt = json.loads((d / 'gt.json').read_text())
            tiles = []
            for k in (0, 20, 40, 60):
                im = np.ascontiguousarray(canvas[HISTORY_IDX + k].copy())
                for fam, t in gt['targets'].items():
                    if 'per_tick' not in t:
                        continue
                    p = t['per_tick'][HISTORY_IDX + k]
                    if p.get('annotated') and p.get('box_canvas'):
                        b = [int(round(v)) for v in p['box_canvas']]
                        cv2.rectangle(im, (b[0], b[1]), (b[2], b[3]), colours.get(fam, (255, 255, 255)), 1)
                        cv2.putText(im, fam, (b[0], max(8, b[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, colours.get(fam, (255, 255, 255)), 1)
                if k == 0:  # the stored t0 boxes of window.json (selection), dashed-looking second outline offset by 1 px
                    wj = json.loads((d / 'window.json').read_text())
                    for fam, t in wj['targets'].items():
                        if t.get('box_canvas'):
                            b = [int(round(v)) for v in t['box_canvas']]
                            cv2.rectangle(im, (b[0] - 1, b[1] - 1), (b[2] + 1, b[3] + 1), (255, 255, 255), 1)
                cv2.putText(im, f"t={k / 10:+.1f}s", (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                tiles.append(im)
            n0 = np.ascontiguousarray(nano[HISTORY_IDX].copy())  # band edges drawn in the Nano frame
            for row_src in BAND_ROWS:
                y = int(round(NANO_AFFINE[1, 1] * row_src + NANO_AFFINE[1, 2])); cv2.line(n0, (0, y), (NANO_W - 1, y), (255, 0, 0), 1)
            n0 = cv2.copyMakeBorder(n0, 0, CANVAS_H - NANO_H, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            cv2.putText(n0, 'nano t0 (band in red)', (4, NANO_H + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(n0, ' '.join(wv['roles'])[:60], (4, NANO_H + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1)
            cv2.putText(n0, wv['name'], (4, NANO_H + 46), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            rows.append(np.hstack([n0] + tiles))
        sheet = np.vstack(rows)
        path = out / f'sheet-{s0 // per_sheet + 1:02d}.png'
        cv2.imwrite(str(path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR)); sheets.append(dict(path=path.name, windows=[w['name'] for w in chosen[s0:s0 + per_sheet]]))
    (out / 'sheets.json').write_text(json.dumps(sheets, indent=1) + '\n')
    print(json.dumps(dict(sheets=len(sheets), windows=len(chosen))))


# ---------------------------------------------------------------------------------------------------------------------
def write_note(root):
    sel = json.loads((root / 'selection.json').read_text())
    rs = json.loads((root / 'render-status.json').read_text()) if (root / 'render-status.json').exists() else {}
    c, g = sel['counts'], sel['geometry']
    L = [f"# Benchmark windows bench-v1 (band geometry), {sel['created'][:10]}", '',
         f"Built by `{sel['script']}` (sha256 {sel['script_sha256'][:16]}...), importing `count_scenario_families.py` and "
         "`make_wm_calibration_windows.py` (unchanged). Selection is past-only on both splits; DEV windows carry `gt.json` for "
         'instrument validation, TEST windows do not.', '',
         '## Geometry', '',
         f"- Source: ring_front_center, {SRC_W}x{SRC_H}.",
         f"- Nano frame: full width, rows [{NANO_ROWS[0]}, {NANO_ROWS[1]}) -> {NANO_W}x{NANO_H} (scale {NANO_W / SRC_W:.6f}); {g['nano']['note']}.",
         f"- Canvas (measurement band, all models): full width, rows [{BAND_ROWS[0]}, {BAND_ROWS[1]}) -> {CANVAS_W}x{CANVAS_H} (scale {CANVAS_W / SRC_W:.6f}).",
         f"- Resampling, render_version {RENDER_VERSION} (clips): {RESAMPLE_V2['method']}; Nano {RESAMPLE_V2['nano']}; canvas {RESAMPLE_V2['canvas']}; "
         f"pixel centres {RESAMPLE_V2['pixel_centres']}.",
         f"- Why version 2: {RESAMPLE_V2['reason']}. Changed {RESAMPLE_V2['changed']}. Selection: {RESAMPLE_V2['selection']}.",
         f"- Version 1 (used by the selection's t0 detector check): `{WARP['call']}`, effective filter {WARP['effective_filter']}.",
         f"- In the Nano frame the band spans rows {NANO_AFFINE[1, 1] * BAND_ROWS[0] + NANO_AFFINE[1, 2]:.1f}-{NANO_AFFINE[1, 1] * BAND_ROWS[1] + NANO_AFFINE[1, 2]:.1f} of 192.", '',
         '## Timing', '', f"- Anchors: {sel['timing']['anchor']}.", f"- Ticks: {sel['timing']['ticks']}.", f"- 20 Hz list: {sel['timing']['list_20hz']}.",
         f"- Clips: {'; '.join(sel['timing']['videos'].values())}.", '',
         '## Families, visibility, selection', '']
    L += [f"- **{k}**: {v}" for k, v in sel['rules']['families'].items()]
    L += [f"- stratum **{k}**: {v}" for k, v in sel['rules']['strata'].items()]
    L += [f"- Common: {sel['rules']['common']}.", f"- Visibility (every non-ego target at t0): {'; '.join(f'{k}: {v}' for k, v in sel['rules']['visibility'].items())}.",
          f"- Selection: {sel['rules']['selection']['score']}; first: {sel['rules']['selection']['first']}; second: {sel['rules']['selection']['second']}; "
          f"at most {sel['rules']['selection']['max_windows_per_log']} per log.", f"- Past-only: {sel['rules']['past_only']}.", '',
          '## Counts', '']
    L += ['| | DEV | TEST |', '|---|---|---|']
    for key, label in (('logs', 'logs'), ('logs_with_candidates', 'logs with >= 1 candidate anchor'), ('windows', 'windows'),
                       ('second_windows', 'second windows')):
        L.append(f"| {label} | {c['DEV'][key]} | {c['TEST'][key]} |")
    for k in ('0', '1', '2'):
        L.append(f"| logs with {k} window(s) | {c['DEV']['windows_per_log'][k]} | {c['TEST']['windows_per_log'][k]} |")
    L += [f"| anchors on the grid / available / with a role | {c['DEV']['grid_anchors']} / {c['DEV']['available_anchors']} / {c['DEV']['candidate_anchors']} | "
          f"{c['TEST']['grid_anchors']} / {c['TEST']['available_anchors']} / {c['TEST']['candidate_anchors']} |", '',
          'Windows per family and stratum (all windows; first windows only in brackets), and candidate anchors n_role:', '',
          '| role | DEV windows | TEST windows | DEV n_role | TEST n_role |', '|---|---|---|---|---|']
    for role in FAMILIES + ALL_STRATA:
        name = role if role in FAMILIES else f'  stratum {role}'
        L.append(f"| {name} | {c['DEV']['per_role'][role]} [{c['DEV']['per_role_first_window_only'][role]}] | {c['TEST']['per_role'][role]} "
                 f"[{c['TEST']['per_role_first_window_only'][role]}] | {c['DEV']['candidate_anchors_per_role'].get(role) if c['DEV']['candidate_anchors_per_role'].get(role) is not None else '-'} | "
                 f"{c['TEST']['candidate_anchors_per_role'].get(role) if c['TEST']['candidate_anchors_per_role'].get(role) is not None else '-'} |")
    L += ['', f"Scan: {sel['detector_calls']} detector calls, {sel['scan_cpu_seconds'] / 3600:.2f} worker CPU-hours; errors: {sel['scan_errors'] or 'none'}; "
          f"map errors: {sel['map_errors'] or 'none'}.", '']
    gen = sel.get('generation_subsample')
    if gen:
        L += ['## Generation subsamples (start at about PAWBench size)', '', f"- Rule: {gen['rule']}",
              f"- DEV: seed {gen['splits']['DEV']['seed']}, `{gen['splits']['DEV']['file']}`, targets {gen['splits']['DEV']['targets']}; "
              f"order {gen['splits']['DEV']['order']}.",
              f"- TEST: seed {gen['splits']['TEST']['seed']}, `{gen['splits']['TEST']['file']}`, targets {gen['splits']['TEST']['targets']}; "
              f"order {gen['splits']['TEST']['order']}.", '',
              '| family | DEV subsample | DEV pool | TEST subsample | TEST pool |', '|---|---|---|---|---|']
        for role in FAMILIES + ('cutin_onset', 'ped_road', 'ped_moving', 'ego_stop_inter'):
            pool = lambda sp: c[sp]['per_role'][role]
            L.append(f"| {role if role in FAMILIES else '  stratum ' + role} | {gen['splits']['DEV']['per_family'][role]} | {pool('DEV')} | "
                     f"{gen['splits']['TEST']['per_family'][role]} | {pool('TEST')} |")
        L += [f"| **total windows** | {gen['splits']['DEV']['total']} | {c['DEV']['windows']} | {gen['splits']['TEST']['total']} | {c['TEST']['windows']} |",
              '', f"Shortfalls (target not reachable at 1 window per log): DEV {gen['splits']['DEV']['shortfall'] or 'none'}; "
              f"TEST {gen['splits']['TEST']['shortfall'] or 'none'}.", '']
        if 'extension' in gen['splits']['DEV']:
            ed, et = gen['splits']['DEV'], gen['splits']['TEST']
            L += ['### Extension to the final generation lists', '',
                  f"- Rule: keep the draw as the prefix, then {ed['extension']['rng']}. DEV seed {ed['extension']['seed']} -> {ed['extension']['total']} "
                  f"windows; TEST seed {et['extension']['seed']} -> {et['extension']['total']} windows.", '',
                  '| family | DEV final (`names-dev-gen.txt`) | TEST final (`names-test-gen.txt`) |', '|---|---|---|']
            for role in FAMILIES + ('cutin_onset', 'ped_road', 'ped_moving', 'ego_stop_inter', 'ego_approach'):
                L.append(f"| {role if role in FAMILIES else '  stratum ' + role} | {ed['final_per_family'][role]} | {et['final_per_family'][role]} |")
            L += [f"| **total windows (logs)** | {ed['final_total']} ({ed['final_logs']}) | {et['final_total']} ({et['final_logs']}) |", '',
                  f"Draw prefix: DEV {ed['total']}, TEST {et['total']}; added by the extension: DEV {ed['extension']['added']}, TEST {et['extension']['added']}.", '']
    if rs:
        L += ['## Render', '', f"DEV: {rs.get('DEV', {}).get('names_file', '-')}; TEST: {rs.get('TEST', {}).get('names_file', '-')}.", '',
              '| | DEV | TEST |', '|---|---|---|']
        for key, label in (('windows', 'windows'), ('ok', 'rendered'), ('all_clips_identical', 'decoded == rendered (all clips)'), ('gt_json', 'gt.json'),
                           ('windows_with_repeated_frames', 'windows with a repeated camera frame'), ('max_abs_frame_offset_ms', 'max abs(frame - tick) (ms)'),
                           ('median_t0_frame_offset_ms', 'median t0 frame - t0 (ms)'), ('t0_frame_after_t0', 'windows whose t0 frame is after t0'),
                           ('history_frames_after_their_tick', 'history frames after their tick (all windows)')):
            L.append(f"| {label} | {rs.get('DEV', {}).get(key, '-')} | {rs.get('TEST', {}).get(key, '-')} |")
        gb = lambda s: f"{rs.get(s, {}).get('bytes', 0) / 1e9:.2f} GB" if s in rs else '-'
        L += [f"| disk | {gb('DEV')} | {gb('TEST')} |", '']
        fails = [f for s in ('DEV', 'TEST') for f in rs.get(s, {}).get('failed', [])]
        L += [f"Failures: {fails or 'none'}.", f"render_version: {rs.get('render_version', 1)} (all DEV windows v2: "
              f"{rs.get('DEV', {}).get('all_render_version_2', '-')}; all TEST windows v2: {rs.get('TEST', {}).get('all_render_version_2', '-')}).", '']
        marks = [m for m in ('render-v2-dev-gen.done', 'render-v2-test-gen.done') if (root / m).exists()]
        L += [f"Done markers: {', '.join(marks) or 'none'}.", '']
    if (root / 'verify-v2-t0.json').exists():
        v = json.loads((root / 'verify-v2-t0.json').read_text())
        L += ['## Check of the t0 boxes on render-v2 frames', '', f"- {v['rule']}.", '', '| split | windows | targets | pass (IoU >= 0.3) |', '|---|---|---|---|']
        for sp, d in v['summary'].items():
            L.append(f"| {sp} | {d['windows']} | {d['targets']} | {d['pass_v2']} |")
        L += ['', 'Failing (window, family, IoU v2 / IoU at selection v1): ' + ('; '.join(f"{x['window']} {x['family']} {x['iou_v2']:.2f}/{x['iou_v1']:.2f}"
                                                                                    for sp in v['summary'] for x in v['summary'][sp]['failing']) or 'none') + '.', '']
    extra = root / 'notes-handwritten.md'
    if extra.exists():
        L += ['', extra.read_text().rstrip(), '']
    (root / 'NOTE.md').write_text('\n'.join(L))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--phase', required=True, choices=['select', 'draw', 'extend', 'render', 'sheets', 'verify'])
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--limit-logs', type=int, help='select: first N logs per split (smoke test; use a scratch --root)')
    parser.add_argument('--split', default='both', choices=['dev', 'test', 'both'])
    parser.add_argument('--count', type=int, default=12, help='sheets: number of DEV windows')
    parser.add_argument('--names', type=Path, help='render/sheets: only these window names (e.g. names-test-gen.txt)')
    parser.add_argument('--done', help='render: marker file (relative to --root) written when every window rendered as version 2')
    parser.add_argument('--sheets-dir', help='sheets: output folder under --root (default contact-sheets-dev)')
    args = parser.parse_args()
    if args.workers > 12:
        raise SystemExit('at most 12 workers on the shared machine')
    os.environ.setdefault('OMP_NUM_THREADS', '2'); os.environ['CUDA_VISIBLE_DEVICES'] = ''
    ROOT_RUNTIME[0] = args.root
    if args.phase == 'select':
        if (args.root / 'selection.json').exists():
            raise SystemExit(f'{args.root}/selection.json exists; selection is written once')
        args.root.mkdir(parents=True, exist_ok=True)
        phase_select(args)
    elif args.phase == 'draw':
        phase_draw(args)
    elif args.phase == 'extend':
        phase_extend(args)
    elif args.phase == 'verify':
        phase_verify(args)
    elif args.phase == 'render':
        if args.split == 'both' and args.names:
            raise SystemExit('--names needs --split dev or --split test')
        phase_render(args)
    else:
        phase_sheets(args)


if __name__ == '__main__':
    main()
