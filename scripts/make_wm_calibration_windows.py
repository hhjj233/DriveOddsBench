#!/usr/bin/env python3
"""Real AV2 front-camera windows for the world-model calibration study, chosen from the past only.

A window is valid for a calibration test only if its choice does not depend on the real future (otherwise the real
outcome is a selected draw and its rank among generated futures is not uniform even for a calibrated model). So every
selection rule here uses the anchor time and before:
  - an annotated vehicle ahead at the anchor: |lateral| < 1.8 m, the nearest such one, its rear face 5-25 m from the
    camera (camera depth of the box centre minus half its length), >= 5 lidar points inside its box;
  - its projected box lies inside the 256 x 320 frame (2 px margin) and is >= 20 px tall;
  - the COCO detector of the measuring instrument finds it in the anchor frame (vehicle class, score >= 0.4,
    IoU >= 0.3 with the projected box).
The first anchor of a log that passes (2.5 s into the log, then every 1 s, while 6 s of future remain) is its
window; one window per log. The annotation over the real future is written for checking the instrument only.

Per window: history.mp4 (21 frames, 2 s at 10 Hz, causal images), real.mp4
(81 frames), lead.json (per tick from -2 s to +6 s: ego-frame position, camera depth, projected box; anchor-time
covariates for strata: ego speed and acceleration from poses at or before the anchor, lead closing speed from the
last 0.5 s, city). Lossless RGB, 256 x 320 through AFFINE. Read only on the dataset; CPU.

  make_wm_calibration_windows.py --split-root DIR --output DIR [--shuffle-seed S --first K --count N] [--workers W]
"""
from __future__ import annotations

import argparse, json, os, subprocess
from multiprocessing import Pool
from pathlib import Path

import numpy as np

AFFINE = np.array([[0.1513978494623656, 0.0, 10.242365591397853], [0.0, 0.1513978494623656, 4.544301075268808]])  # 1550 x 2048 -> 256 x 320
VEHICLES = {'REGULAR_VEHICLE', 'LARGE_VEHICLE', 'BUS', 'BOX_TRUCK', 'TRUCK', 'TRUCK_CAB', 'ARTICULATED_BUS', 'SCHOOL_BUS', 'VEHICULAR_TRAILER'}
HISTORY, FUTURE = 20, 60  # ticks of 0.1 s before and after the anchor
WEIGHTS = Path(os.environ.get('DOB_ROOT', '.'), 'cache/torch/hub/checkpoints/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth')
DETECT = dict(labels=(3, 6, 8), score=0.4, iou=0.3)  # COCO car, bus, truck
GAP_M, LATERAL_M, MIN_BOX_PX, MIN_POINTS = (5.0, 25.0), 1.8, 20.0, 5
_model = None


def write_video(frames, path):
    h, w = frames[0].shape[:2]
    proc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}', '-r', '10', '-i', '-', '-c:v', 'libx264rgb', '-crf', '0',
                             '-preset', 'medium', '-pix_fmt', 'rgb24', '-r', '10', '-video_track_timescale', '10240', '-map_metadata', '-1', '-movflags', '+faststart', str(path)],
                            stdin=subprocess.PIPE)
    for f in frames:
        proc.stdin.write(np.ascontiguousarray(f).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f'ffmpeg failed for {path}')


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1); u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def detections(frame):
    global _model
    import torch, torchvision
    if _model is None:
        torch.set_num_threads(2)
        _model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None)
        _model.load_state_dict(torch.load(WEIGHTS, map_location='cpu', weights_only=True)); _model.eval()
    with torch.no_grad():
        out = _model([torch.as_tensor(np.array(frame)).permute(2, 0, 1).float() / 255.0])[0]
    return [(b.tolist(), float(s)) for b, s, l in zip(out['boxes'], out['scores'], out['labels']) if int(l) in DETECT['labels'] and float(s) >= DETECT['score']]


def project(r, cam_T_ego, K):
    from scipy.spatial.transform import Rotation
    centre = np.array([r.tx_m, r.ty_m, r.tz_m, 1.0]); R = Rotation.from_quat([r.qx, r.qy, r.qz, r.qw]).as_matrix(); L, W, H = r.length_m, r.width_m, r.height_m
    corners = np.array([[sx * L / 2, sy * W / 2, sz * H / 2] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) @ R.T + centre[:3]
    in_cam = (cam_T_ego @ np.hstack([corners, np.ones((8, 1))]).T)[:3].T; centre_cam = (cam_T_ego @ centre)[:3]
    box = None
    if (in_cam[:, 2] > 0.5).all():
        uv = (K @ in_cam.T).T; uv = uv[:, :2] / uv[:, 2:3]
        box = [float(uv[:, 0].min()), float(uv[:, 1].min()), float(uv[:, 0].max()), float(uv[:, 1].max())]
    return float(centre_cam[2]), box


def speed_at(poses_t, poses_xyz, t0, t1):
    """Mean speed over [t0, t1] (ns) from the ego poses inside it, or None."""
    i0, i1 = np.searchsorted(poses_t, t0), np.searchsorted(poses_t, t1, side='right') - 1
    if i1 <= i0:
        return None
    return float(np.linalg.norm(poses_xyz[i1] - poses_xyz[i0]) / ((poses_t[i1] - poses_t[i0]) / 1e9))


def one_log(job):
    log_dir, out_root = Path(job[0]), Path(job[1])
    import cv2, pandas as pd
    from av2.geometry.camera.pinhole_camera import PinholeCamera
    all_ann = pd.read_feather(log_dir / 'annotations.feather'); ann = all_ann[all_ann.category.isin(VEHICLES)]
    sweeps = np.sort(all_ann.timestamp_ns.unique())
    images = sorted((log_dir / 'sensors/cameras/ring_front_center').glob('*.jpg')); stamps = np.array([int(p.stem) for p in images])
    cam = PinholeCamera.from_feather(log_dir, 'ring_front_center'); cam_T_ego = np.linalg.inv(cam.ego_SE3_cam.transform_matrix)
    K = np.vstack([AFFINE, [0, 0, 1]]) @ cam.intrinsics.K
    poses = pd.read_feather(log_dir / 'city_SE3_egovehicle.feather').sort_values('timestamp_ns'); poses_t = poses.timestamp_ns.to_numpy(); poses_xyz = poses[['tx_m', 'ty_m', 'tz_m']].to_numpy()
    city = next((p.name.split('____')[1].split('_')[0] for p in (log_dir / 'map').glob('log_map_archive_*.json')), None)
    tried = []

    def image_at(t, causal):
        choice = np.flatnonzero(stamps <= t)[-1] if causal else int(np.argmin(np.abs(stamps - t)))
        if abs(int(stamps[choice]) - int(t)) > 50_000_000:
            return None
        img = cv2.cvtColor(cv2.imread(str(images[choice])), cv2.COLOR_BGR2RGB)
        return cv2.warpAffine(img, AFFINE, (256, 320), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    for offset in np.arange(2.5, (sweeps[-1] - sweeps[0]) / 1e9 - 6.0 + 1e-6, 1.0):
        anchor = sweeps[np.argmin(np.abs(sweeps - (sweeps[0] + offset * 1e9)))]
        ticks = anchor + np.arange(-HISTORY, FUTURE + 1) * 100_000_000
        if ticks[-1] > sweeps[-1] + 50_000_000 or ticks[0] < stamps[0]:
            continue
        now = ann[(ann.timestamp_ns == anchor) & (ann.tx_m > 0) & (ann.ty_m.abs() < LATERAL_M)]
        if now.empty:
            tried.append((round(float(offset), 1), 'no vehicle ahead')); continue
        lead = now.sort_values('tx_m').iloc[0]
        depth, box = project(lead, cam_T_ego, K); face = depth - lead.length_m / 2
        reason = None
        if not GAP_M[0] <= face <= GAP_M[1]:
            reason = f'rear face {face:.1f} m'
        elif lead.num_interior_pts < MIN_POINTS:
            reason = f'{int(lead.num_interior_pts)} lidar points'
        elif box is None or box[0] < 2 or box[1] < 2 or box[2] > 254 or box[3] > 318:
            reason = 'box not inside the frame'
        elif box[3] - box[1] < MIN_BOX_PX:
            reason = f'box {box[3] - box[1]:.0f} px tall'
        if reason is None:
            frame0 = image_at(anchor, causal=True)
            if frame0 is None:
                reason = 'no anchor image'
            else:
                best = max((iou(b, box) for b, _ in detections(frame0)), default=0.0)
                if best < DETECT['iou']:
                    reason = f'detector IoU {best:.2f}'
        if reason is not None:
            tried.append((round(float(offset), 1), reason)); continue
        frames = [image_at(t, causal=k <= HISTORY) for k, t in enumerate(ticks)]
        if any(f is None for f in frames):
            tried.append((round(float(offset), 1), 'camera gap')); continue
        track = ann[ann.track_uuid == lead.track_uuid]; rows = []
        for k, t in enumerate(ticks):
            sweep = sweeps[np.argmin(np.abs(sweeps - t))]; r = track[track.timestamp_ns == sweep]
            if abs(int(sweep) - int(t)) > 50_000_000 or r.empty:
                rows.append(None); continue
            r = r.iloc[0]; d, b = project(r, cam_T_ego, K)
            rows.append(dict(tick=k - HISTORY, time_s=round((k - HISTORY) * 0.1, 1), ego_xyz_m=[float(r.tx_m), float(r.ty_m), float(r.tz_m)], camera_depth_m=d,
                             rear_face_depth_m=d - float(r.length_m) / 2, distance_m=float(np.linalg.norm([r.tx_m, r.ty_m, r.tz_m])), box_256x320=b))
        prev = rows[HISTORY - 5]  # 0.5 s before the anchor
        closing = (prev['rear_face_depth_m'] - rows[HISTORY]['rear_face_depth_m']) / 0.5 if prev is not None else None
        v0, v1 = speed_at(poses_t, poses_xyz, anchor - 500_000_000, anchor), speed_at(poses_t, poses_xyz, anchor - 1_000_000_000, anchor - 500_000_000)
        covariates = dict(ego_speed_mps=v0, ego_accel_mps2=(v0 - v1) / 0.5 if v0 is not None and v1 is not None else None, lead_closing_speed_mps=closing,
                          rear_face_gap_m=face, lead_box_px=box[3] - box[1], lead_category=lead.category, city=city)
        name = f'{log_dir.name[:8]}-t{offset:.1f}'; out = out_root / name; out.mkdir()
        write_video(frames[:HISTORY + 1], out / 'history.mp4'); write_video(frames, out / 'real.mp4')
        (out / 'lead.json').write_text(json.dumps(dict(log=log_dir.name, split=log_dir.parent.name, anchor_timestamp_ns=int(anchor), anchor_offset_s=float(offset),
                                                       track_uuid=lead.track_uuid, category=lead.category, size_lwh_m=[float(lead.length_m), float(lead.width_m), float(lead.height_m)],
                                                       intrinsics_256x320=K.tolist(), history_frames=HISTORY + 1, future_frames=FUTURE, fps=10, anchor_covariates=covariates,
                                                       selection='past-only (make_wm_calibration_windows.py)', ticks=rows), indent=1) + '\n')
        return dict(log=log_dir.name, name=name, covariates=covariates, tried=tried)
    return dict(log=log_dir.name, name=None, tried=tried)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--split-root', required=True, type=Path); parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--shuffle-seed', type=int, help='order the logs by a seeded permutation (default: sorted)')
    parser.add_argument('--first', type=int, default=0); parser.add_argument('--count', type=int, help='logs to scan from --first (default: all)')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Fresh output required')
    logs = sorted(p for p in args.split_root.iterdir() if p.is_dir())
    if args.shuffle_seed is not None:
        logs = [logs[i] for i in np.random.default_rng(args.shuffle_seed).permutation(len(logs))]
    logs = logs[args.first:args.first + args.count if args.count else None]
    args.output.mkdir(parents=True)
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    with Pool(args.workers) as pool:
        results = pool.map(one_log, [(str(p), str(args.output)) for p in logs], chunksize=1)
    windows = [r for r in results if r['name']]
    (args.output / 'windows.json').write_text(json.dumps(dict(split_root=str(args.split_root), shuffle_seed=args.shuffle_seed, first=args.first, logs_scanned=len(logs),
                                                              selection=dict(gap_m=GAP_M, lateral_m=LATERAL_M, min_box_px=MIN_BOX_PX, min_lidar_points=MIN_POINTS, detector=DETECT),
                                                              windows=[dict(name=r['name'], log=r['log'], **r['covariates']) for r in windows],
                                                              rejected_logs=[dict(log=r['log'], tried=r['tried']) for r in results if not r['name']]), indent=1) + '\n')
    print(json.dumps(dict(logs=len(logs), windows=len(windows))))


if __name__ == '__main__':
    main()
