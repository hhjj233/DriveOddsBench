#!/usr/bin/env python3
"""Vista inputs for bench-v1 (CPU; research server, the bench renderer's environment: OpenCV 5.0.0, the AV2 devkit).

Vista generates at the pinned generation size (pins bench.sizes): `320x576` (the default) =
576 x 320 (W x H); superseded: `canvas` = 512 x 288 and `native` = 1024 x 576. Every model input is rendered from the camera
JPEG by the frozen render_version 2 rule of make_benchmark_windows.py, canvas part: cv2.imread -> BGR2RGB -> source rows
[797, 1669) at full width (the measurement band, 1550 x 872) -> cv2.resize(..., size, INTER_AREA); at 576 x 320 a vertical
squeeze of 1.3 % (aspect 1.8 against 1.7775). Checks per window: run_vista_bench.inter_area (the adapter's resize of its
576 x 320 frames to the canvas) equals cv2.INTER_AREA bit for bit on every frame, and the resized render is compared with
the bench's canvas/real.mp4 (information: two resamplings against one). Vista's own loader
(load_img: centre crop to 16:9, PIL LANCZOS) is not used (the band is already 16:9 to 0.02 %; the crop would drop one source
row). The 10 Hz frames are the bench's ticks (window.json `ticks`: the ring_front_center frame nearest to t0 + k x 0.1 s).

Writes <out>/windows/<split>/<window>/ (lossless libx264rgb -crf 0, rgb24, 10 fps; decoded and checked; <tag> = 512 at the
canvas size, 1024 at the native size):
  history21_<tag>.mp4 21 frames k = -20..0 (t0 - 2.0 .. t0). Frame 20 is the anchor (the only frame the model sees in
                      both tracks); frames 0-19 feed the prefix (the VAE round trip of the real history, mode prefix).
  real81_<tag>.mp4    81 frames k = -20..60: the real reference's input (mode recon uses k = 0..60). DEV; for TEST only
                      with --allow-test-future and the frozen-protocol file.
  actions_t2.json     T2: Vista's trajectory action per rollout round from the AV2 ego poses after t0 (rule below), plus
                      the given ego motion for the DEV action-scale check. DEV; for TEST only with --allow-test-future.
  inputs.json         frame list, checks, file hashes, versions.
An existing video whose decoded frames equal the new render is kept byte for byte (spec hashes stay valid).

Trajectory rule (pins bench.trajectory_rule). Vista's nuScenes annotation (OpenDriveLab/Vista issue #30 script): 25
CAM_FRONT frames at 12 Hz starting at a key frame, `traj` = the LIDAR_TOP origin at the 5 key frames (2 Hz, camera frames
0, 6, 12, 18, 24) in the first key frame's LIDAR_TOP frame (x right, y forward, metres); the model gets traj[2:] (4
waypoints, 8 numbers). Vista treats the frames as 10 Hz. Here round n (n = 0, 1, 2; rollout frames 22n .. 22n + 24, frame
f = tick k = f) gets the waypoints at frames 22n + 6i (i = 1..4), i.e. the ego poses at ticks k = 22n + 6i, expressed at
the point 0.94 m ahead of the rear axle (the nuScenes LIDAR_TOP lever arm) in the ego frame of tick 22n:
x = -(lateral, left positive), y = forward - 0.94. Round 3 needs poses to k = 68 (t0 + 6.8 s); a waypoint after the
log's last pose is extrapolated at the constant speed and yaw rate of the last 1.0 s of ticks and flagged. A diagnostic
variant (`trajectory_time_aligned`: waypoints at 0.5 s steps, ticks 22n + 5i, as nuScenes' real key-frame times) is
recorded for the DEV mapping check only.

  build_vista_inputs.py --bench BENCH_V1_DIR --split dev|test --names FILE --out INPUT_ROOT [--workers 4] [--size canvas|native]
                        [--av2-root DIR (or $WCB_AV2_ROOT)] [--allow-test-future --frozen-file PATH] [--no-actions]
--no-actions skips the T2 action files (DEV; the T1-H0 main table does not need them).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
PINS_PATH = HERE / 'vista_pins.json'
PINS = json.loads(PINS_PATH.read_text())
B = PINS['bench']
SIZES = B['sizes']                              # canvas 288 x 512 (default), native 576 x 1024 (superseded)
CANVAS_WH = tuple(B['canvas_wh'])               # (512, 288)
BAND_ROWS = tuple(B['band_rows'])               # (797, 1669)
FPS = int(B['fps'])
N_ALL, N_HIST = int(B['frames']), 21            # ticks k = -20..60; history k = -20..0
T0 = 20
TR = B['trajectory_rule']
LEVER_M = float(TR['reference_point_m'])        # 0.94 m: nuScenes LIDAR_TOP x in the nuScenes ego frame
ROUNDS, ROUND_STRIDE, WAYPOINT_STRIDE, N_WAYPOINTS = int(B['n_rounds']), 22, 6, 4
TICK_NS, TOL_NS = 100_000_000, 25_000_000
AV2_ROOT_ENV = 'WCB_AV2_ROOT'                  # the AV2 Sensor root (<root>/<train|val>/<log>/), from --av2-root or local.env
SCHEMA = 'wm-calib-bench.vista.inputs.v1'
ACTION_SCHEMA = 'wm-calib-bench.vista.actions_t2.v1'


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


def read_rgb(path):
    import cv2
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


def band_frame(img_rgb, wh):
    """make_benchmark_windows.resample_v2, canvas part (render_version 2), at size wh."""
    import cv2
    return cv2.resize(img_rgb[BAND_ROWS[0]:BAND_ROWS[1]], wh, interpolation=cv2.INTER_AREA)


def write_video(frames, path: Path) -> None:
    """make_wm_calibration_windows.write_video (libx264rgb -crf 0 -preset medium, rgb24, 10 fps, timescale 10240)."""
    import numpy as np
    h, w = frames[0].shape[:2]
    tmp = path.with_name(path.stem + '.part.mp4')
    proc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}', '-r', str(FPS), '-i', '-',
                             '-c:v', 'libx264rgb', '-crf', '0', '-preset', 'medium', '-threads', '2', '-pix_fmt', 'rgb24', '-r', str(FPS),
                             '-video_track_timescale', '10240', '-map_metadata', '-1', '-movflags', '+faststart', str(tmp)], stdin=subprocess.PIPE)
    for f in frames:
        proc.stdin.write(np.ascontiguousarray(f).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f'ffmpeg failed for {path}')
    os.replace(tmp, path)


def decode(path: Path, wh, max_frames: int | None = None):
    import numpy as np
    cmd = ['ffmpeg', '-v', 'error', '-threads', '2', '-i', str(path)]
    if max_frames is not None:
        cmd += ['-frames:v', str(max_frames)]
    raw = subprocess.check_output(cmd + ['-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'])
    return np.frombuffer(raw, np.uint8).reshape(-1, wh[1], wh[0], 3)


def put(frames, path: Path, wh) -> dict:
    """Write frames losslessly unless an existing file already decodes to them; return the file record."""
    import numpy as np
    arr = np.stack(frames)
    kept = False
    if path.is_file():
        try:
            kept = np.array_equal(decode(path, wh), arr)
        except Exception:  # noqa: BLE001
            kept = False
    if not kept:
        write_video(frames, path)
        if not np.array_equal(decode(path, wh), arr):
            raise RuntimeError(f'{path}: decoded frames differ from the rendered ones')
    return dict(frames=len(frames), fps=FPS, size_wh=list(wh), sha256=sha256_file(path), bytes=path.stat().st_size,
                frames_sha256=hashlib.sha256(arr.tobytes()).hexdigest(), kept_existing=kept)


def area2(x):
    """cv2.INTER_AREA at exactly 2x down (uint8): (a + b + c + d + 2) >> 2 (checked equal to cv2 in --self-test)."""
    import numpy as np
    h, w = x.shape[0] // 2, x.shape[1] // 2
    s = x.astype(np.uint16).reshape(h, 2, w, 2, -1).sum((1, 3))
    return ((s + 2) >> 2).astype(np.uint8)


# ---------------------------------------------------------------------------------------------------------------- poses
class Poses:
    """AV2 city_SE3_egovehicle of one log (read as make_benchmark_windows.BenchLog reads it; devkit SE3)."""

    def __init__(self, log_dir: Path):
        import numpy as np
        import pyarrow.feather as pf
        from av2.geometry.geometry import quat_to_mat
        t = pf.read_table(log_dir / 'city_SE3_egovehicle.feather').to_pandas()
        t = t.sort_values('timestamp_ns')
        self.t = t.timestamp_ns.to_numpy().astype(np.int64)
        self.R = quat_to_mat(t[['qw', 'qx', 'qy', 'qz']].to_numpy())
        self.p = t[['tx_m', 'ty_m', 'tz_m']].to_numpy()
        cams = sorted(int(p.stem) for p in (log_dir / 'sensors' / 'cameras' / 'ring_front_center').glob('*.jpg'))
        self.cam_t = np.array(cams, dtype=np.int64)

    def near(self, t_ns: int):
        """Nearest pose (as make_benchmark_windows.dev_gt.pose_near): (pose time, 4x4 city_T_ego) or None past the table."""
        import numpy as np
        i = int(np.argmin(np.abs(self.t - t_ns)))
        if abs(int(self.t[i]) - t_ns) > TOL_NS:
            return None
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = self.R[i], self.p[i]
        return int(self.t[i]), T

    def camera_frame(self, tick_ns: int):
        import numpy as np
        i = int(np.argmin(np.abs(self.cam_t - tick_ns)))
        return int(self.cam_t[i]) if abs(int(self.cam_t[i]) - tick_ns) <= TOL_NS else None


def yaw_of(R) -> float:
    return math.atan2(R[1, 0], R[0, 0])


def ego_track(window: dict, poses: Poses, k_max: int) -> dict:
    """Ego poses (4x4, t0 ego frame) at the camera frame of every tick k = 0..k_max; extrapolated past the log's end."""
    import numpy as np
    ticks = {t['k']: t for t in window['ticks']}
    anchor_tick = int(ticks[0]['tick_timestamp_ns'])
    rec = {}
    for k in range(0, k_max + 1):
        if k in ticks:
            f_ns = int(ticks[k]['frame_timestamp_ns'])
        else:
            f_ns = poses.camera_frame(anchor_tick + k * TICK_NS)
        got = poses.near(f_ns) if f_ns is not None else None
        rec[k] = dict(frame_ns=f_ns, pose_ns=None if got is None else got[0], T=None if got is None else got[1], extrapolated=got is None)
    T0w = rec[0]['T']
    if T0w is None:
        raise ValueError(f'{window["name"]}: no pose at t0')
    inv0 = np.linalg.inv(T0w)
    last = max(k for k in rec if not rec[k]['extrapolated'])
    if any(not rec[k]['extrapolated'] for k in range(last + 1, k_max + 1)) or any(rec[k]['extrapolated'] for k in range(0, last + 1)):
        raise ValueError(f'{window["name"]}: a gap inside the pose track')
    rel = {k: inv0 @ rec[k]['T'] for k in range(0, last + 1)}
    if last < k_max:  # constant speed and yaw rate of the last 1.0 s (10 ticks), planar, from the last real pose
        a, b = rel[last - 10], rel[last]
        d = np.linalg.inv(a) @ b
        dist, dpsi = float(np.hypot(d[0, 3], d[1, 3])), yaw_of(d[:3, :3])
        v, w = dist / 1.0, dpsi / 1.0
        for k in range(last + 1, k_max + 1):
            s = (k - last) * 0.1
            psi = w * s
            if abs(w) > 1e-6:
                dx, dy = v / w * math.sin(psi), v / w * (1 - math.cos(psi))
            else:
                dx, dy = v * s, 0.0
            step = np.eye(4)
            step[:2, :2] = [[math.cos(psi), -math.sin(psi)], [math.sin(psi), math.cos(psi)]]
            step[0, 3], step[1, 3] = dx, dy
            rel[k] = b @ step
    return dict(rel=rel, meta={k: dict(frame_ns=r['frame_ns'], pose_ns=r['pose_ns'], extrapolated=r['extrapolated']) for k, r in rec.items()},
                last_real_k=last)


def trajectory_for(rel: dict, first_k: int, ks: list) -> list:
    """Vista's traj[2:] for one round: the lever-arm point at ticks ks in the ego frame of tick first_k, (x right, y forward)."""
    import numpy as np
    inv = np.linalg.inv(rel[first_k])
    out = []
    for k in ks:
        q = inv @ rel[k] @ np.array([LEVER_M, 0.0, 0.0, 1.0])
        out += [float(-q[1]), float(q[0] - LEVER_M)]
    return out


def actions_for(window: dict, poses: Poses) -> dict:
    k_need = ROUND_STRIDE * (ROUNDS - 1) + WAYPOINT_STRIDE * N_WAYPOINTS      # 68
    trk = ego_track(window, poses, k_need)
    rel = trk['rel']
    rounds = []
    for n in range(ROUNDS):
        first = ROUND_STRIDE * n
        ks = [first + WAYPOINT_STRIDE * i for i in range(1, N_WAYPOINTS + 1)]
        ks_time = [first + 5 * i for i in range(1, N_WAYPOINTS + 1)]
        rounds.append(dict(round=n, first_k=first, waypoint_ks=ks, trajectory=[round(v, 6) for v in trajectory_for(rel, first, ks)],
                           extrapolated=[trk['meta'][k]['extrapolated'] for k in ks],
                           diagnostic_time_aligned=dict(waypoint_ks=ks_time, trajectory=[round(v, 6) for v in trajectory_for(rel, first, ks_time)])))
    given = []
    for k in range(0, 61):  # the given ego motion on the bench horizon (rear axle, t0 ego frame), for the action-scale check
        T = rel[k]
        given.append(dict(k=k, x_m=round(float(T[0, 3]), 4), y_m=round(float(T[1, 3]), 4), yaw_deg=round(math.degrees(yaw_of(T[:3, :3])), 3)))
    return dict(schema=ACTION_SCHEMA, window=window['name'], split=window['split'], mode='trajectory', variant='frame_index', rule=TR, rounds=rounds,
                last_real_pose_k=trk['last_real_k'], extrapolated_waypoints=sum(sum(r['extrapolated']) for r in rounds),
                pose_ticks={str(k): v for k, v in trk['meta'].items()}, given_ego_t0_frame=given)


# ---------------------------------------------------------------------------------------------------------------- build
def build_one(job: dict) -> dict:
    import cv2
    import numpy as np
    cv2.setNumThreads(2)
    t_start = time.time()
    wdir, out, history_only = Path(job['wdir']), Path(job['out']), job['history_only']
    size = job['size']
    h, w = SIZES[size]['hw']
    wh, tag = (w, h), SIZES[size]['input_tag']
    native = wh != CANVAS_WH
    mapping = 'identity' if not native else 'area2' if wh == (2 * CANVAS_WH[0], 2 * CANVAS_WH[1]) else 'inter_area'
    sys.path.insert(0, str(HERE))
    sys.dont_write_bytecode = True
    import run_vista_bench as rvb
    window = json.loads((wdir / 'window.json').read_text())
    if window.get('render_version') != 2:
        raise ValueError(f'{wdir}: render_version {window.get("render_version")}, expected 2')
    ticks = sorted(window['ticks'], key=lambda t: t['k'])
    if [t['k'] for t in ticks] != list(range(-20, 61)):
        raise ValueError(f'{wdir}: ticks are not k = -20..60')
    n = N_HIST if history_only else N_ALL
    renders = [band_frame(read_rgb(t['path']), wh) for t in ticks[:n]]
    bench = decode(wdir / 'canvas' / 'real.mp4', CANVAS_WH, max_frames=N_HIST if history_only else None)
    ia_ok = None
    if mapping == 'area2':
        down = np.stack([area2(x) for x in renders])
    elif mapping == 'inter_area':
        down = rvb.inter_area(np.stack(renders), *CANVAS_WH)
        ia_ok = bool(np.array_equal(down, np.stack([cv2.resize(x, CANVAS_WH, interpolation=cv2.INTER_AREA) for x in renders])))
    else:
        down = np.stack(renders)
    d = np.abs(down.astype(np.int16) - bench.astype(np.int16))
    vs = dict(reference='canvas/real.mp4' + (' frames 0-20' if history_only else ''), frames=int(len(down)), max_abs=int(d.max()),
              mean_abs=round(float(d.mean()), 4), p99_abs=float(np.percentile(d, 99)), identical=bool(not d.any()),
              reference_sha256=sha256_file(wdir / 'canvas' / 'real.mp4'))
    checks = dict(size=size, mapping=mapping, area2_matches_cv2=bool(np.array_equal(area2(renders[T0]), cv2.resize(renders[T0], CANVAS_WH, interpolation=cv2.INTER_AREA)))
                  if mapping == 'area2' else None, inter_area_matches_cv2=ia_ok, model_vs_canvas=vs)
    if native:
        checks['model_down_vs_canvas'] = vs                          # the key of the first (native) inputs
    out.mkdir(parents=True, exist_ok=True)
    files = {f'history21_{tag}.mp4': put(renders[:N_HIST], out / f'history21_{tag}.mp4', wh)}
    actions = None
    if not history_only:
        files[f'real81_{tag}.mp4'] = put(renders, out / f'real81_{tag}.mp4', wh)
    if not history_only and not job['no_actions']:
        log_dir = Path(job['av2_root']) / window['log_source'] / window['log']
        actions = actions_for(window, Poses(log_dir))
        (out / 'actions_t2.json').write_text(json.dumps(actions, indent=1) + '\n')
        files['actions_t2.json'] = dict(sha256=sha256_file(out / 'actions_t2.json'))
        if window['split'] == 'DEV':  # the diagnostic time-aligned variant (DEV mapping check only; never analysed)
            ta = dict(actions, variant='time_aligned', diagnostic_only=True,
                      rounds=[dict(round=r['round'], first_k=r['first_k'], waypoint_ks=r['diagnostic_time_aligned']['waypoint_ks'],
                                   trajectory=r['diagnostic_time_aligned']['trajectory'],
                                   extrapolated=[actions['pose_ticks'][str(k)]['extrapolated'] for k in r['diagnostic_time_aligned']['waypoint_ks']])
                              for r in actions['rounds']])
            (out / 'actions_t2_timealigned.json').write_text(json.dumps(ta, indent=1) + '\n')
            files['actions_t2_timealigned.json'] = dict(sha256=sha256_file(out / 'actions_t2_timealigned.json'))
    rec = dict(schema=SCHEMA, window=window['name'], split=window['split'], history_only=history_only, size=size, size_wh=list(wh), files=files,
               timeline=dict(fps=FPS, frames=n, t0_index=T0, rule='frame i = window.json tick k = i - 20 (the camera frame nearest to t0 + k x 0.1 s)'),
               frames=[dict(i=i, k=t['k'], frame_timestamp_ns=t['frame_timestamp_ns'], offset_ms=t['offset_ms'], path=t['path']) for i, t in enumerate(ticks[:n])],
               checks=checks, window_json_sha256=sha256_file(wdir / 'window.json'),
               actions_summary=None if actions is None else dict(last_real_pose_k=actions['last_real_pose_k'], extrapolated_waypoints=actions['extrapolated_waypoints']),
               software=dict(python=platform.python_version(), opencv=cv2.__version__, numpy=np.__version__,
                             builder_sha256=sha256_file(Path(__file__)), pins_sha256=sha256_file(PINS_PATH)),
               seconds=round(time.time() - t_start, 1), created_utc=datetime.now(timezone.utc).isoformat(timespec='seconds'))
    (out / 'inputs.json').write_text(json.dumps(rec, indent=1) + '\n')
    return dict(window=window['name'], ok=True, seconds=rec['seconds'], files=sorted(files), down_vs_canvas=checks['model_vs_canvas'],
                area2_ok=checks['area2_matches_cv2'] is not False and checks['inter_area_matches_cv2'] is not False, actions=rec['actions_summary'])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bench', type=Path, required=True)
    ap.add_argument('--split', choices=('dev', 'test'), required=True)
    ap.add_argument('--names', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--av2-root', type=Path, default=Path(os.environ[AV2_ROOT_ENV]) if os.environ.get(AV2_ROOT_ENV) else None)
    ap.add_argument('--allow-test-future', action='store_true')
    ap.add_argument('--frozen-file', type=Path, default=None)
    ap.add_argument('--size', choices=tuple(SIZES), default=B['default_size'])
    ap.add_argument('--no-actions', action='store_true', help='DEV without the T2 action files (no AV2 poses needed)')
    args = ap.parse_args()
    names = [l.strip() for l in args.names.read_text().splitlines() if l.strip()]
    pool = set(l.strip() for l in (args.bench / f'names-{args.split}.txt').read_text().splitlines() if l.strip())
    if not set(names) <= pool:
        raise SystemExit(f'names outside names-{args.split}.txt: {sorted(set(names) - pool)[:5]}')
    history_only = args.split == 'test' and not args.allow_test_future
    if not history_only and not args.no_actions and args.av2_root is None:
        raise SystemExit(f'the T2 actions need the AV2 poses: pass --av2-root or set {AV2_ROOT_ENV} (models/vista/local.env)')
    if args.split == 'test' and args.allow_test_future:
        if args.frozen_file is None or not args.frozen_file.is_file() or 'vista' not in args.frozen_file.name:
            raise SystemExit('TEST future frames and poses need the frozen Vista protocol file (--frozen-file protocol-vista-frozen.txt)')
    jobs = [dict(wdir=str(args.bench / f'windows-{args.split}' / w), out=str(args.out / 'windows' / args.split / w), history_only=history_only,
                 av2_root=str(args.av2_root), size=args.size, no_actions=args.no_actions) for w in names]
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(build_one, jobs):
            results.append(r)
            print(json.dumps(r), flush=True)
    summary = dict(split=args.split, size=args.size, windows=len(results), history_only=history_only, seconds=round(time.time() - t0, 1),
                   identical_to_canvas=sum(bool(r['down_vs_canvas'].get('identical')) for r in results),
                   down_vs_canvas_max=max(r['down_vs_canvas']['max_abs'] for r in results),
                   down_vs_canvas_mean=round(sum(r['down_vs_canvas']['mean_abs'] for r in results) / len(results), 4),
                   resize_checks_all_ok=all(r['area2_ok'] for r in results),
                   windows_with_extrapolated_waypoints=None if history_only or args.no_actions else [r['window'] for r in results if r['actions']['extrapolated_waypoints']])
    print(json.dumps(summary))
    (args.out / f'build-{args.split}-summary.json').write_text(json.dumps(summary, indent=1) + '\n')


if __name__ == '__main__':
    main()
