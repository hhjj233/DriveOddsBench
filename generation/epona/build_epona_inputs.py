#!/usr/bin/env python3
"""Epona inputs for bench-v1 windows: the measurement band at Epona's native 1024 x 512 on the 5 Hz grid, plus past ego poses.

Run it with the research server's CPU environment (the one that rendered the bench windows, so the JPEG decoder and OpenCV
build are the same; it needs cv2, numpy, av2 and ffmpeg):

  build_epona_inputs.py --bench-root BENCH --split dev|test --names FILE --out INPUT_ROOT [--workers 2] [--band-rows A B]
                        [--history-only] [--allow-test-real]

Per window, under <INPUT_ROOT>/windows/<split>/<window>/:
  history.mp4      11 frames, 1024x512, 5 fps, lossless: ticks k = -20, -18, ..., 0 (t0 - 2.0 .. t0)
  real.mp4         41 frames, 1024x512, 5 fps, lossless: ticks k = -20, -18, ..., 60 (real reference only; not written
                   with --history-only)
  past_poses.json  ego poses at the frame times of ticks k = -20, -18, ..., 0, in the ego frame at t0 (AV2 egovehicle:
                   x forward, y left, z up), from the log's city_SE3_egovehicle via the AV2 devkit
  build.json       checks and hashes

Rendering = the bench's render_version 2 with another output size: read_rgb (cv2.imread + BGR2RGB) of the tick's camera frame
(window.json "ticks"; the nearest 20 Hz frame), crop the model band at full width (default the bench-v1 band [579, 1451);
--band-rows A B for another 872-row band), cv2.resize to 1024 x 512 with INTER_AREA. The same code at
512 x 288 on the measurement band [797, 1669) must reproduce the window's canvas/real.mp4 frame for frame (checked; with
--history-only and on TEST only for the history frames k <= 0, so no future frame of a TEST window is decoded).

Past only: the poses are taken for ticks k <= 0 only (nearest pose to each frame time, as gt.json's ego entries), relative
to the pose at the anchor (t0); a pose more than 50 ms after the anchor is refused. No annotation is read. On DEV the poses
are also checked against gt.json's ego entries (k <= 0). TEST windows need --history-only (history frames k <= 0 and past
poses only: no future frame is rendered, decoded or written, no annotation is read); a TEST build with the real future
(real.mp4, for the local TEST real reference) needs --allow-test-real and is allowed only after the Epona protocol is frozen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PINS = json.loads((HERE / 'epona_pins.json').read_text())
BAND = tuple(PINS['geometry']['band_rows'])                # the measurement band (canvas), fixed
MODEL_BAND = list(PINS['geometry']['model_band_rows_default'])  # the rows squashed into Epona's 1024x512 frame (--band-rows)
MW, MH = PINS['geometry']['model_size_wh']
CW, CH = PINS['geometry']['canvas_size_wh']
HIST_K = list(range(-20, 1, 2))      # 11 history ticks on the 5 Hz grid
ALL_K = list(range(-20, 61, 2))      # 41 ticks
FPS = 5
POSE_TOL_NS = 50_000_000
SCHEMA = 'wm-calib-bench.epona.inputs.v1'


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


def read_rgb(path):
    import cv2
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


def check_band(rows) -> list:
    a, b = int(rows[0]), int(rows[1])
    if b - a != BAND[1] - BAND[0] or a > BAND[0] or ((BAND[0] - a) * MH) % (b - a):
        raise SystemExit(f'model band {rows}: must be 872 rows, start at or above row {BAND[0]}, and map row {BAND[0]} to an integer model row')
    return [a, b]


def render(img_rgb, model_band):
    """(model 1024x512, canvas 512x288) of one source frame: row crop, then cv2.resize INTER_AREA (model: the model band;
    canvas: the measurement band, for the reproduction check)."""
    import cv2
    return (cv2.resize(img_rgb[model_band[0]:model_band[1]], (MW, MH), interpolation=cv2.INTER_AREA),
            cv2.resize(img_rgb[BAND[0]:BAND[1]], (CW, CH), interpolation=cv2.INTER_AREA))


def write_video(frames, path: Path, fps: int = FPS):
    """make_wm_calibration_windows.write_video's encoder (libx264rgb -crf 0 -preset medium, rgb24) at 5 fps."""
    h, w = frames[0].shape[:2]
    tmp = path.with_name(path.stem + '.part.mp4')
    proc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}', '-r', str(fps), '-i', '-',
                             '-c:v', 'libx264rgb', '-crf', '0', '-preset', 'medium', '-threads', '2', '-pix_fmt', 'rgb24', '-r', str(fps),
                             '-video_track_timescale', '10240', '-map_metadata', '-1', '-movflags', '+faststart', str(tmp)], stdin=subprocess.PIPE)
    for f in frames:
        proc.stdin.write(np.ascontiguousarray(f).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f'ffmpeg failed for {path}')
    os.replace(tmp, path)


def decode(path: Path, w: int, h: int, frames: int | None = None):
    cmd = ['ffmpeg', '-v', 'error', '-threads', '2', '-i', str(path)]
    if frames is not None:
        cmd += ['-frames:v', str(frames)]
    raw = subprocess.check_output(cmd + ['-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'])
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)


def past_poses(log_dir: Path, anchor_ns: int, tick_frames: list[tuple[int, int]]) -> dict:
    """Ego poses at the frame times of ticks k <= 0, relative to the ego pose at the anchor (AV2 devkit, nearest pose)."""
    from av2.utils.io import read_city_SE3_ego
    poses = read_city_SE3_ego(log_dir)
    ts = np.array(sorted(poses))
    if anchor_ns not in poses:
        raise RuntimeError(f'no ego pose at the anchor {anchor_ns}')
    t0_T_city = poses[anchor_ns].inverse()
    rows = []
    for k, f_ns in tick_frames:
        if k > 0:
            raise AssertionError('future tick requested')
        i = int(np.argmin(np.abs(ts - f_ns)))
        pt = int(ts[i])
        if pt > anchor_ns + POSE_TOL_NS:
            raise RuntimeError(f'pose for k={k} at {pt} is after the anchor')
        rel = t0_T_city.compose(poses[pt])
        m = np.eye(4)
        m[:3, :3], m[:3, 3] = rel.rotation, rel.translation
        yaw = float(np.arctan2(rel.rotation[1, 0], rel.rotation[0, 0]))
        rows.append(dict(k=k, frame_timestamp_ns=int(f_ns), pose_timestamp_ns=pt, matrix_4x4=m.tolist(), xyz_m=rel.translation.tolist(), yaw_rad=yaw))
    del poses
    return dict(frame='ego frame at the anchor t0 (AV2 egovehicle: x forward, y left, z up)', source='city_SE3_egovehicle.feather via av2.utils.io.read_city_SE3_ego; nearest pose to each frame time',
                ticks=[r['k'] for r in rows], poses=rows)


def epona_rel(matrices: np.ndarray):
    """numpy port of Epona utils/preprocess.get_rel_pose (for the record; the adapter uses the pinned torch function):
    entry 0 = 0, entry i = inv(R_{i-1}) (t_i - t_{i-1}) [:2] (m) and the yaw of inv(R_{i-1}) R_i (degrees, left positive)."""
    p = np.concatenate([matrices[:1], matrices], axis=0)
    ra, rb, ta, tb = p[:-1, :3, :3], p[1:, :3, :3], p[:-1, :3, 3:], p[1:, :3, 3:]
    rr = np.linalg.inv(ra) @ rb
    tr = np.linalg.inv(ra) @ (tb - ta)
    return tr[:, :2, 0], np.degrees(np.arctan2(rr[:, 1, 0], rr[:, 0, 0]))[:, None]


def build_window(job: dict) -> dict:
    t_start = time.time()
    wdir, out, split, hist_only = Path(job['wdir']), Path(job['out']), job['split'], job['history_only']
    win = json.loads((wdir / 'window.json').read_text())
    name = win['name']
    ticks = {t['k']: t for t in win['ticks']}
    render_k = HIST_K if hist_only else ALL_K
    missing = [k for k in render_k if k not in ticks]
    if missing:
        raise RuntimeError(f'{name}: ticks {missing} missing')
    if hist_only and max(render_k) > 0:
        raise AssertionError('future tick in a history-only build')
    out.mkdir(parents=True, exist_ok=True)
    cache, model, canvas = {}, [], []
    for k in render_k:
        p = ticks[k]['path']
        if p not in cache:
            cache[p] = render(read_rgb(p), job['model_band'])
        model.append(cache[p][0]); canvas.append(cache[p][1])
    # the 512x288 render must equal the window's canvas/real.mp4 (history-only and TEST: history frames only)
    check_k = HIST_K if (hist_only or split == 'test') else ALL_K
    n_dec = 21 if (hist_only or split == 'test') else 81
    ref = decode(wdir / 'canvas/real.mp4', CW, CH, frames=n_dec)
    canvas_ok = all(np.array_equal(canvas[render_k.index(k)], ref[k + 20]) for k in check_k)
    if not canvas_ok:
        bad = [k for k in check_k if not np.array_equal(canvas[render_k.index(k)], ref[k + 20])]
        raise RuntimeError(f'{name}: the 512x288 render differs from canvas/real.mp4 at ticks {bad[:5]}')
    del ref
    clips = {'history.mp4': [model[render_k.index(k)] for k in HIST_K]}
    if not hist_only:
        clips['real.mp4'] = model
    checks = {}
    for rel, frames in clips.items():
        write_video(frames, out / rel)
        back = decode(out / rel, MW, MH)
        same = bool(len(back) == len(frames) and np.array_equal(back, np.stack(frames)))
        checks[rel] = dict(frames=len(frames), decoded=int(len(back)), identical=same, bytes=(out / rel).stat().st_size, sha256=sha256_file(out / rel))
        if not same:
            raise RuntimeError(f'{name}/{rel}: decoded frames differ from the rendered ones')
    # past poses (k <= 0 only)
    log_dir = Path(ticks[0]['path']).parents[3]
    pp = past_poses(log_dir, int(win['anchor']['timestamp_ns']), [(k, int(ticks[k]['frame_timestamp_ns'])) for k in HIST_K])
    mats = np.array([r['matrix_4x4'] for r in pp['poses']])
    cond = [HIST_K.index(k) for k in PINS['timing']['pose_ticks']]
    rel_xy, rel_yaw = epona_rel(mats[cond])
    pp.update(schema='wm-calib-bench.epona.past_poses.v1', window=name, split=split,
              epona_condition_ticks=PINS['timing']['pose_ticks'],
              epona_rel_pose_numpy=dict(rule='utils/preprocess.get_rel_pose over the 10 condition poses (entry 0 = 0); float64 numpy port, for the record',
                                        dx_dy_m=rel_xy.round(6).tolist(), dyaw_deg=rel_yaw.round(6).tolist()),
              speed_mps_from_poses=float(np.linalg.norm(mats[-1, :2, 3] - mats[-2, :2, 3]) / 0.2))
    gt_check = None
    if split == 'dev' and (wdir / 'gt.json').is_file():   # DEV only: compare with gt.json's ego entries (k <= 0)
        gt = json.loads((wdir / 'gt.json').read_text())
        g = {e['k']: e for e in gt['ego'] if e['k'] <= 0}
        del gt
        diffs = [float(np.abs(np.array(g[r['k']]['matrix_3x4']) - np.array(r['matrix_4x4'])[:3]).max()) for r in pp['poses']]
        same_t = all(g[r['k']]['pose_timestamp_ns'] == r['pose_timestamp_ns'] for r in pp['poses'])
        gt_check = dict(max_abs_matrix_diff=max(diffs), pose_timestamps_equal=same_t)
        if max(diffs) > 2e-6 or not same_t:
            raise RuntimeError(f'{name}: past poses differ from gt.json ({gt_check})')
    (out / 'past_poses.json').write_text(json.dumps(pp, indent=1) + '\n')
    rec = dict(schema=SCHEMA, window=name, split=split, created=datetime.now().astimezone().isoformat(timespec='seconds'),
               source_window=str(wdir), log=win['log'], anchor_timestamp_ns=int(win['anchor']['timestamp_ns']),
               history_only=hist_only,
               ticks=dict(history=HIST_K, real=None if hist_only else ALL_K, frame_paths={str(k): ticks[k]['path'] for k in render_k},
                          frame_timestamps_ns={str(k): int(ticks[k]['frame_timestamp_ns']) for k in render_k}),
               render=dict(rule=PINS['geometry']['model_render'], size_wh=[MW, MH], model_band_rows=job['model_band'],
                           canvas_reproduced=dict(ticks=check_k, identical=canvas_ok)),
               clips=checks, past_poses=dict(sha256=sha256_file(out / 'past_poses.json'), ticks=pp['ticks'], gt_json_check=gt_check),
               builder_sha256=sha256_file(Path(__file__)), pins_sha256=sha256_file(HERE / 'epona_pins.json'),
               cv2=__import__('cv2').__version__, numpy=np.__version__, python=sys.version.split()[0], seconds=round(time.time() - t_start, 1))
    (out / 'build.json').write_text(json.dumps(rec, indent=1) + '\n')
    return dict(window=name, ok=True, seconds=rec['seconds'], gt_check=gt_check)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bench-root', type=Path, required=True, help='bench-v1 directory (holds windows-dev/, windows-test/)')
    ap.add_argument('--split', required=True, choices=['dev', 'test'])
    ap.add_argument('--names', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True, help='input root; windows go to <out>/windows/<split>/<window>/')
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--history-only', action='store_true', help='render and write the history (k <= 0) and the past poses only; '
                    'required for --split test unless --allow-test-real')
    ap.add_argument('--allow-test-real', action='store_true', help='TEST with the real future (real.mp4): only after the Epona protocol is frozen')
    ap.add_argument('--force', action='store_true', help='rebuild windows that already have build.json')
    ap.add_argument('--band-rows', type=int, nargs=2, default=MODEL_BAND, help='source rows [a, b) squashed into the model frame '
                    '(default: the bench-v1 band of epona_pins.json; variants in geometry.model_band_variants)')
    args = ap.parse_args()
    if args.split == 'test' and not (args.history_only or args.allow_test_real):
        raise SystemExit('TEST inputs: --history-only (history and past poses only); the real future needs --allow-test-real, '
                         'only after the Epona protocol is frozen')
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    names = args.names.read_text().split()
    band = check_band(args.band_rows)
    jobs = []
    for w in names:
        out = args.out / 'windows' / args.split / w
        if (out / 'build.json').is_file() and not args.force:
            continue
        jobs.append(dict(wdir=str(args.bench_root / f'windows-{args.split}' / w), out=str(out), split=args.split, model_band=band,
                         history_only=bool(args.history_only)))
    print(json.dumps(dict(split=args.split, windows=len(names), to_build=len(jobs))), flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
        for r in ex.map(build_window, jobs):
            results.append(r)
            print(json.dumps(r), flush=True)
    man = dict(schema='wm-calib-bench.epona.inputs_manifest.v1', split=args.split, created=datetime.now().astimezone().isoformat(timespec='seconds'),
               names_file=str(args.names), names_sha256=sha256_file(args.names), windows=names, built_now=len(results), model_band_rows=band,
               history_only=bool(args.history_only),
               builder_sha256=sha256_file(Path(__file__)), pins_sha256=sha256_file(HERE / 'epona_pins.json'))
    (args.out / f'inputs-manifest-{args.split}.json').write_text(json.dumps(man, indent=1) + '\n')


if __name__ == '__main__':
    main()
