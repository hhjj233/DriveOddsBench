#!/usr/bin/env python3
"""Raw AV2 -> DrivingWorld inputs: 10 Hz model frames of the registered band and past-only ego poses.

  build_drivingworld_inputs.py --bench BENCH --split dev|test --names FILE --out INPUT_ROOT [--band centred|top]
                               [--allow-test-future --frozen-file STAGE2_RECORD]

Run with the research server's CPU environment (the one that rendered the bench windows: cv2, numpy, scipy, pyarrow, av2,
ffmpeg). Per window, under <INPUT_ROOT>/windows/<split>/<window>/:
  history10.mp4    21 model frames (512x256, 10 fps, lossless): ticks k = -20..0; the model reads the last 15 (k = -14..0)
  real10.mp4       81 model frames, ticks -20..60 (DEV only, for the real reference; TEST only after the stage-2 freeze)
  past_poses.json  AV2 ego poses at the exact camera timestamps of ticks -20..0 and the released relative-pose encoding
                   of the 15 conditioning frames (no pose after the anchor is read)
  inputs.json      geometry, frames, checks and hashes

Every frame is the window's own tick frame (window.json "ticks", as canvas/real.mp4). The 512x288 canvas render of the
same frame must equal canvas/real.mp4 (checked; history frames only on TEST, so no TEST future frame is decoded) and the
model frame is cut from a render on the same pixel grid, so model rows map onto canvas rows by an integer offset.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import drivingworld_common as dc  # noqa: E402


def past_poses(window, rows):
    """Exact-timestamp AV2 ego poses for the history frames; rows after the anchor are never materialised."""
    import numpy as np
    import pyarrow.dataset as ds
    from av2.geometry.geometry import quat_to_mat
    anchor = int(window['anchor']['timestamp_ns'])
    hist = [r for r in rows if r['k'] <= 0]
    log = Path(hist[-1]['path']).parents[3]
    posefile = log / 'city_SE3_egovehicle.feather'
    first = hist[0]['frame_timestamp_ns']
    table = ds.dataset(str(posefile), format='feather').to_table(
        filter=(ds.field('timestamp_ns') >= first - 1_000_000_000) & (ds.field('timestamp_ns') <= anchor))
    data = table.to_pandas().sort_values('timestamp_ns')
    stamps = {int(t): i for i, t in enumerate(data.timestamp_ns)}
    if int(data.timestamp_ns.max()) > anchor:
        raise ValueError('future pose row loaded')
    missing = [r['k'] for r in hist if r['frame_timestamp_ns'] not in stamps]
    if missing or anchor not in stamps:
        raise ValueError(f'no exact ego pose at the camera timestamps of ticks {missing} or at the anchor')
    idx = [stamps[r['frame_timestamp_ns']] for r in hist] + [stamps[anchor]]
    rot = quat_to_mat(data[['qw', 'qx', 'qy', 'qz']].to_numpy()[idx])
    trans = data[['tx_m', 'ty_m', 'tz_m']].to_numpy()[idx]
    mats = np.tile(np.eye(4), (len(idx), 1, 1))
    mats[:, :3, :3] = rot
    mats[:, :3, 3] = trans
    rel_anchor = np.linalg.inv(mats[-1])[None] @ mats[:-1]   # gt.json / Epona convention: ego frame at the anchor
    mats = mats[:-1]
    cond = [i for i, r in enumerate(hist) if r['k'] in dc.CONDITION_K]
    xy, yaw = dc.rel_poses(mats[cond])
    ix, iy, iyaw = dc.pose_tokens_np(xy, yaw[:, 0])
    dt = (hist[-1]['frame_timestamp_ns'] - hist[-2]['frame_timestamp_ns']) / 1e9
    return dict(schema='wm-calibration.drivingworld.past_poses.v1', window=window['name'],
                frame='AV2 egovehicle (x forward, y left, z up); nuPlan ego poses used in training share the convention',
                source=str(posefile), source_sha256=dc.sha(posefile), anchor_ns=anchor,
                rows_loaded=len(data), latest_row_ns=int(data.timestamp_ns.max()), future_rows_loaded=0,
                history=[dict(k=r['k'], frame_timestamp_ns=r['frame_timestamp_ns'], pose_timestamp_ns=r['frame_timestamp_ns'],
                              matrix_anchor=np.round(rel_anchor[i], 9).tolist()) for i, r in enumerate(hist)],
                condition=dict(ticks=dc.CONDITION_K,
                               rule='released datasets_utils.get_meta_data on the 15 conditioning poses: entry 0 = 0, '
                                    'entry i = inv(P_{i-1}) P_i: (x, y) m and zyx-Euler z angle deg',
                               xy_m=xy.tolist(), yaw_deg=yaw.tolist(),
                               tokens=dict(x=ix.tolist(), y=iy.tolist(), yaw=iyaw.tolist()),
                               clipped=dict(x=int(((xy[:, 0] < 0) | (xy[:, 0] >= 8)).sum()),
                                            y=int((abs(xy[:, 1]) >= 0.5).sum()),
                                            yaw=int((abs(yaw[:, 0]) >= 8).sum()))),
                speed_last_step_mps=float(np.hypot(*xy[-1]) / dt), navigation_command_read=False)


def gt_check(wdir, poses):
    """DEV only: the past poses against gt.json's ego entries (k <= 0)."""
    import numpy as np
    gt = json.loads((wdir / 'gt.json').read_text())
    ego = {e['k']: e for e in gt['ego'] if e['k'] <= 0}
    diffs = [float(np.abs(np.array(ego[r['k']]['matrix_3x4']) - np.array(r['matrix_anchor'])[:3]).max()) for r in poses['history']]
    same = all(ego[r['k']]['pose_timestamp_ns'] == r['pose_timestamp_ns'] for r in poses['history'])
    if max(diffs) > 2e-6 or not same:
        raise ValueError(f'past poses differ from gt.json: {max(diffs)} {same}')
    return dict(max_abs_matrix_diff=max(diffs), pose_timestamps_equal=same)


def build_window(bench, split, window_name, out, band_name, with_future=False, frozen_file=None, diagnostic=False):
    if split == 'test' and with_future:
        dc.verify_freeze(frozen_file, required=[dc.PINS_PATH, Path(__file__)])
    dc.guard_names(bench, split, [window_name], diagnostic)
    wdir = Path(bench) / ('windows-' + split) / window_name
    window = json.loads((wdir / 'window.json').read_text())
    if window['name'] != window_name or window['split'].lower() != split:
        raise ValueError('window split mismatch')
    rows = dc.tick_rows(window, with_future)        # metadata only; history rows never later than the t0 camera frame
    geometry = dc.band(band_name)
    import cv2
    import numpy as np
    cv2.setNumThreads(2)
    dest = Path(out) / 'windows' / split / window_name
    if dest.exists():
        raise FileExistsError('fresh window destination required; existing outputs retained')
    model, canvas = [], []
    for row in rows:
        rgb = cv2.cvtColor(cv2.imread(row['path']), cv2.COLOR_BGR2RGB)
        m, c, tall = dc.render(rgb, geometry)
        off = geometry['canvas_offset']
        if not np.array_equal(tall[off:off + dc.CANVAS_WH[1]], c):
            raise ValueError('tall render and canvas render disagree on the shared rows')
        model.append(m)
        canvas.append(c)
    ref = dc.decode(wdir / 'canvas' / 'real.mp4', dc.CANVAS_WH, frames=len(rows))
    if len(ref) != len(rows) or not np.array_equal(ref, np.stack(canvas)):
        raise ValueError('the canvas render differs from canvas/real.mp4')
    del ref
    dest.mkdir(parents=True, exist_ok=False)
    files = {'history10.mp4': dc.write_checked(model[:len(dc.HISTORY_K)], dest / 'history10.mp4', 10)}
    if with_future:
        files['real10.mp4'] = dc.write_checked(model, dest / 'real10.mp4', 10)
    poses = past_poses(window, rows)
    check = gt_check(wdir, poses) if split == 'dev' else None
    dc.new_json(dest / 'past_poses.json', poses)
    files['past_poses.json'] = dict(sha256=dc.sha(dest / 'past_poses.json'), bytes=(dest / 'past_poses.json').stat().st_size)
    record = dict(schema='wm-calib-bench.drivingworld.inputs.v1', model='drivingworld', split=split, window=window_name,
                  diagnostic=bool(diagnostic),
                  source_window_sha256=dc.sha(wdir / 'window.json'), band=geometry,
                  rows=rows, history_ticks=dc.HISTORY_K, condition_ticks=dc.CONDITION_K, future_present=with_future,
                  checks=dict(canvas_reproduced_ticks=[r['k'] for r in rows], tall_equals_canvas=True, gt_pose_check=check),
                  speed_last_step_mps=poses['speed_last_step_mps'], files=files, timeline=dc.timeline(),
                  builder_sha256=dc.sha(Path(__file__)), common_sha256=dc.sha(dc.HERE / 'drivingworld_common.py'),
                  pins_sha256=dc.sha(dc.PINS_PATH), cv2=cv2.__version__, numpy=np.__version__, python=sys.version.split()[0])
    dc.new_json(dest / 'inputs.json', record)
    return dict(window=window_name, frames=len(rows), band=band_name, speed_last_step_mps=round(poses['speed_last_step_mps'], 3),
                gt_pose_check=check, clipped=poses['condition']['clipped'])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bench', type=Path, required=True)
    ap.add_argument('--split', choices=['dev', 'test'], required=True)
    ap.add_argument('--names', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--band', default=dc.B['model_band'], choices=sorted(dc.B['bands']))
    ap.add_argument('--allow-test-future', action='store_true')
    ap.add_argument('--frozen-file', type=Path)
    ap.add_argument('--diagnostic', action='store_true', help='DEV only: admit any window of names-dev.txt (scoped probes)')
    a = ap.parse_args()
    if a.split == 'test' and a.allow_test_future:
        if not a.frozen_file or a.frozen_file.name != dc.B['test_measurement_freeze']:
            raise ValueError('TEST future frames need the stage-2 freeze record')
        dc.verify_freeze(a.frozen_file, required=[dc.PINS_PATH, Path(__file__)])
    windows = dc.names(a.names)
    dc.guard_names(a.bench, a.split, windows, a.diagnostic)
    future = a.split == 'dev' or a.allow_test_future
    result = [build_window(a.bench, a.split, w, a.out, a.band, future, a.frozen_file, a.diagnostic) for w in windows]
    dc.new_json(a.out / ('build-' + a.split + '.json'), dict(complete=True, split=a.split, band=a.band, windows=result,
                                                             TEST_future_allowed=a.allow_test_future,
                                                             names_sha256=dc.sha(a.names)))
    print(json.dumps(dict(complete=True, windows=len(result), out=str(a.out), GPU_started=False)))


if __name__ == '__main__':
    main()
