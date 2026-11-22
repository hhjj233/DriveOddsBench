#!/usr/bin/env python3
"""Raw AV2 -> native 8 Hz ReWorld inputs in the registered 2:1 model geometry (video only: no prompt, no poses).

Adapted from build_drivelaw_inputs.py: same nearest raw-20-Hz timestamp rule and TEST-future guard. Per window:
history8 (17 real frames, t0-2..t0, codec prefix only), condition8 (its last 9 frames, t0-1..t0: the only frames the
model reads) and, for DEV or after the stage-2 freeze, real8 (65 frames, t0-2..t0+6, for the codec reference).
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import reworld_common as rc


def clip_name(kind):
    return f'{kind}8_{rc.W}.mp4'


def build_window(bench, split, window_name, out, with_future=False, frozen_file=None):
    if split == 'test' and with_future:
        rc.verify_freeze(frozen_file, required=[rc.PINS_PATH, Path(__file__)])
    rc.guard_names(bench, split, [window_name])
    source = Path(bench) / ('windows-' + split) / window_name / 'window.json'
    window = json.loads(source.read_text())
    if window['name'] != window_name or window['split'].lower() != split:
        raise ValueError('window split mismatch')
    # No frame is opened until the split/future guard and all timestamps pass.
    rows = rc.camera_rows(window, with_future)
    import cv2
    import numpy as np
    cv2.setNumThreads(2)
    dest = Path(out) / 'windows' / split / window_name
    if dest.exists():
        raise FileExistsError('fresh window destination required; existing outputs retained')
    dest.mkdir(parents=True, exist_ok=False)
    frames = []
    for row in rows:
        rgb = cv2.cvtColor(cv2.imread(row['path']), cv2.COLOR_BGR2RGB)
        frames.append(rc.render_model_frame(rgb))
    clips = {clip_name('history'): frames[:17], clip_name('condition'): frames[8:17]}
    if with_future:
        clips[clip_name('real')] = frames
    files = {}
    for name, value in clips.items():
        path = dest / name
        rc.write_video(value, path, 8)
        decoded = rc.decode(path, (rc.W, rc.H))
        if not np.array_equal(decoded, np.stack(value)):
            raise ValueError('native input lossless decode mismatch')
        files[name] = dict(sha256=rc.sha(path), bytes=path.stat().st_size, frames=len(value), fps=8, size_wh=[rc.W, rc.H])
    record = dict(schema='wm-calib-bench.reworld_video.inputs.v1', model='reworld_video', split=split, window=window_name,
                  names_list=rc.names_list(bench, split), source_window_sha256=rc.sha(source), raw_camera_rows=rows,
                  history_camera_rows=[r for r in rows if r['native_index'] <= 16], condition_camera_rows=rows[8:17],
                  max_camera_error_ms=max(abs(r['error_ns']) / 1e6 for r in rows),
                  t0_camera_ns=rows[16]['timestamp_ns'], anchor_ns=int(window['anchor']['timestamp_ns']),
                  geometry=dict(band_rows=rc.B['band_rows'], model_source_rows=list(rc.SRC_ROWS), native_wh=[rc.W, rc.H],
                                canvas_wh=[rc.CANVAS_W, rc.CANVAS_H], canvas_valid_rows=list(rc.CANVAS_ROWS), render='cv2.INTER_AREA'),
                  renderer=dict(python=sys.version.split()[0], cv2=cv2.__version__, numpy=np.__version__),
                  text_or_pose_inputs=None, mapping=rc.timeline(), files=files, future_present=with_future,
                  generator_sha256=rc.sha(Path(__file__)), common_sha256=rc.sha(rc.HERE / 'reworld_common.py'),
                  pins_sha256=rc.sha(rc.PINS_PATH))
    rc.new_json(dest / 'inputs.json', record)
    return dict(window=window_name, directory=str(dest), files=len(files), history_frames=17, condition_frames=9,
                real_frames=len(frames) if with_future else 0, max_camera_error_ms=record['max_camera_error_ms'])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bench', type=Path, required=True)
    ap.add_argument('--split', choices=['dev', 'test'], required=True)
    ap.add_argument('--names', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--allow-test-future', action='store_true')
    ap.add_argument('--frozen-file', type=Path)
    a = ap.parse_args()
    if a.split == 'test' and a.allow_test_future:
        rc.verify_freeze(a.frozen_file or '', required=[rc.PINS_PATH, Path(__file__)])
    windows = rc.names(a.names)
    rc.guard_names(a.bench, a.split, windows)
    result = [build_window(a.bench, a.split, w, a.out, a.split == 'dev' or a.allow_test_future, a.frozen_file) for w in windows]
    rc.new_json(a.out / ('build-' + a.split + '.json'), dict(complete=True, split=a.split, windows=result, TEST_future_allowed=a.allow_test_future))
    print(json.dumps(dict(complete=True, windows=len(result), out=str(a.out), GPU_started=False)))


if __name__ == '__main__':
    main()
