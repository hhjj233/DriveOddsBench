#!/usr/bin/env python3
"""Raw AV2 -> VaViM inputs: the 512x288 canvas render of the bench band, which is VaViM's own frame size (DEV by default).

  build_vavim_inputs.py --bench BENCH --split dev|test --names FILE --out INPUT_ROOT
                        [--diagnostic] [--allow-test-future --frozen-file STAGE2_RECORD]

Run with the research server's CPU environment (cv2, numpy, ffmpeg). Per window, under <INPUT_ROOT>/windows/<split>/<window>/:
  history10.mp4   21 frames (512x288, 10 fps, lossless): ticks -20..0 (the 10 Hz prefix option and the source of the context)
  context2.mp4    the frames at the context ticks (pins bench.context_ticks, default -20, -15, -10, -5, 0; 2 fps): the
                  only frames the model reads
  real10.mp4      81 frames, ticks -20..60 (DEV, for the real reference; TEST only after the stage-2 freeze)
  inputs.json     rows, checks, hashes and the canvas timeline

Every frame is the window's own tick frame (window.json "ticks": the 20 Hz camera frame nearest to anchor + k x 0.1 s,
exactly the frames of canvas/real.mp4), rendered by the bench rule (band rows [797, 1669), full width, cv2.INTER_AREA to
512x288). The render must equal canvas/real.mp4 bit for bit on every frame it covers (on TEST only the 21 history frames
are decoded from canvas/real.mp4, so no TEST future frame is decoded). No pose, command or text input exists for VaViM.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import vavim_common as vc  # noqa: E402


def build_window(bench, split, window_name, out, with_future=False, frozen_file=None, diagnostic=False):
    if split == 'test' and with_future:
        if not frozen_file or Path(frozen_file).name != vc.B['test_measurement_freeze']:
            raise ValueError('TEST future frames need the stage-2 freeze record')
        vc.verify_freeze(frozen_file, required=[vc.PINS_PATH, Path(__file__)])
    vc.guard_names(bench, split, [window_name], diagnostic)
    wdir = Path(bench) / ('windows-' + split) / window_name
    window = json.loads((wdir / 'window.json').read_text())
    if window['name'] != window_name or window['split'].lower() != split:
        raise ValueError('window split mismatch')
    rows = vc.tick_rows(window, with_future)        # metadata only; history rows never later than the t0 camera frame
    dest = Path(out) / 'windows' / split / window_name
    if dest.exists():
        raise FileExistsError('fresh window destination required; existing outputs retained')
    import cv2
    import numpy as np
    cv2.setNumThreads(2)
    frames = []
    for row in rows:
        bgr = cv2.imread(row['path'], cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError('unreadable camera frame ' + row['path'])
        frames.append(vc.render(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    ref = vc.decode(wdir / 'canvas' / 'real.mp4', vc.CANVAS_WH, frames=len(rows))
    if len(ref) != len(rows) or not np.array_equal(ref, np.stack(frames)):
        raise ValueError('the render differs from canvas/real.mp4')
    del ref
    dest.mkdir(parents=True, exist_ok=False)
    hist = frames[:len(vc.HISTORY_K)]
    context = [hist[k + 20] for k in vc.CONTEXT_K]
    files = {'history10.mp4': vc.write_checked(hist, dest / 'history10.mp4', 10),
             'context2.mp4': vc.write_checked(context, dest / 'context2.mp4', vc.B['model_fps'])}
    if with_future:
        files['real10.mp4'] = vc.write_checked(frames, dest / 'real10.mp4', 10)
    record = dict(schema='wm-calib-bench.vavim.inputs.v1', model='vavim', split=split, window=window_name,
                  diagnostic=bool(diagnostic), source_window_sha256=vc.sha(wdir / 'window.json'),
                  names_list=dict(file='names-dev.txt' if diagnostic else 'names-' + split + '-gen.txt',
                                  sha256=vc.sha(Path(bench) / ('names-dev.txt' if diagnostic else 'names-' + split + '-gen.txt'))),
                  geometry=dict(band_rows=list(vc.BENCH_BAND), model_wh=list(vc.MODEL_WH), canvas_wh=list(vc.CANVAS_WH),
                                render='crop source rows [797, 1669), full width; cv2.resize INTER_AREA to 512x288 (bench render_version 2)',
                                model_frame_equals_canvas=True),
                  rows=rows, history_ticks=vc.HISTORY_K, context_ticks=vc.CONTEXT_K, future_present=with_future,
                  checks=dict(canvas_reproduced_ticks=[r['k'] for r in rows], render_equals_canvas_real_mp4=True,
                              max_camera_offset_ms=max(abs(r['offset_ms']) for r in rows)),
                  files=files, timeline=vc.timeline(),
                  builder_sha256=vc.sha(Path(__file__)), common_sha256=vc.sha(vc.HERE / 'vavim_common.py'),
                  pins_sha256=vc.sha(vc.PINS_PATH), cv2=cv2.__version__, numpy=np.__version__, python=sys.version.split()[0])
    vc.new_json(dest / 'inputs.json', record)
    return dict(window=window_name, frames=len(rows), context_ticks=vc.CONTEXT_K, future=with_future,
                max_camera_offset_ms=record['checks']['max_camera_offset_ms'])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bench', type=Path, required=True)
    ap.add_argument('--split', choices=['dev', 'test'], required=True)
    ap.add_argument('--names', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--allow-test-future', action='store_true')
    ap.add_argument('--frozen-file', type=Path)
    ap.add_argument('--diagnostic', action='store_true', help='DEV only: admit any window of names-dev.txt (scoped probes)')
    a = ap.parse_args()
    if a.split == 'test' and a.allow_test_future:
        if not a.frozen_file or a.frozen_file.name != vc.B['test_measurement_freeze']:
            raise ValueError('TEST future frames need the stage-2 freeze record')
        vc.verify_freeze(a.frozen_file, required=[vc.PINS_PATH, Path(__file__)])
    windows = vc.names(a.names)
    vc.guard_names(a.bench, a.split, windows, a.diagnostic)
    future = a.split == 'dev' or a.allow_test_future
    result = [build_window(a.bench, a.split, w, a.out, future, a.frozen_file, a.diagnostic) for w in windows]
    vc.new_json(a.out / ('build-' + a.split + '.json'), dict(complete=True, split=a.split, windows=result,
                                                             TEST_future_allowed=a.allow_test_future,
                                                             names_sha256=vc.sha(a.names)))
    print(json.dumps(dict(complete=True, windows=len(result), out=str(a.out), GPU_started=False)))


if __name__ == '__main__':
    main()
