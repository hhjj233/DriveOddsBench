#!/usr/bin/env python3
"""Wan2.1-VACE-1.3B inputs for bench-v1: each window's 16 fps clips at a generation size (CPU, OpenCV): 512 x 288, the
measurement canvas, or 832 x 480, the official 480P size of the 1.3B model.

Wan2.1 runs at a fixed 16 fps (config sample_fps; VACE's processor keeps an 81-frame input frame for frame), and its VAE
needs 4k + 1 frames. The bench's 2.0 s history is therefore 33 = 4 x 8 + 1 frames (t0 - 2.0 .. t0; latents 0-8, t0 the last
frame of latent 8) and the 6.0 s future 96 frames, generated as two first-clip rounds of 48 new frames; 129 frames in all.
AV2's ring_front_center runs at 20 Hz, so the 16 fps grid is filled with real camera frames by the nearest-frame rule on
the camera's own clock (vace_pins.json bench.timeline_rule):

  frame j (0..128) = camera frame c0 + m(j),  m(j) = round_half_even(5 (j - 32) / 4),

c0 the window's t0 camera frame (window.json tick k = 0): the camera frame nearest to t0_cam + (j - 32) / 16 s; the ties
(25 ms each way) go to the even offset, i.e. to the bench's 10 Hz frame. Every frame is a real camera frame, the timing
error is <= 25 ms with no drift, and one camera frame in five is skipped (the 2.0 s history holds 33 of the 41 camera
frames c0-40 .. c0). The bench's 10 Hz ticks k are frames 32 + round(1.6 k) of this grid, exactly the camera frames c0 + 2k
of the bench (checked against window.json ticks below).

Rendering is the frozen render_version 2 of make_benchmark_windows.py, canvas part (read_rgb / resample_v2):
cv2.imread -> BGR2RGB -> the band rows [797, 1669) at full width -> cv2.resize(..., (W, H), INTER_AREA). At 512 x 288 this is
the bench canvas render itself (checked against canvas/real.mp4 at the ticks: identical). At 832 x 480 the band (aspect
1.7775) is squeezed 2.5 % horizontally into the model frame (aspect 1.7333); the adapter's INTER_AREA to 512 x 288 maps it
back onto the bench canvas (recorded per window against canvas/real.mp4: not identical, two area steps).

Writes <out>/windows/<split>/<window>/ (lossless libx264rgb -crf 0, rgb24, 16 fps; decoded and checked), TAG = 512 / 832:
  history16_TAG.mp4  33 frames (t0 - 2.0 .. t0): the model input of every continuation and of the prefix sample
  real16_TAG.mp4     129 frames (t0 - 2.0 .. t0 + 6.0): the real reference's input only. Built for DEV; for TEST only with
                     --allow-test-future, which also requires the frozen-protocol file
  frames16_TAG.json  the frame list (camera timestamps, offsets), checks, file hashes
An existing file whose decoded frames equal the new render is kept byte for byte (so spec hashes stay valid).
TEST is history-only unless --allow-test-future: no TEST future frame is read before the VACE protocol is frozen.

  build_vace_inputs.py --bench BENCH_V1_DIR --split dev|test --names FILE --out INPUT_ROOT [--size canvas|480p]
                       [--workers 4] [--allow-test-future --frozen-file PATH]
Run with the bench instrument's Python (OpenCV 5.0.0, the bench renderer's).
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vace_common as vc  # noqa: E402

SCHEMA = 'wm-calib-bench.wan21_vace13b.inputs.v1'
STEP_NS = 50_000_000
REG_TOL_NS = 100_000          # camera regularity: every 20 Hz step 50 ms within 0.1 ms
TOL_NS = 25_000_000           # a 16 fps instant lies within half a camera frame of its frame


def read_rgb(path):
    import cv2
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


def band_frame(img_rgb, wh):
    """make_benchmark_windows.resample_v2, canvas part (render_version 2), at size wh."""
    import cv2
    return cv2.resize(img_rgb[vc.BAND_ROWS[0]:vc.BAND_ROWS[1]], wh, interpolation=cv2.INTER_AREA)


def camera_list(window: dict, n: int) -> list[dict]:
    """16 fps frames j = 0..n-1 as camera frames of window.json frames_20hz (c0 + m(j)), with checks."""
    src = window['frames_20hz']
    ts = [int(x['timestamp_ns']) for x in src]
    ticks = {t['k']: t for t in window['ticks']}
    c0_ts = int(ticks[0]['frame_timestamp_ns'])
    idx0 = ts.index(c0_ts)
    rows = []
    for j in range(n):
        m = vc.camera_offset(j)
        i = idx0 + m
        if i < 0 or i >= len(src):
            raise ValueError(f'{window["name"]}: camera frame c0{m:+d} is outside frames_20hz')
        d = ts[i] - c0_ts
        if abs(d - m * STEP_NS) > REG_TOL_NS:
            raise ValueError(f'{window["name"]}: camera frame c0{m:+d} is {d / 1e6:.3f} ms from c0 (camera gap)')
        nominal = c0_ts + round((j - vc.T0) * 1e9 / vc.FPS)
        if abs(ts[i] - nominal) > TOL_NS + REG_TOL_NS:
            raise ValueError(f'{window["name"]}: frame {j} is {abs(ts[i] - nominal) / 1e6:.1f} ms from its 16 fps instant')
        rows.append(dict(j=j, camera_offset=m, timestamp_ns=ts[i], offset_ms=round((ts[i] - nominal) / 1e6, 3), path=src[i]['path']))
    for k, t in ticks.items():
        j = vc.frame_of_tick(k)
        if j < n and rows[j]['timestamp_ns'] != int(t['frame_timestamp_ns']):
            raise ValueError(f'{window["name"]}: 16 fps frame {j} is not the bench tick k={k} camera frame')
    return rows


def build_window(job: dict) -> dict:
    import cv2
    import numpy as np
    cv2.setNumThreads(2)          # the server's convention: 2 threads per process
    w, split, out_root, with_future = job['window'], job['split'], Path(job['out']), job['with_future']
    size = vc.SIZES[job['size']]
    names = vc.input_names(job['size'])
    t_start = time.time()
    bench = Path(job['bench'])
    wdir = bench / f'windows-{split}' / w
    window = json.loads((wdir / 'window.json').read_text())
    n = vc.N_ALL if with_future else vc.N_HIST
    rows = camera_list(window, n)
    frames = [band_frame(read_rgb(r['path']), size) for r in rows]
    out = out_root / 'windows' / split / w
    out.mkdir(parents=True, exist_ok=True)
    clips = {names['history']: frames[:vc.N_HIST]}
    if with_future:
        clips[names['real']] = frames
    checks, files = {}, {}
    for name, fr in clips.items():
        p = out / name
        keep = False
        if p.is_file():
            try:
                back = vc.decode(p, size)
                keep = len(back) == len(fr) and np.array_equal(back, np.stack(fr))
            except Exception:  # noqa: BLE001
                keep = False
        if not keep:
            vc.write_video(fr, p, vc.FPS)
        back = vc.decode(p, size)
        same = bool(len(back) == len(fr) and np.array_equal(back, np.stack(fr)))
        if not same:
            raise RuntimeError(f'{p}: decoded frames differ from the rendered ones')
        checks[name] = dict(frames=len(fr), decoded=int(len(back)), identical=same, kept_existing=keep, probe=vc.ffprobe(p))
        files[name] = dict(sha256=vc.sha256_file(p), bytes=p.stat().st_size)
    # the canvas check: the render at the ticks (on the canvas) vs the bench canvas clip (TEST: history ticks only)
    n_ticks = 81 if with_future else 21
    canvas = vc.decode(wdir / 'canvas' / 'real.mp4', vc.CANVAS, max_frames=None if with_future else 21)[:n_ticks]
    diffs = []
    for i in range(n_ticks):
        j = vc.frame_of_tick(i - 20)
        mine = frames[j] if size == vc.CANVAS else cv2.resize(frames[j], vc.CANVAS, interpolation=cv2.INTER_AREA)
        d = np.abs(mine.astype(np.int16) - canvas[i].astype(np.int16))
        diffs.append((int(d.max()), float(d.mean()), float(np.mean(d > 2))))
    rec = dict(schema=SCHEMA, window=w, split=split.upper(), written=datetime.now(timezone.utc).isoformat(timespec='seconds'),
               rule=vc.B['timeline_rule'], tick_rule=vc.B['tick_rule'], fps=vc.FPS, size=job['size'], size_wh=list(size),
               band_rows=list(vc.BAND_ROWS), frames=n, history_frames=vc.N_HIST, t0_index=vc.T0,
               t0_camera_timestamp_ns=rows[vc.T0]['timestamp_ns'], anchor_timestamp_ns=int(window['anchor']['timestamp_ns']),
               history_camera_frames=len({r['camera_offset'] for r in rows[:vc.N_HIST]}),
               history_skipped_camera_frames=sorted(set(range(-40, 1)) - {r['camera_offset'] for r in rows[:vc.N_HIST]}),
               max_abs_offset_ms=max(abs(r['offset_ms']) for r in rows), mean_abs_offset_ms=round(sum(abs(r['offset_ms']) for r in rows) / n, 3),
               ticks_match_bench=True, history_only=not with_future, rows=rows, clips=checks, files=files,
               canvas_check=dict(ticks=n_ticks, max_abs=max(d[0] for d in diffs), mean_abs=round(sum(d[1] for d in diffs) / n_ticks, 4),
                                 share_over_2=round(sum(d[2] for d in diffs) / n_ticks, 5),
                                 note=('the 512 x 288 render vs canvas/real.mp4 (the same direct render: identical expected)' if size == vc.CANVAS else
                                       'INTER_AREA(832x480 render) -> 512x288 vs canvas/real.mp4 (direct render): two area steps')),
               versions=dict(python=platform.python_version(), opencv=cv2.__version__, numpy=np.__version__),
               builder_sha256=vc.sha256_file(Path(__file__)), common_sha256=vc.sha256_file(HERE / 'vace_common.py'),
               pins_sha256=vc.sha256_file(vc.PINS_PATH), seconds=round(time.time() - t_start, 2))
    fp = out / names['frames']
    volatile = ('written', 'seconds', 'clips')     # kept_existing flips on a rebuild; the files' hashes stay in 'files'
    if fp.is_file() and all(c['kept_existing'] for c in checks.values()):
        old = json.loads(fp.read_text())
        if {k: v for k, v in old.items() if k not in volatile} == {k: v for k, v in rec.items() if k not in volatile}:
            rec = old                               # unchanged inputs: the frame list stays byte for byte (spec hashes stay valid)
    if not (fp.is_file() and json.loads(fp.read_text()) == rec):
        fp.write_text(json.dumps(rec, indent=1) + '\n')
    return dict(window=w, ok=True, files=files, canvas_max=rec['canvas_check']['max_abs'], canvas_mean=rec['canvas_check']['mean_abs'],
                seconds=rec['seconds'])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bench', type=Path, required=True); ap.add_argument('--split', choices=('dev', 'test'), required=True)
    ap.add_argument('--names', type=Path, required=True); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--size', choices=sorted(vc.SIZES), default=vc.B['default_size'],
                    help='canvas = 512 x 288 (the measurement canvas); 480p = 832 x 480 (the official 480P size)')
    ap.add_argument('--allow-test-future', action='store_true'); ap.add_argument('--frozen-file', type=Path, default=None)
    a = ap.parse_args()
    bad = vc.check_timeline()
    if bad:
        raise SystemExit('timeline self-check failed: ' + '; '.join(bad[:5]))
    names = [x.strip() for x in a.names.read_text().split() if x.strip()]
    allowed = {x.strip() for x in (a.bench / f'names-{a.split}.txt').read_text().split() if x.strip()}
    if set(names) - allowed:
        raise SystemExit(f'names outside names-{a.split}.txt: {sorted(set(names) - allowed)[:3]}')
    with_future = a.split == 'dev'
    if a.split == 'test' and a.allow_test_future:
        if not (a.frozen_file and a.frozen_file.is_file()):
            raise SystemExit('--allow-test-future needs --frozen-file pointing at the existing frozen-protocol file')
        with_future = True
    jobs = [dict(window=w, split=a.split, out=str(a.out), bench=str(a.bench), with_future=with_future, size=a.size) for w in names]
    t = time.time()
    with ProcessPoolExecutor(max(1, a.workers)) as ex:
        res = list(ex.map(build_window, jobs))
    size = vc.SIZES[a.size]
    manifest = dict(schema=SCHEMA + '.manifest', split=a.split.upper(), windows=len(res), with_future=with_future, size=a.size, size_wh=list(size),
                    written=datetime.now(timezone.utc).isoformat(timespec='seconds'), names_file=str(a.names.name),
                    names_sha256=vc.sha256_file(a.names), builder_sha256=vc.sha256_file(Path(__file__)), results=res,
                    canvas_max=max(r['canvas_max'] for r in res), canvas_mean=round(sum(r['canvas_mean'] for r in res) / len(res), 4),
                    seconds=round(time.time() - t, 1))
    mp = a.out / f'inputs-manifest-{a.split}-{a.names.stem}-{vc.TAGS[a.size]}.json'
    mp.write_text(json.dumps(manifest, indent=1) + '\n')
    print(f'{len(res)} windows ({"history + real" if with_future else "history only"}) at {size}; canvas check max {manifest["canvas_max"]}, '
          f'mean {manifest["canvas_mean"]}; {manifest["seconds"]} s -> {mp}')


if __name__ == '__main__':
    main()
