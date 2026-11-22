#!/usr/bin/env python3
"""LTX-2.5 inputs for bench-v1: the 20 Hz clips of each window at the canvas size and at 1024 x 576 (CPU, OpenCV).

LTX-2.5 needs 8k+1 frames and takes fps as an input. At 20 fps the bench's 2.0 s history is exactly 41 = 8 x 5 + 1
frames and the 8.0 s window exactly 161 = 8 x 20 + 1 frames, so the history fills latents 0-5 and the future starts on
latent 6. Every frame is a real camera frame: AV2's ring_front_center runs at 20 Hz, and frame j of the clip is the
camera frame nearest to t0 + (j - 40) x 0.05 s (j = 0..160), taken from window.json's frames_20hz list. The even frames
0, 2, ..., 160 are the bench's 10 Hz tick frames, and frame 40 is the t0 frame.

Rendering is the frozen render_version 2 of make_benchmark_windows.py, canvas part (resample_v2 / read_rgb):
cv2.imread -> BGR2RGB -> source rows [797, 1669) at full width -> cv2.resize(..., (W, H), INTER_AREA), at two sizes:
  512 x 288   the evaluation canvas (dev variant: model input; distilled variant: stage-1 conditioning). The builder checks
              that its even frames are pixel-identical to the window's canvas/real.mp4 (all 81 frames for DEV; frames
              0-20, decoded with -frames:v 21, for a TEST history-only build).
  1024 x 576  the distilled variant's final size (stage-2 conditioning, and the reference's input): the same band, the
              same call, at 2 x the canvas (a 0.66 downscale of the 1550 x 872 band). Recorded: how far
              INTER_AREA(1024 render -> 512 x 288) is from the direct 512 render (it is not identical: two area steps).

Writes <out>/windows/<split>/<window>/ (lossless libx264rgb -crf 0, rgb24 input, 20 fps; decoded and checked):
  history20.mp4 / history20_1024.mp4   41 frames (t0 - 2.0 .. t0)
  real20.mp4    / real20_1024.mp4      161 frames (t0 - 2.0 .. t0 + 6.0): the real reference's input only. Built for DEV;
                                       for TEST only with --allow-test-future, which also requires the frozen-protocol file
  frames20.json                        the frame list (timestamps, offsets, source paths), checks, file hashes, versions
An existing file whose decoded frames equal the new render is kept byte for byte (so spec hashes stay valid).
TEST is history-only unless --allow-test-future: no TEST future frame is read before the LTX-2.5 protocol is frozen.

  build_ltx25_inputs.py --bench BENCH_V1_DIR --split dev|test --names FILE --out INPUT_ROOT [--sizes 512x288,1024x576]
                        [--workers 4] [--allow-test-future --frozen-file PATH]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
PINS = json.loads((HERE / 'ltx25_pins.json').read_text())
B = PINS['bench']
CANVAS = tuple(B['canvas_wh'])                              # (512, 288)
HR = tuple(PINS['variants']['distilled']['final_size_wh'])  # (1024, 576)
FPS = int(B['fps'])
N_ALL, N_HIST = B['frames'], B['history_frames']          # 161, 41
T0_IDX = N_HIST - 1                                         # 40
STEP_NS = 50_000_000
TOL_NS = 25_000_000                                         # half a 20 Hz frame
BAND_ROWS = (797, 1669)                                     # protocol-v1 section 2 measurement band
SCHEMA = 'wm-calib-bench.ltx25.inputs.v2'
SUFFIX = {CANVAS: '', HR: '_1024'}


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


def write_video(frames, path: Path, fps: int) -> None:
    """make_wm_calibration_windows.write_video with -r fps (libx264rgb -crf 0 -preset medium, rgb24, timescale 10240)."""
    import numpy as np
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


def decode(path: Path, wh, max_frames: int | None = None):
    import numpy as np
    cmd = ['ffmpeg', '-v', 'error', '-threads', '2', '-i', str(path)]
    if max_frames is not None:
        cmd += ['-frames:v', str(max_frames)]
    raw = subprocess.check_output(cmd + ['-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'])
    return np.frombuffer(raw, np.uint8).reshape(-1, wh[1], wh[0], 3)


def frame_list(window: dict, n: int) -> list[dict]:
    """Frames j = 0..n-1 of the 20 Hz timeline: the frames_20hz entry nearest to t0 + (j - 40) x 50 ms."""
    import numpy as np
    src = window['frames_20hz']
    ts = np.array([x['timestamp_ns'] for x in src], dtype=np.int64)
    t0 = int(window['anchor']['timestamp_ns'])
    grid = t0 + (np.arange(n, dtype=np.int64) - T0_IDX) * STEP_NS
    idx = np.abs(ts[None, :] - grid[:, None]).argmin(1)
    off = ts[idx] - grid
    if np.abs(off).max() > TOL_NS:
        raise ValueError(f'{window["name"]}: a 20 Hz tick has no camera frame within 25 ms ({np.abs(off).max() / 1e6:.1f} ms)')
    if len(set(idx.tolist())) != n:
        raise ValueError(f'{window["name"]}: repeated camera frames on the 20 Hz timeline')
    ticks = {t['k']: t for t in window['ticks']}
    for j in range(0, n, 2):
        k = (j - T0_IDX) // 2
        if int(ts[idx[j]]) != int(ticks[k]['frame_timestamp_ns']):
            raise ValueError(f'{window["name"]}: 20 Hz frame {j} is not the 10 Hz tick k={k} frame')
    return [dict(j=int(j), t_s=round((j - T0_IDX) * 0.05, 2), tick_timestamp_ns=int(grid[j]), frame_timestamp_ns=int(ts[idx[j]]),
                 offset_ms=round(float(off[j]) / 1e6, 3), path=src[idx[j]]['path'], tick_k=(j - T0_IDX) // 2 if j % 2 == 0 else None)
            for j in range(n)]


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
        write_video(frames, path, FPS)
        if not np.array_equal(decode(path, wh), arr):
            raise RuntimeError(f'{path}: decoded frames differ from the rendered ones')
    return dict(frames=len(frames), fps=FPS, size_wh=list(wh), sha256=sha256_file(path), bytes=path.stat().st_size,
                frames_sha256=hashlib.sha256(arr.tobytes()).hexdigest(), kept_existing=kept)


def build_one(job: dict) -> dict:
    import cv2
    import numpy as np
    cv2.setNumThreads(2)
    t_start = time.time()
    wdir, out, history_only, sizes = Path(job['wdir']), Path(job['out']), job['history_only'], [tuple(s) for s in job['sizes']]
    window = json.loads((wdir / 'window.json').read_text())
    if window.get('render_version') != 2:
        raise ValueError(f'{wdir}: render_version {window.get("render_version")}, expected 2')
    n = N_HIST if history_only else N_ALL
    frames = frame_list(window, n)
    renders = {wh: [] for wh in sizes}
    for f in frames:
        img = read_rgb(f['path'])
        for wh in sizes:
            renders[wh].append(band_frame(img, wh))
    checks = {}
    if CANVAS in renders:  # identity with the bench's 10 Hz canvas clip on the shared frames
        bench = decode(wdir / 'canvas' / 'real.mp4', CANVAS, max_frames=(T0_IDX // 2 + 1) if history_only else None)
        even = np.stack(renders[CANVAS][0::2])
        if bench.shape != even.shape or not np.array_equal(bench, even):
            d = np.abs(bench.astype(np.int16) - even.astype(np.int16)) if bench.shape == even.shape else None
            raise RuntimeError(f'{window["name"]}: even 20 Hz frames differ from canvas/real.mp4 (shapes {bench.shape} {even.shape}, '
                               f'max diff {None if d is None else int(d.max())})')
        checks['canvas_identity'] = dict(reference='canvas/real.mp4' + (' frames 0-20' if history_only else ''), frames_compared=int(even.shape[0]),
                                         identical=True, reference_sha256=sha256_file(wdir / 'canvas' / 'real.mp4'))
    if CANVAS in renders and HR in renders:  # two area steps vs one (diagnostic, history frames only)
        down = np.stack([cv2.resize(x, CANVAS, interpolation=cv2.INTER_AREA) for x in renders[HR][:N_HIST]])
        d = np.abs(down.astype(np.int16) - np.stack(renders[CANVAS][:N_HIST]).astype(np.int16))
        checks['hr_down_vs_canvas_history'] = dict(max_abs=int(d.max()), mean_abs=round(float(d.mean()), 4), frames=N_HIST)
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for wh in sizes:
        sfx = SUFFIX[wh]
        files[f'history20{sfx}.mp4'] = put(renders[wh][:N_HIST], out / f'history20{sfx}.mp4', wh)
        if not history_only:
            files[f'real20{sfx}.mp4'] = put(renders[wh], out / f'real20{sfx}.mp4', wh)
    rec = dict(schema=SCHEMA, window=window['name'], split=window['split'], history_only=history_only, sizes=[list(s) for s in sizes], files=files,
               timeline=dict(fps=FPS, frames=n, t0_index=T0_IDX, rule='frame j = camera frame nearest to t0 + (j - 40) x 0.05 s',
                             max_abs_offset_ms=max(abs(f['offset_ms']) for f in frames)),
               checks=checks, window_json_sha256=sha256_file(wdir / 'window.json'), frames=frames,
               software=dict(python=platform.python_version(), opencv=cv2.__version__, numpy=np.__version__,
                             builder_sha256=sha256_file(Path(__file__)), pins_sha256=sha256_file(HERE / 'ltx25_pins.json')),
               seconds=round(time.time() - t_start, 1), created_utc=datetime.now(timezone.utc).isoformat(timespec='seconds'))
    (out / 'frames20.json').write_text(json.dumps(rec, indent=1) + '\n')
    return dict(window=window['name'], ok=True, seconds=rec['seconds'], files=sorted(files),
                rewritten=sorted(k for k, v in files.items() if not v['kept_existing']), hr_down=checks.get('hr_down_vs_canvas_history'))


def done(job: dict) -> bool:
    f = Path(job['out']) / 'frames20.json'
    if not f.is_file():
        return False
    rec = json.loads(f.read_text())
    want = {f'history20{SUFFIX[tuple(s)]}.mp4' for s in job['sizes']} | (set() if job['history_only'] else {f'real20{SUFFIX[tuple(s)]}.mp4' for s in job['sizes']})
    return rec.get('schema') == SCHEMA and want <= set(rec.get('files', {})) and all(
        (Path(job['out']) / k).is_file() and sha256_file(Path(job['out']) / k) == rec['files'][k]['sha256'] for k in want)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bench', type=Path, required=True, help='bench-v1 directory (windows-dev/, windows-test/)')
    ap.add_argument('--split', required=True, choices=['dev', 'test'])
    ap.add_argument('--names', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True, help='input root (windows/<split>/<window>/ is created)')
    ap.add_argument('--sizes', default=f'{CANVAS[0]}x{CANVAS[1]},{HR[0]}x{HR[1]}', help='comma-separated WxH (512x288 and/or 1024x576)')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--allow-test-future', action='store_true', help='build TEST real20*.mp4 (needs --frozen-file)')
    ap.add_argument('--frozen-file', type=Path, help='the LTX-2.5 protocol freeze record; must exist for --allow-test-future')
    args = ap.parse_args()
    sizes = [tuple(int(v) for v in s.split('x')) for s in args.sizes.split(',')]
    if any(s not in SUFFIX for s in sizes):
        raise SystemExit(f'sizes must be among {list(SUFFIX)}')
    history_only = args.split == 'test' and not args.allow_test_future
    if args.split == 'test' and args.allow_test_future and not (args.frozen_file and args.frozen_file.is_file()):
        raise SystemExit('--allow-test-future needs --frozen-file pointing at the existing freeze record')
    names = args.names.read_text().split()
    jobs = [dict(wdir=str(args.bench / f'windows-{args.split}' / w), out=str(args.out / 'windows' / args.split / w), history_only=history_only,
                 sizes=[list(s) for s in sizes]) for w in names]
    todo = [j for j in jobs if not done(j)]
    print(json.dumps(dict(split=args.split, windows=len(jobs), to_build=len(todo), history_only=history_only, sizes=args.sizes)), flush=True)
    results, failures = [], []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
        for job, fut in [(j, ex.submit(build_one, j)) for j in todo]:
            try:
                r = fut.result()
                results.append(r)
                print(json.dumps(r), flush=True)
            except Exception as exc:  # noqa: BLE001
                failures.append(dict(window=Path(job['out']).name, error=f'{type(exc).__name__}: {exc}'))
                print(json.dumps(failures[-1]), file=sys.stderr, flush=True)
    man = dict(schema=SCHEMA + '.manifest', split=args.split, names=str(args.names), names_sha256=sha256_file(args.names), windows=len(jobs),
               built_now=len(results), failures=failures, history_only=history_only, sizes=args.sizes, builder_sha256=sha256_file(Path(__file__)),
               created_utc=datetime.now(timezone.utc).isoformat(timespec='seconds'))
    (args.out / 'windows' / args.split).mkdir(parents=True, exist_ok=True)
    (args.out / 'windows' / args.split / f'manifest-{args.names.stem}.json').write_text(json.dumps(man, indent=1) + '\n')
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
