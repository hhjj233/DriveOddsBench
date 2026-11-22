"""DrivingWorld bench helpers: provenance, past-only inputs, geometry and the 10 Hz grid.

No torch import here: the input and spec builders run in the CPU research environment.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import pickletools
import re
import subprocess
import sys
import zipfile

HERE = Path(__file__).resolve().parent
PINS_PATH = HERE / 'drivingworld_pins.json'
PINS = json.loads(PINS_PATH.read_text())
B = PINS['bench']

SRC_WH = (1550, 2048)              # AV2 ring_front_center (portrait)
CANVAS_WH = (512, 288)
MODEL_WH = (512, 256)              # released image_size [256, 512] (H, W)
BENCH_BAND = (797, 1669)           # measurement band, full width; canvas = INTER_AREA of these rows
HISTORY_K = list(range(-20, 1))    # 21 history ticks (canvas prefix)
CONDITION_K = list(range(-14, 1))  # 15 conditioning frames (released condition_frames=15) at 10 Hz
FUTURE_K = list(range(1, 61))      # 60 generated frames
ALL_K = list(range(-20, 61))       # 81 canvas ticks


# ------------------------------------------------------------------------------------------------- files and hashes
def sha(path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def blob(path) -> str:
    """git blob sha1 of a file (the id `git ls-tree` prints)."""
    p = Path(path)
    h = hashlib.sha1(b'blob ' + str(p.stat().st_size).encode() + b'\0')
    with p.open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def new_json(path, value):
    """No-clobber JSON writer."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=1, allow_nan=False)
        f.write('\n')


def safe_relative(root, name) -> Path:
    rel = Path(name)
    if rel.is_absolute() or '..' in rel.parts:
        raise ValueError('relative input path required')
    root = Path(root).resolve()
    p = root / rel
    if not p.resolve().is_relative_to(root) or any(x.is_symlink() for x in [p, *p.parents] if x.is_relative_to(root)):
        raise ValueError('input path escapes root or uses symlink')
    return p


def names(path) -> list:
    values = Path(path).read_text().split()
    if len(values) != len(set(values)) or any(not re.fullmatch(r'[0-9a-f]{8}-t[0-9]+\.[0-9]+', v) for v in values):
        raise ValueError('unique bench window names required')
    return values


def guard_names(bench, split, windows, diagnostic=False):
    """Registered generation windows; diagnostic=True (DEV only) admits the whole DEV pool names-dev.txt."""
    if split not in ('dev', 'test'):
        raise ValueError('split must be dev or test')
    if diagnostic and split != 'dev':
        raise ValueError('diagnostic windows are DEV only')
    allowed = set(names(Path(bench) / ('names-dev.txt' if diagnostic else 'names-' + split + '-gen.txt')))
    if not windows or len(windows) != len(set(windows)) or not set(windows) <= allowed:
        raise ValueError('window set differs from registered split')


def verify_freeze(record, required=()):
    """SHA records only (the coordinator's guard files); call before any TEST future or TEST sample is read."""
    record = Path(record)
    if not record.is_file():
        raise ValueError('coordinator freeze record required')
    checked, pending, visited = {}, [record], set()
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        found = 0
        for line in current.read_text().splitlines():
            match = re.fullmatch(r'([0-9a-f]{64})\s+(.+?)\s*', line.strip())
            if not match:
                continue
            digest, name = match.groups()
            target = Path(name)
            if not target.is_absolute():
                target = current.parent / target
            if sha(target) != digest:
                raise ValueError('frozen hash mismatch: ' + str(target))
            checked[str(target.resolve())] = digest
            found += 1
            if target.name.endswith(('-files.sha256', '-frozen.txt')):
                pending.append(target)
        if not found:
            raise ValueError('empty or unsupported freeze record')
    if any(str(Path(p).resolve()) not in checked for p in required):
        raise ValueError('required runner/pins absent from freeze')
    return dict(path=str(record), sha256=sha(record), entries=len(checked))


# --------------------------------------------------------------------------------------------------- settings
def settings() -> dict:
    """The registered generation settings every spec carries and the runner re-checks."""
    s = B['sampling']
    return dict(size_wh=list(MODEL_WH), fps=B['fps'], canvas_fps=10, condition_frames=B['condition_frames'],
                condition_ticks=[CONDITION_K[0], CONDITION_K[-1]], future_frames=len(FUTURE_K),
                band=B['model_band'], image_sampler=s['image'], pose_sampler=s['pose'], pose_mode=s['pose_mode'],
                token_order=s['order'],
                pose_y_head=s['pose_y_head'], autocast=B['autocast'], batch_layout=B['batch_layout'],
                generator=B['generator'], uint8_rule=B['uint8_rule'], kv_cache=False, deterministic_cudnn=True)


def band(name=None) -> dict:
    """Geometry of a registered model band (pins bench.bands)."""
    name = name or B['model_band']
    if name not in B['bands']:
        raise ValueError('unregistered model band ' + str(name))
    g = dict(B['bands'][name], name=name)
    lo, hi = g['tall_source_rows']
    tw, th = g['tall_size_wh']
    if tw != CANVAS_WH[0] or (hi - lo) * CANVAS_WH[1] != th * (BENCH_BAND[1] - BENCH_BAND[0]):
        raise ValueError('tall render must share the canvas scale (288 rows per 872 source rows)')
    if (BENCH_BAND[0] - lo) * th % (hi - lo):
        raise ValueError('bench band must start on an integer tall-render row')
    g['canvas_offset'] = (BENCH_BAND[0] - lo) * th // (hi - lo)       # tall row of canvas row 0
    r0, r1 = g['tall_rows']
    if r1 - r0 != MODEL_WH[1] or g['canvas_rows'] != [r0 - g['canvas_offset'], r1 - g['canvas_offset']]:
        raise ValueError('model frame must be 256 tall-render rows')
    c0, c1 = g['canvas_rows']
    if g['valid_canvas_rows'] != [max(c0, 0), min(c1, CANVAS_WH[1])] or c0 > 0:
        raise ValueError('valid canvas rows must start at 0 (instrument supports a bottom black band only)')
    return g


def render(rgb, geometry):
    """(model 256x512, canvas 288x512) from one raw RGB camera frame: crop rows, cv2.resize INTER_AREA (render_version 2).

    The model frame is a 256-row window of the 'tall' render, which has the canvas scale and grid, so model rows map to
    canvas rows by an integer offset."""
    import cv2
    if rgb.shape[:2] != (SRC_WH[1], SRC_WH[0]):
        raise ValueError('raw AV2 camera shape changed')
    lo, hi = geometry['tall_source_rows']
    tall = cv2.resize(rgb[lo:hi], tuple(geometry['tall_size_wh']), interpolation=cv2.INTER_AREA)
    r0, r1 = geometry['tall_rows']
    canvas = cv2.resize(rgb[BENCH_BAND[0]:BENCH_BAND[1]], CANVAS_WH, interpolation=cv2.INTER_AREA)
    return tall[r0:r1].copy(), canvas, tall


def to_canvas(frames, geometry):
    """Model frames (N, 256, 512, 3) -> canvas (N, 288, 512, 3): integer row copy, uncovered rows black."""
    import numpy as np
    frames = np.asarray(frames)
    if frames.ndim != 4 or frames.shape[1:] != (MODEL_WH[1], MODEL_WH[0], 3) or frames.dtype != np.uint8:
        raise ValueError('uint8 model frames (N, 256, 512, 3) required')
    c0, c1 = geometry['canvas_rows']
    v0, v1 = geometry['valid_canvas_rows']
    out = np.zeros((len(frames), CANVAS_WH[1], CANVAS_WH[0], 3), np.uint8)
    out[:, v0:v1] = frames[:, v0 - c0:v1 - c0]
    return out


def timeline() -> list:
    """Canvas frame i = tick k = i - 20 = native10 frame i (10 Hz model = canvas rate); exact, no repeats."""
    return [dict(canvas_index=k + 20, k=k, native10_index=k + 20, time_s=k / 10,
                 source='prefix' if k <= 0 else 'generated') for k in ALL_K]


# --------------------------------------------------------------------------------------------- camera and poses
def tick_rows(window, with_future=False):
    """The window's own tick frames (nearest 20 Hz camera frame to anchor + k x 0.1 s, as canvas/real.mp4).

    Metadata only: a history request never selects a frame after the t0 camera frame."""
    ticks = {int(t['k']): t for t in window['ticks']}
    want = ALL_K if with_future else HISTORY_K
    if any(k not in ticks for k in want):
        raise ValueError('window ticks incomplete')
    t0 = int(ticks[0]['frame_timestamp_ns'])
    if t0 > int(window['anchor']['timestamp_ns']):
        raise ValueError('t0 camera image is later than the benchmark anchor')
    rows = [dict(k=k, frame_timestamp_ns=int(ticks[k]['frame_timestamp_ns']), path=ticks[k]['path'],
                 offset_ms=float(ticks[k].get('offset_ms', 0.0))) for k in want]
    if any(r['frame_timestamp_ns'] > t0 for r in rows if r['k'] <= 0):
        raise ValueError('future image in history')
    stamps = [r['frame_timestamp_ns'] for r in rows]
    if stamps != sorted(set(stamps)):
        raise ValueError('tick frames must be strictly increasing')
    return rows


def rel_poses(matrices):
    """Port of the released datasets/datasets_utils.get_meta_data: entry 0 = 0; entry i = inv(P_{i-1}) P_i,
    (x, y) of its translation (m) and the 'zyx' Euler z angle in degrees (scipy, as released)."""
    import numpy as np
    from scipy.spatial.transform import Rotation
    poses = np.asarray(matrices, dtype=np.float64)
    poses = np.concatenate([poses[0:1], poses], axis=0)
    rel = np.linalg.inv(poses[:-1]) @ poses[1:]
    xys = rel[:, :2, 3]
    yaws = np.degrees(Rotation.from_matrix(rel[:, :3, :3]).as_euler('zyx', degrees=False)[:, 0])[:, None]
    return xys, yaws


def pose_bins() -> dict:
    """Released pose tokenizer ranges (modules/tokenizers/pose_tokenizer.py)."""
    return dict(x_m=[0.0, 8.0, 128], y_m=[-0.5, 0.5, 128], yaw_deg=[-8.0, 8.0, 512])


def pose_tokens_np(xy, yaw):
    """numpy copy of the released poses_to_indices / yaws_to_indices (record and tests only)."""
    import numpy as np
    xy = np.asarray(xy, np.float64)
    yaw = np.asarray(yaw, np.float64)
    ix = np.clip(np.floor(xy[..., 0] * 128 / 8), 0, 127).astype(int)
    y_min = -0.5 + (1 / 128) / 2
    iy = np.clip(np.floor((xy[..., 1] - y_min) * 128 / 1), 0, 127).astype(int)
    yaw_min = -16 / 2 + (16 / 512) / 2
    iyaw = np.clip(np.floor((yaw - yaw_min) * 512 / 16), 0, 511).astype(int)
    return ix, iy, iyaw


# ------------------------------------------------------------------------------------------------- shared seeds
def shared_seed(window, sample, snapshot) -> int:
    root = Path(snapshot)
    path = root / PINS['seed_snapshot']['file']
    expected = PINS['seed_snapshot']['sha256']
    listed = {name: digest for digest, name in [line.split('  ', 1) for line in (root / 'MANIFEST.sha256').read_text().splitlines() if '  ' in line]}
    if sha(path) != expected or listed.get(PINS['seed_snapshot']['file']) != expected:
        raise ValueError('shared seed source hash mismatch')
    spec = importlib.util.spec_from_file_location('drivingworld_seed_snapshot', path)
    module = importlib.util.module_from_spec(spec)
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = old
    value = module.seed_of(window, sample)
    local = int.from_bytes(hashlib.sha256(f'bench-v1|{window}|{sample}'.encode()).digest()[:4], 'big') & 0x7fffffff
    if value != local:
        raise ValueError('shared seed rule drift')
    return value


# ----------------------------------------------------------------------------------------- official source/weights
def verify_source(root) -> dict:
    """Every tracked file of the pinned commit by git blob sha1, and no extra Python file in the imported packages."""
    root = Path(root).resolve()
    files = PINS['code']['files']
    for name, row in files.items():
        p = safe_relative(root, name)
        if not p.is_file() or blob(p) != row['git_blob_sha1']:
            raise ValueError('official source changed or absent: ' + name)
    for package in ('models', 'modules', 'utils', 'configs'):
        actual = {str(x.relative_to(root)) for x in (root / package).rglob('*.py') if '__pycache__' not in x.parts}
        expected = {n for n in files if n.startswith(package + '/') and n.endswith('.py')}
        if actual != expected:
            raise ValueError('unregistered Python source under ' + package)
    return dict(commit=PINS['code']['commit'], checked_files=len(files))


def verify_weights(root, full=True) -> dict:
    result = []
    for row in PINS['weights']['files']:
        p = safe_relative(root, row['path'])
        if not p.is_file() or p.stat().st_size != row['bytes']:
            raise ValueError('weight/config absent or wrong size: ' + row['path'])
        if full:
            if row.get('sha256') and sha(p) != row['sha256']:
                raise ValueError('weight SHA mismatch: ' + row['path'])
            if not row.get('sha256') and blob(p) != row['git_blob_sha1']:
                raise ValueError('weight metadata blob mismatch: ' + row['path'])
        result.append(row['path'])
    return dict(revision=PINS['weights']['revision'], files=len(result), full_hashes=full)


def pickle_globals(path) -> list:
    """(module, name) pairs a torch zip checkpoint's pickle would import; read with pickletools, nothing executed."""
    found = set()
    with zipfile.ZipFile(path) as z:
        pkls = [n for n in z.namelist() if n.endswith('data.pkl')]
        if len(pkls) != 1:
            raise ValueError('one data.pkl expected in a torch zip checkpoint')
        data = z.read(pkls[0])
    strings = []
    for op, arg, _ in pickletools.genops(data):
        if op.name in ('SHORT_BINUNICODE', 'BINUNICODE', 'UNICODE', 'BINUNICODE8', 'SHORT_BINSTRING', 'BINSTRING', 'STRING'):
            strings.append(arg)
        elif op.name in ('GLOBAL', 'INST'):
            module, name = arg.split(' ', 1)
            found.add((module, name))
        elif op.name == 'STACK_GLOBAL':
            if len(strings) < 2:
                raise ValueError('unreadable STACK_GLOBAL')
            found.add((strings[-2], strings[-1]))
    return sorted(found)


def check_pickle_globals(path, key) -> list:
    allowed = {tuple(x) for x in PINS['weights']['pickle_globals_allowed'][key]}
    used = pickle_globals(path)
    if not set(used) <= allowed:
        raise ValueError('checkpoint imports unexpected globals: ' + str(sorted(set(used) - allowed)))
    return used


def official_config(src) -> dict:
    """The released demo config (configs/drivingworld_v1/gen_videovq_conf_demo.py), hash-checked, as plain values."""
    import runpy
    path = Path(src) / PINS['code']['config_file']
    if blob(path) != PINS['code']['files'][PINS['code']['config_file']]['git_blob_sha1']:
        raise ValueError('official config changed')
    values = {k: v for k, v in runpy.run_path(str(path)).items() if not k.startswith('__')}
    for key, want in B['official_config'].items():
        if values.get(key) != want:
            raise ValueError(f'official config value drift: {key}')
    return values


# ------------------------------------------------------------------------------------------------- video I/O
def decode(path, wh, frames=None):
    import numpy as np
    cmd = ['ffmpeg', '-nostdin', '-v', 'error', '-xerror', '-threads', '2', '-i', str(path)]
    if frames is not None:
        cmd += ['-frames:v', str(int(frames))]
    data = subprocess.check_output(cmd + ['-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'])
    n = wh[0] * wh[1] * 3
    if len(data) % n:
        raise ValueError('decoded shape mismatch')
    return np.frombuffer(data, dtype=np.uint8).reshape(-1, wh[1], wh[0], 3)


def write_video(frames, path, fps):
    """No-clobber lossless writer (libx264rgb crf 0); a failed partial is retained for diagnosis."""
    import numpy as np
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    partial = path.with_name(path.stem + '.partial.mp4')
    if partial.exists():
        raise FileExistsError(partial)
    h, w = frames[0].shape[:2]
    proc = subprocess.Popen(['ffmpeg', '-nostdin', '-v', 'error', '-n', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}',
                             '-r', str(fps), '-i', '-', '-c:v', 'libx264rgb', '-crf', '0', '-preset', 'medium', '-threads', '2',
                             '-pix_fmt', 'rgb24', '-video_track_timescale', '10240', '-map_metadata', '-1', str(partial)],
                            stdin=subprocess.PIPE)
    try:
        for frame in frames:
            proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError('lossless writer failed; partial retained')
        os.link(partial, path)
        partial.unlink()
    except BaseException:
        if proc.poll() is None:
            proc.terminate()
            proc.wait()
        raise


def write_checked(frames, path, fps):
    """Write losslessly and prove it by decoding; returns the file record."""
    import numpy as np
    write_video(frames, path, fps)
    back = decode(path, (frames[0].shape[1], frames[0].shape[0]))
    if not np.array_equal(back, np.stack(frames)):
        raise ValueError('lossless decode mismatch: ' + str(path))
    return dict(sha256=sha(path), bytes=Path(path).stat().st_size, frames=len(frames), fps=fps,
                size_wh=[int(frames[0].shape[1]), int(frames[0].shape[0])])


def finite(*values) -> bool:
    return all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)
