"""VaViM bench helpers: provenance, guards, the 2 Hz model grid and its map onto the 10 Hz canvas, lossless video I/O.

No torch import at module level: the input and spec builders run in the CPU research environment.

Geometry: VaViM's frame is 512x288 (32x18 tokens of the 16x16 LlamaGen VQ), which is exactly the bench canvas: the model
frame IS the canvas render of the measurement band (cv2.INTER_AREA of source rows [797, 1669), full width), so no crop,
pad or second resize exists between model and canvas.
Time: VaViM is trained on 8-frame clips at 2 Hz (OpenDV/nuPlan 10 Hz subsampled by 5; nuScenes key frames). The model
reads the context ticks (pins bench.context_ticks) and generates one frame per 0.5 s (ticks 5, 10, ..., 60). The canvas is
10 Hz: canvas frame i = tick k = i - 20 shows the model frame of tick model_tick(k) (pins bench.canvas_map).
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
PINS_PATH = HERE / 'vavim_pins.json'
PINS = json.loads(PINS_PATH.read_text())
B = PINS['bench']

SRC_WH = (1550, 2048)                       # AV2 ring_front_center (portrait)
CANVAS_WH = (512, 288)
MODEL_WH = tuple(B['model_wh'])             # (512, 288): the canvas itself
BENCH_BAND = tuple(B['band_rows'])          # (797, 1669)
STEP = B['model_step_ticks']                # 5 ticks = 0.5 s
HISTORY_K = list(range(-20, 1))             # 21 history ticks
CONTEXT_K = list(B['context_ticks'])        # ticks the model reads
FUTURE_K = list(range(STEP, 61, STEP))      # 12 generated ticks: 5, 10, ..., 60
NATIVE2_K = list(range(-20, 61, STEP))      # 17 ticks of the 2 Hz native clip
ALL_K = list(range(-20, 61))                # 81 canvas ticks
TOKEN_HW = tuple(B['token_grid_hw'])        # (18, 32)
CODEBOOK = B['codebook_size']               # 16384 detokenizer entries (the GPT head has 16385 rows)


# ------------------------------------------------------------------------------------------------- files and hashes
def sha(path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def sha_bytes(data) -> str:
    return hashlib.sha256(data).hexdigest()


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


def check_purpose(purpose, split, run, mode, window, sample):
    """main / smoke / diagnostic runs are separated by run prefix, windows and samples (smoke and diagnostic: DEV only)."""
    smoke = B['smoke']
    diag_prefix = B['diagnostic_run_prefix']
    if purpose not in ('main', 'smoke', 'diagnostic'):
        raise ValueError('unregistered purpose')
    if not run.startswith('vavim-' + split + '-'):
        raise ValueError('registered run/split required')
    special = run.startswith((smoke['run_prefix'], diag_prefix))
    if purpose == 'main':
        if special:
            raise ValueError('main runs may not use the smoke or diagnostic run prefixes')
        return
    if split != 'dev':
        raise ValueError('smoke and diagnostic runs are DEV only')
    if purpose == 'diagnostic':
        if not run.startswith(diag_prefix):
            raise ValueError('diagnostic runs are named ' + diag_prefix + '*')
        return
    if not run.startswith(smoke['run_prefix']) or window not in smoke['windows']:
        raise ValueError('smoke: run prefix or window outside the registered set')
    if mode == 'continue' and sample not in smoke['samples']:
        raise ValueError('smoke: sample outside the registered seeds')


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
    return dict(model_wh=list(MODEL_WH), model_fps=B['model_fps'], canvas_fps=B['canvas_fps'],
                context_ticks=CONTEXT_K, future_ticks=FUTURE_K, nb_timesteps=B['nb_timesteps'],
                temperature=s['temperature'], top_k=s['top_k'], use_kv_cache=s['use_kv_cache'],
                generator=s['generator'], autocast=B['autocast'], batch_layout=B['batch_layout'],
                canvas_map=B['canvas_map'], prefix=B['prefix'], prefix_rate=B['prefix_rate'],
                reference_rule=B['reference_rule'], uint8_rule=B['uint8_rule'], input_rule=B['input_rule'],
                deterministic_cudnn=True)


def model_tick(k, rule=None) -> int:
    """Canvas tick k -> the tick of the 2 Hz model frame shown there. 'nearest': 5 round(k / 5) (no ties exist for
    integer k); 'hold': 5 floor(k / 5) (the latest model frame at or before k)."""
    rule = rule or B['canvas_map']
    if k not in range(-20, 61):
        raise ValueError('canvas tick outside the registered range')
    q, r = divmod(k, STEP)
    if rule == 'nearest':
        return STEP * (q + (1 if r > STEP // 2 else 0))
    if rule == 'hold':
        return STEP * q
    raise ValueError('unregistered canvas map ' + str(rule))


def timeline(rule=None, prefix_rate=None) -> list:
    rule = rule or B['canvas_map']
    prefix_rate = prefix_rate or B['prefix_rate']
    out = []
    for k in ALL_K:
        m = model_tick(k, rule)
        if prefix_rate == '10hz' and k <= 0:
            out.append(dict(canvas_index=k + 20, k=k, model_tick=k, source='prefix10', native2_index=None, shown_offset_ms=0))
        else:
            out.append(dict(canvas_index=k + 20, k=k, model_tick=m, source='prefix2' if m <= 0 else 'generated',
                            native2_index=(m + 20) // STEP, shown_offset_ms=(m - k) * 100))
    return out


def assemble_canvas(native2, prefix10=None, rule=None, prefix_rate=None):
    """81 canvas frames from the 17-frame 2 Hz native clip (ticks -20, -15, ..., 60) and, for prefix_rate '10hz', the
    21-frame 10 Hz prefix (ticks -20..0). Pure frame copies: canvas frame i is one of the given frames, bit for bit."""
    import numpy as np
    prefix_rate = prefix_rate or B['prefix_rate']
    native2 = np.asarray(native2)
    if native2.shape != (len(NATIVE2_K), MODEL_WH[1], MODEL_WH[0], 3) or native2.dtype != np.uint8:
        raise ValueError('uint8 native clip (17, 288, 512, 3) required')
    if prefix_rate == '10hz':
        prefix10 = np.asarray(prefix10)
        if prefix10.shape != (len(HISTORY_K), MODEL_WH[1], MODEL_WH[0], 3) or prefix10.dtype != np.uint8:
            raise ValueError('uint8 10 Hz prefix (21, 288, 512, 3) required')
        if not np.array_equal(prefix10[::STEP], native2[:len(HISTORY_K[::STEP])]):
            raise ValueError('10 Hz prefix and 2 Hz prefix disagree at the 2 Hz ticks')
    elif prefix_rate != '2hz':
        raise ValueError('unregistered prefix rate')
    frames = []
    for row in timeline(rule, prefix_rate):
        frames.append(prefix10[row['k'] + 20] if row['source'] == 'prefix10' else native2[row['native2_index']])
    return np.stack(frames)


# --------------------------------------------------------------------------------------------- camera frames
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


def render(rgb):
    """Raw RGB camera frame -> the 512x288 canvas = the VaViM model frame (crop band rows, cv2.resize INTER_AREA)."""
    import cv2
    if rgb.shape[:2] != (SRC_WH[1], SRC_WH[0]):
        raise ValueError('raw AV2 camera shape changed')
    return cv2.resize(rgb[BENCH_BAND[0]:BENCH_BAND[1]], CANVAS_WH, interpolation=cv2.INTER_AREA)


# ------------------------------------------------------------------------------------------------- shared seeds
def local_seed(window, sample) -> int:
    return int.from_bytes(hashlib.sha256(f'bench-v1|{window}|{sample}'.encode()).digest()[:4], 'big') & 0x7fffffff


def shared_seed(window, sample, snapshot) -> int:
    root = Path(snapshot)
    path = root / PINS['seed_snapshot']['file']
    expected = PINS['seed_snapshot']['sha256']
    listed = {name: digest for digest, name in [line.split('  ', 1) for line in (root / 'MANIFEST.sha256').read_text().splitlines() if '  ' in line]}
    if sha(path) != expected or listed.get(PINS['seed_snapshot']['file']) != expected:
        raise ValueError('shared seed source hash mismatch')
    spec = importlib.util.spec_from_file_location('vavim_seed_snapshot', path)
    module = importlib.util.module_from_spec(spec)
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = old
    value = module.seed_of(window, sample)
    if value != local_seed(window, sample):
        raise ValueError('shared seed rule drift')
    return value


# ----------------------------------------------------------------------------------------- official source/weights
def verify_source(root) -> dict:
    """Every tracked file of the pinned commit by git blob sha1, and no extra Python file under vam/."""
    root = Path(root).resolve()
    files = PINS['code']['files']
    for name, row in files.items():
        p = safe_relative(root, name)
        if not p.is_file() or p.stat().st_size != row['bytes'] or blob(p) != row['git_blob_sha1']:
            raise ValueError('official source changed or absent: ' + name)
    actual = {str(x.relative_to(root)) for x in (root / 'vam').rglob('*.py') if '__pycache__' not in x.parts}
    expected = {n for n in files if n.startswith('vam/') and n.endswith('.py')}
    if actual != expected:
        raise ValueError('unregistered Python source under vam/: ' + str(sorted(actual ^ expected)[:5]))
    return dict(commit=PINS['code']['commit'], checked_files=len(files))


def weight_path(root, role):
    rows = [r for r in PINS['weights']['files'] if r['role'] == role]
    if len(rows) != 1:
        raise ValueError('ambiguous weight role ' + role)
    return safe_relative(root, rows[0]['path'])


def verify_weights(root, full=True) -> dict:
    result = []
    for row in PINS['weights']['files']:
        p = safe_relative(root, row['path'])
        if not p.is_file() or p.stat().st_size != row['bytes']:
            raise ValueError('weight file absent or wrong size: ' + row['path'])
        if full and sha(p) != row['sha256']:
            raise ValueError('weight SHA mismatch: ' + row['path'])
        result.append(row['path'])
    return dict(release=PINS['weights']['release']['tag'], files=len(result), full_hashes=full)


def pickle_globals(path) -> list:
    """(module, name) pairs the pickles of a torch zip archive would import; read with pickletools, nothing executed."""
    found = set()
    with zipfile.ZipFile(path) as z:
        pkls = [n for n in z.namelist() if n.endswith('.pkl')]
        if not pkls:
            raise ValueError('no pickle in the archive')
        for name in pkls:
            strings = []
            for op, arg, _ in pickletools.genops(z.read(name)):
                if op.name in ('SHORT_BINUNICODE', 'BINUNICODE', 'UNICODE', 'BINUNICODE8', 'SHORT_BINSTRING', 'BINSTRING', 'STRING'):
                    strings.append(arg)
                elif op.name in ('GLOBAL', 'INST'):
                    module, attr = arg.split(' ', 1)
                    found.add((module, attr))
                elif op.name == 'STACK_GLOBAL':
                    if len(strings) < 2:
                        raise ValueError('unreadable STACK_GLOBAL')
                    found.add((strings[-2], strings[-1]))
    return sorted(found)


def check_pickle_globals(path, key) -> list:
    allowed = {tuple(x) for x in PINS['weights']['pickle_globals_allowed'][key]}
    used = pickle_globals(path)
    if not set(used) <= allowed:
        raise ValueError('archive imports unexpected globals: ' + str(sorted(set(used) - allowed)))
    return used


# --------------------------------------------------------------------------------- official pixel conventions
def to_model_input(torch, frames_u8, device):
    """uint8 (N, 288, 512, 3) -> float32 (N, 3, 288, 512) in [-1, 1]: the official transforms (ToDtype(float32,
    scale=True) = x / 255, then Normalize = 2 x - 1; vam/datalib/transforms.py, token_creator.py)."""
    x = torch.from_numpy(frames_u8.copy()).permute(0, 3, 1, 2).to(torch.float32) / 255.0
    return (2 * x - 1).to(device)


def to_uint8(torch, images):
    """Official torch_image_to_plot: clamp(127.5 x + 128, 0, 255) -> uint8 (truncation), (N, 3, H, W) -> (N, H, W, 3)."""
    return torch.clamp(127.5 * images.float() + 128.0, 0, 255).permute(0, 2, 3, 1).to('cpu', dtype=torch.uint8).numpy()


def check_tokens(tokens):
    """Generated tokens must index the 16384-entry detokenizer codebook (the GPT head has one extra row, 16384)."""
    import numpy as np
    t = np.asarray(tokens)
    bad = int(((t < 0) | (t >= CODEBOOK)).sum())
    if bad:
        raise ValueError(f'{bad} token(s) outside the detokenizer codebook (e.g. the unused GPT row 16384)')
    return t


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
