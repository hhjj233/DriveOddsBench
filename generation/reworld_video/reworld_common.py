"""ReWorld-Video bench helpers: strict provenance, past-only inputs, exact grids and the canvas map.

Adapted from models/drivelaw_video/drivelaw_common.py (same 8 Hz native grid, same raw-camera timestamp rule, same
freeze guard, same shared seed rule and same lossless writer). ReWorld-specific: no text prompt, a 2:1 model frame
(see reworld_pins.json bench.geometry) and its map back to the 512x288 canvas.
"""
from __future__ import annotations
import bisect
from fractions import Fraction
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
PINS_PATH = HERE / 'reworld_pins.json'
PINS = json.loads(PINS_PATH.read_text())
B = PINS['bench']
GEOM = B['geometry']
W, H = B['size_wh']                      # model frame (width, height)
CANVAS_W, CANVAS_H = 512, 288
SRC_W, SRC_H = 1550, 2048                # raw AV2 ring_front_center (portrait)
SRC_ROWS = tuple(GEOM['model_source_rows'])
CANVAS_ROWS = tuple(GEOM['canvas_valid_rows'])


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def blob(path):
    p = Path(path)
    h = hashlib.sha1(b'blob ' + str(p.stat().st_size).encode() + b'\0')
    with p.open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def new_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def safe_relative(root, name):
    rel = Path(name)
    if rel.is_absolute() or '..' in rel.parts:
        raise ValueError('relative input path required')
    root = Path(root).resolve()
    p = root / rel
    if not p.resolve().is_relative_to(root) or any(x.is_symlink() for x in [p, *p.parents] if x.is_relative_to(root)):
        raise ValueError('input path escapes root or uses symlink')
    return p


def names(path):
    values = Path(path).read_text().split()
    if len(values) != len(set(values)) or any(not re.fullmatch(r'[0-9a-f]{8}-t[0-9]+\.[0-9]+', v) for v in values):
        raise ValueError('unique bench window names required')
    return values


def nearest(timestamps, target):
    """Nearest integer timestamp; ties go to the earlier observation."""
    i = bisect.bisect_left(timestamps, target)
    candidates = [j for j in (i - 1, i) if 0 <= j < len(timestamps)]
    if not candidates:
        raise ValueError('empty timestamp source')
    return min(candidates, key=lambda j: (abs(timestamps[j] - target), timestamps[j]))


def native_index(k):
    """10 Hz tick k = -20..60 -> nearest native 8 Hz frame j = 0..64 (t0 at 16); ties go to the earlier frame."""
    if k not in range(-20, 61):
        raise ValueError('canvas tick outside registered range')
    q = Fraction(16) + Fraction(4 * k, 5)
    lo = q.numerator // q.denominator
    return min((lo, lo + 1), key=lambda j: (abs(Fraction(j) - q), j))


def timeline():
    return [dict(k=k, native8_index=native_index(k),
                 native_minus_canvas_ms=float((Fraction(native_index(k) - 16, 8) - Fraction(k, 10)) * 1000))
            for k in range(-20, 61)]


def camera_rows(window, with_future=False):
    """Read metadata only. History requests never select or read a future image."""
    src = window['frames_20hz']
    timestamps = [int(x['timestamp_ns']) for x in src]
    if timestamps != sorted(set(timestamps)):
        raise ValueError('camera timestamps must be strictly increasing')
    tick0 = next(x for x in window['ticks'] if x['k'] == 0)
    t0 = int(tick0['frame_timestamp_ns'])
    anchor = int(window['anchor']['timestamp_ns'])
    if t0 > anchor:
        raise ValueError('t0 camera image is later than benchmark anchor')
    offsets = range(-16, 49) if with_future else range(-16, 1)
    rows = []
    for offset in offsets:
        target = t0 + offset * 125_000_000
        eligible = timestamps if with_future else [t for t in timestamps if t <= t0]
        j = nearest(eligible, target)
        stamp = eligible[j]
        index = bisect.bisect_left(timestamps, stamp)
        if abs(stamp - target) > 25_100_000:
            raise ValueError('raw 20 Hz nearest-frame error exceeds 25.1 ms')
        rows.append(dict(native_index=offset + 16, relative_s=offset / 8, target_timestamp_ns=target,
                         timestamp_ns=stamp, error_ns=stamp - target, path=src[index]['path']))
    if any(r['timestamp_ns'] > t0 for r in rows if r['native_index'] <= 16):
        raise ValueError('future image in history')
    if len({r['timestamp_ns'] for r in rows}) != len(rows):
        raise ValueError('duplicate raw conditioning timestamps')
    return rows


def verify_freeze(record, required=()):
    """SHA records only; call before any TEST future or TEST sample is read."""
    record = Path(record)
    if not record.is_file():
        raise ValueError('coordinator freeze record required')
    checked = {}
    pending = [record]
    visited = set()
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


def guard_names(bench, split, windows):
    if split not in ('dev', 'test'):
        raise ValueError('split must be dev or test')
    allowed = set(names(Path(bench) / ('names-' + split + '-gen.txt')))
    if not windows or len(windows) != len(set(windows)) or not set(windows) <= allowed:
        raise ValueError('window set differs from registered split')


def names_list(bench, split):
    """The window list a spec was validated against (registered list or the authorised diagnostic view)."""
    path = Path(bench) / ('names-' + split + '-gen.txt')
    return dict(file=path.name, sha256=sha(path))


PURPOSES = ('main', 'smoke', 'timescale-diagnostic')


def check_names_list(purpose, split, listed):
    """Main and smoke runs use the registered window list; only the D9 diagnostic uses the authorised bench view."""
    want = (B['timescale_diagnostic']['bench_view_names_dev_gen_sha256'] if purpose == 'timescale-diagnostic'
            else B['registered_names_sha256'][split])
    if not listed or listed.get('sha256') != want:
        raise ValueError(f'{purpose}: window list is not the one registered for this purpose')


def check_purpose(purpose, split, run, mode, window, sample):
    """Smoke and the D9 time-scale diagnostic are separated from the main runs by run prefix, windows and samples."""
    smoke, diag = B['smoke'], B['timescale_diagnostic']
    if purpose not in PURPOSES:
        raise ValueError('unregistered purpose')
    if not run.startswith('reworld-' + split + '-'):
        raise ValueError('registered run/split required')
    special = run.startswith((smoke['run_prefix'], diag['run_prefix']))
    if purpose == 'main':
        if special:
            raise ValueError('main runs may not use the smoke or diagnostic run prefixes')
        return
    if split != 'dev':
        raise ValueError('smoke and diagnostic runs are DEV only')
    rule = smoke if purpose == 'smoke' else diag
    if not run.startswith(rule['run_prefix']) or window not in rule['windows']:
        raise ValueError(purpose + ': run prefix or window outside the registered set')
    if mode == 'continue' and sample not in rule['samples']:
        raise ValueError(purpose + ': sample outside the registered seeds')
    if mode != 'continue' and not rule['references']:
        raise ValueError(purpose + ': continuations only')


def settings():
    """Registered generation settings; specs, runner and checks must agree on this exact dict."""
    lat = B['latent']
    ig = B['internal_guidance']
    if W % lat['spatial_compression'] or H % lat['spatial_compression'] or W != 2 * H:
        raise ValueError('model frame must be 2:1 and divisible by the VAE spatial compression')
    return dict(size_wh=[W, H], model_source_rows=list(SRC_ROWS), canvas_valid_rows=list(CANVAS_ROWS),
                native_fps=B['native_fps'], canvas_fps=B['canvas_fps'], condition_frames=B['condition_frames'],
                num_frames=B['window_frames'], passes=B['passes'], pass2_seed_offset=B['pass2_seed_offset'],
                latent_grid=[lat['latent_frames'], H // lat['spatial_compression'], W // lat['spatial_compression']],
                c_vae=lat['c_vae'], num_inference_steps=B['num_inference_steps'], dtype=B['dtype'],
                null_text_seq_len=B['null_text_seq_len'], ig_block=ig['sampling_block'], ig_scale=ig['scale'],
                ig_min_timestep=ig['min_timestep'], rope_fps=B['rope_fps'], vae_tiling=B['vae_tiling'],
                prefix=B['prefix'], uint8_rule='truncate')


def require_approved():
    """GPU paths only: every decision ruled (decisions_pending empty) and bench.coordinator_approved set at commit."""
    if B.get('coordinator_approved') is not True or PINS.get('decisions_pending'):
        raise ValueError('coordinator approval pending (reworld_pins.json bench.coordinator_approved is not true or '
                         'decisions_pending is not empty); no GPU started')


def aux_seed(window, label):
    """The shared seed formula applied to a non-integer label (prefix, recon, condition-encode draws)."""
    return int.from_bytes(hashlib.sha256(f'bench-v1|{window}|{label}'.encode()).digest()[:4], 'big') & 0x7fffffff


def condition_seeds(window, sample):
    return [aux_seed(window, f'{sample}|cond{p}') for p in (1, 2)]


def shared_seed(window, sample, snapshot):
    root = Path(snapshot)
    path = root / PINS['seed_snapshot']['file']
    expected = PINS['seed_snapshot']['sha256']
    listed = {name: digest for digest, name in [line.split('  ', 1) for line in (root / 'MANIFEST.sha256').read_text().splitlines() if '  ' in line]}
    if sha(path) != expected or listed.get(PINS['seed_snapshot']['file']) != expected:
        raise ValueError('shared seed source hash mismatch')
    spec = importlib.util.spec_from_file_location('reworld_seed_snapshot', path)
    module = importlib.util.module_from_spec(spec)
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = old
    value = module.seed_of(window, sample)
    if value != aux_seed(window, sample):
        raise ValueError('shared seed rule drift')
    return value


def verify_source(root):
    """Every registered official file has its pinned git blob; no unregistered Python under src/dualflow."""
    root = Path(root).resolve()
    registered = dict(PINS['code']['files'])
    for name, digest in registered.items():
        p = safe_relative(root, name)
        if not p.is_file() or blob(p) != digest:
            raise ValueError('official ReWorld source changed: ' + name)
    package = root / 'src' / 'dualflow'
    expected = {n for n in registered if n.startswith('src/dualflow/') and n.endswith('.py')}
    actual = {str(x.relative_to(root)) for x in package.rglob('*.py') if '__pycache__' not in x.parts}
    if actual != expected:
        raise ValueError('unregistered Python source in src/dualflow')
    return dict(commit=PINS['code']['commit'], checked_files=len(registered))


def weight_path(root, suffix):
    rows = [r for r in PINS['weights']['files'] if r['path'].endswith(suffix)]
    if len(rows) != 1:
        raise ValueError('ambiguous weight file ' + suffix)
    return safe_relative(root, rows[0]['path'])


def dit_checkpoint(root):
    return weight_path(root, 'ReWorld-Video/dualflow_dit_best_fvd.safetensors')


def ltx_dir(root):
    return safe_relative(root, PINS['weights']['repos']['ltx095']['dest'])


def verify_weights(root, full=True):
    result = []
    for row in PINS['weights']['files']:
        p = safe_relative(root, row['path'])
        if not p.is_file() or p.stat().st_size != row['bytes']:
            raise ValueError('weight/config absent or wrong size: ' + row['path'])
        if full and sha(p) != row['sha256']:
            raise ValueError('weight SHA mismatch: ' + row['path'])
        result.append(row['path'])
    return dict(revisions={k: v['revision'] for k, v in PINS['weights']['repos'].items()}, files=len(result), full_hashes=full)


def decode(path, wh):
    import numpy as np
    data = subprocess.check_output(['ffmpeg', '-nostdin', '-v', 'error', '-xerror', '-threads', '2', '-i', str(path),
                                    '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'])
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


def render_model_frame(rgb, size_wh=None):
    """Raw 1550x2048 RGB -> full-width rows SRC_ROWS -> cv2.INTER_AREA model frame (2:1, aspect kept)."""
    import cv2
    w, h = size_wh or (W, H)
    if list(rgb.shape[:2]) != [SRC_H, SRC_W]:
        raise ValueError('raw AV2 camera shape changed')
    crop = rgb[SRC_ROWS[0]:SRC_ROWS[1]]
    if crop.shape[1] != 2 * crop.shape[0]:
        raise ValueError('model source crop must be exactly 2:1')
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_AREA)


def to_canvas(frame):
    """One model frame -> the 512x288 canvas: INTER_AREA to 512x(rows) at canvas rows CANVAS_ROWS, the rest black."""
    import cv2
    import numpy as np
    r0, r1 = CANVAS_ROWS
    out = np.zeros((CANVAS_H, CANVAS_W, 3), np.uint8)
    out[r0:r1] = cv2.resize(frame, (CANVAS_W, r1 - r0), interpolation=cv2.INTER_AREA)
    return out


def canvas(native):
    import numpy as np
    if len(native) != B['full_native_frames']:
        raise ValueError('registered native clip must contain 65 frames')
    return np.stack([to_canvas(native[native_index(k)]) for k in range(-20, 61)])
