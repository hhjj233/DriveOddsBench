#!/usr/bin/env python3
"""Vista adapter for the wm-calib-bench calibration benchmark (bench-v1), tracks T1-H0 and T2.

Reads spec JSONs (one per sample; see make_vista_specs.py) and writes, per sample,
  <output-root>/<spec.output_dir>/vision.mp4           81 frames, 512 x 288, 10 fps, lossless (libx264rgb -crf 0, rgb24)
  <output-root>/<spec.output_dir>/sample_outputs.json  status (success / failure), every setting, timings, memory, hashes
  optional latents.safetensors (the rollout latents) and native<tag>.mp4 (the 81 frames at the generation size)

Generation size (pins bench.sizes; named in every spec): `320x576` (the default), i.e.
320 x 576 (H x W), both multiples of 64, so the released UNet runs unmodified; below Vista's 576 x 1024 training
size. The inputs are the bench band rendered by the bench renderer directly to 576 x 320 (history21_576.mp4,
real81_576.mp4; aspect 1.8 against the band's 1.7775, a vertical squeeze of 1.3 %), and every decoded frame (prefix,
reference, continuation) is resized to the 512 x 288 canvas by inter_area(), OpenCV's INTER_AREA bit for bit.
Superseded sizes (refused without --allow-superseded-size): `canvas` 288 x 512 (latent 36 x 64: it needed the one-layer
upsampling adaptation of configure_size(), a disabled option now) and `native` 576 x 1024 (the exact 2x area mean).

Modes (spec "mode"; <tag> = 576 at the default size; 512 / 1024 at the superseded sizes):
  prefix    The window's real history frames k = -20..-1 (history21_<tag>.mp4 frames 0-19) through Vista's VAE, encode then
            decode exactly as a rollout's latents are (encode_first_stage, decode_first_stage), seed_of(w, 'prefix') for the
            posterior sample. prefix.mp4 (20 canvas frames; also prefix<tag>.mp4 at the generation size) becomes frames 0-19
            of every clip of the window.
  recon     The real reference: the real frames k = 0..60 (real81_<tag>.mp4 frames 20-80) through the same encode and the
            same 61-latent decode as a continuation, seed_of(w, 'recon'); vision.mp4 = prefix + these 61 frames.
  continue  Track h0: Vista's released sampler from the anchor frame only (k = 0, history21_<tag>.mp4 frame 20), action-free.
            Track t2: the same plus Vista's trajectory action, one per rollout round (actions_t2.json). 3 rounds of 25
            frames (22 new frames per round after the first), i.e. latents 0-68 = t0 .. t0 + 6.8 s; latents 0-60 are decoded;
            vision.mp4 = prefix + these 61 frames. The real future is never read (for t2 only the ego trajectory).

The sampling call is sample.py's main loop body and sample_utils.do_sample, run as released (VERSION2SPECS 'vwm',
configs/inference/vista.yaml, vista.safetensors, --n_rounds 3, IMG input mode, n_conds 1, 50 steps, CFG 2.5,
TrianglePredictionGuider for the multi-round rollout, cond_aug 0, seed_everything(seed) once per sample), in the memory
layout the specs name (pins bench.layouts: official = without --low_vram, the released setting; local48 = --low_vram plus
the cond-frame encoder one copy at a time, for 48 GB mechanics at the native size only), one sample per call. Differences:
  - do_sample_rounds() is do_sample line for line except (a) before rounds 2 and 3 it sets the trajectory action to that
    round's (do_sample keeps round 1's action for every round; the action-free h0 path is unchanged), (b) it returns the
    latents and leaves the final decode to decode_latents(), which decodes latents 0-60 (t0 .. t0 + 6.0 s) with the
    official decode_first_stage instead of all 69. --equivalence-test checks on the GPU that do_sample_rounds reproduces
    do_sample bit for bit (h0, and t2 with one action for all rounds);
  - the generation size (320 x 576, below the training size; the UNet unmodified);
  - the input frames are the bench renderer's INTER_AREA render of the measurement band at the generation size (no centre
    crop / LANCZOS; at 576 x 320 a vertical squeeze of 1.3 %);
  - FrozenOpenCLIPImageEmbedder is built with pretrained=None (no laion download); vista.safetensors holds all its tensors
    (the adapter requires zero missing and zero unexpected keys);
  - frames are written losslessly (official uint8 truncation, then INTER_AREA to the 512 x 288 canvas).

Resumable: a sample whose sample_outputs.json says success, for the same spec, and whose vision.mp4 matches the recorded
sha256 is skipped. A failed sample is recorded with status "failure" and the run continues; exit code 1 if any failed.

  run_vista_bench.py --specs DIR_OR_JSON [...] --input-root DIR --output-root DIR --vista-src DIR --ckpt FILE
                     [--resume] [--save-native] [--save-latents] [--verify-weights none|size|sha256]
                     [--expect-gpu-uuid UUID] [--min-free-mib N] [--leave-free-mib 6144] [--max-gpu-mib N]
                     [--min-host-avail-gib N] [--stop-file FILE] [--dry-run]
  run_vista_bench.py --self-test [--specs ...] [--input-root DIR] [--vista-src DIR --ckpt FILE --with-model]
  run_vista_bench.py --equivalence-test --input-root DIR --window W --vista-src DIR --ckpt FILE --out FILE.json [--steps 2]
                     [--size 320x576] [--layout official|local48]
Run under CUDA_VISIBLE_DEVICES set to the one allowed GPU (Vista's code uses the device "cuda").
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
PINS_PATH = HERE / 'vista_pins.json'
PINS = json.loads(PINS_PATH.read_text())
D, B = PINS['released_defaults'], PINS['bench']
CW, CH = B['canvas_wh']                      # 512, 288
SIZES = B['sizes']                           # 320x576 (default); canvas 288 x 512 and native 576 x 1024 (superseded)
FPS = int(B['fps'])
N_FRAMES = int(D['n_frames'])                # 25 frames per round
N_ROUNDS = int(B['n_rounds'])                # 3
N_LAT = N_ROUNDS * (N_FRAMES - 3) + 3        # 69
N_DEC = int(B['decoded_latents'])            # 61: t0 .. t0 + 6.0 s
N_PREFIX, N_CLIP = 20, int(B['frames'])      # frames 0-19 prefix, 81 frames per clip
UC_KEYS = list(D['uc_keys'])
SPEC_SCHEMA = 'wm-calib-bench.vista.spec.v1'
OUT_SCHEMA = 'wm-calib-bench.vista.sample_outputs.v1'
ACTION_SCHEMA = 'wm-calib-bench.vista.actions_t2.v1'
assert N_LAT == int(B['rollout_latents'])


# ---------------------------------------------------------------------------------------------------- helpers (no torch)
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def write_json_atomic(path: Path, obj) -> None:
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=1) + '\n')
    os.replace(tmp, path)


def source_digest(root: Path) -> str:
    """The pins' rule: sha256 over the sorted *.py / *.yml / *.yaml files of the checkout."""
    files = sorted(str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p.suffix in ('.py', '.yml', '.yaml')
                   and '.git' not in p.parts and '__pycache__' not in p.parts and not p.name.startswith('.'))
    h = hashlib.sha256()
    for f in files:
        h.update(f.encode() + b'\0' + sha256_file(root / f).encode() + b'\n')
    return h.hexdigest()


def ffprobe(path: Path) -> dict:
    out = subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames', '-show_entries',
                                   'stream=width,height,nb_read_frames,r_frame_rate,pix_fmt,codec_name', '-of', 'json', str(path)], text=True)
    s = json.loads(out)['streams'][0]
    num, den = s['r_frame_rate'].split('/')
    return dict(width=int(s['width']), height=int(s['height']), frames=int(s['nb_read_frames']), fps=float(num) / float(den),
                pix_fmt=s.get('pix_fmt'), codec=s.get('codec_name'))


def decode_rgb(path: Path, threads: int = 2):
    import numpy as np
    info = ffprobe(path)
    raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-threads', str(threads), '-i', str(path), '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'])
    frames = np.frombuffer(raw, np.uint8).reshape(-1, info['height'], info['width'], 3).copy()
    if frames.shape[0] != info['frames']:
        raise RuntimeError(f'{path}: decoded {frames.shape[0]} frames, ffprobe counted {info["frames"]}')
    return frames, info


def write_lossless(frames, path: Path, threads: int = 2) -> None:
    """make_wm_calibration_windows.write_video's encoder (libx264rgb -crf 0 -preset medium, rgb24, 10 fps), plus -threads;
    the decoded frames are checked equal to the written ones."""
    import numpy as np
    frames = np.ascontiguousarray(frames)
    h, w = frames.shape[1:3]
    tmp = path.with_name(path.stem + '.part.mp4')
    proc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}', '-r', str(FPS), '-i', '-',
                             '-c:v', 'libx264rgb', '-crf', '0', '-preset', 'medium', '-threads', str(threads), '-pix_fmt', 'rgb24', '-r', str(FPS),
                             '-video_track_timescale', '10240', '-map_metadata', '-1', '-movflags', '+faststart', str(tmp)], stdin=subprocess.PIPE)
    for f in frames:
        proc.stdin.write(f.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f'ffmpeg failed for {path}')
    back, _ = decode_rgb(tmp, threads)
    if not np.array_equal(back, frames):
        raise RuntimeError(f'{path}: decoded frames differ from the written ones')
    os.replace(tmp, path)


def ffmpeg_version() -> str:
    try:
        return subprocess.check_output(['ffmpeg', '-version'], text=True).splitlines()[0]
    except Exception as exc:  # noqa: BLE001
        return f'unavailable: {exc}'


def area2(x):
    """cv2.INTER_AREA at exactly 2x down, uint8: (a + b + c + d + 2) >> 2 per channel (equal to cv2 5.0.0; the input builder
    records the comparison). x: (..., H, W, C)."""
    import numpy as np
    *lead, h, w, c = x.shape
    s = x.astype(np.uint16).reshape(*lead, h // 2, 2, w // 2, 2, c).sum(axis=(-4, -2))
    return ((s + 2) >> 2).astype(np.uint8)


def _area_tables(ssize: int, dsize: int):
    """OpenCV's computeResizeAreaTab (imgproc/src/resize.cpp) for a down-scale ssize -> dsize with OpenCV's scale
    1 / (dsize / ssize): per output index its (source index, float32 weight) pairs in OpenCV's order, as zero-padded arrays
    (n, dsize) of indices and weights (a zero weight adds an exact 0.0)."""
    import numpy as np
    scale = 1.0 / (dsize / ssize)
    rows = []
    for dx in range(dsize):
        fsx1 = dx * scale
        fsx2 = fsx1 + scale
        cell = min(scale, ssize - fsx1)
        sx1, sx2 = math.ceil(fsx1), math.floor(fsx2)
        sx2 = min(sx2, ssize - 1)
        sx1 = min(sx1, sx2)
        row = []
        if sx1 - fsx1 > 1e-3:
            row.append((sx1 - 1, np.float32((sx1 - fsx1) / cell)))
        row += [(sx, np.float32(1.0 / cell)) for sx in range(sx1, sx2)]
        if fsx2 - sx2 > 1e-3:
            row.append((sx2, np.float32(min(min(fsx2 - sx2, 1.0), cell) / cell)))
        rows.append(row)
    n = max(len(r) for r in rows)
    idx, wt = np.zeros((n, dsize), np.int64), np.zeros((n, dsize), np.float32)
    for dx, r in enumerate(rows):
        for j, (si, a) in enumerate(r):
            idx[j, dx], wt[j, dx] = si, a
    return idx, wt


def inter_area(x, dw: int, dh: int):
    """cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_AREA) for uint8 (..., H, W, C) frames down-scaled by a
    non-integer factor: OpenCV's generic area resampling (ResizeArea_Invoker: per source row the float32 sum of
    pixel x weight over computeResizeAreaTab's x entries, then per output row the float32 sum of weight x row over its y
    entries, in OpenCV's order; saturate_cast = round half to even, clamp). Checked bit for bit against cv2 5.0.0 on every
    input frame by build_vista_inputs.py (the Vista environment has no OpenCV)."""
    import numpy as np
    *lead, h, w, c = x.shape
    if dw >= w or dh >= h or (w % dw == 0 and h % dh == 0):
        raise ValueError(f'inter_area: {w} x {h} -> {dw} x {dh} is not a non-integer down-scale (OpenCV takes another path)')
    xi, xw = _area_tables(w, dw)
    yi, yw = _area_tables(h, dh)
    src = np.ascontiguousarray(x).reshape(-1, h, w, c)
    out = np.empty((src.shape[0], dh, dw, c), np.uint8)
    for f in range(src.shape[0]):
        s = src[f].astype(np.float32)
        buf = s[:, xi[0], :] * xw[0][None, :, None]
        for j in range(1, xi.shape[0]):
            buf = buf + s[:, xi[j], :] * xw[j][None, :, None]
        acc = buf[yi[0]] * yw[0][:, None, None]
        for j in range(1, yi.shape[0]):
            acc = acc + buf[yi[j]] * yw[j][:, None, None]
        out[f] = np.clip(np.rint(acc), 0, 255).astype(np.uint8)
    return out.reshape(*lead, dh, dw, c)


def frames_sha256(arr) -> str:
    import numpy as np
    return sha256_bytes(np.ascontiguousarray(arr).tobytes())


def host_memory() -> dict:
    import resource
    out = dict(peak_rss_gib=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2))
    try:
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith(('VmRSS:', 'VmHWM:')):
                key, val = line.split(':')
                out[key.lower() + '_gib'] = round(int(val.split()[0]) / 2**20, 2)
        out['host_mem_available_gib'] = mem_available_gib()
    except OSError:
        pass
    return out


def mem_available_gib() -> float:
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return round(int(line.split()[1]) / 2**20, 2)
    return float('nan')


def load_specs(paths) -> list:
    files = []
    for p in map(Path, paths):
        files += sorted(p.rglob('*.json')) if p.is_dir() else [p]
    specs = []
    for f in files:
        s = json.loads(f.read_text())
        if s.get('schema') != SPEC_SCHEMA:
            continue  # manifests next to the specs
        s['_file'] = str(f)
        s['_sha256'] = sha256_bytes(f.read_bytes())
        specs.append(s)
    uniq = {}
    for s in specs:  # the same sample listed by two spec sets: allowed only if the spec files are byte-identical
        k = (s['run'], s['name'])
        if k in uniq and uniq[k]['_sha256'] != s['_sha256']:
            raise SystemExit(f'two different specs for {k}')
        uniq.setdefault(k, s)
    specs = list(uniq.values())
    order = {'prefix': 0, 'recon': 1, 'continue': 2}
    specs.sort(key=lambda s: (s['split'], s['window'], order[s['mode']], s.get('track') or '', s.get('k') or 0))
    return specs


def spec_layout(g: dict) -> str:
    """The memory layout (pins bench.layouts) of a generation block: named by 'layout' (newer specs) or inferred from
    low_vram and cond_encode_chunk (the first local DEV specs); the two must agree."""
    match = [name for name, lay in B['layouts'].items() if g.get('low_vram') == lay['low_vram'] and g.get('cond_encode_chunk') == lay['cond_encode_chunk']]
    if len(match) != 1 or (g.get('layout') is not None and g['layout'] != match[0]):
        raise ValueError(f'generation block names no pinned layout (layout {g.get("layout")}, low_vram {g.get("low_vram")}, '
                         f'cond_encode_chunk {g.get("cond_encode_chunk")})')
    return match[0]


def size_name(g: dict) -> str:
    """The generation size (pins bench.sizes) of a generation block: named by 'size' (newer specs) or inferred from height
    and width (the first local DEV specs, native 576 x 1024); the two must agree."""
    match = [n for n, sz in SIZES.items() if [g.get('height'), g.get('width')] == list(sz['hw'])]
    if len(match) != 1 or (g.get('size') is not None and g['size'] != match[0]):
        raise ValueError(f'generation block names no pinned size (size {g.get("size")}, {g.get("height")} x {g.get("width")})')
    return match[0]


class Size:
    """One pinned generation size: model frame h x w, the input files' tag and the mapping to the 512 x 288 canvas
    (identity at the canvas, the exact 2x area mean at twice it, else inter_area)."""

    def __init__(self, name: str):
        sz = SIZES[name]
        self.name, (self.h, self.w), self.tag = name, tuple(sz['hw']), str(sz['input_tag'])
        self.superseded = str(sz.get('status', '')).startswith('superseded')
        self.adaptation = sz.get('unet_adaptation')
        self.mapping = 'identity' if (self.w, self.h) == (CW, CH) else 'area2' if (self.w, self.h) == (2 * CW, 2 * CH) else 'inter_area'
        self.native = self.mapping != 'identity'      # the decode is not the canvas itself: keep the generation-size files
        if self.h % 8 or self.w % 8 or self.w < CW or self.h < CH:
            raise ValueError(f'size {name}: not a multiple of the VAE factor 8, or below the canvas')
        self.history, self.real = f'history21_{self.tag}.mp4', f'real81_{self.tag}.mp4'
        self.native_clip, self.native_prefix = f'native{self.tag}.mp4', f'prefix{self.tag}.mp4'

    def to_canvas(self, frames_u8):
        if self.mapping == 'identity':
            return frames_u8
        return area2(frames_u8) if self.mapping == 'area2' else inter_area(frames_u8, CW, CH)

    def record(self) -> dict:
        return dict(name=self.name, hw=[self.h, self.w], latent_hw=[self.h // 8, self.w // 8], canvas_mapping=SIZES[self.name]['canvas_mapping'],
                    superseded=self.superseded)


def unet_size_plan(lat_h: int, lat_w: int, n_down: int = 3) -> dict:
    """The VideoUNet's skip sizes at a latent size: each stride-2 downsampling (3 x 3, padding 1) gives ceil(s / 2), each
    upsampling doubles. Returns {upsampling input (h, w): skip size (h, w)} for the upsamplings whose double is not the skip
    size (empty for latents that are multiples of 2 ** n_down, e.g. 72 x 128 or 40 x 72; {(5, 8): (9, 16)} at 36 x 64)."""
    sizes = [(int(lat_h), int(lat_w))]
    for _ in range(n_down):
        sizes.append(((sizes[-1][0] + 1) // 2, (sizes[-1][1] + 1) // 2))
    return {sizes[i + 1]: sizes[i] for i in range(n_down) if sizes[i] != (2 * sizes[i + 1][0], 2 * sizes[i + 1][1])}


def check_spec(spec: dict, input_root: Path) -> None:
    for key in ('name', 'run', 'window', 'split', 'mode', 'seed', 'inputs', 'output_dir', 'generation'):
        if key not in spec:
            raise ValueError(f'spec {spec.get("name")}: missing {key}')
    want = {'prefix': {'history'}, 'recon': {'real'}}.get(spec['mode'])
    if spec['mode'] == 'continue':
        want = {'anchor'} | ({'actions'} if spec['track'] == 't2' else set())
        if spec['track'] not in ('h0', 't2'):
            raise ValueError(f'{spec["name"]}: track {spec.get("track")}')
        if spec['inputs']['anchor'].get('use_frames') != [20]:
            raise ValueError(f'{spec["name"]}: a continuation sees the anchor frame (history frame 20) only')
    if want is None or set(spec['inputs']) != want:
        raise ValueError(f'{spec["name"]}: mode {spec["mode"]} needs inputs {sorted(want or [])}, got {sorted(spec["inputs"])}')
    if spec['mode'] != 'prefix' and 'prefix' not in spec:
        raise ValueError(f'{spec["name"]}: no prefix reference')
    for rec in spec['inputs'].values():
        p = Path(rec['path'])
        if p.is_absolute() or '..' in p.parts:
            raise ValueError(f'{spec["name"]}: input paths must be relative to the input root')
        if not (input_root / p).is_file():
            raise FileNotFoundError(f'{spec["name"]}: {input_root / p} missing')
    g = spec['generation']
    size_name(g)
    for key, want_v in (('n_frames', D['n_frames']), ('n_conds', D['n_conds']),
                        ('cfg_scale', D['cfg_scale']), ('cond_aug', D['cond_aug']), ('n_rounds', B['n_rounds']),
                        ('decoded_latents', B['decoded_latents']), ('vista_commit', PINS['code']['commit']),
                        ('hf_revision', PINS['weights']['hf_revision'])):
        if g.get(key) != want_v:
            raise ValueError(f'{spec["name"]}: generation.{key} = {g.get(key)}, the pinned value is {want_v}')
    if g.get('n_steps') != D['n_steps'] and not g.get('mechanics_only'):
        raise ValueError(f'{spec["name"]}: n_steps {g.get("n_steps")} without the mechanics_only mark')
    spec_layout(g)
    av = g.get('action_variant', 'frame_index')
    if av not in WAYPOINT_STEP or (av != 'frame_index' and not (g.get('diagnostic_only') and spec['split'] == 'dev')):
        raise ValueError(f'{spec["name"]}: action variant {av} (only frame_index, or time_aligned as a DEV diagnostic)')


def is_done(spec: dict, output_root: Path) -> bool:
    d = output_root / spec['output_dir']
    f = d / 'sample_outputs.json'
    v = d / ('prefix.mp4' if spec['mode'] == 'prefix' else 'vision.mp4')
    if not (f.is_file() and v.is_file()):
        return False
    try:
        rec = json.loads(f.read_text())
    except json.JSONDecodeError:
        return False
    return rec.get('status') == 'success' and rec.get('spec_sha256') == spec['_sha256'] and rec.get('video_sha256') == sha256_file(v)


WAYPOINT_STEP = {'frame_index': 6, 'time_aligned': 5}   # the pinned rule (frames 22n + 6i) and the DEV-only diagnostic (22n + 5i)


def load_actions(path: Path, window: str, variant: str = 'frame_index'):
    """Vista's trajectory action per round (3 x 8 floats: (x right, y forward) m at 4 waypoints) from actions_t2.json."""
    rec = json.loads(Path(path).read_text())
    if rec.get('schema') != ACTION_SCHEMA or rec.get('window') != window or rec.get('mode') != 'trajectory' or rec.get('variant', 'frame_index') != variant:
        raise ValueError(f'{path}: not the {variant} trajectory actions of {window}')
    rounds = sorted(rec['rounds'], key=lambda r: r['round'])
    if [r['round'] for r in rounds] != list(range(N_ROUNDS)):
        raise ValueError(f'{path}: rounds {[r["round"] for r in rounds]}')
    st = WAYPOINT_STEP[variant]
    for r in rounds:
        if len(r['trajectory']) != 8 or r['first_k'] != 22 * r['round'] or r['waypoint_ks'] != [22 * r['round'] + st * i for i in range(1, 5)]:
            raise ValueError(f'{path}: round {r["round"]} does not follow the {variant} trajectory rule')
    return [list(map(float, r['trajectory'])) for r in rounds], rec


# ---------------------------------------------------------------------------------------------------------- the model
class VistaModel:
    """Vista as sample.py builds it (VERSION2SPECS 'vwm'), with or without --low_vram."""

    def __init__(self, src: Path, ckpt: Path, verify_weights: str, low_vram: bool, allow_unpinned: bool = False, device_check: bool = True,
                 cond_encode_chunk: int | None = None):
        self.src = Path(src).resolve()
        self.digest = source_digest(self.src)
        if self.digest != PINS['code']['source_digest'] and not allow_unpinned:
            raise SystemExit(f'{self.src}: source digest {self.digest}, pinned {PINS["code"]["source_digest"]}')
        self.weights = self._check_weights(Path(ckpt), verify_weights)
        sys.dont_write_bytecode = True                 # keep the checkout clean (the digest ignores __pycache__ anyway)
        sys.path.insert(0, str(self.src))              # sample_utils.py and train.py live at the root of the checkout
        t = time.time()
        import numpy as np
        import torch
        from omegaconf import OmegaConf
        from safetensors.torch import load_file as load_safetensors
        from pytorch_lightning import seed_everything
        import sample_utils as su
        self.np, self.torch, self.su, self.seed_everything = np, torch, su, seed_everything
        su.set_lowvram_mode(low_vram)
        self.low_vram = low_vram
        config = OmegaConf.load(self.src / 'configs' / 'inference' / 'vista.yaml')
        oc = config.model.params.conditioner_config.params.emb_models[0].params.open_clip_embedding_config
        if oc.target != 'vwm.modules.encoders.modules.FrozenOpenCLIPImageEmbedder':
            raise SystemExit('unexpected conditioner layout in configs/inference/vista.yaml')
        oc.params['version'] = None                    # pretrained=None: no laion download; overwritten below (checked)
        mem_before = mem_available_gib()
        # sample_utils.load_model_from_config, step by step, to check the keys
        model = su.instantiate_from_config(config.model)
        sd = load_safetensors(str(ckpt))
        missing, unexpected = model.load_state_dict(sd, strict=False)
        n_tensors = len(sd)
        del sd
        if missing or unexpected:
            raise SystemExit(f'vista.safetensors does not match the model: {len(missing)} missing, {len(unexpected)} unexpected keys '
                             f'(e.g. {list(missing)[:3]} {list(unexpected)[:3]})')
        model = su.initial_model_load(model)
        model.eval()
        enc = [e for e in model.conditioner.embedders if type(e).__name__ == 'VideoPredictionEmbedderWithEncoder']
        if len(enc) != 1 or enc[0].en_and_decode_n_samples_a_time is not None:
            raise SystemExit('unexpected cond-frame encoder configuration')
        if cond_encode_chunk is not None:   # optional memory saver (pinned per run): the class's own chunking parameter
            enc[0].en_and_decode_n_samples_a_time = int(cond_encode_chunk)
        self.cond_encode_chunk = cond_encode_chunk
        self.model = model
        self.size_adaptation = dict(latent_hw=None, upsample_to_skip={})
        self.unique_keys = set(x.input_key for x in model.conditioner.embedders)
        self.load_record = dict(seconds=round(time.time() - t, 1), tensors=n_tensors, missing_keys=0, unexpected_keys=0, host_mem_available_before_gib=mem_before,
                                host_after=host_memory(), unet_dtype=str(next(model.model.parameters()).dtype),
                                conditioner_dtype=str(next(model.conditioner.parameters()).dtype),
                                first_stage_dtype=str(next(model.first_stage_model.parameters()).dtype),
                                open_clip_init='pretrained=None, all tensors from vista.safetensors', low_vram=low_vram,
                                cond_encode_chunk=cond_encode_chunk, cudnn_allow_tf32=bool(torch.backends.cudnn.allow_tf32),
                                matmul_allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
                                scale_factor=float(model.scale_factor), en_and_decode_n_samples_a_time=int(model.en_and_decode_n_samples_a_time),
                                disable_first_stage_autocast=bool(model.disable_first_stage_autocast), use_ema=bool(model.use_ema),
                                embedders=[f'{type(e).__name__}:{getattr(e, "input_key", None)}' for e in model.conditioner.embedders])

    def layout_record(self) -> dict:
        """The memory layout as built (not as requested): where the sub-models live, the UNet's weight dtype, the conditioner's
        chunking, and the UNet's size adaptation in force (configure_size)."""
        m = self.model
        dev = lambda mod: str(next(mod.parameters()).device)  # noqa: E731
        return dict(low_vram=self.low_vram, cond_encode_chunk=self.cond_encode_chunk, unet_dtype=str(next(m.model.parameters()).dtype),
                    unet_device_now=dev(m.model), conditioner_device_now=dev(m.conditioner), first_stage_device_now=dev(m.first_stage_model),
                    name=next((k for k, v in B['layouts'].items() if v['low_vram'] == self.low_vram and v['cond_encode_chunk'] == self.cond_encode_chunk), None),
                    unet_size_adaptation=dict(self.size_adaptation))

    def configure_size(self, size: 'Size') -> dict:
        """The VideoUNet at the latent size of `size`. For every pinned size whose pixels are multiples of 64 (the default
        320 x 576, and 576 x 1024) the plan is empty and nothing is touched: the released UNet runs. Only the superseded
        288 x 512 (pins unet_adaptation upsample_to_skip; refused without --allow-superseded-size) patches the one Upsample
        whose input has no exact double: nearest interpolation to its skip connection's size, then its own convolution
        (diffusers' rule). Any other size with a non-empty plan is refused. Recorded in layout_record()."""
        plan = unet_size_plan(size.h // 8, size.w // 8)
        if not plan:
            self.size_adaptation = dict(latent_hw=[size.h // 8, size.w // 8], upsample_to_skip={}, patched=False, rule='none (released UNet)')
            return self.size_adaptation
        if size.adaptation != 'upsample_to_skip':
            raise SystemExit(f'size {size.name}: the UNet would need an adaptation ({plan}); not enabled for this size')
        import vwm.modules.diffusionmodules.openaimodel as oam
        unet = self.model.model.diffusion_model
        ups = [mod for mod in unet.modules() if isinstance(mod, oam.Upsample)]
        downs = [mod for mod in unet.modules() if isinstance(mod, oam.Downsample)]
        if len(ups) != 3 or len(downs) != 3 or any(mod.dims != 2 for mod in ups + downs):
            raise SystemExit(f'unexpected VideoUNet resampling layers ({len(downs)} down, {len(ups)} up)')
        plan = unet_size_plan(size.h // 8, size.w // 8, len(downs))
        for mod in ups:
            patch_upsample(mod, plan)
        self.size_adaptation = dict(latent_hw=[size.h // 8, size.w // 8],
                                    upsample_to_skip={f'{a[0]}x{a[1]}': f'{b[0]}x{b[1]}' for a, b in sorted(plan.items())}, patched=True,
                                    rule='superseded option: nearest interpolation to the skip size, then the layer\'s own 3 x 3 convolution')
        return self.size_adaptation

    def _check_weights(self, ckpt: Path, mode: str) -> dict:
        want = PINS['weights']['files']['vista.safetensors']
        size = ckpt.stat().st_size
        if size != want['bytes']:
            raise SystemExit(f'{ckpt}: {size} bytes, pinned {want["bytes"]}')
        rec = dict(file=ckpt.name, bytes=size, check=mode, sha256_pinned=want['sha256'])
        if mode == 'sha256':
            got = sha256_file(ckpt)
            if got != want['sha256']:
                raise SystemExit(f'{ckpt}: sha256 {got}, pinned {want["sha256"]}')
            rec['sha256'] = got
        return rec

    # ------------------------------------------------------------------------------------------------------------- io
    def to_input(self, frames_u8):
        """sample.load_img's tensor transform (ToTensor, x * 2 - 1) on frames that already have the model size."""
        torch = self.torch
        from torchvision import transforms
        tf = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x * 2.0 - 1.0)])
        return [tf(f).to('cuda') for f in frames_u8]

    def to_uint8(self, samples_x):
        """do_sample's clamp((x + 1) / 2, 0, 1), then perform_save_locally's (255.0 * x).astype(np.uint8)."""
        from einops import rearrange
        samples = self.torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)
        img_seq = rearrange(samples.cpu().numpy(), 't c h w -> t h w c')
        return (255.0 * img_seq).astype(self.np.uint8)

    # ---------------------------------------------------------------------------------------------- the rollout
    def do_sample_rounds(self, images, sampler, value_dict, num_rounds, num_frames, force_uc_zero_embeddings=None, initial_cond_indices=None,
                         round_actions=None, device='cuda', timer=None):
        """sample_utils.do_sample, line for line, except: before rounds 2.. the trajectory action becomes round_actions[n + 1]
        (with round_actions None nothing changes), and it returns (samples_z, z) without the final decode."""
        torch, su, model = self.torch, self.su, self.model
        from torch import autocast
        timer = timer if timer is not None else {}
        if initial_cond_indices is None:
            initial_cond_indices = [0]
        force_uc_zero_embeddings = su.default(force_uc_zero_embeddings, list())
        precision_scope = autocast
        with torch.no_grad(), precision_scope(device), model.ema_scope("Sampling"):
            t = time.time()
            c, uc = su.get_condition(model, value_dict, num_frames, force_uc_zero_embeddings, device)
            su.load_model(model.first_stage_model)
            z = model.encode_first_stage(images)
            su.unload_model(model.first_stage_model)
            samples_z = torch.zeros((num_rounds * (num_frames - 3) + 3, *z.shape[1:])).to(device)

            def denoiser(x, sigma, cond, cond_mask):
                return model.denoiser(model.model, x, sigma, cond, cond_mask)

            su.load_model(model.denoiser)
            su.load_model(model.model)
            initial_cond_mask = torch.zeros(num_frames).to(device)
            prediction_cond_mask = torch.zeros(num_frames).to(device)
            initial_cond_mask[initial_cond_indices] = 1
            prediction_cond_mask[[0, 1, 2]] = 1
            timer['condition_encode_s'] = round(time.time() - t, 2)
            t = time.time()
            noise = torch.randn_like(z)
            sample = sampler(denoiser, noise, cond=c, uc=uc, cond_frame=z, cond_mask=initial_cond_mask)
            sample[0] = z[0]
            samples_z[:num_frames] = sample
            torch.cuda.synchronize()
            timer['round_s'] = [round(time.time() - t, 2)]
            for n in range(num_rounds - 1):
                t = time.time()
                su.load_model(model.first_stage_model)
                samples_x_for_guidance = model.decode_first_stage(sample[-14:])
                su.unload_model(model.first_stage_model)
                value_dict["cond_frames_without_noise"] = samples_x_for_guidance[[-3]]
                value_dict["cond_frames"] = sample[[-3]] / model.scale_factor
                if round_actions is not None:              # the one change: this round's own action
                    value_dict["trajectory"] = round_actions[n + 1]
                for embedder in model.conditioner.embedders:
                    if hasattr(embedder, "skip_encode"):
                        embedder.skip_encode = True
                c, uc = su.get_condition(model, value_dict, num_frames, force_uc_zero_embeddings, device)
                for embedder in model.conditioner.embedders:
                    if hasattr(embedder, "skip_encode"):
                        embedder.skip_encode = False
                filled_latent = su.fill_latent(sample[-3:], num_frames, [0, 1, 2], device)
                noise = torch.randn_like(filled_latent)
                sample = sampler(denoiser, noise, cond=c, uc=uc, cond_frame=filled_latent, cond_mask=prediction_cond_mask)
                samples_z[(n + 1) * (num_frames - 3) + 3: (n + 1) * (num_frames - 3) + num_frames] = sample[3:]
                torch.cuda.synchronize()
                timer['round_s'].append(round(time.time() - t, 2))
            su.unload_model(model.model)
            su.unload_model(model.denoiser)
        return samples_z, z

    def decode_latents(self, z):
        """do_sample's final decode (decode_first_stage under the same scopes) on the given latents."""
        torch, su, model = self.torch, self.su, self.model
        from torch import autocast
        with torch.no_grad(), autocast('cuda'), model.ema_scope("Sampling"):
            su.load_model(model.first_stage_model)
            samples_x = model.decode_first_stage(z)
            su.unload_model(model.first_stage_model)
        return samples_x

    def prepare(self, anchor_u8, seed: int, n_steps: int, n_rounds: int, cfg_scale: float, cond_aug: float, first_action=None):
        """sample.py's main loop body up to do_sample (IMG mode: the anchor as all 25 input frames)."""
        torch, su = self.torch, self.su
        self.seed_everything(seed)
        img_seq = self.to_input([anchor_u8] * N_FRAMES)
        images = torch.stack(img_seq)
        value_dict = su.init_embedder_options(self.unique_keys)
        cond_img = img_seq[0][None]
        value_dict["cond_frames_without_noise"] = cond_img
        value_dict["cond_aug"] = cond_aug
        value_dict["cond_frames"] = cond_img + cond_aug * torch.randn_like(cond_img)
        if first_action is not None:
            value_dict["trajectory"] = first_action
        guider = "TrianglePredictionGuider" if n_rounds > 1 else "VanillaCFG"
        sampler = su.init_sampling(guider=guider, steps=n_steps, cfg_scale=cfg_scale, num_frames=N_FRAMES)
        return images, value_dict, sampler, guider

    def rollout(self, anchor_u8, seed: int, gen: dict, round_actions=None):
        torch = self.torch
        timer = {}
        acts = None if round_actions is None else [torch.tensor(a) for a in round_actions]   # as get_sample: torch.tensor(traj[2:])
        images, value_dict, sampler, guider = self.prepare(anchor_u8, seed, gen['n_steps'], gen['n_rounds'], gen['cfg_scale'], gen['cond_aug'],
                                                           None if acts is None else acts[0])
        samples_z, z = self.do_sample_rounds(images, sampler, value_dict, gen['n_rounds'], N_FRAMES, force_uc_zero_embeddings=UC_KEYS,
                                             initial_cond_indices=list(range(D['n_conds'])), round_actions=acts, timer=timer)
        settings = self.read_back(sampler, guider)
        return samples_z, z, timer, settings

    def roundtrip(self, frames_u8, seed: int):
        """Encode and decode real frames exactly as a rollout's latents are (seed_everything(seed) for the posterior sample)."""
        torch, su, model = self.torch, self.su, self.model
        from torch import autocast
        self.seed_everything(seed)
        x = torch.stack(self.to_input(list(frames_u8)))
        with torch.no_grad(), autocast('cuda'), model.ema_scope("Sampling"):
            su.load_model(model.first_stage_model)
            z = model.encode_first_stage(x)
            su.unload_model(model.first_stage_model)
        return z, self.decode_latents(z)

    def read_back(self, sampler, guider_name: str) -> dict:
        g = sampler.guider
        sig = sampler.discretization(sampler.num_steps, device='cpu')
        out = dict(sampler=type(sampler).__name__, num_steps=int(sampler.num_steps), s_churn=sampler.s_churn, s_tmin=sampler.s_tmin, s_tmax=sampler.s_tmax,
                   s_noise=sampler.s_noise, discretization=type(sampler.discretization).__name__, sigma_max=float(sampler.discretization.sigma_max),
                   sigma_min=float(sampler.discretization.sigma_min), rho=float(sampler.discretization.rho), sigmas_first_last=[float(sig[0]), float(sig[-2]), float(sig[-1])],
                   guider=type(g).__name__, guider_requested=guider_name)
        if hasattr(g, 'max_scale'):
            out.update(max_scale=float(g.max_scale), min_scale=float(g.min_scale), scale_per_frame=[round(float(v), 4) for v in g.scale.flatten().tolist()])
        else:
            out.update(scale=float(g.scale))
        return out


def patch_upsample(mod, plan: dict) -> None:
    """Instance patch of one openaimodel.Upsample (dims 2): an input whose (h, w) is in `plan` is interpolated (nearest) to
    plan[(h, w)] instead of 2x, then the layer's convolution runs as released; any other input takes the released forward."""
    import torch.nn.functional as F
    mod._wcb_plan = dict(plan)
    if getattr(mod, '_wcb_released_forward', None) is None:
        mod._wcb_released_forward = mod.forward

        def forward(x, _m=mod):
            target = _m._wcb_plan.get(tuple(int(v) for v in x.shape[-2:]))
            if target is None:
                return _m._wcb_released_forward(x)
            assert x.shape[1] == _m.channels and x.dim() == 4
            x = F.interpolate(x, size=target, mode='nearest')
            return _m.conv(x) if _m.use_conv else x
        mod.forward = forward


# ---------------------------------------------------------------------------------------------------------- recording
def software(model: VistaModel | None) -> dict:
    out = dict(python=platform.python_version(), ffmpeg=ffmpeg_version(), adapter_sha256=sha256_file(Path(__file__)), pins_sha256=sha256_file(PINS_PATH),
               vista_commit=PINS['code']['commit'], hf_revision=PINS['weights']['hf_revision'], host=socket.gethostname())
    try:
        import numpy, torch, torchvision, xformers, pytorch_lightning, kornia
        from importlib.metadata import version
        out.update(torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(), torchvision=torchvision.__version__,
                   xformers=xformers.__version__, pytorch_lightning=pytorch_lightning.__version__, kornia=kornia.__version__, numpy=numpy.__version__,
                   open_clip=version('open-clip-torch'), transformers=version('transformers'), cudnn_benchmark=bool(torch.backends.cudnn.benchmark))
    except Exception:  # noqa: BLE001
        pass
    if model is not None:
        out['vista_source_digest'] = model.digest
    try:
        import torch
        out.update(cudnn_allow_tf32=bool(torch.backends.cudnn.allow_tf32), matmul_allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32))
    except Exception:  # noqa: BLE001
        pass
    for key in ('CUDA_VISIBLE_DEVICES', 'PYTORCH_CUDA_ALLOC_CONF', 'OMP_NUM_THREADS'):
        if os.environ.get(key):
            out.setdefault('env', {})[key] = os.environ[key]
    return out


def gpu_info(torch) -> dict:
    if not torch.cuda.is_available():
        return dict(name=None)
    p = torch.cuda.get_device_properties(0)
    info = dict(name=p.name, capability=f'{p.major}.{p.minor}', total_gib=round(p.total_memory / 2**30, 2), torch=torch.__version__,
                cuda_runtime=torch.version.cuda, cudnn=torch.backends.cudnn.version())
    try:  # cuDriverGetVersion: the newest CUDA version the installed driver supports
        import ctypes
        v = ctypes.c_int()
        if ctypes.CDLL('libcuda.so.1').cuDriverGetVersion(ctypes.byref(v)) == 0:
            info['cuda_driver'] = f'{v.value // 1000}.{(v.value % 1000) // 10}'
    except Exception:  # noqa: BLE001
        pass
    try:
        out = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,driver_version', '--format=csv,noheader'], text=True).strip().splitlines()
        vis = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')[0].strip()
        for line in out:
            idx, uuid, name, driver = [x.strip() for x in line.split(',')]
            if vis in (idx, uuid) or (len(out) == 1 and vis in ('', idx)) or (vis.startswith('GPU-') and uuid.startswith(vis)):
                info.update(index=idx, uuid=uuid, nvidia_smi_name=name, driver=driver)
    except Exception:  # noqa: BLE001
        pass
    return info


def gpu_free_mib(torch) -> float:
    free, _ = torch.cuda.mem_get_info(0)
    return free / 2**20


class Mem:
    """Peak GPU memory of the caching allocator per phase (GiB)."""

    def __init__(self, torch):
        self.torch, self.rec = torch, {}

    def start(self):
        self.torch.cuda.synchronize()
        self.torch.cuda.reset_peak_memory_stats()

    def stop(self, name):
        t = self.torch
        t.cuda.synchronize()
        self.rec[name] = dict(peak_allocated_gib=round(t.cuda.max_memory_allocated() / 2**30, 2), peak_reserved_gib=round(t.cuda.max_memory_reserved() / 2**30, 2))


# ---------------------------------------------------------------------------------------------------------------- run
def ensure_prefix(vm: VistaModel, info: dict, history_path: Path, output_root: Path, spec: dict | None, args, size: Size) -> dict:
    """The window's prefix (frames 0-19 of every clip) at the generation size: load it if done, else compute and write it."""
    np = vm.np
    d = output_root / info['output_dir']
    rec_f = d / 'sample_outputs.json'
    if rec_f.is_file():
        rec = json.loads(rec_f.read_text())
        if rec.get('status') == 'success' and rec.get('seed') == info['seed'] and (d / 'prefix.mp4').is_file() and sha256_file(d / 'prefix.mp4') == rec['video_sha256']:
            made = (rec.get('size') or {}).get('name', 'native')     # the first local prefixes were native and name no size
            if made != size.name:
                raise RuntimeError(f'{d}: the prefix was made at the {made} size, this sample is at the {size.name} size')
            canvas, _ = decode_rgb(d / 'prefix.mp4')
            hr = decode_rgb(d / size.native_prefix)[0] if size.native else canvas
            if frames_sha256(canvas) != rec['frames_sha256']:
                raise RuntimeError(f'{d}: prefix frames differ from the recorded hash')
            return dict(canvas=canvas, hr=hr, size=size.name, record=dict(output_dir=info['output_dir'], video_sha256=rec['video_sha256'],
                                                                          frames_sha256=rec['frames_sha256'], computed_now=False))
    t = time.time()
    d.mkdir(parents=True, exist_ok=True)
    frames, vinfo = decode_rgb(history_path)
    if frames.shape != (21, size.h, size.w, 3):
        raise ValueError(f'{history_path}: {frames.shape}, expected 21 frames of {size.w} x {size.h}')
    vm.configure_size(size)
    mem = Mem(vm.torch)
    mem.start()
    z, x = vm.roundtrip(frames[:N_PREFIX], info['seed'])
    mem.stop('prefix_roundtrip')
    hr = vm.to_uint8(x)
    canvas = size.to_canvas(hr)
    write_lossless(canvas, d / 'prefix.mp4')
    if size.native:
        write_lossless(hr, d / size.native_prefix)
    rec = dict(schema=OUT_SCHEMA, status='success', mode='prefix', name=info['name'], run=info['run'], window=(spec or {}).get('window'), seed=info['seed'],
               spec_sha256=(spec or {}).get('_sha256'), size=size.record(),
               input=dict(path=str(history_path), sha256=sha256_file(history_path), frames_used=list(range(N_PREFIX))),
               video_sha256=sha256_file(d / 'prefix.mp4'), frames_sha256=frames_sha256(canvas),
               video1024_sha256=sha256_file(d / size.native_prefix) if size.native else None,
               latents_sha256=sha256_bytes(z.float().cpu().numpy().tobytes()), roundtrip_vs_input=dict(
                   mean_abs_at_generation_size=round(float(np.abs(hr.astype(np.int16) - frames[:N_PREFIX].astype(np.int16)).mean()), 3)),
               decode_windows='decode_first_stage on 20 latents: windows [0, 14), [11, 20), 3-latent overlap averaged',
               seconds=round(time.time() - t, 2), gpu_memory=mem.rec, host=host_memory(), gpu=gpu_info(vm.torch), layout_built=vm.layout_record(),
               finished=now_utc())
    write_json_atomic(rec_f, rec)
    return dict(canvas=canvas, hr=hr, size=size.name, record=dict(output_dir=info['output_dir'], video_sha256=rec['video_sha256'],
                                                                  frames_sha256=rec['frames_sha256'], computed_now=True))


def run_one(vm: VistaModel, spec: dict, args, prefix_cache: dict) -> dict:
    np, torch = vm.np, vm.torch
    d = args.output_root / spec['output_dir']
    d.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    size = Size(size_name(spec['generation']))
    rec = dict(schema=OUT_SCHEMA, name=spec['name'], run=spec['run'], window=spec['window'], split=spec['split'], mode=spec['mode'], track=spec.get('track'),
               k=spec.get('k'), seed=spec['seed'], spec_file=spec['_file'], spec_sha256=spec['_sha256'], generation=spec['generation'], size=size.record(),
               started=now_utc())
    gpu_free_start = gpu_free_mib(torch)
    if gpu_free_start < args.leave_free_mib + 1024:
        raise RuntimeError(f'GPU free {gpu_free_start:.0f} MiB at the start of the sample; refusing (leave-free {args.leave_free_mib})')
    if spec['mode'] == 'prefix':
        info = dict(name=spec['name'], run=spec['run'], output_dir=spec['output_dir'], seed=spec['seed'])
        pre = ensure_prefix(vm, info, args.input_root / spec['inputs']['history']['path'], args.output_root, spec, args, size)
        prefix_cache[(spec['window'], spec['output_dir'])] = pre
        return json.loads((d / 'sample_outputs.json').read_text())
    key = (spec['window'], spec['prefix']['output_dir'])
    if key not in prefix_cache:
        hist = spec['inputs']['anchor']['path'] if spec['mode'] == 'continue' else str(Path(spec['inputs']['real']['path']).with_name(size.history))
        prefix_cache[key] = ensure_prefix(vm, spec['prefix'], args.input_root / hist, args.output_root, None, args, size)
    pre = prefix_cache[key]
    if pre['size'] != size.name:
        raise RuntimeError(f'the prefix of {spec["window"]} is at the {pre["size"]} size, this sample at the {size.name} size')
    vm.configure_size(size)
    mem = Mem(torch)
    timer = {}
    if spec['mode'] == 'recon':
        p = args.input_root / spec['inputs']['real']['path']
        if sha256_file(p) != spec['inputs']['real']['sha256']:
            raise ValueError(f'{p}: sha256 differs from the spec')
        frames, _ = decode_rgb(p)
        if frames.shape != (81, size.h, size.w, 3):
            raise ValueError(f'{p}: {frames.shape}')
        t = time.time()
        mem.start()
        z, x = vm.roundtrip(frames[20:20 + N_DEC], spec['seed'])
        mem.stop('recon_roundtrip')
        timer['roundtrip_s'] = round(time.time() - t, 2)
        lat, extra = z, dict(latents=int(z.shape[0]))
        anchor_sha = sha256_bytes(frames[20].tobytes())
    else:
        p = args.input_root / spec['inputs']['anchor']['path']
        if sha256_file(p) != spec['inputs']['anchor']['sha256']:
            raise ValueError(f'{p}: sha256 differs from the spec')
        frames, _ = decode_rgb(p)
        if frames.shape != (21, size.h, size.w, 3):
            raise ValueError(f'{p}: {frames.shape}')
        anchor = frames[20]
        anchor_sha = sha256_bytes(anchor.tobytes())
        round_actions, act_rec = None, None
        if spec['track'] == 't2':
            ap_ = args.input_root / spec['inputs']['actions']['path']
            if sha256_file(ap_) != spec['inputs']['actions']['sha256']:
                raise ValueError(f'{ap_}: sha256 differs from the spec')
            round_actions, act_rec = load_actions(ap_, spec['window'], spec['generation'].get('action_variant', 'frame_index'))
        t = time.time()
        mem.start()
        samples_z, z0, rtimer, settings = vm.rollout(anchor, spec['seed'], spec['generation'], round_actions)
        mem.stop('rollout')
        timer.update(rtimer, rollout_s=round(time.time() - t, 2))
        if not torch.equal(samples_z[0], z0[0]):
            raise RuntimeError('latent 0 is not the encoded anchor')
        t = time.time()
        mem.start()
        x = vm.decode_latents(samples_z[:N_DEC])
        mem.stop('decode')
        timer['decode_s'] = round(time.time() - t, 2)
        lat = samples_z
        extra = dict(latents=int(samples_z.shape[0]), decoded_latents=N_DEC, settings_read_back=settings,
                     anchor_latent_sha256=sha256_bytes(z0[0].float().cpu().numpy().tobytes()),
                     noise_check='seed_everything(seed) once per sample, as sample.py; the CUDA stream: cond_frames noise (x cond_aug 0), the round-1 noise, then one noise per later round',
                     actions=None if act_rec is None else dict(mode='trajectory', per_round=[dict(round=r['round'], first_k=r['first_k'], waypoint_ks=r['waypoint_ks'],
                                                                                               trajectory=r['trajectory'], extrapolated=r['extrapolated']) for r in act_rec['rounds']],
                                                                 actions_sha256=spec['inputs']['actions']['sha256']))
    t = time.time()
    hr = vm.to_uint8(x)
    if hr.shape != (N_DEC, size.h, size.w, 3):
        raise RuntimeError(f'decoded {hr.shape}')
    clip = np.concatenate([pre['canvas'], size.to_canvas(hr)], axis=0)
    if clip.shape != (N_CLIP, CH, CW, 3):
        raise RuntimeError(f'clip {clip.shape}')
    write_lossless(clip, d / 'vision.mp4')
    save_native = args.save_native and size.native        # at the canvas size vision.mp4 is the model's own frames
    if save_native:
        write_lossless(np.concatenate([pre['hr'], hr], axis=0), d / size.native_clip)
    if args.save_latents:
        from safetensors.torch import save_file
        save_file({'latents': lat.float().cpu().contiguous()}, str(d / 'latents.safetensors'))
    timer['write_s'] = round(time.time() - t, 2)
    timer['total_s'] = round(time.time() - t_all, 2)
    rec.update(extra, status='success', anchor_frame_sha256=anchor_sha, prefix=pre['record'], prefix_frames_sha256=frames_sha256(clip[:N_PREFIX]),
               t0_frame_sha256=frames_sha256(clip[N_PREFIX]), frames_sha256=frames_sha256(clip), video_sha256=sha256_file(d / 'vision.mp4'),
               video=ffprobe(d / 'vision.mp4'), native1024_sha256=sha256_file(d / size.native_clip) if save_native else None,
               latents_file_sha256=sha256_file(d / 'latents.safetensors') if args.save_latents else None,
               latents_sha256=sha256_bytes(lat.float().cpu().numpy().tobytes()), frame_ticks_k=list(range(-20, 61)),
               clip_layout=B['clip_layout'], timings=timer, gpu_memory=mem.rec, gpu_free_start_mib=round(gpu_free_start), host=host_memory(),
               gpu=gpu_info(torch), software=software(vm), layout_built=vm.layout_record(), model_load_s=vm.load_record['seconds'], finished=now_utc())
    write_json_atomic(d / 'sample_outputs.json', rec)
    return rec


def gpu_guard_setup(torch, args) -> dict:
    """Refuse to start without enough free memory; cap this process so that --leave-free-mib of the free memory stays free."""
    if not torch.cuda.is_available():
        raise SystemExit('no CUDA device')
    info = gpu_info(torch)
    if args.expect_gpu_uuid and info.get('uuid') != args.expect_gpu_uuid:
        raise SystemExit(f'GPU {info.get("uuid")} is not the allowed GPU {args.expect_gpu_uuid}')
    free, total = torch.cuda.mem_get_info(0)
    free_mib, total_mib = free / 2**20, total / 2**20
    if free_mib < args.min_free_mib:
        raise SystemExit(f'GPU free {free_mib:.0f} MiB < --min-free-mib {args.min_free_mib}')
    ctx = getattr(args, 'context_mib', 900)       # CUDA context, cuDNN / cuBLAS handles: outside the allocator
    cap = min(args.max_gpu_mib or 1e12, free_mib - args.leave_free_mib - ctx)
    torch.cuda.set_per_process_memory_fraction(max(cap, 1024) / total_mib, 0)
    return dict(gpu=info, free_at_start_mib=round(free_mib), total_mib=round(total_mib), allocator_cap_mib=round(cap), leave_free_mib=args.leave_free_mib,
                context_mib=ctx)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--specs', nargs='+', type=Path, default=[])
    ap.add_argument('--input-root', type=Path)
    ap.add_argument('--output-root', type=Path)
    ap.add_argument('--vista-src', type=Path)
    ap.add_argument('--ckpt', type=Path)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--save-native', action='store_true')
    ap.add_argument('--save-latents', action='store_true')
    ap.add_argument('--verify-weights', choices=('none', 'size', 'sha256'), default='size')
    ap.add_argument('--allow-unpinned', action='store_true')
    ap.add_argument('--expect-gpu-uuid', default=None)
    ap.add_argument('--min-free-mib', type=float, default=20000)
    ap.add_argument('--leave-free-mib', type=float, default=6144)
    ap.add_argument('--max-gpu-mib', type=float, default=None)
    ap.add_argument('--context-mib', type=float, default=900, help='GPU memory outside the allocator, kept out of the cap')
    ap.add_argument('--min-host-avail-gib', type=float, default=28.0)
    ap.add_argument('--stop-file', type=Path, default=None,
                    help='checked before every sample: if it exists the run stops cleanly (exit 3) after the sample in flight')
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--with-model', action='store_true')
    ap.add_argument('--equivalence-test', action='store_true')
    ap.add_argument('--window', default=None)
    ap.add_argument('--steps', type=int, default=2)
    ap.add_argument('--layout', choices=tuple(B['layouts']), default=B['default_layout'], help='equivalence test only (the runs take it from the specs)')
    ap.add_argument('--size', choices=tuple(SIZES), default=B['default_size'], help='equivalence test only (the runs take it from the specs)')
    ap.add_argument('--allow-superseded-size', action='store_true', help='run specs (or the equivalence test) at a superseded size')
    ap.add_argument('--threads', type=int, default=None, help='torch CPU threads (default: OMP_NUM_THREADS or 2)')
    ap.add_argument('--out', type=Path, default=None)
    args = ap.parse_args()
    if args.self_test:
        sys.exit(self_test(args))
    if args.equivalence_test:
        sys.exit(equivalence_test(args))
    specs = load_specs(args.specs)
    for s in specs:
        check_spec(s, args.input_root)
    todo = [s for s in specs if not (args.resume and is_done(s, args.output_root))]
    print(json.dumps(dict(specs=len(specs), todo=len(todo), skipped=len(specs) - len(todo))), flush=True)
    if args.dry_run or not todo:
        return
    avail = mem_available_gib()
    if avail < args.min_host_avail_gib:
        raise SystemExit(f'MemAvailable {avail} GiB < --min-host-avail-gib {args.min_host_avail_gib} (model load peak about 21 GiB + 4 GiB margin)')
    import torch
    torch.set_num_threads(args.threads or int(os.environ.get('OMP_NUM_THREADS', '2')))
    guard = gpu_guard_setup(torch, args)
    print(json.dumps(dict(guard=guard)), flush=True)
    layouts = {spec_layout(s['generation']) for s in todo}
    if len(layouts) != 1:
        raise SystemExit(f'specs with different memory layouts in one process: {sorted(layouts)}')
    sizes = {size_name(s['generation']) for s in todo}
    if len(sizes) != 1:
        raise SystemExit(f'specs with different generation sizes in one process: {sorted(sizes)}')
    if Size(next(iter(sizes))).superseded and not args.allow_superseded_size:
        raise SystemExit(f'size {next(iter(sizes))} is superseded (pins bench.sizes); --allow-superseded-size to run it anyway')
    lay = B['layouts'][layouts.pop()]
    vm = VistaModel(args.vista_src, args.ckpt, args.verify_weights, low_vram=lay['low_vram'], allow_unpinned=args.allow_unpinned,
                    cond_encode_chunk=lay['cond_encode_chunk'])
    vm.configure_size(Size(sizes.pop()))
    print(json.dumps(dict(model_loaded=vm.load_record)), flush=True)
    failed = 0
    prefix_cache = {}
    for s in todo:
        if args.stop_file and args.stop_file.exists():
            print(json.dumps(dict(stopped=True, reason=f'stop file {args.stop_file}', remaining=len(todo) - todo.index(s))), flush=True)
            sys.exit(3)
        t = time.time()
        try:
            rec = run_one(vm, s, args, prefix_cache)
            print(json.dumps(dict(name=s['name'], run=s['run'], status='success', seconds=round(time.time() - t, 1),
                                  gpu=rec.get('gpu_memory'), host_peak=host_memory().get('vmhwm_gib'))), flush=True)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            d = args.output_root / s['output_dir']
            d.mkdir(parents=True, exist_ok=True)
            mem = {}
            try:
                mem = dict(peak_allocated_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2), peak_reserved_gib=round(torch.cuda.max_memory_reserved() / 2**30, 2))
            except Exception:  # noqa: BLE001
                pass
            write_json_atomic(d / 'sample_outputs.json', dict(schema=OUT_SCHEMA, name=s['name'], run=s['run'], status='failure', error=repr(exc),
                                                              traceback=traceback.format_exc(), spec_sha256=s['_sha256'], gpu_memory_at_failure=mem,
                                                              host=host_memory(), finished=now_utc(), model_load=vm.load_record, guard=guard))
            print(json.dumps(dict(name=s['name'], run=s['run'], status='failure', error=repr(exc)[:300])), flush=True)
            if isinstance(exc, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()
    sys.exit(1 if failed else 0)


# -------------------------------------------------------------------------------------------------------------- checks
def self_test(args) -> int:
    """CPU checks without the GPU (and, with --with-model, the model built on the CPU with every key checked)."""
    import numpy as np
    import tempfile
    res = []

    def ok(name, cond, detail=None):
        res.append(dict(test=name, ok=bool(cond), detail=detail))

    ok('pins: rollout latents', N_LAT == 69 and N_DEC == 61 and N_PREFIX + N_DEC == N_CLIP == 81, dict(N_LAT=N_LAT, N_DEC=N_DEC))
    rng = np.random.default_rng(0)
    x = rng.integers(0, 256, (3, 2 * CH, 2 * CW, 3), dtype=np.uint8)
    ref = np.floor(x.astype(np.float64).reshape(3, CH, 2, CW, 2, 3).mean(axis=(2, 4)) + 0.5).astype(np.uint8)
    ok('area2 = round-half-up 2x2 mean', np.array_equal(area2(x), ref) and area2(x).shape == (3, CH, CW, 3))
    ds, cs, ns = Size(B['default_size']), Size('canvas'), Size('native')
    ok('sizes: default 320 x 576 (inter_area to the canvas, not superseded); canvas 288 x 512 and native superseded',
       ds.name == '320x576' and (ds.h, ds.w) == (320, 576) and ds.mapping == 'inter_area' and not ds.superseded and ds.adaptation is None
       and cs.superseded and ns.superseded and cs.mapping == 'identity' and ns.mapping == 'area2' and size_name(dict(height=320, width=576)) == '320x576',
       dict(default=ds.record(), canvas=cs.record(), native=ns.record()))
    ok('UNet size plan: empty at 320 x 576 and 576 x 1024 (the released UNet runs unmodified); 288 x 512 would need one layer',
       unet_size_plan(40, 72) == {} and unet_size_plan(72, 128) == {} and unet_size_plan(36, 64) == {(5, 8): (9, 16)},
       dict(default=str(unet_size_plan(40, 72)), superseded_canvas=str(unet_size_plan(36, 64))))
    y = rng.integers(0, 256, (2, 320, 576, 3), dtype=np.uint8)
    z = inter_area(y, CW, CH)
    flat = inter_area(np.full((1, 320, 576, 3), 77, np.uint8), CW, CH)
    xi, xw = _area_tables(576, 512)
    ok('inter_area 576 x 320 -> 512 x 288: shape, constant image kept, weights sum to 1 (bit-exactness vs cv2: build_vista_inputs.py)',
       z.shape == (2, CH, CW, 3) and bool((flat == 77).all()) and bool(np.allclose(xw.sum(0), 1, atol=1e-5)) and Size(B['default_size']).to_canvas(y).shape == z.shape,
       dict(x_entries_per_output=int(xi.shape[0])))
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / 'a.mp4'
        clip = rng.integers(0, 256, (5, CH, CW, 3), dtype=np.uint8)
        write_lossless(clip, f)
        back, info = decode_rgb(f)
        ok('lossless writer', np.array_equal(back, clip) and info['fps'] == FPS, info)
    if args.vista_src:
        dg = source_digest(args.vista_src)
        ok('Vista source digest', dg == PINS['code']['source_digest'], dg)
        for rel, want in PINS['code']['files_sha256'].items():
            ok(f'source file {rel}', sha256_file(args.vista_src / rel) == want)
        try:
            ok('superseded 288 x 512 option: patched Upsample = released output off the plan, skip size on it', *upsample_patch_test(args.vista_src))
        except Exception as exc:  # noqa: BLE001
            ok('superseded 288 x 512 option: patched Upsample = released output off the plan, skip size on it', False, repr(exc)[:300])
    if args.specs:
        specs = load_specs(args.specs)
        bad = []
        for s in specs:
            try:
                check_spec(s, args.input_root)
                if s['mode'] == 'continue' and s['track'] == 't2':
                    acts, _ = load_actions(args.input_root / s['inputs']['actions']['path'], s['window'], s['generation'].get('action_variant', 'frame_index'))
                    if len(acts) != N_ROUNDS or any(len(a) != 8 for a in acts):
                        raise ValueError('actions shape')
            except Exception as exc:  # noqa: BLE001
                bad.append(f'{s["name"]}: {exc}')
        ok(f'specs ({len(specs)})', not bad, bad[:5])
        seeds = {}
        for s in specs:
            if s['mode'] == 'continue':
                seeds.setdefault((s['window'], s['k']), set()).add(s['seed'])
        ok('h0 and t2 share seed_of(w, k)', all(len(v) == 1 for v in seeds.values()))
    if args.with_model:
        os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
        try:
            vm = VistaModel(args.vista_src, args.ckpt, 'size', low_vram=True)   # CPU: the low_vram build keeps every sub-model on the CPU
            ok('model built, all keys loaded', vm.load_record['missing_keys'] == 0 and vm.load_record['unexpected_keys'] == 0, vm.load_record)
            ok('low_vram: UNet in fp16', vm.load_record['unet_dtype'] == 'torch.float16', vm.load_record['unet_dtype'])
            ok('decode settings', vm.load_record['en_and_decode_n_samples_a_time'] == 14 and vm.load_record['disable_first_stage_autocast'] and not vm.load_record['use_ema'])
            sampler = vm.su.init_sampling(guider='TrianglePredictionGuider', steps=50, cfg_scale=2.5, num_frames=N_FRAMES)
            rb = vm.read_back(sampler, 'TrianglePredictionGuider')
            ok('sampler read back', rb['num_steps'] == 50 and rb['sigma_max'] == 700.0 and rb['max_scale'] == 2.5 and rb['min_scale'] == 1.0
               and rb['scale_per_frame'][0] == 1.0 and rb['scale_per_frame'][12] == 2.5, rb)
            ok('embedder keys', {'trajectory', 'cond_frames', 'cond_frames_without_noise', 'fps_id', 'motion_bucket_id', 'cond_aug'} <= vm.unique_keys,
               sorted(vm.unique_keys))
        except SystemExit as exc:
            ok('model built', False, str(exc))
    out = dict(self_test=res, passed=sum(r['ok'] for r in res), total=len(res), created=now_utc(), adapter_sha256=sha256_file(Path(__file__)))
    print(json.dumps(out, indent=1))
    if args.out:
        write_json_atomic(args.out, out)
    return 0 if all(r['ok'] for r in res) else 1


def upsample_patch_test(vista_src: Path):
    """CPU: an openaimodel.Upsample patched with the canvas plan returns the released output bit for bit for inputs off the
    plan (9 x 16 -> 18 x 32, and with an empty plan), and nearest-to-9 x 16 then its convolution on the plan (5 x 8)."""
    sys.dont_write_bytecode = True
    if str(Path(vista_src).resolve()) not in sys.path:
        sys.path.insert(0, str(Path(vista_src).resolve()))
    import torch
    import torch.nn.functional as F
    import vwm.modules.diffusionmodules.openaimodel as oam
    torch.manual_seed(0)
    ref, mod, empty = oam.Upsample(8, True, dims=2), oam.Upsample(8, True, dims=2), oam.Upsample(8, True, dims=2)
    mod.load_state_dict(ref.state_dict()); empty.load_state_dict(ref.state_dict())
    patch_upsample(mod, unet_size_plan(36, 64))
    patch_upsample(empty, {})
    a, b = torch.randn(2, 8, 9, 16), torch.randn(2, 8, 5, 8)
    with torch.no_grad():
        off = torch.equal(mod(a), ref(a)) and torch.equal(empty(b), ref(b)) and tuple(ref(b).shape[-2:]) == (10, 16)
        on = mod(b)
        want = ref.conv(F.interpolate(b, size=(9, 16), mode='nearest'))
        rows = F.interpolate(torch.arange(5.0).view(1, 1, 5, 1), size=(9, 1), mode='nearest').flatten().tolist()
    return off and tuple(on.shape[-2:]) == (9, 16) and torch.equal(on, want) and rows == [0, 0, 1, 1, 2, 2, 3, 3, 4], \
        dict(off_plan_identical=off, on_plan_shape=list(on.shape), source_rows_for_9=rows)


def equivalence_test(args) -> int:
    """GPU: the adapter's rollout vs sample_utils.do_sample with the same seed, at --steps sampling steps (a mechanics check
    of the code path: every round, conditioning update, guider and decode). Bit-identical latents are required."""
    import torch
    torch.set_num_threads(2)
    guard = gpu_guard_setup(torch, args)
    lay = B['layouts'][args.layout]
    size = Size(args.size)
    if size.superseded and not args.allow_superseded_size:
        raise SystemExit(f'size {size.name} is superseded; --allow-superseded-size to test it anyway')
    vm = VistaModel(args.vista_src, args.ckpt, 'size', low_vram=lay['low_vram'], cond_encode_chunk=lay['cond_encode_chunk'])
    vm.configure_size(size)
    w = args.window
    frames, _ = decode_rgb(args.input_root / 'windows' / 'dev' / w / size.history)
    if frames.shape[1:3] != (size.h, size.w):
        raise SystemExit(f'{w}: {size.history} is {frames.shape[2]} x {frames.shape[1]}, not the {size.name} size')
    anchor = frames[20]
    acts, _ = load_actions(args.input_root / 'windows' / 'dev' / w / 'actions_t2.json', w)
    gen = dict(n_steps=args.steps, n_rounds=N_ROUNDS, cfg_scale=D['cfg_scale'], cond_aug=D['cond_aug'])
    seed = 12345
    out = dict(window=w, steps=args.steps, seed=seed, layout=args.layout, size=size.record(), layout_built=vm.layout_record(), guard=guard,
               gpu=gpu_info(torch), software=software(vm), cases=[])
    for case, act in (('h0', None), ('t2-constant-action', [acts[0]] * N_ROUNDS)):
        t = time.time()
        z_ad, z0, _, _ = vm.rollout(anchor, seed, gen, act)
        x_ad69 = vm.decode_latents(z_ad)
        x_ad61 = vm.decode_latents(z_ad[:N_DEC])
        t_ad = time.time() - t
        # the official path: sample.py main loop body + sample_utils.do_sample (same action for every round)
        t = time.time()
        images, value_dict, sampler, _ = vm.prepare(anchor, seed, gen['n_steps'], gen['n_rounds'], gen['cfg_scale'], gen['cond_aug'],
                                                    None if act is None else torch.tensor(act[0]))
        samples, samples_z, _ = vm.su.do_sample(images, vm.model, sampler, value_dict, num_rounds=gen['n_rounds'], num_frames=N_FRAMES,
                                                force_uc_zero_embeddings=UC_KEYS, initial_cond_indices=[0])
        t_off = time.time() - t
        u8_off = (255.0 * vm.np.transpose(samples.cpu().numpy(), (0, 2, 3, 1))).astype(vm.np.uint8)
        u8_ad69, u8_ad61 = vm.to_uint8(x_ad69), vm.to_uint8(x_ad61)
        d61 = vm.np.abs(u8_ad61.astype(int) - u8_off[:N_DEC].astype(int))
        mem_peak = dict(peak_allocated_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2), peak_reserved_gib=round(torch.cuda.max_memory_reserved() / 2**30, 2))
        torch.cuda.reset_peak_memory_stats()
        out['cases'].append(dict(case=case, latents_identical=bool(torch.equal(z_ad, samples_z)), decode69_identical=bool(vm.np.array_equal(u8_ad69, u8_off)),
                                 gpu_memory_case=mem_peak,
                                 decode61_vs_official_first61=dict(identical_frames=[int(i) for i in range(N_DEC) if not d61[i].any()][-3:],
                                                                   first_differing_frame=next((int(i) for i in range(N_DEC) if d61[i].any()), None),
                                                                   max_abs=int(d61.max()), mean_abs_frames_55_60=round(float(d61[55:].mean()), 4)),
                                 seconds_adapter=round(t_ad, 1), seconds_official=round(t_off, 1)))
        print(json.dumps(out['cases'][-1]), flush=True)
    out['passed'] = all(c['latents_identical'] and c['decode69_identical'] for c in out['cases'])
    out['created'] = now_utc()
    write_json_atomic(args.out, out)
    return 0 if out['passed'] else 1


if __name__ == '__main__':
    main()
