"""Shared helpers of the Wan2.1-VACE-1.3B tools (stdlib + numpy only: imported by the input builder in the bench
instrument's environment and by the adapter in the VACE environment).

- local.env parsing (the rules of local_env.sh: KEY=VALUE, optional 'export ' and quotes, '#' comments, the environment
  wins, the first line for a key counts);
- the pins (vace_pins.json) and sha256 helpers;
- the 16 fps timeline of bench-v1 (vace_pins.json bench.timeline_rule / tick_rule) and the two extension rounds;
- seed_of() from the read-only code snapshot (the shared seed table of bench-v1);
- the lossless writer and the decoder of the bench clips (libx264rgb -crf 0, rgb24; ffmpeg rawvideo rgb24).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
PINS_PATH = HERE / 'vace_pins.json'
PINS = json.loads(PINS_PATH.read_text())
B = PINS['bench']
FPS = int(B['fps'])                          # 16: Wan2.1's sample_fps
N_HIST = int(B['history_frames'])            # 33 (t0 - 2.0 .. t0)
N_NEW = int(B['new_frames_per_round'])       # 48 (3.0 s)
N_ROUND = int(B['round_frames'])             # 81 = 33 + 48 (frame_num)
N_ROUNDS = int(B['rounds'])                  # 2
N_ALL = int(B['total_frames'])               # 129 = 33 + 2 x 48 (t0 - 2.0 .. t0 + 6.0)
T0 = int(B['t0_index'])                      # 32
R2_OFFSET = int(B['round2_offset'])          # 48: round 2's frame 0 is global frame 48 (t0 + 1.0 s)
CANVAS = tuple(B['canvas_wh'])               # (512, 288)
BAND_ROWS = tuple(B['band_rows'])            # (797, 1669)
GRAY = int(B['gray_value'])                  # 127: the VACE user guide's "missing" value of src_video
SIZES = {name: tuple(rec['wh']) for name, rec in B['sizes'].items()}   # canvas (512, 288), 480p (832, 480)
TAGS = {name: rec['tag'] for name, rec in B['sizes'].items()}
TICKS = list(range(-20, 61))                 # bench-v1 10 Hz ticks k
NANO_SPECS_SHA256 = '108c4cc974487284182bcc4fc6c84ff12477437059e9a90a50e1fa6fd441968c'  # make_bench_nano_specs.py (bench-v1)
SNAPSHOT_TAR_SHA256 = '2f9f615394f5cb779de0bfee8b478af8b24f33c3c8c980c702b514aab1e59aa0'  # orbis2-code-snapshot-v1.tar


def size_name(name_or_wh) -> str:
    """'canvas' / '480p' for a size given by name or as a (W, H) pair; refuses anything else."""
    if isinstance(name_or_wh, str):
        if name_or_wh not in SIZES:
            raise ValueError(f'generation size {name_or_wh}: not one of {sorted(SIZES)}')
        return name_or_wh
    wh = tuple(int(x) for x in name_or_wh)
    for n, s in SIZES.items():
        if s == wh:
            return n
    raise ValueError(f'generation size {wh}: not one of {sorted(SIZES.values())}')


def input_names(name: str) -> dict:
    """Input file names of one generation size."""
    tag = TAGS[size_name(name)]
    return dict(history=f'history16_{tag}.mp4', real=f'real16_{tag}.mp4', frames=f'frames16_{tag}.json')


# ------------------------------------------------------------------------------------------------------------ local.env
def load_local_env(path: str | os.PathLike | None = None) -> dict:
    """Parse local.env into os.environ (a variable already set wins); returns the parsed pairs."""
    f = Path(path or os.environ.get('WCB_LOCAL_ENV') or HERE / 'local.env')
    got = {}
    if not f.is_file():
        return got
    for line in f.read_text().splitlines():
        line = line.split('#', 1)[0].strip()
        if line.startswith('export '):
            line = line[len('export '):]
        if '=' not in line:
            continue
        key, val = line.split('=', 1)
        key, val = key.strip(), val.strip()
        if not key.replace('_', 'a').isalnum() or key[0].isdigit():
            continue
        if len(val) >= 2 and val[0] == val[-1] and val[0] in '"\'':
            val = val[1:-1]
        if key in got:
            continue
        got[key] = val
        if not os.environ.get(key):
            os.environ[key] = val
    return got


def setting(name: str, required: bool = True, default: str | None = None) -> str | None:
    v = os.environ.get(name) or default
    if required and not v:
        raise SystemExit(f'{name} is not set: put it into models/wan21_vace13b/local.env (template local.env.example) or the environment')
    if v and ('REPLACE' in v or v.startswith('/path/to/')):
        raise SystemExit(f'{name} still holds the placeholder of local.env.example: set it in models/wan21_vace13b/local.env')
    return v


# ---------------------------------------------------------------------------------------------------------------- hashes
def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_blob_sha1(path) -> str:
    data = Path(path).read_bytes()
    return hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()


def source_digest(root: Path) -> str:
    """The pins' rule: sha256 over the sorted *.py / *.yml / *.yaml files of the checkout (path NUL sha256 newline)."""
    root = Path(root)
    files = sorted(str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p.suffix in ('.py', '.yml', '.yaml')
                   and '.git' not in p.parts and '__pycache__' not in p.parts and not p.name.startswith('.'))
    h = hashlib.sha256()
    for f in files:
        h.update(f.encode() + b'\0' + sha256_file(root / f).encode() + b'\n')
    return h.hexdigest()


# --------------------------------------------------------------------------------------------------------- the timeline
def _round_half_even(x: Fraction) -> int:
    """round() of a Fraction: ties go to the even integer (exact arithmetic)."""
    return round(x)


def camera_offset(j: int) -> int:
    """Camera-frame offset m(j) from the t0 camera frame for global 16 fps frame j (0..128): the 20 Hz camera frame nearest
    to t0_cam + (j - 32) / 16 s, ties (every j - 32 = 2 mod 4, 25 ms each way) to the even offset, i.e. to the bench's
    10 Hz frame. Exact: m = round_half_even(5 (j - 32) / 4)."""
    return _round_half_even(Fraction(5 * (j - T0), 4))


def frame_of_tick(k: int) -> int:
    """Global 16 fps frame of bench tick k (t0 + 0.1 k s): 32 + round(1.6 k); 1.6 k never ends in .5, so no ties."""
    return T0 + _round_half_even(Fraction(8 * k, 5))


def round_of_frame(j: int) -> tuple:
    """(round, index in that round's 81-frame output) of the generated global frame j > 32: round 1 holds frames 33-80
    (indices 33-80), round 2 frames 81-128 (indices 33-80 of round 2, whose frame 0 is global frame 48)."""
    if T0 < j < N_ROUND:
        return 1, j
    if N_ROUND <= j < N_ALL:
        return 2, j - R2_OFFSET
    raise ValueError(f'frame {j} is not a generated frame')


def timeline_table() -> dict:
    """The whole mapping (written into frames16_*.json and checked by the self-test)."""
    rows = []
    for j in range(N_ALL):
        m = camera_offset(j)
        rows.append(dict(j=j, model_time_s=round((j - T0) / FPS, 6), camera_offset=m,
                         camera_minus_model_ms=round(float((Fraction(m, 20) - Fraction(j - T0, FPS)) * 1000), 3)))
    ticks = []
    for k in TICKS:
        j = frame_of_tick(k)
        src = 'prefix' if j <= T0 else f'round{round_of_frame(j)[0]}'
        ticks.append(dict(k=k, frame16=j, source=src, model_minus_tick_ms=round(float((Fraction(j - T0, FPS) - Fraction(k, 10)) * 1000), 3),
                          camera_offset=camera_offset(j)))
    return dict(rows=rows, ticks=ticks)


def check_timeline() -> list[str]:
    """Self-checks of the 16 fps rule; returns failures."""
    bad = []
    hist = [camera_offset(j) for j in range(N_HIST)]
    if hist[0] != -40 or hist[-1] != 0 or len(set(hist)) != N_HIST or hist != sorted(hist):
        bad.append('history: camera frames are not 33 distinct increasing frames c0-40 .. c0')
    allm = [camera_offset(j) for j in range(N_ALL)]
    if allm[-1] != 120 or len(set(allm)) != N_ALL or allm != sorted(allm):
        bad.append('clip: camera frames are not 129 distinct increasing frames c0-40 .. c0+120')
    for j in range(N_ALL):
        m = camera_offset(j)
        e = abs(Fraction(m, 20) - Fraction(j - T0, FPS))
        if e > Fraction(1, 40):
            bad.append(f'j={j}: camera frame {m} is {float(e) * 1000:.1f} ms off')
        for mm in (m - 1, m + 1):            # nearest; on a tie the even one
            e2 = abs(Fraction(mm, 20) - Fraction(j - T0, FPS))
            if e2 < e or (e2 == e and mm % 2 == 0):
                bad.append(f'j={j}: camera frame {m} is not the nearest (ties to the even offset)')
    for k in TICKS:
        j = frame_of_tick(k)
        dt = abs(Fraction(j - T0, FPS) - Fraction(k, 10))
        if dt > Fraction(1, 40):
            bad.append(f'tick {k}: frame {j} more than 25 ms off')
        if k % 5 == 0 and dt != 0:
            bad.append(f'tick {k}: a 0.5 s multiple is not exact')
        if camera_offset(j) != 2 * k:
            bad.append(f'tick {k}: 16 fps frame {j} is camera frame {camera_offset(j)}, not the bench tick frame {2 * k}')
    if [frame_of_tick(k) for k in (-20, 0, 10, 20, 30, 31, 40, 60)] != [0, 32, 48, 64, 80, 82, 96, 128]:
        bad.append('ticks at -2, 0, 1, 2, 3, 3.1, 4, 6 s are not frames 0, 32, 48, 64, 80, 82, 96, 128')
    if [round_of_frame(j) for j in (33, 80, 81, 128)] != [(1, 33), (1, 80), (2, 33), (2, 80)]:
        bad.append('round mapping')
    return bad


# ------------------------------------------------------------------------------------------------------------- seed_of
def snapshot_dir(arg=None) -> Path:
    d = Path(arg) if arg else Path(os.environ['WCB_SNAPSHOT_DIR']) if os.environ.get('WCB_SNAPSHOT_DIR') else HERE.parents[1] / 'snapshot'
    if not (d / 'scripts' / 'make_bench_nano_specs.py').is_file():
        raise SystemExit(f'code snapshot not found at {d}: unpack orbis2-code-snapshot-v1.tar (it creates snapshot/) and pass '
                         '--snapshot DIR or set WCB_SNAPSHOT_DIR')
    return d


def load_seed_of(snapshot=None):
    """seed_of() and PROMPTS from the snapshot of the Nano spec generator, after its sha256 is checked against the pinned
    value and the snapshot's MANIFEST.sha256 (the loader of models/vista and models/skyreels_v2_df)."""
    d = snapshot_dir(snapshot)
    nano = d / 'scripts' / 'make_bench_nano_specs.py'
    listed = None
    for line in (d / 'MANIFEST.sha256').read_text().splitlines():
        digest, _, rel = line.partition('  ')
        if rel.strip() == 'scripts/make_bench_nano_specs.py':
            listed = digest
    got = sha256_file(nano)
    if got != NANO_SPECS_SHA256 or listed != NANO_SPECS_SHA256:
        raise SystemExit(f'{nano}: sha256 {got} (manifest {listed}), pinned {NANO_SPECS_SHA256}')
    spec = importlib.util.spec_from_file_location('make_bench_nano_specs_snapshot', nano)
    mod = importlib.util.module_from_spec(spec)
    keep, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = keep
    return mod.seed_of, mod.PROMPTS, got


def seed_of_local(window: str, k) -> int:
    """The shared seed table's rule (make_bench_nano_specs.seed_of), for checks without the snapshot."""
    return int.from_bytes(hashlib.sha256(f'bench-v1|{window}|{k}'.encode()).digest()[:4], 'big') & 0x7FFFFFFF


# ---------------------------------------------------------------------------------------------------------------- video
def write_video(frames, path: Path, fps: int, threads: int = 2) -> None:
    """The bench's lossless writer (make_wm_calibration_windows.write_video with -r fps): libx264rgb -crf 0 -preset medium,
    rgb24 in and out, timescale 10240, no metadata."""
    import numpy as np
    path = Path(path)
    h, w = frames[0].shape[:2]
    tmp = path.with_name(path.stem + '.part.mp4')
    proc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}', '-r', str(fps), '-i', '-',
                             '-c:v', 'libx264rgb', '-crf', '0', '-preset', 'medium', '-threads', str(threads), '-pix_fmt', 'rgb24', '-r', str(fps),
                             '-video_track_timescale', '10240', '-map_metadata', '-1', '-movflags', '+faststart', str(tmp)], stdin=subprocess.PIPE)
    for f in frames:
        proc.stdin.write(np.ascontiguousarray(f, dtype=np.uint8).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f'ffmpeg failed for {path}')
    os.replace(tmp, path)


def decode(path: Path, wh, max_frames: int | None = None, threads: int = 2):
    import numpy as np
    cmd = ['ffmpeg', '-v', 'error', '-threads', str(threads), '-i', str(path)]
    if max_frames is not None:
        cmd += ['-frames:v', str(max_frames)]
    raw = subprocess.check_output(cmd + ['-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'])
    return np.frombuffer(raw, np.uint8).reshape(-1, wh[1], wh[0], 3)


def ffprobe(path: Path) -> dict:
    out = subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames', '-show_entries',
                                   'stream=width,height,nb_read_frames,r_frame_rate,pix_fmt,codec_name', '-of', 'json', str(path)], text=True)
    s = json.loads(out)['streams'][0]
    num, den = s['r_frame_rate'].split('/')
    return dict(width=int(s['width']), height=int(s['height']), frames=int(s['nb_read_frames']), fps=float(num) / float(den),
                pix_fmt=s.get('pix_fmt'), codec=s.get('codec_name'))
