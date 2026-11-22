#!/usr/bin/env python3
"""Epona adapter for the wm-calib-bench benchmark (bench-v1).

Reads spec JSONs (one per sample; see make_epona_specs.py) and writes, per sample,
  <output-root>/<spec.output_dir>/vision.mp4           41 frames, 512x288 (the evaluation canvas), 5 fps, lossless (libx264rgb -crf 0)
  <output-root>/<spec.output_dir>/sample_outputs.json  status (success / failure), every setting, hardware, versions, timings, memory
  <output-root>/<spec.output_dir>/pred_traj.npy        continue mode: Epona's self-predicted trajectories, float32 [30, 15, 3]
                                                       (per generated frame: 15 future steps of (dx m, dy m, dyaw deg) at 5 Hz)
  <output-root>/<spec.output_dir>/native.mp4           with --save-native: the same 41 frames at the model's 1024x512

Frame i of every clip is tick k = -20 + 2 i (t0 + k x 0.1 s), i = 0..40: the bench's 10 Hz ticks on Epona's 5 Hz grid.

Modes (spec "mode"):
  continue  Epona's released self-predicted-trajectory rollout (scripts/test/test_demo.py / test_free.py), as is:
            - condition: the 10 history frames k = -18..0 (5 Hz, 1.8 s; history.mp4 frames 1-10, the band rendered at 1024x512),
              encoded together (tokenizer.encode_to_z on the 10 frames); the 10 ego poses at the same ticks give the relative
              poses with Epona's own get_rel_pose (test_free.py: entry 0 = 0, entry i = pose i-1 -> i in frame i-1, x forward,
              y left, yaw left positive in degrees); the 11th entry (the next frame's pose, which step_eval replaces by the
              self-predicted one) is a copy of the 10th, as the demo's own rollout fills it ("pred, pred");
            - 30 autoregressive steps (6.0 s at 5 Hz), each: STT, TrajDiT (100 Euler steps), VisDiT (100 Euler steps) under
              bfloat16 autocast; the first predicted pose conditions the frame and enters the pose window ("pred, pred");
            - generated latents decoded in groups of 6 with is_video=False (the demo's z_to_image), uint8 by the demo's rule.
            vision.mp4 = frames 0-10: the per-frame DCAE round trip of the 11 history frames (k = -20..0; the same function as
            the real reference, so these frames equal the reference's), then frames 11-40: the 30 generated frames, each mapped
            to the canvas (default band: cv2.resize 1024x512 -> 512x288, INTER_AREA; spec generation.model_band_rows other than
            [797, 1669], an exploratory variant: the rows covering the canvas, resized the same way, uncovered canvas rows
            black). The spec references the history clip and the past poses only.
  recon     The real reference: the 41 real frames (real.mp4) through the same DCAE: frames 0-10 by the per-frame round trip,
            frames 11-40 encoded one at a time and decoded in groups of 6 like the generated frames; same uint8 rule and resize.

The model is built as the demo builds it (TrainTransformersDiT(cfg, condition_frames=10) and VAETokenizer(cfg) from the pinned
configs/dit_config_dcae_nuplan.py, vae_ckpt set). Differences from the demo, none of which changes the sampled distribution:
  - the config file is read with runpy (its top-level values; see load_config): the released utils/config_utils.Config
    cannot load any config (undefined helpers; Epona issues #5, #7, #14, #22);
  - the world-model checkpoint is read with torch.load(mmap=True, weights_only=True) and copied into the modules with the
    demo's own key mapping (module.model./module.dit./module.traj_dit.), instead of an unmapped CPU load;
  - each sample re-seeds random, numpy and torch (CPU and CUDA) with its seed right before it runs; the demo seeds once at
    start, before model construction. The initial noise comes from the CPU generator (torch.randn(...).to(gpu)), so the
    streams for a given seed differ from the demo's, with the same distribution;
  - cuDNN deterministic, benchmark off (the demo turns benchmark on);
  - the demo's crash line (rot_matrix undefined) is not used; the poses go to the GPU as in test_free.py;
  - output frames are written losslessly instead of as PNG + mp4 at 5 fps.

Batched option (--batch-k N, N >= 2; not used by the default path): the continuation samples of a window run in fixed
chunks of N consecutive k (rollout_batch / step_eval_gen: the released step_eval line for line, except that row b draws its
initial noise from torch.Generator().manual_seed(seed_b) with the shapes of a batch-1 call, so each seed keeps its meaning;
with N = 1 this reproduces the released path bit for bit). Batched outputs differ from batch-1 outputs only by kernel-level
numerics (GEMM tiling for the larger batch), which autoregression can grow into visibly different futures of the same law;
the layout (size, members, position) is recorded per sample, and a rerun with the same layout is bit-identical on the same
GPU type. The chunk is the unit of resumption: if any member is unfinished the chunk runs again whole, finished members are
checked for bit identity and kept. --writer-preset ultrafast writes the same lossless frames faster into larger files.

Resumable: a sample whose sample_outputs.json says success and whose vision.mp4 matches the recorded sha256 is skipped.
A failed sample is recorded with status "failure" and the run continues; the exit code is 1 if any sample failed.

  run_epona_bench.py --specs DIR_OR_JSON [...] --input-root DIR --output-root DIR --models-dir DIR --epona-src DIR
                     [--resume] [--dry-run] [--min-free-mib N] [--max-gpu-mib N] [--leave-free-mib N] [--min-host-avail-gib G]
                     [--expect-gpu-uuid UUID] [--verify-weights size|sha256] [--save-native] [--batch-k N]
                     [--writer-preset medium|veryfast|ultrafast] [--dcae-only] [--recon-prefix-root DIR]
Specs made with make_epona_specs.py --batch-k N are bound to that layout: the run refuses another --batch-k and refuses
specs of different layouts together. --dcae-only (real references only) skips the world model. --recon-prefix-root DIR
(TEST real references, after freezing; the cluster TEST package carries no future frames) takes frames 0-10 of each reference
from a finished continuation of the same window under DIR, so all 17 clips share them exactly.
On the research server use run_dev_local.sh (GPU 1, shared lock, memory guards).
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import platform
import random
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
PINS_PATH = HERE / 'epona_pins.json'
PINS = json.loads(PINS_PATH.read_text())
DEF = PINS['released_defaults']
MW, MH = PINS['geometry']['model_size_wh']
CW, CH = PINS['geometry']['canvas_size_wh']
OUT_FRAMES, FPS = PINS['output']['frames'], PINS['output']['fps']
HIST_TICKS = PINS['timing']['history_ticks']
N_HIST = len(HIST_TICKS)                      # 11
CF = DEF['condition_frames']                  # 10
N_FUT = DEF['future_frames']                  # 30
GROUP = DEF['decode_group']                   # 6
TICKS = list(range(-20, 61, 2))               # 41
SPEC_SCHEMA = 'wm-calib-bench.epona.spec.v1'
OUT_SCHEMA = 'wm-calib-bench.epona.sample_outputs.v1'


# ----------------------------------------------------------------------------------------------------------------------
# small helpers (no torch)
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
    """The pins' rule: sha256 over the sorted *.py / *.yml / *.yaml files of the Epona source tree."""
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


def decode_rgb(path: Path, threads: int):
    import numpy as np
    info = ffprobe(path)
    raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-threads', str(threads), '-i', str(path), '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'])
    frames = np.frombuffer(raw, np.uint8).reshape(-1, info['height'], info['width'], 3).copy()
    if frames.shape[0] != info['frames']:
        raise RuntimeError(f'{path}: decoded {frames.shape[0]} frames, ffprobe counted {info["frames"]}')
    return frames, info


def write_lossless(frames, path: Path, threads: int, preset: str = 'medium') -> None:
    """The bench's encoder (libx264rgb -crf 0 -preset medium, rgb24, video_track_timescale 10240) at 5 fps. Any preset
    is lossless at -crf 0 (the decoded frames are checked); 'ultrafast' only makes the file larger and the write faster."""
    import numpy as np
    h, w = frames.shape[1:3]
    tmp = path.with_name(path.stem + '.part.mp4')
    proc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}', '-r', str(FPS), '-i', '-',
                             '-c:v', 'libx264rgb', '-crf', '0', '-preset', preset, '-threads', str(threads), '-pix_fmt', 'rgb24', '-r', str(FPS),
                             '-video_track_timescale', '10240', '-map_metadata', '-1', '-movflags', '+faststart', str(tmp)], stdin=subprocess.PIPE)
    for f in frames:
        proc.stdin.write(np.ascontiguousarray(f).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f'ffmpeg failed for {path}')
    os.replace(tmp, path)


def ffmpeg_version() -> str:
    try:
        return subprocess.check_output(['ffmpeg', '-version'], text=True).splitlines()[0]
    except Exception as exc:  # noqa: BLE001
        return f'unavailable: {exc}'


def load_specs(paths) -> list:
    files = []
    for p in map(Path, paths):
        files += sorted(p.rglob('*.json')) if p.is_dir() else [p]
    specs = []
    for f in files:
        s = json.loads(f.read_text())
        if s.get('schema') != SPEC_SCHEMA:
            continue  # manifests and other JSON files next to the specs
        s['_file'] = str(f)
        s['_sha256'] = sha256_bytes(f.read_bytes())
        specs.append(s)
    names = [s['name'] for s in specs]
    if len(set(names)) != len(names):
        raise SystemExit('duplicate sample names among the specs')
    specs.sort(key=lambda s: (s['split'], s['window'], 0 if s['mode'] == 'recon' else 1, s.get('k') or 0))
    return specs


def check_spec(spec: dict, input_root: Path) -> None:
    for key in ('name', 'window', 'split', 'mode', 'seed', 'inputs', 'output_dir', 'generation'):
        if key not in spec:
            raise ValueError(f'spec {spec.get("name")}: missing {key}')
    if spec['mode'] == 'continue':
        if set(spec['inputs']) != {'history', 'poses'}:
            raise ValueError(f'{spec["name"]}: a continue spec must reference the history clip and the past poses only, got {sorted(spec["inputs"])}')
        if max(spec['inputs']['poses'].get('ticks', [1])) > 0:
            raise ValueError(f'{spec["name"]}: the pose file must hold past ticks only')
    elif spec['mode'] == 'recon':
        if set(spec['inputs']) != {'real'}:
            raise ValueError(f'{spec["name"]}: a recon spec must reference the real clip only, got {sorted(spec["inputs"])}')
    else:
        raise ValueError(f'{spec["name"]}: unknown mode {spec["mode"]}')
    for rec in spec['inputs'].values():
        p = Path(rec['path'])
        if p.is_absolute() or '..' in p.parts:
            raise ValueError(f'{spec["name"]}: input paths must be relative to the input root')
        if not (input_root / p).is_file():
            raise FileNotFoundError(f'{spec["name"]}: {input_root / p} missing')
    gen = spec['generation']
    # the model band: the spec's, checked against the band the inputs were rendered with (build.json); specs written before
    # the band was recorded take the inputs' band, never a default
    first_input = input_root / Path(next(iter(spec['inputs'].values()))['path'])
    build = first_input.parent / 'build.json'
    # (build records written before the band option existed have no model_band_rows: they were rendered with [797, 1669))
    band_in = json.loads(build.read_text())['render'].get('model_band_rows', [797, 1669]) if build.is_file() else None
    band = gen.get('model_band_rows') or band_in
    if band is None:
        raise ValueError(f'{spec["name"]}: no model band in the spec and no build.json next to its inputs')
    if band_in is not None and list(band) != list(band_in):
        raise ValueError(f'{spec["name"]}: spec band {band} differs from the inputs\' band {band_in}')
    canvas_map(band)  # validates the band
    spec['_band'] = list(band)
    for key, want in (('num_sampling_steps', DEF['config_checked']['num_sampling_steps']), ('condition_frames', CF), ('future_frames', N_FUT),
                      ('decode_group', GROUP), ('self_pred_traj', True), ('epona_commit', PINS['code']['commit']),
                      ('hf_revision', PINS['weights']['hf_revision'])):
        if gen.get(key) != want:
            raise ValueError(f'{spec["name"]}: generation.{key} = {gen.get(key)}, the pinned value is {want}')


def canvas_map(band) -> dict:
    """Which model-frame rows cover the canvas, for a 16:9 full-width model band [a, a + 872) with a <= 797 (pins rule)."""
    a, b = int(band[0]), int(band[1])
    s0, s1 = PINS['geometry']['band_rows']
    if b - a != s1 - s0 or a > s0 or ((s0 - a) * MH) % (b - a):
        raise ValueError(f'model band {band} breaks the pins rule ({PINS["geometry"]["model_band_rule"]})')
    top = (s0 - a) * MH // (b - a)                               # model row of source row 797
    rows_c = round((min(b, s1) - s0) * CH / (s1 - s0))           # canvas rows covered
    return dict(model_band_rows=[a, b], model_rows=[top, MH], canvas_rows=[0, rows_c])


def is_done(spec: dict, output_root: Path) -> bool:
    d = output_root / spec['output_dir']
    f, v = d / 'sample_outputs.json', d / 'vision.mp4'
    if not (f.is_file() and v.is_file()):
        return False
    try:
        rec = json.loads(f.read_text())
    except json.JSONDecodeError:
        return False
    return rec.get('status') == 'success' and rec.get('spec_sha256') == spec['_sha256'] and rec.get('video_sha256') == sha256_file(v)


def load_config(path: Path, overrides: dict):
    """The pinned config file's module-level values as an attribute namespace (missing attributes raise), plus the demo's
    command-line values that the model reads (batch_size 1) and the checkpoint paths.

    The released utils/config_utils.Config (a partial copy of mmengine's Config) cannot load any config: _file2dict calls
    names that are not defined in the file (check_file_exist, re, RemoveAssignFromAST, ...; issues #5, #7, #14, #22; the
    maintainer: comment them out, "these functions are not actually used"). For a plain-assignment config like
    dit_config_dcae_nuplan.py (no _base_, no imports, literals only) its result is the file's top-level variables, which is
    what runpy.run_path returns."""
    import runpy
    import types
    ns = runpy.run_path(str(path))
    vals = {k: v for k, v in ns.items() if not k.startswith('_') and not isinstance(v, types.ModuleType) and not callable(v)}
    vals.update(overrides)
    return types.SimpleNamespace(**vals)


# ----------------------------------------------------------------------------------------------------------------------
class Epona:
    """The released Epona world model (epona_nuplan+china.pkl) and temporal-aware DCAE, built as scripts/test/test_demo.py builds them."""

    def __init__(self, models_dir: Path, src: Path, verify_weights: str, allow_unpinned: bool, threads: int, world_model: bool = True):
        self.models_dir, self.src, self.threads = models_dir, src, threads
        self.digest = source_digest(src)
        if self.digest != PINS['code']['source_digest'] and not allow_unpinned:
            raise SystemExit(f'Epona source at {src} has digest {self.digest}, pinned {PINS["code"]["source_digest"]} (commit {PINS["code"]["commit"]})')
        self.weight_checks = self._check_weights(verify_weights, allow_unpinned)
        sys.dont_write_bytecode = True   # keep the pinned source tree free of __pycache__
        sys.path.insert(0, str(src))

        import numpy as np
        import torch
        from einops import rearrange
        from models.model import TrainTransformersDiT
        from models.modules.tokenizer import VAETokenizer, poses_to_indices, yaws_to_indices
        from models.modules.sampling import prepare_ids, get_schedule
        from utils.preprocess import get_rel_pose
        self.torch, self.np, self.rearrange, self.get_rel_pose = torch, np, rearrange, get_rel_pose
        self.poses_to_indices, self.yaws_to_indices, self.prepare_ids, self.get_schedule = poses_to_indices, yaws_to_indices, prepare_ids, get_schedule

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        t0 = time.time()
        cfg = load_config(src / DEF['config'], dict(batch_size=1, exp_name='wm-calib-bench', config=DEF['config'],
                                                     resume_path=str(models_dir / 'epona_nuplan+china.pkl'),
                                                     vae_ckpt=str(models_dir / 'dcae_td_20000.pkl')))
        self.config_values = {k: v for k, v in vars(cfg).items() if k not in ('resume_path', 'vae_ckpt')}
        bad = {}
        for k, want in DEF['config_checked'].items():
            got = getattr(cfg, k)
            got = list(got) if isinstance(got, (tuple, list)) else got
            if got != want:
                bad[k] = (got, want)
        if bad and not allow_unpinned:
            raise SystemExit(f'the pinned config differs from the pins: {bad}')
        self.cfg = cfg
        self.lat_h = cfg.image_size[0] // (cfg.downsample_size * cfg.patch_size)
        self.lat_w = cfg.image_size[1] // (cfg.downsample_size * cfg.patch_size)
        self.ckpt_keys = None
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            model = None
            if world_model:   # a DCAE-only process (real references only) skips the 2.6 B-parameter world model
                model = TrainTransformersDiT(cfg, load_path=None, local_rank=0, condition_frames=cfg.condition_frames)
                self._load_world_model(model, models_dir / 'epona_nuplan+china.pkl')
                model.eval()
            tokenizer = VAETokenizer(cfg, 0)
        self.build_log = buf.getvalue()[-4000:]
        self.model, self.tokenizer = model, tokenizer
        self.load_s = round(time.time() - t0, 1)
        self.settings = self._read_back_settings()
        self._install_counters()

    # -- pins ----------------------------------------------------------------------------------------------------------
    def _check_weights(self, mode: str, allow_unpinned: bool) -> dict:
        out = {}
        for rel in ('epona_nuplan+china.pkl', 'dcae_td_20000.pkl'):
            rec = PINS['weights']['files'][rel]
            p = self.models_dir / rel
            if not p.is_file():
                raise SystemExit(f'{p} missing')
            entry = dict(path=rel, bytes=p.stat().st_size)
            if entry['bytes'] != rec['bytes'] and not allow_unpinned:
                raise SystemExit(f'{p}: {entry["bytes"]} bytes, pinned {rec["bytes"]}')
            if mode == 'sha256':
                entry['sha256'] = sha256_file(p)
                if entry['sha256'] != rec['sha256'] and not allow_unpinned:
                    raise SystemExit(f'{p}: sha256 {entry["sha256"]}, pinned {rec["sha256"]}')
            out[rel] = entry
        out['_mode'] = mode
        return out

    def _load_world_model(self, model, path: Path):
        """models/model.py's loading loop (prefixes module.model., module.traj_dit., module.dit.; strict), from an mmap'd
        weights-only load instead of a full CPU copy."""
        torch = self.torch
        state = torch.load(str(path), map_location='cpu', mmap=True, weights_only=True)['model_state_dict']
        self.ckpt_keys = dict(total=len(state), unused=sorted({k.split('.')[1] for k in state} - {'model', 'dit', 'traj_dit'}))
        for sub, prefix in ((model.model, 'module.model.'), (model.traj_dit, 'module.traj_dit.'), (model.dit, 'module.dit.')):
            sd = sub.state_dict()
            for k in sd.keys():
                sd[k] = state[prefix + k]
            sub.load_state_dict(sd)
        del state

    def _read_back_settings(self) -> dict:
        m, cfg, torch = self.model, self.cfg, self.torch
        npar = lambda mod: int(sum(p.numel() for p in mod.parameters()))  # noqa: E731
        if m is None:
            got = dict(world_model='not loaded (DCAE-only process: real references only)', tokenizer=type(self.tokenizer.vae).__name__,
                       params_m=dict(dcae=round(npar(self.tokenizer.vae) / 1e6, 1)), latent_hw=[self.lat_h, self.lat_w], image_size=list(cfg.image_size),
                       decoder_temporal=bool(self.tokenizer.vae.add_decoder_temporal), encoder_temporal=bool(self.tokenizer.vae.add_encoder_temporal),
                       dcae_temporal_frames=int(getattr(self.tokenizer.vae, 'condition_frames', 0)), config_values=self.config_values,
                       config_loader='runpy.run_path (the released Config.fromfile is broken, see load_config)',
                       tf32_matmul=bool(torch.backends.cuda.matmul.allow_tf32), tf32_cudnn=bool(torch.backends.cudnn.allow_tf32),
                       cudnn_deterministic=bool(torch.backends.cudnn.deterministic), cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
                       training_flags=dict(dcae=self.tokenizer.vae.training))
            expect = dict(latent_hw=[16, 32], image_size=[512, 1024], decoder_temporal=True, encoder_temporal=False, dcae_temporal_frames=6)
            bad = {k: (got[k], v) for k, v in expect.items() if got[k] != v}
            if bad:
                raise SystemExit(f'the loaded DCAE does not run the pinned released defaults: {bad}')
            return got
        got = dict(world_model=type(m).__name__, stt=type(m.model).__name__, vis_dit=type(m.dit).__name__, traj_dit=type(m.traj_dit).__name__,
                   tokenizer=type(self.tokenizer.vae).__name__, params_m=dict(stt=round(npar(m.model) / 1e6, 1), vis_dit=round(npar(m.dit) / 1e6, 1),
                   traj_dit=round(npar(m.traj_dit) / 1e6, 1), dcae=round(npar(self.tokenizer.vae) / 1e6, 1)),
                   param_dtype=str(next(m.dit.parameters()).dtype), condition_frames=int(m.condition_frames), traj_len=int(m.traj_len),
                   latent_hw=[self.lat_h, self.lat_w], img_token_size=int(m.img_token_size), num_sampling_steps=int(cfg.num_sampling_steps),
                   diffusion_model_type=cfg.diffusion_model_type, downsample_fps=int(cfg.downsample_fps), image_size=list(cfg.image_size),
                   decoder_temporal=bool(self.tokenizer.vae.add_decoder_temporal), encoder_temporal=bool(self.tokenizer.vae.add_encoder_temporal),
                   dcae_temporal_frames=int(getattr(self.tokenizer.vae, 'condition_frames', 0)), checkpoint_keys=self.ckpt_keys,
                   config_values=self.config_values, config_loader='runpy.run_path (the released Config.fromfile is broken, see load_config)',
                   tf32_matmul=bool(torch.backends.cuda.matmul.allow_tf32), tf32_cudnn=bool(torch.backends.cudnn.allow_tf32),
                   cudnn_deterministic=bool(torch.backends.cudnn.deterministic), cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
                   training_flags=dict(stt=m.model.training, dit=m.dit.training, traj_dit=m.traj_dit.training, dcae=self.tokenizer.vae.training))
        expect = dict(condition_frames=CF, traj_len=15, latent_hw=[16, 32], img_token_size=512, num_sampling_steps=100, diffusion_model_type='flow',
                      downsample_fps=5, image_size=[512, 1024], decoder_temporal=True, encoder_temporal=False, dcae_temporal_frames=6)
        bad = {k: (got[k], v) for k, v in expect.items() if got[k] != v}
        if bad:
            raise SystemExit(f'the loaded model does not run the pinned released defaults: {bad}')
        return got

    def _install_counters(self):
        """Count network evaluations (forward hooks; no effect on the numbers): STT (via its image projector, once per
        evaluate), TrajDiT and VisDiT (once per Euler step)."""
        self.calls = dict(stt=0, traj_dit=0, vis_dit=0)
        if self.model is None:
            return

        def inc(key):
            return lambda *a: self.calls.__setitem__(key, self.calls[key] + 1)
        self.model.model.img_projector.register_forward_hook(inc('stt'))
        self.model.traj_dit.register_forward_hook(inc('traj_dit'))
        self.model.dit.register_forward_hook(inc('vis_dit'))

    # -- tensors -------------------------------------------------------------------------------------------------------
    def normalize(self, frames_u8):
        """dataset_demo.DemoDataset: uint8 HWC -> float [F, 3, H, W] in [-1, 1] ((x / 255 - 0.5) * 2)."""
        torch = self.torch
        imgs = torch.stack([torch.from_numpy(f.copy()).permute(2, 0, 1) for f in frames_u8], 0)
        imgs = imgs / 255.0
        return (imgs - 0.5) * 2

    def to_u8(self, img_chw):
        """The demo's rule: (img.permute(1, 2, 0).numpy() * 255).astype('uint8') on z_to_image's [0, 1] output (kept RGB)."""
        return (img_chw.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')

    def to_canvas(self, frames_u8, cmap: dict):
        """Model frames -> canvas: the rows covering the canvas, cv2.resize INTER_AREA; uncovered canvas rows black."""
        import cv2
        np = self.np
        (m0, m1), (c0, c1) = cmap['model_rows'], cmap['canvas_rows']
        out = np.zeros((len(frames_u8), CH, CW, 3), np.uint8)
        for i, f in enumerate(frames_u8):
            out[i, c0:c1] = cv2.resize(np.ascontiguousarray(f[m0:m1]), (CW, c1 - c0), interpolation=cv2.INTER_AREA)
        return out

    def z_img(self, z_tokens):
        """[b, L, c] latent tokens -> [b, h, w, c] (the demo's rearrange before z_to_image)."""
        return self.rearrange(z_tokens, 'b (h w) c -> b h w c', h=self.lat_h, w=self.lat_w)

    def roundtrip(self, frames_u8):
        """Per-frame DCAE round trip: encode_to_z on one frame, z_to_image on its latent (batch 1, is_video=False)."""
        torch, out = self.torch, []
        with torch.no_grad():
            for f in frames_u8:
                x = self.normalize([f]).unsqueeze(0).cuda()
                z = self.tokenizer.encode_to_z(x)
                out.append(self.to_u8(self.tokenizer.z_to_image(self.z_img(z[:, 0]))[0]))
        return self.np.stack(out)

    def encode_frames(self, frames_u8):
        """encode_to_z one frame at a time -> [F, L, c] latents."""
        torch = self.torch
        with torch.no_grad():
            return torch.cat([self.tokenizer.encode_to_z(self.normalize([f]).unsqueeze(0).cuda())[:, 0] for f in frames_u8], 0)

    def decode_groups(self, z_tokens):
        """The demo's decoding of generated frames: z_to_image on groups of 6 latents, is_video=False."""
        torch, out = self.torch, []
        with torch.no_grad():
            for g in range(0, z_tokens.shape[0], GROUP):
                imgs = self.tokenizer.z_to_image(self.z_img(z_tokens[g:g + GROUP]), is_video=False).cpu()
                out += [self.to_u8(im) for im in imgs]
        return self.np.stack(out)

    def condition_poses(self, mats_4x4):
        """test_free.py: get_rel_pose over the 10 condition poses (float32, on the GPU); entry 10 = copy of entry 9."""
        torch = self.torch
        abs_p = torch.from_numpy(self.np.asarray(mats_4x4, dtype=self.np.float64)).float().unsqueeze(0).cuda()
        pose_total, yaw_total = self.get_rel_pose(abs_p)
        pose = torch.cat([pose_total[:, :CF], pose_total[:, CF - 1:CF]], dim=1)
        yaw = torch.cat([yaw_total[:, :CF], yaw_total[:, CF - 1:CF]], dim=1)
        return pose, yaw

    @staticmethod
    def seed_all(seed: int, torch, np):
        """The demo's seeding block, per sample."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def rollout(self, hist_u8, mats_4x4, seed: int, decode_context: bool = True):
        """test_demo.py's test_sliding_window_img body for one sample: 30 self-predicted steps from the 10 condition frames.
        decode_context=False skips the demo's decoding of the 10 condition latents (a diagnostic here; it does not feed the
        rollout and is the same for every sample of a window) and returns None for it."""
        torch, np, model, tok, rearrange = self.torch, self.np, self.model, self.tokenizer, self.rearrange
        cond = hist_u8[N_HIST - CF:]                                   # ticks -18..0
        img = self.normalize(cond).unsqueeze(0).cuda()                 # [1, 10, 3, 512, 1024]
        pose, yaw = self.condition_poses(mats_4x4)
        cond_record = dict(dx_dy_m=pose[0].cpu().numpy().round(6).tolist(), dyaw_deg=yaw[0, :, 0].cpu().numpy().round(6).tolist())
        self.seed_all(int(seed), torch, np)
        self.calls.update(stt=0, traj_dit=0, vis_dit=0)
        timing = dict(steps=[])
        sink = io.StringIO()
        with torch.no_grad():
            model.eval()
            t = time.time()
            start_latents = tok.encode_to_z(img[:, :CF])
            context = [self.to_u8(tok.z_to_image(self.z_img(start_latents[:, j]))[0]) for j in range(CF)] if decode_context else None
            timing['encode_context'] = round(time.time() - t, 2)
            frames, trajs, first = [], [], []
            for t1 in range(0, N_FUT, GROUP):
                patch = []
                for t2 in range(GROUP):
                    ts = time.time()
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16), contextlib.redirect_stdout(sink):
                        predict_traj, predict_latents = model.step_eval(start_latents, pose, yaw, self_pred_traj=True)
                    predict_pose, predict_yaw = predict_traj[:, 0:1, 0:2], predict_traj[:, 0:1, 2:3]
                    pose = torch.cat((pose[:, 1:CF, ...], predict_pose, predict_pose), dim=1)
                    yaw = torch.cat((yaw[:, 1:CF, ...], predict_yaw, predict_yaw), dim=1)
                    trajs.append(predict_traj[0].float().cpu().numpy())
                    patch.append(predict_latents)
                    predict_latents = rearrange(predict_latents, '(b f) h w c -> b f (h w) c', f=1)
                    start_latents = torch.cat((start_latents[:, 1:CF, ...], predict_latents), dim=1)
                    torch.cuda.synchronize()
                    timing['steps'].append(round(time.time() - ts, 3))
                td = time.time()
                imgs = tok.z_to_image(torch.cat(patch, dim=0), is_video=False).cpu()
                frames += [self.to_u8(im) for im in imgs]
                timing.setdefault('decode', 0.0)
                timing['decode'] = round(timing['decode'] + time.time() - td, 2)
        traj = np.stack(trajs).astype(np.float32)                      # [30, 15, 3]
        counts = dict(stt_evals=self.calls['stt'], traj_dit_evals=self.calls['traj_dit'], vis_dit_evals=self.calls['vis_dit'])
        return np.stack(frames), (np.stack(context) if context is not None else None), traj, cond_record, counts, timing


    def step_eval_gen(self, latents, rel_pose, rel_yaw, gens):
        """models/model.py TrainTransformersDiT.step_eval(latents, rel_pose, rel_yaw, sample_last=True, self_pred_traj=True,
        traj_only=False), line for line (prints dropped), with one change: the initial noise of batch row b comes from
        gens[b], a CPU torch.Generator, drawn with the shapes a batch-1 call draws and in the same order (trajectory noise,
        then image noise), instead of one draw for the whole batch from the global CPU generator. Row b therefore gets
        exactly the noise that a batch-1 run seeded with gens[b]'s seed gets; with batch 1 the call reproduces step_eval
        bit for bit (checked by profile_epona.py)."""
        torch, m = self.torch, self.model
        with torch.no_grad():
            m.model.eval()
            pose_total = self.poses_to_indices(rel_pose, m.pose_x_vocab_size, m.pose_y_vocab_size)  # (b, t+1, 2)
            yaw_total = self.yaws_to_indices(rel_yaw, m.yaw_vocab_size)  # (b, t+1, 1)
            stt_features, pose_emb = m.model.evaluate(latents, pose_total, yaw_total, sample_last=True)
            bsz = stt_features.shape[0]
            if bsz != len(gens):
                raise ValueError(f'{bsz} batch rows, {len(gens)} generators')
            img_ids, cond_ids, traj_ids = self.prepare_ids(bsz, m.h, m.w, m.total_token_size, m.traj_len)
            m.traj_dit.eval()
            noise_traj = torch.cat([torch.randn(1, m.traj_len, m.traj_token_size, generator=g) for g in gens], 0).to(stt_features)
            timesteps_traj = self.get_schedule(int(m.args.num_sampling_steps), m.traj_len)
            predict_traj = m.traj_dit.sample(noise_traj, traj_ids, stt_features, cond_ids, timesteps_traj)
            predict_traj = m.denormalize_traj(predict_traj)
            predict_pose, predict_yaw = predict_traj[:, 0:1, 0:2], predict_traj[:, 0:1, 2:3]
            predict_pose = self.poses_to_indices(predict_pose, m.pose_x_vocab_size, m.pose_y_vocab_size)  # (b, 1, 2)
            predict_yaw = self.yaws_to_indices(predict_yaw, m.yaw_vocab_size)  # (b, 1, 1)
            pose_emb = m.model.get_pose_emb(predict_pose, predict_yaw)
            m.dit.eval()
            noise = torch.cat([torch.randn(1, m.img_token_size, m.vae_emb_dim, generator=g) for g in gens], 0).to(stt_features)
            timesteps = self.get_schedule(int(m.args.num_sampling_steps), m.img_token_size)
            predict_latents = m.dit.sample(noise, img_ids, stt_features, cond_ids, pose_emb, timesteps)
            predict_latents = self.rearrange(predict_latents, 'b (h w) c -> b h w c', h=m.h, w=m.w)
        return predict_traj, predict_latents

    def rollout_batch(self, hist_u8, mats_4x4, seeds, decode_context: bool = True):
        """rollout() for the samples of one window in one batch: the same history, poses and steps, one batch row per
        seed, the noise of row b from torch.Generator().manual_seed(seeds[b]) (step_eval_gen). The history is encoded
        once (the same call as rollout()); each row's latents are decoded in groups of 6 on their own, as rollout() does.
        Returns per-row lists (future frames, trajectories) and the shared context frames, conditioning, counts, timing.
        Checks that neither the global CPU nor the CUDA generator is consumed."""
        torch, np, tok, rearrange = self.torch, self.np, self.tokenizer, self.rearrange
        B = len(seeds)
        cond = hist_u8[N_HIST - CF:]                                   # ticks -18..0
        img = self.normalize(cond).unsqueeze(0).cuda()
        pose1, yaw1 = self.condition_poses(mats_4x4)
        cond_record = dict(dx_dy_m=pose1[0].cpu().numpy().round(6).tolist(), dyaw_deg=yaw1[0, :, 0].cpu().numpy().round(6).tolist())
        gens = [torch.Generator().manual_seed(int(s)) for s in seeds]
        cpu_state, cuda_state = torch.get_rng_state(), torch.cuda.get_rng_state()
        self.calls.update(stt=0, traj_dit=0, vis_dit=0)
        timing = dict(steps=[], decode=0.0)
        with torch.no_grad():
            self.model.eval()
            t = time.time()
            start1 = tok.encode_to_z(img[:, :CF])
            context = [self.to_u8(tok.z_to_image(self.z_img(start1[:, j]))[0]) for j in range(CF)] if decode_context else None
            timing['encode_context'] = round(time.time() - t, 2)
            start_latents = start1.repeat(B, 1, 1, 1)
            pose, yaw = pose1.repeat(B, 1, 1), yaw1.repeat(B, 1, 1)
            frames, trajs = [[] for _ in range(B)], [[] for _ in range(B)]
            for t1 in range(0, N_FUT, GROUP):
                patch = []
                for t2 in range(GROUP):
                    ts = time.time()
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        predict_traj, predict_latents = self.step_eval_gen(start_latents, pose, yaw, gens)
                    predict_pose, predict_yaw = predict_traj[:, 0:1, 0:2], predict_traj[:, 0:1, 2:3]
                    pose = torch.cat((pose[:, 1:CF, ...], predict_pose, predict_pose), dim=1)
                    yaw = torch.cat((yaw[:, 1:CF, ...], predict_yaw, predict_yaw), dim=1)
                    tr = predict_traj.float().cpu().numpy()
                    for b in range(B):
                        trajs[b].append(tr[b])
                    patch.append(predict_latents)
                    predict_latents = rearrange(predict_latents, '(b f) h w c -> b f (h w) c', f=1)
                    start_latents = torch.cat((start_latents[:, 1:CF, ...], predict_latents), dim=1)
                    torch.cuda.synchronize()
                    timing['steps'].append(round(time.time() - ts, 3))
                td = time.time()
                for b in range(B):
                    imgs = tok.z_to_image(torch.cat([p[b:b + 1] for p in patch], dim=0), is_video=False).cpu()
                    frames[b] += [self.to_u8(im) for im in imgs]
                timing['decode'] = round(timing['decode'] + time.time() - td, 2)
        if not (torch.equal(cpu_state, torch.get_rng_state()) and torch.equal(cuda_state, torch.cuda.get_rng_state())):
            raise RuntimeError('the batched rollout consumed the global CPU or CUDA generator')
        counts = dict(stt_evals=self.calls['stt'], traj_dit_evals=self.calls['traj_dit'], vis_dit_evals=self.calls['vis_dit'], batch=B)
        return ([np.stack(f) for f in frames], (np.stack(context) if context is not None else None), [np.stack(t).astype(np.float32) for t in trajs],
                cond_record, counts, timing)


# ----------------------------------------------------------------------------------------------------------------------
def software(model: Epona | None) -> dict:
    out = dict(python=platform.python_version(), ffmpeg=ffmpeg_version(), adapter_sha256=sha256_file(Path(__file__)), pins_sha256=sha256_file(PINS_PATH),
               epona_commit=PINS['code']['commit'], hf_revision=PINS['weights']['hf_revision'])
    try:
        import numpy, torch, cv2, einops, PIL, scipy
        out.update(torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(), numpy=numpy.__version__, opencv=cv2.__version__,
                   einops=einops.__version__, pillow=PIL.__version__, scipy=scipy.__version__)
        import triton
        out['triton'] = triton.__version__
    except Exception:  # noqa: BLE001
        pass
    if model is not None:
        out['epona_source_digest'] = model.digest
    for key in ('CUDA_VISIBLE_DEVICES', 'APPTAINER_CONTAINER', 'SINGULARITY_CONTAINER', 'WCB_IMAGE', '_CONDOR_JOB_AD'):
        if os.environ.get(key):
            out.setdefault('env', {})[key] = os.environ[key]
    return out


def gpu_info(torch) -> dict:
    """The GPU this process uses (device 0 under CUDA_VISIBLE_DEVICES), its driver and CUDA versions."""
    if not torch.cuda.is_available():
        return dict(name=None)
    p = torch.cuda.get_device_properties(0)
    info = dict(name=p.name, capability=f'{p.major}.{p.minor}', total_gib=round(p.total_memory / 2**30, 2), torch_device_count=torch.cuda.device_count(),
                torch=torch.__version__, cuda_runtime=torch.version.cuda, cudnn=torch.backends.cudnn.version())
    uuid = getattr(p, 'uuid', None)
    if uuid is not None:
        info.update(uuid=str(uuid) if str(uuid).startswith('GPU-') else 'GPU-' + str(uuid), uuid_source='torch')
    try:  # cuDriverGetVersion: the newest CUDA version the installed driver supports
        import ctypes
        v = ctypes.c_int()
        if ctypes.CDLL('libcuda.so.1').cuDriverGetVersion(ctypes.byref(v)) == 0:
            info['cuda_driver'] = f'{v.value // 1000}.{(v.value % 1000) // 10}'
    except Exception:  # noqa: BLE001
        pass
    sel = (os.environ.get('CUDA_VISIBLE_DEVICES') or '0').split(',')[0]
    try:
        out = subprocess.check_output(['nvidia-smi', '-i', sel, '--query-gpu=name,uuid,driver_version,memory.total,pci.bus_id', '--format=csv,noheader'],
                                      text=True, stderr=subprocess.DEVNULL).strip().splitlines()[0]
        name, uuid, driver, mem, bus = [x.strip() for x in out.split(',')]
        info.update(nvidia_smi_name=name, uuid=uuid, uuid_source='nvidia-smi', driver=driver, driver_source='nvidia-smi',
                    nvidia_smi_memory=mem, pci_bus_id=bus, nvidia_smi_selector=sel)
    except Exception:  # noqa: BLE001
        pass
    # cluster containers have no nvidia-smi (the same gap 974e1e8 and 4cb49f3 closed for Orbis 2 and LTX-2.5). The UUID then
    # comes from torch or from CUDA_VISIBLE_DEVICES, which batch-system sets to the assigned GPU's UUID; the driver version
    # from NVML (libnvidia-ml.so.1, bound into the container with the driver), else from the kernel module.
    if 'uuid' not in info and sel.startswith('GPU-'):
        info.update(uuid=sel, uuid_source='CUDA_VISIBLE_DEVICES')
    if 'driver' not in info:
        try:
            import ctypes
            nvml = ctypes.CDLL('libnvidia-ml.so.1')
            if nvml.nvmlInit_v2() == 0:
                try:
                    buf = ctypes.create_string_buffer(96)
                    if nvml.nvmlSystemGetDriverVersion(buf, ctypes.c_uint(len(buf))) == 0 and buf.value:
                        info.update(driver=buf.value.decode(), driver_source='nvml')
                finally:
                    nvml.nvmlShutdown()
        except Exception:  # noqa: BLE001
            pass
    if 'driver' not in info:
        try:
            import re
            first = Path('/proc/driver/nvidia/version').read_text().splitlines()[0]  # "NVRM version: ... Kernel Module  570.195.03  ..."
            m = re.search(r'\s(\d+\.\d+(?:\.\d+)?)(?:\s|$)', first)
            if m:
                info.update(driver=m.group(1), driver_source='/proc/driver/nvidia/version')
        except Exception:  # noqa: BLE001
            pass
    return info


def gpu_used_by_others_mib(sel: str) -> float | None:
    try:
        out = subprocess.check_output(['nvidia-smi', '-i', sel, '--query-gpu=memory.used,memory.free', '--format=csv,noheader,nounits'], text=True)
        used, free = [float(x) for x in out.strip().split(',')]
        return dict(used_mib=used, free_mib=free)
    except Exception:  # noqa: BLE001
        return None


def host_memory() -> dict:
    """Peak and current resident memory of this process (GiB) and the host's MemAvailable."""
    import resource
    out = dict(peak_rss=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2))
    try:
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith(('VmRSS:', 'RssAnon:', 'RssFile:', 'VmHWM:')):
                key, val = line.split(':')
                out[key.lower()] = round(int(val.split()[0]) / 2**20, 2)
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemAvailable:'):
                out['host_mem_available'] = round(int(line.split()[1]) / 2**20, 2)
    except OSError:
        pass
    return out


def mem_available_gib() -> float:
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) / 2**20
    return float('nan')


def load_poses(path: Path, np):
    pp = json.loads(path.read_text())
    if pp['ticks'] != HIST_TICKS or max(r['k'] for r in pp['poses']) > 0:
        raise ValueError(f'{path}: pose ticks {pp["ticks"]}, expected {HIST_TICKS} (past only)')
    by_k = {r['k']: np.array(r['matrix_4x4'], dtype=np.float64) for r in pp['poses']}
    return np.stack([by_k[k] for k in PINS['timing']['pose_ticks']])


def make_units(specs: list, todo: list, batch_k: int) -> list:
    """Work units in spec order: each real reference alone; continuation samples alone (batch_k < 2, the released
    per-sample path) or, with batch_k >= 2, in fixed chunks per window: the window's continuation specs sorted by k, cut
    into consecutive groups of batch_k. The chunk layout depends only on the spec set, so a resumed run rebuilds the same
    batches; a chunk runs again whole if any member is unfinished (its finished members are regenerated, checked for bit
    identity against their records and not rewritten)."""
    todo_names = {s['name'] for s in todo}
    by_window: dict = {}
    for s in specs:
        by_window.setdefault((s['split'], s['window']), []).append(s)
    units = []
    for key in sorted(by_window):
        ws = by_window[key]
        units += [[s] for s in ws if s['mode'] == 'recon' and s['name'] in todo_names]
        cont = sorted((s for s in ws if s['mode'] == 'continue'), key=lambda s: s['k'])
        if batch_k >= 2:
            for i in range(0, len(cont), batch_k):
                chunk = cont[i:i + batch_k]
                if any(c['name'] in todo_names for c in chunk):
                    units.append(chunk)
        else:
            units += [[s] for s in cont if s['name'] in todo_names]
    return units


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--specs', nargs='+', required=True, help='spec JSON files or directories (searched recursively)')
    ap.add_argument('--input-root', type=Path, required=True)
    ap.add_argument('--output-root', type=Path, required=True)
    ap.add_argument('--models-dir', type=Path, default=os.environ.get('EPONA_MODELS_DIR'), help='holds epona_nuplan+china.pkl and dcae_td_20000.pkl')
    ap.add_argument('--epona-src', type=Path, default=os.environ.get('EPONA_SRC'), help='Epona source at the pinned commit')
    ap.add_argument('--resume', action='store_true', help='skip samples already finished (status success, video sha256 matches)')
    ap.add_argument('--dry-run', action='store_true', help='check the specs and inputs, run nothing')
    ap.add_argument('--min-free-mib', type=int, default=0, help='refuse to start if the GPU has less free memory (shared GPUs)')
    ap.add_argument('--max-gpu-mib', type=int, default=0, help='cap this process at N MiB (torch.cuda.set_per_process_memory_fraction)')
    ap.add_argument('--leave-free-mib', type=int, default=0, help='cap this process so that at least N MiB (+1 GiB for the CUDA context) '
                    'of the GPU memory free at start stays free; combined with --max-gpu-mib by taking the smaller cap')
    ap.add_argument('--min-host-avail-gib', type=float, default=0.0, help='refuse to start if MemAvailable is lower')
    ap.add_argument('--expect-gpu-uuid', default=None, help='refuse to start unless the visible GPU has this UUID')
    ap.add_argument('--verify-weights', choices=('size', 'sha256'), default='size', help='checkpoint check before loading')
    ap.add_argument('--save-native', action='store_true', help='also write native.mp4 (1024x512)')
    ap.add_argument('--threads', type=int, default=int(os.environ.get('OMP_NUM_THREADS', '2')), help='CPU threads for torch and ffmpeg')
    ap.add_argument('--allow-unpinned', action='store_true', help='development only: accept other code or weights (recorded)')
    ap.add_argument('--batch-k', type=int, default=0, help='>= 2: generate the continuation samples of a window in batches of this size '
                    '(rollout_batch: one torch.Generator per sample seeded with its own seed; the layout is recorded per sample). '
                    'Default: one sample per call on the released path (model.step_eval)')
    ap.add_argument('--dcae-only', action='store_true', help='real references only: do not load the world model (about 3-5 GB of GPU '
                    'memory instead of 17); refused if a continuation spec is to run')
    ap.add_argument('--recon-prefix-root', type=Path, default=None, help='TEST real references after freezing: take frames 0-10 from a '
                    'finished continuation of the same window under this collected run root (cluster outputs), so that all 17 clips '
                    'of a window share frames 0-10 exactly; the local round trip is compared and recorded')
    ap.add_argument('--writer-preset', default='medium', choices=('medium', 'veryfast', 'ultrafast'),
                    help='x264 preset of the lossless writer (-crf 0 is lossless with any preset; medium is the bench encoder)')
    args = ap.parse_args()
    if args.models_dir is None or args.epona_src is None:
        raise SystemExit('--models-dir and --epona-src (or $EPONA_MODELS_DIR, $EPONA_SRC) are required')
    input_root, output_root = args.input_root.resolve(), args.output_root.resolve()

    specs = load_specs(args.specs)
    if not specs:
        raise SystemExit('no specs found')
    for s in specs:
        check_spec(s, input_root)
    bound = {s['generation'].get('batch_k') for s in specs if s['mode'] == 'continue'}
    if len(bound) > 1:
        raise SystemExit(f'the specs mix batch layouts {sorted(bound, key=str)}: never mix layouts within one run')
    bound = next(iter(bound)) if bound else None
    if bound is not None and args.batch_k != bound:
        raise SystemExit(f'the continuation specs are bound to --batch-k {bound} (generation.batch_k); got --batch-k {args.batch_k}')
    todo = [s for s in specs if not (args.resume and is_done(s, output_root))]
    if args.dcae_only and any(s['mode'] != 'recon' for s in todo):
        raise SystemExit('--dcae-only runs real references only')
    if not args.resume:
        clash = [s['name'] for s in specs if (output_root / s['output_dir'] / 'sample_outputs.json').exists()]
        if clash:
            raise SystemExit(f'{len(clash)} samples already have outputs (first: {clash[0]}); use --resume or a fresh output root')
    print(json.dumps(dict(specs=len(specs), to_run=len(todo), windows=len({s['window'] for s in specs}))), flush=True)
    if args.dry_run or not todo:
        return

    avail = mem_available_gib()
    if avail < args.min_host_avail_gib:
        raise SystemExit(f'host MemAvailable {avail:.1f} GiB < --min-host-avail-gib {args.min_host_avail_gib}')
    os.environ.setdefault('OMP_NUM_THREADS', str(args.threads))
    import numpy as np
    import torch
    torch.set_num_threads(args.threads)
    if not torch.cuda.is_available():
        raise SystemExit('CUDA is not available')
    gpu = gpu_info(torch)
    if args.expect_gpu_uuid and gpu.get('uuid') != args.expect_gpu_uuid:
        raise SystemExit(f'visible GPU {gpu.get("uuid")} is not the expected {args.expect_gpu_uuid}')
    free, total = torch.cuda.mem_get_info()
    if free / 2**20 < args.min_free_mib:
        raise SystemExit(f'GPU has {free / 2**20:.0f} MiB free, --min-free-mib {args.min_free_mib}')
    caps = [c for c in (args.max_gpu_mib, (free / 2**20 - args.leave_free_mib - 1024) if args.leave_free_mib else 0) if c]
    cap_mib = int(min(caps)) if caps else 0
    if cap_mib:
        if cap_mib < max(args.min_free_mib - args.leave_free_mib, 1024):
            raise SystemExit(f'GPU cap {cap_mib} MiB is too small (free {free / 2**20:.0f} MiB, leave {args.leave_free_mib} MiB)')
        torch.cuda.set_per_process_memory_fraction(min(1.0, cap_mib * 2**20 / total), 0)
    sel = (os.environ.get('CUDA_VISIBLE_DEVICES') or '0').split(',')[0]
    run_log = dict(schema='wm-calib-bench.epona.adapter_run.v1', started_utc=now_utc(), host=socket.gethostname(), argv=sys.argv,
                   input_root=str(input_root), output_root=str(output_root), specs=len(specs), to_run=len(todo), host_memory_start=host_memory(),
                   gpu_free_at_start_mib=round(free / 2**20), gpu_cap_mib=cap_mib or None, gpu_smi_start=gpu_used_by_others_mib(sel))
    model = Epona(args.models_dir.resolve(), args.epona_src.resolve(), args.verify_weights, args.allow_unpinned, args.threads,
                  world_model=not args.dcae_only)
    sw = software(model)
    run_log.update(model_load_s=model.load_s, settings=model.settings, weights=model.weight_checks, software=sw, gpu=gpu, samples=[],
                   host_memory_after_load=host_memory(), gpu_memory_after_load_gib=round(torch.cuda.memory_allocated() / 2**30, 2),
                   gpu_smi_after_load=gpu_used_by_others_mib(sel))
    log_dir = output_root / 'adapter-runs'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{datetime.now().strftime('%Y%m%dT%H%M%S')}-{socket.gethostname()}-{os.getpid()}.json"
    write_json_atomic(log_path, run_log)
    print(json.dumps(dict(model_load_s=model.load_s, gpu=gpu.get('name'), host_peak_rss_gib=run_log['host_memory_after_load'].get('peak_rss'))), flush=True)

    prefixes: dict = {}      # window -> (11 round-tripped native history frames, canvas frames, sha256 of the canvas frames)
    recon_prefix: dict = {}  # window -> sha256 of the real reference's canvas frames 0-10 (this process)
    context_diag: dict = {}  # window -> model_context_vs_prefix_native (the same for every sample of a window)
    failures = 0
    batch_k = args.batch_k if args.batch_k >= 2 else 0
    run_log['batch_k'] = batch_k or None
    run_log['writer_preset'] = args.writer_preset
    run_log['dcae_only'] = bool(args.dcae_only)
    run_log['recon_prefix_root'] = str(args.recon_prefix_root) if args.recon_prefix_root else None
    step = lambda a, b: float(np.abs(a.astype(np.int16) - b).mean())  # noqa: E731

    def base_rec(spec):
        return dict(schema=OUT_SCHEMA, name=spec['name'], window=spec['window'], split=spec['split'], mode=spec['mode'], k=spec.get('k'),
                    seed=spec['seed'], run=spec.get('run'), spec=spec['_file'], spec_sha256=spec['_sha256'], started_utc=now_utc(),
                    host=socket.gethostname(), gpu=gpu, software=sw, generation=spec['generation'], settings_read_back=model.settings,
                    frame_ticks_k=TICKS, frame_times_s=[round(k * 0.1, 1) for k in TICKS], fps=FPS, encoder_preset=args.writer_preset)

    def finish(spec, rec, clip, native, t_start, out_dir, traj=None):
        """Write and verify the clip(s) of one sample, then its record (the record is written last: resumability)."""
        nonlocal failures
        try:
            if clip is None:
                raise RuntimeError(rec.get('error') or 'no clip')
            if clip.shape != (OUT_FRAMES, CH, CW, 3):
                raise RuntimeError(f'clip has shape {clip.shape}')
            rec.setdefault('prefix11_frames_sha256', sha256_bytes(clip[:N_HIST].tobytes()))
            if spec['mode'] == 'continue' and spec['window'] in recon_prefix:
                rec['prefix_identical_to_recon_in_process'] = rec['prefix11_frames_sha256'] == recon_prefix[spec['window']]
                if not rec['prefix_identical_to_recon_in_process']:
                    raise RuntimeError('history prefix differs from the real reference frames 0-10')
            t3 = time.time()
            if traj is not None:
                np.save(out_dir / 'pred_traj.npy', traj)
            video = out_dir / 'vision.mp4'
            write_lossless(clip, video, args.threads, args.writer_preset)
            back, vinfo = decode_rgb(video, args.threads)
            if not np.array_equal(back, clip):
                raise RuntimeError('vision.mp4 does not decode to the written frames')
            if args.save_native:
                write_lossless(native, out_dir / 'native.mp4', args.threads, args.writer_preset)
                rec['native'] = dict(video='native.mp4', sha256=sha256_file(out_dir / 'native.mp4'), frames_sha256=sha256_bytes(native.tobytes()))
            rec.setdefault('timing_s', {})['write'] = round(time.time() - t3, 2)
            rec.update(status='success', video='vision.mp4', video_sha256=sha256_file(video), video_bytes=video.stat().st_size,
                       frames=vinfo['frames'], width=vinfo['width'], height=vinfo['height'], fps=vinfo['fps'], pix_fmt=vinfo['pix_fmt'],
                       frames_sha256=sha256_bytes(clip.tobytes()))
        except Exception as exc:  # noqa: BLE001
            failures += 1
            rec.update(status='failure', error=rec.get('error') or f'{type(exc).__name__}: {exc}', traceback=rec.get('traceback') or traceback.format_exc())
            print(rec['traceback'], file=sys.stderr, flush=True)
        rec['gpu_memory_gib'] = dict(max_allocated=round(torch.cuda.max_memory_allocated() / 2**30, 2), max_reserved=round(torch.cuda.max_memory_reserved() / 2**30, 2))
        rec['gpu_smi'] = gpu_used_by_others_mib(sel)
        rec['host_memory_gib'] = host_memory()
        rec['timing_s'] = dict(rec.get('timing_s', {}), total=round(time.time() - t_start, 2))
        rec['finished_utc'] = now_utc()
        write_json_atomic(out_dir / 'sample_outputs.json', rec)
        run_log['samples'].append(dict(name=spec['name'], status=rec['status'], timing_s=rec['timing_s'], gpu_memory_gib=rec.get('gpu_memory_gib'),
                                       host_peak_rss_gib=rec['host_memory_gib'].get('peak_rss'), batch=rec.get('batch', {}).get('size')))
        write_json_atomic(log_path, run_log)
        print(json.dumps(dict(sample=spec['name'], status=rec['status'], **{k: v for k, v in rec['timing_s'].items() if not isinstance(v, list)})), flush=True)

    def run_recon(spec):
        out_dir = output_root / spec['output_dir']
        out_dir.mkdir(parents=True, exist_ok=True)
        rec, t_start, clip, native = base_rec(spec), time.time(), None, None
        try:
            cmap = canvas_map(spec['_band'])
            rec['canvas_map'] = cmap
            torch.cuda.reset_peak_memory_stats()
            src = input_root / spec['inputs']['real']['path']
            frames, info = decode_rgb(src, args.threads)
            if frames.shape != (OUT_FRAMES, MH, MW, 3) or abs(info['fps'] - FPS) > 1e-6:
                raise ValueError(f'real clip {src}: {frames.shape} at {info["fps"]} fps')
            rec['input'] = dict(path=spec['inputs']['real']['path'], file_sha256=sha256_file(src), frames_sha256=sha256_bytes(frames.tobytes()))
            if spec['inputs']['real'].get('sha256') and rec['input']['file_sha256'] != spec['inputs']['real']['sha256']:
                raise ValueError('real clip sha256 differs from the spec')
            t0 = time.time()
            head = model.roundtrip(frames[:N_HIST])
            t1 = time.time()
            z = model.encode_frames(frames[N_HIST:])
            tail = model.decode_groups(z)
            t2 = time.time()
            with torch.no_grad():   # diagnostic: the same latents decoded one at a time
                single = np.stack([model.to_u8(model.tokenizer.z_to_image(model.z_img(z[i:i + 1]))[0]) for i in range(z.shape[0])])
            native = np.concatenate([head, tail], axis=0)
            rec['timing_s'] = dict(roundtrip_history=round(t1 - t0, 2), encode_decode_future=round(t2 - t1, 2))
            d = np.abs(head.astype(np.int16) - frames[:N_HIST])
            dg = np.abs(tail.astype(np.int16) - single)
            rec['diagnostics'] = dict(roundtrip_vs_input_history_native=dict(frames=[0, N_HIST - 1], max_abs=int(d.max()), mean_abs=float(d.mean())),
                                      decode_group6_vs_single=dict(max_abs=int(dg.max()), mean_abs=float(dg.mean()), pixels_differing=int((dg.max(axis=-1) > 0).sum())))
            clip = model.to_canvas(native, cmap)
            if args.recon_prefix_root:
                rec['prefix_source'] = external_prefix(spec['window'], clip)
            recon_prefix[spec['window']] = sha256_bytes(clip[:N_HIST].tobytes())
        except Exception as exc:  # noqa: BLE001
            rec.update(error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
        finish(spec, rec, clip, native, t_start, out_dir)

    def external_prefix(window, clip):
        """Replace clip frames 0-10 (in place) by those of a finished continuation of the window under --recon-prefix-root."""
        cands = []
        for f in sorted(args.recon_prefix_root.glob(f'*/native/{window}-s*/sample_outputs.json')):
            r = json.loads(f.read_text())
            if r.get('status') == 'success' and r.get('window') == window and (f.parent / 'vision.mp4').is_file():
                cands.append((r.get('k', 99), f.parent, r))
        if not cands:
            raise FileNotFoundError(f'no finished continuation of {window} under {args.recon_prefix_root}')
        k, d, r = sorted(cands, key=lambda c: c[0])[0]
        ext, _ = decode_rgb(d / 'vision.mp4', args.threads)
        ext = ext[:N_HIST]
        if sha256_bytes(ext.tobytes()) != r.get('prefix11_frames_sha256'):
            raise RuntimeError(f'{d}: frames 0-10 do not match the recorded prefix hash')
        diff = np.abs(ext.astype(np.int16) - clip[:N_HIST])
        out = dict(sample=r['name'], path=str(d), prefix11_frames_sha256=r['prefix11_frames_sha256'], gpu=r.get('gpu', {}).get('name'),
                   local_vs_external=dict(identical=bool(diff.max() == 0), max_abs=int(diff.max()), mean_abs=float(diff.mean())),
                   note='frames 0-10 are the continuations\' own prefix (computed on their GPU); frames 11-40 are this process\'s round trip')
        clip[:N_HIST] = ext
        return out

    def run_chunk(chunk):
        """One continuation sample (released path) or one batch of samples of the same window (rollout_batch)."""
        batched = batch_k >= 2
        t_start = time.time()
        recs = {s['name']: base_rec(s) for s in chunk}
        done = {s['name'] for s in chunk if is_done(s, output_root)}
        results, err = {}, None
        try:
            spec0 = chunk[0]
            if len({(s['window'], s['inputs']['history']['path'], s['inputs']['poses']['path'], json.dumps(s['generation'], sort_keys=True), tuple(s['_band'])) for s in chunk}) != 1:
                raise ValueError('a batch must share one window, its inputs and the generation settings')
            cmap = canvas_map(spec0['_band'])
            torch.cuda.reset_peak_memory_stats()
            src = input_root / spec0['inputs']['history']['path']
            psrc = input_root / spec0['inputs']['poses']['path']
            frames, info = decode_rgb(src, args.threads)
            if frames.shape != (N_HIST, MH, MW, 3) or abs(info['fps'] - FPS) > 1e-6:
                raise ValueError(f'history clip {src}: {frames.shape} at {info["fps"]} fps')
            inp = dict(path=spec0['inputs']['history']['path'], file_sha256=sha256_file(src), frames_sha256=sha256_bytes(frames.tobytes()),
                       poses_path=spec0['inputs']['poses']['path'], poses_sha256=sha256_file(psrc))
            for s in chunk:
                if s['inputs']['history'].get('sha256') and inp['file_sha256'] != s['inputs']['history']['sha256']:
                    raise ValueError(f'{s["name"]}: history clip sha256 differs from the spec')
                if s['inputs']['poses'].get('sha256') and inp['poses_sha256'] != s['inputs']['poses']['sha256']:
                    raise ValueError(f'{s["name"]}: pose file sha256 differs from the spec')
            mats = load_poses(psrc, np)
            t0 = time.time()
            if spec0['window'] not in prefixes:
                p = model.roundtrip(frames)
                pc = model.to_canvas(p, cmap)
                prefixes[spec0['window']] = (p, pc, sha256_bytes(pc.tobytes()))
            prefix_native, prefix_canvas, prefix_sha = prefixes[spec0['window']]
            t1 = time.time()
            need_ctx = spec0['window'] not in context_diag
            if batched:
                futures, context, trajs, cond, counts, steps = model.rollout_batch(frames, mats, [s['seed'] for s in chunk], decode_context=need_ctx)
            else:
                future, context, traj, cond, counts, steps = model.rollout(frames, mats, spec0['seed'], decode_context=need_ctx)
                futures, trajs = [future], [traj]
            t2 = time.time()
            if need_ctx:
                d = np.abs(context.astype(np.int16) - prefix_native[N_HIST - CF:])
                context_diag[spec0['window']] = dict(frames=list(range(N_HIST - CF, N_HIST)), max_abs=int(d.max()), mean_abs=float(d.mean()),
                                                     pixels_differing=int((d.max(axis=-1) > 0).sum()))
            step_s = steps.pop('steps')
            B = len(chunk)
            for i, s in enumerate(chunk):
                traj = trajs[i]
                native = np.concatenate([prefix_native, futures[i]], axis=0) if args.save_native else None
                clip = np.concatenate([prefix_canvas, model.to_canvas(futures[i], cmap)], axis=0)
                rec = recs[s['name']]
                rec.update(canvas_map=cmap, input=inp, prefix11_frames_sha256=prefix_sha)
                rec['rollout'] = dict(counts, condition_poses=cond,
                                      first_step_pred=dict(dx_m=traj[:, 0, 0].round(4).tolist(), dy_m=traj[:, 0, 1].round(4).tolist(),
                                                           dyaw_deg=traj[:, 0, 2].round(4).tolist()),
                                      pred_speed_mps=dict(first=round(float(traj[0, 0, 0] / 0.2), 3), mean=round(float(traj[:, 0, 0].mean() / 0.2), 3)),
                                      pred_traj_file='pred_traj.npy')
                rec['batch'] = (dict(size=B, position=i, members=[c['name'] for c in chunk], ks=[c['k'] for c in chunk], batch_k=batch_k,
                                     noise='row b: torch.Generator().manual_seed(seed of member b); trajectory then image noise per frame, the shapes of a batch-1 call',
                                     note='numerics depend on the batch layout; a rerun with the same layout is bit-identical on the same GPU type')
                                if batched else dict(size=1, position=0, members=[s['name']], ks=[s['k']], batch_k=None,
                                                     noise='global CPU generator seeded per sample (the released step_eval)'))
                rec['timing_s'] = dict(prefix=round(t1 - t0, 2), rollout=round(t2 - t1, 2), rollout_per_sample=round((t2 - t1) / B, 2),
                                       per_frame_mean=round(sum(step_s) / len(step_s), 3), per_frame_max=max(step_s), **steps)
                rec['diagnostics'] = dict(
                    model_context_vs_prefix_native=context_diag[spec0['window']],
                    step_mean_abs_canvas=dict(f9_f10=step(clip[9], clip[10]), f10_f11=step(clip[10], clip[11]), f11_f12=step(clip[11], clip[12]),
                                              future_mean=float(np.mean([step(clip[j], clip[j + 1]) for j in range(N_HIST, OUT_FRAMES - 1)])),
                                              history_mean=float(np.mean([step(clip[j], clip[j + 1]) for j in range(0, N_HIST - 1)]))))
                results[s['name']] = (clip, native, traj)
        except Exception as exc:  # noqa: BLE001
            err = (f'{type(exc).__name__}: {exc}', traceback.format_exc())
        for s in chunk:
            out_dir = output_root / s['output_dir']
            out_dir.mkdir(parents=True, exist_ok=True)
            rec = recs[s['name']]
            if s['name'] in done:
                old = json.loads((out_dir / 'sample_outputs.json').read_text())
                same = s['name'] in results and sha256_bytes(results[s['name']][0].tobytes()) == old.get('frames_sha256')
                print(json.dumps(dict(sample=s['name'], status='kept', regenerated_identical=same)), flush=True)
                run_log.setdefault('kept', []).append(dict(name=s['name'], regenerated_identical=same))
                if not same and s['name'] in results:
                    raise SystemExit(f'{s["name"]}: regenerated batch differs from the recorded sample (layout or determinism changed)')
                continue
            if err:
                rec.update(error=err[0], traceback=err[1])
                finish(s, rec, None, None, t_start, out_dir)
            else:
                clip, native, traj = results[s['name']]
                finish(s, rec, clip, native, t_start, out_dir, traj=traj)

    for unit in make_units(specs, todo, batch_k):
        if unit[0]['mode'] == 'recon':
            run_recon(unit[0])
        else:
            run_chunk(unit)
    run_log.update(finished_utc=now_utc(), failures=failures, host_memory_end=host_memory(),
                   gpu_memory_peak_gib=max((x['gpu_memory_gib'] or {}).get('max_allocated', 0) for x in run_log['samples']) if run_log['samples'] else None)
    write_json_atomic(log_path, run_log)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
