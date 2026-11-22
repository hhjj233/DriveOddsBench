#!/usr/bin/env python3
"""Orbis 2 adapter for the wm-calib-bench calibration benchmark (bench-v1).

Reads spec JSONs (one per sample; see make_orbis2_specs.py) and writes, per sample,
  <output-root>/<spec.output_dir>/vision.mp4           81 frames, 512x288, 10 fps, lossless (libx264rgb -crf 0, rgb24)
  <output-root>/<spec.output_dir>/sample_outputs.json  status (success / failure), every setting, timings, hashes

Modes (spec "mode"):
  continue  The 21-frame 10 Hz history clip (t0 - 2.0 .. t0) conditions the released Orbis 2 rollout: L1 context = history
            frames 16-20 (10 Hz), L2 context = frames 0, 5, 10, 15, 20 (2 Hz), 12 rollout steps of 5 frames = 60 frames
            (t0 + 0.1 .. t0 + 6.0 s), no steering (the framework's all-NaN placeholder). vision.mp4 = frames 0-20: the
            per-frame tokenizer round trip of the 21 history frames (the same function as the real reference, so these 21
            frames are identical to the reference's), then frames 21-80: the 60 rolled-out frames. The spec references
            the history clip only; the real future is never read.
  recon     The real reference: the 81-frame real clip through the same tokenizer, frame by frame (rec branch:
            encode -> x enc_scale -> / enc_scale -> decode), under the same fp16 autocast as the rollout.

The model is built exactly as Orbis 2's evaluate/rollout_demo_v2.py builds it (released L1/config.yaml, standard L2,
EMA weights, fp16 autocast, --l1_nfe 30 passed and ignored by the configured Heun+Euler schedule, L2 at its configured
20 NFE). Differences from the demo, none of which changes the sampled distribution:
  - the model is loaded once per process; each sample re-seeds (pytorch_lightning.seed_everything(seed)) right before
    its rollout, which leaves the CUDA generator in the same state as the demo's start-of-run seeding (nothing between
    the demo's seeding and its rollout draws from the CUDA generator);
  - input frames are decoded with ffmpeg to rgb24 (the demo uses decord), then go through the demo's own
    ClipAugmenter(DecordFrameAdapter(), ResizeCenterPolicy((288, 512))), which is the identity on 512x288 frames;
  - output frames are written losslessly instead of as JPEG/GIF; uint8 = clamp(round((x + 1) / 2 * 255)) in float32.

Resumable: a sample whose sample_outputs.json says success and whose vision.mp4 matches the recorded sha256 is skipped.
A failed sample is recorded with status "failure" and the run continues; the exit code is 1 if any sample failed.

  run_orbis2_bench.py --specs DIR_OR_JSON [...] --input-root DIR --output-root DIR [--models-dir DIR] [--orbis2-src DIR]
                      [--resume] [--dry-run] [--min-free-mib N] [--verify-weights none|size|sha256]
Run it as a script path (not with python -m): Orbis 2's top-level packages (models, data, util, ...) must win the imports.
"""
from __future__ import annotations

import argparse
import hashlib
import json
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
PINS_PATH = HERE / 'orbis2_pins.json'
PINS = json.loads(PINS_PATH.read_text())
DEF = PINS['released_defaults']
H, W = DEF['image_size_hw']
HIST, OUT_FRAMES, FPS = DEF['history_frames'], PINS['output']['frames'], PINS['output']['fps']
SPEC_SCHEMA = 'wm-calib-bench.orbis2.spec.v1'
OUT_SCHEMA = 'wm-calib-bench.orbis2.sample_outputs.v1'


# ----------------------------------------------------------------------------------------------------------------------
# small helpers (no torch)
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def write_json_atomic(path: Path, obj) -> None:
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=1) + '\n')
    os.replace(tmp, path)


def source_digest(root: Path) -> str:
    """The pins' rule: sha256 over the sorted *.py / *.yml / *.yaml files of the Orbis 2 source tree."""
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
    frames = np.frombuffer(raw, np.uint8).reshape(-1, info['height'], info['width'], 3).copy()  # writable for torch.from_numpy
    if frames.shape[0] != info['frames']:
        raise RuntimeError(f'{path}: decoded {frames.shape[0]} frames, ffprobe counted {info["frames"]}')
    return frames, info


def write_lossless(frames, path: Path, threads: int) -> None:
    """make_wm_calibration_windows.write_video's encoder (libx264rgb -crf 0 -preset medium, rgb24, 10 fps), plus -threads."""
    import numpy as np
    h, w = frames.shape[1:3]
    tmp = path.with_name(path.stem + '.part.mp4')
    proc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}', '-r', str(FPS), '-i', '-',
                             '-c:v', 'libx264rgb', '-crf', '0', '-preset', 'medium', '-threads', str(threads), '-pix_fmt', 'rgb24', '-r', str(FPS),
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
    # per window: the real reference first, then the samples by k
    specs.sort(key=lambda s: (s['split'], s['window'], 0 if s['mode'] == 'recon' else 1, s.get('k') or 0))
    return specs


def check_spec(spec: dict, input_root: Path) -> None:
    for key in ('name', 'window', 'split', 'mode', 'seed', 'inputs', 'output_dir', 'generation'):
        if key not in spec:
            raise ValueError(f'spec {spec.get("name")}: missing {key}')
    if spec['mode'] == 'continue':
        if set(spec['inputs']) != {'history'}:
            raise ValueError(f'{spec["name"]}: a continue spec must reference the history clip only, got {sorted(spec["inputs"])}')
    elif spec['mode'] == 'recon':
        if set(spec['inputs']) != {'real'}:
            raise ValueError(f'{spec["name"]}: a recon spec must reference the real clip only, got {sorted(spec["inputs"])}')
    else:
        raise ValueError(f'{spec["name"]}: unknown mode {spec["mode"]}')
    for key, rec in spec['inputs'].items():
        p = Path(rec['path'])
        if p.is_absolute() or '..' in p.parts:
            raise ValueError(f'{spec["name"]}: input paths must be relative to the input root')
        if not (input_root / p).is_file():
            raise FileNotFoundError(f'{spec["name"]}: {input_root / p} missing')
    gen = spec['generation']
    for key in ('num_rollout_steps', 'l2_pred_nfe', 'l1_nfe_cli', 'history_frames'):
        if gen.get(key) != DEF[key]:
            raise ValueError(f'{spec["name"]}: generation.{key} = {gen.get(key)}, the pinned default is {DEF[key]}')
    if gen.get('hf_revision') != PINS['weights']['hf_revision'] or gen.get('orbis2_commit') != PINS['code']['commit']:
        raise ValueError(f'{spec["name"]}: spec pins a different code or weight revision')


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


# ----------------------------------------------------------------------------------------------------------------------
class Orbis2:
    """The released Orbis 2 L1-L2 model, built as evaluate/rollout_demo_v2.py builds it."""

    def __init__(self, models_dir: Path, src: Path, device: str, verify_weights: str, allow_unpinned: bool):
        self.models_dir, self.src, self.device = models_dir, src, device
        self.digest = source_digest(src)
        if self.digest != PINS['code']['source_digest'] and not allow_unpinned:
            raise SystemExit(f'Orbis 2 source at {src} has digest {self.digest}, pinned {PINS["code"]["source_digest"]} '
                             f'(commit {PINS["code"]["commit"]})')
        self.weight_checks = self._check_weights(verify_weights, allow_unpinned)
        os.environ['ORBIS2_MODELS_DIR'] = str(models_dir)  # the configs refer to $ORBIS2_MODELS_DIR
        sys.path.insert(0, str(src))

        import torch
        from omegaconf import OmegaConf
        from util import instantiate_from_config
        from data.video_loaders import ClipAugmenter, DecordFrameAdapter, ResizeCenterPolicy
        from evaluate import rollout_demo_v2 as demo
        from evaluate.utils import get_rollout_future_frame_count
        self.torch, self.demo = torch, demo
        self.get_rollout_future_frame_count = get_rollout_future_frame_count

        # as the demo's `if args.seed > 0` block (its `cudnn.enable = False` is a typo with no effect, so cuDNN stays on)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        t0 = time.time()
        config = OmegaConf.load(models_dir / DEF['config'])
        config = OmegaConf.merge(config, OmegaConf.from_dotlist([]))
        model = instantiate_from_config(config.model)
        # The demo calls torch.load(args.ckpt) with the default weights_only (True in torch >= 2.6).
        state = torch.load(models_dir / DEF['ckpt'], map_location='cpu')['state_dict']
        result = model.load_state_dict(state, strict=False)
        del state
        exempt = tuple(getattr(model, 'checkpoint_exempt_key_prefixes', ()))
        missing = [k for k in result.missing_keys if not k.startswith(exempt)]
        if missing:
            raise RuntimeError(f'unexpected missing keys: {missing[:10]}')
        self.unexpected_keys = len(result.unexpected_keys)
        model = model.to(device)
        model.eval()
        demo.require_l1l2_model(model)
        self.model, self.config = model, config
        self.load_s = round(time.time() - t0, 1)

        self.hw = demo.resolve_context_image_size(config)
        self.l1_rate = demo.resolve_l1_frame_rate(config)
        self.l2_rate = demo.resolve_l2_frame_rate(model)
        self.l1_ctx = int(model.vit.num_context_frames)
        self.l2_ctx = int(model.condition_preprocessor.num_context_frames)
        self.augmenter = ClipAugmenter(DecordFrameAdapter(), ResizeCenterPolicy(self.hw))
        self.settings = self._read_back_settings()
        self._install_counters()

    # -- pins ----------------------------------------------------------------------------------------------------------
    def _check_weights(self, mode: str, allow_unpinned: bool) -> dict:
        out = {}
        for rel, rec in PINS['weights']['files'].items():
            if rel.startswith('L2_distilled/') or rel.endswith('config_distill.yaml'):
                continue
            p = self.models_dir / rel
            if not p.is_file():
                raise SystemExit(f'{p} missing (models dir layout: <dir>/L1, <dir>/L2, <dir>/tok)')
            entry = dict(path=rel)
            if 'bytes' in rec:
                entry['bytes'] = p.stat().st_size
                if entry['bytes'] != rec['bytes'] and not allow_unpinned:
                    raise SystemExit(f'{p}: {entry["bytes"]} bytes, pinned {rec["bytes"]}')
            if rel.endswith('.yaml') or mode == 'sha256':
                entry['sha256'] = sha256_file(p)
                if entry['sha256'] != rec['sha256'] and not allow_unpinned:
                    raise SystemExit(f'{p}: sha256 {entry["sha256"]}, pinned {rec["sha256"]}')
            out[rel] = entry
        out['_mode'] = mode
        return out

    def _read_back_settings(self) -> dict:
        m, cp = self.model, self.model.condition_preprocessor
        l2 = cp.l2_predictor
        sched = getattr(m.sampler, 'step_schedule', None)
        got = dict(
            l1_module=type(m).__name__, l1_net=type(m.vit).__name__, l1_sampler=type(m.sampler).__name__,
            l1_step_schedule=[list(x) for x in sched] if sched else None,
            l1_network_evals_per_step=sum(2 if s == 'heun' else 1 for s, _ in sched) if sched else None,
            l1_integration_t_eps=getattr(m.sampler, 'integration_t_eps', None),
            num_pred_frames=int(m.num_pred_frames), l1_context_frames=self.l1_ctx, l2_context_frames=self.l2_ctx,
            condition_preprocessor=type(cp).__name__, l2_pred_nfe=int(cp.l2_pred_NFE), use_z_l2_start=bool(cp.use_z_l2_start),
            l2_module=type(l2).__name__, l2_net=type(l2.vit).__name__, l2_sampler=type(l2.sampler).__name__,
            l2_timestep_conditioning=getattr(l2.sampler, 'timestep_conditioning', None), l2_encoder_branch=cp.get_l2_predictor_encoder_branch(),
            l1_encoder_branch=getattr(m.first_stage, 'encoder_branch', None), enc_scale=float(m.enc_scale),
            speed_scale=float(cp.speed_scale), yaw_rate_scale=float(cp.yaw_rate_scale),
            l1_frame_rate=self.l1_rate, l2_frame_rate=self.l2_rate, image_size_hw=list(self.hw),
            tokenizer=type(m.ae).__name__, unexpected_checkpoint_keys=self.unexpected_keys,
            tf32_matmul=bool(self.torch.backends.cuda.matmul.allow_tf32), tf32_cudnn=bool(self.torch.backends.cudnn.allow_tf32),
            cudnn_deterministic=bool(self.torch.backends.cudnn.deterministic), cudnn_benchmark=bool(self.torch.backends.cudnn.benchmark))
        expect = dict(l1_sampler='FlowMatchingSamplerHeunPlusEuler', l1_step_schedule=[['heun', 0.4], ['heun', 0.4], ['euler', 0.2]],
                      l1_network_evals_per_step=5, num_pred_frames=DEF['num_pred_frames'], l1_context_frames=DEF['l1_context_frames'],
                      l2_context_frames=DEF['l2_context_frames'], l2_pred_nfe=DEF['l2_pred_nfe'], l2_sampler='FlowMatchingSamplerEuler',
                      speed_scale=1.0, yaw_rate_scale=1.0, l1_frame_rate=DEF['l1_frame_rate'], l2_frame_rate=DEF['l2_frame_rate'],
                      image_size_hw=DEF['image_size_hw'], l1_encoder_branch='rec')
        bad = {k: (got[k], v) for k, v in expect.items() if got[k] != v}
        if bad:
            raise SystemExit(f'the loaded model does not run the pinned released defaults: {bad}')
        return got

    def _install_counters(self):
        """Count network evaluations of the L1 and L2 EMA nets (forward hooks; no effect on the numbers)."""
        self.calls = dict(l1=0, l2=0)
        self.model.ema_vit.register_forward_hook(lambda *a: self.calls.__setitem__('l1', self.calls['l1'] + 1))
        self.model.condition_preprocessor.l2_predictor.ema_vit.register_forward_hook(lambda *a: self.calls.__setitem__('l2', self.calls['l2'] + 1))

    # -- tensors -------------------------------------------------------------------------------------------------------
    def to_uint8(self, x):
        """[F, 3, H, W] in [-1, 1] -> [F, H, W, 3] uint8, torchvision save_image's rounding, in float32."""
        y = ((x.float() + 1.0) / 2.0).mul(255).add_(0.5).clamp_(0, 255).to(self.torch.uint8)
        return y.permute(0, 2, 3, 1).contiguous().cpu().numpy()

    def roundtrip(self, frames_u8):
        """Per-frame tokenizer round trip (the encode/decode the rollout applies to its context), batch of one frame."""
        torch = self.torch
        x = self.augmenter(frames_u8)  # [F, 3, H, W] in [-1, 1]
        out = []
        with torch.no_grad(), torch.autocast(dtype=torch.float16, device_type='cuda', enabled=self.device.startswith('cuda')):
            for i in range(x.shape[0]):
                lat = self.model.encode_frames(x[i:i + 1].unsqueeze(0).to(self.device))
                out.append(self.model.decode_frames(lat, output_device='cpu')[0, 0])
        return self.to_uint8(torch.stack(out))

    def rollout(self, frames_u8, seed: int):
        """The demo's generate_images() on an in-memory 21-frame history; returns (60 future frames, 5 decoded context frames)."""
        torch, demo, model = self.torch, self.demo, self.model
        from pytorch_lightning import seed_everything
        n = frames_u8.shape[0]
        indexer = demo._L1L2FrameIndexer(frame_interval=1, stored_data_frame_rate=float(FPS), num_l2_context=self.l2_ctx,
                                         l2_frame_rate=self.l2_rate, l1_context_frames=self.l1_ctx)
        start = n - ((self.l1_ctx - 1) * 1 + 1)  # the demo's default --start_frame: the latest window that fits
        if start < indexer.get_required_l1_start_offset():
            raise ValueError('not enough lookback for the L2 context')
        l1_idx, l2_idx = indexer.get_l1_and_l2_indices(start, self.l1_ctx)
        if l1_idx != DEF['l1_context_indices'] or l2_idx != DEF['l2_context_indices']:
            raise ValueError(f'context indices {l1_idx} {l2_idx} differ from the pinned ones')
        all_t = self.augmenter(frames_u8[l1_idx + l2_idx])
        l1 = all_t[:len(l1_idx)].unsqueeze(0).to(self.device)
        l2 = all_t[len(l1_idx):].unsqueeze(0).to(self.device)
        steps = DEF['num_rollout_steps']
        n_future = self.get_rollout_future_frame_count(model, steps)
        frame_rate = torch.tensor(self.l1_rate, device=self.device)
        batch = {'images': l1, 'l2_context': l2, 'frame_rate': frame_rate}
        rows = model.condition_preprocessor.get_required_rollout_odometry_steps(
            validation_params=None, num_condition_frames=self.l1_ctx, num_gen_frames=n_future, rollout_steps=steps)
        if rows != DEF['required_steering_rows']:
            raise ValueError(f'steering rows {rows}, pinned {DEF["required_steering_rows"]}')
        batch['steering'] = demo.make_unconditional_steering(rows, dtype=l1.dtype, device=self.device)
        batch['steering_format'] = demo.STEERING_FORMAT
        condition_kwargs = model.condition_preprocessor.get_condition_kwargs_from_batch(batch, split='rollout')
        seed_everything(int(seed), verbose=False)
        self.calls.update(l1=0, l2=0)
        with torch.no_grad(), torch.autocast(dtype=torch.float16, device_type='cuda', enabled=self.device.startswith('cuda')):
            _latents, gen = model.roll_out(x_0={'images': l1}, num_gen_frames=steps, latent_input=False, NFE=DEF['l1_nfe_cli'], eta=DEF['eta'],
                                           sample_with_ema=DEF['use_ema'], num_samples=l1.size(0), frame_rate=frame_rate.reshape(1).repeat(l1.size(0)),
                                           condition_kwargs=condition_kwargs, decode_device=DEF['decode_device'], num_condition_frames=l1.size(1))
        if tuple(gen.shape) != (1, self.l1_ctx + n_future, 3, H, W):
            raise RuntimeError(f'rollout returned {tuple(gen.shape)}')
        u8 = self.to_uint8(gen[0])
        return u8[self.l1_ctx:], u8[:self.l1_ctx], dict(l1_network_evals=self.calls['l1'], l2_network_evals=self.calls['l2'],
                                                         context_indices=dict(l1=l1_idx, l2=l2_idx), steering_rows=rows)


# ----------------------------------------------------------------------------------------------------------------------
def software(model: Orbis2 | None) -> dict:
    out = dict(python=platform.python_version(), ffmpeg=ffmpeg_version(), adapter_sha256=sha256_file(Path(__file__)), pins_sha256=sha256_file(PINS_PATH),
               orbis2_commit=PINS['code']['commit'], hf_revision=PINS['weights']['hf_revision'])
    try:
        import numpy, torch, torchvision, timm, pytorch_lightning, cv2
        out.update(torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(), torchvision=torchvision.__version__,
                   timm=timm.__version__, pytorch_lightning=pytorch_lightning.__version__, numpy=numpy.__version__, opencv=cv2.__version__)
    except Exception:  # noqa: BLE001
        pass
    if model is not None:
        out['orbis2_source_digest'] = model.digest
    for key in ('APPTAINER_CONTAINER', 'SINGULARITY_CONTAINER', 'WCB_IMAGE', 'CUDA_VISIBLE_DEVICES', '_CONDOR_JOB_AD'):
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
        info['uuid'] = 'GPU-' + str(uuid) if not str(uuid).startswith('GPU-') else str(uuid)
    try:  # cuDriverGetVersion: the newest CUDA version the installed driver supports (e.g. 12080 -> 12.8)
        import ctypes
        v = ctypes.c_int()
        if ctypes.CDLL('libcuda.so.1').cuDriverGetVersion(ctypes.byref(v)) == 0:
            info['cuda_driver'] = f'{v.value // 1000}.{(v.value % 1000) // 10}'
    except Exception:  # noqa: BLE001
        pass
    for sel in ([info['uuid']] if 'uuid' in info else []) + ['0']:
        try:
            out = subprocess.check_output(['nvidia-smi', '-i', sel, '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'],
                                          text=True, stderr=subprocess.DEVNULL).strip().splitlines()[0]
            name, driver, mem = [x.strip() for x in out.split(',')]
            info.update(driver=driver, driver_source='nvidia-smi', nvidia_smi_name=name, nvidia_smi_memory=mem,
                        nvidia_smi_selector='uuid' if sel != '0' else 'index 0')
            break
        except Exception:  # noqa: BLE001
            continue
    if 'driver' not in info:  # cluster containers have no nvidia-smi: ask NVML (bound in with the driver), then the kernel module
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


def host_memory() -> dict:
    """Peak and current resident memory of this process (GiB) and the host's MemAvailable."""
    import resource
    out = dict(peak_rss=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2))
    try:
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith(('VmRSS:', 'RssAnon:', 'RssFile:')):
                key, val = line.split(':')
                out[key.lower()] = round(int(val.split()[0]) / 2**20, 2)
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemAvailable:'):
                out['host_mem_available'] = round(int(line.split()[1]) / 2**20, 2)
    except OSError:
        pass
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--specs', nargs='+', required=True, help='spec JSON files or directories (searched recursively)')
    ap.add_argument('--input-root', type=Path, required=True)
    ap.add_argument('--output-root', type=Path, required=True)
    ap.add_argument('--models-dir', type=Path, default=os.environ.get('ORBIS2_MODELS_DIR'), help='holds L1/, L2/, tok/ (default $ORBIS2_MODELS_DIR)')
    ap.add_argument('--orbis2-src', type=Path, default=os.environ.get('ORBIS2_SRC', '/opt/orbis2'), help='Orbis 2 source at the pinned commit')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--resume', action='store_true', help='skip samples already finished (status success, video sha256 matches)')
    ap.add_argument('--dry-run', action='store_true', help='check the specs and inputs, run nothing')
    ap.add_argument('--min-free-mib', type=int, default=0, help='refuse to start if the GPU has less free memory (shared GPUs)')
    ap.add_argument('--verify-weights', choices=('size', 'sha256'), default='size', help='checkpoint check before loading')
    ap.add_argument('--threads', type=int, default=int(os.environ.get('OMP_NUM_THREADS', '2')), help='CPU threads for torch and ffmpeg')
    ap.add_argument('--allow-unpinned', action='store_true', help='development only: accept other code or weights (recorded)')
    args = ap.parse_args()
    if args.models_dir is None:
        raise SystemExit('--models-dir or $ORBIS2_MODELS_DIR is required')
    input_root, output_root = args.input_root.resolve(), args.output_root.resolve()

    specs = load_specs(args.specs)
    if not specs:
        raise SystemExit('no specs found')
    for s in specs:
        check_spec(s, input_root)
    todo = [s for s in specs if not (args.resume and is_done(s, output_root))]
    if not args.resume:
        clash = [s['name'] for s in specs if (output_root / s['output_dir'] / 'sample_outputs.json').exists()]
        if clash:
            raise SystemExit(f'{len(clash)} samples already have outputs (first: {clash[0]}); use --resume or a fresh output root')
    print(json.dumps(dict(specs=len(specs), to_run=len(todo), windows=len({s['window'] for s in specs}))), flush=True)
    if args.dry_run or not todo:
        return

    os.environ.setdefault('OMP_NUM_THREADS', str(args.threads))
    import numpy as np
    import torch
    torch.set_num_threads(args.threads)
    if args.device.startswith('cuda'):
        if not torch.cuda.is_available():
            raise SystemExit('CUDA is not available')
        free, total = torch.cuda.mem_get_info()
        if free / 2**20 < args.min_free_mib:
            raise SystemExit(f'GPU has {free / 2**20:.0f} MiB free, --min-free-mib {args.min_free_mib}')
    run_log = dict(schema='wm-calib-bench.orbis2.adapter_run.v1', started_utc=now_utc(), host=socket.gethostname(), argv=sys.argv,
                   input_root=str(input_root), output_root=str(output_root), specs=len(specs), to_run=len(todo), host_memory_start=host_memory())
    model = Orbis2(args.models_dir.resolve(), args.orbis2_src.resolve(), args.device, args.verify_weights, args.allow_unpinned)
    sw, gpu = software(model), gpu_info(torch)
    run_log.update(model_load_s=model.load_s, settings=model.settings, weights=model.weight_checks, software=sw, gpu=gpu, samples=[],
                   host_memory_after_load=host_memory(),
                   gpu_memory_after_load_gib=round(torch.cuda.memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None)
    log_dir = output_root / 'adapter-runs'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{datetime.now().strftime('%Y%m%dT%H%M%S')}-{socket.gethostname()}-{os.getpid()}.json"
    print(json.dumps(dict(model_load_s=model.load_s, gpu=gpu.get('name'))), flush=True)

    prefixes: dict = {}   # window -> (21 round-tripped history frames, their sha256)
    recon_prefix: dict = {}  # window -> sha256 of the real reference's frames 0-20 (this process)
    failures = 0
    for spec in todo:
        out_dir = output_root / spec['output_dir']
        out_dir.mkdir(parents=True, exist_ok=True)
        rec = dict(schema=OUT_SCHEMA, name=spec['name'], window=spec['window'], split=spec['split'], mode=spec['mode'], k=spec.get('k'),
                   seed=spec['seed'], run=spec.get('run'), spec=spec['_file'], spec_sha256=spec['_sha256'], started_utc=now_utc(),
                   host=socket.gethostname(), gpu=gpu, software=sw, generation=spec['generation'], settings_read_back=model.settings)
        t_start = time.time()
        try:
            torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
            if spec['mode'] == 'recon':
                src = input_root / spec['inputs']['real']['path']
                frames, info = decode_rgb(src, args.threads)
                if frames.shape != (OUT_FRAMES, H, W, 3) or abs(info['fps'] - FPS) > 1e-6:
                    raise ValueError(f'real clip {src}: {frames.shape} at {info["fps"]} fps')
                rec['input'] = dict(path=spec['inputs']['real']['path'], file_sha256=sha256_file(src), frames_sha256=sha256_bytes(frames.tobytes()))
                if spec['inputs']['real'].get('sha256') and rec['input']['file_sha256'] != spec['inputs']['real']['sha256']:
                    raise ValueError('real clip sha256 differs from the spec')
                t0 = time.time()
                clip = model.roundtrip(frames)
                rec['timing_s'] = dict(roundtrip=round(time.time() - t0, 2))
                recon_prefix[spec['window']] = sha256_bytes(clip[:HIST].tobytes())
                # codec fidelity on the history frames only (nothing is computed over the real future)
                d = np.abs(clip[:HIST].astype(np.int16) - frames[:HIST])
                rec['diagnostics'] = dict(roundtrip_vs_input_history=dict(frames=[0, HIST - 1], max_abs=int(d.max()), mean_abs=float(d.mean())))
            else:
                src = input_root / spec['inputs']['history']['path']
                frames, info = decode_rgb(src, args.threads)
                if frames.shape != (HIST, H, W, 3) or abs(info['fps'] - FPS) > 1e-6:
                    raise ValueError(f'history clip {src}: {frames.shape} at {info["fps"]} fps')
                rec['input'] = dict(path=spec['inputs']['history']['path'], file_sha256=sha256_file(src), frames_sha256=sha256_bytes(frames.tobytes()))
                if spec['inputs']['history'].get('sha256') and rec['input']['file_sha256'] != spec['inputs']['history']['sha256']:
                    raise ValueError('history clip sha256 differs from the spec')
                t0 = time.time()
                if spec['window'] not in prefixes:
                    p = model.roundtrip(frames)
                    prefixes[spec['window']] = (p, sha256_bytes(p.tobytes()))
                prefix, prefix_sha = prefixes[spec['window']]
                t1 = time.time()
                future, context, counts = model.rollout(frames, spec['seed'])
                t2 = time.time()
                clip = np.concatenate([prefix, future], axis=0)
                d = np.abs(context.astype(np.int16) - prefix[DEF['l1_context_indices'][0]:HIST])
                step = lambda a, b: float(np.abs(a.astype(np.int16) - b).mean())  # noqa: E731
                rec['rollout'] = counts
                rec['timing_s'] = dict(prefix=round(t1 - t0, 2), rollout=round(t2 - t1, 2))
                rec['diagnostics'] = dict(
                    model_context_vs_prefix=dict(frames=list(range(DEF['l1_context_indices'][0], HIST)), max_abs=int(d.max()), mean_abs=float(d.mean()),
                                                 pixels_differing=int((d.max(axis=-1) > 0).sum())),
                    step_mean_abs=dict(f19_f20=step(clip[19], clip[20]), f20_f21=step(clip[20], clip[21]), f21_f22=step(clip[21], clip[22])))
                rec['prefix21_frames_sha256'] = prefix_sha
            if clip.shape != (OUT_FRAMES, H, W, 3):
                raise RuntimeError(f'clip has shape {clip.shape}')
            rec.setdefault('prefix21_frames_sha256', sha256_bytes(clip[:HIST].tobytes()))
            if spec['mode'] == 'continue' and spec['window'] in recon_prefix:
                rec['prefix_identical_to_recon_in_process'] = rec['prefix21_frames_sha256'] == recon_prefix[spec['window']]
                if not rec['prefix_identical_to_recon_in_process']:
                    raise RuntimeError('history prefix differs from the real reference frames 0-20')
            t3 = time.time()
            video = out_dir / 'vision.mp4'
            write_lossless(clip, video, args.threads)
            back, vinfo = decode_rgb(video, args.threads)
            if not np.array_equal(back, clip):
                raise RuntimeError('vision.mp4 does not decode to the written frames')
            rec['timing_s']['write'] = round(time.time() - t3, 2)
            rec.update(status='success', video='vision.mp4', video_sha256=sha256_file(video), video_bytes=video.stat().st_size,
                       frames=vinfo['frames'], width=vinfo['width'], height=vinfo['height'], fps=vinfo['fps'], pix_fmt=vinfo['pix_fmt'],
                       frames_sha256=sha256_bytes(clip.tobytes()))
        except Exception as exc:  # noqa: BLE001
            failures += 1
            rec.update(status='failure', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
            print(rec['traceback'], file=sys.stderr, flush=True)
        if torch.cuda.is_available():
            rec['gpu_memory_gib'] = dict(max_allocated=round(torch.cuda.max_memory_allocated() / 2**30, 2), max_reserved=round(torch.cuda.max_memory_reserved() / 2**30, 2))
        rec['host_memory_gib'] = host_memory()
        rec['timing_s'] = dict(rec.get('timing_s', {}), total=round(time.time() - t_start, 2))
        rec['finished_utc'] = now_utc()
        write_json_atomic(out_dir / 'sample_outputs.json', rec)
        run_log['samples'].append(dict(name=spec['name'], status=rec['status'], timing_s=rec['timing_s'], gpu_memory_gib=rec.get('gpu_memory_gib'),
                                       host_peak_rss_gib=rec['host_memory_gib'].get('peak_rss')))
        write_json_atomic(log_path, run_log)
        print(json.dumps(dict(sample=spec['name'], status=rec['status'], **{k: v for k, v in rec['timing_s'].items()})), flush=True)
    run_log.update(finished_utc=now_utc(), failures=failures, host_memory_end=host_memory(),
                   gpu_memory_peak_gib=max((x['gpu_memory_gib'] or {}).get('max_allocated', 0) for x in run_log['samples']) if run_log['samples'] else None)
    write_json_atomic(log_path, run_log)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
