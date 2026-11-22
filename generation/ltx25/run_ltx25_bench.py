#!/usr/bin/env python3
"""LTX-2.5 adapter for the wm-calib-bench benchmark (bench-v1).

Reads spec JSONs (one per sample; make_ltx25_specs.py) and writes, per sample,
  <output-root>/<spec.output_dir>/vision.mp4           81 frames, 512x288 (canvas), 10 fps, lossless: canvas frames 0, 2, .., 160
  <output-root>/<spec.output_dir>/latents.safetensors  the final video latent, the audio latent and the DiffVAE generator state
                                                       right before decoding (any decode can be repeated or redone later)
  <output-root>/<spec.output_dir>/native20.mp4         with --keep-native20: the 161 canvas frames at 20 fps, lossless
  <output-root>/<spec.output_dir>/full1024.mp4         distilled, with --keep-full-res: the 161 decoded frames at 1024x576
  <output-root>/<spec.output_dir>/sample_outputs.json  status (success / failure), every setting, timings, memory, hashes

Variants (ltx25_pins.json; the spec names its variant; default pins.default_variant = distilled):
  distilled  ltx_pipelines.distilled.DistilledPipeline: its components are built exactly as its __init__ builds them (without
             the prompt encoder: the prompt comes from a cache; without the audio decoder: audio is not decoded), and its
             stream_chunks is re-implemented step for step for one chunk (no chunking, no carry). Precision: the spec's
             generation.quantization (bf16 'none' for every analysed clip; 'fp8-cast' only in local mechanics-smoke specs). Stage 1 at 512x288: 8 Euler-ancestral steps (eta 1, noise seed
             seed + 10000) on the distilled sigmas from pure noise; the x2 latent upsampler; stage 2 at 1024x576: 3
             Euler-ancestral steps (seed + 20000) from sigma 0.909375 (the run generator re-noises the upsampled latent);
             no guidance. In each stage the only video conditioning is VideoConditionByLatentIndex(encoder(the 41-frame
             history at that stage's size), strength 1.0, latent_idx 0); the stage rebuilds it at its own resolution, so
             the history the decoder sees is the encoded real 1024x576 history. The DiffVAE decodes the stage-2 latent at
             1024x576 with the run generator; the frames go through the official RGB24 conversion and cv2.INTER_AREA to
             the 512x288 canvas. The joint audio latent is generated, not decoded.
  dev        ltx_pipelines.ti2vid_one_stage.TI2VidOneStagePipeline at 512x288, its __call__ re-implemented step for step
             (30 Euler steps, CFG 3 + STG 1 + modality 3, negative prompt), history conditioned the same way. Kept
             available, not the default.
Modes (spec "mode"):
  continue  The history clip(s) only (t0 - 2.0 .. t0, 41 frames at 20 fps); the real future is never read.
  recon     The real reference: the real 161-frame clip at the variant's decode size through the same VAE (deterministic
            encoder, then the same DiffVAE call seeded with seed_of(window, 'recon')), then to the canvas exactly as a
            generated clip.
  redecode  Decode-only DEV diagnostics (queue dev-diag, specs from build_diag_package.py; distilled): a saved final latent
            (latents.safetensors of a DEV sample) through the DiffVAE or the conv decoder with the same decode call, the
            generator either restored to the sample's saved decoder state (rng saved: the DiffVAE must reproduce the DEV
            clip bit for bit) or freshly seeded (rng seed: an extra decoder draw). Only the decoders are built; such specs
            run alone, without a prompt cache.

Differences from the official pipelines, none of which changes the sampled distribution:
  - the conditioning list is the history latent instead of still-image items;
  - the prompt embeddings come from a cache made once per variant by --encode-prompt with the pipeline's PromptEncoder;
  - the decode tiling is resolved once from a fixed memory budget (pins: decode_free_bytes) instead of the GPU's free
    memory at call time (AUTO_TILING);
  - transformer residency (--residency): distilled per-call = the official lifecycle (built per stage call), per-sample
    = built once per sample and freed before decoding (local fp8 smoke), resident = kept across samples (the cluster H200
    runs, the default); dev per-sample (official) or resident. Each sample builds its torch.Generator(seed) exactly as
    the pipeline does.
Resumable: a sample whose sample_outputs.json says success and whose videos match the recorded sha256 is skipped. A
failed sample is recorded with status "failure" and the run continues; the exit code is 1 if any sample failed, 3 if the
run stopped early because GPU or host memory fell below the guard, 5 if --stop-after-s ended it before all samples ran
(rerun with --resume to finish).

  run_ltx25_bench.py --specs DIR_OR_JSON [...] --input-root DIR --output-root DIR --prompt-cache FILE
                     [--variant distilled|dev] [--residency MODE] [--models-dir DIR] [--resume] [--dry-run]
                     [--keep-free-mib 6144] [--min-host-avail-mib 10240] [--verify-weights size|sha256] [--keep-native20]
                     [--keep-full-res]
  run_ltx25_bench.py --encode-prompt FILE [--variant ...] [--models-dir DIR] [--te-offload none|disk]
  run_ltx25_bench.py --specs REDECODE_SPECS --input-root DIR --output-root DIR      (decode-only: queue dev-diag)
  run_ltx25_bench.py --self-test [--specs ... --input-root DIR]                       (CPU, no weights needed)
Machine-specific defaults (--models-dir, --ltx-src) come from the environment variables LTX25_MODELS_DIR, LTX25_SRC and
LTX25_ENV, else from the git-ignored models/ltx25/local.env (template local.env.example); the cluster jobs pass both
explicitly. GPU jobs on the research server run under `flock "$WCB_GPU_LOCK"` on the GPU its rules allow (local.env).
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
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_ltx25_specs as SPECGEN  # noqa: E402  (stdlib only at import; shares the generation-block rule)

PINS_PATH = HERE / 'ltx25_pins.json'
PINS = json.loads(PINS_PATH.read_text())
BENCH, OUTP, VARIANTS = PINS['bench'], PINS['output'], PINS['variants']
CW, CH = BENCH['canvas_wh']                                 # 512, 288
FPS = float(BENCH['fps'])
N_ALL, N_HIST, N_LAT_HIST = BENCH['frames'], BENCH['history_frames'], BENCH['history_latent_frames']
VISION_FPS, VISION_FRAMES = 10, (N_ALL + 1) // 2             # 81 frames at the bench's 10 Hz ticks
T0_NATIVE, T0_VISION = N_HIST - 1, (N_HIST - 1) // 2         # 40, 20
SPEC_SCHEMA = SPECGEN.SCHEMA
OUT_SCHEMA = 'wm-calib-bench.ltx25.sample_outputs.v2'
CACHE_SCHEMA = 'wm-calib-bench.ltx25.prompt_cache.v2'
LOCAL_ENV = Path(os.environ.get('WCB_LOCAL_ENV') or HERE / 'local.env')
TEXT_ENCODER = 'text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors'
VIDEO_VAE = 'vae/ltx-2.5-video-vae-bf16.safetensors'
AUDIO_VAE = 'vae/ltx-2.5-audio-vae-bf16.safetensors'
CONV_VAE = 'vae/ltx-2.5-video-vae-conv-bf16.safetensors'
DECODER_FILES = {'diffvae': VIDEO_VAE, 'conv': CONV_VAE}
RESIDENCY = {'distilled': ('per-call', 'per-sample', 'resident'), 'dev': ('per-sample', 'resident')}
GIB = 2 ** 30


# ----------------------------------------------------------------------------------------------------------------------
# small helpers (no torch)
def local_setting(key: str, path: Path | None = None) -> str | None:
    """A machine-specific default: the environment variable KEY, else KEY=VALUE in models/ltx25/local.env (git-ignored; template
    local.env.example; WCB_LOCAL_ENV=FILE points elsewhere), else None. Lines are KEY=VALUE with optional 'export ' and quotes;
    '#' starts a comment."""
    if os.environ.get(key):
        return os.environ[key]
    try:
        lines = Path(path or LOCAL_ENV).read_text().splitlines()
    except OSError:
        return None
    for line in lines:
        line = line.split('#', 1)[0].strip()
        if line.startswith('export '):
            line = line[len('export '):].strip()
        k, eq, v = line.partition('=')
        if eq and k.strip() == key:
            return v.strip().strip('"').strip("'") or None
    return None


def default_ltx_src() -> Path | None:
    """LTX25_SRC, else $LTX25_ENV/src/LTX-2 (environment or local.env)."""
    src = local_setting('LTX25_SRC')
    if src:
        return Path(src)
    env = local_setting('LTX25_ENV')
    return Path(env) / 'src' / 'LTX-2' if env else None


def default_models_dir() -> Path | None:
    d = local_setting('LTX25_MODELS_DIR')
    return Path(d) if d else None


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


def source_digest(root: Path) -> tuple[str, int]:
    """The pins' rule: sha256 over the sorted *.py files under packages/ltx-core/src and packages/ltx-pipelines/src."""
    files = []
    for sub in ('packages/ltx-core/src', 'packages/ltx-pipelines/src'):
        for p in (root / sub).rglob('*.py'):
            rel = p.relative_to(root)
            if '__pycache__' in rel.parts or any(x.startswith('.') for x in rel.parts):
                continue
            files.append(str(rel))
    files.sort()
    h = hashlib.sha256()
    for f in files:
        h.update(f.encode() + b'\0' + sha256_file(root / f).encode() + b'\n')
    return h.hexdigest(), len(files)


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


def write_lossless(frames, path: Path, fps: int, threads: int) -> None:
    """make_wm_calibration_windows.write_video's encoder (libx264rgb -crf 0 -preset medium, rgb24, timescale 10240) at fps."""
    import numpy as np
    h, w = frames.shape[1:3]
    tmp = path.with_name(path.stem + '.part.mp4')
    proc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}', '-r', str(fps), '-i', '-',
                             '-c:v', 'libx264rgb', '-crf', '0', '-preset', 'medium', '-threads', str(threads), '-pix_fmt', 'rgb24', '-r', str(fps),
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


def mem_available_mib() -> int:
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) // 1024
    return -1


def host_memory() -> dict:
    """Peak and current resident memory of this process (GiB) and the host's MemAvailable."""
    import resource
    out = dict(peak_rss=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2))
    try:
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith(('VmRSS:', 'VmHWM:', 'RssAnon:', 'RssFile:')):
                key, val = line.split(':')
                out[key.lower()] = round(int(val.split()[0]) / 2**20, 2)
        out['host_mem_available'] = round(mem_available_mib() / 1024, 2)
    except OSError:
        pass
    return out


def transformer_file(variant: str) -> str:
    return VARIANTS[variant]['transformer']


def needed_files(variant: str, purpose: str = 'generate') -> tuple[str, ...]:
    """The pinned files a <variant>:<purpose> process opens (purpose: generate | encode-prompt)."""
    tag = f'{variant}:{purpose}'
    return tuple(k for k, r in PINS['weights']['files'].items() if tag in r.get('needed_for', ()))


def spec_quantization(specs: list) -> str:
    """The one generation.quantization all specs of a run share."""
    q = {s['generation'].get('quantization') for s in specs}
    if len(q) != 1:
        raise SystemExit(f'the specs mix precisions {sorted(map(str, q))}; run them separately')
    return q.pop()


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
    names = [(s['run'], s['name']) for s in specs]
    if len(set(names)) != len(names):
        raise SystemExit('duplicate (run, sample name) pairs among the specs')
    # per window: the real reference first (its prefix is compared with the continuations'), then the samples by k
    specs.sort(key=lambda s: (s['split'], s['window'], 0 if s['mode'] == 'recon' else 1, s.get('k') or 0, s['run']))
    return specs


def check_spec(spec: dict, input_root: Path, variant: str | None = None) -> None:
    for key in ('name', 'window', 'split', 'mode', 'seed', 'inputs', 'output_dir', 'generation', 'variant'):
        if key not in spec:
            raise ValueError(f'spec {spec.get("name")}: missing {key}')
    v = spec['variant']
    if variant is not None and v != variant:
        raise ValueError(f'{spec["name"]}: spec variant {v}, this run is {variant}')
    if spec['mode'] == 'redecode':
        check_redecode_spec(spec, input_root)
        return
    if spec['mode'] not in ('continue', 'recon'):
        raise ValueError(f'{spec["name"]}: unknown mode {spec["mode"]}')
    want = SPECGEN.INPUTS[v]['continue_' if spec['mode'] == 'continue' else 'recon']
    if set(spec['inputs']) != set(want):
        raise ValueError(f'{spec["name"]}: a {v} {spec["mode"]} spec must reference {sorted(want)} only, got {sorted(spec["inputs"])}')
    for role, rec in spec['inputs'].items():
        p = Path(rec['path'])
        if p.is_absolute() or '..' in p.parts:
            raise ValueError(f'{spec["name"]}: input paths must be relative to the input root')
        if p.name != want[role][0] or rec.get('frames') != want[role][1]:
            raise ValueError(f'{spec["name"]}: input {role} is {p.name} with {rec.get("frames")} frames, expected {want[role]}')
        if not (input_root / p).is_file():
            raise FileNotFoundError(f'{spec["name"]}: {input_root / p} missing')
    want_gen = SPECGEN.generation_block(v, spec['generation'].get('quantization'))
    if spec['generation'] != want_gen:
        bad = sorted(k for k in set(spec['generation']) | set(want_gen) if spec['generation'].get(k) != want_gen.get(k))
        raise ValueError(f'{spec["name"]}: generation block differs from the current pins in {bad}')


def check_redecode_spec(spec: dict, input_root: Path) -> None:
    """A decode-only spec: one saved latent file, a pinned decoder, the generator either restored (rng saved, seed null) or
    seeded (rng seed, integer seed), and the decode block of the current pins."""
    n = spec['name']
    for key in ('decoder', 'rng', 'source'):
        if key not in spec:
            raise ValueError(f'redecode spec {n}: missing {key}')
    if spec['variant'] != 'distilled':
        raise ValueError(f'{n}: redecode specs are defined for the distilled variant')
    if spec['decoder'] not in DECODER_FILES:
        raise ValueError(f'{n}: unknown decoder {spec["decoder"]}')
    if spec['rng'] == 'seed':
        if not isinstance(spec['seed'], int) or isinstance(spec['seed'], bool):
            raise ValueError(f'{n}: rng seed needs an integer seed')
    elif spec['rng'] == 'saved':
        if spec['seed'] is not None:
            raise ValueError(f'{n}: rng saved restores the source\'s decoder state; seed must be null')
    else:
        raise ValueError(f'{n}: rng must be saved or seed')
    for key in ('run', 'name', 'latents_sha256'):
        if not spec['source'].get(key):
            raise ValueError(f'{n}: source.{key} missing')
    if set(spec['inputs']) != {'latents'}:
        raise ValueError(f'{n}: a redecode spec references one latents file, got {sorted(spec["inputs"])}')
    p = Path(spec['inputs']['latents']['path'])
    if p.is_absolute() or '..' in p.parts or p.suffix != '.safetensors':
        raise ValueError(f'{n}: the latents path must be a relative .safetensors path')
    if not (input_root / p).is_file():
        raise FileNotFoundError(f'{n}: {input_root / p} missing')
    want = SPECGEN.decode_block(spec['variant'], spec['decoder'])
    if spec['generation'] != want:
        bad = sorted(k for k in set(spec['generation']) | set(want) if spec['generation'].get(k) != want.get(k))
        raise ValueError(f'{n}: decode block differs from the current pins in {bad}')


def is_done(spec: dict, output_root: Path) -> bool:
    d = output_root / spec['output_dir']
    f = d / 'sample_outputs.json'
    if not f.is_file():
        return False
    try:
        rec = json.loads(f.read_text())
    except json.JSONDecodeError:
        return False
    if rec.get('status') != 'success' or rec.get('spec_sha256') != spec['_sha256']:
        return False
    return all((d / name).is_file() and info.get('sha256') == sha256_file(d / name) for name, info in rec.get('files', {}).items())


def check_weights(models_dir: Path, mode: str, allow_unpinned: bool, needed: tuple[str, ...]) -> dict:
    out = {}
    for rel in needed:
        rec = PINS['weights']['files'][rel]
        p = models_dir / rel
        if not p.is_file():
            raise SystemExit(f'{p} missing (weights not downloaded: see download_ltx25_weights.sh)')
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


def to_u8(torch, frames):
    """[F, H, W, C] float in [0, 1] -> uint8 numpy, ltx_core.color.yuv.FrameConverter.RGB24's rule (clamp, * 255, truncate)."""
    return frames.float().clamp(0.0, 1.0).mul(255.0).to(torch.uint8).cpu().numpy()


def to_canvas(frames_u8):
    """Decoded uint8 frames -> the 512x288 canvas: identity at 512x288, else cv2.INTER_AREA (the bench renderer's call)."""
    import numpy as np
    if frames_u8.shape[1:3] == (CH, CW):
        return frames_u8
    import cv2
    return np.stack([cv2.resize(f, (CW, CH), interpolation=cv2.INTER_AREA) for f in frames_u8])


def max_abs(a, b) -> float:
    return float((a.float() - b.float()).abs().max().item())


# ----------------------------------------------------------------------------------------------------------------------
class LTX25:
    """The released LTX-2.5 pipelines, built as the official CLIs build them (distilled: DistilledPipeline; dev:
    TI2VidOneStagePipeline)."""

    def __init__(self, variant: str, models_dir: Path, src: Path, device: str, verify_weights: str, allow_unpinned: bool, te_offload: str = 'none',
                 purpose: str = 'generate', quantization: str | None = None):
        self.variant, self.v, self.purpose = variant, VARIANTS[variant], purpose
        self.quantization = quantization or self.v['quantization']
        if self.quantization not in self.v['allowed_quantization']:
            raise SystemExit(f'quantization {self.quantization} is not allowed for the {variant} variant')
        self.models_dir, self.src = models_dir, src
        self.digest, nfiles = source_digest(src)
        if self.digest != PINS['code']['source_digest'] and not allow_unpinned:
            raise SystemExit(f'LTX-2 source at {src} has digest {self.digest} ({nfiles} files), pinned {PINS["code"]["source_digest"]} '
                             f'(commit {PINS["code"]["commit"]})')
        self.weight_checks = check_weights(models_dir, verify_weights, allow_unpinned, needed_files(variant, purpose))

        import torch
        from ltx_core.model.video_vae.transformer import DiffVAEMode
        from ltx_pipelines.utils.constants import DEFAULT_NEGATIVE_PROMPT, detect_model_version, detect_params
        from ltx_pipelines.utils.helpers import tiling_config_for_vae
        from ltx_pipelines.utils.model_paths import ModelPaths
        from ltx_pipelines.utils.quantization_factory import QuantizationKind
        from ltx_pipelines.utils.types import OffloadMode
        self.torch = torch
        torch.backends.cudnn.deterministic = BENCH['cudnn']['deterministic']
        torch.backends.cudnn.benchmark = BENCH['cudnn']['benchmark']
        self.device = torch.device(device)
        t0 = time.time()
        p = lambda rel: str(models_dir / rel) if (models_dir / rel).exists() else None  # noqa: E731
        tf = p(transformer_file(variant))
        self.model_paths = ModelPaths.from_split(transformer_path=tf, text_encoder_path=p(TEXT_ENCODER), video_vae_path=p(VIDEO_VAE),
                                                 audio_vae_path=p(AUDIO_VAE))
        self.model_version = tuple(detect_model_version(tf))
        policy = QuantizationKind(self.quantization).to_policy(checkpoint_path=tf) if self.quantization != 'none' else None
        offload = OffloadMode.DISK if te_offload == 'disk' else OffloadMode.NONE
        mode = DiffVAEMode(self.v['diffvae_optimization'])
        if variant == 'distilled':
            from ltx_pipelines.distilled import DistilledPipeline, should_use_ancestral_sampler
            from ltx_pipelines.utils.blocks import DiffusionStage, ImageConditioner, PromptEncoder, VideoDecoder, VideoUpsampler
            # DistilledPipeline.__init__, component by component, minus the PromptEncoder (the prompt comes from a cache;
            # built alone for --encode-prompt) and the AudioDecoder (audio is not decoded); every block gets its own default
            # registry, as there (registry=None)
            pipe = object.__new__(DistilledPipeline)
            pipe.device, pipe.dtype = self.device, torch.bfloat16
            pipe.prompt_encoder = pipe.audio_decoder = pipe.duration_predictor = None
            if purpose == 'encode-prompt':
                pipe.prompt_encoder = PromptEncoder(self.model_paths, pipe.dtype, self.device, offload_mode=offload)
            else:
                pipe.image_conditioner = ImageConditioner(self.model_paths.video_vae(), pipe.dtype, self.device)
                pipe.stage = DiffusionStage.from_checkpoint(self.model_paths.transformer(), pipe.dtype, self.device, loras=(), quantization=policy)
                pipe.upsampler = VideoUpsampler(self.model_paths.video_vae(), p(self.v['spatial_upsampler']), pipe.dtype, self.device)
                pipe.video_decoder = VideoDecoder(self.model_paths.video_vae(), pipe.dtype, self.device, diffvae_optimization=mode)
                pipe.use_ancestral_sampler = should_use_ancestral_sampler(self.model_paths.transformer())
            self.pipe = pipe
            self.negative_prompt = None
        else:
            from ltx_pipelines.ti2vid_one_stage import TI2VidOneStagePipeline
            self.pipe = TI2VidOneStagePipeline(model_paths=self.model_paths, loras=(), device=self.device, quantization=policy, offload_mode=offload,
                                               diffvae_optimization=mode)
            self.negative_prompt = DEFAULT_NEGATIVE_PROMPT
            self.params = detect_params(tf)
        if purpose == 'encode-prompt':
            self.settings = dict(variant=variant, model_version=list(self.model_version), purpose=purpose)
        else:
            self.settings = self._read_back_settings()
            dw, dh = self.v['decode_size_wh']
            self.tiling = tiling_config_for_vae(p(VIDEO_VAE), height=dh, width=dw, num_frames=N_ALL, diffvae_optimization=mode, device=self.device,
                                                free_bytes=int(self.v['decode_free_bytes']))
            self.settings['decode_tiling'] = repr(self.tiling)
        self.construct_s = round(time.time() - t0, 1)
        self.cache = None

    def _read_back_settings(self) -> dict:
        torch = self.torch
        got = dict(variant=self.variant, model_version=list(self.model_version), pipeline=type(self.pipe).__name__,
                   video_scale_factors=repr(self.pipe.stage.video_scale_factors), duration_predictor=self.pipe.duration_predictor is not None,
                   quantization=self.quantization, cudnn_deterministic=bool(torch.backends.cudnn.deterministic),
                   cudnn_benchmark=bool(torch.backends.cudnn.benchmark), tf32_matmul=bool(torch.backends.cuda.matmul.allow_tf32),
                   tf32_cudnn=bool(torch.backends.cudnn.allow_tf32))
        try:
            from ltx_core.model.video_vae.transformer.attention import _NATTEN_AVAILABLE
            got['natten_available'] = bool(_NATTEN_AVAILABLE)
        except Exception:  # noqa: BLE001
            got['natten_available'] = None
        bad = {}
        if not got['natten_available']:
            bad['natten_available'] = (got['natten_available'], True)
        if self.variant == 'distilled':
            from ltx_pipelines import distilled as D
            from ltx_pipelines.utils.constants import DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES
            got.update(use_ancestral_sampler=bool(self.pipe.use_ancestral_sampler), ancestral_eta=D.ANCESTRAL_ETA, ancestral_s_noise=D.ANCESTRAL_S_NOISE,
                       ancestral_offsets=[D.ANCESTRAL_NOISE_SEED_OFFSET, D.ANCESTRAL_STAGE_2_NOISE_SEED_OFFSET],
                       stage_1_sigmas=list(DISTILLED_SIGMA_VALUES), stage_2_sigmas=list(STAGE_2_DISTILLED_SIGMA_VALUES))
            expect = dict(pipeline='DistilledPipeline', use_ancestral_sampler=True, ancestral_eta=self.v['sampler']['eta'],
                          ancestral_s_noise=self.v['sampler']['s_noise'],
                          ancestral_offsets=[self.v['stage_1']['ancestral_noise_seed_offset'], self.v['stage_2']['ancestral_noise_seed_offset']],
                          stage_1_sigmas=self.v['stage_1']['sigmas'], stage_2_sigmas=self.v['stage_2']['sigmas'])
            if self.model_version < (2, 5):
                bad['model_version'] = (got['model_version'], '>= 2.5')
        else:
            prm = self.params
            g = lambda gp: dict(cfg_scale=gp.cfg_scale, stg_scale=gp.stg_scale, stg_blocks=list(gp.stg_blocks), rescale_scale=gp.rescale_scale,  # noqa: E731
                                modality_scale=gp.modality_scale, skip_step=gp.skip_step)
            got.update(num_inference_steps=prm.num_inference_steps, video_guider=g(prm.video_guider_params), audio_guider=g(prm.audio_guider_params),
                       negative_prompt_sha256=sha256_bytes(self.negative_prompt.encode()))
            expect = dict(pipeline='TI2VidOneStagePipeline', num_inference_steps=self.v['num_inference_steps'], video_guider=self.v['video_guider'],
                          audio_guider=self.v['audio_guider'], negative_prompt_sha256=self.v['negative_prompt_sha256'])
            if self.model_version < (2, 4):
                bad['model_version'] = (got['model_version'], '>= 2.4')
        bad.update({k: (got[k], e) for k, e in expect.items() if got[k] != e})
        if bad:
            raise SystemExit(f'the loaded checkpoint/pipeline does not run the pinned released defaults: {bad}')
        return got

    # -- prompt cache --------------------------------------------------------------------------------------------------
    def prompts(self) -> list[str]:
        return [BENCH['prompt']] if self.variant == 'distilled' else [BENCH['prompt'], self.negative_prompt]

    def encode_prompt(self, out: Path) -> dict:
        from safetensors.torch import save_file
        t0 = time.time()
        ctxs = self.pipe.prompt_encoder(self.prompts(), enhance_first_prompt=False, enhance_static_cache=False, enhance_prompt_image=None)
        tensors = {}
        for tag, ctx in zip(('p', 'n'), ctxs):
            for field in ('video_encoding', 'audio_encoding', 'attention_mask'):
                v = getattr(ctx, field)
                if v is not None:
                    tensors[f'{tag}.{field}'] = v.detach().contiguous().cpu()
        meta = dict(schema=CACHE_SCHEMA, variant=self.variant, prompt_sha256=BENCH['prompt_sha256'],
                    negative_prompt_sha256=None if self.negative_prompt is None else sha256_bytes(self.negative_prompt.encode()),
                    ltx2_commit=PINS['code']['commit'], hf_revision=PINS['weights']['hf_revision'],
                    text_encoder_sha256=PINS['weights']['files'][TEXT_ENCODER]['sha256'],
                    transformer_sha256=PINS['weights']['files'][transformer_file(self.variant)]['sha256'],
                    shapes={k: list(v.shape) for k, v in tensors.items()}, dtypes={k: str(v.dtype) for k, v in tensors.items()}, created_utc=now_utc())
        tmp = out.with_name(out.name + '.part')
        save_file(tensors, str(tmp), metadata={'bench': json.dumps(meta)})
        os.replace(tmp, out)
        meta['file_sha256'] = sha256_file(out)
        meta['seconds'] = round(time.time() - t0, 1)
        return meta

    def load_prompt_cache(self, path: Path) -> dict:
        from safetensors import safe_open
        with safe_open(str(path), framework='pt', device='cpu') as f:
            meta = json.loads((f.metadata() or {}).get('bench', '{}'))
            tensors = {k: f.get_tensor(k) for k in f.keys()}
        want = dict(schema=CACHE_SCHEMA, variant=self.variant, prompt_sha256=BENCH['prompt_sha256'], ltx2_commit=PINS['code']['commit'],
                    hf_revision=PINS['weights']['hf_revision'], transformer_sha256=PINS['weights']['files'][transformer_file(self.variant)]['sha256'])
        bad = {k: (meta.get(k), v) for k, v in want.items() if meta.get(k) != v}
        if bad:
            raise SystemExit(f'prompt cache {path} was made for other settings: {bad}')
        dev = lambda k: tensors[k].to(self.device) if k in tensors else None  # noqa: E731
        self.cache = dict(v_p=dev('p.video_encoding'), a_p=dev('p.audio_encoding'), v_n=dev('n.video_encoding'), a_n=dev('n.audio_encoding'))
        if self.variant == 'dev' and self.cache['v_n'] is None:
            raise SystemExit('the dev variant needs the negative prompt in the cache')
        self.cache_info = dict(path=str(path), sha256=sha256_file(path), meta=meta)
        return self.cache_info

    # -- tensors -------------------------------------------------------------------------------------------------------
    def preprocess(self, frames_u8):
        """The official video conditioning preprocessing (video_preprocess) at the frames' own size: [F,H,W,3] uint8 ->
        (1, 3, F, H, W) in [-1, 1], bf16 (resize-and-centre-crop is the identity at the same size)."""
        from ltx_pipelines.utils.media_io.decode import video_preprocess
        torch = self.torch
        h, w = frames_u8.shape[1:3]
        return video_preprocess((torch.from_numpy(f).unsqueeze(0) for f in frames_u8), h, w, torch.bfloat16, self.device)

    def encode(self, *clips):
        """Encode one or more pixel clips with one encoder build (ImageConditioner owns the encoder lifecycle)."""
        return self.pipe.image_conditioner(lambda enc: tuple(enc(x) for x in clips))

    def decode(self, latent, generator):
        """The variant's official video decode call; video only. Returns [F, H, W, C] float in [0, 1]. The generator's state
        right before the call is kept (self.last_decoder_rng_state) so the decode can be repeated from the saved latent."""
        torch = self.torch
        self.last_decoder_rng_state = generator.get_state().clone().cpu()
        if self.variant == 'distilled':  # chunks.operators._decode_chunk_video
            chunks = self.pipe.video_decoder(latent, self.tiling, generator, dtype=torch.bfloat16, device_fn=lambda _batch, device=latent.device: device)
        else:  # TI2VidOneStagePipeline.__call__
            chunks = self.pipe.video_decoder(latent, self.tiling, generator=generator, dtype=torch.bfloat16)
        return torch.cat([c.float() for c in chunks], dim=0)

    def generate_distilled(self, hist_lo, hist_hr, seed: int, residency: str):
        """DistilledPipeline.stream_chunks for one chunk, step for step, with the history latent as the video conditioning."""
        torch = self.torch
        from ltx_core.components.noisers import GaussianNoiser
        from ltx_core.conditioning.types.latent_cond import VideoConditionByLatentIndex
        from ltx_core.types import VIDEO_SCALE_FACTORS, VideoPixelShape
        from ltx_pipelines.chunks import ChunkConfig, denoise_chunks, generate_uniform_chunks, spatially_upsample_chunks
        from ltx_pipelines.distilled import ANCESTRAL_NOISE_SEED_OFFSET, ANCESTRAL_STAGE_2_NOISE_SEED_OFFSET
        from ltx_pipelines.utils import SimpleDenoiser
        from ltx_pipelines.utils.constants import DISTILLED_SIGMAS, STAGE_2_DISTILLED_SIGMAS
        from ltx_pipelines.utils.helpers import snap_frames_to_grid
        from ltx_pipelines.utils.types import VideoAudio
        pipe, dtype, c = self.pipe, torch.bfloat16, self.cache
        t, taps, encoded, calls = {}, {}, {}, dict(stage_1=0, stage_2=0)
        t_start = time.time()

        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        noiser = GaussianNoiser(generator=generator)
        num_frames = snap_frames_to_grid(N_ALL)
        if num_frames != N_ALL:
            raise RuntimeError(f'snap_frames_to_grid({N_ALL}) = {num_frames}')
        history = {tuple(hist_lo.shape[1:3]): hist_lo, tuple(hist_hr.shape[1:3]): hist_hr}

        def make_video_conditionings(chunk):  # replaces DistilledPipeline._video_conditionings
            hw = (chunk.video.shape[-2] * VIDEO_SCALE_FACTORS.height, chunk.video.shape[-1] * VIDEO_SCALE_FACTORS.width)
            t1 = time.time()
            (lat,) = self.encode(self.preprocess(history[hw]))
            t[f'encode_{hw[1]}x{hw[0]}'] = round(time.time() - t1, 2)
            if tuple(lat.shape[2:]) != (N_LAT_HIST, chunk.video.shape[-2], chunk.video.shape[-1]):
                raise RuntimeError(f'history latent {tuple(lat.shape)} does not fit the chunk {tuple(chunk.video.shape)}')
            encoded[hw] = lat
            return [VideoConditionByLatentIndex(latent=lat, strength=1.0, latent_idx=0)]

        def counting(stage):
            def make(chunk):
                inner = SimpleDenoiser(chunk.video_context, chunk.audio_context)  # denoise_chunks' default denoiser

                def call(*a, **k):
                    calls[stage] += 1
                    return inner(*a, **k)
                return call
            return make

        def tap(it, name):
            for chunk in it:
                taps[name] = dict(t=time.time(), video=chunk.video, shape=list(chunk.video.shape))
                yield chunk

        (sw, sh) = self.v['stage_1']['size_wh']
        target = VideoPixelShape(batch=1, frames=num_frames, height=sh, width=sw, fps=FPS)
        chunks = generate_uniform_chunks(target=target, context=VideoAudio(video=c['v_p'], audio=c['a_p']), device=self.device, dtype=dtype,
                                         config=ChunkConfig(chunk_pixel_frames=num_frames, next_video_carry_frames=0),
                                         video_scale_factors=pipe.stage.video_scale_factors, make_video_conditionings=make_video_conditionings,
                                         generated_keyframes=0)
        s1 = DISTILLED_SIGMAS.to(dtype=torch.float32, device=self.device)
        s2 = STAGE_2_DISTILLED_SIGMAS.to(dtype=torch.float32, device=self.device)
        base = tap(denoise_chunks(chunks, pipe.stage, sigmas=s1, noiser=noiser, fps=FPS, noise_scale=1.0, make_denoiser=counting('stage_1'),
                                  **pipe._sampler_kwargs(seed, ANCESTRAL_NOISE_SEED_OFFSET)), 'stage_1')
        upscaled = tap(spatially_upsample_chunks(base, pipe.upsampler), 'upsample')
        refined = tap(denoise_chunks(upscaled, pipe.stage, sigmas=s2, noiser=noiser, fps=FPS, make_denoiser=counting('stage_2'),
                                     **pipe._sampler_kwargs(seed, ANCESTRAL_STAGE_2_NOISE_SEED_OFFSET)), 'stage_2')
        with (pipe.stage.resident() if residency == 'per-sample' else nullcontext()):
            out = list(refined)  # runs stage 1, the upsampler and stage 2; per-sample residency frees the transformer here
        torch.cuda.synchronize() if self.device.type == 'cuda' else None
        if len(out) != 1:
            raise RuntimeError(f'{len(out)} chunks, expected 1')
        lat = out[0].video
        lo, hr = tuple(self.v['stage_1']['size_wh'][::-1]), tuple(self.v['stage_2']['size_wh'][::-1])
        t['stage_1'] = round(taps['stage_1']['t'] - t_start, 2)
        t['upsample'] = round(taps['upsample']['t'] - taps['stage_1']['t'], 2)
        t['stage_2'] = round(taps['stage_2']['t'] - taps['upsample']['t'], 2)
        self.last_latents = dict(video_latent=lat, audio_latent=out[0].audio)
        info = dict(denoiser_calls=[calls['stage_1'], calls['stage_2']], stage_1_latent_shape=taps['stage_1']['shape'],
                    upsampled_latent_shape=taps['upsample']['shape'], video_latent_shape=list(lat.shape),
                    audio_latent_shape=list(out[0].audio.shape) if out[0].audio is not None else None,
                    history_latent_pinned_max_abs_stage_1=max_abs(taps['stage_1']['video'][:, :, :N_LAT_HIST], encoded[lo]),
                    upsampled_history_vs_encoded_hr_max_abs=max_abs(taps['upsample']['video'][:, :, :N_LAT_HIST], encoded[hr]),
                    history_latent_pinned_max_abs=max_abs(lat[:, :, :N_LAT_HIST], encoded[hr]),
                    history_latent_sha256=sha256_bytes(encoded[hr].float().cpu().numpy().tobytes()), residency=residency)
        t2 = time.time()
        frames = self.decode(lat, generator)
        t['decode'] = round(time.time() - t2, 2)
        return frames, info, t, encoded[hr]

    def generate_dev(self, history_u8, seed: int):
        """TI2VidOneStagePipeline.__call__, step for step, with the history latent as the conditioning."""
        torch = self.torch
        from ltx_core.components.guiders import MultiModalGuiderParams, create_multimodal_guider_factory
        from ltx_core.components.noisers import GaussianNoiser
        from ltx_core.conditioning.types.latent_cond import VideoConditionByLatentIndex
        from ltx_pipelines.utils.denoisers import FactoryGuidedDenoiser
        from ltx_pipelines.utils.helpers import create_initial_av_latents
        from ltx_pipelines.utils.types import ModalitySpec, VideoAudio
        pipe, dtype, c, t = self.pipe, torch.bfloat16, self.cache, {}
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        noiser = GaussianNoiser(generator=generator)
        t0 = time.time()
        (hist_latent,) = self.encode(self.preprocess(history_u8))
        conditionings = [VideoConditionByLatentIndex(latent=hist_latent, strength=1.0, latent_idx=0)]
        t['encode'] = round(time.time() - t0, 2)
        sigmas = pipe._scheduler.execute(steps=self.v['num_inference_steps']).to(dtype=torch.float32, device=self.device)
        vgf = create_multimodal_guider_factory(params=MultiModalGuiderParams(**self.v['video_guider']), negative_context=c['v_n'])
        agf = create_multimodal_guider_factory(params=MultiModalGuiderParams(**self.v['audio_guider']), negative_context=c['a_n'])
        mw, mh = self.v['model_size_wh']
        video_latent, audio_latent = create_initial_av_latents(width=mw, height=mh, frames=N_ALL, fps=FPS, device=self.device, dtype=dtype,
                                                               video_scale_factors=pipe.stage.video_scale_factors)
        denoiser = FactoryGuidedDenoiser(v_context=c['v_p'], a_context=c['a_p'], video_guider_factory=vgf, audio_guider_factory=agf)
        calls = dict(n=0)

        def counted(*a, **k):
            calls['n'] += 1
            return denoiser(*a, **k)

        t1 = time.time()
        video_state, audio_state = pipe.stage(
            denoiser=counted, sigmas=sigmas, noiser=noiser,
            modalities=VideoAudio(video=ModalitySpec(latent=video_latent, conditioning_fps=FPS, context=c['v_p'], conditionings=conditionings),
                                  audio=ModalitySpec(latent=audio_latent, conditioning_fps=FPS, context=c['a_p'])),
            max_batch_size=self.v['max_batch_size'])
        torch.cuda.synchronize() if self.device.type == 'cuda' else None
        t['denoise'] = round(time.time() - t1, 2)
        lat = video_state.latent
        self.last_latents = dict(video_latent=lat, audio_latent=audio_state.latent if audio_state is not None else None)
        info = dict(denoiser_calls=[calls['n']], video_latent_shape=list(lat.shape),
                    audio_latent_shape=list(audio_state.latent.shape) if audio_state is not None else None,
                    history_latent_pinned_max_abs=max_abs(lat[:, :, :N_LAT_HIST], hist_latent),
                    history_latent_sha256=sha256_bytes(hist_latent.float().cpu().numpy().tobytes()))
        t2 = time.time()
        frames = self.decode(lat, generator)
        t['decode'] = round(time.time() - t2, 2)
        return frames, info, t, hist_latent

    def reconstruct(self, real_u8, seed: int):
        """The real clip at the decode size through the same VAE: encode 161 frames (and, for the causality check, the first
        41), then the variant's decode call with the reference seed."""
        torch = self.torch
        t = {}
        t0 = time.time()
        full, hist = self.encode(self.preprocess(real_u8), self.preprocess(real_u8[:N_HIST]))
        t['encode'] = round(time.time() - t0, 2)
        self.last_latents = dict(video_latent=full, audio_latent=None)
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        t1 = time.time()
        frames = self.decode(full, generator)
        t['decode'] = round(time.time() - t1, 2)
        info = dict(video_latent_shape=list(full.shape), causal_encode_hist_vs_full_max_abs=max_abs(full[:, :, :N_LAT_HIST], hist),
                    history_latent_sha256=sha256_bytes(hist.float().cpu().numpy().tobytes()))
        return frames, info, t, hist


class Decoders:
    """Decode-only process (mode redecode): the video decode call of the distilled runs -- ltx_pipelines.utils.blocks.VideoDecoder
    with the tiling of ltx_pipelines.utils.helpers.tiling_config_for_vae, built and called exactly as LTX25 does -- for the
    DiffVAE and, for the sensitivity arm, the convolutional VAE. No transformer, upsampler, encoder or prompt."""

    def __init__(self, variant: str, models_dir: Path, src: Path, device: str, verify_weights: str, allow_unpinned: bool, decoders):
        if variant != 'distilled':
            raise SystemExit('redecode runs are defined for the distilled variant')
        unknown = set(decoders) - set(DECODER_FILES)
        if unknown:
            raise SystemExit(f'unknown decoders {sorted(unknown)}')
        self.variant, self.v, self.models_dir, self.src = variant, VARIANTS[variant], models_dir, src
        self.digest, nfiles = source_digest(src)
        if self.digest != PINS['code']['source_digest'] and not allow_unpinned:
            raise SystemExit(f'LTX-2 source at {src} has digest {self.digest} ({nfiles} files), pinned {PINS["code"]["source_digest"]} '
                             f'(commit {PINS["code"]["commit"]})')
        self.weight_checks = check_weights(models_dir, verify_weights, allow_unpinned, tuple(DECODER_FILES[d] for d in sorted(decoders)))
        import torch
        from ltx_core.model.video_vae import is_diffusion_video_vae
        from ltx_core.model.video_vae.transformer import DiffVAEMode
        from ltx_pipelines.utils.blocks import VideoDecoder
        from ltx_pipelines.utils.helpers import tiling_config_for_vae
        self.torch = torch
        torch.backends.cudnn.deterministic = BENCH['cudnn']['deterministic']
        torch.backends.cudnn.benchmark = BENCH['cudnn']['benchmark']
        self.device = torch.device(device)
        t0 = time.time()
        mode = DiffVAEMode(self.v['diffvae_optimization'])
        dw, dh = self.v['decode_size_wh']
        self.decoders, self.tiling, info = {}, {}, {}
        for d in sorted(decoders):
            path = str(models_dir / DECODER_FILES[d])
            diffusion = bool(is_diffusion_video_vae(path))
            if diffusion != (d == 'diffvae'):
                raise SystemExit(f'{DECODER_FILES[d]}: is_diffusion_video_vae is {diffusion}, expected {d == "diffvae"}')
            self.decoders[d] = VideoDecoder(path, torch.bfloat16, self.device, diffvae_optimization=mode)
            self.tiling[d] = tiling_config_for_vae(path, height=dh, width=dw, num_frames=N_ALL, diffvae_optimization=mode, device=self.device,
                                                   free_bytes=int(self.v['decode_free_bytes']))
            info[d] = dict(checkpoint=DECODER_FILES[d], diffusion_vae=diffusion, tiling=repr(self.tiling[d]))
        self.settings = dict(variant=variant, purpose='decode', diffvae_optimization=self.v['diffvae_optimization'], decode_size_wh=[dw, dh], decoders=info,
                             cudnn_deterministic=bool(torch.backends.cudnn.deterministic), cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
                             tf32_matmul=bool(torch.backends.cuda.matmul.allow_tf32), tf32_cudnn=bool(torch.backends.cudnn.allow_tf32))
        try:
            from ltx_core.model.video_vae.transformer.attention import _NATTEN_AVAILABLE
            self.settings['natten_available'] = bool(_NATTEN_AVAILABLE)
        except Exception:  # noqa: BLE001
            self.settings['natten_available'] = None
        if 'diffvae' in self.decoders and not self.settings['natten_available']:
            raise SystemExit('natten is not available: the DiffVAE would not run the pinned decode')
        self.construct_s = round(time.time() - t0, 1)

    def decode(self, decoder: str, latent, generator):
        """LTX25.decode's call (distilled) with the chosen decoder. Returns [F, H, W, C] float in [0, 1]."""
        torch = self.torch
        chunks = self.decoders[decoder](latent, self.tiling[decoder], generator, dtype=torch.bfloat16,
                                        device_fn=lambda _batch, device=latent.device: device)
        return torch.cat([c.float() for c in chunks], dim=0)


# ----------------------------------------------------------------------------------------------------------------------
def software(model) -> dict:
    out = dict(python=platform.python_version(), ffmpeg=ffmpeg_version(), adapter_sha256=sha256_file(Path(__file__)), pins_sha256=sha256_file(PINS_PATH),
               specgen_sha256=sha256_file(HERE / 'make_ltx25_specs.py'), ltx2_commit=PINS['code']['commit'], hf_revision=PINS['weights']['hf_revision'])
    try:
        import cv2, numpy, torch, torchvision, transformers, safetensors, natten, triton
        out.update(torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(), torchvision=torchvision.__version__,
                   transformers=transformers.__version__, safetensors=safetensors.__version__, natten=natten.__version__, triton=triton.__version__,
                   numpy=numpy.__version__, opencv=cv2.__version__)
    except Exception:  # noqa: BLE001
        pass
    if model is not None:
        out['ltx2_source_digest'] = model.digest
    for key in ('APPTAINER_CONTAINER', 'SINGULARITY_CONTAINER', 'WCB_IMAGE', 'CUDA_VISIBLE_DEVICES', '_CONDOR_JOB_AD'):
        if os.environ.get(key):
            out.setdefault('env', {})[key] = os.environ[key]
    return out


def gpu_info(torch) -> dict:
    """The GPU this process uses (device 0 under CUDA_VISIBLE_DEVICES), its driver and CUDA versions."""
    if not torch.cuda.is_available():
        return dict(name=None)
    p = torch.cuda.get_device_properties(0)
    info = dict(name=p.name, capability=f'{p.major}.{p.minor}', total_gib=round(p.total_memory / GIB, 2), torch_device_count=torch.cuda.device_count(),
                torch=torch.__version__, cuda_runtime=torch.version.cuda, cudnn=torch.backends.cudnn.version())
    uuid = getattr(p, 'uuid', None)
    if uuid is not None:
        info['uuid'] = str(uuid) if str(uuid).startswith('GPU-') else 'GPU-' + str(uuid)
    try:
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


def gpu_free_mib(torch) -> int:
    free, _total = torch.cuda.mem_get_info()
    return int(free // 2**20)


def guard_gpu(torch, keep_free_mib: int, need_mib: int) -> dict:
    """Refuse to start unless the GPU can give need_mib and still keep keep_free_mib free; then cap this process's allocator at
    (free - keep_free - 1 GiB for the CUDA context and libraries), so it can never take the reserve."""
    free, total = torch.cuda.mem_get_info()
    free_mib, total_mib = free // 2**20, total // 2**20
    budget_mib = free_mib - keep_free_mib - 1024
    if budget_mib < need_mib:
        raise SystemExit(f'GPU has {free_mib} MiB free; keeping {keep_free_mib} MiB free leaves {budget_mib} MiB, this step needs about {need_mib} MiB')
    torch.cuda.set_per_process_memory_fraction(budget_mib / total_mib, 0)
    return dict(free_mib_at_start=int(free_mib), total_mib=int(total_mib), keep_free_mib=keep_free_mib, allocator_cap_mib=int(budget_mib))


def run_samples(args) -> int:
    input_root, output_root = args.input_root.resolve(), args.output_root.resolve()
    specs = load_specs(args.specs)
    if not specs:
        raise SystemExit('no specs found')
    for s in specs:
        check_spec(s, input_root, args.variant)
    todo = [s for s in specs if not (args.resume and is_done(s, output_root))]
    if not args.resume:
        clash = [s['name'] for s in specs if (output_root / s['output_dir'] / 'sample_outputs.json').exists()]
        if clash:
            raise SystemExit(f'{len(clash)} samples already have outputs (first: {clash[0]}); use --resume or a fresh output root')
    print(json.dumps(dict(variant=args.variant, specs=len(specs), to_run=len(todo), windows=len({s['window'] for s in specs}), residency=args.residency)),
          flush=True)
    if args.dry_run or not todo:
        return 0
    modes = {s['mode'] for s in specs}
    if 'redecode' in modes:
        if modes != {'redecode'}:
            raise SystemExit('redecode specs run alone (a decode-only process); split the spec set')
        return run_redecode(args, specs, todo, input_root, output_root)
    if args.models_dir is None:
        raise SystemExit('--models-dir is required (or LTX25_MODELS_DIR in the environment or models/ltx25/local.env)')
    if args.prompt_cache is None or not args.prompt_cache.is_file():
        raise SystemExit('--prompt-cache FILE (made by --encode-prompt for this variant) is required')
    avail = mem_available_mib()
    if avail < args.min_host_avail_mib:
        raise SystemExit(f'host MemAvailable {avail} MiB < --min-host-avail-mib {args.min_host_avail_mib}')

    os.environ.setdefault('OMP_NUM_THREADS', str(args.threads))
    import cv2
    import numpy as np
    import torch
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(args.threads)
    if not torch.cuda.is_available():
        raise SystemExit('CUDA is not available')
    guard = guard_gpu(torch, args.keep_free_mib, args.need_mib)
    run_log = dict(schema='wm-calib-bench.ltx25.adapter_run.v2', variant=args.variant, residency=args.residency, started_utc=now_utc(), host=socket.gethostname(),
                   argv=sys.argv, input_root=str(input_root), output_root=str(output_root), specs=len(specs), to_run=len(todo), gpu_guard=guard,
                   host_memory_start=host_memory())
    quantization = spec_quantization(specs)
    model = LTX25(args.variant, args.models_dir.resolve(), args.ltx_src.resolve(), args.device, args.verify_weights, args.allow_unpinned,
                  quantization=quantization)
    model.settings['transformer_residency'] = args.residency
    run_log['quantization'] = quantization
    cache_info = model.load_prompt_cache(args.prompt_cache.resolve())
    sw, gpu = software(model), gpu_info(torch)
    run_log.update(construct_s=model.construct_s, settings=model.settings, weights=model.weight_checks, software=sw, gpu=gpu, prompt_cache=cache_info,
                   samples=[], host_memory_after_construct=host_memory())
    log_dir = output_root / 'adapter-runs'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{datetime.now().strftime('%Y%m%dT%H%M%S')}-{socket.gethostname()}-{os.getpid()}.json"
    print(json.dumps(dict(construct_s=model.construct_s, gpu=gpu.get('name'), guard=guard, tiling=model.settings['decode_tiling'])), flush=True)
    roles = SPECGEN.INPUTS[args.variant]

    recon_prefix: dict = {}   # window -> (canvas frames 0..40 of the reference, its 41-frame history latent) from this process
    failures, stopped, t_run = 0, None, time.time()
    outer = model.pipe.stage.resident() if args.residency == 'resident' else nullcontext()
    with outer:
        for spec in todo:
            if args.stop_after_s and time.time() - t_run > args.stop_after_s:
                stopped = f'time budget: {args.stop_after_s} s reached before {spec["name"]}'
                print(json.dumps(dict(stopped=stopped)), flush=True)
                break
            out_dir = output_root / spec['output_dir']
            out_dir.mkdir(parents=True, exist_ok=True)
            rec = dict(schema=OUT_SCHEMA, variant=args.variant, quantization=quantization, name=spec['name'], window=spec['window'], split=spec['split'], mode=spec['mode'],
                       k=spec.get('k'), seed=spec['seed'], seed_rule=spec.get('seed_rule'), run=spec.get('run'), spec=spec['_file'],
                       spec_sha256=spec['_sha256'], started_utc=now_utc(), host=socket.gethostname(), gpu=gpu, software=sw, generation=spec['generation'],
                       settings_read_back=model.settings, prompt_cache_sha256=cache_info['sha256'], gpu_free_mib_before=gpu_free_mib(torch))
            t_start = time.time()
            try:
                torch.cuda.reset_peak_memory_stats()
                clips, rec['inputs'] = {}, {}
                want = roles['continue_' if spec['mode'] == 'continue' else 'recon']
                for role, (fname, nframes) in want.items():
                    src = input_root / spec['inputs'][role]['path']
                    frames, info = decode_rgb(src, args.threads)
                    if frames.shape[0] != nframes or abs(info['fps'] - FPS) > 1e-6:
                        raise ValueError(f'{role} clip {src}: {frames.shape} at {info["fps"]} fps')
                    rec['inputs'][role] = dict(path=spec['inputs'][role]['path'], file_sha256=sha256_file(src), frames_sha256=sha256_bytes(frames.tobytes()),
                                               size_wh=[info['width'], info['height']])
                    if spec['inputs'][role].get('sha256') and rec['inputs'][role]['file_sha256'] != spec['inputs'][role]['sha256']:
                        raise ValueError(f'{role} clip sha256 differs from the spec')
                    clips[role] = frames
                with torch.no_grad():
                    if spec['mode'] == 'recon':
                        real = clips['real_hr' if args.variant == 'distilled' else 'real']
                        clipf, info_m, timing, hist_lat = model.reconstruct(real, spec['seed'])
                    elif args.variant == 'distilled':
                        clipf, info_m, timing, hist_lat = model.generate_distilled(clips['history'], clips['history_hr'], spec['seed'], args.residency)
                    else:
                        clipf, info_m, timing, hist_lat = model.generate_dev(clips['history'], spec['seed'])
                full = to_u8(torch, clipf)
                del clipf
                dw, dh = model.v['decode_size_wh']
                if full.shape != (N_ALL, dh, dw, 3):
                    raise RuntimeError(f'decoded clip has shape {full.shape}')
                t_ds = time.time()
                clip = to_canvas(full)
                timing['to_canvas'] = round(time.time() - t_ds, 2)
                rec['model'], rec['timing_s'] = info_m, timing
                # diagnostics on the history frames only (nothing is computed over the real future)
                hist_in = clips['history'] if 'history' in clips else to_canvas(clips['real_hr'][:N_HIST]) if 'real_hr' in clips else clips['real'][:N_HIST]
                d_in = np.abs(clip[:N_HIST].astype(np.int16) - hist_in[:N_HIST].astype(np.int16))
                step = lambda a, b: float(np.abs(clip[a].astype(np.int16) - clip[b]).mean())  # noqa: E731
                diag = dict(prefix_vs_input_canvas=dict(native_frames=[0, N_HIST - 1], max_abs=int(d_in.max()), mean_abs=float(d_in.mean()),
                                                        mean_abs_t0=float(d_in[T0_NATIVE].mean())))
                if args.variant == 'distilled':
                    hr_in = clips.get('history_hr', clips.get('real_hr'))
                    d_hr = np.abs(full[:N_HIST].astype(np.int16) - hr_in[:N_HIST].astype(np.int16))
                    diag['prefix_vs_input_1024'] = dict(max_abs=int(d_hr.max()), mean_abs=float(d_hr.mean()), mean_abs_t0=float(d_hr[T0_NATIVE].mean()))
                if spec['mode'] == 'recon':
                    recon_prefix[spec['window']] = (clip[:N_HIST].copy(), hist_lat.float().cpu())
                else:
                    diag['step_mean_abs_native'] = {f'f{a}_f{a + 1}': step(a, a + 1) for a in (T0_NATIVE - 2, T0_NATIVE - 1, T0_NATIVE, T0_NATIVE + 1)}
                    if spec['window'] in recon_prefix:
                        rp, rl = recon_prefix[spec['window']]
                        d_r = np.abs(clip[:N_HIST].astype(np.int16) - rp.astype(np.int16))
                        diag['prefix_vs_recon_in_process'] = dict(identical=bool(d_r.max() == 0), max_abs=int(d_r.max()), mean_abs=float(d_r.mean()),
                                                                  mean_abs_per_frame={str(j): float(d_r[j].mean()) for j in (0, 10, 20, 30, 38, 39, 40)},
                                                                  history_latent_identical=bool(torch.equal(hist_lat.float().cpu(), rl)))
                rec['diagnostics'] = diag
                t3 = time.time()
                files = {}
                outs = [('vision.mp4', clip[0::2], VISION_FPS)]
                if args.keep_native20:
                    outs.append(('native20.mp4', clip, int(FPS)))
                if args.keep_full_res and args.variant == 'distilled':
                    outs.append(('full1024.mp4', full, int(FPS)))
                for name, arr, fps in outs:
                    p = out_dir / name
                    write_lossless(arr, p, fps, args.threads)
                    back, vinfo = decode_rgb(p, args.threads)
                    if not np.array_equal(back, arr):
                        raise RuntimeError(f'{name} does not decode to the written frames')
                    files[name] = dict(sha256=sha256_file(p), bytes=p.stat().st_size, frames=vinfo['frames'], fps=vinfo['fps'], width=vinfo['width'],
                                       height=vinfo['height'], pix_fmt=vinfo['pix_fmt'], frames_sha256=sha256_bytes(arr.tobytes()))
                if files['vision.mp4']['frames'] != VISION_FRAMES or (files['vision.mp4']['width'], files['vision.mp4']['height']) != (CW, CH):
                    raise RuntimeError(f'vision.mp4 is {files["vision.mp4"]}')
                # latents.safetensors: the final video latent, the audio latent, and the DiffVAE generator state before decoding
                from safetensors.torch import save_file
                lat = {k: v.detach().to('cpu').contiguous() for k, v in model.last_latents.items() if v is not None}
                lat['decoder_rng_state'] = model.last_decoder_rng_state
                lmeta = dict(schema='wm-calib-bench.ltx25.latents.v1', name=spec['name'], window=spec['window'], mode=spec['mode'], seed=spec['seed'],
                             variant=args.variant, quantization=quantization, shapes={k: list(v.shape) for k, v in lat.items()},
                             vision_frames_sha256=files['vision.mp4']['frames_sha256'],
                             note='decoder_rng_state is torch.Generator.get_state() of the run generator right before the DiffVAE call')
                lp = out_dir / 'latents.safetensors'
                save_file(lat, str(lp.with_name(lp.name + '.part')), metadata={'bench': json.dumps(lmeta)})
                os.replace(lp.with_name(lp.name + '.part'), lp)
                files['latents.safetensors'] = dict(sha256=sha256_file(lp), bytes=lp.stat().st_size, shapes=lmeta['shapes'],
                                                    video_latent_sha256=sha256_bytes(lat['video_latent'].float().numpy().tobytes()))
                rec['timing_s']['write'] = round(time.time() - t3, 2)
                rec.update(status='success', files=files, video='vision.mp4', video_sha256=files['vision.mp4']['sha256'],
                           full_res_frames_sha256=sha256_bytes(full.tobytes()), prefix21_frames_sha256=sha256_bytes(clip[0:N_HIST:2].tobytes()),
                           layout=OUTP['files'])
            except Exception as exc:  # noqa: BLE001
                failures += 1
                rec.update(status='failure', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
                print(rec['traceback'], file=sys.stderr, flush=True)
            rec['gpu_memory_gib'] = dict(max_allocated=round(torch.cuda.max_memory_allocated() / GIB, 2), max_reserved=round(torch.cuda.max_memory_reserved() / GIB, 2))
            rec['gpu_free_mib_after'] = gpu_free_mib(torch)
            rec['host_memory_gib'] = host_memory()
            rec['timing_s'] = dict(rec.get('timing_s', {}), total=round(time.time() - t_start, 2))
            rec['finished_utc'] = now_utc()
            write_json_atomic(out_dir / 'sample_outputs.json', rec)
            run_log['samples'].append(dict(name=spec['name'], run=spec.get('run'), status=rec['status'], timing_s=rec['timing_s'],
                                           gpu_memory_gib=rec.get('gpu_memory_gib'), host_peak_rss_gib=rec['host_memory_gib'].get('peak_rss'),
                                           gpu_free_mib_after=rec['gpu_free_mib_after']))
            write_json_atomic(log_path, run_log)
            print(json.dumps(dict(sample=spec['name'], run=spec.get('run'), status=rec['status'], **rec['timing_s'],
                                  gpu_max_reserved_gib=rec['gpu_memory_gib']['max_reserved'], gpu_free_mib=rec['gpu_free_mib_after'])), flush=True)
            if rec['gpu_free_mib_after'] < args.keep_free_mib:
                stopped = f'GPU free memory {rec["gpu_free_mib_after"]} MiB < {args.keep_free_mib} MiB'
            elif mem_available_mib() < 4096:
                stopped = f'host MemAvailable {mem_available_mib()} MiB < 4096 MiB'
            if stopped:
                print(json.dumps(dict(stopped=stopped)), flush=True)
                break
    run_log.update(finished_utc=now_utc(), failures=failures, stopped=stopped, host_memory_end=host_memory(),
                   gpu_memory_peak_gib=max((x['gpu_memory_gib'] or {}).get('max_reserved', 0) for x in run_log['samples']) if run_log['samples'] else None)
    write_json_atomic(log_path, run_log)
    if stopped and stopped.startswith('time budget'):
        return 5
    return 3 if stopped else (1 if failures else 0)


def run_redecode(args, specs: list, todo: list, input_root: Path, output_root: Path) -> int:
    """Decode-only run (every spec mode redecode): each spec re-decodes a saved latent (latents.safetensors of a DEV sample) with
    the DiffVAE or the conv decoder; the generator is restored to the sample's saved decoder state (rng saved) or freshly seeded
    (rng seed). Writes vision.mp4 (and with --keep-native20 native20.mp4) and sample_outputs.json as the generation runs do."""
    if args.models_dir is None:
        raise SystemExit('--models-dir is required (or LTX25_MODELS_DIR in the environment or models/ltx25/local.env)')
    avail = mem_available_mib()
    if avail < args.min_host_avail_mib:
        raise SystemExit(f'host MemAvailable {avail} MiB < --min-host-avail-mib {args.min_host_avail_mib}')
    os.environ.setdefault('OMP_NUM_THREADS', str(args.threads))
    import cv2
    import numpy as np
    import torch
    from safetensors import safe_open
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(args.threads)
    if not torch.cuda.is_available():
        raise SystemExit('CUDA is not available')
    guard = guard_gpu(torch, args.keep_free_mib, args.need_mib)
    decs = Decoders(args.variant, args.models_dir.resolve(), args.ltx_src.resolve(), args.device, args.verify_weights, args.allow_unpinned,
                    {s['decoder'] for s in specs})
    sw, gpu = software(decs), gpu_info(torch)
    run_log = dict(schema='wm-calib-bench.ltx25.adapter_run.v2', variant=args.variant, purpose='redecode', started_utc=now_utc(), host=socket.gethostname(),
                   argv=sys.argv, input_root=str(input_root), output_root=str(output_root), specs=len(specs), to_run=len(todo), gpu_guard=guard,
                   host_memory_start=host_memory(), construct_s=decs.construct_s, settings=decs.settings, weights=decs.weight_checks, software=sw,
                   gpu=gpu, samples=[])
    log_dir = output_root / 'adapter-runs'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{datetime.now().strftime('%Y%m%dT%H%M%S')}-{socket.gethostname()}-{os.getpid()}.json"
    print(json.dumps(dict(construct_s=decs.construct_s, gpu=gpu.get('name'), guard=guard, decoders=decs.settings['decoders'])), flush=True)
    failures, stopped, t_run = 0, None, time.time()
    for spec in todo:
        if args.stop_after_s and time.time() - t_run > args.stop_after_s:
            stopped = f'time budget: {args.stop_after_s} s reached before {spec["name"]}'
            print(json.dumps(dict(stopped=stopped)), flush=True)
            break
        out_dir = output_root / spec['output_dir']
        out_dir.mkdir(parents=True, exist_ok=True)
        src = spec['source']
        rec = dict(schema=OUT_SCHEMA, variant=args.variant, quantization='none', name=spec['name'], window=spec['window'], split=spec['split'],
                   mode='redecode', k=spec.get('k'), seed=spec['seed'], seed_rule=spec.get('seed_rule'), rng=spec['rng'], decoder=spec['decoder'],
                   source=src, run=spec.get('run'), spec=spec['_file'], spec_sha256=spec['_sha256'], started_utc=now_utc(), host=socket.gethostname(),
                   gpu=gpu, software=sw, generation=spec['generation'], settings_read_back=decs.settings, gpu_free_mib_before=gpu_free_mib(torch))
        t_start = time.time()
        try:
            torch.cuda.reset_peak_memory_stats()
            timing = {}
            lp = input_root / spec['inputs']['latents']['path']
            lsha = sha256_file(lp)
            if lsha != src['latents_sha256']:
                raise ValueError(f'{lp.name}: sha256 {lsha}, the source record says {src["latents_sha256"]}')
            with safe_open(str(lp), framework='pt', device='cpu') as f:
                lmeta = json.loads((f.metadata() or {}).get('bench', '{}'))
                latent = f.get_tensor('video_latent')
                state = f.get_tensor('decoder_rng_state') if spec['rng'] == 'saved' else None
            if lmeta.get('name') != src['name'] or lmeta.get('window') != spec['window']:
                raise ValueError(f'{lp.name} holds {lmeta.get("name")} of {lmeta.get("window")}, the spec names {src["name"]}')
            if lmeta.get('variant') != 'distilled' or lmeta.get('quantization') != 'none':
                raise ValueError(f'{lp.name}: variant {lmeta.get("variant")}, quantization {lmeta.get("quantization")} (DEV analysis runs are distilled bf16)')
            rec['inputs'] = dict(latents=dict(path=spec['inputs']['latents']['path'], file_sha256=lsha, video_latent_shape=list(latent.shape),
                                              video_latent_dtype=str(latent.dtype)))
            generator = torch.Generator(device=decs.device)
            if spec['rng'] == 'saved':
                generator.set_state(state)
            else:
                generator.manual_seed(int(spec['seed']))
            t1 = time.time()
            with torch.no_grad():
                frames = decs.decode(spec['decoder'], latent.to(decs.device), generator)
            timing['decode'] = round(time.time() - t1, 2)
            full = to_u8(torch, frames)
            del frames
            dw, dh = decs.v['decode_size_wh']
            if full.shape != (N_ALL, dh, dw, 3):
                raise RuntimeError(f'decoded clip has shape {full.shape}')
            t2 = time.time()
            clip = to_canvas(full)
            timing['to_canvas'] = round(time.time() - t2, 2)
            t3 = time.time()
            files = {}
            outs = [('vision.mp4', clip[0::2], VISION_FPS)] + ([('native20.mp4', clip, int(FPS))] if args.keep_native20 else [])
            for name, arr, fps in outs:
                vp = out_dir / name
                write_lossless(arr, vp, fps, args.threads)
                back, vinfo = decode_rgb(vp, args.threads)
                if not np.array_equal(back, arr):
                    raise RuntimeError(f'{name} does not decode to the written frames')
                files[name] = dict(sha256=sha256_file(vp), bytes=vp.stat().st_size, frames=vinfo['frames'], fps=vinfo['fps'], width=vinfo['width'],
                                   height=vinfo['height'], pix_fmt=vinfo['pix_fmt'], frames_sha256=sha256_bytes(arr.tobytes()))
            if files['vision.mp4']['frames'] != VISION_FRAMES or (files['vision.mp4']['width'], files['vision.mp4']['height']) != (CW, CH):
                raise RuntimeError(f'vision.mp4 is {files["vision.mp4"]}')
            timing['write'] = round(time.time() - t3, 2)
            full_sha = sha256_bytes(full.tobytes())
            model = dict(full_res_frames_sha256=full_sha, vision_frames_sha256=files['vision.mp4']['frames_sha256'])
            if spec['rng'] == 'saved' and spec['decoder'] == 'diffvae':
                model['reproduces_source'] = full_sha == src.get('full_res_frames_sha256')
            rec.update(status='success', files=files, video='vision.mp4', video_sha256=files['vision.mp4']['sha256'], full_res_frames_sha256=full_sha,
                       model=model, timing_s=timing, layout=OUTP['files'])
        except Exception as exc:  # noqa: BLE001
            failures += 1
            rec.update(status='failure', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
            print(rec['traceback'], file=sys.stderr, flush=True)
        rec['gpu_memory_gib'] = dict(max_allocated=round(torch.cuda.max_memory_allocated() / GIB, 2), max_reserved=round(torch.cuda.max_memory_reserved() / GIB, 2))
        rec['gpu_free_mib_after'] = gpu_free_mib(torch)
        rec['host_memory_gib'] = host_memory()
        rec['timing_s'] = dict(rec.get('timing_s', {}), total=round(time.time() - t_start, 2))
        rec['finished_utc'] = now_utc()
        write_json_atomic(out_dir / 'sample_outputs.json', rec)
        run_log['samples'].append(dict(name=spec['name'], run=spec.get('run'), status=rec['status'], timing_s=rec['timing_s'],
                                       gpu_memory_gib=rec.get('gpu_memory_gib'), host_peak_rss_gib=rec['host_memory_gib'].get('peak_rss'),
                                       gpu_free_mib_after=rec['gpu_free_mib_after']))
        write_json_atomic(log_path, run_log)
        print(json.dumps(dict(sample=spec['name'], run=spec.get('run'), status=rec['status'], **rec['timing_s'],
                              gpu_max_reserved_gib=rec['gpu_memory_gib']['max_reserved'], gpu_free_mib=rec['gpu_free_mib_after'])), flush=True)
        if rec['gpu_free_mib_after'] < args.keep_free_mib:
            stopped = f'GPU free memory {rec["gpu_free_mib_after"]} MiB < {args.keep_free_mib} MiB'
        elif mem_available_mib() < 4096:
            stopped = f'host MemAvailable {mem_available_mib()} MiB < 4096 MiB'
        if stopped:
            print(json.dumps(dict(stopped=stopped)), flush=True)
            break
    run_log.update(finished_utc=now_utc(), failures=failures, stopped=stopped, host_memory_end=host_memory(),
                   gpu_memory_peak_gib=max((x['gpu_memory_gib'] or {}).get('max_reserved', 0) for x in run_log['samples']) if run_log['samples'] else None)
    write_json_atomic(log_path, run_log)
    if stopped and stopped.startswith('time budget'):
        return 5
    return 3 if stopped else (1 if failures else 0)


def run_encode_prompt(args) -> int:
    avail = mem_available_mib()
    if avail < args.min_host_avail_mib:
        raise SystemExit(f'host MemAvailable {avail} MiB < --min-host-avail-mib {args.min_host_avail_mib}')
    import torch
    torch.set_num_threads(args.threads)
    if args.models_dir is None:
        raise SystemExit('--models-dir is required (or LTX25_MODELS_DIR in the environment or models/ltx25/local.env)')
    guard = guard_gpu(torch, args.keep_free_mib, args.need_mib if args.te_offload == 'none' else 6144)
    model = LTX25(args.variant, args.models_dir.resolve(), args.ltx_src.resolve(), args.device, args.verify_weights, args.allow_unpinned,
                  te_offload=args.te_offload, purpose='encode-prompt')
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        meta = model.encode_prompt(args.encode_prompt.resolve())
    meta.update(gpu=gpu_info(torch), gpu_guard=guard, gpu_memory_gib=dict(max_reserved=round(torch.cuda.max_memory_reserved() / GIB, 2)),
                host_memory_gib=host_memory(), software=software(model), te_offload=args.te_offload)
    args.encode_prompt.resolve().with_suffix('.json').write_text(json.dumps(meta, indent=1) + '\n')
    print(json.dumps({k: meta[k] for k in ('variant', 'file_sha256', 'seconds', 'shapes', 'gpu_memory_gib', 'host_memory_gib')}), flush=True)
    return 0


# ----------------------------------------------------------------------------------------------------------------------
def self_test(args) -> int:
    """CPU checks that need no weights: pinned defaults vs the installed code (both variants), latent geometry, the
    conditioning masks at both distilled stages, the official samplers keeping pinned tokens clean (deterministic Euler and
    the ancestral loop with noise injection), the preprocessing identity, the RGB24 conversion, the INTER_AREA canvas step,
    the writers, the native -> vision mapping, the local.env parser, the redecode spec rules, the specs."""
    import tempfile
    import cv2
    import numpy as np
    import torch
    from ltx_core.color.yuv import FrameConverter
    from ltx_core.components.noisers import GaussianNoiser
    from ltx_core.components.patchifiers import VideoLatentPatchifier
    from ltx_core.components.schedulers import LTX2Scheduler
    from ltx_core.conditioning.types.latent_cond import VideoConditionByLatentIndex
    from ltx_core.tools import VideoLatentTools
    from ltx_core.types import VIDEO_SCALE_FACTORS, AudioLatentShape, VideoLatentShape, VideoPixelShape
    from ltx_pipelines import distilled as D
    from ltx_pipelines.utils.constants import DEFAULT_NEGATIVE_PROMPT, DISTILLED_SIGMA_VALUES, LTX_2_4_PARAMS, STAGE_2_DISTILLED_SIGMA_VALUES
    from ltx_pipelines.utils.helpers import create_initial_av_latents, create_noised_state, post_process_latent, snap_frames_to_grid
    from ltx_pipelines.utils.media_io.decode import video_preprocess
    from ltx_pipelines.utils.samplers import euler_ancestral_denoising_loop
    from ltx_pipelines.utils.types import DenoisedLatentResult, VideoAudio
    torch.manual_seed(0)
    res, ok = {}, True
    vd, vv = VARIANTS['distilled'], VARIANTS['dev']

    def check(name, cond, detail=None):
        nonlocal ok
        res[name] = dict(ok=bool(cond), detail=detail)
        ok &= bool(cond)

    digest, n = source_digest(args.ltx_src.resolve())
    check('source_digest', digest == PINS['code']['source_digest'], dict(digest=digest, files=n))
    # distilled defaults
    close = lambda a, b: len(a) == len(b) and all(abs(float(x) - float(y)) < 1e-9 for x, y in zip(a, b))  # noqa: E731
    check('distilled_sigmas', close(DISTILLED_SIGMA_VALUES, vd['stage_1']['sigmas']) and close(STAGE_2_DISTILLED_SIGMA_VALUES, vd['stage_2']['sigmas'])
          and len(vd['stage_1']['sigmas']) - 1 == vd['stage_1']['steps'] and len(vd['stage_2']['sigmas']) - 1 == vd['stage_2']['steps'],
          dict(stage_1=list(DISTILLED_SIGMA_VALUES), stage_2=list(STAGE_2_DISTILLED_SIGMA_VALUES)))
    check('distilled_ancestral_rule', D.ANCESTRAL_SAMPLER_SINCE_VERSION == (2, 5) and D.ANCESTRAL_ETA == vd['sampler']['eta']
          and D.ANCESTRAL_S_NOISE == vd['sampler']['s_noise'] and D.ANCESTRAL_NOISE_SEED_OFFSET == vd['stage_1']['ancestral_noise_seed_offset']
          and D.ANCESTRAL_STAGE_2_NOISE_SEED_OFFSET == vd['stage_2']['ancestral_noise_seed_offset'],
          dict(since=D.ANCESTRAL_SAMPLER_SINCE_VERSION, eta=D.ANCESTRAL_ETA, s_noise=D.ANCESTRAL_S_NOISE,
               offsets=[D.ANCESTRAL_NOISE_SEED_OFFSET, D.ANCESTRAL_STAGE_2_NOISE_SEED_OFFSET]))
    fw, fh = vd['final_size_wh']
    check('distilled_size_rule', fw % 64 == 0 and fh % 64 == 0 and [fw // 2, fh // 2] == vd['stage_1']['size_wh'] == BENCH['canvas_wh']
          and snap_frames_to_grid(N_ALL) == N_ALL, dict(final=[fw, fh], stage_1=vd['stage_1']['size_wh']))
    # dev defaults
    p = LTX_2_4_PARAMS
    g = lambda gp: dict(cfg_scale=gp.cfg_scale, stg_scale=gp.stg_scale, stg_blocks=list(gp.stg_blocks), rescale_scale=gp.rescale_scale,  # noqa: E731
                        modality_scale=gp.modality_scale, skip_step=gp.skip_step)
    check('dev_defaults_LTX_2_4_PARAMS', p.num_inference_steps == vv['num_inference_steps'] and g(p.video_guider_params) == vv['video_guider']
          and g(p.audio_guider_params) == vv['audio_guider'] and sha256_bytes(DEFAULT_NEGATIVE_PROMPT.encode()) == vv['negative_prompt_sha256'])
    sig = LTX2Scheduler().execute(steps=vv['num_inference_steps'])
    check('dev_scheduler', len(sig) == vv['num_inference_steps'] + 1 and abs(float(sig[0]) - 1.0) < 1e-6 and float(sig[-1]) == 0.0
          and abs(float(sig[-2]) - 0.1) < 1e-6)
    check('prompt_sha256', sha256_bytes(BENCH['prompt'].encode()) == BENCH['prompt_sha256'], dict(words=len(BENCH['prompt'].split())))

    # geometry and conditioning masks at both stage sizes
    for tag, (w, h) in (('stage_1', vd['stage_1']['size_wh']), ('stage_2', vd['stage_2']['size_wh'])):
        shape = VideoLatentShape.from_pixel_shape(VideoPixelShape(batch=1, frames=N_ALL, height=h, width=w, fps=FPS), scale_factors=VIDEO_SCALE_FACTORS)
        hshape = VideoLatentShape.from_pixel_shape(VideoPixelShape(batch=1, frames=N_HIST, height=h, width=w, fps=FPS), scale_factors=VIDEO_SCALE_FACTORS)
        want = vd[tag]['latent_thw']
        vlat, alat = create_initial_av_latents(width=w, height=h, frames=N_ALL, fps=FPS, device=torch.device('cpu'), dtype=torch.bfloat16,
                                               video_scale_factors=VIDEO_SCALE_FACTORS)
        tools = VideoLatentTools(VideoLatentPatchifier(patch_size=1), VideoLatentShape.from_torch_shape(vlat.shape), FPS, scale_factors=VIDEO_SCALE_FACTORS)
        hist = torch.randn(1, 128, N_LAT_HIST, want[1], want[2]).to(torch.bfloat16)
        state = create_noised_state(tools=tools, conditionings=[VideoConditionByLatentIndex(latent=hist, strength=1.0, latent_idx=0)],
                                    noiser=GaussianNoiser(generator=torch.Generator().manual_seed(714236784)), dtype=torch.bfloat16,
                                    device=torch.device('cpu'), noise_scale=1.0 if tag == 'stage_1' else vd['stage_2']['sigmas'][0], initial_latent=vlat)
        ntok = N_LAT_HIST * want[1] * want[2]
        mask = state.denoise_mask.tensor
        hist_tok = VideoLatentPatchifier(patch_size=1).patchify(hist)
        check(f'{tag}_geometry_and_mask', [shape.frames, shape.height, shape.width] == want and hshape.frames == N_LAT_HIST
              and shape.frames * shape.height * shape.width == vd[tag]['video_tokens'] and bool((mask[:, :ntok] == 0).all()) and bool((mask[:, ntok:] == 1).all())
              and torch.equal(state.latent[:, :ntok].float(), hist_tok.float()),
              dict(latent=list(vlat.shape), audio=list(alat.shape), tokens=int(state.latent.shape[1]), pinned_tokens=ntok))
    a_ref = AudioLatentShape.from_duration(batch=1, duration=N_ALL / FPS)
    check('audio_latent', list(alat.shape) == list(a_ref.to_torch_shape()), dict(audio=list(alat.shape), seconds=N_ALL / FPS))

    # the official samplers keep pinned tokens clean: post_process_latent, and the ancestral loop with eta 1 on a tiny state
    den = torch.randn_like(state.latent.float()).to(state.latent.dtype)
    pp = post_process_latent(den, state.denoise_mask.tensor, state.clean_latent)
    check('post_process_keeps_history', torch.equal(pp[:, :ntok].float(), hist_tok.float()))
    tiny = torch.zeros(1, 128, 3, 2, 2, dtype=torch.bfloat16)
    ttools = VideoLatentTools(VideoLatentPatchifier(patch_size=1), VideoLatentShape.from_torch_shape(tiny.shape), FPS, scale_factors=VIDEO_SCALE_FACTORS)
    thist = torch.randn(1, 128, 1, 2, 2).to(torch.bfloat16)
    tstate = create_noised_state(tools=ttools, conditionings=[VideoConditionByLatentIndex(latent=thist, strength=1.0, latent_idx=0)],
                                 noiser=GaussianNoiser(generator=torch.Generator().manual_seed(1)), dtype=torch.bfloat16, device=torch.device('cpu'),
                                 noise_scale=1.0, initial_latent=tiny)
    fake = lambda _tr, vs, _as, _sig, _i: VideoAudio(video=DenoisedLatentResult(denoised=torch.randn_like(vs.latent.float()).to(vs.latent.dtype)))  # noqa: E731
    sigmas = torch.tensor(vd['stage_1']['sigmas'], dtype=torch.float32)
    v_out, _ = euler_ancestral_denoising_loop(sigmas, tstate, None, None, fake, noise_seed=714236784 + 10000, eta=1.0, s_noise=1.0)
    ntiny = 1 * 2 * 2
    check('ancestral_loop_keeps_history', torch.equal(v_out.latent[:, :ntiny].float(), VideoLatentPatchifier(patch_size=1).patchify(thist).float())
          and not torch.equal(v_out.latent[:, ntiny:], tstate.latent[:, ntiny:]), dict(steps=len(sigmas) - 1, pinned_tokens=ntiny))

    u8 = np.random.default_rng(0).integers(0, 256, size=(N_HIST, CH, CW, 3), dtype=np.uint8)
    x = video_preprocess((torch.from_numpy(f).unsqueeze(0) for f in u8), CH, CW, torch.float32, torch.device('cpu'))
    ref = (torch.from_numpy(u8).permute(3, 0, 1, 2).unsqueeze(0).float() / 127.5 - 1.0)
    check('video_preprocess_identity', tuple(x.shape) == (1, 3, N_HIST, CH, CW) and torch.equal(x, ref), dict(shape=list(x.shape)))
    f = torch.rand(5, fh, fw, 3) * 1.2 - 0.1
    ours = to_u8(torch, f)
    official = FrameConverter.RGB24(f.clone().movedim(-1, -3)).numpy()
    check('rgb24_conversion', np.array_equal(ours, official))
    canvas = to_canvas(ours)
    check('to_canvas_inter_area', canvas.shape == (5, CH, CW, 3) and np.array_equal(canvas[0], cv2.resize(ours[0], (CW, CH), interpolation=cv2.INTER_AREA))
          and np.array_equal(to_canvas(canvas), canvas), dict(opencv=cv2.__version__))
    with tempfile.TemporaryDirectory() as td:
        clip = np.random.default_rng(1).integers(0, 256, size=(N_ALL, CH, CW, 3), dtype=np.uint8)
        for name, arr, fps in (('native20.mp4', clip, int(FPS)), ('vision.mp4', clip[0::2], VISION_FPS)):
            pth = Path(td) / name
            write_lossless(arr, pth, fps, args.threads)
            back, info = decode_rgb(pth, args.threads)
            check(f'writer_{name}', np.array_equal(back, arr) and info['frames'] == len(arr) and abs(info['fps'] - fps) < 1e-6,
                  dict(frames=info['frames'], fps=info['fps']))
    t_native = [(j - T0_NATIVE) * 0.05 for j in range(N_ALL)]
    t_vision = [(i - T0_VISION) * 0.1 for i in range(VISION_FRAMES)]
    check('vision_mapping', all(abs(t_native[2 * i] - t_vision[i]) < 1e-9 for i in range(VISION_FRAMES)), dict(vision_t0_index=T0_VISION))
    with tempfile.TemporaryDirectory() as td:
        # machine-specific defaults: the environment variable wins, else KEY=VALUE from local.env
        envf = Path(td) / 'local.env'
        envf.write_text('# comment\nexport WCB_SELFTEST_A=/x/y  # trailing comment\nWCB_SELFTEST_B = "q r"\nWCB_SELFTEST_C=from-file\n')
        os.environ['WCB_SELFTEST_C'] = 'from-env'
        try:
            got = [local_setting(k, envf) for k in ('WCB_SELFTEST_A', 'WCB_SELFTEST_B', 'WCB_SELFTEST_C', 'WCB_SELFTEST_D')]
        finally:
            del os.environ['WCB_SELFTEST_C']
        check('local_settings', got == ['/x/y', 'q r', 'from-env', None], dict(got=got))
        # redecode specs: one valid spec per decoder passes; a saved-state spec with a seed, a seeded spec without one, or a decode
        # block that differs from the pins is rejected
        lat = Path(td) / 'latents' / 'ltx25d-dev-recon' / 'w-recon.safetensors'
        lat.parent.mkdir(parents=True)
        lat.write_bytes(b'x')
        base = dict(schema=SPEC_SCHEMA, variant='distilled', split='dev', window='w', mode='redecode', k=None, name='w-recon-rep', run='ltx25d-dev-redecode',
                    output_dir='ltx25d-dev-redecode/native/w-recon-rep', inputs=dict(latents=dict(path='latents/ltx25d-dev-recon/w-recon.safetensors')),
                    source=dict(run='ltx25d-dev-recon', name='w-recon', latents_sha256='0' * 64))
        good = [dict(base, decoder=d, rng=r, seed=sd, generation=SPECGEN.decode_block('distilled', d)) for d, r, sd in (('diffvae', 'saved', None), ('conv', 'seed', 7))]
        bad = [dict(good[0], seed=7), dict(good[1], seed=None), dict(good[0], generation=dict(good[0]['generation'], decode_free_bytes=1))]
        valid = rejected = 0
        for sp in good:
            try:
                check_spec(sp, Path(td))
                valid += 1
            except Exception:  # noqa: BLE001
                pass
        for sp in bad:
            try:
                check_spec(sp, Path(td))
            except Exception:  # noqa: BLE001
                rejected += 1
        check('redecode_specs', valid == len(good) and rejected == len(bad), dict(valid=valid, rejected=rejected))
    if args.specs:
        specs = load_specs(args.specs)
        errs, by_variant = [], {}
        for s in specs:
            try:
                check_spec(s, args.input_root.resolve())
                by_variant[s['variant']] = by_variant.get(s['variant'], 0) + 1
            except Exception as exc:  # noqa: BLE001
                errs.append(f'{s["name"]}: {exc}')
        check('specs', not errs and bool(specs), dict(specs=len(specs), by_variant=by_variant, errors=errs[:5]))
    print(json.dumps(dict(self_test='PASS' if ok else 'FAIL', checks=res), indent=1))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--specs', nargs='+', help='spec JSON files or directories (searched recursively)')
    ap.add_argument('--input-root', type=Path)
    ap.add_argument('--output-root', type=Path)
    ap.add_argument('--variant', choices=sorted(VARIANTS), default=PINS['default_variant'])
    ap.add_argument('--residency', help='distilled: per-call | per-sample | resident; dev: per-sample | resident (default: the variant\'s pinned default)')
    ap.add_argument('--models-dir', type=Path, default=default_models_dir(),
                    help='pinned weights in the Hugging Face layout (default: LTX25_MODELS_DIR, environment or models/ltx25/local.env)')
    ap.add_argument('--ltx-src', type=Path, default=default_ltx_src(),
                    help='LTX-2 checkout at the pinned commit (default: LTX25_SRC, else $LTX25_ENV/src/LTX-2; environment or local.env)')
    ap.add_argument('--prompt-cache', type=Path, help='prompt cache made by --encode-prompt for this variant')
    ap.add_argument('--encode-prompt', type=Path, metavar='FILE', help='encode the fixed prompt (and, dev, the negative prompt) and write the cache to FILE')
    ap.add_argument('--te-offload', choices=('none', 'disk'), default='none', help='--encode-prompt only: stream the text encoder from disk')
    ap.add_argument('--self-test', action='store_true', help='CPU checks without weights')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--resume', action='store_true', help='skip samples already finished (status success, every output sha256 matches)')
    ap.add_argument('--dry-run', action='store_true', help='check the specs and inputs, run nothing')
    ap.add_argument('--keep-native20', action='store_true', help='also write native20.mp4 (the 161 canvas frames at 20 fps, lossless)')
    ap.add_argument('--keep-full-res', action='store_true', help='distilled: also write full1024.mp4 (the 1024x576 decode, lossless)')
    ap.add_argument('--keep-free-mib', type=int, default=6144, help='GPU memory that must stay free for other users (default 6 GiB)')
    ap.add_argument('--need-mib', type=int, default=None,
                    help='GPU memory this process needs; refuse to start below it (default 24576; resident residency 61440: bf16 on an 80-141 GB GPU)')
    ap.add_argument('--min-host-avail-mib', type=int, default=10240, help='refuse to start if host MemAvailable is lower')
    ap.add_argument('--stop-after-s', type=int, default=0, help='start no new sample after this many seconds (0: no limit); exit code 5')
    ap.add_argument('--verify-weights', choices=('size', 'sha256'), default='size', help='checkpoint check before loading')
    ap.add_argument('--threads', type=int, default=int(os.environ.get('OMP_NUM_THREADS', '2')), help='CPU threads for torch, OpenCV and ffmpeg')
    ap.add_argument('--allow-unpinned', action='store_true', help='development only: accept other code or weights (recorded)')
    args = ap.parse_args()
    args.residency = args.residency or VARIANTS[args.variant]['residency_default']
    if args.residency not in RESIDENCY[args.variant]:
        ap.error(f'--residency {args.residency} is not available for the {args.variant} variant {RESIDENCY[args.variant]}')
    if args.need_mib is None:
        args.need_mib = 61440 if (args.residency == 'resident' and not args.encode_prompt) else 24576
    if args.ltx_src is None:
        ap.error('--ltx-src is required (or LTX25_SRC / LTX25_ENV in the environment or models/ltx25/local.env)')
    if args.self_test:
        sys.exit(self_test(args))
    if args.encode_prompt:
        sys.exit(run_encode_prompt(args))
    if not (args.specs and args.input_root and args.output_root):
        ap.error('--specs, --input-root and --output-root are required')
    sys.exit(run_samples(args))


if __name__ == '__main__':
    main()
