#!/usr/bin/env python3
"""Wan2.1-VACE-1.3B adapter for bench-v1 (table T1-H2): spec JSONs -> vision.mp4 + sample_outputs.json.

Official code path (Wan-Video/Wan2.1 at the pinned commit, read-only checkout on sys.path; nothing is copied or edited):
wan.WanVace (task vace-1.3B, the t2v_1_3B config) and its generate(), i.e. the body of generate.py --task vace-1.3B,
called once per extension round with the first-clip task of VACE (vace/annotators/frameref.py mode 'firstclip': the
given clip with mask 0 = kept, then gray frames with mask 1 = generated; the VACE user guide's procedure for long videos:
5 s clips one by one, each a first-clip extension of the last):
  round 1  src_video = the window's 33 history frames (16 fps, t0 - 2.0 .. t0) + 48 gray (127) frames, src_mask 0 x 33 +
           255 x 48 -> 81 frames (frame_num 81);
  round 2  src_video = round 1's output frames 48-80 (t0 + 1.0 .. t0 + 3.0) + 48 gray frames, the same mask -> 81 frames.
The src tensors are built in memory exactly as WanVace.prepare_source builds them from a lossless file pair (the official
VaceVideoProcessor.resize_crop on the uint8 frames, then .to(device), mask (m[:1] + 1) / 2); --check-source-equivalence
writes the pair losslessly and compares with prepare_source itself. Sampler as released for vace tasks: UniPC flow
solver, 50 steps, shift 16, guidance 5.0, context scale 1.0, the config's negative prompt, seed per round through
generate(seed=...) (torch.Generator(device).manual_seed). The official generate() runs unchanged with two substitutions
in the wan.vace module namespace, neither of which changes a number: T5EncoderModel -> the prompt cache (the official T5
call's output tensors, made once by --make-prompt-cache; generate() calls it as it calls the T5) and
FlowUniPCMultistepScheduler -> a subclass that records the timesteps and sigmas it sets (the settings read back).
offload_model is False (the DiT stays on the GPU between samples; no numeric effect). VACE regenerates the masked-0
prefix (it is not pinned): the adapter records, per round, how far the regenerated prefix lies from its conditioning
(latents and pixels).

Modes (one spec per sample; make_vace_specs.py):
  prefix    the window's 33 history frames through the Wan VAE alone (encode, decode, the official uint8 conversion of
            cache_video): prefix33.mp4 (on the canvas, lossless, 16 fps). Run first in every window; frames 0-20 of every
            clip of the window (continuations and the reference) are copied from it, so they are byte-identical;
  continue  the two rounds above; seeds seed_of(w, k) and seed_of(w, f'{k}|r2');
  recon     the real reference: the window's 129 real frames through the same VAE, window by window as the two rounds
            (frames 0-80 and 48-128 each encoded and decoded as one 81-frame clip); no randomness.

Output (OUT/native/<window>-<sample>/, sample s<k>, recon or prefix):
  vision.mp4           81 frames at 10 fps, 512 x 288, lossless (libx264rgb -crf 0): frame i = tick k = i - 20 = global
                       16 fps frame 32 + round(1.6 k): k <= 0 from the window's prefix, k = 1..30 from round 1, 31..60 from
                       round 2 (generated at 512 x 288: as is; at 832 x 480: cv2.resize(..., (512, 288), INTER_AREA))
  prefix33.mp4         (prefix mode) the 33 prefix frames on the canvas, 16 fps; prefix33_native.mp4 at 832 x 480 runs
  native16.mp4         optional (--keep native16): the official outputs joined on the canvas at 16 fps (round 1's 81 frames,
                       round 2's 48 new frames; the reference: the prefix and its two windows' new frames)
  latents.safetensors  optional (--keep latents): per round the final latents and the VACE context latents
  sample_outputs.json  written last (status success = done; --resume skips it)

  run_vace_bench.py --specs DIR|FILE... --input-root DIR --output-root DIR --prompt-cache FILE [--resume] [--limit N]
                    [--max-continuations N] [--prefix-root DIR] [--keep native16,latents] [--verify-weights size|sha256] [--stop-file PATH] [--src DIR] [--ckpt DIR]
                    [--gpu-uuid UUID|any] [--min-host-avail-gib G] [--frozen-file FILE] [--check-source-equivalence]
  run_vace_bench.py --make-prompt-cache FILE [--src DIR] [--ckpt DIR] [--snapshot DIR]   (GPU, the official T5; once)
  run_vace_bench.py --self-test [--specs DIR...] [--input-root DIR] [--src DIR] [--ckpt DIR] [--prompt-cache FILE] (CPU)
  run_vace_bench.py --count-remaining --specs DIR --input-root DIR --output-root DIR   (no model)
Machine paths come from the options, else from the environment or the git-ignored local.env (template local.env.example):
--src / VACE_SRC (else VACE_ENV/src-git), --ckpt / VACE_CKPT, --prompt-cache / VACE_PROMPT_CACHE, --gpu-uuid /
WCB_LOCAL_GPU_UUID ("any" on a scheduler-assigned GPU, as on cluster), --frozen-file / VACE_FROZEN_FILE (required by TEST
specs; its sha256 goes into every record). The wan package cannot be imported without a GPU (wan/modules/t5.py evaluates
torch.cuda.current_device() in a default argument), so the self-test reads the official files instead of importing them.
"""
from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import importlib.util
import json
import os
import platform
import resource
import socket
import subprocess
import sys
import tempfile
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vace_common as vc  # noqa: E402

warnings.filterwarnings('ignore', category=FutureWarning)      # torch.cuda.amp.autocast in the official code
SCHEMA_OUT = 'wm-calib-bench.wan21_vace13b.sample_outputs.v1'
SPEC_SCHEMA = 'wm-calib-bench.wan21_vace13b.spec.v1'
CACHE_SCHEMA = 'wm-calib-bench.wan21_vace13b.prompt_cache.v1'
# wan/configs/shared_config.py (pinned commit): sample_neg_prompt, verbatim (checked against the file)
# Kept in Chinese as released by the Wan authors. In English it lists what to avoid: vivid oversaturated colours, overexposure,
# static frames, blurred details, subtitles, style, artwork, painting, still image, overall greyness, worst and low quality,
# JPEG artefacts, ugly or incomplete content, extra fingers, badly drawn hands and faces, deformed or disfigured limbs, fused
# fingers, motionless frames, cluttered backgrounds, three legs, crowded backgrounds and walking backwards.
NEGATIVE_PROMPT = ('色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，'
                   '多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走')
SETTINGS = dict(task='vace-1.3B', size_config='832*480', frame_num=vc.N_ROUND, rounds=vc.N_ROUNDS, prefix_frames=vc.N_HIST,
                new_frames=vc.N_NEW, round2_prefix_start=vc.R2_OFFSET, sample_solver='unipc', sampling_steps=50, shift=16.0,
                guide_scale=5.0, context_scale=1.0, n_prompt='', offload_model=False, fps=vc.FPS, gray=vc.GRAY,
                prompt_extend=False, t5='prompt cache (official T5 call, bf16)')
KEEP_CHOICES = ('native16', 'latents')
N_PREFIX_LAT = (vc.N_HIST - 1) // 4 + 1          # 9 latents hold the 33 kept frames
N_ROUND_LAT = (vc.N_ROUND - 1) // 4 + 1          # 21 latents per round
CALLS_PER_ROUND = 2 * SETTINGS['sampling_steps']  # cond + uncond per step


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def host_mem_available_gib() -> float:
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) / 1048576
    return float('nan')


def rss_peak_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576


def load_specs(paths) -> list[tuple[Path, dict]]:
    files = []
    for p in paths:
        p = Path(p)
        files += sorted(p.rglob('*.json')) if p.is_dir() else [p]
    out = []
    for f in files:
        d = json.loads(f.read_text())
        if d.get('schema') == SPEC_SCHEMA:
            out.append((f, d))
    order = {'prefix': 0, 'recon': 1, 'continue': 2}       # a window's prefix first (the others copy frames 0-20 from it)
    out.sort(key=lambda fs: (fs[1]['window'], order[fs[1]['mode']], int(fs[1]['sample']) if str(fs[1]['sample']).isdigit() else -1))
    return out


def spec_size(spec) -> tuple:
    return vc.SIZES[vc.size_name((spec['settings']['width'], spec['settings']['height']))]


def check_spec(spec: dict) -> list[str]:
    bad = []
    if spec.get('mode') not in ('prefix', 'continue', 'recon'):
        bad.append(f'mode {spec.get("mode")}')
    st = spec.get('settings', {})
    for k, v in SETTINGS.items():
        if st.get(k) != v:
            bad.append(f'settings.{k} = {st.get(k)} (adapter: {v})')
    try:
        spec_size(spec)
    except (ValueError, TypeError, KeyError):
        bad.append(f'generation size {st.get("width")} x {st.get("height")} is not one of {sorted(vc.SIZES.values())}')
    ins = spec.get('inputs', {})
    if 'history' not in ins:
        bad.append('inputs.history missing')
    if spec.get('mode') == 'continue' and (not isinstance(spec.get('seed'), int) or not isinstance(spec.get('seed_round2'), int)):
        bad.append('seeds missing')
    if spec.get('mode') == 'recon' and 'real' not in ins:
        bad.append('inputs.real missing')
    if spec.get('mode') == 'recon' and spec.get('split') == 'TEST' and not spec.get('frozen_reference'):
        bad.append('a TEST reference spec without the frozen_reference mark (TEST references are made after the freeze)')
    if spec.get('negative_prompt_sha256') != vc.sha256_text(NEGATIVE_PROMPT):
        bad.append('negative prompt differs from the adapter constant')
    if spec.get('prompt_sha256') != vc.sha256_text(spec.get('prompt', '')):
        bad.append('prompt sha256 mismatch')
    return bad


# ----------------------------------------------------------------------------------------------- official file checks
def read_source_file(src: Path, rel: str) -> str:
    return (Path(src) / rel).read_text()


def config_negative_prompt(src: Path) -> str:
    """sample_neg_prompt of wan/configs/shared_config.py (the value generate() uses for n_prompt '')."""
    tree = ast.parse(read_source_file(src, 'wan/configs/shared_config.py'))
    vals = [n.value.value for n in ast.walk(tree) if isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Attribute) and t.attr == 'sample_neg_prompt' for t in n.targets) and isinstance(n.value, ast.Constant)]
    if len(vals) != 1:
        raise SystemExit(f'shared_config.py: {len(vals)} sample_neg_prompt literals')
    return vals[0]


def official_default_checks(src: Path) -> dict:
    """The released VACE defaults, read from the pinned files (generate.py, wan/vace.py, the configs)."""
    g = read_source_file(src, 'generate.py')
    v = read_source_file(src, 'wan/vace.py')
    c = read_source_file(src, 'wan/configs/__init__.py')
    s = read_source_file(src, 'wan/configs/shared_config.py')
    checks = {
        'generate.py: 50 steps unless i2v': 'args.sample_steps = 50' in g,
        'generate.py: shift 16 for vace': 'elif "flf2v" in args.task or "vace" in args.task:\n            args.sample_shift = 16' in g,
        'generate.py: guidance default 5.0': '"--sample_guide_scale",\n        type=float,\n        default=5.0' in g,
        'generate.py: solver default unipc': "default='unipc'" in g,
        'generate.py: frame_num 81': 'args.frame_num = 1 if "t2i" in args.task else 81' in g,
        'configs: vace-1.3B = t2v_1_3B': "'vace-1.3B': t2v_1_3B" in c,
        'configs: vace-1.3B sizes 480*832, 832*480': "'vace-1.3B': ('480*832', '832*480')" in c,
        'shared_config: sample_fps 16': 'wan_shared_cfg.sample_fps = 16' in s,
        'shared_config: param_dtype bfloat16': 'wan_shared_cfg.param_dtype = torch.bfloat16' in s,
        'vace.py: masks binarised at 0.5, inactive / reactive': 'torch.where(m > 0.5, 1.0, 0.0)' in v and 'inactive = [i * (1 - m) + 0 * m' in v,
        'vace.py: noise from torch.Generator seeded with seed': 'seed_g.manual_seed(seed)' in v and 'generator=seed_g)' in v,
        'vace.py: area 480*832 -> seq_len 32760': 'elif area == 480 * 832:\n            self.vid_proc.set_seq_len(32760)' in v,
        'vace.py: mask = clamp((m[:1] + 1) / 2)': '(src_mask[i][:1, :, :, :] + 1) / 2, min=0, max=1)' in v,
    }
    return checks


def load_vace_processor_module(src: Path):
    """wan/utils/vace_processor.py imported by path (numpy, torch, PIL only), without the wan package's __init__."""
    spec = importlib.util.spec_from_file_location('wan_vace_processor_pinned', Path(src) / 'wan/utils/vace_processor.py')
    mod = importlib.util.module_from_spec(spec)
    keep, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = keep
    return mod


def official_processor(vp_mod):
    """The VaceVideoProcessor WanVace builds, with prepare_source's settings for --size 832*480."""
    proc = vp_mod.VaceVideoProcessor(downsample=(4, 16, 16), min_area=720 * 1280, max_area=720 * 1280, min_fps=16, max_fps=16,
                                     zero_start=True, seq_len=75600, keep_last=True)
    proc.set_area(480 * 832)
    proc.set_seq_len(32760)
    return proc


def processor_geometry(proc, np, wh) -> dict:
    """What the official loader does with an 81-frame, 16 fps input of size wh: its output size and frame ids
    (_get_frameid_bbox_adjust_last with the input's frame timestamps [i / 16, (i + 1) / 16])."""
    ts = np.array([[i / 16, (i + 1) / 16] for i in range(vc.N_ROUND)], dtype=np.float32)
    ids, box, (oh, ow), fps = proc._get_frameid_bbox_adjust_last(16.0, ts, wh[1], wh[0], None, None)
    return dict(out_hw=[int(oh), int(ow)], frame_ids_identity=list(ids) == list(range(vc.N_ROUND)), fps=float(fps),
                keeps_size=(int(ow), int(oh)) == tuple(wh))


def uint8_frames(video, np):
    """cache_video's conversion of a decoded [C, T, H, W] tensor in [-1, 1] (clamp, make_grid normalize, x 255, uint8
    truncation), as numpy [T, H, W, C]."""
    import torch
    t = video.clamp(-1, 1)
    t = (t + 1) / 2                        # make_grid(normalize=True, value_range=(-1, 1)): sub_(-1).div_(2)
    t = (t * 255).type(torch.uint8)
    return t.permute(1, 2, 3, 0).cpu().numpy()


def to_canvas(frame, size):
    """The canvas mapping: the identity when the model generated on the canvas (512 x 288), else INTER_AREA."""
    if tuple(size) == vc.CANVAS:
        return frame
    import cv2
    return cv2.resize(frame, vc.CANVAS, interpolation=cv2.INTER_AREA)


def psnr(a, b, np) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float('inf') if mse == 0 else round(10 * np.log10(255.0 ** 2 / mse), 2)


def pixel_stats(a, b, np) -> dict:
    d = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return dict(max_abs=int(d.max()), mean_abs=round(float(d.mean()), 4), psnr_db=psnr(a, b, np))


def check_weights(ckpt: Path, how: str, need_t5: bool = False) -> dict:
    files = vc.PINS['weights']['files']
    sums = {}
    p = ckpt / 'SHA256SUMS'
    if not p.is_file():
        raise SystemExit(f'{p} missing: run download_vace_weights.sh (research server) or stage_weights.sh (cluster)')
    for line in p.read_text().splitlines():
        h, _, f = line.partition('  ')
        sums[f] = h
    bad = []
    for rel, rec in files.items():
        if not rec['cluster'] and not (need_t5 and rel.startswith(('models_t5', 'google/'))):
            continue
        f = ckpt / rel
        if not f.is_file() or f.stat().st_size != rec['bytes']:
            bad.append(f'{rel}: missing or size differs')
            continue
        if sums.get(rel) != rec['sha256']:
            bad.append(f'{rel}: SHA256SUMS differs from the pins')
        if how == 'sha256' and vc.sha256_file(f) != rec['sha256']:
            bad.append(f'{rel}: sha256 differs from the pins')
    if bad:
        raise SystemExit('weights: ' + '; '.join(bad))
    return dict(dir_sha256sums=vc.sha256_file(p), verified=how, revision=vc.PINS['weights']['hf_revision'])


# ---------------------------------------------------------------------------------------------------- the prompt cache
def tensor_sha256(t) -> str:
    import torch
    return hashlib.sha256(t.detach().to('cpu').contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def make_prompt_cache(out: Path, src: Path, ckpt: Path, snapshot=None) -> None:
    """The official T5 call of WanVace.generate (t5_cpu False): T5EncoderModel built on the CPU in bf16, moved to the GPU,
    called on [prompt] and on [negative prompt]; the returned tensors are saved with their provenance."""
    import torch
    from safetensors.torch import save_file
    if vc.source_digest(src) != vc.PINS['code']['source_digest']:
        raise SystemExit('source digest differs from the pins')
    check_weights(ckpt, 'sha256', need_t5=True)
    if config_negative_prompt(src) != NEGATIVE_PROMPT:
        raise SystemExit('the negative prompt of shared_config.py differs from the adapter constant')
    _, prompts, nano_sha = vc.load_seed_of(snapshot)
    prompt = prompts['long']
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(src))
    from wan.configs import WAN_CONFIGS
    from wan.modules.t5 import T5EncoderModel
    cfg = WAN_CONFIGS['vace-1.3B']
    dev = torch.device('cuda:0')
    t0 = time.time()
    te = T5EncoderModel(text_len=cfg.text_len, dtype=cfg.t5_dtype, device=torch.device('cpu'),
                        checkpoint_path=str(ckpt / cfg.t5_checkpoint), tokenizer_path=str(ckpt / cfg.t5_tokenizer), shard_fn=None)
    te.model.to(dev)
    with torch.no_grad():
        ctx = te([prompt], dev)
        ctx_null = te([NEGATIVE_PROMPT], dev)
    tensors = {'prompt': ctx[0].detach().to('cpu').contiguous(), 'negative': ctx_null[0].detach().to('cpu').contiguous()}
    props = torch.cuda.get_device_properties(0)
    meta = dict(schema=CACHE_SCHEMA, created=now(), prompt=prompt, prompt_sha256=vc.sha256_text(prompt), prompt_key='long',
                negative_prompt_sha256=vc.sha256_text(NEGATIVE_PROMPT), nano_specs_sha256=nano_sha,
                tensors={k: dict(shape=list(v.shape), dtype=str(v.dtype), sha256=tensor_sha256(v)) for k, v in tensors.items()},
                call='T5EncoderModel(text_len=512, dtype=bfloat16, device=cpu, umT5-XXL bf16).model.to(cuda); te([text], cuda) as WanVace.generate',
                code_commit=vc.PINS['code']['commit'], source_digest=vc.PINS['code']['source_digest'],
                weights_revision=vc.PINS['weights']['hf_revision'], t5_sha256=vc.PINS['weights']['files']['models_t5_umt5-xxl-enc-bf16.pth']['sha256'],
                torch=torch.__version__, cuda=torch.version.cuda, gpu=props.name, seconds=round(time.time() - t0, 1),
                host=socket.gethostname())
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix('.part')
    save_file(tensors, str(tmp), metadata={'wcb': json.dumps(meta, ensure_ascii=False)})
    os.replace(tmp, out)
    rec = dict(meta, file=out.name, file_sha256=vc.sha256_file(out), bytes=out.stat().st_size)
    out.with_suffix('.json').write_text(json.dumps(rec, indent=1, ensure_ascii=False) + '\n')
    print(json.dumps({k: rec[k] for k in ('file', 'file_sha256', 'tensors', 'gpu', 'seconds')}, ensure_ascii=False))


def load_prompt_cache(path: Path) -> tuple[dict, dict]:
    from safetensors import safe_open
    with safe_open(str(path), 'pt') as f:
        meta = json.loads(f.metadata()['wcb'])
        tensors = {k: f.get_tensor(k) for k in f.keys()}
    if meta.get('schema') != CACHE_SCHEMA or meta['negative_prompt_sha256'] != vc.sha256_text(NEGATIVE_PROMPT):
        raise SystemExit(f'{path}: not a prompt cache of this adapter')
    for k, v in tensors.items():
        if tensor_sha256(v) != meta['tensors'][k]['sha256']:
            raise SystemExit(f'{path}: tensor {k} differs from its recorded sha256')
    if meta['code_commit'] != vc.PINS['code']['commit'] or meta['weights_revision'] != vc.PINS['weights']['hf_revision']:
        raise SystemExit(f'{path}: made with another code commit or weight revision')
    return tensors, dict(meta, file_sha256=vc.sha256_file(path))


class CachedT5:
    """Stands in for wan.modules.t5.T5EncoderModel inside wan.vace: WanVace.__init__ builds it with the T5 arguments
    (nothing is loaded) and generate() calls .model.to(device), then self([text], device) for the prompt and the negative
    prompt; it returns copies of the official call's tensors on that device."""
    cache: dict = {}

    def __init__(self, **kwargs):
        self.init_kwargs = {k: str(v) for k, v in kwargs.items()}

        class _NoModel:
            def to(self, *a, **k):
                return self

            def cpu(self):
                return self
        self.model = _NoModel()

    def __call__(self, texts, device):
        out = []
        for t in texts:
            if t not in CachedT5.cache:
                raise KeyError('text not in the prompt cache (it holds the bench prompt and the config negative prompt)')
            out.append(CachedT5.cache[t].to(device).clone())
        return out


# ------------------------------------------------------------------------------------------------------------ the model
class VaceModel:
    def __init__(self, src: Path, ckpt: Path, verify_weights: str, cache_path: Path, gpu_uuid: str, min_avail_gib: float):
        t0 = time.time()
        self.src, self.ckpt = Path(src), Path(ckpt)
        self.digest = vc.source_digest(self.src)
        if self.digest != vc.PINS['code']['source_digest']:
            raise SystemExit(f'{self.src}: source digest {self.digest}, pinned {vc.PINS["code"]["source_digest"]}')
        self.weights = check_weights(self.ckpt, verify_weights)
        if config_negative_prompt(self.src) != NEGATIVE_PROMPT:
            raise SystemExit('the negative prompt of shared_config.py differs from the adapter constant')
        avail = host_mem_available_gib()
        if avail < min_avail_gib:
            raise SystemExit(f'host MemAvailable {avail:.1f} GiB < {min_avail_gib:g} GiB')
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(self.src))
        import numpy as np
        import torch
        self.np, self.torch = np, torch
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise SystemExit('needs exactly one visible CUDA device (CUDA_VISIBLE_DEVICES set by the GPU runner)')
        props = torch.cuda.get_device_properties(0)
        uuid = str(getattr(props, 'uuid', '') or '')
        if not uuid:
            idx = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')[0]
            uuid = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader', '-i', idx], text=True).strip()
        uuid = uuid if uuid.startswith('GPU-') else f'GPU-{uuid}'
        if gpu_uuid != 'any' and uuid != gpu_uuid:
            raise SystemExit(f'visible GPU {uuid} is not the allowed GPU {gpu_uuid} (--gpu-uuid / WCB_LOCAL_GPU_UUID)')
        self.gpu = dict(name=props.name, uuid=uuid, total_gib=round(props.total_memory / 2**30, 2), capability=f'{props.major}.{props.minor}')
        self.guards = dict(gpu_uuid=gpu_uuid, min_host_avail_gib=min_avail_gib, host_mem_available_gib_at_start=round(avail, 1))
        # the prompt cache and the two read-only substitutions in the wan.vace namespace
        tensors, self.cache_meta = load_prompt_cache(cache_path)
        CachedT5.cache = {self.cache_meta['prompt']: tensors['prompt'], NEGATIVE_PROMPT: tensors['negative']}
        import wan
        import wan.vace as wv
        from wan.configs import SIZE_CONFIGS, WAN_CONFIGS
        from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
        self.record = {}
        record = self.record

        class RecordingUniPC(FlowUniPCMultistepScheduler):
            def set_timesteps(self, *a, **k):
                out = super().set_timesteps(*a, **k)
                record['scheduler'] = dict(cls='FlowUniPCMultistepScheduler', num_inference_steps=int(self.num_inference_steps),
                                           shift_arg=float(k.get('shift')) if k.get('shift') is not None else None,
                                           config_shift=float(self.config.shift), solver_order=int(self.config.solver_order),
                                           prediction_type=self.config.prediction_type, use_dynamic_shifting=bool(self.config.use_dynamic_shifting),
                                           timesteps=[int(x) for x in self.timesteps.tolist()], sigmas=[round(float(x), 7) for x in self.sigmas.tolist()])
                return out
        wv.T5EncoderModel = CachedT5
        wv.FlowUniPCMultistepScheduler = RecordingUniPC
        self.size_config = SIZE_CONFIGS[SETTINGS['size_config']]
        cfg = WAN_CONFIGS[SETTINGS['task']]
        self.cfg = cfg
        self.pipe = wv.WanVace(config=cfg, checkpoint_dir=str(self.ckpt), device_id=0, rank=0, t5_fsdp=False, dit_fsdp=False,
                               use_usp=False, t5_cpu=False)
        if self.pipe.sample_neg_prompt != NEGATIVE_PROMPT:
            raise SystemExit('the pipeline negative prompt differs from the adapter constant')
        # from_pretrained (diffusers) must have filled every tensor of the file
        from safetensors import safe_open
        with safe_open(str(self.ckpt / 'diffusion_pytorch_model.safetensors'), 'pt') as f:
            keys = set(f.keys())
        sd_keys = set(self.pipe.model.state_dict().keys())
        self.load_check = dict(file_tensors=len(keys), model_tensors=len(sd_keys), n_missing=len(sd_keys - keys), n_unexpected=len(keys - sd_keys),
                               missing=sorted(sd_keys - keys)[:10], unexpected=sorted(keys - sd_keys)[:10])
        if self.load_check['n_missing'] or self.load_check['n_unexpected']:
            raise SystemExit(f'DiT weights do not match the model: {self.load_check}')
        import collections
        self.dtypes = dict(dit_params=dict(collections.Counter(str(p.dtype) for p in self.pipe.model.parameters())),
                           dit_n_params=int(sum(p.numel() for p in self.pipe.model.parameters())),
                           autocast=str(self.pipe.param_dtype), vae=str(self.pipe.vae.dtype),
                           vae_params=dict(collections.Counter(str(p.dtype) for p in self.pipe.vae.model.parameters())),
                           prompt_cache=str(tensors['prompt'].dtype), t5_stub=self.pipe.text_encoder.init_kwargs)
        self.model_config = {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in dict(self.pipe.model.config).items() if not k.startswith('_')}
        self.vp_mod = sys.modules[type(self.pipe.vid_proc).__module__]
        self.calls = [0]
        self.pipe.model.register_forward_pre_hook(lambda m, a: self.calls.__setitem__(0, self.calls[0] + 1))
        # record the final latents and the VACE context latents of every generate() call (no change to either)
        self.captured = {}
        orig_decode, orig_encode = self.pipe.decode_latent, self.pipe.vace_encode_frames

        def decode_latent(zs, ref_images=None, vae=None):
            self.captured['x0'] = zs[0].detach().clone()
            return orig_decode(zs, ref_images, vae)

        def vace_encode_frames(frames, ref_images, masks=None, vae=None):
            z = orig_encode(frames, ref_images, masks=masks, vae=vae)
            self.captured['z0'] = z[0].detach().clone()
            return z
        self.pipe.decode_latent = decode_latent
        self.pipe.vace_encode_frames = vace_encode_frames
        proc = official_processor(self.vp_mod)
        self.geometry = {name: processor_geometry(proc, np, wh) for name, wh in vc.SIZES.items()}
        for name, g in self.geometry.items():
            if not (g['keeps_size'] and g['frame_ids_identity']):
                raise SystemExit(f'the official VaceVideoProcessor would change a {name} input: {g}')
        self.load_s = round(time.time() - t0, 1)
        self.rss_after_load = round(rss_peak_gib(), 2)
        self.build_peak = dict(allocated_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                               reserved_gib=round(torch.cuda.max_memory_reserved() / 2**30, 2))
        torch.cuda.reset_peak_memory_stats()
        self.wan_file = str(Path(wan.__file__).resolve())

    # -------------------------------------------------------------------------------------------------- VACE inputs
    def vace_source(self, kept_u8, size):
        """src_video [3, 81, H, W] and src_mask [1, 81, H, W] on the GPU, as WanVace.prepare_source loads a lossless file
        pair (the kept frames then gray 127 frames; the mask 0 then 255): VaceVideoProcessor.resize_crop on the decord-style
        uint8 [T, H, W, C] tensors (no resize at this size: checked), .to(device), mask clamp((m[:1] + 1) / 2, 0, 1)."""
        torch, np = self.torch, self.np
        W, H = size
        n_keep = len(kept_u8)
        video = np.concatenate([kept_u8, np.full((vc.N_ROUND - n_keep, H, W, 3), vc.GRAY, np.uint8)])
        mask = np.concatenate([np.zeros((n_keep, H, W, 3), np.uint8), np.full((vc.N_ROUND - n_keep, H, W, 3), 255, np.uint8)])
        rc = self.vp_mod.VaceVideoProcessor.resize_crop
        dev = self.pipe.device
        v = rc(torch.from_numpy(video), H, W).to(dev)
        m = rc(torch.from_numpy(mask), H, W).to(dev)
        m = torch.clamp((m[:1, :, :, :] + 1) / 2, min=0, max=1)
        return v, m, video, mask

    def source_equivalence(self, video_u8, mask_u8, v, m, size) -> bool:
        """Write the pair losslessly and load it with the official prepare_source: bit-identical tensors?"""
        torch = self.torch
        with tempfile.TemporaryDirectory() as td:
            pv, pm = Path(td) / 'src_video.mp4', Path(td) / 'src_mask.mp4'
            vc.write_video(list(video_u8), pv, vc.FPS)
            vc.write_video(list(mask_u8), pm, vc.FPS)
            sv, sm, _ = self.pipe.prepare_source([str(pv)], [str(pm)], [None], vc.N_ROUND, self.size_config, self.pipe.device)
        return bool(torch.equal(sv[0], v) and torch.equal(sm[0], m))

    def vae_roundtrip(self, frames_u8, size):
        """frames (uint8 [T, H, W, 3]) -> the official loader's float conversion -> WanVAE.encode -> decode -> uint8."""
        torch = self.torch
        rc = self.vp_mod.VaceVideoProcessor.resize_crop
        x = rc(torch.from_numpy(np_copy(frames_u8, self.np)), size[1], size[0]).to(self.pipe.device)
        with torch.no_grad():
            z = self.pipe.vae.encode([x])[0]
            dec = self.pipe.vae.decode([z])[0]
        return uint8_frames(dec, self.np), z

    def generate_round(self, src_v, src_m, seed: int, prompt: str):
        torch = self.torch
        self.captured.clear()
        self.record.pop('scheduler', None)
        c0 = self.calls[0]
        t0 = time.time()
        with torch.no_grad():
            video = self.pipe.generate(prompt, [src_v], [src_m], [None], size=self.size_config, frame_num=SETTINGS['frame_num'],
                                       context_scale=SETTINGS['context_scale'], shift=SETTINGS['shift'], sample_solver=SETTINGS['sample_solver'],
                                       sampling_steps=SETTINGS['sampling_steps'], guide_scale=SETTINGS['guide_scale'], n_prompt=SETTINGS['n_prompt'],
                                       seed=int(seed), offload_model=SETTINGS['offload_model'])
        torch.cuda.synchronize()
        sec = time.time() - t0
        calls = self.calls[0] - c0
        if calls != CALLS_PER_ROUND:
            raise RuntimeError(f'{calls} DiT calls in a round, expected {CALLS_PER_ROUND}')
        sched = self.record.get('scheduler')
        if not sched or sched['num_inference_steps'] != SETTINGS['sampling_steps'] or sched['shift_arg'] != SETTINGS['shift']:
            raise RuntimeError(f'scheduler read back {sched}')
        frames = uint8_frames(video, self.np)
        return frames, dict(self.captured), sched, round(sec, 2), calls

    # ------------------------------------------------------------------------------------------------------ samples
    def prefix(self, spec, hist, size):
        frames, z = self.vae_roundtrip(hist, size)
        return frames, dict(latents=list(z.shape), latent_sha256=tensor_sha256(z))

    def continue_(self, spec, hist, size, check_equiv: bool):
        np, torch = self.np, self.torch
        info = dict(rounds=[])
        r1_v, r1_m, v_u8, m_u8 = self.vace_source(hist, size)
        eq = [self.source_equivalence(v_u8, m_u8, r1_v, r1_m, size)] if check_equiv else []
        out1, cap1, sched, s1, c1 = self.generate_round(r1_v, r1_m, spec['seed'], spec['prompt'])
        cond2 = out1[vc.R2_OFFSET:vc.N_ROUND]                              # round 1's frames 48-80 (t0 + 1.0 .. t0 + 3.0)
        r2_v, r2_m, v2_u8, m2_u8 = self.vace_source(cond2, size)
        if check_equiv:
            eq.append(self.source_equivalence(v2_u8, m2_u8, r2_v, r2_m, size))
        out2, cap2, sched2, s2, c2 = self.generate_round(r2_v, r2_m, spec['seed_round2'], spec['prompt'])
        if sched2 != sched:
            raise RuntimeError('the two rounds read back different scheduler settings')
        for r, (cap, kept, out, sec, calls, seed) in enumerate(((cap1, hist, out1, s1, c1, spec['seed']), (cap2, cond2, out2, s2, c2, spec['seed_round2'])), 1):
            x0, z0 = cap['x0'], cap['z0']
            if x0.shape[1] != N_ROUND_LAT or z0.shape[0] != 32:
                raise RuntimeError(f'round {r}: latents {tuple(x0.shape)}, context {tuple(z0.shape)}')
            inactive = z0[:16, :N_PREFIX_LAT].float()
            dl = (x0[:, :N_PREFIX_LAT].float() - inactive)
            info['rounds'].append(dict(
                round=r, seed=seed, seconds=sec, dit_calls=calls, latents=list(x0.shape), context=list(z0.shape),
                noise_sha256=noise_sha256(torch, x0.shape, seed, self.pipe.device),
                prefix_latent_vs_context=dict(max_abs=round(float(dl.abs().max()), 5), mean_abs=round(float(dl.abs().mean()), 5),
                                              rel_l2=round(float(dl.norm() / inactive.norm()), 5)),
                prefix_pixels_vs_kept=pixel_stats(out[:vc.N_HIST], kept, np),
                last_kept_pixels=pixel_stats(out[vc.N_HIST - 1], kept[-1], np),
                final_latent_sha256=tensor_sha256(x0), context_sha256=tensor_sha256(z0)))
        info['source_equivalence'] = eq if check_equiv else None
        info['scheduler'] = sched
        return out1, out2, (cap1, cap2), info

    def recon(self, spec, real, size):
        out1, z1 = self.vae_roundtrip(real[0:vc.N_ROUND], size)
        out2, z2 = self.vae_roundtrip(real[vc.R2_OFFSET:vc.N_ALL], size)
        return out1, out2, dict(round_latents=[list(z1.shape), list(z2.shape)], latent_sha256=[tensor_sha256(z1), tensor_sha256(z2)])


def np_copy(a, np):
    return np.ascontiguousarray(a).copy()


def noise_sha256(torch, shape, seed, device) -> str:
    """The initial noise generate() draws for this seed (recomputed with the same call; the stream is not touched)."""
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    n = torch.randn(shape[0], shape[1], shape[2], shape[3], dtype=torch.float32, device=device, generator=g)
    return tensor_sha256(n)


def versions(torch, uuid: str) -> dict:
    v = dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version())
    for mod in ('flash_attn', 'diffusers', 'transformers', 'decord', 'cv2', 'numpy', 'safetensors'):
        try:
            v[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            v[mod] = None
    try:
        v['driver'] = subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader', '-i', uuid],
                                              text=True).strip()
    except Exception:  # noqa: BLE001
        v['driver'] = None
    if not v['driver']:
        try:
            import ctypes
            nv = ctypes.CDLL('libnvidia-ml.so.1')
            nv.nvmlInit_v2()
            buf = ctypes.create_string_buffer(96)
            nv.nvmlSystemGetDriverVersion(buf, 96)
            v['driver'] = buf.value.decode()
        except Exception:  # noqa: BLE001
            pass
    v['cuda_driver'] = None
    try:
        import ctypes
        d = ctypes.c_int()
        ctypes.CDLL('libcuda.so.1').cuDriverGetVersion(ctypes.byref(d))
        v['cuda_driver'] = f'{d.value // 1000}.{(d.value % 1000) // 10}'
    except Exception:  # noqa: BLE001
        pass
    v['tf32'] = dict(matmul=bool(torch.backends.cuda.matmul.allow_tf32), cudnn=bool(torch.backends.cudnn.allow_tf32))
    v['cudnn_benchmark'] = bool(torch.backends.cudnn.benchmark)
    return v


def gpu_used_mib(uuid: str) -> int | None:
    try:
        return int(subprocess.check_output(['nvidia-smi', '--id=' + uuid, '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip())
    except Exception:  # noqa: BLE001
        return None


def prefix_dir(out_root: Path, window: str) -> Path:
    return out_root / 'native' / f'{window}-prefix'


def load_prefix(out_root: Path, window: str, size) -> tuple:
    """The window's prefix sample (mode prefix, run first): the 33 canvas frames and the record."""
    d = prefix_dir(out_root, window)
    rf = d / 'sample_outputs.json'
    if not rf.is_file():
        raise RuntimeError(f'{window}: the prefix sample is missing (run the window\'s prefix spec first)')
    rec = json.loads(rf.read_text())
    if rec.get('status') != 'success' or vc.sha256_file(d / 'prefix33.mp4') != rec['outputs']['prefix33.mp4']['sha256']:
        raise RuntimeError(f'{window}: the prefix sample is not finished or its file changed')
    if tuple(rec['size']['wh']) != tuple(size):
        raise RuntimeError(f'{window}: the prefix was made at {rec["size"]["wh"]}, this sample is at {list(size)}')
    frames = vc.decode(d / 'prefix33.mp4', vc.CANVAS)
    if len(frames) != vc.N_HIST:
        raise RuntimeError(f'{window}: prefix33.mp4 has {len(frames)} frames')
    return frames, rec


def run_sample(model: VaceModel, spec_path: Path, spec: dict, input_root: Path, out_root: Path, keep: set, n_in_process: int,
               check_equiv: bool, prefix_root: Path | None = None) -> dict:
    prefix_root = prefix_root or out_root
    np, torch = model.np, model.torch
    name = spec['name']
    out = out_root / 'native' / name
    out.mkdir(parents=True, exist_ok=True)
    (out / 'sample_outputs.json').unlink(missing_ok=True)
    keep = set(keep) | set(spec.get('keep', []))
    ins = {}
    for k, rec in spec['inputs'].items():
        p = input_root / rec['path']
        got = vc.sha256_file(p)
        if got != rec['sha256']:
            raise RuntimeError(f'{p}: sha256 {got[:16]}, spec {rec["sha256"][:16]}')
        ins[k] = dict(path=rec['path'], sha256=got)
    size = spec_size(spec)
    hist = vc.decode(input_root / spec['inputs']['history']['path'], size)
    if len(hist) != vc.N_HIST:
        raise RuntimeError(f'history has {len(hist)} frames, expected {vc.N_HIST}')
    model.calls[0] = 0
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    files, extra, latents_out = {}, {}, {}
    if spec['mode'] == 'prefix':
        frames, extra = model.prefix(spec, hist, size)
        canvas = [to_canvas(f, size) for f in frames]
        vc.write_video(canvas, out / 'prefix33.mp4', vc.FPS)
        if not np.array_equal(vc.decode(out / 'prefix33.mp4', vc.CANVAS), np.stack(canvas)):
            raise RuntimeError('prefix33.mp4 does not decode to the written frames')
        files['prefix33.mp4'] = dict(sha256=vc.sha256_file(out / 'prefix33.mp4'), frames=vc.N_HIST, fps=vc.FPS, size=list(vc.CANVAS))
        if tuple(size) != vc.CANVAS:
            vc.write_video(list(frames), out / 'prefix33_native.mp4', vc.FPS, threads=4)
            files['prefix33_native.mp4'] = dict(sha256=vc.sha256_file(out / 'prefix33_native.mp4'), frames=vc.N_HIST, fps=vc.FPS, size=list(size))
        extra['vs_history'] = pixel_stats(frames, hist, np)
        extra['t0_vs_history'] = pixel_stats(frames[vc.T0], hist[vc.T0], np)
        extra['frames_sha256'] = hashlib.sha256(np.stack(canvas).tobytes()).hexdigest()
        torch.cuda.synchronize()
        t_gen = time.time() - t0
    else:
        pre, pre_rec = load_prefix(prefix_root, spec['window'], size)
        if spec['mode'] == 'continue':
            out1, out2, caps, extra = model.continue_(spec, hist, size, check_equiv)
        else:
            real = vc.decode(input_root / spec['inputs']['real']['path'], size)
            if len(real) != vc.N_ALL or not np.array_equal(real[:vc.N_HIST], hist):
                raise RuntimeError('the real clip has the wrong length or its first 33 frames are not the history')
            out1, out2, extra = model.recon(spec, real, size)
            caps = None
            # the Wan VAE is causal: round 1's frames 0-32 should be the prefix (diagnostic)
            pre_native = vc.decode(prefix_dir(prefix_root, spec['window']) / 'prefix33_native.mp4', size) if tuple(size) != vc.CANVAS else pre
            extra['causal_round1_prefix_vs_prefix'] = pixel_stats(out1[:vc.N_HIST], pre_native, np)
        torch.cuda.synchronize()
        t_gen = time.time() - t0
        vis = []
        for k in vc.TICKS:
            j = vc.frame_of_tick(k)
            if j <= vc.T0:
                vis.append(pre[j])
            else:
                r, idx = vc.round_of_frame(j)
                vis.append(to_canvas((out1 if r == 1 else out2)[idx], size))
        vc.write_video(vis, out / 'vision.mp4', 10)
        back = vc.decode(out / 'vision.mp4', vc.CANVAS)
        if not np.array_equal(back, np.stack(vis)):
            raise RuntimeError('vision.mp4 does not decode to the written frames')
        files['vision.mp4'] = dict(sha256=vc.sha256_file(out / 'vision.mp4'), frames=81, fps=10, size=list(vc.CANVAS),
                                   frames_sha256=hashlib.sha256(back.tobytes()).hexdigest(),
                                   prefix21_frames_sha256=hashlib.sha256(back[:21].tobytes()).hexdigest())
        # the seams (canvas frames): the history's last step, t0 -> the first generated tick, round 1 -> round 2
        g = back.astype(np.float32).mean(axis=3)
        steps = np.abs(np.diff(g, axis=0)).mean(axis=(1, 2))
        extra['seams'] = dict(history_last_second_step_mean=round(float(steps[10:20].mean()), 4), t0_step=round(float(steps[20]), 4),
                              round_step_30_31=round(float(steps[50]), 4), future_step_median=round(float(np.median(steps[21:])), 4))
        if spec['mode'] == 'continue':
            # the regenerated t0 (round 1's frame 32) against the shared prefix's t0, on the canvas
            extra['regenerated_t0_vs_prefix_t0'] = pixel_stats(to_canvas(out1[vc.T0], size), pre[vc.T0], np)
            extra['regenerated_prefix_vs_prefix'] = pixel_stats(np.stack([to_canvas(f, size) for f in out1[:vc.N_HIST]]), pre, np)
            extra['round2_frame32_vs_round1_frame80'] = pixel_stats(to_canvas(out2[vc.T0], size), to_canvas(out1[vc.N_ROUND - 1], size), np)
        if 'native16' in keep:
            nat = [to_canvas(f, size) for f in out1] + [to_canvas(f, size) for f in out2[vc.N_HIST:]]
            if spec['mode'] == 'recon':
                nat = list(pre) + nat[vc.N_HIST:]
            vc.write_video(nat, out / 'native16.mp4', vc.FPS)
            files['native16.mp4'] = dict(sha256=vc.sha256_file(out / 'native16.mp4'), frames=len(nat), fps=vc.FPS, size=list(vc.CANVAS))
        if 'latents' in keep and caps:
            from safetensors.torch import save_file
            tensors = {}
            for r, cap in enumerate(caps, 1):
                tensors[f'round{r}_final_latents'] = cap['x0'].float().cpu().contiguous()
                tensors[f'round{r}_context'] = cap['z0'].float().cpu().contiguous()
            save_file(tensors, str(out / 'latents.safetensors'))
            files['latents.safetensors'] = dict(sha256=vc.sha256_file(out / 'latents.safetensors'))
        extra['prefix_sample'] = dict(name=f'{spec["window"]}-prefix', root=str(prefix_root) if prefix_root != out_root else 'output root',
                                      record_sha256=vc.sha256_file(prefix_dir(prefix_root, spec['window']) / 'sample_outputs.json'),
                                      prefix33_sha256=pre_rec['outputs']['prefix33.mp4']['sha256'])
    peak_alloc = torch.cuda.max_memory_allocated() / 2**30
    peak_res = torch.cuda.max_memory_reserved() / 2**30
    used_now = gpu_used_mib(model.gpu['uuid'])
    t_write = time.time() - t0 - t_gen
    res = dict(schema=SCHEMA_OUT, status='success', written=now(), name=name, window=spec['window'], sample=spec['sample'], mode=spec['mode'],
               run=spec['run'], split=spec['split'], spec_file=spec_path.name, spec_sha256=vc.sha256_file(spec_path), seed=spec.get('seed'),
               seed_round2=spec.get('seed_round2'), settings=dict(SETTINGS, width=size[0], height=size[1]),
               size=dict(name=vc.size_name(size), wh=list(size), canvas_mapping='identity' if tuple(size) == vc.CANVAS else 'cv2.INTER_AREA'),
               read_back=dict(scheduler=extra.get('scheduler'), dtypes=model.dtypes, model_config=model.model_config,
                              processor_geometry=model.geometry, size_config=list(model.size_config),
                              negative_prompt_is_config=model.pipe.sample_neg_prompt == NEGATIVE_PROMPT),
               prompt_sha256=spec['prompt_sha256'], negative_prompt_sha256=vc.sha256_text(NEGATIVE_PROMPT),
               prompt_cache=dict(file_sha256=model.cache_meta['file_sha256'], tensors=model.cache_meta['tensors'], gpu=model.cache_meta['gpu']),
               inputs=ins, outputs=files,
               timeline=dict(rule=vc.B['timeline_rule'], tick_rule=vc.B['tick_rule'], round_rule=vc.B['round_rule'], prefix_rule=vc.B['prefix_rule'],
                             vision_frames16=[vc.frame_of_tick(k) for k in vc.TICKS]),
               generation={k: v for k, v in extra.items() if k != 'scheduler'}, dit_calls=model.calls[0],
               timing=dict(generate_s=round(t_gen, 2), write_s=round(t_write, 2), model_load_s=model.load_s, sample_in_process=n_in_process),
               memory=dict(peak_allocated_gib=round(peak_alloc, 2), peak_reserved_gib=round(peak_res, 2), nvidia_smi_used_mib_after=used_now,
                           build_peak=model.build_peak, host_rss_peak_gib=round(rss_peak_gib(), 2), host_rss_after_load_gib=model.rss_after_load,
                           host_mem_available_gib=round(host_mem_available_gib(), 1)),
               hardware=dict(host=socket.gethostname(), gpu=model.gpu, image=os.environ.get('WCB_IMAGE'), **versions(torch, model.gpu['uuid'])),
               code=dict(commit=vc.PINS['code']['commit'], source_digest=model.digest, wan_package=model.wan_file,
                         adapter_sha256=vc.sha256_file(Path(__file__)), common_sha256=vc.sha256_file(HERE / 'vace_common.py'),
                         pins_sha256=vc.sha256_file(vc.PINS_PATH)),
               weights=model.weights, load_check=model.load_check, guards=model.guards,
               frozen=model.frozen, frozen_sha256=(model.frozen or {}).get('sha256'), pid=os.getpid())
    tmp = out / 'sample_outputs.json.part'
    tmp.write_text(json.dumps(res, indent=1, ensure_ascii=False) + '\n')
    os.replace(tmp, out / 'sample_outputs.json')
    return res


def done(out_root: Path, spec: dict, spec_path: Path | None = None) -> bool:
    p = out_root / 'native' / spec['name'] / 'sample_outputs.json'
    if not p.is_file():
        return False
    try:
        d = json.loads(p.read_text())
        if d.get('status') != 'success' or (spec_path is not None and d.get('spec_sha256') != vc.sha256_file(spec_path)):
            return False
        main = 'prefix33.mp4' if spec['mode'] == 'prefix' else 'vision.mp4'
        v = out_root / 'native' / spec['name'] / main
        return v.is_file() and vc.sha256_file(v) == d['outputs'][main]['sha256']
    except Exception:  # noqa: BLE001
        return False


# --------------------------------------------------------------------------------------------------------- self-test
def self_test(specs, input_root: Path | None, src: str | None = None, ck: str | None = None, cache: str | None = None) -> int:
    """CPU checks: timeline, pins, source digest and key files, the released defaults in the pinned files, the negative
    prompt, the official processor's geometry for both sizes and the in-memory source = prepare_source's file loading
    (decord), writer and decoder, INTER_AREA, weights listing, the prompt cache, specs (settings, seeds, inputs)."""
    import numpy as np
    res = []

    def item(name, ok, detail=''):
        res.append(dict(item=name, ok=bool(ok), detail=detail)); print(f'{"PASS" if ok else "FAIL"} {name} {detail}', flush=True)
    item('timeline (16 fps, ticks, rounds)', not vc.check_timeline(), f'{len(vc.check_timeline())} problems')
    if src and Path(src).is_dir():
        d = vc.source_digest(Path(src))
        item('source digest', d == vc.PINS['code']['source_digest'], d[:16])
        badf = [rel for rel, h in vc.PINS['code']['files_sha256'].items() if vc.sha256_file(Path(src) / rel) != h]
        item('key source files = pins', not badf, ','.join(badf))
        item('negative prompt = shared_config.py sample_neg_prompt', config_negative_prompt(Path(src)) == NEGATIVE_PROMPT,
             vc.sha256_text(NEGATIVE_PROMPT)[:16])
        chk = official_default_checks(Path(src))
        item('released VACE defaults in the pinned files', all(chk.values()), '; '.join(k for k, v in chk.items() if not v))
        try:
            import torch
            vp = load_vace_processor_module(Path(src))
            proc = official_processor(vp)
            geo = {n: processor_geometry(proc, np, wh) for n, wh in vc.SIZES.items()}
            item('official processor keeps both sizes and all 81 frames', all(g['keeps_size'] and g['frame_ids_identity'] for g in geo.values()),
                 json.dumps({n: g['out_hw'] for n, g in geo.items()}))
            import tempfile
            ok_all = []
            for name, (W, H) in vc.SIZES.items():
                rng = np.random.default_rng(1)
                kept = rng.integers(0, 256, (vc.N_HIST, H, W, 3), dtype=np.uint8)
                video = np.concatenate([kept, np.full((vc.N_NEW, H, W, 3), vc.GRAY, np.uint8)])
                mask = np.concatenate([np.zeros((vc.N_HIST, H, W, 3), np.uint8), np.full((vc.N_NEW, H, W, 3), 255, np.uint8)])
                v = vp.VaceVideoProcessor.resize_crop(torch.from_numpy(video), H, W)
                m = vp.VaceVideoProcessor.resize_crop(torch.from_numpy(mask), H, W)
                m = torch.clamp((m[:1] + 1) / 2, min=0, max=1)
                with tempfile.TemporaryDirectory() as td:
                    pv, pm = Path(td) / 'v.mp4', Path(td) / 'm.mp4'
                    vc.write_video(list(video), pv, vc.FPS)
                    vc.write_video(list(mask), pm, vc.FPS)
                    sv, sm, ids, ohw, fps = proc.load_video_pair(str(pv), str(pm))
                    sm = torch.clamp((sm[:1] + 1) / 2, min=0, max=1)
                ok_all.append(bool(torch.equal(sv, v) and torch.equal(sm, m) and list(ids) == list(range(vc.N_ROUND)) and tuple(ohw) == (H, W)))
            item('in-memory VACE source = the official file loading (decord, lossless pair), both sizes', all(ok_all), str(ok_all))
        except ImportError as e:
            item('official processor checks', False, f'import: {e}')
    if ck and Path(ck).is_dir():
        try:
            w = check_weights(Path(ck), 'size')
            item('weights (sizes, SHA256SUMS = pins)', True, w['dir_sha256sums'][:16])
        except SystemExit as e:
            item('weights', False, str(e))
    if cache and Path(cache).is_file():
        try:
            _, meta = load_prompt_cache(Path(cache))
            item('prompt cache (metadata, tensor hashes, code / weights pins)', True, f'{meta["file_sha256"][:16]} made on {meta["gpu"]}')
        except SystemExit as e:
            item('prompt cache', False, str(e))
    import tempfile
    import cv2
    rng = np.random.default_rng(0)
    fr = [np.clip(rng.normal(128, 60, (480, 832, 3)), 0, 255).astype(np.uint8) for _ in range(3)]
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / 't.mp4'
        vc.write_video(fr, p, vc.FPS)
        item('lossless writer', np.array_equal(vc.decode(p, (832, 480)), np.stack(fr)))
    a = cv2.resize(fr[0], vc.CANVAS, interpolation=cv2.INTER_AREA)
    item('INTER_AREA 832x480 -> 512x288 shape', a.shape == (288, 512, 3), cv2.__version__)
    try:
        import torch
        x = torch.linspace(-1.2, 1.2, 4001).view(1, 1, 1, -1).repeat(3, 1, 1, 1)
        u = uint8_frames(x, np)
        ref = np.floor(np.clip((np.linspace(-1.2, 1.2, 4001).astype(np.float32) + 1) / 2, 0, 1) * 255).astype(np.uint8)
        item('uint8 conversion = cache_video (clamp, (x + 1) / 2, x 255, truncation)', int(np.abs(u[0, 0, :, 0].astype(int) - ref.astype(int)).max()) <= 1,
             f'max level difference {int(np.abs(u[0, 0, :, 0].astype(int) - ref.astype(int)).max())} vs float64 (fp32 rounding)')
    except ImportError:
        pass
    if specs:
        seed_of = prompts = None
        try:
            seed_of, prompts, _ = vc.load_seed_of()
        except SystemExit as e:
            item('seed table (snapshot)', False, str(e))
        nbad = 0
        for f, s in specs:
            bad = check_spec(s)
            k = s['sample']
            if s['mode'] == 'continue':
                sof = seed_of or vc.seed_of_local
                if s['seed'] != sof(s['window'], int(k)) or s['seed_round2'] != sof(s['window'], f'{int(k)}|r2'):
                    bad.append('seeds differ from seed_of')
            if prompts and s.get('prompt') != prompts['long']:
                bad.append('prompt is not the Nano prompt "long"')
            if input_root:
                for kk, rec in s['inputs'].items():
                    q = input_root / rec['path']
                    if not q.is_file() or vc.sha256_file(q) != rec['sha256']:
                        bad.append(f'input {kk} missing or changed')
            if bad:
                nbad += 1
                print(f'  {f.name}: {bad}')
        item(f'specs ({len(specs)})', nbad == 0, f'{nbad} bad')
    ok = all(r['ok'] for r in res)
    print('self-test', 'PASS' if ok else 'FAIL', f'{sum(r["ok"] for r in res)}/{len(res)}')
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--specs', nargs='*', default=[]); ap.add_argument('--input-root', type=Path, default=None)
    ap.add_argument('--output-root', type=Path, default=None); ap.add_argument('--resume', action='store_true')
    ap.add_argument('--limit', type=int, default=0, help='run at most N samples (skipped ones not counted), then exit 0')
    ap.add_argument('--max-continuations', type=int, default=0, help='run at most N continuations (prefix and reference samples not '
                    'counted), then exit 0: short calls on a shared GPU (run_chunk.sh WCB_MAX_CONTINUATIONS; not used on cluster)')
    ap.add_argument('--keep', default='', help=f'comma list of {KEEP_CHOICES}')
    ap.add_argument('--verify-weights', choices=('size', 'sha256'), default='size')
    ap.add_argument('--stop-file', type=Path, default=None, help='exit 3 between samples when this file exists')
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--count-remaining', action='store_true', help='print the number of specs not done yet and exit (no model)')
    ap.add_argument('--make-prompt-cache', type=Path, default=None, help='write the prompt cache (official T5 on the GPU) and exit')
    ap.add_argument('--snapshot', type=Path, default=None)
    ap.add_argument('--src', default=None, help='the pinned Wan2.1 source (default VACE_SRC, else VACE_ENV/src-git)')
    ap.add_argument('--ckpt', default=None, help='the checkpoint directory (default VACE_CKPT)')
    ap.add_argument('--prompt-cache', default=None, help='the prompt cache (default VACE_PROMPT_CACHE)')
    ap.add_argument('--gpu-uuid', default=None, help='the only GPU allowed (default WCB_LOCAL_GPU_UUID); "any": the one visible GPU')
    ap.add_argument('--min-host-avail-gib', type=float, default=24.0, help='refuse to build the model below this MemAvailable (0: off)')
    ap.add_argument('--frozen-file', type=Path, default=None, help='the freeze record (default VACE_FROZEN_FILE); TEST needs it')
    ap.add_argument('--check-source-equivalence', action='store_true', help='also load each round\'s src pair with the official prepare_source')
    ap.add_argument('--prefix-root', type=Path, default=None, help='where the windows\' prefix samples are (native/<w>-prefix; default the output root)')
    a = ap.parse_args()
    vc.load_local_env()
    specs = load_specs(a.specs)
    src = a.src or vc.setting('VACE_SRC', required=False) or (str(Path(os.environ['VACE_ENV']) / 'src-git') if os.environ.get('VACE_ENV') else None)
    ck = a.ckpt or vc.setting('VACE_CKPT', required=False)
    cache = a.prompt_cache or vc.setting('VACE_PROMPT_CACHE', required=False)
    if a.self_test:
        sys.exit(self_test(specs, a.input_root, src, ck, cache))
    if a.make_prompt_cache:
        if not src or not ck:
            raise SystemExit('--src and --ckpt (or VACE_SRC / VACE_CKPT) are needed')
        make_prompt_cache(a.make_prompt_cache, Path(src), Path(ck), a.snapshot); return
    if not specs or not a.input_root or not a.output_root:
        raise SystemExit('--specs, --input-root and --output-root are required')
    keep = {k for k in a.keep.split(',') if k}
    if keep - set(KEEP_CHOICES):
        raise SystemExit(f'--keep: unknown {keep - set(KEEP_CHOICES)}')
    for f, s in specs:
        bad = check_spec(s)
        if bad:
            raise SystemExit(f'{f}: {bad}')
    frozen_path = a.frozen_file or (Path(os.environ['VACE_FROZEN_FILE']) if os.environ.get('VACE_FROZEN_FILE') else None)
    frozen = None
    if frozen_path:
        if not frozen_path.is_file():
            raise SystemExit(f'{frozen_path}: the freeze record does not exist')
        frozen = dict(file=frozen_path.name, sha256=vc.sha256_file(frozen_path))
    if any(s['split'] == 'TEST' for _, s in specs) and not frozen:
        raise SystemExit('TEST specs need the freeze record of this model (--frozen-file / VACE_FROZEN_FILE)')
    todo = [(f, s) for f, s in specs if not (a.resume and done(a.output_root, s, f))]
    if a.count_remaining:
        print(len([1 for f, s in specs if not done(a.output_root, s, f)])); return
    print(f'{now()} {len(specs)} specs, {len(specs) - len(todo)} done, {len(todo)} to run', flush=True)
    if not todo:
        return
    stop = a.stop_file or (a.output_root / 'STOP')
    if stop.exists():
        print(f'{stop} exists: not starting'); sys.exit(3)
    if not src or not ck or not cache:
        raise SystemExit('the Wan2.1 source, the checkpoint and the prompt cache are needed: --src / VACE_SRC, --ckpt / VACE_CKPT, '
                         '--prompt-cache / VACE_PROMPT_CACHE')
    prompts = {s['prompt'] for _, s in todo}
    model = VaceModel(Path(src), Path(ck), a.verify_weights, Path(cache), a.gpu_uuid or vc.setting('WCB_LOCAL_GPU_UUID'), a.min_host_avail_gib)
    if prompts - set(CachedT5.cache):
        raise SystemExit('a spec prompt is not in the prompt cache')
    model.frozen = frozen
    print(f'{now()} model ready in {model.load_s} s ({model.gpu["name"]}, host RSS peak {model.rss_after_load} GiB, '
          f'build peak {model.build_peak["reserved_gib"]} GiB reserved)', flush=True)
    n = n_cont = 0
    log = a.output_root / 'run-log.jsonl'
    a.output_root.mkdir(parents=True, exist_ok=True)
    for f, s in todo:
        if stop.exists():
            print(f'{now()} {stop} exists: stopping after {n} samples', flush=True); sys.exit(3)
        if a.limit and n >= a.limit:
            break
        if s['mode'] == 'continue':
            if a.max_continuations and n_cont >= a.max_continuations:
                break
            n_cont += 1
        n += 1
        r = run_sample(model, f, s, a.input_root, a.output_root, keep, n, a.check_source_equivalence, a.prefix_root)
        gen = r['generation']
        line = dict(t=now(), name=s['name'], mode=s['mode'], seconds=r['timing']['generate_s'], peak_alloc_gib=r['memory']['peak_allocated_gib'],
                    peak_reserved_gib=r['memory']['peak_reserved_gib'], rss_gib=r['memory']['host_rss_peak_gib'],
                    main=(r['outputs'].get('vision.mp4') or r['outputs'].get('prefix33.mp4'))['sha256'][:16])
        if s['mode'] == 'continue':
            line.update(rounds_s=[x['seconds'] for x in gen['rounds']], regen_t0_psnr=gen['regenerated_t0_vs_prefix_t0']['psnr_db'],
                        prefix_latent_mean_abs=[x['prefix_latent_vs_context']['mean_abs'] for x in gen['rounds']],
                        seams=gen['seams'], source_equivalence=gen.get('source_equivalence'))
        with log.open('a') as fh:
            fh.write(json.dumps(line, ensure_ascii=False) + '\n')
        print(json.dumps(line, ensure_ascii=False), flush=True)
        gc.collect()
    print(f'{now()} {n} samples run', flush=True)


if __name__ == '__main__':
    main()
