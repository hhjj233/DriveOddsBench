#!/usr/bin/env python3
"""Two unchanged official ReWorld Stage-1 calls per continuation (self-guided sampling); keep all native outputs.

The model is loaded ONCE per process with the official sample_dualflow loading lines, then every spec runs:
  prefix  : the real 17-frame history (t0-2..t0) through the official VAE encode/decode, seeded (codec prefix);
  pass 1  : official condition loader on the 9 real frames t0-1..t0 + official run_vae_latent_sampling
            (30 flow-match steps, Internal Guidance block 8, scale 1.4) + official decode_video -> 33 frames;
  pass 2  : the same on pass 1's last 9 GENERATED frames (written as a lossless 8 fps conditioning file), seed + 1000003;
  native65 = prefix[:8] + pass1[0:33] + pass2[9:33] at 8 Hz (t0 = index 16) -> nearest 8 -> 10 Hz canvas (81 frames).
A reference (mode recon) replaces both passes by VAE round trips of the real spans [-1, 3] s and [2, 6] s.
Adapted from run_drivelaw_bench.py (same spec/freeze/no-overwrite/failure rules and the same output layout).
I/O: every path argument must be absolute; the runner reads only the spec files, the inputs, the source and
the weights, and writes only under --output-root (sample directories, their .partial files, failure.json); it never uses
the working directory. Library caches (TMPDIR, HF_HOME, CUDA_CACHE_PATH, ...) are set by the launching worker.
"""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys
import time
from types import SimpleNamespace
sys.dont_write_bytecode = True
import reworld_common as rc
import build_reworld_inputs as bi

OFFICIAL_IG = dict(ig_head=None, ig_block_idx=8, ig_scale=1.4, ig_min_timestep=None)
DEVIATIONS = [
    'frame size 1024x512 (ruling D1; the shipped sample_reworld.yaml uses 768x384)',
    'model frame = full-width 2:1 crop of the top of the bench band, rendered by the kit (ruling D2; the official loader '
    'would centre-crop the clip it is given; at the model size its resize/crop are identities)',
    'seed per call = bench seed (official: cfg.seed + 17389 * scene index)',
    'condition-encode posterior draw seeded with torch.manual_seed(condition seed) right before the official loader '
    '(ruling D4; official: ambient global RNG state)',
    'pass 2 (6 s horizon) conditions on pass 1 frames 24..32 through the same official condition loader, seed + 1000003 (ruling D3)',
    'decoded frames written losslessly (libx264rgb crf 0) instead of export_to_video quality 5; the uint8 frames are the '
    'official (clamp(x, 0, 1) * 255).astype(uint8) conversion (ruling D8)',
    'whole spec under torch.inference_mode (the official code builds autograd graphs for the VAE calls; values unchanged)',
]


def validate_spec(spec, bench, input_root, frozen_file=None):
    if spec.get('schema') != 'wm-calib-bench.reworld_video.spec.v1' or spec.get('model') != 'reworld_video':
        raise ValueError('ReWorld spec required')
    split, mode = spec['split'], spec['mode']
    if split == 'test':
        expected = rc.B['test_generation_freeze'] if mode == 'continue' else rc.B['test_measurement_freeze']
        if not frozen_file or Path(frozen_file).name != expected:
            raise ValueError('correct TEST generation/measurement record required')
        rc.verify_freeze(frozen_file, required=[Path(__file__), rc.PINS_PATH])
    rc.guard_names(bench, split, [spec['window']])
    if mode not in ('continue', 'recon', 'prefix'):
        raise ValueError('unregistered mode')
    if spec['settings'] != rc.settings() or spec['generator']['pins_sha256'] != rc.sha(rc.PINS_PATH):
        raise ValueError('spec settings/pins changed')
    for key, name in [('common_sha256', 'reworld_common.py'), ('spec_builder_sha256', 'make_reworld_specs.py'),
                      ('runner_sha256', 'run_reworld_bench.py'), ('input_builder_sha256', 'build_reworld_inputs.py')]:
        if spec['generator'].get(key) != rc.sha(rc.HERE / name):
            raise ValueError('spec code provenance changed')
    seed = spec['seed']
    if seed != rc.aux_seed(spec['window'], spec['sample']) or spec['pass_seeds'] != [seed, seed + rc.B['pass2_seed_offset']]:
        raise ValueError('seed rule changed')
    if spec['prefix_seed'] != rc.aux_seed(spec['window'], 'prefix'):
        raise ValueError('prefix seed rule changed')
    if mode == 'continue' and spec['condition_encode_seeds'] != rc.condition_seeds(spec['window'], spec['sample']):
        raise ValueError('condition-encode seed rule changed')
    if spec.get('prompt') is not None:
        raise ValueError('ReWorld Stage-1 takes no prompt')
    rc.check_purpose(spec.get('purpose'), split, spec['run'], mode, spec['window'], spec['sample'])
    if spec.get('names_list') != rc.names_list(bench, split):
        raise ValueError('spec was built against a different window list (bench view)')
    rc.check_names_list(spec['purpose'], split, spec['names_list'])
    for row in spec['inputs'].values():
        if rc.sha(rc.safe_relative(input_root, row['path'])) != row['sha256']:
            raise ValueError('input hash mismatch')
    return spec


def check_lock():
    lock = (rc.HERE / 'requirements-reworld-video.lock.txt').read_text()
    pins = re.findall(r'^([A-Za-z0-9_.-]+)==([^\s\\]+)', lock, re.M)
    for package, version in pins:
        if importlib.metadata.version(package) != version:
            raise ValueError('inference environment differs from exact lock: ' + package)
    return len(pins)


def official_config(src, weights, size_wh=None, num_inference_steps=None):
    """The shipped configs/dualflow/sample_reworld.yaml with local paths and the registered frame size."""
    import yaml
    src = Path(src).resolve()
    data = yaml.safe_load((src / 'configs/dualflow/sample_reworld.yaml').read_text())
    shipped = dict(data)
    w, h = size_wh or (rc.W, rc.H)
    data.update(transformer_config=str(src / 'configs/dualflow/transformer_config.json'),
                dit_checkpoint=str(rc.dit_checkpoint(weights)), vae_model_source=str(rc.ltx_dir(weights)),
                video_width=w, video_height=h, conditioning_dir=None, output_root=None, max_scenes=0)
    if num_inference_steps is not None:
        data['num_inference_steps'] = num_inference_steps
    changed = sorted(k for k in data if data[k] != shipped.get(k))
    return data, changed


def check_registered(cfg):
    lat, ig = rc.B['latent'], rc.B['internal_guidance']
    want = dict(dit_latent_mode='vae_only', c_vae=lat['c_vae'], total_source_frames=rc.B['window_frames'],
                condition_source_frames=rc.B['condition_frames'], temporal_compression=lat['temporal_compression'],
                spatial_compression=lat['spatial_compression'], dtype=rc.B['dtype'], fps=rc.B['rope_fps'],
                null_text_seq_len=rc.B['null_text_seq_len'], internal_guidance_dit_block=ig['train_block'],
                internal_guidance_sampling_block=ig['sampling_block'], internal_guidance_sampling_scale=ig['scale'],
                internal_guidance_sampling_min_timestep=ig['min_timestep'])
    got = {k: getattr(cfg, k) for k in want}
    if got != want:
        raise ValueError('official sampling config differs from the registered settings: ' + json.dumps(got))


def load_official(src, weights, device='cuda', size_wh=None, num_inference_steps=None):
    """Official model loading (sample_dualflow_async, lines 140-151) + strict completeness checks."""
    rc.verify_source(src)
    rc.verify_weights(weights, full=True)
    locked = check_lock()
    if 'diffusers' in sys.modules or 'dualflow' in sys.modules:
        raise RuntimeError('clean inference process required; diffusers already imported')
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    sys.path.insert(0, str(Path(src).resolve() / 'src'))   # what scripts/sample_dualflow_dit.py does
    import torch
    from dualflow import sample_dualflow as official
    from dualflow.train_configs import DualFlowSampleConfig
    from dualflow.model_utils import load_ltx_transformer_from_json_channel_override
    from dualflow.ltx.model_loader import load_vae
    data, changed = official_config(src, weights, size_wh, num_inference_steps)
    cfg = DualFlowSampleConfig(**data)
    if size_wh is None and num_inference_steps is None:
        if cfg.num_inference_steps != rc.B['num_inference_steps'] or [cfg.video_width, cfg.video_height] != [rc.W, rc.H]:
            raise ValueError('registered size/steps not applied')
    check_registered(cfg)
    device = torch.device(device)
    dtype = torch.bfloat16 if cfg.dtype == 'bf16' else torch.float32
    # ---- official loading, same calls and order as dualflow/sample_dualflow.py sample_dualflow_async ----
    dit = load_ltx_transformer_from_json_channel_override(cfg.transformer_config, in_channels=cfg.c_vae,
                                                          out_channels=cfg.c_vae, torch_dtype=dtype).to(device).eval()
    official._load_dit_checkpoint(dit, cfg.dit_checkpoint)
    ig_kw = official._build_ig_kwargs(dit, cfg, device, dtype)
    dit.to(device)
    vae = load_vae(cfg.vae_model_source, dtype=dtype)
    vae.to(device).eval()
    # ---- bench checks: the permissive official loads (strict=False) must equal exact strict loads ----
    if ig_kw != OFFICIAL_IG:
        raise ValueError('official Internal Guidance kwargs differ: ' + repr(ig_kw))
    if sorted(dit.ig_heads.keys()) != [str(rc.B['internal_guidance']['sampling_block'])]:
        raise ValueError('unexpected IG heads')
    strict = [strict_check(dit, rc.dit_checkpoint(weights), 'transformer.', 'transformer+ig_heads.8'),
              strict_check(vae, rc.ltx_dir(weights) / 'vae/diffusion_pytorch_model.safetensors', '', 'vae')]
    return SimpleNamespace(cfg=cfg, config_dict=data, config_changed_from_shipped=changed, dit=dit, vae=vae, ig_kw=ig_kw,
                           device=device, dtype=dtype, official=official, strict=strict, locked_packages=locked, timings=[])


def strict_check(module, path, prefix, label):
    """Every checkpoint tensor is already in the module (same key set, bit-identical after the official dtype cast),
    then a strict=True reload (a value no-op) confirms zero missing / unexpected keys."""
    import torch
    from safetensors.torch import load_file
    state = load_file(str(path), device='cpu')
    if prefix:
        if any(not k.startswith(prefix) for k in state):
            raise ValueError('unprefixed checkpoint keys')
        state = {k[len(prefix):]: v for k, v in state.items()}
    own = module.state_dict()
    if set(own) != set(state):
        raise ValueError(f'{label}: key sets differ (missing {len(set(own) - set(state))}, unexpected {len(set(state) - set(own))})')
    for k, v in state.items():
        if not torch.equal(own[k], v.to(device=own[k].device, dtype=own[k].dtype)):
            raise ValueError(f'{label}: official load left {k} different from the checkpoint')
    loaded = module.load_state_dict(state, strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise ValueError('strict state-dict compatibility failed')
    dtypes = sorted({str(v.dtype) for v in state.values()})
    del state
    return dict(component=label, keys=len(own), missing_keys=[], unexpected_keys=[], checkpoint_dtypes=dtypes,
                official_load_bit_identical=True)


def to_uint8(vid):
    """Official sample_dualflow post-processing: [C,F,H,W] in [0,1] -> clamp -> float -> (x*255).astype(uint8)."""
    import numpy as np
    if vid.dim() == 5:
        vid = vid[0]
    vid_fhwc = vid.permute(1, 2, 3, 0).contiguous().clamp(0, 1).float().cpu().detach().numpy()
    return (vid_fhwc * 255).astype(np.uint8)


def check_reader(engine, path, expected):
    """The official decord reader must return exactly the intended uint8 frames at the registered 8 fps."""
    import numpy as np
    import torch
    from dualflow.ltx.video_io import read_video_leading_frames
    frames, fps = read_video_leading_frames(path, len(expected))
    got = (frames.permute(0, 2, 3, 1) * 255).round().to(torch.uint8).numpy()
    if got.shape != expected.shape or not np.array_equal(got, expected):
        raise ValueError('official reader does not reproduce the lossless conditioning frames: ' + str(path))
    if float(fps) != float(rc.B['native_fps']):
        raise ValueError('conditioning file fps differs from 8')


def official_pass(engine, cond_path, expected, seed, cond_seed):
    """One unchanged official 33-frame call (sample_dualflow_async conditioned branch, lines 158-215)."""
    import torch
    from dualflow.async_dualflow_inference import run_vae_latent_sampling
    from dualflow.dual_conditioning import conditioning_mask_packed, rgb_frames_to_latent_frames
    from dualflow.ltx.latents import decode_video
    from dualflow.sample_vae_resources import load_and_encode_condition_anchor
    cfg, dev, dt = engine.cfg, engine.device, engine.dtype
    nf, nh, nw = cfg.latent_frames, cfg.latent_height, cfg.latent_width
    if len(expected) != cfg.condition_source_frames:
        raise ValueError('exactly 9 conditioning frames required')
    check_reader(engine, cond_path, expected)
    sync(dev)
    started = time.monotonic()
    torch.manual_seed(cond_seed)
    z_pix_a, fps_rd = load_and_encode_condition_anchor(cfg, cond_path, engine.vae, dev, dt)
    sample_fps = float(fps_rd) if fps_rd > 0 else float(cfg.fps)
    if sample_fps != rc.B['rope_fps']:
        raise ValueError('RoPE fps differs from the registered 8.0')
    lf = rgb_frames_to_latent_frames(cfg.condition_source_frames, cfg.temporal_compression)
    cond_mask = conditioning_mask_packed(1, nf * nh * nw, lf, nh, nw, dev)
    z_pix = run_vae_latent_sampling(engine.dit, engine.vae, c_vae=cfg.c_vae, nf=nf, nh=nh, nw=nw,
                                    num_inference_steps=cfg.num_inference_steps, vae_model_source=cfg.vae_model_source,
                                    fps=sample_fps, caption_channels=int(engine.dit.config.caption_channels),
                                    null_text_seq_len=cfg.null_text_seq_len, seed=seed, device=dev, dtype=dt,
                                    output_video_path=None, conditioning_mask=cond_mask, z_pix_anchor=z_pix_a, **engine.ig_kw)
    vid = decode_video(engine.vae, z_pix, num_frames=nf, height=nh, width=nw, device=dev, dtype=dt)
    frames = to_uint8(vid)
    sync(dev)
    engine.timings.append(dict(kind='official_pass', seed=seed, condition_seed=cond_seed, seconds=time.monotonic() - started,
                               frames=len(frames), rope_fps=sample_fps))
    if frames.shape != (cfg.total_source_frames, cfg.video_height, cfg.video_width, 3):
        raise ValueError('official pass output shape mismatch')
    return frames


def vae_roundtrip(engine, path, expected, seed):
    """Codec path for the prefix and the real reference: official loader -> encode_video -> decode_video."""
    import torch
    from dualflow.ltx.latents import decode_video, encode_video
    from dualflow.sample_vae_resources import load_condition_bcfhw
    cfg, dev, dt = engine.cfg, engine.device, engine.dtype
    check_reader(engine, path, expected)
    sync(dev)
    started = time.monotonic()
    torch.manual_seed(seed)
    bcfhw, _ = load_condition_bcfhw(video_path=path, condition_source_frames=len(expected), video_width=cfg.video_width,
                                    video_height=cfg.video_height, device=dev, dtype=dt)
    enc = encode_video(engine.vae, bcfhw, device=dev, dtype=dt)
    vid = decode_video(engine.vae, enc['latents'], num_frames=enc['num_frames'], height=enc['height'], width=enc['width'],
                       device=dev, dtype=dt)
    frames = to_uint8(vid)
    sync(dev)
    engine.timings.append(dict(kind='vae_roundtrip', seed=seed, seconds=time.monotonic() - started, frames=len(frames)))
    if frames.shape != expected.shape:
        raise ValueError('codec round trip changed the clip shape')
    return frames


def sync(device):
    import torch
    if device.type == 'cuda':
        torch.cuda.synchronize()


def continuation_passes(engine, cond1_path, cond1, seeds, cond_seeds, save_first, write_condition2):
    if len(cond1) != 9 or len(seeds) != 2 or seeds[1] != seeds[0] + rc.B['pass2_seed_offset'] or len(cond_seeds) != 2:
        raise ValueError('registered two-pass context/seed rule required')
    p1 = official_pass(engine, cond1_path, cond1, seeds[0], cond_seeds[0])
    save_first(p1)
    ctx2 = p1[-9:]
    cond2_path = write_condition2(ctx2)
    p2 = official_pass(engine, cond2_path, ctx2, seeds[1], cond_seeds[1])
    return p1, p2


def execute_one(engine, spec, input_root, output_root, specfile):
    import numpy as np
    import torch
    cfg = engine.cfg
    wh = (cfg.video_width, cfg.video_height)
    dest = rc.safe_relative(output_root, spec['output_dir'])
    if dest.exists():
        raise FileExistsError('generated sample already exists; no overwrite or automatic retry')
    dest.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    engine.timings = []
    if engine.device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    hist_path = rc.safe_relative(input_root, spec['inputs'][bi.clip_name('history')]['path'])
    cond_path = rc.safe_relative(input_root, spec['inputs'][bi.clip_name('condition')]['path'])
    history = rc.decode(hist_path, wh)
    condition = rc.decode(cond_path, wh)
    if len(history) != 17 or len(condition) != 9 or not np.array_equal(history[-9:], condition):
        raise ValueError('conditioning must equal the last 9 real history frames')
    files = []
    with torch.inference_mode():
        prefix = vae_roundtrip(engine, hist_path, history, spec['prefix_seed'])
        rc.write_video(prefix, dest / 'prefix-native8.mp4', 8)
        files.append('prefix-native8.mp4')
        if spec['mode'] == 'prefix':
            rc.new_json(dest / 'sample_outputs.json', dict(schema='wm-calib-bench.reworld_video.output.v1', status='success', mode='prefix',
                                                           run=spec['run'], name=spec['name'], window=spec['window'], split=spec['split'],
                                                           purpose=spec.get('purpose'), seed_rules=seed_rules(spec), spec_sha256=rc.sha(specfile), frames=17,
                                                           files={'prefix-native8.mp4': dict(sha256=rc.sha(dest / 'prefix-native8.mp4'))}))
            return

        def write_condition2(frames):
            rc.write_video(frames, dest / 'pass2-condition8.mp4', 8)
            files.append('pass2-condition8.mp4')
            return dest / 'pass2-condition8.mp4'

        def save_first(frames):
            rc.write_video(frames, dest / 'pass1-native8.mp4', 8)
            files.append('pass1-native8.mp4')

        if spec['mode'] == 'continue':
            p1, p2 = continuation_passes(engine, cond_path, condition, spec['pass_seeds'], spec['condition_encode_seeds'],
                                         save_first, write_condition2)
        else:
            real = rc.decode(rc.safe_relative(input_root, spec['inputs'][bi.clip_name('real')]['path']), wh)
            if len(real) != 65 or not np.array_equal(real[:17], history):
                raise ValueError('reference needs the 65 raw 8 Hz frames whose first 17 are the history')
            segments = []
            for i, (lo, hi) in enumerate(((8, 41), (32, 65)), 1):
                rc.write_video(real[lo:hi], dest / f'ref-seg{i}-input8.mp4', 8)
                files.append(f'ref-seg{i}-input8.mp4')
                segments.append(real[lo:hi])
            p1 = vae_roundtrip(engine, dest / 'ref-seg1-input8.mp4', segments[0], spec['pass_seeds'][0])
            save_first(p1)
            p2 = vae_roundtrip(engine, dest / 'ref-seg2-input8.mp4', segments[1], spec['pass_seeds'][1])
        rc.write_video(p2, dest / 'pass2-native8.mp4', 8)
        files.append('pass2-native8.mp4')
    if p1.shape != (33, wh[1], wh[0], 3) or p2.shape != p1.shape:
        raise ValueError('two 33-frame passes required')
    native = np.concatenate([prefix[:8], p1, p2[9:]], axis=0)
    rc.write_video(native, dest / 'native8.mp4', 8)
    vision = rc.canvas(native)
    rc.write_video(vision, dest / 'vision.mp4', 10)
    files += ['native8.mp4', 'vision.mp4']
    sync(engine.device)
    record = dict(schema='wm-calib-bench.reworld_video.output.v1', status='success', mode=spec['mode'], run=spec['run'],
                  name=spec['name'], window=spec['window'], split=spec['split'], sample=spec['sample'], seed=spec['seed'],
                  pass_seeds=spec['pass_seeds'], condition_encode_seeds=spec.get('condition_encode_seeds'),
                  prefix_seed=spec['prefix_seed'], seed_rules=seed_rules(spec), purpose=spec.get('purpose'),
                  purpose_note=spec.get('purpose_note'), names_list=spec.get('names_list'),
                  spec_sha256=rc.sha(specfile), settings=spec['settings'], inputs=spec['inputs'],
                  code_commit=rc.PINS['code']['commit'],
                  weight_revisions={k: v['revision'] for k, v in rc.PINS['weights']['repos'].items()},
                  files={n: dict(sha256=rc.sha(dest / n), bytes=(dest / n).stat().st_size) for n in files},
                  frames=81, native_frames=65,
                  prefix_rule='per_clip: native 0..7 from the seeded 17-frame codec prefix; native 8..16 are pass 1 decoded conditions',
                  mapping=rc.timeline(), geometry=dict(model_source_rows=list(rc.SRC_ROWS), canvas_valid_rows=list(rc.CANVAS_ROWS)),
                  official_config={k: v for k, v in engine.config_dict.items() if k not in ('dit_checkpoint', 'vae_model_source', 'transformer_config')},
                  config_keys_changed_from_shipped_yaml=engine.config_changed_from_shipped, deviations_from_official_sampler=DEVIATIONS,
                  internal_guidance=engine.ig_kw, seconds=time.monotonic() - started, pass_wall_times=engine.timings,
                  strict_state_dict_loads=engine.strict, locked_packages=engine.locked_packages,
                  software=software(), attention_processors=sorted({type(p).__module__ + '.' + type(p).__name__ for p in engine.dit.attn_processors.values()}),
                  video_sha256=rc.sha(dest / 'vision.mp4'))
    record.update(device_record(engine.device))
    rc.new_json(dest / 'sample_outputs.json', record)


def seed_rules(spec):
    """Ruling D4: the rules and the values actually used, per call."""
    window, k = spec['window'], spec['sample']
    calls = [dict(call='prefix (17-frame codec round trip)', rule=rc.B['prefix_seed_rule'], seed=spec['prefix_seed'])]
    if spec['mode'] == 'continue':
        for p in (0, 1):
            calls.append(dict(call=f'pass {p + 1} condition-encode posterior draw (torch.manual_seed before the official loader)',
                              rule=rc.B['condition_encode_seed_rule'], label=f'bench-v1|{window}|{k}|cond{p + 1}',
                              seed=spec['condition_encode_seeds'][p]))
            calls.append(dict(call=f'pass {p + 1} sampling (official torch.manual_seed in run_vae_latent_sampling)',
                              rule=rc.B['seed_rule'] + ('' if p == 0 else ' + pass2_seed_offset'), seed=spec['pass_seeds'][p]))
    elif spec['mode'] == 'recon':
        for p in (0, 1):
            calls.append(dict(call=f'reference segment {p + 1} codec round trip', rule='seed_of(window, "recon")' + ('' if p == 0 else ' + pass2_seed_offset'),
                              seed=spec['pass_seeds'][p]))
    return dict(pass2_seed_offset=rc.B['pass2_seed_offset'], calls=calls)


def software():
    import diffusers
    import torch
    out = dict(torch=torch.__version__, diffusers=diffusers.__version__, python=sys.version.split()[0],
               python_executable=sys.executable, env_prefix=sys.prefix)
    for name in ('decord', 'transformers', 'numpy', 'opencv-python'):
        out[name] = importlib.metadata.version(name)
    try:
        from diffusers.models.attention_dispatch import _AttentionBackendRegistry
        backend, _ = _AttentionBackendRegistry.get_active_backend()
        out['diffusers_attention_backend'] = getattr(backend, 'value', str(backend))
    except Exception as error:  # recorded, never silently assumed
        out['diffusers_attention_backend'] = 'unavailable: ' + type(error).__name__
    return out


def device_record(device):
    import torch
    if device.type != 'cuda':
        return dict(gpu=None, device='cpu')
    gpu = torch.cuda.get_device_properties(0)
    return dict(gpu=dict(name=gpu.name, total_MiB=gpu.total_memory / 2**20, device=str(torch.cuda.current_device()),
                         uuid=str(getattr(gpu, 'uuid', ''))),
                gpu_memory=dict(peak_allocated_MiB=torch.cuda.max_memory_allocated() / 2**20,
                                peak_reserved_MiB=torch.cuda.max_memory_reserved() / 2**20),
                attention_backend=dict(torch_SDPA_flash=torch.backends.cuda.flash_sdp_enabled(),
                                       torch_SDPA_mem_efficient=torch.backends.cuda.mem_efficient_sdp_enabled(),
                                       torch_SDPA_math=torch.backends.cuda.math_sdp_enabled(),
                                       note='enabled backend flags; the selected kernel per call is not profiled'))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--specs', type=Path, nargs='+', required=True)
    ap.add_argument('--bench', type=Path, required=True)
    ap.add_argument('--input-root', type=Path, required=True)
    ap.add_argument('--output-root', type=Path, required=True)
    ap.add_argument('--src', type=Path, required=True)
    ap.add_argument('--weights', type=Path, required=True)
    ap.add_argument('--frozen-file', type=Path)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    paths = [a.bench, a.input_root, a.output_root, a.src, a.weights, *a.specs] + ([a.frozen_file] if a.frozen_file else [])
    if not all(p.is_absolute() for p in paths):
        raise ValueError('explicit absolute paths required for every path argument (ruling D10)')
    specs = []
    for path in a.specs:
        spec = json.loads(path.read_text())
        validate_spec(spec, a.bench, a.input_root, a.frozen_file)
        specs.append((path, spec))
    rc.verify_source(a.src)
    if a.dry_run:
        try:
            rc.require_approved()
            approved = True
        except ValueError:
            approved = False
        print(json.dumps(dict(CPU_spec_and_source_hashes='PASS', specs=len(specs), coordinator_approved=approved, GPU_started=False)))
        return
    rc.require_approved()
    if os.environ.get('CUDA_VISIBLE_DEVICES') is None:
        raise ValueError('launch through the approved GPU wrapper (CUDA_VISIBLE_DEVICES unset)')
    if not (a.weights / 'download-verified.json').is_file():
        raise ValueError('owner-approved pinned weight download proof required')
    engine = load_official(a.src, a.weights)
    for path, spec in specs:
        try:
            execute_one(engine, spec, a.input_root, a.output_root, path)
        except BaseException as error:
            dest = rc.safe_relative(a.output_root, spec['output_dir'])
            if dest.is_dir() and not (dest / 'failure.json').exists():
                rc.new_json(dest / 'failure.json', dict(error_type=type(error).__name__, failed=True, no_retry=True,
                                                        all_partial_outputs_retained=True))
            raise
        print(json.dumps(dict(name=spec['name'], status='success')), flush=True)


if __name__ == '__main__':
    main()
