#!/usr/bin/env python3
"""CPU-only official load + strict state-dict, API and codec-reader checks before any GPU lease.

Runs the runner's load_official on the CPU (the official sample_dualflow loading calls), which proves that the
permissive official loads (strict=False) leave the Video DiT, its Internal Guidance head (block 8) and the LTX VAE
bit-identical to the pinned checkpoints with zero missing / unexpected keys. Then checks every official function the
runner calls by signature, the scheduler configuration, the uint8 rule, the identity of the official resize/crop at
the model size and the exactness of the official decord reader on a lossless 8 fps clip. No GPU, no AV2 frames.
"""
import argparse
import inspect
import json
from pathlib import Path
import sys
import tempfile
import time
sys.dont_write_bytecode = True
import reworld_common as rc
import run_reworld_bench as runner


def params(fn):
    return list(inspect.signature(fn).parameters)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--src', type=Path, required=True)
    ap.add_argument('--weights', type=Path, required=True)
    ap.add_argument('--report', type=Path, required=True)
    a = ap.parse_args()
    if a.report.exists():
        raise FileExistsError('fresh check report required')
    started = time.monotonic()
    engine = runner.load_official(a.src, a.weights, device='cpu')
    load_seconds = time.monotonic() - started
    import numpy as np
    import torch
    import diffusers
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.utils import export_utils
    from dualflow.async_dualflow_inference import run_vae_latent_sampling
    from dualflow.sample_vae_resources import load_and_encode_condition_anchor, load_condition_bcfhw
    from dualflow.ltx.latents import decode_video, encode_video, get_rope_scale_factors
    from dualflow.ltx.video_io import crop_video, read_video_leading_frames, resize_video
    from dualflow.ltx.model_loader import load_scheduler
    from dualflow.ltx.transformer_ltx_ig import LTXVideoTransformer3DModelWithIG
    from dualflow.model_utils import null_text_embeddings
    cfg = engine.cfg
    checks = {}
    want = {'run_vae_latent_sampling': (run_vae_latent_sampling, ['transformer', 'vae', 'c_vae', 'nf', 'nh', 'nw', 'num_inference_steps',
                                                                  'vae_model_source', 'fps', 'caption_channels', 'null_text_seq_len', 'seed',
                                                                  'device', 'dtype', 'output_video_path', 'conditioning_mask',
                                                                  'z_pix_anchor', 'ig_head', 'ig_block_idx', 'ig_scale', 'ig_min_timestep']),
            'load_and_encode_condition_anchor': (load_and_encode_condition_anchor, ['cfg', 'video_path', 'vae', 'device', 'dtype']),
            'load_condition_bcfhw': (load_condition_bcfhw, ['video_path', 'condition_source_frames', 'video_width', 'video_height', 'device', 'dtype']),
            'encode_video': (encode_video, ['vae', 'image_or_video', 'patch_size', 'patch_size_t', 'device', 'dtype', 'generator', 'pixel_values_in_zero_one']),
            'decode_video': (decode_video, ['vae', 'latents', 'num_frames', 'height', 'width', 'device', 'dtype', 'patch_size', 'patch_size_t',
                                            'decode_timestep', 'decode_noise_scale', 'generator']),
            'read_video_leading_frames': (read_video_leading_frames, ['video_path', 'num_frames'])}
    for name, (fn, expected) in want.items():
        if params(fn) != expected:
            raise ValueError('official signature changed: ' + name)
        checks[name] = 'signature PASS'
    dv = inspect.signature(decode_video).parameters
    if dv['decode_timestep'].default != 0.0 or dv['decode_noise_scale'].default is not None:
        raise ValueError('decode defaults changed')
    if 'per_token_timesteps' not in inspect.signature(FlowMatchEulerDiscreteScheduler.step).parameters:
        raise ValueError('scheduler lacks per_token_timesteps')
    fwd = inspect.signature(LTXVideoTransformer3DModelWithIG.forward).parameters
    if not {'return_intermediate_velocity', 'intermediate_block_idx', 'rope_interpolation_scale', 'video_coords'} <= set(fwd):
        raise ValueError('IG forward signature changed')
    sched = load_scheduler(cfg.vae_model_source)
    sched.set_timesteps(cfg.num_inference_steps, device='cpu')
    ts = sched.timesteps
    sc = dict(sched.config)
    if sc['use_dynamic_shifting'] or sc['shift'] != 1.0 or len(ts) != rc.B['num_inference_steps']:
        raise ValueError('scheduler configuration differs from the pinned LTX-Video 0.9.5 config')
    checks['scheduler'] = dict(class_name=type(sched).__name__, use_dynamic_shifting=sc['use_dynamic_shifting'], shift=sc['shift'],
                               first_timesteps=[round(float(x), 4) for x in ts[:3]], last_timesteps=[round(float(x), 4) for x in ts[-3:]],
                               sigmas_tail=[round(float(x), 6) for x in sched.sigmas[-3:]])
    src_export = inspect.getsource(export_utils.export_to_video)
    if '(frame * 255).astype(np.uint8)' not in src_export:
        raise ValueError('official uint8 conversion changed')
    checks['uint8_rule'] = 'export_to_video uses (frame * 255).astype(np.uint8): truncation (the runner applies the same rule)'
    rope = get_rope_scale_factors(rc.B['rope_fps'])
    if rope != [1.0, 32.0, 32.0]:
        raise ValueError('RoPE scale factors at 8 fps changed')
    checks['rope_scale_at_8fps'] = rope
    enc, mask = null_text_embeddings(1, cfg.null_text_seq_len, int(engine.dit.config.caption_channels), torch.device('cpu'), torch.bfloat16)
    if tuple(enc.shape) != (1, 1, 4096) or enc.abs().max() != 0 or not bool(mask.all()):
        raise ValueError('null text embedding changed')
    checks['null_text'] = 'zeros [1, 1, 4096], mask ones'
    x = torch.rand(9, 3, cfg.video_height, cfg.video_width)
    if not torch.equal(crop_video(resize_video(x, cfg.video_width, cfg.video_height), cfg.video_width, cfg.video_height), x):
        raise ValueError('official resize/crop is not the identity at the model size')
    checks['official_resize_crop_at_model_size'] = 'identity'
    with tempfile.TemporaryDirectory() as td:
        clip = np.random.default_rng(0).integers(0, 256, (9, cfg.video_height, cfg.video_width, 3), dtype=np.uint8)
        rc.write_video(clip, Path(td) / 'c.mp4', 8)
        runner.check_reader(engine, Path(td) / 'c.mp4', clip)
    checks['decord_reader'] = 'lossless libx264rgb 8 fps clip read bit-exactly at 8.0 fps'
    vae_cfg = dict(engine.vae.config)
    checks['vae'] = dict(decoder_causal=vae_cfg['decoder_causal'], encoder_causal=vae_cfg['encoder_causal'],
                         timestep_conditioning=vae_cfg['timestep_conditioning'], tiling_enabled=bool(getattr(engine.vae, 'use_tiling', False)))
    if checks['vae']['tiling_enabled']:
        raise ValueError('the official ReWorld sampler never enables VAE tiling')
    report = dict(PASS=True, strict=True, components=engine.strict, internal_guidance=engine.ig_kw,
                  ig_heads=sorted(engine.dit.ig_heads.keys()), official_config={k: v for k, v in engine.config_dict.items()
                                                                                 if k not in ('dit_checkpoint', 'vae_model_source', 'transformer_config')},
                  config_keys_changed_from_shipped_yaml=engine.config_changed_from_shipped, api_checks=checks,
                  locked_packages=engine.locked_packages, software=runner.software(), load_seconds=load_seconds,
                  attention_processors=sorted({type(p).__module__ + '.' + type(p).__name__ for p in engine.dit.attn_processors.values()}),
                  GPU_started=False, CPU_only=True, pins_sha256=rc.sha(rc.PINS_PATH), runner_sha256=rc.sha(Path(runner.__file__)),
                  torch=torch.__version__, diffusers=diffusers.__version__)
    rc.new_json(a.report, report)
    print(json.dumps(dict(PASS=True, components=len(engine.strict), GPU_started=False)))


if __name__ == '__main__':
    main()
