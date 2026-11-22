#!/usr/bin/env python3
"""Orbis 2 real references from the tokenizer only (DEV validation of the canvas instrument; research server, GPU 1).

The real reference of the Orbis 2 entry is the per-frame tokenizer round trip of the 81-frame real clip
(run_orbis2_bench.py --mode recon). This tool computes the same function without loading the two predictors
(about 3 GB of GPU memory instead of about 19 GB):
  - the tokenizer module is built from tok/config.yaml exactly as PredictorModule.build_tokenizer builds it
    (loss and entropy configs dropped, distill_model_type None, no pretrained-weight download);
  - its weights are the L1 checkpoint's ae.* tensors (read with mmap), i.e. the tokenizer state of the loaded L1 model
    (in the full model, the L1 checkpoint overwrites every tok/ tensor; they differ only in the unused VQ codebooks);
  - encode / decode go through Orbis 2's own IFSelectBranchFirstStageHandler (rec branch, enc_scale 4), frame by frame,
    under the same fp16 autocast, with the demo's ClipAugmenter input path and the adapter's uint8 rule.
Outputs use the run layout of the adapter: <out>/<run>/native/<window>-recon/vision.mp4 (81 frames, 512x288, 10 fps,
lossless) and sample_outputs.json. DEV only. The clips stand in for the cluster run orbis2-dev-recon
in the DEV validation (another GPU type: close, not bit-identical).

  tokenizer_roundtrip.py --bench BENCH_V1 --names FILE --models-dir DIR --orbis2-src DIR --out DIR [--run orbis2-dev-recon-tok]
                         [--max-gpu-gib 8] [--leave-free-mib 6144]
Run under the shared GPU lock (`flock "$WCB_GPU_LOCK" ...`) with CUDA_VISIBLE_DEVICES set to the local GPU.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
import time
import types
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_orbis2_bench as rob  # noqa: E402  (pins, decode / write helpers, source digest)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bench', type=Path, required=True); ap.add_argument('--names', type=Path, required=True)
    ap.add_argument('--models-dir', type=Path, required=True); ap.add_argument('--orbis2-src', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True); ap.add_argument('--run', default='orbis2-dev-recon-tok')
    ap.add_argument('--max-gpu-gib', type=float, default=8.0); ap.add_argument('--leave-free-mib', type=int, default=6144)
    ap.add_argument('--threads', type=int, default=2)
    a = ap.parse_args()
    names = a.names.read_text().split()
    dev = set((a.bench / 'names-dev.txt').read_text().split())
    if not set(names) <= dev:
        sys.exit('refusing: DEV windows only')
    digest = rob.source_digest(a.orbis2_src)
    if digest != rob.PINS['code']['source_digest']:
        sys.exit(f'Orbis 2 source digest {digest} differs from the pins')
    for rel in ('L1/checkpoints/last.ckpt', 'tok/config.yaml'):
        rec = rob.PINS['weights']['files'][rel]
        p = a.models_dir / rel
        if 'bytes' in rec and p.stat().st_size != rec['bytes']:
            sys.exit(f'{p}: size differs from the pins')
        if rel.endswith('.yaml') and rob.sha256_file(p) != rec['sha256']:
            sys.exit(f'{p}: sha256 differs from the pins')
    os.environ['ORBIS2_MODELS_DIR'] = str(a.models_dir)
    sys.path.insert(0, str(a.orbis2_src))
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from util import instantiate_from_config
    from data.video_loaders import ClipAugmenter, DecordFrameAdapter, ResizeCenterPolicy
    from models.second_stage.fm_model_v2 import IFSelectBranchFirstStageHandler
    torch.set_num_threads(a.threads)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    free, total = torch.cuda.mem_get_info(0)
    if free / 2**20 < a.leave_free_mib + 4096:
        sys.exit(f'GPU free {free / 2**20:.0f} MiB: not enough to stay {a.leave_free_mib} MiB above zero')
    torch.cuda.set_per_process_memory_fraction(min(1.0, a.max_gpu_gib * 2**30 / total), 0)

    t0 = time.time()
    tcfg = OmegaConf.load(a.models_dir / 'tok' / 'config.yaml')                 # PredictorModule.build_tokenizer, step by step
    model_cfg = OmegaConf.to_container(tcfg.model, resolve=False)
    for key in ('loss_config', 'entropy_loss_weight_scheduler_config'):
        model_cfg.get('params', {}).pop(key, None)
    model_cfg.get('params', {})['distill_model_type'] = None
    enc = model_cfg.get('params', {}).get('encoder_config', {}).get('params', {})
    if isinstance(enc, dict):
        enc['use_pretrained_weights'] = False
    ae = instantiate_from_config(OmegaConf.create(model_cfg))
    sd = torch.load(a.models_dir / 'L1' / 'checkpoints' / 'last.ckpt', map_location='cpu', weights_only=True, mmap=True)['state_dict']
    ae_sd = {k[3:]: v for k, v in sd.items() if k.startswith('ae.')}
    res = ae.load_state_dict(ae_sd, strict=False)
    del sd, ae_sd
    if res.missing_keys:
        sys.exit(f'tokenizer keys missing from the L1 checkpoint: {res.missing_keys[:5]}')
    ae.eval().to('cuda')
    stub = types.SimpleNamespace(ae=ae, vit=types.SimpleNamespace(in_channels=16, input_size=(18, 32)), enc_scale=4.0, enc_scale_dino=4.0,
                                 use_precomputed_training_inputs=False)
    fs = IFSelectBranchFirstStageHandler(stub, encoder_branch='rec')
    augmenter = ClipAugmenter(DecordFrameAdapter(), ResizeCenterPolicy((288, 512)))
    load_s = round(time.time() - t0, 1)
    gpu = rob.gpu_info(torch); sw = rob.software(None)

    def roundtrip(frames_u8):
        x = augmenter(frames_u8)
        out = []
        with torch.no_grad(), torch.autocast(dtype=torch.float16, device_type='cuda'):
            for i in range(x.shape[0]):
                lat = fs.encode_frames(x[i:i + 1].unsqueeze(0).to('cuda'))
                out.append(fs.decode_frames(lat, output_device='cpu')[0, 0])
        y = ((torch.stack(out).float() + 1.0) / 2.0).mul(255).add_(0.5).clamp_(0, 255).to(torch.uint8)
        return y.permute(0, 2, 3, 1).contiguous().numpy()

    done = 0
    for w in names:
        d = a.out / a.run / 'native' / f'{w}-recon'
        if (d / 'sample_outputs.json').is_file() and json.loads((d / 'sample_outputs.json').read_text()).get('status') == 'success':
            continue
        d.mkdir(parents=True, exist_ok=True)
        free, _ = torch.cuda.mem_get_info(0)
        if free / 2**20 < a.leave_free_mib:
            sys.exit(f'GPU free {free / 2**20:.0f} MiB < {a.leave_free_mib}: stopping')
        src = a.bench / 'windows-dev' / w / 'canvas' / 'real.mp4'
        frames, info = rob.decode_rgb(src, a.threads)
        if frames.shape != (81, 288, 512, 3):
            sys.exit(f'{src}: {frames.shape}')
        ts = time.time(); torch.cuda.reset_peak_memory_stats()
        clip = roundtrip(frames)
        rt = round(time.time() - ts, 2)
        rob.write_lossless(clip, d / 'vision.mp4', a.threads)
        back, vinfo = rob.decode_rgb(d / 'vision.mp4', a.threads)
        if not np.array_equal(back, clip):
            sys.exit(f'{w}: vision.mp4 does not decode to the written frames')
        h = np.abs(clip[:21].astype(np.int16) - frames[:21])
        rec = dict(schema='wm-calib-bench.orbis2.sample_outputs.v1', status='success', name=f'{w}-recon', window=w, split='dev', mode='recon', k=None, seed=0,
                   run=a.run, tool='tokenizer_roundtrip.py (tokenizer only; validation stand-in for orbis2-dev-recon)', tool_sha256=rob.sha256_file(Path(__file__)),
                   input=dict(path=str(src.relative_to(a.bench)), file_sha256=rob.sha256_file(src), frames_sha256=hashlib.sha256(frames.tobytes()).hexdigest()),
                   video='vision.mp4', video_sha256=rob.sha256_file(d / 'vision.mp4'), video_bytes=(d / 'vision.mp4').stat().st_size, frames=vinfo['frames'],
                   width=vinfo['width'], height=vinfo['height'], fps=vinfo['fps'], frames_sha256=hashlib.sha256(clip.tobytes()).hexdigest(),
                   prefix21_frames_sha256=hashlib.sha256(clip[:21].tobytes()).hexdigest(),
                   diagnostics=dict(roundtrip_vs_input_history=dict(frames=[0, 20], max_abs=int(h.max()), mean_abs=float(h.mean()))),
                   timing_s=dict(roundtrip=rt), gpu=gpu, software=sw, model_load_s=load_s, host=socket.gethostname(),
                   gpu_memory_gib=dict(max_allocated=round(torch.cuda.max_memory_allocated() / 2**30, 2)), host_memory_gib=rob.host_memory(),
                   finished_utc=datetime.now(timezone.utc).isoformat(timespec='seconds'))
        rob.write_json_atomic(d / 'sample_outputs.json', rec)
        done += 1
        print(json.dumps(dict(window=w, roundtrip_s=rt, gpu_gib=rec['gpu_memory_gib']['max_allocated'], rss_gib=rec['host_memory_gib'].get('peak_rss'))), flush=True)
    print(json.dumps(dict(done=done, of=len(names), load_s=load_s)), flush=True)


if __name__ == '__main__':
    main()
