#!/usr/bin/env python3
"""CPU checks with the real VaViM-L weights (no GPU is touched; run with CUDA_VISIBLE_DEVICES empty).

  check_strict_load.py --src SRC --weights WEIGHTS --input-root INPUTS --windows W [W ...] --out NEW_JSON
                       [--equivalence-frames N] [--top-k K] [--precision fp32|bf16] [--threads 8]

1. The runner's load_official on CPU: source and weight hashes, pickle allow-lists, the exact lock, the released
   load_pretrained_gpt sequence with a strict state-dict load, mup base shapes (file = checkpoint record = model) and the
   readout scaling (lm_head width_mult 8, tied to the token embedding).
2. Tokenizer round trip on real DEV history frames (ticks -20, -10, 0 of each window's history10.mp4): token grids,
   determinism, PSNR of the released uint8 conversion against the input; float32 against CPU bfloat16 autocast
   (whether autocast reaches the TorchScript graphs, token agreement, decode difference).
3. Optional (--equivalence-frames N > 0): the released forward_inference (torch.manual_seed) against the kit sampling path
   (per-row generator) at batch 1 on the first window's context tokens, N frames, same seed, and the time per token step
   at batch 1 and batch 2 (CPU).
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
import types

sys.dont_write_bytecode = True
import vavim_common as vc  # noqa: E402
import run_vavim_bench as rb  # noqa: E402


def psnr(a, b):
    import numpy as np
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float('inf') if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', type=Path, required=True)
    ap.add_argument('--weights', type=Path, required=True)
    ap.add_argument('--input-root', type=Path, required=True)
    ap.add_argument('--windows', nargs='+', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--equivalence-frames', type=int, default=0)
    ap.add_argument('--top-k', type=int, default=None, help='CPU check only; default: the registered top_k')
    ap.add_argument('--precision', choices=['fp32', 'bf16'], default='fp32')
    ap.add_argument('--threads', type=int, default=8)
    a = ap.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise SystemExit('CPU check: set CUDA_VISIBLE_DEVICES= (empty)')
    if a.out.exists():
        raise FileExistsError(a.out)
    import numpy as np
    t0 = time.monotonic()
    engine = rb.load_official(a.src, a.weights, 'cpu')
    torch = engine.torch
    torch.set_num_threads(a.threads)
    result = dict(schema='wm-calib-bench.vavim.cpu-check.v1', device='cpu', threads=a.threads, torch=torch.__version__,
                  load=engine.record, load_wall_seconds=time.monotonic() - t0, kit_files=dict(
                      common=vc.sha(vc.HERE / 'vavim_common.py'), runner=vc.sha(vc.HERE / 'run_vavim_bench.py'),
                      check=vc.sha(Path(__file__)), pins=vc.sha(vc.PINS_PATH)))
    print(json.dumps(dict(loaded=True, strict=engine.record['strict_state_dict_load'])), flush=True)

    # ---- 2. tokenizer round trip on real DEV frames ----
    rt = []
    contexts = {}
    with torch.inference_mode():
        for w in a.windows:
            root = a.input_root / 'windows' / 'dev' / w
            hist = vc.decode(root / 'history10.mp4', vc.MODEL_WH)
            pick = hist[[0, 10, 20]]
            x = vc.to_model_input(torch, pick, 'cpu')
            t = time.monotonic()
            tok32 = engine.encoder(x)
            t_enc = (time.monotonic() - t) / len(pick)
            tok32b = engine.encoder(x)
            t = time.monotonic()
            img32 = engine.decoder(tok32)
            t_dec = (time.monotonic() - t) / len(pick)
            with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
                tok16 = engine.encoder(x)
                img16 = engine.decoder(tok32)
            u32, u16 = vc.to_uint8(torch, img32), vc.to_uint8(torch, img16)
            rt.append(dict(window=w, ticks=[-20, -10, 0], token_shape=list(tok32.shape), token_dtype=str(tok32.dtype),
                           token_range=[int(tok32.min()), int(tok32.max())], deterministic=bool(torch.equal(tok32, tok32b)),
                           unique_tokens_per_frame=[int(len(torch.unique(f))) for f in tok32],
                           psnr_fp32=[round(psnr(u, p), 3) for u, p in zip(u32, pick)],
                           psnr_bf16_decode=[round(psnr(u, p), 3) for u, p in zip(u16, pick)],
                           bf16_autocast_token_agreement=float((tok16 == tok32).float().mean()),
                           bf16_vs_fp32_decode_max_abs_uint8=int(np.abs(u16.astype(int) - u32.astype(int)).max()),
                           seconds_per_frame=dict(encode_fp32=t_enc, decode_fp32=t_dec),
                           decoded_range=[float(img32.min()), float(img32.max())]))
            print(json.dumps(rt[-1]), flush=True)
            full = engine.encoder(vc.to_model_input(torch, hist, 'cpu'))
            contexts[w] = full[[k + 20 for k in vc.CONTEXT_K]]
            rt[-1]['context_tokens_sha256'] = vc.sha_bytes(contexts[w].numpy().astype('int16').tobytes())
    result['round_trip'] = rt

    # ---- 3. released forward_inference vs kit path, batch 1, real weights ----
    if a.equivalence_frames > 0:
        top_k = a.top_k if a.top_k is not None else vc.B['sampling']['top_k']
        if top_k is None:
            raise SystemExit('give --top-k (no registered top_k yet)')
        s = dict(vc.B['sampling'], top_k=top_k)
        old = vc.B['sampling']
        vc.B['sampling'] = s                     # CPU check only: the registered dict is restored below
        old_autocast = vc.B['autocast']
        if a.precision == 'fp32':
            vc.B['autocast'] = dict(old_autocast, gpt='float32 (CPU check)')
        try:
            w = a.windows[0]
            burnin = contexts[w]
            path = a.out.with_name(a.out.stem + '-equivalence.json')
            with torch.inference_mode():
                eq = rb.equivalence_check(engine, burnin, w, path, a.equivalence_frames)
            steps = a.equivalence_frames * vc.B['tokens_per_frame']
            result['equivalence'] = dict(eq, precision=a.precision, record=str(path),
                                         seconds_per_token_step_batch1=eq['kit_seconds'] / steps)
            print(json.dumps(result['equivalence']), flush=True)
            # batch-2 timing for one frame (kit path, seeds of samples 0 and 1)
            seeds = [vc.local_seed(w, 0), vc.local_seed(w, 1)]
            gens = [torch.Generator(device='cpu').manual_seed(x) for x in seeds]
            b2 = burnin.unsqueeze(0).expand(2, *burnin.shape).contiguous()
            t = time.monotonic()
            with torch.inference_mode(), rb.kit_sampling(engine, gens), rb.autocast(engine, 'gpt'):
                out2 = engine.gpt.forward_inference(number_of_future_frames=1, burnin_visual_tokens=b2, temperature=s['temperature'],
                                                    topk_sampler=top_k, use_kv_cache=True)
            result['timing_batch2'] = dict(frames=1, seconds=time.monotonic() - t,
                                           seconds_per_token_step=(time.monotonic() - t) / vc.B['tokens_per_frame'],
                                           rows_differ=bool(not torch.equal(out2[0], out2[1])))
            print(json.dumps(result['timing_batch2']), flush=True)
        finally:
            vc.B['sampling'] = old
            vc.B['autocast'] = old_autocast
    result['wall_seconds'] = time.monotonic() - t0
    vc.new_json(a.out, result)
    print(json.dumps(dict(written=str(a.out), GPU_started=False)))


if __name__ == '__main__':
    main()
