#!/usr/bin/env python3
"""GPU check of the LTX-2.5 environment without weights (well under 1 GB of GPU memory).

Checks, on the GPU this process sees (CUDA_VISIBLE_DEVICES):
  - CUDA initialisation, device, driver, torch / natten / triton versions;
  - bf16 SDPA (the transformer's attention on non-Hopper GPUs) on a 3024 + 201 token sequence: two calls bit-identical;
  - the official fp8-cast Linear (ltx_core.quantization.fp8_cast: weights rounded to e4m3 at load, upcast to bf16 in
    the matmul): equals a bf16 Linear on the dequantized weights, bit for bit;
  - natten.na3d with the CUTLASS FNA backend (the DiffVAE's neighbourhood attention in chunked_eager mode) against
    LTX's eager fallback (fallback_na.eager.na3d) on a small volume, and two natten calls bit-identical;
  - the CUDA generator: two torch.Generator(cuda).manual_seed(s) streams identical.
Prints one JSON record. Run it under the research server's GPU lock, on the GPU its rules allow (models/ltx25/local.env):
  flock "$WCB_GPU_LOCK" env CUDA_VISIBLE_DEVICES="$WCB_LOCAL_GPU" "$LTX25_ENV/venv/bin/python" probe_ltx25_env.py
"""
from __future__ import annotations

import json
import platform
import sys
import time


def main() -> int:
    import torch
    out, ok = {}, True
    t0 = time.time()
    if not torch.cuda.is_available():
        print(json.dumps(dict(probe='FAIL', error='CUDA is not available')))
        return 1
    free, total = torch.cuda.mem_get_info()
    p = torch.cuda.get_device_properties(0)
    import natten
    import triton
    out['device'] = dict(name=p.name, capability=f'{p.major}.{p.minor}', total_gib=round(total / 2**30, 2), free_gib_at_start=round(free / 2**30, 2),
                         torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(), natten=natten.__version__,
                         triton=triton.__version__, python=platform.python_version())
    dev = torch.device('cuda')
    g = torch.Generator(device=dev).manual_seed(714236784)

    # SDPA, bf16, 24 heads x 128 (the LTX transformer's video attention size class), 3024 video + 201 audio tokens
    q, k, v = (torch.randn(1, 24, 3225, 128, device=dev, dtype=torch.bfloat16, generator=g) for _ in range(3))
    a1 = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    a2 = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    out['sdpa_repeat_identical'] = bool(torch.equal(a1, a2))
    ok &= out['sdpa_repeat_identical']
    del q, k, v, a1, a2

    # fp8-cast Linear, as ltx_core.quantization.fp8_cast applies it
    from ltx_core.quantization.fp8_cast import _replace_fwd_with_upcast
    lin = torch.nn.Linear(4096, 4096, bias=True, device=dev, dtype=torch.bfloat16)
    w8, b8 = lin.weight.detach().to(torch.float8_e4m3fn), lin.bias.detach().to(torch.float8_e4m3fn)
    ref = torch.nn.Linear(4096, 4096, bias=True, device=dev, dtype=torch.bfloat16)
    with torch.no_grad():
        ref.weight.copy_(w8.to(torch.bfloat16))
        ref.bias.copy_(b8.to(torch.bfloat16))
        _replace_fwd_with_upcast(lin, with_stochastic_rounding=False)
        lin.weight.copy_(w8)
        lin.bias.copy_(b8)
        x = torch.randn(3225, 4096, device=dev, dtype=torch.bfloat16, generator=g)
        y8, yr = lin(x), ref(x)
    out['fp8_cast_linear'] = dict(class_=type(lin).__name__, weight_dtype=str(lin.weight.dtype), equals_dequantized_bf16=bool(torch.equal(y8, yr)))
    ok &= out['fp8_cast_linear']['equals_dequantized_bf16']
    del lin, ref, x, y8, yr, w8, b8

    # natten (CUTLASS FNA) vs LTX's eager neighbourhood attention, (B, T, H, W, heads, head_dim)
    from ltx_core.model.video_vae.transformer.fallback_na.eager import na3d as eager_na3d
    shape, ks = (1, 6, 18, 32, 4, 64), (3, 7, 7)
    q, k, v = (torch.randn(*shape, device=dev, dtype=torch.bfloat16, generator=g) for _ in range(3))
    q = q * (shape[-1] ** -0.5)
    n1 = natten.na3d(q, k, v, kernel_size=ks, scale=1.0, backend='cutlass-fna')
    n2 = natten.na3d(q, k, v, kernel_size=ks, scale=1.0, backend='cutlass-fna')
    e = eager_na3d(q.float(), k.float(), v.float(), kernel_size=list(ks), scale=1.0)
    diff = (n1.float() - e.float()).abs()
    out['natten_cutlass_fna'] = dict(shape=list(shape), kernel=list(ks), repeat_identical=bool(torch.equal(n1, n2)), max_abs_vs_eager_fp32=float(diff.max()),
                                     mean_abs_vs_eager_fp32=float(diff.mean()))
    ok &= out['natten_cutlass_fna']['repeat_identical'] and out['natten_cutlass_fna']['max_abs_vs_eager_fp32'] < 0.05
    del q, k, v, n1, n2, e, diff

    g1 = torch.Generator(device=dev).manual_seed(123)
    g2 = torch.Generator(device=dev).manual_seed(123)
    out['cuda_generator_reproducible'] = bool(torch.equal(torch.randn(1000, device=dev, generator=g1), torch.randn(1000, device=dev, generator=g2)))
    ok &= out['cuda_generator_reproducible']
    torch.cuda.synchronize()
    out['peak_gpu_gib'] = round(torch.cuda.max_memory_reserved() / 2**30, 3)
    out['seconds'] = round(time.time() - t0, 1)
    print(json.dumps(dict(probe='PASS' if ok else 'FAIL', **out), indent=1))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
