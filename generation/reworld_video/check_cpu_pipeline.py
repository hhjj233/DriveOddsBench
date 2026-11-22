#!/usr/bin/env python3
"""CPU code-path check of the whole runner at a tiny size (NOT a sample, NOT a quality or timing result).

Loads the official model on the CPU at a check-only frame size (default 256x128) with 2 sampling steps, rebuilds one
DEV smoke window's inputs at that size, and runs runner.execute_one for one continuation (both official passes,
pass 2 on generated frames) and one reference, then applies the smoke structural checks. This exercises every call
the GPU run makes (official loader, decord reader, condition encode, Internal Guidance forward, per-token scheduler,
decode, uint8 rule, assembly, canvas map, records) without a GPU. DEV only; outputs go to a fresh scratch directory.
"""
import argparse
import json
from pathlib import Path
import sys
import time
sys.dont_write_bytecode = True
import reworld_common as rc
import build_reworld_inputs as bi
import run_reworld_bench as runner
import check_smoke


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--src', type=Path, required=True)
    ap.add_argument('--weights', type=Path, required=True)
    ap.add_argument('--bench', type=Path, required=True)
    ap.add_argument('--input-root', type=Path, required=True, help='registered DEV smoke inputs (model size)')
    ap.add_argument('--window', required=True)
    ap.add_argument('--snapshot', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--size', default='256x128')
    ap.add_argument('--steps', type=int, default=2)
    a = ap.parse_args()
    rc.guard_names(a.bench, 'dev', [a.window])
    w, h = (int(v) for v in a.size.split('x'))
    if w != 2 * h or w % 32 or (w, h) == (rc.W, rc.H):
        raise ValueError('check-only 2:1 size different from the registered size required')
    a.out.mkdir(parents=True, exist_ok=False)
    import cv2
    import numpy as np
    started = time.monotonic()
    engine = runner.load_official(a.src, a.weights, device='cpu', size_wh=(w, h), num_inference_steps=a.steps)
    load_seconds = time.monotonic() - started
    src_dir = a.input_root / 'windows' / 'dev' / a.window
    tiny_root = a.out / 'inputs'
    tiny_dir = tiny_root / 'windows' / 'dev' / a.window
    inputs = {}
    for kind in ('history', 'condition', 'real'):
        name = bi.clip_name(kind)
        frames = rc.decode(src_dir / name, (rc.W, rc.H))
        small = np.stack([cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA) for f in frames])
        rc.write_video(small, tiny_dir / name, 8)
        inputs[name] = dict(path=str((tiny_dir / name).relative_to(tiny_root)), sha256=rc.sha(tiny_dir / name))
    settings = dict(rc.settings(), size_wh=[w, h], num_inference_steps=a.steps, latent_grid=[5, h // 32, w // 32], check_only=True)
    results = []
    for mode, k in (('continue', 0), ('recon', 'recon')):
        seed = rc.shared_seed(a.window, k, a.snapshot)
        name = a.window + ('-s0' if mode == 'continue' else '-recon')
        spec = dict(schema='wm-calib-bench.reworld_video.spec.v1', model='reworld_video', split='dev', window=a.window, mode=mode,
                    run='reworld-dev-cpucheck', name=name, sample=k, seed=seed, pass_seeds=[seed, seed + rc.B['pass2_seed_offset']],
                    condition_encode_seeds=rc.condition_seeds(a.window, k) if mode == 'continue' else None,
                    prefix_seed=rc.aux_seed(a.window, 'prefix'),
                    inputs={n: v for n, v in inputs.items() if mode == 'recon' or not n.startswith('real')},
                    prompt=None, purpose='cpu-code-path-check', names_list=None, settings=settings,
                    output_dir='reworld-dev-cpucheck/native/' + name)
        specfile = a.out / 'specs' / (name + '.json')
        rc.new_json(specfile, spec)
        t = time.monotonic()
        runner.execute_one(engine, spec, tiny_root, a.out / 'outputs', specfile)
        sample = a.out / 'outputs' / spec['output_dir']
        row = check_smoke.check(sample, expect_settings=False)
        row['wall_seconds_cpu'] = time.monotonic() - t
        native = rc.decode(sample / 'native8.mp4', (w, h))
        cond = rc.decode(tiny_dir / bi.clip_name('condition'), (w, h))
        row['decoded_condition_vs_input_mean_abs'] = float(np.abs(native[8:17].astype(float) - cond).mean())
        results.append(row)
    report = dict(code_path_check='PASS', not_a_sample=True, size_wh=[w, h], steps=a.steps, window=a.window, device='cpu',
                  load_seconds=load_seconds, strict=engine.strict, results=results, software=runner.software(),
                  runner_sha256=rc.sha(Path(runner.__file__)), pins_sha256=rc.sha(rc.PINS_PATH), GPU_started=False)
    rc.new_json(a.out / 'cpu-pipeline-report.json', report)
    print(json.dumps(dict(code_path_check='PASS', samples=len(results), GPU_started=False)))


if __name__ == '__main__':
    main()
