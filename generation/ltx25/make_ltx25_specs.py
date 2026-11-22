#!/usr/bin/env python3
"""LTX-2.5 sample specs for bench-v1 (one JSON per sample), with every path relative to an input root.

Variants (ltx25_pins.json; default: pins.default_variant = distilled):
  distilled  DistilledPipeline, two stages (512 x 288 -> x2 upsampler -> 1024 x 576), output INTER_AREA to the canvas.
             continue specs reference history20.mp4 (stage 1) and history20_1024.mp4 (stage 2); recon specs reference
             real20_1024.mp4. Default run prefix ltx25d-<split>.
  dev        TI2VidOneStagePipeline at 512 x 288 (kept available, not the default). continue: history20.mp4; recon:
             real20.mp4. Default run prefix ltx25-<split>.

Input root layout (build_ltx25_inputs.py): windows/<split>/<window>/{history20,real20}{,_1024}.mp4 (20 fps, lossless).

Specs written to <out>/<window>/<name>.json (default <input-root>/specs/<variant>/<split>/):
  <window>-s<k>    mode continue, k in --samples (default 0..15): references the history clip(s) only; seed = seed_of(window, k)
  <window>-recon   mode recon: references the real clip; seed = seed_of(window, 'recon'), which seeds the DiffVAE decoder
                   (the reference's only randomness; one independent decoder draw per window)
  <window>-recon-d<j>  (--recon-draws N, j = 1..N; DEV diagnostics only) the same real latent decoded with seed_of(window,
                   'recon-d<j>'): the DiffVAE's noise floor on the instrument's readings. Never part of the analysis set
Decode-only DEV diagnostics (mode redecode: saved latents through the DiffVAE or the conv decoder, queue dev-diag) are made
by build_diag_package.py after the DEV runs; their generation block is decode_block() below.

seed_of() is imported from the read-only code snapshot of make_bench_nano_specs.py (package orbis2-code-snapshot-v1,
unpacked anywhere: --snapshot / $WCB_SNAPSHOT_DIR / <repo>/snapshot), after checking its sha256 against the pinned
value and the snapshot's MANIFEST.sha256, so every model of the benchmark shares one seed table: the seed of sample k
of window w is the first 4 bytes of sha256("bench-v1|w|k"), big-endian, masked to 31 bits.

  make_ltx25_specs.py --input-root DIR --split dev|test --names FILE [--variant distilled|dev] [--precision none|fp8-cast]
                      [--samples 0-15] [--no-recon] [--recon-draws N] [--run-prefix PREFIX] [--out DIR] [--snapshot DIR]
--precision sets generation.quantization: the variant's pinned value by default (distilled: none = bf16, the analysis
precision); fp8-cast only for the local mechanics smoke on a 48 GB GPU (its clips never enter an analysis).
A same-seed rerun (determinism check) is a second spec set with another --run-prefix and --out, e.g.
  --names first-window.txt --samples 0 --run-prefix ltx25d-dev-smoke-rerun --out <input-root>/specs/distilled/dev-smoke-rerun
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PINS = json.loads((HERE / 'ltx25_pins.json').read_text())
SNAPSHOT_DEFAULT = REPO / 'snapshot'
NANO_SPECS_SHA256 = '108c4cc974487284182bcc4fc6c84ff12477437059e9a90a50e1fa6fd441968c'  # bench-v1 Nano spec runs' generator_sha256
SCHEMA = 'wm-calib-bench.ltx25.spec.v2'
SEED_RULE = 'make_bench_nano_specs.seed_of: int.from_bytes(sha256(f"bench-v1|{window}|{k}").digest()[:4], "big") & 0x7FFFFFFF'
RECON_SEED_RULE = "seed_of(window, 'recon'): the same hash with k = 'recon'; seeds the DiffVAE decoder of the real reference"
PREFIX = {'distilled': 'ltx25d', 'dev': 'ltx25'}
INPUTS = {  # role -> (file name, frames)
    'distilled': dict(continue_={'history': ('history20.mp4', 41), 'history_hr': ('history20_1024.mp4', 41)}, recon={'real_hr': ('real20_1024.mp4', 161)}),
    'dev': dict(continue_={'history': ('history20.mp4', 41)}, recon={'real': ('real20.mp4', 161)}),
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot_dir(arg: Path | None = None) -> Path:
    d = Path(arg) if arg else Path(os.environ['WCB_SNAPSHOT_DIR']) if os.environ.get('WCB_SNAPSHOT_DIR') else SNAPSHOT_DEFAULT
    if not (d / 'scripts' / 'make_bench_nano_specs.py').is_file():
        raise SystemExit(f'code snapshot not found at {d}: unpack orbis2-code-snapshot-v1.tar (it creates snapshot/) and pass '
                         '--snapshot DIR or set WCB_SNAPSHOT_DIR')
    return d


def load_seed_of(snapshot: Path | None = None):
    """seed_of() from the snapshot of the Nano spec generator, after its sha256 is checked against the pinned value and
    the snapshot's MANIFEST.sha256."""
    d = snapshot_dir(snapshot)
    nano = d / 'scripts' / 'make_bench_nano_specs.py'
    listed = None
    for line in (d / 'MANIFEST.sha256').read_text().splitlines():
        digest, _, rel = line.partition('  ')
        if rel.strip() == 'scripts/make_bench_nano_specs.py':
            listed = digest
    got = sha256_file(nano)
    if got != NANO_SPECS_SHA256 or listed != NANO_SPECS_SHA256:
        raise SystemExit(f'{nano}: sha256 {got} (manifest {listed}), pinned {NANO_SPECS_SHA256}')
    spec = importlib.util.spec_from_file_location('make_bench_nano_specs_snapshot', nano)
    mod = importlib.util.module_from_spec(spec)
    keep, sys.dont_write_bytecode = sys.dont_write_bytecode, True  # no __pycache__ inside the read-only snapshot
    try:
        spec.loader.exec_module(mod)  # defines constants and functions only; no I/O at import
    finally:
        sys.dont_write_bytecode = keep
    if mod.PROMPTS['long'] != PINS['bench']['prompt']:
        raise SystemExit('the pinned LTX-2.5 prompt is not the Nano spec generator\'s prompt "long"')
    return mod.seed_of, got


def parse_range(text: str) -> list[int]:
    lo, _, hi = text.partition('-')
    return list(range(int(lo), int(hi or lo) + 1))


def generation_block(variant: str, quantization: str | None = None) -> dict:
    v, b = PINS['variants'][variant], PINS['bench']
    quantization = quantization or v['quantization']
    if quantization not in v['allowed_quantization']:
        raise SystemExit(f'quantization {quantization} is not allowed for the {variant} variant {v["allowed_quantization"]}')
    out = dict(variant=variant)
    if variant == 'distilled':
        out.update({k: v[k] for k in ('transformer', 'spatial_upsampler', 'final_size_wh', 'guidance', 'denoiser_calls', 'enhance_prompt', 'dtype',
                                      'quantization', 'diffvae_optimization', 'decode_size_wh', 'decode_free_bytes')})
        out['stage_1'] = {k: v['stage_1'][k] for k in ('size_wh', 'sigmas', 'steps', 'noise_scale', 'ancestral_noise_seed_offset')}
        out['stage_2'] = {k: v['stage_2'][k] for k in ('size_wh', 'sigmas', 'steps', 'ancestral_noise_seed_offset')}
        out['sampler'] = {k: v['sampler'][k] for k in ('eta', 's_noise')} | dict(loop='euler_ancestral_denoising_loop')
        out['output_downsize'] = 'RGB24 truncation, then cv2.INTER_AREA 1024x576 -> 512x288'
        out['quantization'] = quantization
    else:
        out.update({k: v[k] for k in ('transformer', 'num_inference_steps', 'video_guider', 'audio_guider', 'scheduler', 'sampler', 'negative_prompt_sha256',
                                      'enhance_prompt', 'dtype', 'max_batch_size', 'diffvae_optimization', 'decode_size_wh', 'decode_free_bytes')})
        out['quantization'] = quantization
    out.update({k: b[k] for k in ('fps', 'frames', 'history_frames', 'history_latent_frames', 'future_frames', 'canvas_wh', 'prompt_sha256')})
    out.update(ltx2_commit=PINS['code']['commit'], hf_repo=PINS['weights']['hf_repo'], hf_revision=PINS['weights']['hf_revision'],
               output_files=sorted(k for k in PINS['output']['files'] if k != 'full1024.mp4' or variant == 'distilled'))
    return out


def decode_block(variant: str, decoder: str) -> dict:
    """The generation block of a redecode spec (decode-only DEV diagnostics: a saved final latent through one of the pinned video
    decoders with the call the generation runs use; build_diag_package.py)."""
    v, b = PINS['variants'][variant], PINS['bench']
    decoders = v.get('diag', {}).get('decoders', {})
    if decoder not in decoders:
        raise SystemExit(f'no {decoder} decoder pinned for the {variant} variant')
    out = dict(variant=variant, kind='redecode', decoder=decoder, checkpoint=decoders[decoder], dtype=v['dtype'], decode_size_wh=v['decode_size_wh'],
               decode_call=v['diag']['decode_call'], output_downsize='RGB24 truncation, then cv2.INTER_AREA 1024x576 -> 512x288')
    if decoder == 'diffvae':
        out.update(diffvae_optimization=v['diffvae_optimization'], decode_free_bytes=v['decode_free_bytes'])
    out.update({k: b[k] for k in ('fps', 'frames', 'canvas_wh')})
    out.update(ltx2_commit=PINS['code']['commit'], hf_repo=PINS['weights']['hf_repo'], hf_revision=PINS['weights']['hf_revision'])
    return out


def build(input_root: Path, split: str, names: list[str], samples: list[int], recon: bool, run_prefix: str, out: Path,
          snapshot: Path | None = None, recon_draws: int = 0, variant: str = PINS['default_variant'], quantization: str | None = None) -> list[dict]:
    if recon_draws and split != 'dev':
        raise SystemExit('--recon-draws is a DEV diagnostic')
    seed_of, nano_sha = load_seed_of(snapshot)
    gen = generation_block(variant, quantization)
    k_run = f'{run_prefix}-k{len(samples)}' if samples == list(range(len(samples))) else f'{run_prefix}-k{samples[0]}-{samples[-1]}'

    def inputs(wdir: Path, which: dict) -> dict:
        rec = {}
        for role, (fname, frames) in which.items():
            p = input_root / wdir / fname
            if not p.is_file():
                raise SystemExit(f'{p} missing' + (' (TEST real clips exist only after the protocol is frozen)' if fname.startswith('real') else ''))
            rec[role] = dict(path=str(wdir / fname), sha256=sha256_file(p), frames=frames)
        return rec

    specs = []
    for w in names:
        wdir = Path('windows') / split / w
        base = dict(schema=SCHEMA, benchmark='bench-v1', model='ltx25d' if variant == 'distilled' else 'ltx25', variant=variant, split=split, window=w)
        if recon or recon_draws:
            rin = inputs(wdir, INPUTS[variant]['recon'])
        if recon:
            name = f'{w}-recon'
            specs.append(dict(base, mode='recon', k=None, seed=int(seed_of(w, 'recon')), seed_rule=RECON_SEED_RULE, name=name, run=f'{run_prefix}-recon',
                              inputs=rin, output_dir=f'{run_prefix}-recon/native/{name}', generation=gen))
        for j in range(1, recon_draws + 1):
            name = f'{w}-recon-d{j}'
            specs.append(dict(base, mode='recon', k=None, seed=int(seed_of(w, f'recon-d{j}')),
                              seed_rule=f"seed_of(window, 'recon-d{j}'): an extra DiffVAE decoder draw (DEV diagnostic of the decoder noise floor; not part of the analysis set)",
                              name=name, run=f'{run_prefix}-recon-draws', inputs=rin, output_dir=f'{run_prefix}-recon-draws/native/{name}', generation=gen))
        if samples:
            cin = inputs(wdir, INPUTS[variant]['continue_'])
            for k in samples:
                name = f'{w}-s{k}'
                specs.append(dict(base, mode='continue', k=k, seed=int(seed_of(w, k)), seed_rule=SEED_RULE, name=name, run=k_run, inputs=cin,
                                  output_dir=f'{k_run}/native/{name}', generation=gen))
    seeds = [s['seed'] for s in specs]
    if len(set(seeds)) != len(seeds):
        raise SystemExit('two specs share a seed')
    for s in specs:
        d = out / s['window']
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{s['name']}.json").write_text(json.dumps(s, indent=1) + '\n')
    me = Path(__file__).read_bytes()
    manifest = dict(schema='wm-calib-bench.ltx25.spec_manifest.v2', created=datetime.now().astimezone().isoformat(timespec='seconds'), variant=variant,
                    quantization=gen['quantization'],
                    split=split, windows=len(names), samples=samples, recon=recon, recon_draws=recon_draws, runs=sorted({s['run'] for s in specs}),
                    specs=len(specs), generator_sha256=hashlib.sha256(me).hexdigest(), pins_sha256=sha256_file(HERE / 'ltx25_pins.json'),
                    seed_source='orbis2-code-snapshot-v1: snapshot/scripts/make_bench_nano_specs.py', seed_source_sha256=nano_sha,
                    seed_table_sha256=hashlib.sha256(json.dumps([[s['name'], s['seed']] for s in specs]).encode()).hexdigest())
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    return specs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input-root', type=Path, required=True)
    ap.add_argument('--split', required=True, choices=['dev', 'test'])
    ap.add_argument('--names', type=Path, required=True, help='window names, one per line')
    ap.add_argument('--variant', choices=sorted(PINS['variants']), default=PINS['default_variant'])
    ap.add_argument('--precision', choices=('none', 'fp8-cast'), help="generation.quantization (default: the variant's pinned value; fp8-cast only for the local smoke)")
    ap.add_argument('--samples', default='0-15', help="sample indices k ('' for none)")
    ap.add_argument('--no-recon', action='store_true')
    ap.add_argument('--recon-draws', type=int, default=0, help='DEV only: extra decoder draws of the real reference per window')
    ap.add_argument('--run-prefix', help='default ltx25d-<split> (distilled) or ltx25-<split> (dev)')
    ap.add_argument('--out', type=Path, help='default <input-root>/specs/<variant>/<split>')
    ap.add_argument('--snapshot', type=Path, help='unpacked code snapshot (default $WCB_SNAPSHOT_DIR or <repo>/snapshot)')
    args = ap.parse_args()
    names = args.names.read_text().split()
    out = args.out or args.input_root / 'specs' / args.variant / args.split
    if out.exists() and any(out.rglob('*.json')):
        raise SystemExit(f'{out} already holds specs; choose a new directory')
    specs = build(args.input_root, args.split, names, parse_range(args.samples) if args.samples else [], not args.no_recon,
                  args.run_prefix or f'{PREFIX[args.variant]}-{args.split}', out, args.snapshot, args.recon_draws, args.variant, args.precision)
    print(json.dumps(dict(variant=args.variant, quantization=specs[0]['generation']['quantization'] if specs else None, split=args.split,
                          windows=len(names), specs=len(specs), out=str(out))))


if __name__ == '__main__':
    main()
