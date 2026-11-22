#!/usr/bin/env python3
"""Vista sample specs for bench-v1 (one JSON per sample), every path relative to an input root (build_vista_inputs.py).

Input root layout: windows/<split>/<window>/{history21_<tag>.mp4, real81_<tag>.mp4, actions_t2.json}; <tag> = 1024 at the
generation size `native` (576 x 1024, the size of the TEST run of the paper, package vista-inputs-v4-native; pass --size native),
576 at `320x576` (the pins default) and 512 at `canvas` (288 x 512).

Specs written to <out>/<window>/<run>__<name>.json (default <input-root>/specs/<split>/):
  <window>-prefix   mode prefix (one per window; run <prefix>-prefix): the VAE round trip of the real history frames
                    k = -20..-1 (history21_<tag>.mp4 frames 0-19), seed_of(window, 'prefix'); frames 0-19 of every clip
  <window>-recon    mode recon (run <prefix>-recon): the real frames k = 0..60 (real81_<tag>.mp4 frames 20-80) through the
                    same encode and decode as a rollout's latents, seed_of(window, 'recon'); frames 20-80 of the reference
  <window>-s<k>     mode continue, track h0 (run <prefix>-h0-k16) and/or t2 (run <prefix>-t2-k16), k in --samples: the anchor
                    (history21_<tag>.mp4 frame 20) only, plus for t2 the window's actions_t2.json; seed = seed_of(window, k),
                    the same for both tracks (h0 and t2 differ only by the action)
Every continue / recon spec names its window's prefix spec (`prefix`), whose frames the adapter puts in front.

seed_of() is imported from the read-only code snapshot of make_bench_nano_specs.py (package orbis2-code-snapshot-v1,
unpacked anywhere: --snapshot / $WCB_SNAPSHOT_DIR / <repo>/snapshot), after checking its sha256 against the pinned value
and the snapshot's MANIFEST.sha256: the seed of sample k of window w is the first 4 bytes of sha256("bench-v1|w|k"),
big-endian, masked to 31 bits, one table for every model of the benchmark.

  make_vista_specs.py --input-root DIR --split dev|test --names FILE [--tracks h0,t2] [--samples 0-15] [--no-recon]
                      [--no-prefix] [--run-prefix PREFIX] [--steps N] [--action-variant frame_index|time_aligned]
                      [--layout official|local48] [--size canvas|native] [--out DIR] [--snapshot DIR]
--steps overrides the released 50 sampling steps (only for a mechanics check; such specs are marked and never analysed).
A same-seed rerun (determinism check) is a second spec set with another --run-prefix and --out.
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
PINS_PATH = HERE / 'vista_pins.json'
PINS = json.loads(PINS_PATH.read_text())
SNAPSHOT_DEFAULT = REPO / 'snapshot'
NANO_SPECS_SHA256 = '108c4cc974487284182bcc4fc6c84ff12477437059e9a90a50e1fa6fd441968c'  # bench-v1 Nano spec runs' generator_sha256
SCHEMA = 'wm-calib-bench.vista.spec.v1'
SEED_RULE = 'make_bench_nano_specs.seed_of: int.from_bytes(sha256(f"bench-v1|{window}|{k}").digest()[:4], "big") & 0x7FFFFFFF'


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
    return mod.seed_of, got


def parse_range(text: str) -> list[int]:
    lo, _, hi = text.partition('-')
    return list(range(int(lo), int(hi or lo) + 1))


def generation_block(steps: int | None = None, layout: str | None = None, size: str | None = None) -> dict:
    d, b = PINS['released_defaults'], PINS['bench']
    layout = layout or b['default_layout']
    size = size or b['default_size']
    lay = b['layouts'][layout]
    h, w = b['sizes'][size]['hw']
    out = dict(size=size, height=h, width=w, n_frames=d['n_frames'], n_conds=d['n_conds'], n_rounds=b['n_rounds'], n_steps=d['n_steps'],
               cfg_scale=d['cfg_scale'], cond_aug=d['cond_aug'], guider='TrianglePredictionGuider' if b['n_rounds'] > 1 else 'VanillaCFG',
               sampler='EulerEDMSampler', discretization='EDMDiscretization', layout=layout, low_vram=lay['low_vram'], rollout_latents=b['rollout_latents'],
               decoded_latents=b['decoded_latents'], frames=b['frames'], fps=b['fps'], canvas_wh=b['canvas_wh'], model_wh=[w, h],
               vista_commit=PINS['code']['commit'], hf_repo=PINS['weights']['hf_repo'], hf_revision=PINS['weights']['hf_revision'],
               weights_sha256=PINS['weights']['files']['vista.safetensors']['sha256'], mechanics_only=False, cond_encode_chunk=lay['cond_encode_chunk'])
    if steps is not None and steps != d['n_steps']:
        out.update(n_steps=int(steps), mechanics_only=True)
    return out


def build(input_root: Path, split: str, names: list[str], samples: list[int], tracks: list[str], recon: bool, prefix: bool, run_prefix: str,
          out: Path, snapshot: Path | None = None, steps: int | None = None, action_variant: str = 'frame_index',
          layout: str | None = None, size: str | None = None) -> list[dict]:
    seed_of, nano_sha = load_seed_of(snapshot)
    gen = generation_block(steps, layout, size)
    tag = PINS['bench']['sizes'][gen['size']]['input_tag']
    hist_name, real_name = f'history21_{tag}.mp4', f'real81_{tag}.mp4'
    gen_t2 = dict(gen, action_variant=action_variant)
    if action_variant == 'time_aligned':
        if split != 'dev':
            raise SystemExit('the time-aligned trajectory variant is a DEV diagnostic')
        gen_t2['diagnostic_only'] = True
    k_tag = f'k{len(samples)}' if samples == list(range(len(samples))) else f'k{samples[0]}-{samples[-1]}'

    def rec(wdir: Path, fname: str, **extra) -> dict:
        p = input_root / wdir / fname
        if not p.is_file():
            raise SystemExit(f'{p} missing' + (' (TEST future inputs exist only after the protocol is frozen)' if fname != hist_name else ''))
        return dict(path=str(wdir / fname), sha256=sha256_file(p), **extra)

    specs = []
    for w in names:
        wdir = Path('windows') / split / w
        base = dict(schema=SCHEMA, benchmark='bench-v1', model='vista', split=split, window=w)
        pname = f'{w}-prefix'
        prun = f'{run_prefix}-prefix'
        pref = dict(name=pname, run=prun, output_dir=f'{prun}/native/{pname}', seed=int(seed_of(w, 'prefix')))
        hist = rec(wdir, hist_name, frames=21)
        if prefix:
            specs.append(dict(base, mode='prefix', track=None, k=None, seed=pref['seed'],
                              seed_rule="seed_of(window, 'prefix'): seeds the VAE posterior sample of the history round trip", name=pname, run=prun,
                              inputs=dict(history=dict(hist, use_frames=list(range(20)))), output_dir=pref['output_dir'], generation=gen))
        if recon:
            name, run = f'{w}-recon', f'{run_prefix}-recon'
            specs.append(dict(base, mode='recon', track=None, k=None, seed=int(seed_of(w, 'recon')),
                              seed_rule="seed_of(window, 'recon'): seeds the VAE posterior sample of the real frames k = 0..60", name=name, run=run,
                              inputs=dict(real=rec(wdir, real_name, frames=81, use_frames=list(range(20, 81)))), prefix=pref,
                              output_dir=f'{run}/native/{name}', generation=gen))
        for track in tracks:
            ta = track == 't2' and action_variant == 'time_aligned'
            run = f'{run_prefix}-{track}{"ta" if ta else ""}-{k_tag}'
            inputs = dict(anchor=dict(hist, use_frames=[20]))
            if track == 't2':
                inputs['actions'] = rec(wdir, 'actions_t2_timealigned.json' if ta else 'actions_t2.json', mode='trajectory')
            for k in samples:
                name = f'{w}-s{k}'
                specs.append(dict(base, mode='continue', track=track, k=k, seed=int(seed_of(w, k)), seed_rule=SEED_RULE, name=name, run=run,
                                  inputs=inputs, prefix=pref, output_dir=f'{run}/native/{name}', generation=gen_t2 if track == 't2' else gen))
    keys = [(s['run'], s['name']) for s in specs]
    if len(set(keys)) != len(keys):
        raise SystemExit('two specs share a run and a name')
    for run in {s['run'] for s in specs if s['mode'] == 'continue'}:  # within a run every sample has its own seed
        seeds = [s['seed'] for s in specs if s['run'] == run]
        if len(set(seeds)) != len(seeds):
            raise SystemExit(f'two {track} specs share a seed')
    for s in specs:
        d = out / s['window']
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{s['run']}__{s['name']}.json").write_text(json.dumps(s, indent=1) + '\n')
    manifest = dict(schema='wm-calib-bench.vista.spec_manifest.v1', created=datetime.now().astimezone().isoformat(timespec='seconds'), split=split,
                    windows=len(names), samples=samples, tracks=tracks, recon=recon, prefix=prefix, runs=sorted({s['run'] for s in specs}), specs=len(specs),
                    mechanics_only=gen['mechanics_only'], n_steps=gen['n_steps'], action_variant=action_variant, layout=gen['layout'], size=gen['size'], generator_sha256=sha256_file(Path(__file__)), pins_sha256=sha256_file(PINS_PATH),
                    seed_source='orbis2-code-snapshot-v1: snapshot/scripts/make_bench_nano_specs.py', seed_source_sha256=nano_sha,
                    seed_table_sha256=hashlib.sha256(json.dumps([[s['run'], s['name'], s['seed']] for s in specs]).encode()).hexdigest())
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    return specs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input-root', type=Path, required=True)
    ap.add_argument('--split', choices=('dev', 'test'), required=True)
    ap.add_argument('--names', type=Path, required=True)
    ap.add_argument('--tracks', default='h0,t2')
    ap.add_argument('--samples', default='0-15')
    ap.add_argument('--no-recon', action='store_true')
    ap.add_argument('--no-prefix', action='store_true')
    ap.add_argument('--run-prefix', default=None)
    ap.add_argument('--steps', type=int, default=None)
    ap.add_argument('--action-variant', choices=('frame_index', 'time_aligned'), default='frame_index')
    ap.add_argument('--layout', choices=tuple(PINS['bench']['layouts']), default=PINS['bench']['default_layout'],
                    help='memory layout (pins bench.layouts): official = the released setting (default); local48 = --low_vram and '
                         'the cond-frame encoder one copy at a time, for 48 GB mechanics at the native size only')
    ap.add_argument('--size', choices=tuple(PINS['bench']['sizes']), default=PINS['bench']['default_size'],
                    help='generation size (pins bench.sizes): native = 576 x 1024, the TEST run of the paper; 320x576 = the pins default; canvas = 288 x 512')
    ap.add_argument('--out', type=Path, default=None)
    ap.add_argument('--snapshot', type=Path, default=None)
    args = ap.parse_args()
    names = [l.strip() for l in args.names.read_text().splitlines() if l.strip()]
    tracks = [t for t in args.tracks.split(',') if t]
    if not set(tracks) <= {'h0', 't2'}:
        raise SystemExit('--tracks: h0 and/or t2')
    samples = parse_range(args.samples) if args.samples else []
    specs = build(args.input_root, args.split, names, samples, tracks, recon=not args.no_recon, prefix=not args.no_prefix,
                  run_prefix=args.run_prefix or f'vista-{args.split}', out=args.out or args.input_root / 'specs' / args.split,
                  snapshot=args.snapshot, steps=args.steps, action_variant=args.action_variant, layout=args.layout, size=args.size)
    print(json.dumps(dict(specs=len(specs), runs=sorted({s['run'] for s in specs}))))


if __name__ == '__main__':
    main()
