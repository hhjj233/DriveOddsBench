#!/usr/bin/env python3
"""Orbis 2 sample specs for bench-v1 (one JSON per sample), with every path relative to an input root.

Input root layout (the cluster input package, see build_input_package.py):
  windows/<split>/<window>/history.mp4   21 frames, 512x288, 10 fps (t0 - 2.0 .. t0), lossless
  windows/<split>/<window>/real.mp4      81 frames, 512x288, 10 fps (t0 - 2.0 .. t0 + 6.0), lossless (real reference only)

Specs written to <input-root>/specs/<split>/<window>/<name>.json:
  <window>-s<k>    mode continue, k = 0..15: references the history clip only; seed = seed_of(window, k)
  <window>-recon   mode recon: references the real clip; seed 0 (no randomness is involved)

seed_of() is imported from the read-only code snapshot of make_bench_nano_specs.py (the Cosmos3-Nano spec generator;
package orbis2-code-snapshot-v1, unpacked anywhere: --snapshot / $WCB_SNAPSHOT_DIR / <repo>/snapshot), after checking its
sha256 against the pinned value and the snapshot's MANIFEST.sha256, so every model of the benchmark shares one seed table:
the seed of sample k of window w is the first 4 bytes of sha256("bench-v1|w|k"), big-endian, masked to 31 bits.

  make_orbis2_specs.py --input-root DIR --split dev|test --names FILE [--samples 0-15] [--no-recon] [--run-prefix orbis2-dev]
                       [--snapshot DIR]
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
PINS = json.loads((HERE / 'orbis2_pins.json').read_text())
# The code snapshot is not part of the repository: it is the separate package orbis2-code-snapshot-v1, whose tar
# unpacks to snapshot/. Its location: --snapshot, else $WCB_SNAPSHOT_DIR, else <repo>/snapshot (git-ignored).
SNAPSHOT_DEFAULT = REPO / 'snapshot'
NANO_SPECS_SHA256 = '108c4cc974487284182bcc4fc6c84ff12477437059e9a90a50e1fa6fd441968c'  # bench-v1 Nano spec runs' generator_sha256
SCHEMA = 'wm-calib-bench.orbis2.spec.v1'
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


def generation_block() -> dict:
    d = PINS['released_defaults']
    keep = ('config', 'ckpt', 'l1_sampler', 'l1_nfe_cli', 'l2_sampler', 'l2_pred_nfe', 'eta', 'guidance', 'use_ema', 'autocast', 'steering',
            'num_rollout_steps', 'num_pred_frames', 'image_size_hw', 'l1_frame_rate', 'l2_frame_rate', 'history_frames', 'l1_context_indices',
            'l2_context_indices', 'decode_device', 'compile')
    out = {k: d[k] for k in keep}
    out.update(orbis2_commit=PINS['code']['commit'], hf_repo=PINS['weights']['hf_repo'], hf_revision=PINS['weights']['hf_revision'],
               output_frames=PINS['output']['frames'], output_fps=PINS['output']['fps'], output_layout=PINS['output']['layout'])
    return out


def build(input_root: Path, split: str, names: list[str], samples: list[int], recon: bool, run_prefix: str, out: Path,
          snapshot: Path | None = None) -> list[dict]:
    seed_of, nano_sha = load_seed_of(snapshot)
    gen = generation_block()
    k_run = f'{run_prefix}-k{len(samples)}' if samples == list(range(len(samples))) else f'{run_prefix}-k{samples[0]}-{samples[-1]}'
    specs = []
    for w in names:
        wdir = Path('windows') / split / w
        hist, real = input_root / wdir / 'history.mp4', input_root / wdir / 'real.mp4'
        if recon:
            if not real.is_file():
                raise SystemExit(f'{real} missing')
            name = f'{w}-recon'
            specs.append(dict(schema=SCHEMA, benchmark='bench-v1', model='orbis2', split=split, window=w, mode='recon', k=None, seed=0,
                              seed_rule='recon: no randomness (fixed 0)', name=name, run=f'{run_prefix}-recon',
                              inputs=dict(real=dict(path=str(wdir / 'real.mp4'), sha256=sha256_file(real), frames=PINS['output']['frames'])),
                              output_dir=f'{run_prefix}-recon/native/{name}', generation=gen))
        if samples:
            if not hist.is_file():
                raise SystemExit(f'{hist} missing')
            hist_sha = sha256_file(hist)
            for k in samples:
                name = f'{w}-s{k}'
                specs.append(dict(schema=SCHEMA, benchmark='bench-v1', model='orbis2', split=split, window=w, mode='continue', k=k,
                                  seed=int(seed_of(w, k)), seed_rule=SEED_RULE, name=name, run=k_run,
                                  inputs=dict(history=dict(path=str(wdir / 'history.mp4'), sha256=hist_sha,
                                                           frames=PINS['released_defaults']['history_frames'])),
                                  output_dir=f'{k_run}/native/{name}', generation=gen))
    for s in specs:
        d = out / s['window']
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{s['name']}.json").write_text(json.dumps(s, indent=1) + '\n')
    me = Path(__file__).read_bytes()
    manifest = dict(schema='wm-calib-bench.orbis2.spec_manifest.v1', created=datetime.now().astimezone().isoformat(timespec='seconds'),
                    split=split, windows=len(names), samples=samples, recon=recon, runs=sorted({s['run'] for s in specs}), specs=len(specs),
                    generator_sha256=hashlib.sha256(me).hexdigest(), pins_sha256=sha256_file(HERE / 'orbis2_pins.json'),
                    seed_source='orbis2-code-snapshot-v1: snapshot/scripts/make_bench_nano_specs.py', seed_source_sha256=nano_sha,
                    seed_table_sha256=hashlib.sha256(json.dumps([[s['name'], s['seed']] for s in specs]).encode()).hexdigest())
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    return specs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input-root', type=Path, required=True)
    ap.add_argument('--split', required=True, choices=['dev', 'test'])
    ap.add_argument('--names', type=Path, required=True, help='window names, one per line')
    ap.add_argument('--samples', default='0-15', help="sample indices k ('' for none)")
    ap.add_argument('--no-recon', action='store_true')
    ap.add_argument('--run-prefix', help='default orbis2-<split>')
    ap.add_argument('--out', type=Path, help='default <input-root>/specs/<split>')
    ap.add_argument('--snapshot', type=Path, help='unpacked code snapshot (default $WCB_SNAPSHOT_DIR or <repo>/snapshot)')
    args = ap.parse_args()
    names = args.names.read_text().split()
    out = args.out or args.input_root / 'specs' / args.split
    if out.exists() and any(out.rglob('*.json')):
        raise SystemExit(f'{out} already holds specs; choose a new directory')
    specs = build(args.input_root, args.split, names, parse_range(args.samples) if args.samples else [], not args.no_recon,
                  args.run_prefix or f'orbis2-{args.split}', out, args.snapshot)
    print(json.dumps(dict(split=args.split, windows=len(names), specs=len(specs), out=str(out))))


if __name__ == '__main__':
    main()
