#!/usr/bin/env python3
"""Epona sample specs for bench-v1 (one JSON per sample), with every path relative to an input root.

Input root layout (build_epona_inputs.py):
  windows/<split>/<window>/history.mp4      11 frames, 1024x512, 5 fps (ticks k = -20..0 step 2), lossless
  windows/<split>/<window>/past_poses.json  ego poses at ticks k = -20..0 step 2 (t0 ego frame; past only)
  windows/<split>/<window>/real.mp4         41 frames, 1024x512, 5 fps (k = -20..60 step 2), lossless (real reference only)

Specs written to <out>/<window>/<name>.json (default <input-root>/specs/<split>):
  <window>-s<k>    mode continue, k = 0..15: references history.mp4 and past_poses.json only; seed = seed_of(window, k)
  <window>-recon   mode recon: references real.mp4 only; seed 0 (no randomness is involved)

seed_of() is imported from the read-only code snapshot of make_bench_nano_specs.py (package orbis2-code-snapshot-v1,
unpacked anywhere: --snapshot / $WCB_SNAPSHOT_DIR / <repo>/snapshot) after its sha256 is checked against the pinned value
and the snapshot's MANIFEST.sha256, so every model of the benchmark shares one seed table: the seed of sample k of window w
is the first 4 bytes of sha256("bench-v1|w|k"), big-endian, masked to 31 bits.

  make_epona_specs.py --input-root DIR --split dev|test --names FILE [--samples 0-15] [--no-recon] [--run-prefix epona-dev]
                      [--batch-k N] [--out DIR] [--snapshot DIR]

--batch-k N binds the continuation specs to a batch layout (generation.batch_k = N: the adapter must run them with
--batch-k N, in chunks of N consecutive k per window; bench-v1 TEST and DEV-full use 16 = one window per batch). Specs
without batch_k run one sample per call on the released path. A TEST input root built with --history-only has no
real.mp4: use --no-recon (the TEST real references are made on the research server after freezing).
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
PINS = json.loads((HERE / 'epona_pins.json').read_text())
SNAPSHOT_DEFAULT = REPO / 'snapshot'
NANO_SPECS_SHA256 = '108c4cc974487284182bcc4fc6c84ff12477437059e9a90a50e1fa6fd441968c'  # bench-v1 Nano spec runs' generator_sha256
SCHEMA = 'wm-calib-bench.epona.spec.v1'
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
    """seed_of() from the snapshot of the Nano spec generator (sha256 checked against the pin and MANIFEST.sha256)."""
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
    out = dict(config=d['config'], checkpoint='epona_nuplan+china.pkl', vae_checkpoint='dcae_td_20000.pkl',
               num_sampling_steps=d['config_checked']['num_sampling_steps'], condition_frames=d['condition_frames'],
               frame_rate_hz=d['frame_rate_hz'], future_frames=d['future_frames'], decode_group=d['decode_group'],
               decode_is_video=d['decode_is_video'], self_pred_traj=d['self_pred_traj'], image_size_hw=d['config_checked']['image_size'],
               autocast=d['autocast'], sampler=d['sampler'], pose_rule='test_free.py: get_rel_pose over the 10 condition poses (entry 0 = 0); '
               'entry 10 (the next frame, unused with self_pred_traj) = a copy of entry 9', guidance='none')
    out.update(epona_commit=PINS['code']['commit'], hf_repo=PINS['weights']['hf_repo'], hf_revision=PINS['weights']['hf_revision'],
               output_frames=PINS['output']['frames'], output_fps=PINS['output']['fps'], output_size_wh=PINS['output']['size_wh'],
               output_layout=PINS['output']['layout'], history_ticks=PINS['timing']['history_ticks'],
               model_condition_ticks=PINS['timing']['model_condition_ticks'], pose_ticks=PINS['timing']['pose_ticks'])
    return out


def build(input_root: Path, split: str, names: list[str], samples: list[int], recon: bool, run_prefix: str, out: Path,
          snapshot: Path | None = None, batch_k: int | None = None) -> list[dict]:
    seed_of, nano_sha = load_seed_of(snapshot)
    gen = generation_block()
    bands = {tuple(json.loads((input_root / 'windows' / split / w / 'build.json').read_text())['render'].get('model_band_rows', [797, 1669])) for w in names}
    if len(bands) != 1:
        raise SystemExit(f'the windows were rendered with different model bands: {sorted(bands)}')
    gen['model_band_rows'] = list(bands.pop())
    gen_cont = dict(gen)   # continuation specs: the same block plus the batch layout
    if batch_k:
        if batch_k < 2:
            raise SystemExit('--batch-k must be >= 2 (omit it for the per-sample released path)')
        gen_cont.update(batch_k=batch_k, batch_layout=('one window per batch: k ' + f'{samples[0]}..{samples[-1]}' if batch_k >= len(samples)
                                                       else f'chunks of {batch_k} consecutive k within a window'),
                        batch_noise='row b: torch.Generator().manual_seed(seed of sample b) (step_eval_gen)')
    k_run = f'{run_prefix}-k{len(samples)}' if samples == list(range(len(samples))) else f'{run_prefix}-k{samples[0]}-{samples[-1]}'
    specs = []
    for w in names:
        wdir = Path('windows') / split / w
        hist, poses, real = input_root / wdir / 'history.mp4', input_root / wdir / 'past_poses.json', input_root / wdir / 'real.mp4'
        if recon:
            if not real.is_file():
                raise SystemExit(f'{real} missing')
            name = f'{w}-recon'
            specs.append(dict(schema=SCHEMA, benchmark='bench-v1', model='epona', split=split, window=w, mode='recon', k=None, seed=0,
                              seed_rule='recon: no randomness (fixed 0)', name=name, run=f'{run_prefix}-recon',
                              inputs=dict(real=dict(path=str(wdir / 'real.mp4'), sha256=sha256_file(real), frames=PINS['output']['frames'])),
                              output_dir=f'{run_prefix}-recon/native/{name}', generation=gen))
        if samples:
            for f in (hist, poses):
                if not f.is_file():
                    raise SystemExit(f'{f} missing')
            pp = json.loads(poses.read_text())
            if max(pp['ticks']) > 0:
                raise SystemExit(f'{poses}: holds a pose after t0')
            hist_sha, poses_sha = sha256_file(hist), sha256_file(poses)
            for k in samples:
                name = f'{w}-s{k}'
                specs.append(dict(schema=SCHEMA, benchmark='bench-v1', model='epona', split=split, window=w, mode='continue', k=k,
                                  seed=int(seed_of(w, k)), seed_rule=SEED_RULE, name=name, run=k_run,
                                  inputs=dict(history=dict(path=str(wdir / 'history.mp4'), sha256=hist_sha, frames=len(PINS['timing']['history_ticks'])),
                                              poses=dict(path=str(wdir / 'past_poses.json'), sha256=poses_sha, ticks=pp['ticks'])),
                                  output_dir=f'{k_run}/native/{name}', generation=gen_cont))
    for s in specs:
        d = out / s['window']
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{s['name']}.json").write_text(json.dumps(s, indent=1) + '\n')
    manifest = dict(schema='wm-calib-bench.epona.spec_manifest.v1', created=datetime.now().astimezone().isoformat(timespec='seconds'),
                    split=split, windows=len(names), samples=samples, recon=recon, batch_k=batch_k, runs=sorted({s['run'] for s in specs}), specs=len(specs),
                    generator_sha256=sha256_file(Path(__file__)), pins_sha256=sha256_file(HERE / 'epona_pins.json'),
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
    ap.add_argument('--run-prefix', help='default epona-<split>')
    ap.add_argument('--out', type=Path, help='default <input-root>/specs/<split>')
    ap.add_argument('--snapshot', type=Path, help='unpacked code snapshot (default $WCB_SNAPSHOT_DIR or <repo>/snapshot)')
    ap.add_argument('--batch-k', type=int, default=None, help='bind the continuation specs to batches of N (see above)')
    args = ap.parse_args()
    names = args.names.read_text().split()
    out = args.out or args.input_root / 'specs' / args.split
    if out.exists() and any(out.rglob('*.json')):
        raise SystemExit(f'{out} already holds specs; choose a new directory')
    specs = build(args.input_root, args.split, names, parse_range(args.samples) if args.samples else [], not args.no_recon,
                  args.run_prefix or f'epona-{args.split}', out, args.snapshot, args.batch_k)
    print(json.dumps(dict(split=args.split, windows=len(names), specs=len(specs), out=str(out))))


if __name__ == '__main__':
    main()
