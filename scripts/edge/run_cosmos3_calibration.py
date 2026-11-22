#!/usr/bin/env python3
"""Cosmos3-Edge runner for bench-v1 (2026-11-03): the frozen scripts/run_cosmos3_calibration.py with the checkpoint
swapped, plus a 'wait' mode for the last step of the Edge TEST chain.

Why this file has the frozen runner's name: scripts/edge/run_bench_chain.sh is a byte-identical copy of the frozen
scripts/run_bench_chain.sh (same sha256), which runs `run_cosmos3_calibration.py` from its own directory. Placing this
wrapper next to the copy lets Edge use the frozen chain logic unchanged (PLAN files, `require` lines, resume, the
HOLD_DIR pause/resume and the environment of the inference process).

  --mode specs  calls the frozen run_specs() of scripts/run_cosmos3_calibration.py (imported read-only, never modified)
                after setting that module's CHECKPOINT constant to the pinned Cosmos3-Edge snapshot. Everything else is
                the frozen code path: the same framework (cosmos-framework a36f6b0), the same flags (--parallelism-preset
                latency, --offload-guardrail-models, no torch.compile, no CUDA graphs, no diffusion cache), the spec
                files run as they are, chunks of --chunk samples per inference process, --resume skipping samples whose
                sample_outputs.json says success, the manifest/outputs records.
  --mode wait   blocks until every --wait-for file exists (checked every 60 s) or --wait-hours pass, then exits 0. It is
                the final plan line of the Edge TEST chain, so that the chain, and with it the GPU-hold pause, ends
                only after the blind measurement watchers are done with GPU 0 (on 2026-11-02 the chain resumed the hold
                at plan end and starved the Nano watcher for about 20 min).
GPU memory gate: before each chunk the GPU must show --min-free-mib free (the frozen check). The frozen runner refuses
values below Nano's 37,756 MiB; for Edge the floor is EDGE_PEAK_MIB (measured in the DEV smoke) + 1 GiB, so that the
measurement watchers can share GPU 0 with generation.

  run_cosmos3_calibration.py --mode specs --specs-dir DIR --output DIR [--resume] [--allow-incomplete] [--min-free-mib M]
  run_cosmos3_calibration.py --mode wait --wait-for FILE [FILE ...] --output DIR [--wait-hours H]
"""
from __future__ import annotations

import argparse, hashlib, importlib.util, json, os, time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
FROZEN = HERE.parent / 'run_cosmos3_calibration.py'
EDGE = Path(os.environ.get('DOB_ROOT', '.')) / 'checkpoints/cosmos3-pinned/nvidia/Cosmos3-Edge'
EDGE_REPOSITORY, EDGE_REVISION = 'nvidia/Cosmos3-Edge', '344d602b128d1bbdacb43b08d0a3626f46343e29'
EDGE_PEAK_MIB = 10934         # peak GPU memory of one Edge inference process (nvidia-smi, DEV smoke 2026-11-03: 10,622 continuation, 10,934 recon)
FLOOR_MARGIN_MIB = 1024
HOST_MIN_AVAILABLE_MIB = 12 * 1024   # host RAM: Edge process peak RSS 7.8 GiB (DEV smoke) + the owner's 4 GB margin


def mem_available_mib():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) // 1024
    return 0


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def frozen_runner():
    spec = importlib.util.spec_from_file_location('run_cosmos3_calibration_frozen', FROZEN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def record(args, extra):
    """Edge provenance next to the frozen manifest (written after the frozen code ran; never inside it)."""
    if not args.output.is_dir():
        return
    f = args.output / 'edge-runner.json'
    rec = json.loads(f.read_text()) if f.is_file() else dict(calls=[])
    rec.update(model='Cosmos3-Edge', checkpoint=str(EDGE), repository=EDGE_REPOSITORY, revision=EDGE_REVISION,
               frozen_runner=str(FROZEN), frozen_runner_sha256=sha(FROZEN), wrapper=str(Path(__file__).resolve()),
               wrapper_sha256=sha(Path(__file__).resolve()), edge_peak_mib=EDGE_PEAK_MIB)
    rec['calls'].append(dict(extra, finished_utc=datetime.now(timezone.utc).isoformat()))
    f.write_text(json.dumps(rec, indent=1) + '\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mode', choices=('specs', 'wait'), required=True)
    ap.add_argument('--specs-dir', type=Path); ap.add_argument('--allow-incomplete', action='store_true')
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--chunk', type=int, default=400); ap.add_argument('--min-free-mib', type=int, default=37756)
    ap.add_argument('--wait-hours', type=float, default=12)
    ap.add_argument('--resume', action='store_true'); ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--wait-for', type=Path, nargs='*', default=[], help='--mode wait: files that must all exist')
    args = ap.parse_args()
    if args.mode == 'wait':
        if not args.wait_for:
            raise SystemExit('--mode wait needs --wait-for FILE ...')
        args.output.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        while not all(f.is_file() for f in args.wait_for):
            if time.time() - t0 > args.wait_hours * 3600:
                print(json.dumps(dict(wait='timed_out', hours=args.wait_hours, missing=[str(f) for f in args.wait_for if not f.is_file()])), flush=True)
                break
            time.sleep(60)
        state = 'all_present' if all(f.is_file() for f in args.wait_for) else 'timed_out'
        (args.output / 'wait.json').write_text(json.dumps(dict(state=state, wait_for=[str(f) for f in args.wait_for], waited_s=round(time.time() - t0),
                                                               finished_utc=datetime.now(timezone.utc).isoformat()), indent=1) + '\n')
        print(json.dumps(dict(wait=state, waited_s=round(time.time() - t0))), flush=True)
        return
    gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if gpu not in ('0', '1') or os.environ.get('COSMOS_TRAINING') != '0':
        raise SystemExit('One explicit GPU (0 or 1) and the inference-only environment are required')
    floor = 37756 if EDGE_PEAK_MIB is None else EDGE_PEAK_MIB + FLOOR_MARGIN_MIB
    if args.min_free_mib < floor:
        raise SystemExit(f'--min-free-mib {args.min_free_mib} is below the Edge floor {floor} MiB (peak {EDGE_PEAK_MIB} + {FLOOR_MARGIN_MIB})')
    if args.specs_dir is None:
        raise SystemExit('--mode specs needs --specs-dir')
    if not (EDGE / 'modular_model_index.json').is_file():
        raise SystemExit(f'Edge checkpoint missing: {EDGE}')
    R = frozen_runner()
    R.CHECKPOINT = EDGE                              # the checkpoint: the only change to what is generated
    gpu_free = R.free_mib

    def gated_free(g):
        """Host-memory gate before each model load (owner rule 2026-11-03: keep the peak >= 4 GB below MemAvailable):
        reporting 0 MiB free makes the frozen pre-chunk loop wait (30 s steps, up to --wait-hours) until both the GPU and
        the host have room. It changes when a chunk starts, never what it computes."""
        return 0 if mem_available_mib() < HOST_MIN_AVAILABLE_MIB else gpu_free(g)
    R.free_mib = gated_free
    t0 = time.time()
    try:
        R.run_specs(args, gpu)
    finally:
        record(args, dict(mode='specs', specs_dir=str(args.specs_dir), min_free_mib=args.min_free_mib, chunk=args.chunk, resume=args.resume,
                          wall_s=round(time.time() - t0, 1)))


if __name__ == '__main__':
    main()
