#!/usr/bin/env python3
"""Cosmos3 samples for the world-model calibration study (2026-10-29): continuations of real windows, or the real clips
passed through the same tokenizer.

  continue     video2video without a control video: the first --cond-latents latent frames of history.mp4 (6 = the 21
               history frames) condition an 81-frame clip; one sample per seed.
  reconstruct  real.mp4 with all 21 latent frames conditioned. Conditioned latents carry sigma = 0, so the output is the
               real clip encoded and decoded by the same tokenizer as every generated clip, and the real and generated
               futures are measured with the same degradation. One sample per window; --steps can be small.
  anchor-i2v   (protocol-v1.2) image2video from <windows>/<name>/anchor.png: 61 frames from the anchor; one sample per seed.
  reconstruct-anchor  (protocol-v1.2) <windows>/<name>/real-anchor.mp4 (61 frames from the anchor) with all 16 latents
               conditioned: the real reference of anchor-i2v, its latent grid starting at the anchor.
  specs        prebuilt spec files from <--specs-dir>/specs/*.json, run as they are (e.g. the forward-dynamics / wam /
               inverse-dynamics sets of make_fd_calibration_specs.py). A sample counts as done when its
               sample_outputs.json reports success, with or without a video (inverse dynamics writes none).

Sampling defaults are the framework's image2video defaults (35 UniPC steps, guidance 6.0, default negative prompt), as in
the smoke test; --guidance and --steps override. Samples run in chunks, each a fresh inference process that loads the
model once. Before each chunk the GPU must show --min-free-mib free (waits, then fails after --wait-hours). Finished
samples are skipped when the same output is run again with --resume. Run in the inference environment
(run_cosmos3_calibration.sh or run_wm_calibration_chain.sh).

  run_cosmos3_calibration.py --mode continue --windows DIR --names-file F --seeds 0-15 --output DIR [--guidance G]
"""
from __future__ import annotations

import argparse, hashlib, json, os, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

SOURCE = Path(os.environ.get('DOB_ROOT', '.')) / 'references/backend-route-20261010/cosmos-framework'
CHECKPOINT = Path(os.environ.get('DOB_ROOT', '.')) / 'checkpoints/cosmos3-pinned/nvidia/Cosmos3-Nano'
DEFAULTS = SOURCE / 'cosmos_framework/inference/defaults'
PROMPT = 'A realistic front-facing driving camera video on a city street, continuing the given footage.'  # as the smoke test
FLAGS = ['--parallelism-preset', 'latency', '--offload-guardrail-models', '--no-use-torch-compile', '--no-use-cuda-graphs', '--no-diffusion-cache']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seeds_of(text):
    out = []
    for part in text.split(','):
        a, _, b = part.partition('-')
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


def free_mib(gpu):
    return int(subprocess.check_output(['nvidia-smi', f'--id={gpu}', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())


def done(native, name):
    f = native / name / 'sample_outputs.json'
    return (native / name / 'vision.mp4').is_file() and f.is_file() and json.loads(f.read_text()).get('status') == 'success'


def run_specs(args, gpu):
    """--mode specs: run prebuilt spec files unchanged, chunked and resumable like the other modes."""
    src = sorted((args.specs_dir / 'specs').glob('*.json'))
    if not src:
        raise SystemExit(f'no specs in {args.specs_dir}/specs')
    digest = hashlib.sha256(b''.join(f.read_bytes() for f in src)).hexdigest()
    settings = dict(mode='specs', specs_dir=str(args.specs_dir), specs=[f.name for f in src], specs_sha256=digest)
    record_path = args.output / 'manifest.json'
    if args.output.exists():
        if not args.resume:
            raise SystemExit('Fresh output required (or --resume)')
        record = json.loads(record_path.read_text())
        if record['settings'] != settings:
            raise SystemExit('--resume with different settings')
    else:
        args.output.mkdir(parents=True); (args.output / 'specs').mkdir()
        record = dict(schema='cosmos3_calibration_run_v1', created_utc=datetime.now(timezone.utc).isoformat(), checkpoint=str(CHECKPOINT), physical_GPU=int(gpu), settings=settings,
                      control_video=None, chunks=[], status='running')
    files = []
    for f in src:
        dst = args.output / 'specs' / f.name
        if not dst.exists():
            dst.write_bytes(f.read_bytes())
        files.append((json.loads(dst.read_text())['name'], dst))
    native = args.output / 'native'
    finished = lambda n: (native / n / 'sample_outputs.json').is_file() and json.loads((native / n / 'sample_outputs.json').read_text()).get('status') == 'success'
    todo = [(n, p) for n, p in files if not finished(n)]
    print(json.dumps(dict(samples=len(files), to_run=len(todo))), flush=True)
    record_path.write_text(json.dumps(record, indent=1) + '\n')
    if args.dry_run:
        return
    for start in range(0, len(todo), args.chunk):
        part = todo[start:start + args.chunk]; waited = time.time()
        while free_mib(gpu) < args.min_free_mib:
            if time.time() - waited > args.wait_hours * 3600:
                record['status'] = 'waiting_failed'; record_path.write_text(json.dumps(record, indent=1) + '\n')
                raise SystemExit(f'GPU {gpu}: less than {args.min_free_mib} MiB free for {args.wait_hours} h')
            time.sleep(30)
        t0 = time.time()
        code = subprocess.call([sys.executable, '-m', 'cosmos_framework.scripts.inference', '--checkpoint-path', str(CHECKPOINT), *FLAGS, '-i', *[str(p) for _, p in part], '-o', str(native)],
                               cwd=SOURCE)
        record['chunks'].append(dict(first=part[0][0], last=part[-1][0], samples=len(part), returncode=code, waited_s=round(t0 - waited, 1), wall_s=round(time.time() - t0, 1),
                                     finished_utc=datetime.now(timezone.utc).isoformat()))
        record_path.write_text(json.dumps(record, indent=1) + '\n')
        if code != 0:
            break
    outputs = {n: dict(video=str(native / n / 'vision.mp4') if (native / n / 'vision.mp4').is_file() else None,
                       sha256=sha(native / n / 'vision.mp4') if (native / n / 'vision.mp4').is_file() else None) for n, _ in files if finished(n)}
    record.update(status='complete' if len(outputs) == len(files) else 'incomplete', outputs=len(outputs), missing=[n for n, _ in files if n not in outputs],
                  finished_utc=datetime.now(timezone.utc).isoformat())
    (args.output / 'outputs.json').write_text(json.dumps(outputs, indent=1) + '\n'); record_path.write_text(json.dumps(record, indent=1) + '\n')
    print(json.dumps({k: record[k] for k in ('status', 'outputs')}), flush=True)
    if record['status'] != 'complete' and not args.allow_incomplete:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mode', choices=('continue', 'reconstruct', 'anchor-i2v', 'reconstruct-anchor', 'specs'), required=True)
    parser.add_argument('--windows', type=Path); parser.add_argument('--names-file', type=Path, help='one window name per line (default: all in --windows)')
    parser.add_argument('--specs-dir', type=Path, help='with --mode specs: a directory holding specs/*.json')
    parser.add_argument('--allow-incomplete', action='store_true', help='with --mode specs: record failed samples and exit 0 (for pilots that must not stop a chain)')
    parser.add_argument('--output', required=True, type=Path); parser.add_argument('--seeds', default='0')
    parser.add_argument('--guidance', type=float); parser.add_argument('--steps', type=int); parser.add_argument('--cond-latents', type=int, default=6)
    parser.add_argument('--chunk', type=int, default=400); parser.add_argument('--min-free-mib', type=int, default=37756); parser.add_argument('--wait-hours', type=float, default=12)
    parser.add_argument('--resume', action='store_true', help='continue an existing output with the same settings')
    parser.add_argument('--dry-run', action='store_true', help='write the specs and the manifest, run nothing')
    args = parser.parse_args()
    gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if gpu not in ('0', '1') or os.environ.get('COSMOS_TRAINING') != '0':
        raise SystemExit('One explicit GPU (0 or 1) and the inference-only environment are required')
    if args.min_free_mib < 37756:
        raise SystemExit('Cosmos3-Nano peaked at 36756 MiB; keep at least 1 GiB margin')
    if args.mode == 'specs':
        return run_specs(args, gpu)
    if args.windows is None:
        raise SystemExit('--windows is required for this mode')
    names = [l.strip() for l in args.names_file.read_text().splitlines() if l.strip()] if args.names_file else sorted(p.name for p in args.windows.iterdir() if (p / 'history.mp4').is_file() or (p / 'anchor.png').is_file())
    seeds = [0] if args.mode in ('reconstruct', 'reconstruct-anchor') else seeds_of(args.seeds)
    defaults = json.loads((DEFAULTS / 'image2video/sample_args.json').read_text())
    defaults['negative_prompt'] = json.dumps(json.loads((DEFAULTS / defaults.pop('negative_prompt_file')).read_text()))  # as the framework's own defaults loader does
    if args.guidance is not None:
        defaults['guidance'] = args.guidance
    if args.steps is not None:
        defaults['num_steps'] = args.steps
    cond_latents = {'continue': args.cond_latents, 'reconstruct': 21, 'anchor-i2v': 1, 'reconstruct-anchor': 16}[args.mode]
    settings = dict(mode=args.mode, windows=str(args.windows), names=names, seeds=seeds, defaults=defaults, prompt=PROMPT, cond_latents=cond_latents)
    record_path = args.output / 'manifest.json'
    if args.output.exists():
        if not args.resume:
            raise SystemExit('Fresh output required (or --resume)')
        record = json.loads(record_path.read_text())
        if record['settings'] != settings:
            raise SystemExit('--resume with different settings')
    else:
        args.output.mkdir(parents=True); (args.output / 'specs').mkdir()
        record = dict(schema='cosmos3_calibration_run_v1', created_utc=datetime.now(timezone.utc).isoformat(), checkpoint=str(CHECKPOINT), physical_GPU=int(gpu), settings=settings,
                      control_video=None, chunks=[], status='running')
    (args.output / 'defaults.json').write_text(json.dumps(defaults, indent=1) + '\n')
    native = args.output / 'native'; files = []
    for name in names:
        for seed in seeds:
            sample = f'{name}-s{seed}' if args.mode in ('continue', 'anchor-i2v') else f'{name}-recon'
            if args.mode == 'anchor-i2v':
                spec = dict(name=sample, model_mode='image2video', defaults_file=str(args.output / 'defaults.json'), vision_path=str(args.windows / name / 'anchor.png'),
                            condition_frame_indexes_vision=[0], num_frames=61, fps=10, resolution='256', aspect_ratio='3,4', seed=seed, prompt=PROMPT, enable_sound=False)
            elif args.mode == 'reconstruct-anchor':
                spec = dict(name=sample, model_mode='video2video', defaults_file=str(args.output / 'defaults.json'), vision_path=str(args.windows / name / 'real-anchor.mp4'),
                            condition_frame_indexes_vision=list(range(16)), condition_video_keep='first', num_frames=61, fps=10, resolution='256', aspect_ratio='3,4',
                            seed=0, prompt=PROMPT, enable_sound=False)
            elif args.mode == 'continue':
                spec = dict(name=sample, model_mode='video2video', defaults_file=str(args.output / 'defaults.json'), vision_path=str(args.windows / name / 'history.mp4'),
                            condition_frame_indexes_vision=list(range(args.cond_latents)), condition_video_keep='first', num_frames=81, fps=10, resolution='256', aspect_ratio='3,4',
                            seed=seed, prompt=PROMPT, enable_sound=False)
            else:
                spec = dict(name=sample, model_mode='video2video', defaults_file=str(args.output / 'defaults.json'), vision_path=str(args.windows / name / 'real.mp4'),
                            condition_frame_indexes_vision=list(range(21)), condition_video_keep='first', num_frames=81, fps=10, resolution='256', aspect_ratio='3,4',
                            seed=0, prompt=PROMPT, enable_sound=False)
            path = args.output / 'specs' / f'{sample}.json'
            path.write_text(json.dumps(spec, indent=1) + '\n'); files.append((sample, path))
    todo = [(n, p) for n, p in files if not done(native, n)]
    print(json.dumps(dict(samples=len(files), to_run=len(todo))), flush=True)
    record_path.write_text(json.dumps(record, indent=1) + '\n')
    if args.dry_run:
        return
    for start in range(0, len(todo), args.chunk):
        part = todo[start:start + args.chunk]; waited = time.time()
        while free_mib(gpu) < args.min_free_mib:
            if time.time() - waited > args.wait_hours * 3600:
                record['status'] = 'waiting_failed'; record_path.write_text(json.dumps(record, indent=1) + '\n')
                raise SystemExit(f'GPU {gpu}: less than {args.min_free_mib} MiB free for {args.wait_hours} h')
            time.sleep(30)
        t0 = time.time()
        code = subprocess.call([sys.executable, '-m', 'cosmos_framework.scripts.inference', '--checkpoint-path', str(CHECKPOINT), *FLAGS, '-i', *[str(p) for _, p in part], '-o', str(native)],
                               cwd=SOURCE)
        record['chunks'].append(dict(first=part[0][0], last=part[-1][0], samples=len(part), returncode=code, waited_s=round(t0 - waited, 1), wall_s=round(time.time() - t0, 1),
                                     finished_utc=datetime.now(timezone.utc).isoformat()))
        record_path.write_text(json.dumps(record, indent=1) + '\n')
        if code != 0:
            break
    outputs = {n: dict(video=str(native / n / 'vision.mp4'), sha256=sha(native / n / 'vision.mp4')) for n, _ in files if done(native, n)}
    record.update(status='complete' if len(outputs) == len(files) else 'incomplete', outputs=len(outputs), missing=[n for n, _ in files if n not in outputs],
                  finished_utc=datetime.now(timezone.utc).isoformat())
    (args.output / 'outputs.json').write_text(json.dumps(outputs, indent=1) + '\n'); record_path.write_text(json.dumps(record, indent=1) + '\n')
    print(json.dumps({k: record[k] for k in ('status', 'outputs')}), flush=True)
    if record['status'] != 'complete':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
