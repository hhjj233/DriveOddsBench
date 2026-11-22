#!/usr/bin/env python3
"""Cosmos3-Edge sample specs for bench-v1 (2026-11-03): a thin wrapper around the frozen make_bench_nano_specs.py.

The frozen spec maker writes to <bench>/specs/<run>; the Edge specs go to <bench>/edge/specs/<run>. Everything that
defines a sample comes from the frozen module, imported read-only and unchanged:
  - the defaults files (DEFAULTS: study-v1's abl-h5 / recon-hist5 defaults; 35 steps, guidance 6, shift 10, sigma_max 80,
    the default negative prompt; recon 2 steps), copied byte for byte;
  - the prompt (PROMPTS['long'], the benchmark's fixed generic description), HISTORY_LATENTS = 6, TOTAL_FRAMES = 81,
    FPS = 10, resolution tier '256' with aspect ratio '16,9' (320x192), video2video, condition_video_keep 'first';
  - seed_of(window, k) (sha256("bench-v1|w|k"), first 4 bytes, 31 bits) and parse_range;
  - the windows: <bench>/windows-<split>/<name>/nano/{history,real}.mp4, the same files as Nano's specs.
The checkpoint is not a spec field: the runner passes it to the framework (--checkpoint-path; see
scripts/edge/run_cosmos3_calibration.py), and this script records it in the specs manifest. So an Edge spec equals the
Nano spec of the same name in every field except defaults_file, which points to a byte-identical copy in the Edge run
directory. `verify` checks exactly that against existing Nano spec directories.

  make_bench_edge_specs.py make   --bench DIR --split dev|test --mode continue|recon --run NAME [--names FILE] [--samples 0-15]
  make_bench_edge_specs.py verify --edge EDGE_SPECS_RUN_DIR --nano NANO_SPECS_RUN_DIR [NANO_SPECS_RUN_DIR ...] [--out F]
"""
from __future__ import annotations
import os

import argparse, hashlib, importlib.util, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FROZEN = HERE / 'make_bench_nano_specs.py'
CHECKPOINT = dict(repository='nvidia/Cosmos3-Edge', revision='344d602b128d1bbdacb43b08d0a3626f46343e29',
                  local_dir=os.path.join(os.environ.get('DOB_ROOT', '.'), 'checkpoints/cosmos3-pinned/nvidia/Cosmos3-Edge'))


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def frozen_module():
    spec = importlib.util.spec_from_file_location('make_bench_nano_specs_frozen', FROZEN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cmd_make(args):
    nano = frozen_module()
    names = (args.names or args.bench / f'names-{args.split}.txt').read_text().split()
    windows = args.bench / f'windows-{args.split}'
    out = args.bench / 'edge' / 'specs' / args.run
    if out.exists():
        raise SystemExit(f'{out} exists; choose a new run name')
    (out / 'specs').mkdir(parents=True)
    source = nano.DEFAULTS[args.mode]
    (out / 'defaults.json').write_bytes(source.read_bytes())          # byte-identical copy (no overrides offered)
    defaults = json.loads(source.read_text())
    # the same field order and values as make_bench_nano_specs.py main()
    common = dict(defaults_file=str(out / 'defaults.json'), model_mode='video2video', condition_video_keep='first', num_frames=nano.TOTAL_FRAMES,
                  fps=nano.FPS, resolution='256', aspect_ratio='16,9', prompt=nano.PROMPTS['long'], enable_sound=False)
    specs = []
    for name in names:
        d = windows / name / 'nano'
        if args.mode == 'continue':
            if not (d / 'history.mp4').is_file():
                raise SystemExit(f'{d}/history.mp4 missing')
            for k in nano.parse_range(args.samples):
                specs.append(dict(name=f'{name}-s{k}', vision_path=str(d / 'history.mp4'), condition_frame_indexes_vision=list(range(nano.HISTORY_LATENTS)),
                                  seed=nano.seed_of(name, k), **common))
        else:
            if not (d / 'real.mp4').is_file():
                raise SystemExit(f'{d}/real.mp4 missing')
            specs.append(dict(name=f'{name}-recon', vision_path=str(d / 'real.mp4'), condition_frame_indexes_vision=list(range((nano.TOTAL_FRAMES - 1) // 4 + 1)),
                              seed=0, **common))
    for spec in specs:
        (out / 'specs' / f"{spec['name']}.json").write_text(json.dumps(spec, indent=1) + '\n')
    (out / 'manifest.json').write_text(json.dumps(dict(
        model='Cosmos3-Edge', checkpoint=CHECKPOINT, split=args.split, mode=args.mode, prompt='long',
        samples=args.samples if args.mode == 'continue' else None, windows=len(names), specs=len(specs),
        names_file=str(args.names or args.bench / f'names-{args.split}.txt'), names_sha256=sha(args.names or args.bench / f'names-{args.split}.txt'),
        defaults_source=str(source), defaults_source_sha256=sha(source), num_steps=defaults['num_steps'], guidance=defaults['guidance'],
        shift=defaults.get('shift'), wrapper_sha256=sha(Path(__file__)), frozen_spec_maker=str(FROZEN), frozen_spec_maker_sha256=sha(FROZEN)), indent=1) + '\n')
    print(json.dumps(dict(run=args.run, specs=len(specs), out=str(out))))


def cmd_verify(args):
    """Every Edge spec vs the Nano spec of the same name: all fields equal except defaults_file, and the two defaults
    files byte-identical."""
    nano_index = {}
    for d in args.nano:
        for f in sorted((d / 'specs').glob('*.json')):
            nano_index.setdefault(f.stem, f)
    res = dict(edge=str(args.edge), nano=[str(d) for d in args.nano], specs=0, matched=0, identical_except_defaults_file=0,
               defaults_identical=0, unmatched=[], differing=[])
    for f in sorted((args.edge / 'specs').glob('*.json')):
        res['specs'] += 1
        e = json.loads(f.read_text())
        g = nano_index.get(f.stem)
        if g is None:
            res['unmatched'].append(f.stem); continue
        n = json.loads(g.read_text()); res['matched'] += 1
        keys = sorted(set(e) | set(n))
        diff = [k for k in keys if e.get(k) != n.get(k)]
        if diff == ['defaults_file'] or not diff:
            res['identical_except_defaults_file'] += 1
        else:
            res['differing'].append(dict(name=f.stem, fields=diff))
        if sha(Path(e['defaults_file'])) == sha(Path(n['defaults_file'])):
            res['defaults_identical'] += 1
        elif len(res['differing']) < 20:
            res['differing'].append(dict(name=f.stem, fields=['defaults_file content']))
    res['ok'] = res['matched'] == res['specs'] and res['identical_except_defaults_file'] == res['specs'] and res['defaults_identical'] == res['specs']
    text = json.dumps(res, indent=1)
    if args.out:
        args.out.write_text(text + '\n')
    print(json.dumps({k: res[k] for k in ('specs', 'matched', 'identical_except_defaults_file', 'defaults_identical', 'ok')}))
    if not res['ok']:
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('make')
    p.add_argument('--bench', required=True, type=Path); p.add_argument('--split', required=True, choices=['dev', 'test'])
    p.add_argument('--mode', required=True, choices=['continue', 'recon']); p.add_argument('--run', required=True)
    p.add_argument('--names', type=Path, help='subset of window names (default: names-<split>.txt)')
    p.add_argument('--samples', default='0-15', help='sample indices k for --mode continue')
    p = sub.add_parser('verify')
    p.add_argument('--edge', required=True, type=Path); p.add_argument('--nano', required=True, type=Path, nargs='+'); p.add_argument('--out', type=Path)
    args = ap.parse_args()
    {'make': cmd_make, 'verify': cmd_verify}[args.cmd](args)


if __name__ == '__main__':
    main()
