#!/usr/bin/env python3
"""Pinned VaViM specs: K continuation samples per window (shared seed table) and the real reference.

  make_vavim_specs.py --bench BENCH --input-root INPUTS --split dev|test --names FILE --mode continue|recon
                      --run vavim-<split>-<name> [--samples 0-15] [--purpose main|smoke|diagnostic]
                      --snapshot SEED_SNAPSHOT --out SPEC_DIR [--frozen-file RECORD]

A continuation spec reads only history10.mp4 and context2.mp4 (the model reads the context frames only; history10 gives
the shared codec prefix); a reference spec also reads real10.mp4 (DEV, or TEST after the stage-2 freeze). Each spec
carries its seed seed_of(window, k), the registered settings and the hashes of the kit files that will run it; the
runner refuses any drift. Purposes: main (registered windows), smoke (the registered smoke windows and samples, run prefix
vavim-dev-smoke-) and diagnostic (DEV only: any names-dev.txt window, run prefix vavim-dev-diag-; never part of the main
DEV analysis).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import vavim_common as vc  # noqa: E402

KIT_FILES = ['vavim_common.py', 'build_vavim_inputs.py', 'make_vavim_specs.py', 'run_vavim_bench.py']


def kit_hashes():
    return dict(pins_sha256=vc.sha(vc.PINS_PATH), **{name: vc.sha(vc.HERE / name) for name in KIT_FILES})


def parse_samples(text):
    result = []
    for item in text.split(','):
        lo, _, hi = item.partition('-')
        result.extend(range(int(lo), int(hi or lo) + 1))
    if len(result) != len(set(result)) or any(k not in range(16) for k in result):
        raise ValueError('unique samples 0..15 required')
    return result


def build(bench, input_root, split, windows, mode, run, samples, snapshot, out, frozen_file=None, purpose='main'):
    diagnostic = purpose == 'diagnostic'
    vc.guard_names(bench, split, windows, diagnostic)
    if mode not in ('continue', 'recon'):
        raise ValueError('registered mode required')
    for w in windows:
        for k in (samples if mode == 'continue' else [None]):
            vc.check_purpose(purpose, split, run, mode, w, k)
    if split == 'test' and mode != 'continue':
        if not frozen_file or Path(frozen_file).name != vc.B['test_measurement_freeze']:
            raise ValueError('TEST reference specs need the stage-2 freeze record')
        vc.verify_freeze(frozen_file, required=[vc.PINS_PATH, Path(__file__)])
    settings = vc.settings()
    if settings['top_k'] is None:
        raise ValueError('sampling top_k is not registered yet (vavim_pins.json bench.sampling.top_k)')
    hashes = kit_hashes()
    listed_name = 'names-dev.txt' if diagnostic else 'names-' + split + '-gen.txt'
    listed = dict(file=listed_name, sha256=vc.sha(Path(bench) / listed_name))
    written = []
    for w in windows:
        root = Path(input_root) / 'windows' / split / w
        record = json.loads((root / 'inputs.json').read_text())
        if record['window'] != w or record['split'] != split:
            raise ValueError('input split mismatch')
        if (record['pins_sha256'] != hashes['pins_sha256'] or record['builder_sha256'] != hashes['build_vavim_inputs.py']
                or record['common_sha256'] != hashes['vavim_common.py']):
            raise ValueError('input build provenance changed; prepare a new input directory')
        if bool(record.get('diagnostic')) != diagnostic or record.get('names_list') != listed:
            raise ValueError('inputs were built for another purpose or window list')
        if record['context_ticks'] != vc.CONTEXT_K:
            raise ValueError('inputs were built for other context ticks')
        needed = ['history10.mp4', 'context2.mp4'] + (['real10.mp4'] if mode == 'recon' else [])
        inputs = {}
        for name in needed:
            path = root / name
            if name not in record['files'] or vc.sha(path) != record['files'][name]['sha256']:
                raise ValueError('input digest changed: ' + name)
            inputs[name] = dict(path=str(path.relative_to(input_root)), sha256=record['files'][name]['sha256'])
        inputs['inputs.json'] = dict(path=str((root / 'inputs.json').relative_to(input_root)), sha256=vc.sha(root / 'inputs.json'))
        keys = samples if mode == 'continue' else ['recon']
        for k in keys:
            name = w + ('-s' + str(k) if mode == 'continue' else '-recon')
            seed = vc.shared_seed(w, k, snapshot) if mode == 'continue' else None
            spec = dict(schema='wm-calib-bench.vavim.spec.v1', model='vavim', split=split, window=w, purpose=purpose,
                        diagnostic=diagnostic, mode=mode, run=run, name=name, sample=k if mode == 'continue' else None, seed=seed,
                        seed_rule=vc.B['seed_rule'] if mode == 'continue' else 'none: deterministic VQ round trip',
                        inputs=inputs, names_list=listed, settings=settings, output_dir=run + '/native/' + name,
                        keep=['vision.mp4', 'native2.mp4', 'tokens.npz', 'sample_outputs.json'], generator=hashes)
            if split == 'test' and mode != 'continue':
                spec['measurement_freeze'] = vc.verify_freeze(frozen_file)
            path = Path(out) / run / (name + '.json')
            vc.new_json(path, spec)
            written.append(str(path))
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bench', type=Path, required=True)
    ap.add_argument('--input-root', type=Path, required=True)
    ap.add_argument('--split', choices=['dev', 'test'], required=True)
    ap.add_argument('--names', type=Path, required=True)
    ap.add_argument('--mode', choices=['continue', 'recon'], required=True)
    ap.add_argument('--run', required=True)
    ap.add_argument('--samples', default='0-15')
    ap.add_argument('--purpose', choices=['main', 'smoke', 'diagnostic'], default='main')
    ap.add_argument('--snapshot', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--frozen-file', type=Path)
    a = ap.parse_args()
    files = build(a.bench, a.input_root, a.split, vc.names(a.names), a.mode, a.run, parse_samples(a.samples),
                  a.snapshot, a.out, a.frozen_file, a.purpose)
    print(json.dumps(dict(specs=len(files), mode=a.mode, split=a.split, run=a.run, purpose=a.purpose)))


if __name__ == '__main__':
    main()
