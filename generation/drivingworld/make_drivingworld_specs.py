#!/usr/bin/env python3
"""Pinned DrivingWorld specs: K continuation samples per window (shared seed table) and the real reference.

  make_drivingworld_specs.py --bench BENCH --input-root INPUTS --split dev|test --names FILE --mode continue|recon
                             --run drivingworld-<split>-<name> [--samples 0-15] --snapshot SEED_SNAPSHOT --out SPEC_DIR
                             [--frozen-file RECORD]

A continuation spec reads only history10.mp4 and past_poses.json; a reference spec also reads real10.mp4 (DEV, or TEST
after the stage-2 freeze). Each spec carries its seed seed_of(window, k), the registered settings and the hashes of the
kit files that will run it; the runner refuses any drift.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import drivingworld_common as dc  # noqa: E402

KIT_FILES = ['drivingworld_common.py', 'drivingworld_rollout.py', 'build_drivingworld_inputs.py',
             'make_drivingworld_specs.py', 'run_drivingworld_bench.py']


def kit_hashes():
    return dict(pins_sha256=dc.sha(dc.PINS_PATH), **{name: dc.sha(dc.HERE / name) for name in KIT_FILES})


def parse_samples(text):
    result = []
    for item in text.split(','):
        lo, _, hi = item.partition('-')
        result.extend(range(int(lo), int(hi or lo) + 1))
    if len(result) != len(set(result)) or any(k not in range(16) for k in result):
        raise ValueError('unique samples 0..15 required')
    return result


def build(bench, input_root, split, windows, mode, run, samples, snapshot, out, frozen_file=None, diagnostic=False):
    dc.guard_names(bench, split, windows, diagnostic)
    settings = dc.settings()
    if not run.startswith('drivingworld-' + split + '-') or mode not in ('continue', 'recon'):
        raise ValueError('registered run/split/mode required')
    if diagnostic != run.startswith('drivingworld-dev-diag-'):
        raise ValueError('diagnostic runs (DEV only) are named drivingworld-dev-diag-*')
    if split == 'test' and mode != 'continue':
        if not frozen_file or Path(frozen_file).name != dc.B['test_measurement_freeze']:
            raise ValueError('TEST reference specs need the stage-2 freeze record')
        dc.verify_freeze(frozen_file, required=[dc.PINS_PATH, Path(__file__)])
    hashes = kit_hashes()
    written = []
    for w in windows:
        root = Path(input_root) / 'windows' / split / w
        record = json.loads((root / 'inputs.json').read_text())
        if record['window'] != w or record['split'] != split:
            raise ValueError('input split mismatch')
        if (record['pins_sha256'] != hashes['pins_sha256'] or record['builder_sha256'] != hashes['build_drivingworld_inputs.py']
                or record['common_sha256'] != hashes['drivingworld_common.py']):
            raise ValueError('input build provenance changed; prepare a new input directory')
        if bool(record.get('diagnostic')) != bool(diagnostic):
            raise ValueError('diagnostic inputs and specs must match')
        if record['band']['name'] != settings['band']:
            raise ValueError('inputs were built for another model band')
        needed = ['history10.mp4', 'past_poses.json'] + (['real10.mp4'] if mode == 'recon' else [])
        inputs = {}
        for name in needed:
            path = root / name
            if name not in record['files'] or dc.sha(path) != record['files'][name]['sha256']:
                raise ValueError('input digest changed: ' + name)
            inputs[name] = dict(path=str(path.relative_to(input_root)), sha256=record['files'][name]['sha256'])
        inputs['inputs.json'] = dict(path=str((root / 'inputs.json').relative_to(input_root)), sha256=dc.sha(root / 'inputs.json'))
        keys = samples if mode == 'continue' else ['recon']
        for k in keys:
            name = w + ('-s' + str(k) if mode == 'continue' else '-recon')
            seed = dc.shared_seed(w, k, snapshot) if mode == 'continue' else None
            spec = dict(schema='wm-calib-bench.drivingworld.spec.v1', model='drivingworld', split=split, window=w, diagnostic=bool(diagnostic),
                        mode=mode, run=run, name=name, sample=k if mode == 'continue' else None, seed=seed,
                        seed_rule=dc.B['seed_rule'] if mode == 'continue' else 'none: VQ argmin round trip',
                        inputs=inputs, settings=settings, band=record['band'], output_dir=run + '/native/' + name,
                        keep=['vision.mp4', 'native10.mp4', 'tokens.npz', 'pred_poses.json'] if mode == 'continue'
                        else ['vision.mp4', 'native10.mp4', 'tokens.npz'], generator=hashes)
            if split == 'test' and mode != 'continue':
                spec['measurement_freeze'] = dc.verify_freeze(frozen_file)
            path = Path(out) / run / (name + '.json')
            dc.new_json(path, spec)
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
    ap.add_argument('--snapshot', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--frozen-file', type=Path)
    ap.add_argument('--diagnostic', action='store_true', help='DEV only: windows of names-dev.txt, run drivingworld-dev-diag-*')
    a = ap.parse_args()
    files = build(a.bench, a.input_root, a.split, dc.names(a.names), a.mode, a.run, parse_samples(a.samples),
                  a.snapshot, a.out, a.frozen_file, a.diagnostic)
    print(json.dumps(dict(specs=len(files), mode=a.mode, split=a.split, run=a.run)))


if __name__ == '__main__':
    main()
