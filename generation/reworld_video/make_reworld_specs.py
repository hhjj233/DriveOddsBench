#!/usr/bin/env python3
"""Pinned K=16 ReWorld specs with the shared seed table (video only; no prompt).

Modes: continue (two official 33-frame calls, pass-2 seed = seed + 1000003), recon (the real reference through the
same VAE, seeded seed_of(window, 'recon')), prefix (the 17-frame codec prefix alone). Adapted from
make_drivelaw_specs.py; TEST reference/prefix specs need the stage-2 measurement freeze.
Purposes: main (registered windows), smoke (the 2 registered smoke windows, s0/s1 + references, run prefix
reworld-dev-smoke-) and timescale-diagnostic (ruling D9: the DriveLaW predeclared moving subset, s0/s1, run prefix
reworld-dev-tsdiag-, built against the authorised diagnostic bench view; never part of the main DEV analysis).
"""
from pathlib import Path
import argparse
import json
import sys
sys.dont_write_bytecode = True
import reworld_common as rc
import build_reworld_inputs as bi


def parse_samples(text):
    result = []
    for item in text.split(','):
        lo, _, hi = item.partition('-')
        result.extend(range(int(lo), int(hi or lo) + 1))
    if len(result) != len(set(result)) or any(k not in range(16) for k in result):
        raise ValueError('unique samples 0..15 required')
    return result


def code_provenance():
    return dict(pins_sha256=rc.sha(rc.PINS_PATH), spec_builder_sha256=rc.sha(Path(__file__)),
                common_sha256=rc.sha(rc.HERE / 'reworld_common.py'), runner_sha256=rc.sha(rc.HERE / 'run_reworld_bench.py'),
                input_builder_sha256=rc.sha(rc.HERE / 'build_reworld_inputs.py'))


def build(bench, input_root, split, windows, mode, run, samples, snapshot, out, frozen_file=None, purpose='main'):
    rc.guard_names(bench, split, windows)
    settings = rc.settings()
    if not run.startswith('reworld-' + split + '-') or mode not in ('continue', 'recon', 'prefix'):
        raise ValueError('registered run/split/mode required')
    keys_all = samples if mode == 'continue' else [mode]
    for w in windows:
        for k in keys_all:
            rc.check_purpose(purpose, split, run, mode, w, k)
    listed = rc.names_list(bench, split)
    rc.check_names_list(purpose, split, listed)
    if split == 'test' and mode != 'continue':
        rc.verify_freeze(frozen_file or '', required=[rc.PINS_PATH, Path(__file__)])
    written = []
    for w in windows:
        root = Path(input_root) / 'windows' / split / w
        record = json.loads((root / 'inputs.json').read_text())
        if record['window'] != w or record['split'] != split:
            raise ValueError('input split mismatch')
        if (record['pins_sha256'] != rc.sha(rc.PINS_PATH) or record['generator_sha256'] != rc.sha(rc.HERE / 'build_reworld_inputs.py')
                or record['common_sha256'] != rc.sha(rc.HERE / 'reworld_common.py')):
            raise ValueError('input build provenance changed; prepare a new candidate input directory')
        if record['names_list'] != listed:
            raise ValueError('inputs were built against a different window list (bench view)')
        needed = [bi.clip_name('history'), bi.clip_name('condition')]
        if mode == 'recon':
            needed.append(bi.clip_name('real'))
        inputs = {}
        for name in needed:
            path = root / name
            if rc.sha(path) != record['files'][name]['sha256']:
                raise ValueError('input digest changed')
            inputs[name] = dict(path=str(path.relative_to(input_root)), sha256=rc.sha(path))
        keys = samples if mode == 'continue' else [mode]
        for k in keys:
            seed = rc.shared_seed(w, k, snapshot)
            name = w + ('-s' + str(k) if mode == 'continue' else '-' + mode)
            spec = dict(schema='wm-calib-bench.reworld_video.spec.v1', model='reworld_video', split=split, window=w,
                        mode=mode, run=run, name=name, sample=k, seed=seed,
                        pass_seeds=[seed, seed + rc.B['pass2_seed_offset']],
                        condition_encode_seeds=rc.condition_seeds(w, k) if mode == 'continue' else None,
                        prefix_seed=rc.aux_seed(w, 'prefix'),
                        inputs=inputs, prompt=None, conditioning='video only: last 9 real frames (t0-1..t0); no text, actions or poses',
                        purpose=purpose, names_list=listed,
                        purpose_note=rc.B['timescale_diagnostic']['rule'] if purpose == 'timescale-diagnostic' else None,
                        settings=settings, output_dir=run + '/native/' + name,
                        keep=['prefix-native8', 'pass1-native8', 'pass2-condition8', 'pass2-native8', 'native8', 'vision'],
                        generator=code_provenance())
            if split == 'test' and mode != 'continue':
                spec['measurement_freeze'] = rc.verify_freeze(frozen_file)
            path = Path(out) / run / (name + '.json')
            rc.new_json(path, spec)
            written.append(str(path))
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bench', type=Path, required=True)
    ap.add_argument('--input-root', type=Path, required=True)
    ap.add_argument('--split', choices=['dev', 'test'], required=True)
    ap.add_argument('--names', type=Path, required=True)
    ap.add_argument('--mode', choices=['continue', 'recon', 'prefix'], required=True)
    ap.add_argument('--run', required=True)
    ap.add_argument('--samples', default='0-15')
    ap.add_argument('--snapshot', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--frozen-file', type=Path)
    ap.add_argument('--purpose', choices=rc.PURPOSES, default='main')
    a = ap.parse_args()
    if a.split == 'test' and a.mode != 'continue':
        rc.verify_freeze(a.frozen_file or '', required=[rc.PINS_PATH, Path(__file__)])
    files = build(a.bench, a.input_root, a.split, rc.names(a.names), a.mode, a.run, parse_samples(a.samples), a.snapshot, a.out,
                  a.frozen_file, a.purpose)
    print(json.dumps(dict(specs=len(files), mode=a.mode, split=a.split, purpose=a.purpose)))


if __name__ == '__main__':
    main()
