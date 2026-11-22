#!/usr/bin/env python3
"""Trajectory-level calibration v1, extension 1 (PLAN-EXT-1.md): the frozen v1 and addendum-1 scripts on nine models.
  run_ext1.py --bench BENCH_V1_ROOT --out RUN_DIR
Refuses to run unless frozen-ext-1.txt lists this script and PLAN-EXT-1.md; the frozen scripts' own guards run too."""
import hashlib, importlib.util, json, math, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; V1 = HERE.parent / 'trajectory-calibration-v1'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(name, path):
    s = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


def new(root, pre):
    O = f'{root}/stage2-p84'
    return (f'{O}/analysis-test-v1', [f'{O}/outcomes/test/outcomes-{pre}-test-k16.csv'], [f'{O}/measure-test/measure-{pre}-test-k16.json'],
            f'{O}/measure-test/measure-{pre}-test-recon.json', [f'{O}/test-instrument/ego-vo/clips/{pre}-test-k16'],
            f'{O}/test-instrument/ego-vo/clips/{pre}-test-recon')


NEW = {'DriveLaW-Video': new('drivelaw-video', 'drivelaw'), 'DrivingWorld': new('drivingworld', 'drivingworld'),
       'ReWorld-Video': new('reworld-video', 'reworld')}


def main():
    pins = dict(reversed(l.split()[:2]) for l in (HERE / 'frozen-ext-1.txt').read_text().splitlines() if l and not l.startswith('#'))
    for n in ('PLAN-EXT-1.md', 'run_ext1.py'):
        if pins.get(n) != sha(HERE / n): sys.exit(f'{n} changed since the freeze')
    a = dict(zip(sys.argv[1::2], sys.argv[2::2])); bench, out = Path(a['--bench']), Path(a['--out'])
    at = load('at', V1 / 'analyze_trajectories.py'); cl = load('cl', V1 / 'analyze_trajectory_climatology.py')
    for mod in (at, cl.at): mod.MODELS.update(NEW)
    sys.argv = ['analyze_trajectories.py', '--bench', str(bench), '--out', str(out), '--models', ','.join(at.MODELS)]; at.main()
    sys.argv = ['analyze_trajectory_climatology.py', '--bench', str(bench), '--out', str(out)]; cl.main()
    old = V1 / 'run-20261110-v1'; R, R0 = json.loads((out / 'results.json').read_text()), json.loads((old / 'results.json').read_text())
    C, C0 = json.loads((out / 'addendum-1-climatology.json').read_text()), json.loads((old / 'addendum-1-climatology.json').read_text())
    strip = lambda r: {f: {k: v for k, v in o.items() if not k.endswith('_q')} for f, o in r['result']['families'].items()}
    bad = [m for m in R0['models'] if strip(R['models'][m]) != strip(R0['models'][m]) or R['models'][m]['result']['pooled'] != R0['models'][m]['result']['pooled']
           or C['models'][m] != C0['models'][m]]
    rec = dict(by='coordinator p-inter-84', plan_ext_sha256=sha(HERE / 'PLAN-EXT-1.md'), script_sha256=sha(__file__), new_models=NEW,
               earlier_six_reproduced=not bad, mismatched=bad)
    (out / 'EXT-1-RECORD.json').write_text(json.dumps(rec, indent=1) + '\n'); print(json.dumps(dict(earlier_six_reproduced=not bad, mismatched=bad)))
    if bad: sys.exit('earlier models not reproduced: ' + ', '.join(bad))


if __name__ == '__main__':
    main()
