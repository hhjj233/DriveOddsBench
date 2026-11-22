#!/usr/bin/env python3
"""Build the Epona cluster input package (epona-inputs-v1) from built Epona input roots.

Input root: built with build_epona_inputs.py in the bench-v1 band [579, 1451):
  windows/dev/<w>/{history.mp4, real.mp4, past_poses.json, build.json}    60 DEV windows (real.mp4 feeds the DEV references)
  windows/test/<w>/{history.mp4, past_poses.json, build.json}             200 TEST windows, --history-only: frames k <= 0 and
                                                                          past poses only (checked: no real.mp4, no tick > 0)
Specs (make_epona_specs.build, seed table of make_bench_nano_specs.seed_of, bound to --batch-k 16 = one window per batch):
  DEV  run epona-dev-b16:  16 continuations + 1 real reference per window
  TEST run epona-test-b16: 16 continuations per window (the TEST references are made on the research server after freezing,
                           from the real frames, with frames 0-10 taken from the cluster continuations: run_epona_bench.py
                           --recon-prefix-root)
Chunks (one batch-system job each): chunks/epona-inputs-v1-<id>.tar = chunk-<id>.json + windows/ + specs/ of up to
--windows-per-chunk windows; ids dev-cNN, test-cNN and dev-smoke (2 DEV windows). queues/*.txt list chunk ids:
dev-smoke, dev, test (all TEST chunks), test-a / test-b (halves). Outer tars epona-inputs-v1-dev.tar / -test.tar hold the
chunk tars, queues, names, manifests and README; both extract into epona-inputs-v1/. Nothing is measured or decoded here.

  build_input_package.py --inputs ROOT --bench DIR --out DIR [--windows-per-chunk 8] [--smoke W1,W2] [--snapshot DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import sys
import tarfile
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_epona_specs as mes  # noqa: E402

PACKAGE = 'epona-inputs-v1'
NAMES_SHA256 = {'dev': '1e7b3878a485fb45fa295dddd29e63237f72a7423230cb0de7c012808f47fbd1',   # protocol-v1.md section 1
                'test': 'e8343f526ed2e25112af41a63a45a9866171f5b74d56cff7663dd1c2db086233'}
BAND = [579, 1451]
BATCH_K = 16
FILES = {'dev': ('history.mp4', 'real.mp4', 'past_poses.json', 'build.json'), 'test': ('history.mp4', 'past_poses.json', 'build.json')}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


def add_file(tar: tarfile.TarFile, path: Path, arcname: str) -> None:
    ti = tar.gettarinfo(str(path), arcname)
    ti.uid = ti.gid = 0
    ti.uname = ti.gname = ''
    ti.mtime = int(path.stat().st_mtime)
    with open(path, 'rb') as f:
        tar.addfile(ti, f)


def add_bytes(tar: tarfile.TarFile, data: bytes, arcname: str) -> None:
    ti = tarfile.TarInfo(arcname)
    ti.size, ti.mode, ti.mtime = len(data), 0o644, int(time.time())
    tar.addfile(ti, io.BytesIO(data))


def check_window(root: Path, split: str, w: str) -> dict:
    d = root / 'windows' / split / w
    b = json.loads((d / 'build.json').read_text())
    problems = []
    if b['render'].get('model_band_rows') != BAND:
        problems.append(f'band {b["render"].get("model_band_rows")}')
    if not b['render']['canvas_reproduced']['identical']:
        problems.append('canvas not reproduced')
    for f in FILES[split]:
        if not (d / f).is_file():
            problems.append(f'{f} missing')
    for rel, c in b['clips'].items():
        if not c['identical'] or sha256_file(d / rel) != c['sha256']:
            problems.append(f'{rel} differs from its build record')
    pp = json.loads((d / 'past_poses.json').read_text())
    if max(pp['ticks']) > 0 or pp['ticks'] != list(range(-20, 1, 2)):
        problems.append(f'pose ticks {pp["ticks"]}')
    if split == 'test':
        if not b.get('history_only') or (d / 'real.mp4').exists() or any(int(k) > 0 for k in b['ticks']['frame_paths']):
            problems.append('TEST window is not history-only')
    if problems:
        raise SystemExit(f'{split}/{w}: {problems}')
    return dict(window=w, split=split, files={f: dict(sha256=sha256_file(d / f), bytes=(d / f).stat().st_size) for f in FILES[split]},
                band=b['render']['model_band_rows'], history_only=bool(b.get('history_only')), builder_sha256=b['builder_sha256'])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--inputs', type=Path, required=True, help='Epona input root (build_epona_inputs.py, band [579, 1451))')
    ap.add_argument('--bench', type=Path, required=True, help='bench-v1 directory (names lists)')
    ap.add_argument('--out', type=Path, required=True, help=f'package directory (e.g. .../cluster-packages/{PACKAGE})')
    ap.add_argument('--windows-per-chunk', type=int, default=8)
    ap.add_argument('--smoke', default='2eeaccb0-t2.5,47a9fb77-t9.5', help='DEV windows of the dev-smoke chunk')
    ap.add_argument('--snapshot', type=Path, help='unpacked code snapshot (seed_of; default $WCB_SNAPSHOT_DIR or <repo>/snapshot)')
    ap.add_argument('--keep-stage', action='store_true')
    args = ap.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f'{args.out} exists and is not empty')
    args.out.mkdir(parents=True, exist_ok=True)
    stage = args.out / '.stage'
    t_start = time.time()

    names = {}
    for split in ('dev', 'test'):
        f = args.bench / f'names-{split}-gen.txt'
        if sha256_file(f) != NAMES_SHA256[split]:
            raise SystemExit(f'{f}: sha256 differs from protocol-v1.md')
        names[split] = f.read_text().split()
    smoke = args.smoke.split(',')
    if not set(smoke) <= set(names['dev']):
        raise SystemExit('smoke windows must be DEV generation windows')

    windows = []
    for split in ('dev', 'test'):
        for w in names[split]:
            windows.append(check_window(args.inputs, split, w))
            dst = stage / 'windows' / split / w
            dst.mkdir(parents=True, exist_ok=True)
            for f in FILES[split]:
                shutil.copyfile(args.inputs / 'windows' / split / w / f, dst / f)
    print(json.dumps(dict(windows=len(windows), seconds=round(time.time() - t_start))), flush=True)

    spec_counts = {}
    spec_counts['dev'] = len(mes.build(stage, 'dev', names['dev'], list(range(16)), True, 'epona-dev-b16', stage / 'specs' / 'dev', args.snapshot, BATCH_K))
    spec_counts['test'] = len(mes.build(stage, 'test', names['test'], list(range(16)), False, 'epona-test-b16', stage / 'specs' / 'test', args.snapshot, BATCH_K))

    chunks = []
    for split in ('dev', 'test'):
        ws = names[split]
        for i in range(0, len(ws), args.windows_per_chunk):
            chunks.append(dict(id=f'{split}-c{i // args.windows_per_chunk + 1:02d}', split=split, windows=ws[i:i + args.windows_per_chunk]))
    chunks.append(dict(id='dev-smoke', split='dev', windows=smoke))
    (args.out / 'chunks').mkdir(exist_ok=True)
    inner = []
    for c in chunks:
        tar_path = args.out / 'chunks' / f"{PACKAGE}-{c['id']}.tar"
        files = []
        for w in c['windows']:
            for p in sorted((stage / 'windows' / c['split'] / w).iterdir()) + sorted((stage / 'specs' / c['split'] / w).iterdir()):
                files.append(dict(path=str(p.relative_to(stage)), sha256=sha256_file(p), bytes=p.stat().st_size))
        c['files'] = files
        c['specs'] = sum(1 for f in files if f['path'].startswith('specs/'))
        c['batch_k'] = BATCH_K
        c['model_band_rows'] = BAND
        chunk_json = json.dumps(dict(schema='wm-calib-bench.epona.chunk.v1', package=PACKAGE, **c), indent=1).encode() + b'\n'
        with tarfile.open(tar_path, 'w', format=tarfile.PAX_FORMAT) as tar:
            add_bytes(tar, chunk_json, f"chunk-{c['id']}.json")
            for f in files:
                add_file(tar, stage / f['path'], f['path'])
        c['tar'] = dict(path=f'chunks/{tar_path.name}', sha256=sha256_file(tar_path), bytes=tar_path.stat().st_size)
        if c['id'] != 'dev-smoke':
            inner += [(f['path'], f['sha256'], f['bytes']) for f in files]

    (args.out / 'queues').mkdir(exist_ok=True)
    dev_ids = [c['id'] for c in chunks if c['split'] == 'dev' and c['id'] != 'dev-smoke']
    test_ids = [c['id'] for c in chunks if c['split'] == 'test']
    half = -(-len(test_ids) // 2)
    queues = {'dev-smoke': ['dev-smoke'], 'dev': dev_ids, 'test': test_ids, 'test-a': test_ids[:half], 'test-b': test_ids[half:]}
    for q, ids in queues.items():
        (args.out / 'queues' / f'{q}.txt').write_text(''.join(f'{i}\n' for i in ids))

    spec_manifests = {s: json.loads((stage / 'specs' / s / 'manifest.json').read_text()) for s in ('dev', 'test')}
    summary = dict(schema='wm-calib-bench.epona.input_package.v1', package=PACKAGE, created=datetime.now().astimezone().isoformat(timespec='seconds'),
                   source_bench='bench-v1', names_sha256=NAMES_SHA256, windows={s: len(names[s]) for s in names}, specs=spec_counts,
                   model_band_rows=BAND, batch_k=BATCH_K, test_history_only=True, smoke_windows=smoke, windows_per_chunk=args.windows_per_chunk,
                   queues={q: len(v) for q, v in queues.items()}, builder_sha256=sha256_file(Path(__file__)),
                   spec_generator_sha256=sha256_file(HERE / 'make_epona_specs.py'), pins_sha256=sha256_file(HERE / 'epona_pins.json'),
                   input_builder_sha256=sorted({w['builder_sha256'] for w in windows}), spec_manifests=spec_manifests,
                   chunks=[{k: c[k] for k in ('id', 'split', 'windows', 'specs', 'tar')} for c in chunks], window_files=windows)
    for split in ('dev', 'test'):
        part = dict(summary, chunks=[x for x in summary['chunks'] if x['split'] == split],
                    window_files=[r for r in windows if r['split'] == split], spec_manifests={split: spec_manifests[split]})
        (args.out / f'manifest-{split}.json').write_text(json.dumps(part, indent=1) + '\n')
        lines = [f'{sha}  {p}' for p, sha, _ in sorted(x for x in inner if x[0].split('/')[1] == split)]
        lines += [f"{c['tar']['sha256']}  {c['tar']['path']}" for c in chunks if c['split'] == split]
        (args.out / f'MANIFEST-{split}.sha256').write_text('\n'.join(lines) + '\n')
        (args.out / f'names-{split}-gen.txt').write_bytes((args.bench / f'names-{split}-gen.txt').read_bytes())

    readme = (HERE / 'package_README.md').read_text().format(
        created=summary['created'], dev_windows=len(names['dev']), test_windows=len(names['test']), dev_chunks=len(dev_ids),
        test_chunks=len(test_ids), wpc=args.windows_per_chunk, smoke=', '.join(smoke), half_a=len(queues['test-a']), half_b=len(queues['test-b']))
    (args.out / 'README.md').write_text(readme)

    for split in ('dev', 'test'):
        outer = args.out / f'{PACKAGE}-{split}.tar'
        members = ['README.md', f'manifest-{split}.json', f'MANIFEST-{split}.sha256', f'names-{split}-gen.txt']
        members += [f'queues/{q}.txt' for q in queues if q.startswith(split)]
        members += [c['tar']['path'] for c in chunks if c['split'] == split]
        with tarfile.open(outer, 'w', format=tarfile.PAX_FORMAT) as tar:
            for m in members:
                add_file(tar, args.out / m, f'{PACKAGE}/{m}')
        (args.out / f'{outer.name}.sha256').write_text(f'{sha256_file(outer)}  {outer.name}\n')
    (args.out / 'SHA256SUMS').write_text(''.join((args.out / f'{PACKAGE}-{s}.tar.sha256').read_text() for s in ('dev', 'test')))
    shutil.rmtree(args.out / 'chunks')
    if not args.keep_stage:
        shutil.rmtree(stage)
    print(json.dumps(dict(package=str(args.out), seconds=round(time.time() - t_start), chunks=dict(dev=len(dev_ids), test=len(test_ids)),
                          tars={s: (args.out / f'{PACKAGE}-{s}.tar').stat().st_size for s in ('dev', 'test')})))


if __name__ == '__main__':
    main()
