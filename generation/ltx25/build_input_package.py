#!/usr/bin/env python3
"""Build the LTX-2.5 distilled input package for cluster (ltx25d-inputs-v1) from the verified LTX inputs (research server only).

Source: the input root built by build_ltx25_inputs.py (outputs/wm-calibration/ltx25-inputs-v1), whose frames20.json records
every file's sha256 and the pixel identity of the 512 x 288 history with the bench clips. Per window the package holds
  DEV:  windows/dev/<w>/history20.mp4        41 frames 512x288 20 fps (stage-1 conditioning)
        windows/dev/<w>/history20_1024.mp4   41 frames 1024x576 (stage-2 conditioning)
        windows/dev/<w>/real20_1024.mp4      161 frames 1024x576 (the real reference's input)
        windows/dev/<w>/frames20.json        frame list and checks
        specs/dev/<w>/<w>-s{0..15}.json, <w>-recon.json
  TEST: windows/test/<w>/history20.mp4, history20_1024.mp4, frames20.json (history only: no TEST frame after t0)
        specs/test/<w>/<w>-s{0..15}.json   (the TEST references come after freezing, in a separate add-on package)
The specs are made by make_ltx25_specs.build (variant distilled, quantization none = bf16, the shared seed table).
Nothing is measured; no frame after t0 of a TEST window is read.

Packing: windows are grouped into chunk tars (one batch-system job each, model resident across the chunk),
chunks/<package>-<split>-cNN.tar, plus the smoke chunk dev-smoke; queues/*.txt list chunk ids for `queue chunk from ...`.
The chunk tars, queue lists, names lists, manifests and README go into one outer tar per split (ltx25d-inputs-v1-dev.tar,
-test.tar); both extract into the same directory.

  build_input_package.py --inputs LTX25_INPUT_ROOT --bench BENCH_V1_DIR --out DIR [--windows-per-chunk 20] [--test-queues 2]
                         [--smoke-names FILE --smoke-count 2] [--snapshot DIR]
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
import make_ltx25_specs as mls  # noqa: E402

PACKAGE = 'ltx25d-inputs-v1'
NAMES_SHA256 = {'dev': '1e7b3878a485fb45fa295dddd29e63237f72a7423230cb0de7c012808f47fbd1',   # protocol-v1.md section 1
                'test': 'e8343f526ed2e25112af41a63a45a9866171f5b74d56cff7663dd1c2db086233'}
FILES = {'dev': ('history20.mp4', 'history20_1024.mp4', 'real20_1024.mp4', 'frames20.json'),
         'test': ('history20.mp4', 'history20_1024.mp4', 'frames20.json')}


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


def check_window(inputs: Path, split: str, w: str) -> dict:
    """The window's files against its frames20.json (sha256), the canvas identity check, and TEST history-only."""
    d = inputs / 'windows' / split / w
    rec = json.loads((d / 'frames20.json').read_text())
    if rec.get('schema') != 'wm-calib-bench.ltx25.inputs.v2' or rec.get('window') != w:
        raise SystemExit(f'{d}/frames20.json: unexpected record')
    if split == 'test' and not rec.get('history_only'):
        raise SystemExit(f'{w}: a TEST window must be history-only')
    if split == 'test' and any(p.name.startswith('real20') for p in d.iterdir()):
        raise SystemExit(f'{w}: TEST real clips present in the source; refusing to package them')
    if not rec['checks']['canvas_identity']['identical']:
        raise SystemExit(f'{w}: canvas identity check not passed')
    out = {}
    for name in FILES[split]:
        p = d / name
        if name != 'frames20.json':
            want = rec['files'][name]['sha256']
            got = sha256_file(p)
            if got != want:
                raise SystemExit(f'{p}: sha256 {got}, frames20.json says {want}')
        out[name] = dict(path=f'windows/{split}/{w}/{name}', sha256=sha256_file(p), bytes=p.stat().st_size)
    return dict(window=w, split=split, files=out, history_only=rec['history_only'], max_abs_offset_ms=rec['timeline']['max_abs_offset_ms'],
                hr_down_vs_canvas=rec['checks'].get('hr_down_vs_canvas_history'))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--inputs', type=Path, required=True, help='the LTX input root (outputs/wm-calibration/ltx25-inputs-v1)')
    ap.add_argument('--bench', type=Path, required=True, help='bench-v1 directory (names lists)')
    ap.add_argument('--out', type=Path, required=True, help=f'package directory (e.g. .../cluster-packages/{PACKAGE})')
    ap.add_argument('--windows-per-chunk', type=int, default=20)
    ap.add_argument('--test-queues', type=int, default=2, help='split the TEST chunks into this many queue files (/home quota batches)')
    ap.add_argument('--smoke-names', type=Path, help='DEV names to draw the smoke chunk from (default <bench>/names-smoke-dev.txt)')
    ap.add_argument('--smoke-count', type=int, default=2)
    ap.add_argument('--snapshot', type=Path, help='unpacked code snapshot (default $WCB_SNAPSHOT_DIR or <repo>/snapshot)')
    args = ap.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f'{args.out} exists and is not empty')
    args.out.mkdir(parents=True, exist_ok=True)
    stage = args.out / '.stage'
    stage.mkdir()
    t_start = time.time()

    names = {}
    for split in ('dev', 'test'):
        f = args.bench / f'names-{split}-gen.txt'
        if sha256_file(f) != NAMES_SHA256[split]:
            raise SystemExit(f'{f}: sha256 differs from protocol-v1.md')
        names[split] = f.read_text().split()
        (stage / f'names-{split}-gen.txt').write_bytes(f.read_bytes())
    smoke_src = (args.smoke_names or args.bench / 'names-smoke-dev.txt').read_text().split()
    smoke = [w for w in smoke_src if w in set(names['dev'])][:args.smoke_count]
    if len(smoke) != args.smoke_count:
        raise SystemExit('not enough smoke windows')

    windows = {(s, w): check_window(args.inputs, s, w) for s in ('dev', 'test') for w in names[s]}
    print(json.dumps(dict(windows=len(windows), seconds=round(time.time() - t_start))), flush=True)

    spec_counts = {}
    for split in ('dev', 'test'):
        specs = mls.build(args.inputs, split, names[split], list(range(16)), split == 'dev', f'ltx25d-{split}', stage / 'specs' / split, args.snapshot,
                          variant='distilled')
        if any(s['generation']['quantization'] != 'none' for s in specs):
            raise SystemExit('package specs must be bf16 (quantization none)')
        spec_counts[split] = len(specs)

    chunks = []
    for split in ('dev', 'test'):
        ws = names[split]
        for i in range(0, len(ws), args.windows_per_chunk):
            chunks.append(dict(id=f'{split}-c{i // args.windows_per_chunk + 1:02d}', split=split, windows=ws[i:i + args.windows_per_chunk]))
    chunks.append(dict(id='dev-smoke', split='dev', windows=smoke))
    (args.out / 'chunks').mkdir()
    inner = []
    for c in chunks:
        tar_path = args.out / 'chunks' / f"{PACKAGE}-{c['id']}.tar"
        files = []
        for w in c['windows']:
            for name in FILES[c['split']]:
                rec = windows[(c['split'], w)]['files'][name]
                files.append(dict(path=rec['path'], sha256=rec['sha256'], bytes=rec['bytes'], src=str(args.inputs / rec['path'])))
            for p in sorted((stage / 'specs' / c['split'] / w).iterdir()):
                files.append(dict(path=str(p.relative_to(stage)), sha256=sha256_file(p), bytes=p.stat().st_size, src=str(p)))
        c['files'] = [{k: f[k] for k in ('path', 'sha256', 'bytes')} for f in files]
        c['specs'] = sum(1 for f in files if f['path'].startswith('specs/'))
        chunk_json = json.dumps(dict(schema='wm-calib-bench.ltx25d.chunk.v1', package=PACKAGE, **{k: c[k] for k in ('id', 'split', 'windows', 'specs', 'files')}),
                                indent=1).encode() + b'\n'
        with tarfile.open(tar_path, 'w', format=tarfile.PAX_FORMAT) as tar:
            add_bytes(tar, chunk_json, f"chunk-{c['id']}.json")
            for f in files:
                add_file(tar, Path(f['src']), f['path'])
        c['tar'] = dict(path=f'chunks/{tar_path.name}', sha256=sha256_file(tar_path), bytes=tar_path.stat().st_size)
        if c['id'] != 'dev-smoke':
            inner += [(f['path'], f['sha256']) for f in c['files']]
        print(json.dumps(dict(chunk=c['id'], windows=len(c['windows']), specs=c['specs'], bytes=c['tar']['bytes'])), flush=True)

    (args.out / 'queues').mkdir()
    dev_ids = [c['id'] for c in chunks if c['split'] == 'dev' and c['id'] != 'dev-smoke']
    test_ids = [c['id'] for c in chunks if c['split'] == 'test']
    queues = {'dev-smoke': ['dev-smoke'], 'dev': dev_ids}
    per = -(-len(test_ids) // args.test_queues)
    for q in range(args.test_queues):
        queues[f'test-{chr(ord("a") + q)}'] = test_ids[q * per:(q + 1) * per]
    for q, ids in queues.items():
        (args.out / 'queues' / f'{q}.txt').write_text(''.join(f'{i}\n' for i in ids))

    spec_manifests = {s: json.loads((stage / 'specs' / s / 'manifest.json').read_text()) for s in ('dev', 'test')}
    summary = dict(schema='wm-calib-bench.ltx25d.input_package.v1', package=PACKAGE, created=datetime.now().astimezone().isoformat(timespec='seconds'),
                   source_inputs=str(args.inputs), source_bench='bench-v1', names_sha256=NAMES_SHA256, windows={s: len(names[s]) for s in names},
                   specs=spec_counts, smoke_windows=smoke, windows_per_chunk=args.windows_per_chunk, queues={q: len(v) for q, v in queues.items()},
                   builder_sha256=sha256_file(Path(__file__)), spec_generator_sha256=sha256_file(HERE / 'make_ltx25_specs.py'),
                   pins_sha256=sha256_file(HERE / 'ltx25_pins.json'), spec_manifests=spec_manifests,
                   chunks=[{k: c[k] for k in ('id', 'split', 'windows', 'specs', 'tar')} for c in chunks],
                   window_files=[windows[(s, w)] for s in ('dev', 'test') for w in names[s]])
    for split in ('dev', 'test'):
        part = dict(summary, chunks=[x for x in summary['chunks'] if x['split'] == split],
                    window_files=[r for r in summary['window_files'] if r['split'] == split], spec_manifests={split: spec_manifests[split]})
        (args.out / f'manifest-{split}.json').write_text(json.dumps(part, indent=1) + '\n')
        lines = [f'{sha}  {p}' for p, sha in sorted(x for x in inner if x[0].split('/')[1] == split)]
        lines += [f"{c['tar']['sha256']}  {c['tar']['path']}" for c in chunks if c['split'] == split]
        (args.out / f'MANIFEST-{split}.sha256').write_text('\n'.join(lines) + '\n')
        (args.out / f'names-{split}-gen.txt').write_bytes((stage / f'names-{split}-gen.txt').read_bytes())

    readme = (HERE / 'package_README.md').read_text().format(
        created=summary['created'], dev_windows=len(names['dev']), test_windows=len(names['test']), dev_chunks=len(dev_ids),
        test_chunks=len(test_ids), wpc=args.windows_per_chunk, smoke_count=len(smoke), smoke=', '.join(smoke),
        test_queues=', '.join(q for q in queues if q.startswith('test')))
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
    shutil.rmtree(stage)
    print(json.dumps(dict(package=str(args.out), seconds=round(time.time() - t_start),
                          tars={s: (args.out / f'{PACKAGE}-{s}.tar').stat().st_size for s in ('dev', 'test')}, queues=summary['queues'])))


if __name__ == '__main__':
    main()
