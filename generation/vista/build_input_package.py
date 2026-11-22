#!/usr/bin/env python3
"""Build the Vista cluster input package from the built Vista input root (research server only).

Package vista-inputs-v3 (default): generation size `320x576` (the UNet code
unmodified). Earlier packages, kept: vista-inputs-v2 (`canvas`, 288 x 512) and vista-inputs-v1 (`native`, 576 x 1024).
Input root (build_vista_inputs.py --size ...): windows/dev/<w>/{history21_<tag>.mp4, real81_<tag>.mp4, actions_t2.json,
inputs.json} for the 60 DEV generation windows; windows/test/<w>/{history21_<tag>.mp4, inputs.json} for the 200 TEST
windows, history only (frames k = -20..0; checked: no real clip, no action file, 21 frames); <tag> = 512 / 1024.
Specs (make_vista_specs.build, layout `official` = the released setting without --low_vram, seed_of table):
  DEV  runs vista-dev-prefix, vista-dev-recon, vista-dev-h0-k16 (prefix, real reference, 16 h0 continuations per window)
  TEST runs vista-test-prefix, vista-test-h0-k16 (prefix and 16 h0 continuations; the prefix uses frames k <= -1 only).
       The TEST references are made after the freeze (mode --recon-addon, or on the research server).
  T2 (--tracks h0,t2): adds vista-<split>-t2-k16 to DEV only (TEST T2 needs future poses, so only after its own freeze);
       off by default.
Chunks (one batch-system job each): chunks/<package>-<id>.tar = chunk-<id>.json + windows/ + specs/ of up to
--windows-per-chunk windows (default: pins bench.cluster.windows_per_chunk); ids dev-cNN, test-cNN and dev-smoke (2 DEV
windows: prefix, reference, h0 s0 and s1, the same spec files as their DEV chunk). queues/*.txt: dev-smoke, dev, test,
test-a, test-b. Outer tars <package>-dev.tar / -test.tar hold the chunk tars, queues, names, manifests and README; both
extract into <package>/. Nothing is measured or decoded here.

  build_input_package.py --inputs ROOT --bench DIR --out DIR [--size canvas] [--package vista-inputs-v2]
                         [--windows-per-chunk N] [--tracks h0] [--snapshot DIR]
  build_input_package.py --recon-addon test --inputs ROOT --prefix-root COLLECTED_RUNS --out DIR --frozen-file PATH
      (after the freeze: the TEST references as an add-on, <package>-test-recon.tar, with the collected cluster prefixes so
       that every reference shares its window's prefix byte for byte; --recon-addon dev builds the same for DEV as a test)
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
import make_vista_specs as mvs  # noqa: E402

PINS = mvs.PINS
PACKAGES = {'320x576': 'vista-inputs-v3', 'canvas': 'vista-inputs-v2', 'native': 'vista-inputs-v1'}
PACKAGE = PACKAGES[PINS['bench']['default_size']]   # set from --size / --package in main()
SIZE = PINS['bench']['default_size']          # set from --size in main()
NAMES_SHA256 = {'dev': '1e7b3878a485fb45fa295dddd29e63237f72a7423230cb0de7c012808f47fbd1',   # protocol-v1.md section 1
                'test': 'e8343f526ed2e25112af41a63a45a9866171f5b74d56cff7663dd1c2db086233'}
FILES = {}                                    # split -> input files of a window at SIZE (set_size)


def set_size(size: str, package: str) -> None:
    global SIZE, PACKAGE, FILES
    SIZE, PACKAGE = size, package
    tag = PINS['bench']['sizes'][size]['input_tag']
    FILES = {'dev': (f'history21_{tag}.mp4', f'real81_{tag}.mp4', 'actions_t2.json', 'inputs.json'), 'test': (f'history21_{tag}.mp4', 'inputs.json')}
SMOKE_SAMPLES = (0, 1)


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
    rec = json.loads((d / 'inputs.json').read_text())
    problems = []
    for f in FILES[split]:
        if not (d / f).is_file():
            problems.append(f'{f} missing')
    for name, f in rec['files'].items():
        if name != 'actions_t2.json' and sha256_file(d / name) != f['sha256']:
            problems.append(f'{name} differs from inputs.json')
    chk = rec['checks'].get('model_vs_canvas') or rec['checks']['model_down_vs_canvas']
    mapping = rec['checks'].get('mapping') or ('area2' if 'model_down_vs_canvas' in rec['checks'] else 'identity')
    if rec.get('size', 'native') != SIZE:
        problems.append(f'inputs at size {rec.get("size", "native")}, the package is at {SIZE}')
    elif (mapping == 'area2' and (chk['max_abs'] > 1 or not rec['checks']['area2_matches_cv2'])) or (mapping == 'identity' and chk['max_abs'] != 0) \
            or (mapping == 'inter_area' and rec['checks'].get('inter_area_matches_cv2') is not True):
        problems.append(f'render check ({mapping}: {chk})')
    if split == 'test':
        extra = sorted(p.name for p in d.iterdir() if p.name not in FILES['test'])
        if not rec['history_only'] or rec['timeline']['frames'] != 21 or extra or any(f['k'] > 0 for f in rec['frames']):
            problems.append(f'TEST window is not history-only (extra files {extra})')
    if problems:
        raise SystemExit(f'{split}/{w}: {problems}')
    return dict(window=w, split=split, files={f: dict(sha256=sha256_file(d / f), bytes=(d / f).stat().st_size) for f in FILES[split]},
                history_only=bool(rec['history_only']), builder_sha256=rec['software']['builder_sha256'])


def write_chunk(stage: Path, root: Path, chunk_id: str, split: str, windows: list, spec_files: dict, layout: str, out_dir: Path) -> dict:
    """spec_files: window -> list of spec paths (in the staged spec tree). Returns the chunk record."""
    files, specs = [], []
    tar_path = out_dir / 'chunks' / f'{PACKAGE}-{chunk_id}.tar'
    with tarfile.open(tar_path, 'w') as tar:
        for w in windows:
            for f in FILES[split]:
                p = root / 'windows' / split / w / f
                arc = f'windows/{split}/{w}/{f}'
                add_file(tar, p, arc)
                files.append(dict(path=arc, sha256=sha256_file(p)))
            for sp in spec_files[w]:
                arc = f'specs/{split}/{w}/{sp.name}'
                add_file(tar, sp, arc)
                files.append(dict(path=arc, sha256=sha256_file(sp)))
                specs.append(sp.name)
        rec = dict(schema='wm-calib-bench.vista.chunk.v1', package=PACKAGE, chunk=chunk_id, split=split, windows=windows, specs=len(specs),
                   spec_names=specs, layout=layout, size=SIZE, size_hw=PINS['bench']['sizes'][SIZE]['hw'], files=files,
                   created=datetime.now().astimezone().isoformat(timespec='seconds'))
        add_bytes(tar, (json.dumps(rec, indent=1) + '\n').encode(), f'chunk-{chunk_id}.json')
    return dict(rec, tar=tar_path.name, tar_sha256=sha256_file(tar_path), tar_bytes=tar_path.stat().st_size)


def build_specs(stage: Path, root: Path, split: str, names: list, tracks: list, snapshot: Path | None) -> dict:
    """Spec tree of the split in the stage (official layout); returns window -> spec paths."""
    out = stage / 'specs' / split
    mvs.build(root, split, names, list(range(16)), tracks, recon=split == 'dev', prefix=True, run_prefix=f'vista-{split}', out=out,
              snapshot=snapshot, layout='official', size=SIZE)
    return {w: sorted((out / w).glob('*.json')) for w in names}


def outer_tar(out_dir: Path, split: str, members: list) -> Path:
    p = out_dir / f'{PACKAGE}-{split}.tar'
    with tarfile.open(p, 'w') as tar:
        for rel in members:
            add_file(tar, out_dir / rel, f'{PACKAGE}/{rel}')
    return p


def main_package(args):
    if args.out.exists() and any(p for p in args.out.iterdir() if p.name != '.stage'):
        raise SystemExit(f'{args.out} exists and is not empty')
    args.out.mkdir(parents=True, exist_ok=True)
    stage = args.out / '.stage'
    shutil.rmtree(stage, ignore_errors=True)
    (args.out / 'chunks').mkdir(parents=True, exist_ok=True)
    (args.out / 'queues').mkdir(parents=True, exist_ok=True)
    tracks = [t for t in args.tracks.split(',') if t]
    names = {}
    for split in ('dev', 'test'):
        f = args.bench / f'names-{split}-gen.txt'
        if sha256_file(f) != NAMES_SHA256[split]:
            raise SystemExit(f'{f}: sha256 differs from protocol-v1.md')
        names[split] = f.read_text().split()
        shutil.copy2(f, args.out / f'names-{split}-gen.txt')
    smoke = args.smoke.split(',')
    if not set(smoke) <= set(names['dev']):
        raise SystemExit('smoke windows must be DEV generation windows')
    manifests = {}
    for split in ('dev', 'test'):
        wins = [check_window(args.inputs, split, w) for w in names[split]]
        split_tracks = tracks if split == 'dev' else ['h0']
        spec_files = build_specs(stage, args.inputs, split, names[split], split_tracks, args.snapshot)
        chunks = []
        wl = names[split]
        for i in range(0, len(wl), args.windows_per_chunk):
            cid = f'{split}-c{i // args.windows_per_chunk + 1:02d}'
            chunks.append(write_chunk(stage, args.inputs, cid, split, wl[i:i + args.windows_per_chunk], spec_files, 'official', args.out))
        if split == 'dev':
            smoke_specs = {w: [p for p in spec_files[w] if p.name.endswith(('-prefix.json', '-recon.json'))
                               or (('-h0-k16__' in p.name) and p.stem.rsplit('-s', 1)[-1].isdigit() and int(p.stem.rsplit('-s', 1)[-1]) in SMOKE_SAMPLES)]
                           for w in smoke}
            chunks.insert(0, write_chunk(stage, args.inputs, 'dev-smoke', 'dev', smoke, smoke_specs, 'official', args.out))
        ids = [c['chunk'] for c in chunks]
        if split == 'dev':
            (args.out / 'queues' / 'dev-smoke.txt').write_text('dev-smoke\n')
            (args.out / 'queues' / 'dev.txt').write_text('\n'.join(i for i in ids if i != 'dev-smoke') + '\n')
        else:
            half = (len(ids) + 1) // 2
            (args.out / 'queues' / 'test.txt').write_text('\n'.join(ids) + '\n')
            (args.out / 'queues' / 'test-a.txt').write_text('\n'.join(ids[:half]) + '\n')
            (args.out / 'queues' / 'test-b.txt').write_text('\n'.join(ids[half:]) + '\n')
        spec_manifest = json.loads((stage / 'specs' / split / 'manifest.json').read_text())
        manifests[split] = dict(schema='wm-calib-bench.vista.package_manifest.v1', package=PACKAGE, split=split, windows=wins,
                                chunks=[{k: c[k] for k in ('chunk', 'windows', 'specs', 'tar', 'tar_sha256', 'tar_bytes', 'layout')} for c in chunks],
                                spec_manifest=spec_manifest, tracks=split_tracks, layout='official', size=SIZE, windows_per_chunk=args.windows_per_chunk,
                                builder_sha256=sha256_file(Path(__file__)), generator_sha256=sha256_file(HERE / 'make_vista_specs.py'),
                                pins_sha256=sha256_file(HERE / 'vista_pins.json'), created=datetime.now().astimezone().isoformat(timespec='seconds'))
        (args.out / f'manifest-{split}.json').write_text(json.dumps(manifests[split], indent=1) + '\n')
        lines = [f'{c["tar_sha256"]}  chunks/{c["tar"]}' for c in chunks]
        for c in chunks:
            lines += [f'{f["sha256"]}  {f["path"]}' for f in c['files']]
        (args.out / f'MANIFEST-{split}.sha256').write_text('\n'.join(sorted(set(lines))) + '\n')
    readme = (HERE / 'package_README.md').read_text().format(
        created=datetime.now().astimezone().isoformat(timespec='minutes'), dev_windows=len(names['dev']), test_windows=len(names['test']),
        dev_chunks=len([c for c in manifests['dev']['chunks'] if c['chunk'] != 'dev-smoke']), test_chunks=len(manifests['test']['chunks']),
        wpc=args.windows_per_chunk, smoke=', '.join(smoke), half_a=len((args.out / 'queues' / 'test-a.txt').read_text().split()),
        half_b=len((args.out / 'queues' / 'test-b.txt').read_text().split()), tracks=', '.join(tracks), package=PACKAGE,
        size=f"{PINS['bench']['sizes'][SIZE]['hw'][0]}×{PINS['bench']['sizes'][SIZE]['hw'][1]}（{SIZE}）")
    (args.out / 'README.md').write_text(readme)
    common = ['README.md', 'queues/dev-smoke.txt', 'queues/dev.txt', 'queues/test.txt', 'queues/test-a.txt', 'queues/test-b.txt']
    for split in ('dev', 'test'):
        members = common + [f'names-{split}-gen.txt', f'manifest-{split}.json', f'MANIFEST-{split}.sha256'] + \
                  [f'chunks/{c["tar"]}' for c in manifests[split]['chunks']]
        outer_tar(args.out, split, members)
    sums = [f'{sha256_file(args.out / f"{PACKAGE}-{s}.tar")}  {PACKAGE}-{s}.tar' for s in ('dev', 'test')]
    (args.out / 'SHA256SUMS').write_text('\n'.join(sums) + '\n')
    for s in ('dev', 'test'):
        (args.out / f'{PACKAGE}-{s}.tar.sha256').write_text(next(x for x in sums if x.endswith(f'-{s}.tar')) + '\n')
    shutil.rmtree(args.out / 'chunks')                   # the chunk tars live inside the outer tars
    for split in ('dev', 'test'):
        (args.out / f'MANIFEST-{split}.sha256').unlink()  # (also inside); manifest-*.json stay next to the tars
    shutil.rmtree(stage)
    print(json.dumps(dict(out=str(args.out), sums=sums, dev_chunks=len(manifests['dev']['chunks']), test_chunks=len(manifests['test']['chunks']),
                          bytes={s: (args.out / f'{PACKAGE}-{s}.tar').stat().st_size for s in ('dev', 'test')}), indent=1))


def main_recon_addon(args):
    split = args.recon_addon
    if split == 'test' and (args.frozen_file is None or not args.frozen_file.is_file() or 'vista' not in args.frozen_file.name):
        raise SystemExit('TEST references need the frozen Vista protocol file (--frozen-file protocol-vista-frozen.txt)')
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    stage = out / '.stage-recon'
    shutil.rmtree(stage, ignore_errors=True)
    names = (args.bench / f'names-{split}-gen.txt').read_text().split()
    spec_dir = stage / 'specs' / split
    mvs.build(args.inputs, split, names, [], [], recon=True, prefix=False, run_prefix=f'vista-{split}', out=spec_dir, snapshot=args.snapshot, layout='official',
              size=SIZE)
    real_name = FILES['dev'][1]
    tar_path = out / f'{PACKAGE}-{split}-recon.tar'
    files = []
    with tarfile.open(tar_path, 'w') as tar:
        for w in names:
            p = args.inputs / 'windows' / split / w / real_name
            if not p.is_file():
                raise SystemExit(f'{p} missing (build_vista_inputs.py --allow-test-future after the freeze)')
            add_file(tar, p, f'windows/{split}/{w}/{real_name}'); files.append(dict(path=f'windows/{split}/{w}/{real_name}', sha256=sha256_file(p)))
            pd = args.prefix_root / f'vista-{split}-prefix' / 'native' / f'{w}-prefix'
            rec = json.loads((pd / 'sample_outputs.json').read_text())
            if rec.get('status') != 'success' or sha256_file(pd / 'prefix.mp4') != rec['video_sha256']:
                raise SystemExit(f'{pd}: not a finished prefix')
            if (rec.get('size') or {}).get('name', 'native') != SIZE:
                raise SystemExit(f'{pd}: a prefix of another size')
            native_prefix = f'prefix{(rec.get("size") or {}).get("hw", [576, 1024])[1]}.mp4'   # the generation-size prefix (record key video1024_sha256)
            for f in ('prefix.mp4', 'sample_outputs.json') + ((native_prefix,) if rec.get('video1024_sha256') else ()):
                arc = f'prefixes/vista-{split}-prefix/native/{w}-prefix/{f}'
                add_file(tar, pd / f, arc); files.append(dict(path=arc, sha256=sha256_file(pd / f)))
            for sp in sorted((spec_dir / w).glob('*.json')):
                arc = f'specs/{split}/{w}/{sp.name}'
                add_file(tar, sp, arc); files.append(dict(path=arc, sha256=sha256_file(sp)))
        rec = dict(schema='wm-calib-bench.vista.recon_addon.v1', package=PACKAGE, split=split, windows=names, files=files, layout='official', size=SIZE,
                   created=datetime.now().astimezone().isoformat(timespec='seconds'))
        add_bytes(tar, (json.dumps(rec, indent=1) + '\n').encode(), f'recon-addon-{split}.json')
    (out / f'{PACKAGE}-{split}-recon.tar.sha256').write_text(f'{sha256_file(tar_path)}  {tar_path.name}\n')
    shutil.rmtree(stage)
    print(json.dumps(dict(tar=str(tar_path), bytes=tar_path.stat().st_size, windows=len(names))))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--inputs', type=Path, required=True, help='Vista input root (build_vista_inputs.py)')
    ap.add_argument('--bench', type=Path, required=True, help='bench-v1 directory (names lists)')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--windows-per-chunk', type=int, default=int(PINS['bench']['cluster']['windows_per_chunk']))
    ap.add_argument('--size', choices=tuple(PINS['bench']['sizes']), default=PINS['bench']['default_size'])
    ap.add_argument('--package', default=None, help='package name (default by size: 320x576 v3, canvas v2, native v1)')
    ap.add_argument('--tracks', default='h0', help='DEV tracks (h0, or h0,t2); TEST is h0 only')
    ap.add_argument('--smoke', default='2eeaccb0-t2.5,47a9fb77-t9.5')
    ap.add_argument('--snapshot', type=Path, default=None, help='unpacked code snapshot (seed_of; default $WCB_SNAPSHOT_DIR or <repo>/snapshot)')
    ap.add_argument('--recon-addon', choices=('dev', 'test'), default=None)
    ap.add_argument('--prefix-root', type=Path, default=None, help='collected cluster runs holding vista-<split>-prefix/native/')
    ap.add_argument('--frozen-file', type=Path, default=None)
    args = ap.parse_args()
    set_size(args.size, args.package or PACKAGES[args.size])
    if args.recon_addon:
        if args.prefix_root is None:
            raise SystemExit('--recon-addon needs --prefix-root (the collected cluster prefix runs)')
        main_recon_addon(args)
    else:
        main_package(args)


if __name__ == '__main__':
    main()
