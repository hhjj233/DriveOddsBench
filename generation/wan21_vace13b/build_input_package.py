#!/usr/bin/env python3
"""Build the Wan2.1-VACE-1.3B cluster input package from the built input root (research server only).

One package per generation size (vace_pins.json bench.sizes): wan21-vace13b-512-inputs-v1 (canvas, 512 x 288) and
wan21-vace13b-832-inputs-v1 (480p, 832 x 480); the size switch of the kit is the choice of package.
Input root (build_vace_inputs.py --size ...): windows/dev/<w>/{history16_TAG.mp4, real16_TAG.mp4, frames16_TAG.json} for the
60 DEV generation windows; windows/test/<w>/{history16_TAG.mp4, frames16_TAG.json} for the 200 TEST windows, history only
(checked: no real clip, 33 frames, no camera frame after t0).
Specs (make_vace_specs.build, the official sampler, seed_of table):
  DEV  runs vace<TAG>-dev-prefix, vace<TAG>-dev-recon, vace<TAG>-dev-k16 (per window: the prefix, the real reference and
       16 continuations: 18 samples)
  TEST runs vace<TAG>-test-prefix, vace<TAG>-test-k16 (17 samples per window); the TEST references are made on the research
       server after the freeze (build_vace_inputs.py --allow-test-future, make_vace_specs.py --mode recon --frozen-file,
       run_vace_bench.py --prefix-root on the collected TEST prefixes), never as a cluster queue.
Chunks (one batch-system job each): chunks/<package>-<id>.tar = chunk-<id>.json + windows/ + specs/<split>/<w>/ of up to
--windows-per-chunk windows (default pins bench.cluster.windows_per_chunk = 4); ids dev-cNN, test-cNN and dev-smoke (2 DEV
windows: prefix, reference, s0 and s1, the same spec files as their DEV chunk). The prompt cache
(prompt-cache/vace-t5-long.safetensors + .json) is shared by every job. queues/*.txt: dev-smoke, dev, test, test-a, test-b.
Outer tars <package>-dev.tar / -test.tar hold the chunk tars, the prompt cache, queues, names, manifests and the README;
both extract into <package>/. Nothing is decoded here.

  build_input_package.py --inputs ROOT --bench DIR --out DIR --prompt-cache FILE [--size canvas|480p]
                         [--windows-per-chunk 4] [--snapshot DIR]
"""
from __future__ import annotations

import argparse
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
import make_vace_specs as mvs  # noqa: E402
import vace_common as vc  # noqa: E402

NAMES_SHA256 = {'dev': '1e7b3878a485fb45fa295dddd29e63237f72a7423230cb0de7c012808f47fbd1',   # protocol-v1.md section 1
                'test': 'e8343f526ed2e25112af41a63a45a9866171f5b74d56cff7663dd1c2db086233'}
SMOKE_SAMPLES = (0, 1)
K = 16
CACHE_NAME = 'vace-t5-long.safetensors'


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


class Package:
    def __init__(self, size_name: str):
        self.size_name = size_name
        self.size = vc.SIZES[size_name]
        self.name = vc.B['sizes'][size_name]['package']
        n = vc.input_names(size_name)
        self.n = n
        self.files = {'dev': (n['history'], n['real'], n['frames']), 'test': (n['history'], n['frames'])}

    def check_window(self, root: Path, split: str, w: str) -> dict:
        d = root / 'windows' / split / w
        problems = []
        for f in self.files[split]:
            if not (d / f).is_file():
                problems.append(f'{f} missing')
        rec = json.loads((d / self.n['frames']).read_text())
        for name, f in rec['files'].items():
            if vc.sha256_file(d / name) != f['sha256']:
                problems.append(f'{name} differs from {self.n["frames"]}')
        if rec['size_wh'] != list(self.size) or (self.size == vc.CANVAS and rec['canvas_check']['max_abs'] != 0):
            problems.append('size or canvas check')
        if split == 'test':
            if rec['frames'] != vc.N_HIST or any(r['camera_offset'] > 0 for r in rec['rows']) or (d / self.n['real']).exists() \
                    or not rec['history_only'] or set(rec['files']) != {self.n['history']}:
                problems.append('TEST window is not history-only')
        if problems:
            raise SystemExit(f'{split}/{w}: {problems}')
        return dict(window=w, split=split, files={f: dict(sha256=vc.sha256_file(d / f), bytes=(d / f).stat().st_size) for f in self.files[split]},
                    builder_sha256=rec['builder_sha256'])

    def write_chunk(self, root: Path, chunk_id: str, split: str, windows: list, spec_files: dict, out_dir: Path) -> dict:
        """spec_files: window -> list of spec paths (in the staged spec tree). Returns the chunk record."""
        files, specs = [], []
        tar_path = out_dir / 'chunks' / f'{self.name}-{chunk_id}.tar'
        with tarfile.open(tar_path, 'w') as tar:
            for w in windows:
                for f in self.files[split]:
                    p = root / 'windows' / split / w / f
                    arc = f'windows/{split}/{w}/{f}'
                    add_file(tar, p, arc)
                    files.append(dict(path=arc, sha256=vc.sha256_file(p)))
                for sp in spec_files[w]:
                    spec = json.loads(sp.read_text())
                    arc = f'specs/{split}/{w}/{spec["run"]}__{sp.name}'
                    add_file(tar, sp, arc)
                    files.append(dict(path=arc, sha256=vc.sha256_file(sp)))
                    specs.append(f'{spec["run"]}__{sp.name}')
            rec = dict(schema='wm-calib-bench.wan21_vace13b.chunk.v1', package=self.name, chunk=chunk_id, split=split, windows=windows,
                       specs=len(specs), spec_names=specs, size=self.size_name, size_wh=list(self.size), files=files,
                       created=datetime.now().astimezone().isoformat(timespec='seconds'))
            add_bytes(tar, (json.dumps(rec, indent=1) + '\n').encode(), f'chunk-{chunk_id}.json')
        return dict(rec, tar=tar_path.name, tar_sha256=vc.sha256_file(tar_path), tar_bytes=tar_path.stat().st_size)

    def build_specs(self, stage: Path, root: Path, split: str, names: list, snapshot) -> dict:
        """Spec trees of the split in the stage; returns window -> spec paths (prefix, reference (DEV), s0..s15)."""
        out = stage / 'specs' / split
        runs = [mvs.build(root, split, names, 'prefix', mvs.run_name(self.size_name, split, 'prefix'), [], [], self.size_name,
                          out / mvs.run_name(self.size_name, split, 'prefix'), snapshot)]
        if split == 'dev':
            runs.append(mvs.build(root, split, names, 'recon', mvs.run_name(self.size_name, split, 'recon'), [], [], self.size_name,
                                  out / mvs.run_name(self.size_name, split, 'recon'), snapshot))
        runs.append(mvs.build(root, split, names, 'continue', mvs.run_name(self.size_name, split, 'k16'), list(range(K)), [], self.size_name,
                              out / mvs.run_name(self.size_name, split, 'k16'), snapshot))
        by_w = {w: [] for w in names}
        for paths in runs:
            for p in paths:
                by_w[json.loads(p.read_text())['window']].append(p)

        def order(p):
            s = p.stem.rsplit('-', 1)[1]
            return (0, 0) if s == 'prefix' else (1, 0) if s == 'recon' else (2, int(s[1:]))
        return {w: sorted(v, key=order) for w, v in by_w.items()}

    def outer_tar(self, out_dir: Path, split: str, members: list) -> Path:
        p = out_dir / f'{self.name}-{split}.tar'
        with tarfile.open(p, 'w') as tar:
            for rel in members:
                add_file(tar, out_dir / rel, f'{self.name}/{rel}')
        return p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--inputs', type=Path, required=True, help='VACE input root (build_vace_inputs.py)')
    ap.add_argument('--bench', type=Path, required=True, help='bench-v1 directory (names lists)')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--prompt-cache', type=Path, required=True, help='run_vace_bench.py --make-prompt-cache output (+ its .json)')
    ap.add_argument('--size', choices=sorted(vc.SIZES), default=vc.B['default_size'])
    ap.add_argument('--windows-per-chunk', type=int, default=int(vc.B['cluster']['windows_per_chunk']))
    ap.add_argument('--smoke', default='2eeaccb0-t2.5,47a9fb77-t9.5')
    ap.add_argument('--snapshot', type=Path, default=None, help='unpacked code snapshot (seed_of; default $WCB_SNAPSHOT_DIR or <repo>/snapshot)')
    args = ap.parse_args()
    vc.load_local_env()
    pkg = Package(args.size)
    if args.out.exists() and any(p for p in args.out.iterdir() if p.name != '.stage'):
        raise SystemExit(f'{args.out} exists and is not empty')
    args.out.mkdir(parents=True, exist_ok=True)
    stage = args.out / '.stage'
    shutil.rmtree(stage, ignore_errors=True)
    for d in ('chunks', 'queues', 'prompt-cache'):
        (args.out / d).mkdir(parents=True, exist_ok=True)
    cache_meta = json.loads(args.prompt_cache.with_suffix('.json').read_text())
    if vc.sha256_file(args.prompt_cache) != cache_meta['file_sha256'] or args.prompt_cache.name != CACHE_NAME:
        raise SystemExit(f'{args.prompt_cache}: not the recorded prompt cache')
    shutil.copy2(args.prompt_cache, args.out / 'prompt-cache' / CACHE_NAME)
    shutil.copy2(args.prompt_cache.with_suffix('.json'), args.out / 'prompt-cache' / (Path(CACHE_NAME).stem + '.json'))
    names = {}
    for split in ('dev', 'test'):
        f = args.bench / f'names-{split}-gen.txt'
        if vc.sha256_file(f) != NAMES_SHA256[split]:
            raise SystemExit(f'{f}: sha256 differs from protocol-v1.md')
        names[split] = f.read_text().split()
        shutil.copy2(f, args.out / f'names-{split}-gen.txt')
    smoke = args.smoke.split(',')
    if not set(smoke) <= set(names['dev']):
        raise SystemExit('smoke windows must be DEV generation windows')
    manifests = {}
    for split in ('dev', 'test'):
        wins = [pkg.check_window(args.inputs, split, w) for w in names[split]]
        spec_files = pkg.build_specs(stage, args.inputs, split, names[split], args.snapshot)
        chunks = []
        wl = names[split]
        for i in range(0, len(wl), args.windows_per_chunk):
            cid = f'{split}-c{i // args.windows_per_chunk + 1:02d}'
            chunks.append(pkg.write_chunk(args.inputs, cid, split, wl[i:i + args.windows_per_chunk], spec_files, args.out))
        if split == 'dev':
            smoke_specs = {}
            for w in smoke:
                keep = []
                for p in spec_files[w]:
                    s = p.stem.rsplit('-', 1)[1]
                    if s in ('prefix', 'recon') or (s[0] == 's' and s[1:].isdigit() and int(s[1:]) in SMOKE_SAMPLES):
                        keep.append(p)
                smoke_specs[w] = keep
            chunks.insert(0, pkg.write_chunk(args.inputs, 'dev-smoke', 'dev', smoke, smoke_specs, args.out))
        ids = [c['chunk'] for c in chunks]
        if split == 'dev':
            (args.out / 'queues' / 'dev-smoke.txt').write_text('dev-smoke\n')
            (args.out / 'queues' / 'dev.txt').write_text('\n'.join(i for i in ids if i != 'dev-smoke') + '\n')
        else:
            half = (len(ids) + 1) // 2
            (args.out / 'queues' / 'test.txt').write_text('\n'.join(ids) + '\n')
            (args.out / 'queues' / 'test-a.txt').write_text('\n'.join(ids[:half]) + '\n')
            (args.out / 'queues' / 'test-b.txt').write_text('\n'.join(ids[half:]) + '\n')
        manifests[split] = dict(schema='wm-calib-bench.wan21_vace13b.package_manifest.v1', package=pkg.name, split=split, size=pkg.size_name,
                                size_wh=list(pkg.size), windows=wins,
                                chunks=[{k: c[k] for k in ('chunk', 'windows', 'specs', 'tar', 'tar_sha256', 'tar_bytes')} for c in chunks],
                                windows_per_chunk=args.windows_per_chunk, prompt_cache=dict(file=f'prompt-cache/{CACHE_NAME}',
                                                                                            sha256=cache_meta['file_sha256'], gpu=cache_meta['gpu']),
                                builder_sha256=vc.sha256_file(Path(__file__)), generator_sha256=vc.sha256_file(HERE / 'make_vace_specs.py'),
                                adapter_sha256=vc.sha256_file(HERE / 'run_vace_bench.py'), pins_sha256=vc.sha256_file(vc.PINS_PATH),
                                created=datetime.now().astimezone().isoformat(timespec='seconds'))
        (args.out / f'manifest-{split}.json').write_text(json.dumps(manifests[split], indent=1) + '\n')
        lines = [f'{c["tar_sha256"]}  chunks/{c["tar"]}' for c in chunks]
        for c in chunks:
            lines += [f'{f["sha256"]}  {f["path"]}' for f in c['files']]
        lines.append(f'{cache_meta["file_sha256"]}  prompt-cache/{CACHE_NAME}')
        (args.out / f'MANIFEST-{split}.sha256').write_text('\n'.join(sorted(set(lines))) + '\n')
    n_dev_specs = sum(c['specs'] for c in manifests['dev']['chunks'] if c['chunk'] != 'dev-smoke')
    n_test_specs = sum(c['specs'] for c in manifests['test']['chunks'])
    readme = (HERE / 'package_README.md').read_text().format(
        package=pkg.name, created=datetime.now().astimezone().isoformat(timespec='minutes'), size=f'{pkg.size[0]} x {pkg.size[1]}',
        size_name=pkg.size_name, tag=vc.TAGS[pkg.size_name], dev_windows=len(names['dev']), test_windows=len(names['test']),
        dev_chunks=len([c for c in manifests['dev']['chunks'] if c['chunk'] != 'dev-smoke']), test_chunks=len(manifests['test']['chunks']),
        wpc=args.windows_per_chunk, smoke=', '.join(smoke), dev_specs=n_dev_specs, test_specs=n_test_specs,
        half_a=len((args.out / 'queues' / 'test-a.txt').read_text().split()), half_b=len((args.out / 'queues' / 'test-b.txt').read_text().split()),
        cache_sha=cache_meta['file_sha256'])
    (args.out / 'README.md').write_text(readme)
    common = ['README.md', 'queues/dev-smoke.txt', 'queues/dev.txt', 'queues/test.txt', 'queues/test-a.txt', 'queues/test-b.txt',
              f'prompt-cache/{CACHE_NAME}', f'prompt-cache/{Path(CACHE_NAME).stem}.json']
    for split in ('dev', 'test'):
        members = common + [f'names-{split}-gen.txt', f'manifest-{split}.json', f'MANIFEST-{split}.sha256'] + \
                  [f'chunks/{c["tar"]}' for c in manifests[split]['chunks']]
        pkg.outer_tar(args.out, split, members)
    sums = [f'{vc.sha256_file(args.out / f"{pkg.name}-{s}.tar")}  {pkg.name}-{s}.tar' for s in ('dev', 'test')]
    (args.out / 'SHA256SUMS').write_text('\n'.join(sums) + '\n')
    for s in ('dev', 'test'):
        (args.out / f'{pkg.name}-{s}.tar.sha256').write_text(next(x for x in sums if x.endswith(f'-{s}.tar')) + '\n')
    shutil.rmtree(args.out / 'chunks')                   # the chunk tars live inside the outer tars
    shutil.rmtree(args.out / 'prompt-cache')
    for split in ('dev', 'test'):
        (args.out / f'MANIFEST-{split}.sha256').unlink()  # (also inside); manifest-*.json stay next to the tars
    shutil.rmtree(stage)
    print(json.dumps(dict(out=str(args.out), sums=sums, dev_chunks=len(manifests['dev']['chunks']), test_chunks=len(manifests['test']['chunks']),
                          bytes={s: (args.out / f'{pkg.name}-{s}.tar').stat().st_size for s in ('dev', 'test')}), indent=1))


if __name__ == '__main__':
    main()
