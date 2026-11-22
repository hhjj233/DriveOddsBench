#!/usr/bin/env python3
"""Build the Orbis 2 input package for cluster from the bench-v1 windows (runs on the research server only).

For every window of names-dev-gen.txt (60) and names-test-gen.txt (200):
  windows/<split>/<w>/history.mp4  frames 0-20 of <bench>/windows-<split>/<w>/canvas/real.mp4 (t0 - 2.0 .. t0), re-encoded
                                   losslessly with the renders' encoder and checked frame for frame against the source
                                   (only those 21 frames are decoded);
  windows/<split>/<w>/real.mp4     byte copy of canvas/real.mp4 (81 frames, the input of the real reference; sha256 checked);
  specs/<split>/<w>/*.json         16 continue specs + 1 recon spec (make_orbis2_specs.py, seed table of make_bench_nano_specs).
Nothing is measured: no frame after t0 is decoded here, and no outcome is computed or summarised.

Packing: windows are grouped into chunk tars (one batch-system job each), chunks/<package>-<split>-cNN.tar, plus a DEV smoke
chunk; queues/*.txt list chunk ids for `queue chunk from ...`. The chunk tars, queue lists, names lists, manifests and
README go into one outer tar per split (orbis2-inputs-v1-dev.tar, -test.tar); both extract into the same directory.

  build_input_package.py --bench DIR --out DIR [--windows-per-chunk 4] [--smoke-names FILE --smoke-count 2] [--workers 6]
                         [--snapshot DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import subprocess
import sys
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_orbis2_specs as mos  # noqa: E402

PACKAGE = 'orbis2-inputs-v1'
NAMES_SHA256 = {'dev': '1e7b3878a485fb45fa295dddd29e63237f72a7423230cb0de7c012808f47fbd1',   # protocol-v1.md section 1
                'test': 'e8343f526ed2e25112af41a63a45a9866171f5b74d56cff7663dd1c2db086233'}
W, H, FPS, HIST, FRAMES = 512, 288, 10, 21, 81


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


def ffprobe(path: Path) -> dict:
    out = subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames', '-show_entries',
                                   'stream=width,height,nb_read_frames,r_frame_rate,pix_fmt,codec_name', '-of', 'json', str(path)], text=True)
    s = json.loads(out)['streams'][0]
    num, den = s['r_frame_rate'].split('/')
    return dict(width=int(s['width']), height=int(s['height']), frames=int(s['nb_read_frames']), fps=float(num) / float(den), pix_fmt=s['pix_fmt'], codec=s['codec_name'])


def decode_first(path: Path, n: int, threads: int = 2) -> np.ndarray:
    raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-threads', str(threads), '-i', str(path), '-frames:v', str(n), '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'])
    return np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3)


def write_video(frames: np.ndarray, path: Path, threads: int = 2) -> None:
    """make_wm_calibration_windows.write_video (libx264rgb -crf 0 -preset medium, rgb24, 10 fps), with -threads."""
    proc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                             '-c:v', 'libx264rgb', '-crf', '0', '-preset', 'medium', '-threads', str(threads), '-pix_fmt', 'rgb24', '-r', str(FPS),
                             '-video_track_timescale', '10240', '-map_metadata', '-1', '-movflags', '+faststart', str(path)], stdin=subprocess.PIPE)
    for f in frames:
        proc.stdin.write(np.ascontiguousarray(f).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f'ffmpeg failed for {path}')


def one_window(job):
    bench, stage, split, w = job
    src_dir = Path(bench) / f'windows-{split}' / w
    render = json.loads((src_dir / 'render.json').read_text())
    if not render.get('ok') or render.get('render_version') != 2:
        raise RuntimeError(f'{w}: render.json not ok / not render_version 2')
    src = src_dir / 'canvas' / 'real.mp4'
    info = ffprobe(src)
    if (info['width'], info['height'], info['frames']) != (W, H, FRAMES) or abs(info['fps'] - FPS) > 1e-9:
        raise RuntimeError(f'{w}: canvas/real.mp4 is {info}')
    dst = Path(stage) / 'windows' / split / w
    dst.mkdir(parents=True, exist_ok=True)
    hist = decode_first(src, HIST)
    if hist.shape != (HIST, H, W, 3):
        raise RuntimeError(f'{w}: decoded {hist.shape}')
    write_video(hist, dst / 'history.mp4')
    back = decode_first(dst / 'history.mp4', HIST + 5)
    if not np.array_equal(back, hist):
        raise RuntimeError(f'{w}: history.mp4 does not decode to frames 0-20 of canvas/real.mp4')
    hinfo = ffprobe(dst / 'history.mp4')
    shutil.copyfile(src, dst / 'real.mp4')
    src_sha, dst_sha = sha256_file(src), sha256_file(dst / 'real.mp4')
    if src_sha != dst_sha:
        raise RuntimeError(f'{w}: copy of canvas/real.mp4 differs')
    return dict(window=w, split=split, source=f'windows-{split}/{w}/canvas/real.mp4', source_sha256=src_sha, render_version=render['render_version'],
                history=dict(path=f'windows/{split}/{w}/history.mp4', sha256=sha256_file(dst / 'history.mp4'), frames=hinfo['frames'], fps=hinfo['fps'],
                             size=[hinfo['width'], hinfo['height']], pix_fmt=hinfo['pix_fmt'], frames_sha256=hashlib.sha256(hist.tobytes()).hexdigest(),
                             derived_from='canvas/real.mp4 frames 0-20, decoded to rgb24 and re-encoded losslessly; decoded frames verified identical'),
                real=dict(path=f'windows/{split}/{w}/real.mp4', sha256=dst_sha, frames=info['frames'], fps=info['fps'], size=[W, H], pix_fmt=info['pix_fmt']))


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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bench', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True, help=f'package directory (e.g. .../cluster-packages/{PACKAGE})')
    ap.add_argument('--stage', type=Path, help='scratch input root (default <out>/.stage, removed at the end)')
    ap.add_argument('--windows-per-chunk', type=int, default=4)
    ap.add_argument('--smoke-names', type=Path, help='DEV names to draw the smoke chunk from (default <bench>/names-smoke-dev.txt)')
    ap.add_argument('--smoke-count', type=int, default=2)
    ap.add_argument('--test-queues', type=int, default=2, help='split the TEST chunks into this many queue files (/home quota batches)')
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--keep-stage', action='store_true')
    ap.add_argument('--snapshot', type=Path, help='unpacked code snapshot (default $WCB_SNAPSHOT_DIR or <repo>/snapshot)')
    args = ap.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f'{args.out} exists and is not empty')
    args.out.mkdir(parents=True, exist_ok=True)
    stage = args.stage or args.out / '.stage'
    stage.mkdir(parents=True, exist_ok=True)
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

    jobs = [(str(args.bench), str(stage), split, w) for split in ('dev', 'test') for w in names[split]]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        windows = list(pool.map(one_window, jobs, chunksize=4))
    print(json.dumps(dict(windows=len(windows), seconds=round(time.time() - t_start))), flush=True)

    spec_counts = {}
    for split in ('dev', 'test'):
        specs = mos.build(stage, split, names[split], list(range(16)), True, f'orbis2-{split}', stage / 'specs' / split, args.snapshot)
        spec_counts[split] = len(specs)

    # chunks
    by_window = {(r['split'], r['window']): r for r in windows}
    chunks = []
    for split in ('dev', 'test'):
        ws = names[split]
        for i in range(0, len(ws), args.windows_per_chunk):
            chunks.append(dict(id=f'{split}-c{i // args.windows_per_chunk + 1:02d}', split=split, windows=ws[i:i + args.windows_per_chunk]))
    chunks.append(dict(id='dev-smoke', split='dev', windows=smoke))
    (args.out / 'chunks').mkdir(exist_ok=True)
    inner = []  # (relative path in the input root, sha256, bytes)
    for c in chunks:
        tar_path = args.out / 'chunks' / f"{PACKAGE}-{c['id']}.tar"
        files = []
        for w in c['windows']:
            for p in sorted((stage / 'windows' / c['split'] / w).iterdir()) + sorted((stage / 'specs' / c['split'] / w).iterdir()):
                rel = str(p.relative_to(stage))
                files.append(dict(path=rel, sha256=sha256_file(p), bytes=p.stat().st_size))
        c['files'] = files
        c['specs'] = sum(1 for f in files if f['path'].startswith('specs/'))
        chunk_json = json.dumps(dict(schema='wm-calib-bench.orbis2.chunk.v1', package=PACKAGE, **c), indent=1).encode() + b'\n'
        with tarfile.open(tar_path, 'w', format=tarfile.PAX_FORMAT) as tar:
            add_bytes(tar, chunk_json, f"chunk-{c['id']}.json")
            for f in files:
                add_file(tar, stage / f['path'], f['path'])
        c['tar'] = dict(path=f'chunks/{tar_path.name}', sha256=sha256_file(tar_path), bytes=tar_path.stat().st_size)
        if c['id'] != 'dev-smoke':
            inner += [(f['path'], f['sha256'], f['bytes']) for f in files]

    # queue lists
    (args.out / 'queues').mkdir(exist_ok=True)
    dev_ids = [c['id'] for c in chunks if c['split'] == 'dev' and c['id'] != 'dev-smoke']
    test_ids = [c['id'] for c in chunks if c['split'] == 'test']
    queues = {'dev-smoke': ['dev-smoke'], 'dev': dev_ids}
    per = -(-len(test_ids) // args.test_queues)
    for q in range(args.test_queues):
        queues[f'test-{chr(ord("a") + q)}'] = test_ids[q * per:(q + 1) * per]
    for q, ids in queues.items():
        (args.out / 'queues' / f'{q}.txt').write_text(''.join(f'{i}\n' for i in ids))

    # manifests
    spec_manifests = {s: json.loads((stage / 'specs' / s / 'manifest.json').read_text()) for s in ('dev', 'test')}
    summary = dict(schema='wm-calib-bench.orbis2.input_package.v1', package=PACKAGE, created=datetime.now().astimezone().isoformat(timespec='seconds'),
                   source_bench='bench-v1', names_sha256=NAMES_SHA256, windows={s: len(names[s]) for s in names}, specs=spec_counts,
                   smoke_windows=smoke, windows_per_chunk=args.windows_per_chunk, queues={q: len(v) for q, v in queues.items()},
                   builder_sha256=sha256_file(Path(__file__)), spec_generator_sha256=sha256_file(HERE / 'make_orbis2_specs.py'),
                   pins_sha256=sha256_file(HERE / 'orbis2_pins.json'), spec_manifests=spec_manifests,
                   chunks=[{k: c[k] for k in ('id', 'split', 'windows', 'specs', 'tar')} for c in chunks], window_files=windows)
    for split in ('dev', 'test'):
        part = dict(summary, chunks=[x for x in summary['chunks'] if x['split'] == split],
                    window_files=[r for r in windows if r['split'] == split], spec_manifests={split: spec_manifests[split]})
        (args.out / f'manifest-{split}.json').write_text(json.dumps(part, indent=1) + '\n')
        lines = [f'{sha}  {p}' for p, sha, _ in sorted(x for x in inner if x[0].split('/')[1] == split)]
        lines += [f"{c['tar']['sha256']}  {c['tar']['path']}" for c in chunks if c['split'] == split]
        (args.out / f'MANIFEST-{split}.sha256').write_text('\n'.join(lines) + '\n')
        (args.out / f'names-{split}-gen.txt').write_bytes((stage / f'names-{split}-gen.txt').read_bytes())

    readme = (HERE / 'package_README.md').read_text().format(
        created=summary['created'], dev_windows=len(names['dev']), test_windows=len(names['test']), dev_chunks=len(dev_ids),
        test_chunks=len(test_ids), wpc=args.windows_per_chunk, smoke_count=len(smoke), smoke=', '.join(smoke))
    (args.out / 'README.md').write_text(readme)

    # outer tars, one per split
    for split in ('dev', 'test'):
        outer = args.out / f'{PACKAGE}-{split}.tar'
        members = ['README.md', f'manifest-{split}.json', f'MANIFEST-{split}.sha256', f'names-{split}-gen.txt']
        members += [f'queues/{q}.txt' for q in queues if q.startswith(split)]
        members += [c['tar']['path'] for c in chunks if c['split'] == split]
        with tarfile.open(outer, 'w', format=tarfile.PAX_FORMAT) as tar:
            for m in members:
                add_file(tar, args.out / m, f'{PACKAGE}/{m}')
        digest = sha256_file(outer)
        (args.out / f'{outer.name}.sha256').write_text(f'{digest}  {outer.name}\n')
    sums = [(args.out / f'{PACKAGE}-{s}.tar.sha256').read_text() for s in ('dev', 'test')]
    (args.out / 'SHA256SUMS').write_text(''.join(sums))
    # the chunk tars live inside the outer tars; the small manifests and queue lists stay loose for inspection
    shutil.rmtree(args.out / 'chunks')
    if not args.keep_stage and stage == args.out / '.stage':
        shutil.rmtree(stage)
    print(json.dumps(dict(package=str(args.out), seconds=round(time.time() - t_start),
                          tars={s: (args.out / f'{PACKAGE}-{s}.tar').stat().st_size for s in ('dev', 'test')})))


if __name__ == '__main__':
    main()
