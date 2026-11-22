#!/usr/bin/env python3
"""Build the LTX-2.5 distilled DEV diagnostics package for cluster (ltx25d-diag-v1, queue dev-diag) on the research server, AFTER
the DEV runs are collected (cluster/ltx25/collect_outputs.py: <runs>/ltx25d-dev-recon and <runs>/ltx25d-dev-k16).

Nothing is generated again. Every diagnostic re-decodes a saved final latent (latents.safetensors: video_latent, and the
decoder state right before the DiffVAE call) with the decode call of the generation runs (run_ltx25_bench.py, mode redecode).
Windows: the 10 DEV windows with the smallest sha256('bench-v1|ltx25d-diag|' + window) (pins variants.distilled.diag), a rule
fixed before any DEV clip exists. Specs (212 for 10 windows and 3 draws):
  ltx25d-dev-recon-draws/<w>-recon-d<j>   DiffVAE on the reference latent, a fresh generator seeded seed_of(w, 'recon-d<j>'),
                                          j = 1..3: the decoder noise floor (the names and seeds of make_ltx25_specs --recon-draws)
  ltx25d-dev-redecode/<w>-recon-rep       DiffVAE on the reference latent with the saved decoder state, and the same for s0 of
  ltx25d-dev-redecode/<w0>-s0-rep         the first window: must reproduce the DEV clips bit for bit
  ltx25d-dev-conv-recon/<w>-recon         conv decoder on the reference latent, saved decoder state
  ltx25d-dev-conv-k16/<w>-s<k>            conv decoder on the 16 continuations' final latents, each with its saved state
  ltx25d-dev-conv-probe/<w0>-recon-probe  conv decoder on the first window's reference with seed_of(w0, 'conv-probe'): identical
                                          to its ltx25d-dev-conv-recon clip = the conv decoder ignores the generator
Package (DEV only: any source record that is not a DEV distilled bf16 sample stops the build):
  <out>/ltx25d-diag-v1-dev-diag.tar  the chunk: latents/<run>/<name>.safetensors (170 files, about 0.55 GB),
                                     specs/dev/<w>/<name>.json, chunk-dev-diag.json (windows, every file's sha256, spec count)
  <out>/queues/dev-diag.txt, manifest.json (selection rule, windows, sources, runs, seed table), README.md, SHA256SUMS

  build_diag_package.py --runs DIR --bench BENCH_V1_DIR --out DIR [--windows 10] [--draws 3] [--snapshot DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_input_package as bip  # noqa: E402
import make_ltx25_specs as mls  # noqa: E402

DIAG = mls.PINS['variants']['distilled']['diag']
PACKAGE, CHUNK = DIAG['package'], DIAG['queue']           # ltx25d-diag-v1, dev-diag
RUN_RECON, RUN_K16 = 'ltx25d-dev-recon', 'ltx25d-dev-k16'
RUNS = dict(draws='ltx25d-dev-recon-draws', rep='ltx25d-dev-redecode', conv_recon='ltx25d-dev-conv-recon', conv_k16='ltx25d-dev-conv-k16',
            probe='ltx25d-dev-conv-probe')
K = 16


def rank(window: str) -> str:
    return hashlib.sha256(f'bench-v1|ltx25d-diag|{window}'.encode()).hexdigest()


def select(names: list[str], n: int) -> list[str]:
    """The pinned window rule: the n DEV windows with the smallest sha256('bench-v1|ltx25d-diag|' + window)."""
    return sorted(names, key=rank)[:n]


def source(runs: Path, run: str, name: str) -> dict:
    """A collected DEV sample: status success, DEV, distilled bf16, its latents.safetensors and vision.mp4 as recorded."""
    d = runs / run / 'native' / name
    rec = json.loads((d / 'sample_outputs.json').read_text())
    if rec.get('split') != 'dev':
        raise SystemExit(f'{d}: split {rec.get("split")}; the diagnostics are DEV only')
    if rec.get('status') != 'success' or rec.get('variant') != 'distilled' or rec.get('quantization') != 'none':
        raise SystemExit(f'{d}: status {rec.get("status")}, variant {rec.get("variant")}, quantization {rec.get("quantization")}')
    lat, vis = d / 'latents.safetensors', d / 'vision.mp4'
    lsha = bip.sha256_file(lat)
    if lsha != rec['files']['latents.safetensors']['sha256']:
        raise SystemExit(f'{lat}: sha256 differs from its record')
    if bip.sha256_file(vis) != rec['video_sha256']:
        raise SystemExit(f'{vis}: sha256 differs from its record')
    return dict(run=run, name=name, mode=rec['mode'], k=rec.get('k'), seed=rec['seed'], latents_sha256=lsha,
                full_res_frames_sha256=rec['full_res_frames_sha256'], vision_frames_sha256=rec['files']['vision.mp4']['frames_sha256'],
                video_sha256=rec['video_sha256'], gpu=(rec.get('gpu') or {}).get('name'), latents_file=lat)


def make_spec(window: str, name: str, run: str, decoder: str, rng: str, seed: int | None, seed_rule: str, src: dict) -> dict:
    rel = f'latents/{src["run"]}/{src["name"]}.safetensors'
    return dict(schema=mls.SCHEMA, benchmark='bench-v1', model='ltx25d', variant='distilled', split='dev', window=window, mode='redecode', k=src['k'],
                seed=seed, seed_rule=seed_rule, rng=rng, decoder=decoder, name=name, run=run,
                source={k: v for k, v in src.items() if k != 'latents_file'},
                inputs=dict(latents=dict(path=rel, sha256=src['latents_sha256'])), output_dir=f'{run}/native/{name}',
                generation=mls.decode_block('distilled', decoder))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', type=Path, required=True, help='collected DEV runs (parent of ltx25d-dev-recon/ and ltx25d-dev-k16/)')
    ap.add_argument('--bench', type=Path, required=True, help='bench-v1 directory (names-dev-gen.txt)')
    ap.add_argument('--out', type=Path, required=True, help=f'package directory (e.g. .../cluster-packages/{PACKAGE})')
    ap.add_argument('--windows', type=int, default=DIAG['windows'])
    ap.add_argument('--draws', type=int, default=DIAG['recon_draws'])
    ap.add_argument('--snapshot', type=Path, help='unpacked code snapshot (default $WCB_SNAPSHOT_DIR or <repo>/snapshot)')
    args = ap.parse_args()
    names_file = args.bench / 'names-dev-gen.txt'
    if bip.sha256_file(names_file) != bip.NAMES_SHA256['dev']:
        raise SystemExit(f'{names_file}: sha256 is not the pinned DEV list')
    names = [x.strip() for x in names_file.read_text().splitlines() if x.strip()]
    windows = select(names, args.windows)
    seed_of, nano_sha = mls.load_seed_of(args.snapshot)
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f'{args.out} is not empty')
    stage = args.out / f'.build-{os.getpid()}'
    (stage / 'specs' / 'dev').mkdir(parents=True)

    specs, sources = [], {}
    for i, w in enumerate(windows):
        ref = source(args.runs, RUN_RECON, f'{w}-recon')
        conts = [source(args.runs, RUN_K16, f'{w}-s{k}') for k in range(K)]
        sources[w] = [ref] + conts
        for j in range(1, args.draws + 1):
            specs.append(make_spec(w, f'{w}-recon-d{j}', RUNS['draws'], 'diffvae', 'seed', int(seed_of(w, f'recon-d{j}')),
                                   f"seed_of(window, 'recon-d{j}'): an extra DiffVAE draw of the saved reference latent (decoder noise floor; not in the analysis set)", ref))
        specs.append(make_spec(w, f'{w}-recon-rep', RUNS['rep'], 'diffvae', 'saved', None,
                               'the saved decoder state of the reference (reproduction check)', ref))
        specs.append(make_spec(w, f'{w}-recon', RUNS['conv_recon'], 'conv', 'saved', None, 'the saved decoder state of the reference', ref))
        for c in conts:
            specs.append(make_spec(w, c['name'], RUNS['conv_k16'], 'conv', 'saved', None, 'the saved decoder state of the continuation', c))
        if i == 0:
            specs.append(make_spec(w, f'{w}-s0-rep', RUNS['rep'], 'diffvae', 'saved', None,
                                   'the saved decoder state of s0 (reproduction check of a continuation)', conts[0]))
            specs.append(make_spec(w, f'{w}-recon-probe', RUNS['probe'], 'conv', 'seed', int(seed_of(w, 'conv-probe')),
                                   "seed_of(window, 'conv-probe'): the conv decoder with another generator (determinism probe)", ref))
    seeds = [s['seed'] for s in specs if s['rng'] == 'seed']
    if len(set(seeds)) != len(seeds):
        raise SystemExit('two seeded diagnostics share a seed')
    if (args.windows, args.draws) == (DIAG['windows'], DIAG['recon_draws']) and len(specs) != DIAG['specs']:
        raise SystemExit(f'{len(specs)} specs, the pins say {DIAG["specs"]}')

    files = []
    for w, srcs in sources.items():
        for src in srcs:
            rel = f'latents/{src["run"]}/{src["name"]}.safetensors'
            (stage / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src['latents_file'], stage / rel)
            if bip.sha256_file(stage / rel) != src['latents_sha256']:
                raise SystemExit(f'copy of {rel} differs')
            files.append(dict(path=rel, sha256=src['latents_sha256'], bytes=(stage / rel).stat().st_size))
    for s in specs:
        p = stage / 'specs' / 'dev' / s['window'] / f'{s["name"]}.json'
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(s, indent=1) + '\n')
        files.append(dict(path=str(p.relative_to(stage)), sha256=bip.sha256_file(p), bytes=p.stat().st_size))
    runs = {}
    for s in specs:
        runs[s['run']] = runs.get(s['run'], 0) + 1
    chunk = dict(schema='wm-calib-bench.ltx25d.diag_chunk.v1', id=CHUNK, package=PACKAGE, split='dev', windows=windows, specs=len(specs), runs=runs,
                 files=sorted(files, key=lambda f: f['path']))
    (stage / f'chunk-{CHUNK}.json').write_text(json.dumps(chunk, indent=1) + '\n')

    tar_path = args.out / f'{PACKAGE}-{CHUNK}.tar'
    with tarfile.open(tar_path, 'w') as tar:
        bip.add_file(tar, stage / f'chunk-{CHUNK}.json', f'chunk-{CHUNK}.json')
        for f in chunk['files']:
            bip.add_file(tar, stage / f['path'], f['path'])
    shutil.rmtree(stage)
    (args.out / 'queues').mkdir()
    (args.out / 'queues' / f'{CHUNK}.txt').write_text(CHUNK + '\n')
    gpus = sorted({src['gpu'] or '?' for srcs in sources.values() for src in srcs})
    manifest = dict(schema='wm-calib-bench.ltx25d.diag_package.v1', package=PACKAGE, created=datetime.now().astimezone().isoformat(timespec='seconds'),
                    builder_sha256=bip.sha256_file(Path(__file__)), pins_sha256=bip.sha256_file(HERE / 'ltx25_pins.json'),
                    window_rule=DIAG['window_rule'], windows=windows, draws=args.draws, specs=len(specs), runs=runs, dev_gpus=gpus,
                    seed_source_sha256=nano_sha, chunk_tar=dict(name=tar_path.name, sha256=bip.sha256_file(tar_path), bytes=tar_path.stat().st_size),
                    seed_table=[[s['name'], s['seed']] for s in specs if s['rng'] == 'seed'],
                    sources={w: [{k: v for k, v in src.items() if k != 'latents_file'} for src in srcs] for w, srcs in sources.items()},
                    note='decode-only DEV diagnostics: nothing is generated again; run on the GPU type of the DEV runs (dev_gpus)')
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    readme = (HERE / 'diag_package_README.md').read_text().format(created=manifest['created'], windows=', '.join(windows), specs=len(specs),
                                                                   tar=tar_path.name, gpus=', '.join(gpus))
    (args.out / 'README.md').write_text(readme)
    sums = [f'{bip.sha256_file(args.out / rel)}  {rel}' for rel in (tar_path.name, 'manifest.json', 'README.md', f'queues/{CHUNK}.txt')]
    (args.out / 'SHA256SUMS').write_text('\n'.join(sums) + '\n')
    print(json.dumps(dict(package=str(args.out), windows=len(windows), specs=len(specs), runs=runs, tar_bytes=tar_path.stat().st_size, dev_gpus=gpus)))


if __name__ == '__main__':
    main()
