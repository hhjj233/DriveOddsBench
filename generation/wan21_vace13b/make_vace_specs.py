#!/usr/bin/env python3
"""Wan2.1-VACE-1.3B sample specs for bench-v1 (one JSON per sample), every path relative to an input root
(build_vace_inputs.py: windows/<split>/<window>/{history16_TAG.mp4, real16_TAG.mp4, frames16_TAG.json}, TAG 512 / 832).

  --mode prefix    one spec per window: the 33 history frames through the Wan VAE (frames 0-20 of every clip of the
                   window are copied from it); name <window>-prefix; no randomness
  --mode continue  one spec per window and sample k (--samples, default 0-15): the window's history16_TAG.mp4, seeds
                   seed_of(window, k) (round 1) and seed_of(window, f'{k}|r2') (round 2); name <window>-s<k>
  --mode recon     one spec per window: the real reference from real16_TAG.mp4 (no randomness); name <window>-recon.
                   TEST references only after the freeze (--frozen-file: the spec carries the frozen_reference mark)
Specs go to <out>/<run>/<name>.json (default <input-root>/specs/<split>/<run>/).

seed_of() is imported from the read-only code snapshot of make_bench_nano_specs.py (package orbis2-code-snapshot-v1,
unpacked anywhere: --snapshot / $WCB_SNAPSHOT_DIR), after checking its sha256 against the pinned value and the snapshot's
MANIFEST.sha256: the seed of sample k of window w is the first 4 bytes of sha256("bench-v1|w|k"), big-endian, masked to 31
bits, one table for every model of the benchmark. The prompt is the Nano prompt "long" of the same file, verbatim (a fixed
generic description: no motion, no events, no future).

  make_vace_specs.py --input-root DIR --split dev|test --names FILE --mode prefix|continue|recon --run NAME
                     [--size canvas|480p] [--samples 0-15] [--keep native16,latents] [--out DIR] [--snapshot DIR]
                     [--frozen-file PATH]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_vace_bench as rvb  # noqa: E402  (stdlib at import: SETTINGS, NEGATIVE_PROMPT, SPEC_SCHEMA)
import vace_common as vc  # noqa: E402


def parse_range(text: str) -> list[int]:
    out = []
    for part in text.split(','):
        lo, _, hi = part.partition('-')
        out += list(range(int(lo), int(hi or lo) + 1))
    return out


def run_name(size_name: str, split: str, what: str) -> str:
    return f'vace{vc.TAGS[size_name]}-{split}-{what}'


def build(input_root: Path, split: str, names: list, mode: str, run: str, samples: list, keep: list, size_name: str,
          out: Path, snapshot=None, frozen_file: Path | None = None) -> list:
    """Write the specs of one run into out/ (one JSON per sample); returns the spec paths. Used by main() and by
    build_input_package.py (the cluster package)."""
    seed_of, prompts, nano_sha = vc.load_seed_of(snapshot)
    prompt = prompts['long']
    out.mkdir(parents=True, exist_ok=True)
    size = vc.SIZES[size_name]
    files = vc.input_names(size_name)
    settings = dict(rvb.SETTINGS, width=size[0], height=size[1])
    if mode == 'recon' and split == 'test' and not (frozen_file and frozen_file.is_file()):
        raise SystemExit('TEST reference specs need --frozen-file (the frozen VACE protocol)')
    written = []
    for w in names:
        wd = input_root / 'windows' / split / w
        fr = json.loads((wd / files['frames']).read_text())
        base = dict(schema=rvb.SPEC_SCHEMA, run=run, mode=mode, split=split.upper(), window=w, prompt_key='long', prompt=prompt,
                    prompt_sha256=vc.sha256_text(prompt), negative_prompt_sha256=vc.sha256_text(rvb.NEGATIVE_PROMPT),
                    settings=settings, keep=list(keep), size=dict(name=size_name, wh=list(size)),
                    timeline=dict(rule=vc.B['timeline_rule'], tick_rule=vc.B['tick_rule'], round_rule=vc.B['round_rule'],
                                  frames16_sha256=vc.sha256_file(wd / files['frames'])),
                    generator=dict(script_sha256=vc.sha256_file(Path(__file__)), adapter_sha256=vc.sha256_file(HERE / 'run_vace_bench.py'),
                                   common_sha256=vc.sha256_file(HERE / 'vace_common.py'), pins_sha256=vc.sha256_file(vc.PINS_PATH),
                                   nano_specs_sha256=nano_sha, seed_rule=vc.B['seed_rule']),
                    written=datetime.now(timezone.utc).isoformat(timespec='seconds'))
        hist = dict(path=f'windows/{split}/{w}/{files["history"]}', sha256=fr['files'][files['history']]['sha256'])
        if mode == 'continue':
            for k in samples:
                spec = dict(base, name=f'{w}-s{k}', sample=k, seed=seed_of(w, k), seed_round2=seed_of(w, f'{k}|r2'), inputs=dict(history=hist))
                p = out / f'{spec["name"]}.json'
                p.write_text(json.dumps(spec, indent=1, ensure_ascii=False) + '\n'); written.append(p)
        elif mode == 'prefix':
            spec = dict(base, name=f'{w}-prefix', sample='prefix', seed=None, seed_round2=None, inputs=dict(history=hist))
            p = out / f'{spec["name"]}.json'
            p.write_text(json.dumps(spec, indent=1, ensure_ascii=False) + '\n'); written.append(p)
        else:
            if files['real'] not in fr['files']:
                raise SystemExit(f'{w}: no {files["real"]} (TEST real clips exist only after the freeze)')
            spec = dict(base, name=f'{w}-recon', sample='recon', seed=None, seed_round2=None,
                        inputs=dict(history=hist, real=dict(path=f'windows/{split}/{w}/{files["real"]}', sha256=fr['files'][files['real']]['sha256'])))
            if split == 'test':
                spec['frozen_reference'] = dict(file=frozen_file.name, sha256=vc.sha256_file(frozen_file))
            p = out / f'{spec["name"]}.json'
            p.write_text(json.dumps(spec, indent=1, ensure_ascii=False) + '\n'); written.append(p)
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input-root', type=Path, required=True); ap.add_argument('--split', choices=('dev', 'test'), required=True)
    ap.add_argument('--names', type=Path, required=True); ap.add_argument('--mode', choices=('prefix', 'continue', 'recon'), required=True)
    ap.add_argument('--run', default=None, help='run name (default vace<TAG>-<split>-{prefix,k16,recon})'); ap.add_argument('--samples', default='0-15')
    ap.add_argument('--keep', default=''); ap.add_argument('--out', type=Path, default=None); ap.add_argument('--snapshot', type=Path, default=None)
    ap.add_argument('--size', choices=sorted(vc.SIZES), default=vc.B['default_size'])
    ap.add_argument('--frozen-file', type=Path, default=None)
    a = ap.parse_args()
    vc.load_local_env()
    names = [x.strip() for x in a.names.read_text().split() if x.strip()]
    run = a.run or run_name(a.size, a.split, {'prefix': 'prefix', 'continue': 'k16', 'recon': 'recon'}[a.mode])
    out = (a.out or a.input_root / 'specs' / a.split) / run
    written = build(a.input_root, a.split, names, a.mode, run, parse_range(a.samples), [k for k in a.keep.split(',') if k], a.size, out,
                    a.snapshot, a.frozen_file)
    print(f'{len(written)} specs -> {out}')


if __name__ == '__main__':
    main()
