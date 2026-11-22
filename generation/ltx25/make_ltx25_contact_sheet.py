#!/usr/bin/env python3
"""Contact sheet of LTX-2.5 DEV clips for eyeballing (CPU; numpy + Pillow). AV2 frames carry no face or plate blurring:
the sheet stays in private storage, never in git or on a public page.

One block per window: rows = clips (the raw real canvas clip, the reference, the continuations, the rerun), columns =
times after t0 read from the 10 Hz vision.mp4 on the 512 x 288 canvas (frame 20 = t0; distilled clips are already
INTER_AREA-downsized from 1024 x 576). Tiles are half size (256 x 144) with a white t0 column border.

  make_ltx25_contact_sheet.py --bench BENCH_V1_DIR --output-root SMOKE_DIR --windows W1 W2 [--samples s0 s1] [--rerun-run NAME]
                              --out SHEET.png [--times -2 -1 -0.5 0 0.5 1 2 3 4 5 6]
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

W, H, TW, TH, LABEL_W, PAD = 512, 288, 256, 144, 150, 4


def decode(path: Path) -> np.ndarray:
    raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-threads', '2', '-i', str(path), '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'])
    return np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3)


def find(output_root: Path, name: str, rerun_prefix: str, want_rerun: bool) -> Path | None:
    """vision.mp4 of the successful sample `name`, from the rerun run (want_rerun) or from any other run."""
    for f in sorted(output_root.rglob('sample_outputs.json')):
        r = json.loads(f.read_text())
        if r.get('name') == name and r.get('status') == 'success' and r.get('run', '').startswith(rerun_prefix) == want_rerun:
            return f.parent / 'vision.mp4'
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bench', type=Path, required=True)
    ap.add_argument('--output-root', type=Path, required=True)
    ap.add_argument('--windows', nargs='+', required=True)
    ap.add_argument('--samples', nargs='+', default=['s0', 's1'])
    ap.add_argument('--rerun-run', default='ltx25d-dev-smoke-rerun', help='run-name prefix of the same-seed rerun (dev variant: ltx25-dev-smoke-rerun)')
    ap.add_argument('--times', nargs='+', type=float, default=[-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    cols = [int(round(20 + 10 * t)) for t in args.times]
    blocks = []
    for w in args.windows:
        rows = [('real (raw canvas)', args.bench / 'windows-dev' / w / 'canvas' / 'real.mp4'),
                ('reference (LTX VAE)', find(args.output_root, f'{w}-recon', args.rerun_run, False))]
        rows += [(f'{s} (generated)', find(args.output_root, f'{w}-{s}', args.rerun_run, False)) for s in args.samples]
        rr = find(args.output_root, f'{w}-s0', args.rerun_run, True)
        if rr is not None:
            rows.append(('s0 rerun (same seed)', rr))
        rows = [(lab, p) for lab, p in rows if p is not None and p.is_file()]
        img = Image.new('RGB', (LABEL_W + len(cols) * (TW + PAD), 26 + len(rows) * (TH + PAD)), (20, 20, 20))
        d = ImageDraw.Draw(img)
        d.text((6, 6), w, fill=(255, 255, 255))
        for j, t in enumerate(args.times):
            d.text((LABEL_W + j * (TW + PAD) + 4, 6), f't = {t:+.1f} s', fill=(255, 255, 0) if t == 0 else (220, 220, 220))
        for i, (lab, p) in enumerate(rows):
            v = decode(p)
            y = 26 + i * (TH + PAD)
            d.text((6, y + TH // 2 - 6), lab, fill=(255, 255, 255))
            for j, c in enumerate(cols):
                tile = Image.fromarray(v[c]).resize((TW, TH), Image.BILINEAR)
                x = LABEL_W + j * (TW + PAD)
                img.paste(tile, (x, y))
                if args.times[j] == 0:
                    d.rectangle([x - 1, y - 1, x + TW, y + TH], outline=(255, 255, 255))
        blocks.append(img)
    sheet = Image.new('RGB', (max(b.width for b in blocks), sum(b.height + 10 for b in blocks)), (0, 0, 0))
    y = 0
    for b in blocks:
        sheet.paste(b, (0, y))
        y += b.height + 10
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out)
    print(json.dumps(dict(out=str(args.out), size=list(sheet.size), windows=args.windows)))


if __name__ == '__main__':
    main()
