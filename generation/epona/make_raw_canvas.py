#!/usr/bin/env python3
"""DEV raw (codec-free) clips of the Epona band on the evaluation canvas, for the shared canvas-instrument work (DEV only).

For every DEV window of the Epona input root (band [579, 1451), build_epona_inputs.py), writes <out>/<window>/:
  raw-band.mp4       41 frames (k = -20..60 step 2, 5 fps): real.mp4 (the band at 1024x512) mapped to the canvas exactly as
                     the adapter maps Epona's frames (run_epona_bench.canvas_map / Epona.to_canvas: model rows [128, 512)
                     -> canvas rows [0, 216) with cv2.INTER_AREA, canvas rows [216, 288) black). The same pipeline as the
                     real reference without the DCAE: raw vs reference isolates the codec.
  canvas-even.mp4    41 frames: bench-v1 canvas/real.mp4 at the even ticks with canvas rows [216, 288) set to black (the
                     bench's direct render of the canvas; differs from raw-band.mp4 only by the resampling path)
  raw.json           ticks, mappings, frame and file hashes
Both are lossless (libx264rgb -crf 0, preset ultrafast) and decoded back for checking.

  make_raw_canvas.py --inputs ROOT --bench DIR --names FILE --out DIR
"""
from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_epona_bench as rb  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--inputs', type=Path, required=True)
    ap.add_argument('--bench', type=Path, required=True)
    ap.add_argument('--names', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    fake = types.SimpleNamespace(np=np)
    done = 0
    for w in args.names.read_text().split():
        b = json.loads((args.inputs / 'windows' / 'dev' / w / 'build.json').read_text())
        if b['split'] != 'dev' or b.get('history_only'):
            raise SystemExit(f'{w}: DEV windows with real.mp4 only')
        cmap = rb.canvas_map(b['render']['model_band_rows'])
        real, _ = rb.decode_rgb(args.inputs / 'windows' / 'dev' / w / 'real.mp4', 2)
        raw = rb.Epona.to_canvas(fake, real, cmap)
        canvas, _ = rb.decode_rgb(args.bench / 'windows-dev' / w / 'canvas' / 'real.mp4', 2)
        even = canvas[0::2].copy()
        even[:, cmap['canvas_rows'][1]:] = 0
        if raw.shape != (41, 288, 512, 3) or even.shape != raw.shape:
            raise SystemExit(f'{w}: shapes {raw.shape} {even.shape}')
        od = args.out / w
        od.mkdir(parents=True, exist_ok=True)
        rec = dict(schema='wm-calib-bench.epona.raw_canvas.v1', window=w, split='dev', frame_ticks_k=rb.TICKS, fps=rb.FPS, canvas_map=cmap,
                   sources=dict(real=b['clips']['real.mp4']['sha256'], canvas=rb.sha256_file(args.bench / 'windows-dev' / w / 'canvas' / 'real.mp4')))
        for name, fr in (('raw-band.mp4', raw), ('canvas-even.mp4', even)):
            rb.write_lossless(fr, od / name, 2, 'ultrafast')
            back, info = rb.decode_rgb(od / name, 2)
            if not np.array_equal(back, fr):
                raise SystemExit(f'{w}/{name}: decoded frames differ')
            rec[name] = dict(sha256=rb.sha256_file(od / name), frames_sha256=rb.sha256_bytes(fr.tobytes()), frames=info['frames'], fps=info['fps'])
        d = np.abs(raw[:, :cmap['canvas_rows'][1]].astype(np.int16) - even[:, :cmap['canvas_rows'][1]])
        rec['raw_band_vs_canvas_even'] = dict(max_abs=int(d.max()), mean_abs=round(float(d.mean()), 4))
        rec['script_sha256'] = rb.sha256_file(Path(__file__))
        (od / 'raw.json').write_text(json.dumps(rec, indent=1) + '\n')
        done += 1
        print(json.dumps(dict(window=w, **rec['raw_band_vs_canvas_even'])), flush=True)
    print(json.dumps(dict(windows=done, out=str(args.out))))


if __name__ == '__main__':
    main()
