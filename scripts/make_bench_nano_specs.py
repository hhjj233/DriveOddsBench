#!/usr/bin/env python3
"""Cosmos3-Nano sample specs for the band-geometry benchmark windows (bench-v1, 2026-11-01).

The windows come from make_benchmark_windows.py: <bench>/windows-<split>/<name>/nano/history.mp4 holds 21 frames
(t0-2.0 .. t0 s, 10 Hz) and nano/real.mp4 holds 81 frames (t0-2.0 .. t0+6.0 s), both 320x192, which is Cosmos3's
256-tier "16,9" bucket, so the framework does not resize or crop them.

  --mode continue  video2video from the 2 s history: latents 0-5 (21 frames) conditioned, 81 frames out, 10 fps.
                   One spec per window and sample k; the seed is a fixed function of (window, k), the same table for
                   every model, so samples are independent across windows (in study-v1 one seed shared its noise
                   across all windows).
  --mode recon     the real reference: nano/real.mp4 with all 21 latents conditioned (2 steps; the output does not
                   depend on the step count, checked in study-v1), seed 0.

The sampler defaults are copied from study-v1 (35 steps, guidance 6, shift 10, the default negative prompt); only
num_steps and guidance can be overridden. The prompt is the benchmark's fixed generic description (no motion, no
events, no future) unless --prompt short selects the study-v1 prompt (a sensitivity arm).

  make_bench_nano_specs.py --bench DIR --split dev|test --mode continue|recon --run NAME [--names FILE] [--samples 0-15]
"""
from __future__ import annotations
import os

import argparse, hashlib, json
from pathlib import Path

STUDY = Path(os.environ.get('DOB_ROOT', '.'), 'outputs/wm-calibration/study-v1')
DEFAULTS = {'continue': STUDY / 'hist5-specs/abl-h5/defaults.json', 'recon': STUDY / 'hist5-specs/recon-hist5/defaults.json'}
PROMPTS = {
    'long': ('A continuous, unedited video from a forward-facing dashcam mounted behind the windshield of a passenger car on '
             'public roads in a North American city. The camera is rigidly fixed to the car at a constant height and angle, '
             'looking straight ahead along the road. The view shows the road surface with lane markings, curbs and '
             'crosswalks, together with other vehicles, pedestrians, cyclists, traffic lights, road signs, poles, trees, '
             'parked cars and buildings along the street. The video seamlessly continues the given footage from its last '
             'frame, with the same camera, lens, exposure, weather and lighting. There are no cuts, no scene changes, no '
             'transitions, no zoom, no text, no overlays and no watermarks. Photorealistic, natural colors, sharp details, '
             'and consistent geometry and perspective throughout.'),
    'short': 'A realistic front-facing driving camera video on a city street, continuing the given footage.',
}
HISTORY_LATENTS, TOTAL_FRAMES, FPS = 6, 81, 10


def seed_of(window: str, k: int) -> int:
    return int.from_bytes(hashlib.sha256(f'bench-v1|{window}|{k}'.encode()).digest()[:4], 'big') & 0x7FFFFFFF


def parse_range(text: str) -> list[int]:
    lo, _, hi = text.partition('-')
    return list(range(int(lo), int(hi or lo) + 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--bench', required=True, type=Path); parser.add_argument('--split', required=True, choices=['dev', 'test'])
    parser.add_argument('--mode', required=True, choices=['continue', 'recon']); parser.add_argument('--run', required=True)
    parser.add_argument('--names', type=Path, help='subset of window names (default: names-<split>.txt)')
    parser.add_argument('--samples', default='0-15', help='sample indices k for --mode continue')
    parser.add_argument('--prompt', default='long', choices=sorted(PROMPTS))
    parser.add_argument('--steps', type=int); parser.add_argument('--guidance', type=float)
    args = parser.parse_args()

    names = (args.names or args.bench / f'names-{args.split}.txt').read_text().split()
    windows = args.bench / f'windows-{args.split}'
    out = args.bench / 'specs' / args.run
    if out.exists():
        raise SystemExit(f'{out} exists; choose a new run name')
    (out / 'specs').mkdir(parents=True)

    source = DEFAULTS[args.mode]
    defaults = json.loads(source.read_text())
    if args.steps is not None:
        defaults['num_steps'] = args.steps
    if args.guidance is not None:
        defaults['guidance'] = args.guidance
    (out / 'defaults.json').write_text(json.dumps(defaults, indent=1) + '\n')

    common = dict(defaults_file=str(out / 'defaults.json'), model_mode='video2video', condition_video_keep='first', num_frames=TOTAL_FRAMES,
                  fps=FPS, resolution='256', aspect_ratio='16,9', prompt=PROMPTS[args.prompt], enable_sound=False)
    specs = []
    for name in names:
        nano = windows / name / 'nano'
        if args.mode == 'continue':
            if not (nano / 'history.mp4').is_file():
                raise SystemExit(f'{nano}/history.mp4 missing')
            for k in parse_range(args.samples):
                specs.append(dict(name=f'{name}-s{k}', vision_path=str(nano / 'history.mp4'), condition_frame_indexes_vision=list(range(HISTORY_LATENTS)),
                                  seed=seed_of(name, k), **common))
        else:
            if not (nano / 'real.mp4').is_file():
                raise SystemExit(f'{nano}/real.mp4 missing')
            specs.append(dict(name=f'{name}-recon', vision_path=str(nano / 'real.mp4'), condition_frame_indexes_vision=list(range((TOTAL_FRAMES - 1) // 4 + 1)),
                              seed=0, **common))
    for spec in specs:
        (out / 'specs' / f"{spec['name']}.json").write_text(json.dumps(spec, indent=1) + '\n')
    me = Path(__file__).read_bytes()
    (out / 'manifest.json').write_text(json.dumps(dict(
        split=args.split, mode=args.mode, prompt=args.prompt, samples=args.samples if args.mode == 'continue' else None, windows=len(names),
        specs=len(specs), defaults_source=str(source), defaults_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        num_steps=defaults['num_steps'], guidance=defaults['guidance'], generator_sha256=hashlib.sha256(me).hexdigest()), indent=1) + '\n')
    print(json.dumps(dict(run=args.run, specs=len(specs))))


if __name__ == '__main__':
    main()
