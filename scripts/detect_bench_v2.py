#!/usr/bin/env python3
"""Per-frame COCO detections on the 512x288 evaluation canvas for bench-v1 (instrument v2, 2026-11-01).

The expensive, cached part of the bench-v1 instrument (measure_bench_v2.py does the tracking on CPU). Every clip is
first brought onto the evaluation canvas, then the detector runs on all 81 frames:
  canvas-raw   windows-dev/<w>/canvas/real.mp4       (512x288 already: the source band rendered directly)
  nano-raw     windows-dev/<w>/nano/real.mp4         (320x192 -> canvas, mapping below)
  <run>        runs/<run>/native/<sample>/vision.mp4 (320x192 -> canvas, mapping below; recon and generated clips)
Nano -> canvas mapping (fixed, identical for nano-raw, recon and generated clips): crop Nano rows 6..185 (180 rows,
the measurement band, source rows 797-1669 to 0.02 canvas px), then cv2.resize(crop, (512, 288),
interpolation=cv2.INTER_CUBIC) (scale 1.6 on both axes). Canvas pixel = 1.6 x Nano pixel (x), 1.6 x (Nano - 6) (y),
in pixel-edge coordinates.

Detector: COCO Faster R-CNN v2, local weights (the selection's and study-v1's detector), torchvision defaults (internal
resize to 1333x750). Kept: person 1, bicycle 2, car 3, motorcycle 4, bus 6, truck 8 with score >= 0.1 (trackers apply
their own thresholds). Boxes are torchvision pixel-edge coordinates on the canvas (x1, y1, x2, y2, score, label).

GPU etiquette (shared GPUs): the process caps its own memory (--max-gib), and before every batch it checks the GPU's
free memory with torch.cuda.mem_get_info(); with less than --min-free-mib + the batch's need free, it releases its
cache and waits (never touches other processes). One JSON per clip under BENCH/detections/<source>/<sample>.json;
clips already done are skipped (resumable). The device and the mapping are recorded in each file.

  detect_bench_v2.py --bench DIR --names-file F [--canvas-raw] [--nano-raw] [--runs RUN ...] [--max-gib 3.5]
"""
from __future__ import annotations

import argparse, json, os, subprocess, sys, threading, queue, time
from pathlib import Path

import numpy as np

WEIGHTS = Path(os.environ.get('DOB_ROOT', '.'), 'cache/torch/hub/checkpoints/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth')
LABELS = (1, 2, 3, 4, 6, 8)
MIN_SCORE = 0.1
CANVAS_W, CANVAS_H = 512, 288
NANO_W, NANO_H = 320, 192
NANO_CROP_ROWS = (6, 186)
NANO_TO_CANVAS = dict(crop_rows=list(NANO_CROP_ROWS), resize=[CANVAS_W, CANVAS_H], interpolation='cv2.INTER_CUBIC', scale=1.6,
                      note='canvas_x = 1.6 * nano_x, canvas_y = 1.6 * (nano_y - 6) (pixel-edge coordinates); '
                           'the direct canvas mapping differs by 0.02 canvas px')


def decode(path, w, h):
    raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-threads', '2', '-i', str(path), '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'])
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)


def nano_to_canvas(frames):
    import cv2
    r0, r1 = NANO_CROP_ROWS
    return np.stack([cv2.resize(np.ascontiguousarray(f[r0:r1]), (CANVAS_W, CANVAS_H), interpolation=cv2.INTER_CUBIC) for f in frames])


def canvas_frames(path, kind):
    """Frames of a clip on the evaluation canvas (kind 'canvas' = already 512x288, 'nano' = 320x192 mapped)."""
    if kind == 'canvas':
        return decode(path, CANVAS_W, CANVAS_H)
    return nano_to_canvas(decode(path, NANO_W, NANO_H))


def clip_jobs(bench: Path, names, canvas_raw: bool, nano_raw: bool, runs, windows_dir='windows-dev'):
    det = bench / 'detections'
    for name in names:
        if canvas_raw:
            yield bench / windows_dir / name / 'canvas' / 'real.mp4', 'canvas', det / 'canvas-raw' / f'{name}.json'
        if nano_raw:
            yield bench / windows_dir / name / 'nano' / 'real.mp4', 'nano', det / 'nano-raw' / f'{name}.json'
        for run in runs:
            for d in sorted((bench / 'runs' / run / 'native').glob(f'{name}-*')):
                if (d / 'vision.mp4').is_file():
                    yield d / 'vision.mp4', 'nano', det / run / f'{d.name}.json'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bench', type=Path, required=True); ap.add_argument('--names-file', type=Path, required=True)
    ap.add_argument('--canvas-raw', action='store_true'); ap.add_argument('--nano-raw', action='store_true'); ap.add_argument('--runs', nargs='*', default=[])
    ap.add_argument('--batch', type=int, default=9); ap.add_argument('--max-gib', type=float, default=3.5)
    ap.add_argument('--min-free-mib', type=float, default=1500, help='free memory that must remain on the GPU besides this batch')
    ap.add_argument('--batch-need-mib', type=float, default=2500, help='memory one batch may add (measured peak, rounded up)')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--allow-test', action='store_true', help='TEST windows; refused unless BENCH/instrument-frozen.txt exists')
    args = ap.parse_args()
    # refuse anything outside DEV: TEST material is reserved until protocol-v1.1 is frozen
    names = [l.strip() for l in args.names_file.read_text().splitlines() if l.strip()]
    dev = set(l.strip() for l in (args.bench / 'names-dev.txt').read_text().splitlines() if l.strip())
    if not set(names) <= dev and not (args.allow_test and (args.bench / 'instrument-frozen.txt').is_file()):
        sys.exit('refusing: names outside names-dev.txt (DEV only until protocol-v1.1 is frozen)')
    if set(names) & dev and not set(names) <= dev:
        sys.exit('refusing: a names file must not mix DEV and TEST windows')
    wdir = 'windows-dev' if set(names) <= dev else 'windows-test'
    jobs = [j for j in clip_jobs(args.bench, names, args.canvas_raw, args.nano_raw, args.runs, wdir) if not j[2].is_file()]
    if args.limit:
        jobs = jobs[:args.limit]
    print(json.dumps(dict(to_detect=len(jobs))), flush=True)
    if not jobs:
        return
    import torch, torchvision
    torch.set_num_threads(2)
    dev_name = 'cuda:0'
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, args.max_gib * 2 ** 30 / total), 0)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None)
    model.load_state_dict(torch.load(WEIGHTS, map_location='cpu', weights_only=True)); model.eval().to(dev_name)
    gpu_uuid = None
    try:
        gpu_uuid = subprocess.check_output(['nvidia-smi', '-i', os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0], '--query-gpu=uuid,name',
                                            '--format=csv,noheader']).decode().strip()
    except Exception:
        pass

    q: queue.Queue = queue.Queue(maxsize=4)

    def reader():
        for job in jobs:
            try:
                q.put((job, canvas_frames(job[0], job[1])))
            except Exception as e:  # unreadable video: report and go on
                q.put((job, e))
        q.put(None)
    threading.Thread(target=reader, daemon=True).start()

    def wait_for_memory():
        waited = 0.0
        while True:
            free, _ = torch.cuda.mem_get_info(0)
            mine = torch.cuda.memory_reserved(0)
            # memory this process already holds can be reused; it needs at most batch_need beyond that
            need = max(0.0, args.batch_need_mib * 2 ** 20 - mine) + args.min_free_mib * 2 ** 20
            if free >= need:
                return waited
            torch.cuda.empty_cache(); time.sleep(20); waited += 20
            if waited % 600 < 20:
                print(json.dumps(dict(waiting_for_gpu_memory_s=waited, free_mib=round(free / 2 ** 20))), flush=True)

    t0, done, errors, waited = time.time(), 0, [], 0.0
    for item in iter(q.get, None):
        (video, kind, out), frames = item
        if isinstance(frames, Exception) or len(frames) != 81:
            errors.append(dict(video=str(video), error=str(frames) if isinstance(frames, Exception) else f'{len(frames)} frames')); continue
        dets = []
        with torch.no_grad():
            for s in range(0, len(frames), args.batch):
                waited += wait_for_memory()
                outs = model([torch.as_tensor(np.array(f)).permute(2, 0, 1).float().div(255.0).to(dev_name) for f in frames[s:s + args.batch]])
                for o in outs:
                    dets.append([[round(float(v), 2) for v in b] + [round(float(sc), 4), int(l)] for b, sc, l in
                                 zip(o['boxes'].cpu(), o['scores'].cpu(), o['labels'].cpu()) if int(l) in LABELS and float(sc) >= MIN_SCORE])
        out.parent.mkdir(parents=True, exist_ok=True)
        rec = dict(video=str(video), kind=kind, mapping=None if kind == 'canvas' else NANO_TO_CANVAS, frames=list(range(81)), canvas=[CANVAS_W, CANVAS_H],
                   detector=WEIGHTS.name, labels=list(LABELS), min_score=MIN_SCORE, device=dev_name, gpu=gpu_uuid, box_convention='torchvision pixel edges',
                   detections=dets)
        tmp = out.with_suffix('.tmp'); tmp.write_text(json.dumps(rec) + '\n'); tmp.rename(out)
        done += 1
        if done % 20 == 0:
            print(json.dumps(dict(done=done, of=len(jobs), s=round(time.time() - t0), waited_s=waited,
                                  peak_gib=round(torch.cuda.max_memory_allocated(0) / 2 ** 30, 2))), flush=True)
    print(json.dumps(dict(done=done, errors=errors[:10], n_errors=len(errors), s=round(time.time() - t0), waited_s=waited,
                          peak_gib=round(torch.cuda.max_memory_allocated(0) / 2 ** 30, 2))), flush=True)


if __name__ == '__main__':
    main()
