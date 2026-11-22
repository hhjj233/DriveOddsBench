#!/usr/bin/env python3
"""Canvas-path detection for the shared canvas instrument (canvas-v1, DEV only): the frozen bench-v1 detector
(detect_bench_v2.py: COCO Faster R-CNN v2, local weights, torchvision defaults, labels 1/2/3/4/6/8, score >= 0.1, batch
9, boxes in torchvision pixel edges) on 512x288 canvas clips of any frame count (81 frames at 10 Hz, 41 at 5 Hz).

Sources (--source NAME --from KIND[:ARG]):
  run:DIR         DIR/native/<window>-<sample>/vision.mp4 (the adapters' run layout)  -> <out>/NAME/<window>-<sample>.json
  canvas-even     BENCH/windows-<split>/<w>/canvas/real.mp4 at the even frames (5 Hz ticks k = -20..60 step 2), canvas
                  rows >= the spec's valid_canvas_rows[1] set to 0 (Epona's uncovered rows)  -> <out>/NAME/<w>.json
  file:PATTERN    PATTERN with {w} for the window (a raw clip already on the canvas)       -> <out>/NAME/<w>.json
Every clip must decode to the spec's frame count at 512x288. Resumable (clips with a JSON are skipped).

GPU etiquette (research server, GPU 1 only, run under a GPU lock file): the process
caps its own memory (--max-gib), refuses to start with less than --leave-free-mib + --batch-need-mib free, and before
every batch waits (never touches other processes) until that much is free.

  detect_canvas.py --bench B --names-file F --model orbis2|ltx25|epona --source NAME --from KIND[:ARG] --out DIR [--allow-test]
DEV by default; TEST windows only with --allow-test once the model's frozen protocol file exists (ci_core.split_of).
"""
from __future__ import annotations

import argparse, json, os, queue, subprocess, sys, threading, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import ci_core as ci  # noqa: E402
import detect_bench_v2 as dbv  # noqa: E402  (frozen detector settings and decoder, read only)


def jobs_for(args, spec, names):
    kind, _, arg = args.from_.partition(':')
    out = args.out / args.source
    for w in names:
        if kind == 'run':
            for d in sorted((Path(arg) / 'native').glob(f'{w}-*')):
                if (d / 'vision.mp4').is_file():
                    yield dict(video=d / 'vision.mp4', select='all', black_from=None, out=out / f'{d.name}.json', window=w, sample=d.name[len(w) + 1:])
        elif kind == 'canvas-even':
            if spec['step'] != 2:
                raise SystemExit('canvas-even is the 5 Hz raw source')
            yield dict(video=args.bench / getattr(args, 'windows_dir', 'windows-dev') / w / 'canvas' / 'real.mp4', select='even', black_from=spec['valid_canvas_rows'][1],
                       out=out / f'{w}.json', window=w, sample='raw')
        elif kind == 'file':
            p = Path(arg.format(w=w))
            if p.is_file():
                yield dict(video=p, select='all', black_from=None, out=out / f'{w}.json', window=w, sample='raw')
        else:
            raise SystemExit(f'unknown source kind {kind}')


def frames_of(job):
    fr = dbv.decode(job['video'], dbv.CANVAS_W, dbv.CANVAS_H)
    if job['select'] == 'even':
        if len(fr) != 81:
            raise ValueError(f'{len(fr)} frames, canvas-even needs the 81-frame canvas clip')
        fr = fr[0::2].copy()
    if job['black_from'] is not None and job['black_from'] < dbv.CANVAS_H:
        fr = fr.copy(); fr[:, job['black_from']:] = 0
    return fr


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bench', type=Path, required=True); ap.add_argument('--names-file', type=Path, required=True)
    ap.add_argument('--model', required=True); ap.add_argument('--source', required=True); ap.add_argument('--from', dest='from_', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--batch', type=int, default=9); ap.add_argument('--max-gib', type=float, default=3.5)
    ap.add_argument('--leave-free-mib', type=float, default=6144); ap.add_argument('--batch-need-mib', type=float, default=2500)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--allow-test', action='store_true', help="TEST windows; refused unless the spec's test_freeze_file exists in the bench")
    args = ap.parse_args()
    spec = ci.load_spec(args.model)
    names = ci.names_from(args.names_file); split = ci.split_of(args.bench, names, spec, args.allow_test)
    args.windows_dir = ci.windows_dir(split)
    jobs = [j for j in jobs_for(args, spec, names) if not j['out'].is_file()]
    if args.limit:
        jobs = jobs[:args.limit]
    print(json.dumps(dict(to_detect=len(jobs), source=args.source, model=spec['model'], frames=spec['n_frames'])), flush=True)
    if not jobs:
        return
    import torch, torchvision
    torch.set_num_threads(2)
    dev_name = 'cuda:0'
    free, total = torch.cuda.mem_get_info(0)
    if free / 2 ** 20 < args.leave_free_mib + args.batch_need_mib + 1024:
        sys.exit(f'GPU free {free / 2 ** 20:.0f} MiB: not enough to keep {args.leave_free_mib:.0f} MiB free')
    torch.cuda.set_per_process_memory_fraction(min(1.0, args.max_gib * 2 ** 30 / total), 0)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None)
    model.load_state_dict(torch.load(dbv.WEIGHTS, map_location='cpu', weights_only=True)); model.eval().to(dev_name)
    gpu_uuid = None
    try:
        gpu_uuid = subprocess.check_output(['nvidia-smi', '-i', os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0], '--query-gpu=uuid,name',
                                            '--format=csv,noheader']).decode().strip()
    except Exception:
        pass
    script_sha = ci.sha256_file(__file__)
    q: queue.Queue = queue.Queue(maxsize=4)

    def reader():
        for job in jobs:
            try:
                q.put((job, frames_of(job)))
            except Exception as e:
                q.put((job, e))
        q.put(None)
    threading.Thread(target=reader, daemon=True).start()

    def wait_for_memory():
        waited = 0.0
        while True:
            free, _ = torch.cuda.mem_get_info(0)
            need = max(0.0, args.batch_need_mib * 2 ** 20 - torch.cuda.memory_reserved(0)) + args.leave_free_mib * 2 ** 20
            if free >= need:
                return waited
            torch.cuda.empty_cache(); time.sleep(20); waited += 20
            if waited % 600 < 20:
                print(json.dumps(dict(waiting_for_gpu_memory_s=waited, free_mib=round(free / 2 ** 20))), flush=True)

    t0, done, errors, waited = time.time(), 0, [], 0.0
    for item in iter(q.get, None):
        job, frames = item
        if isinstance(frames, Exception) or frames.shape != (spec['n_frames'], dbv.CANVAS_H, dbv.CANVAS_W, 3):
            errors.append(dict(video=str(job['video']), error=str(frames) if isinstance(frames, Exception) else f'{frames.shape}')); continue
        dets = []
        with torch.no_grad():
            for s in range(0, len(frames), args.batch):
                waited += wait_for_memory()
                outs = model([torch.as_tensor(np.array(f)).permute(2, 0, 1).float().div(255.0).to(dev_name) for f in frames[s:s + args.batch]])
                for o in outs:
                    dets.append([[round(float(v), 2) for v in b] + [round(float(sc), 4), int(l)] for b, sc, l in
                                 zip(o['boxes'].cpu(), o['scores'].cpu(), o['labels'].cpu()) if int(l) in dbv.LABELS and float(sc) >= dbv.MIN_SCORE])
        out = job['out']; out.parent.mkdir(parents=True, exist_ok=True)
        rec = dict(video=str(job['video']), kind='canvas', mapping=None, frames=list(range(len(frames))), canvas=[dbv.CANVAS_W, dbv.CANVAS_H],
                   detector=dbv.WEIGHTS.name, labels=list(dbv.LABELS), min_score=dbv.MIN_SCORE, device=dev_name, gpu=gpu_uuid,
                   box_convention='torchvision pixel edges', detections=dets,
                   canvas_instrument=dict(model=spec['model'], source=args.source, sample=job['sample'], window=job['window'], select=job['select'],
                                          black_from_row=job['black_from'], fps=spec['fps'], ticks_k=list(range(-20, 61, spec['step'])),
                                          script_sha256=script_sha, batch=args.batch))
        tmp = out.with_suffix('.tmp'); tmp.write_text(json.dumps(rec) + '\n'); tmp.rename(out)
        done += 1
        if done % 20 == 0:
            print(json.dumps(dict(done=done, of=len(jobs), s=round(time.time() - t0), waited_s=waited,
                                  peak_gib=round(torch.cuda.max_memory_allocated(0) / 2 ** 30, 2))), flush=True)
    print(json.dumps(dict(done=done, errors=errors[:10], n_errors=len(errors), s=round(time.time() - t0), waited_s=waited,
                          peak_gib=round(torch.cuda.max_memory_allocated(0) / 2 ** 30, 2))), flush=True)


if __name__ == '__main__':
    main()
