#!/usr/bin/env python3
"""DEV check of Epona's pose convention on AV2 (exploratory; DEV windows only).

Epona's README says the relative pose has "y right positive"; its training code (get_rel_pose on nuPlan ego poses) yields
y LEFT positive, and the adapter follows the code (AV2's egovehicle frame is also x forward, y left, z up). This script asks
the model which reading it expects: for each DEV window, Epona's own trajectory head (TrajDiT, traj_only=True, the released
100 steps) predicts the next 3.0 s (15 steps at 5 Hz, cumulative, in the frame of the last condition frame) from the real
history under three readings of the past poses:
  code       the adapter's convention (get_rel_pose of the AV2 poses: dy left +, dyaw left +)
  y_flip     dy negated (the README's literal "y right positive")
  both_flip  dy and dyaw negated
and compares it with the real future trajectory from the DEV gt.json ego poses (k = 2..30). Reported per reading: ADE and
FDE (m), the share of turning windows whose predicted final heading change has the real sign, and the lateral sign share.

  check_epona_pose_sign.py --input-root DIR --bench-root BENCH --names FILE --out FILE.json --models-dir DIR --epona-src DIR
                           [--seeds 4] [--expect-gpu-uuid UUID] [--leave-free-mib 6144] [--max-gpu-mib 30000]
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_epona_bench as rb  # noqa: E402


def real_future(gt_path: Path, np):
    """DEV only: the real ego trajectory at k = 2..30 relative to the pose at k = 0, as Epona's get_rel_traj (x, y, yaw deg)."""
    g = json.loads(gt_path.read_text())
    m = {}
    for e in g['ego']:
        a = np.eye(4)
        a[:3] = np.array(e['matrix_3x4'])
        m[e['k']] = a
    inv0 = np.linalg.inv(m[0])
    out = []
    for k in range(2, 31, 2):
        r = inv0 @ m[k]
        out.append([r[0, 3], r[1, 3], np.degrees(np.arctan2(r[1, 0], r[0, 0]))])
    return np.array(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input-root', type=Path, required=True)
    ap.add_argument('--bench-root', type=Path, required=True)
    ap.add_argument('--names', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--models-dir', type=Path, required=True)
    ap.add_argument('--epona-src', type=Path, required=True)
    ap.add_argument('--seeds', type=int, default=4)
    ap.add_argument('--expect-gpu-uuid', default=None)
    ap.add_argument('--leave-free-mib', type=int, default=6144)
    ap.add_argument('--max-gpu-mib', type=int, default=30000)
    args = ap.parse_args()
    import numpy as np
    import torch
    torch.set_num_threads(2)
    gpu = rb.gpu_info(torch)
    if args.expect_gpu_uuid and gpu.get('uuid') != args.expect_gpu_uuid:
        raise SystemExit(f'visible GPU {gpu.get("uuid")} is not the expected {args.expect_gpu_uuid}')
    free, total = torch.cuda.mem_get_info()
    cap = min(args.max_gpu_mib, free / 2**20 - args.leave_free_mib - 1024)
    torch.cuda.set_per_process_memory_fraction(min(1.0, cap * 2**20 / total), 0)
    model = rb.Epona(args.models_dir.resolve(), args.epona_src.resolve(), 'size', False, 2)
    names = args.names.read_text().split()
    rows = []
    t_start = time.time()
    for w in names:
        wdir = args.input_root / 'windows' / 'dev' / w
        hist, _ = rb.decode_rgb(wdir / 'history.mp4', 2)
        mats = rb.load_poses(wdir / 'past_poses.json', np)
        real = real_future(args.bench_root / 'windows-dev' / w / 'gt.json', np)
        cond = hist[rb.N_HIST - rb.CF:]
        with torch.no_grad():
            img = model.normalize(cond).unsqueeze(0).cuda()
            lat = model.tokenizer.encode_to_z(img[:, :rb.CF])
            pose0, yaw0 = model.condition_poses(mats)
            row = dict(window=w, real_final=real[-1].round(3).tolist(), past_speed_mps=round(float(pose0[0, -2, 0].item() / 0.2), 3),
                       past_dyaw_sum_deg=round(float(yaw0[0, :-1, 0].sum().item()), 3), readings={})
            for tag, sy, syaw in (('code', 1.0, 1.0), ('y_flip', -1.0, 1.0), ('both_flip', -1.0, -1.0)):
                pose = pose0.clone()
                pose[..., 1] *= sy
                yaw = yaw0 * syaw
                preds = []
                for s in range(args.seeds):
                    rb.Epona.seed_all(int(rb.sha256_bytes(f'pose-sign|{w}|{s}'.encode())[:8], 16) & 0x7FFFFFFF, torch, np)
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16), contextlib.redirect_stdout(io.StringIO()):
                        tr, _ = model.model.step_eval(lat, pose, yaw, self_pred_traj=True, traj_only=True)
                    preds.append(tr[0].float().cpu().numpy())
                p = np.stack(preds)                                   # [S, 15, 3]
                err = np.linalg.norm(p[:, :, :2] - real[None, :, :2], axis=-1)
                row['readings'][tag] = dict(ade=round(float(err.mean()), 3), fde=round(float(err[:, -1].mean()), 3),
                                            final_dy=round(float(p[:, -1, 1].mean()), 3), final_dyaw=round(float(p[:, -1, 2].mean()), 3),
                                            final_dx=round(float(p[:, -1, 0].mean()), 3))
        rows.append(row)
        print(json.dumps(dict(window=w, **{k: v['ade'] for k, v in row['readings'].items()})), flush=True)
    turning = [r for r in rows if abs(r['real_final'][2]) >= 5.0 and r['real_final'][0] >= 3.0]
    lateral = [r for r in rows if abs(r['real_final'][1]) >= 1.0 and r['real_final'][0] >= 3.0]
    summ = {}
    for tag in ('code', 'y_flip', 'both_flip'):
        summ[tag] = dict(ade_mean=round(float(np.mean([r['readings'][tag]['ade'] for r in rows])), 3),
                         fde_mean=round(float(np.mean([r['readings'][tag]['fde'] for r in rows])), 3),
                         ade_turning=round(float(np.mean([r['readings'][tag]['ade'] for r in turning])), 3) if turning else None,
                         yaw_sign_agree_turning=round(float(np.mean([np.sign(r['readings'][tag]['final_dyaw']) == np.sign(r['real_final'][2]) for r in turning])), 3) if turning else None,
                         lateral_sign_agree=round(float(np.mean([np.sign(r['readings'][tag]['final_dy']) == np.sign(r['real_final'][1]) for r in lateral])), 3) if lateral else None)
    res = dict(schema='wm-calib-bench.epona.pose_sign_check.v1', split='dev', windows=len(rows), turning=len(turning), lateral=len(lateral),
               seeds=args.seeds, summary=summ, rows=rows, gpu=gpu, seconds=round(time.time() - t_start, 1),
               note='DEV only; exploratory; uses DEV real future ego poses (gt.json) for the comparison')
    args.out.write_text(json.dumps(res, indent=1) + '\n')
    print(json.dumps(dict(windows=len(rows), turning=len(turning), lateral=len(lateral), summary=summ), indent=1))


if __name__ == '__main__':
    main()
