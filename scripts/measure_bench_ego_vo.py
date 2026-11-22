#!/usr/bin/env python3
"""bench-v1 ego family: ego motion from road-surface optical flow on 320x192 Nano frames (VO v2 ported, 2026-11-01).

A port of study-v1's validated VO v2 (scripts/measure_ego_motion_raft.py, imported read-only; its algorithm is unchanged)
to the band geometry. The gate this instrument must pass is fixed in bench-v1/instrument/ego-vo/VO-GATE.md (written
before any evaluation).

Changes from VO v2 (geometry only, plus planar odometry):
  - frame: the 320x192 Nano frame (native resolution of reconstructed and generated clips; raw = nano/real.mp4); camera
    K_nano and ego_SE3_cam from window.json (AV2 calibration); road 0.33 m below the ego origin, as in VO v2;
  - RAFT on the whole frame (no hood, no side bars in the band), upsampled x2 (640x384), 20 updates;
  - road pixels rows 60-190, columns 2-317 (step 2), ground distance 3-35 m; flow targets within rows 50-191;
  - masks: canvas detections (bench-v1/detections, all kept classes, score >= 0.3) mapped to Nano pixels, padded 4 px;
  - planar odometry: per stride-4 group the advance is VO v2's final value (accepted refined, else interpolated) and the
    yaw that of the accepted refined motion, else of the composed stride-1 chain; positions integrate the advance along
    the mid-group heading. D(T) = forward displacement (x at T in the t0 ego frame), Psi(T) = heading change.

  measure_bench_ego_vo.py measure --bench B --names-file F --sources nano-raw RUN... [--part i/n]
  measure_bench_ego_vo.py gate    --bench B
"""
from __future__ import annotations

import argparse, json, math, os, queue, subprocess, sys, threading, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import measure_ego_motion_raft as vo  # noqa: E402  (study-v1 VO v2, read only)

H, W = 192, 320
CROP = (0, 192, 0, 320)
ROI_ROWS, ROI_COLS, TGT_ROW_MAX, TGT_ROW_MIN = (60, 190), (2, 317), 191, 50
DIST_M = (3.0, 35.0)
DET_SCORE, DET_PAD = 0.3, 4
HORIZONS_S = (1.0, 2.0, 4.0, 6.0)
OUT_SUB = 'instrument/ego-vo'


def box_mask(dets):
    """Canvas detections (pixel edges) -> mask on the Nano frame (canvas = 1.6 x Nano, rows offset by 6 Nano px)."""
    m = np.zeros((H, W), bool)
    for d in dets or []:
        if d[4] < DET_SCORE:
            continue
        x1, y1, x2, y2 = d[0] / 1.6, d[1] / 1.6 + 6, d[2] / 1.6, d[3] / 1.6 + 6
        m[max(0, int(y1) - DET_PAD):min(H, int(np.ceil(y2)) + DET_PAD), max(0, int(x1) - DET_PAD):min(W, int(np.ceil(x2)) + DET_PAD)] = True
    return m


class Grid:
    def __init__(self, cam: vo.Camera, step=2):
        vv, uu = np.mgrid[ROI_ROWS[0]:ROI_ROWS[1] + 1:step, ROI_COLS[0]:ROI_COLS[1] + 1:step]
        self.u = uu.ravel().astype(float); self.v = vv.ravel().astype(float)
        self.iu = self.u.astype(int); self.iv = self.v.astype(int)
        self.P = cam.ground(cam.rays(self.u, self.v))
        dist = self.P @ cam.fwd
        self.base = np.isfinite(dist) & (dist >= DIST_M[0]) & (dist <= DIST_M[1])


def fit_pair(cam, grid, flow, fb_ok, dets_i, dets_j, theta0=None):
    F = flow[:, grid.iv, grid.iu].T
    tu, tv = grid.u + F[:, 0], grid.v + F[:, 1]
    ok = grid.base & np.isfinite(F).all(1) & fb_ok[grid.iv, grid.iu] & ~box_mask(dets_i)[grid.iv, grid.iu]
    ok &= (tu >= ROI_COLS[0]) & (tu <= ROI_COLS[1]) & (tv <= TGT_ROW_MAX) & (tv >= TGT_ROW_MIN)
    ok &= ~box_mask(dets_j)[np.clip(np.round(np.nan_to_num(tv)).astype(int), 0, H - 1), np.clip(np.round(np.nan_to_num(tu)).astype(int), 0, W - 1)]
    n = int(ok.sum())
    if n < vo.MIN_POINTS:
        return None, dict(points=n)
    tgt = np.stack([(tu[ok] - cam.cx) / cam.fx, (tv[ok] - cam.cy) / cam.fy], 1)
    th, info = vo.fit_motion(grid.P[ok], tgt, cam.fy, theta0=theta0)
    return th, dict(points=n, **info)


def ego_yaw(theta, R_ec):
    """Yaw (rad, left positive) of the ego's rotation from frame i to frame j for camera motion theta = (C, w)."""
    Rc = vo.rodrigues(theta[3:6]).T            # camera j axes in camera i coordinates
    Re = R_ec @ Rc @ R_ec.T
    return float(math.atan2(Re[1, 0], Re[0, 0]))


def planar(rec):
    """Planar ego odometry from a finalised record: D(T), Psi(T) at the horizons and per group end."""
    R_ec = np.asarray(rec['camera']['ego_R_cam'], float)
    groups = rec['groups'][1:]                 # after the anchor: 20->24 ... 76->80
    psi, x, y = 0.0, 0.0, 0.0
    traj = [dict(t=0.0, x=0.0, y=0.0, psi_deg=0.0)]
    for k, g in enumerate(groups):
        th = g['passes'][0]['theta'] if g.get('accepted_m') is not None and g['passes'][0]['theta'] is not None else g['init_theta']
        yaw = ego_yaw(np.asarray(th, float), R_ec) if th is not None else 0.0
        mid = psi + yaw / 2; adv = g['final_m']
        x += adv * math.cos(mid); y += adv * math.sin(mid); psi += yaw
        traj.append(dict(t=round((k + 1) * 0.4, 1), x=round(x, 4), y=round(y, 4), psi_deg=round(math.degrees(psi), 4), yaw_source='refined' if g.get('accepted_m') is not None else 'chain'))
    ts = np.array([p['t'] for p in traj])
    out = {}
    for h in HORIZONS_S:
        out[str(h)] = dict(D_m=round(float(np.interp(h, ts, [p['x'] for p in traj])), 4), lateral_m=round(float(np.interp(h, ts, [p['y'] for p in traj])), 4),
                           Psi_deg=round(float(np.interp(h, ts, [p['psi_deg'] for p in traj])), 4))
    return dict(horizons=out, trajectory=traj)


def frames_of(path):
    raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-threads', '1', '-i', str(path), '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'])
    return np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3)


def clip_list(bench: Path, names, sources, windows_dir='windows-dev'):
    det = bench / 'detections'
    for name in names:
        for source in sources:
            if source == 'nano-raw':
                yield name, source, 'raw', bench / windows_dir / name / 'nano' / 'real.mp4', det / 'nano-raw' / f'{name}.json'
            else:
                for d in sorted((bench / 'runs' / source / 'native').glob(f'{name}-*')):
                    sample = d.name[len(name) + 1:]
                    yield name, source, ('real' if sample == 'recon' else sample.lstrip('s')), d / 'vision.mp4', det / source / f'{d.name}.json'


def camera_of(bench, name, windows_dir='windows-dev'):
    w = json.loads((bench / windows_dir / name / 'window.json').read_text())
    T = np.asarray(w['intrinsics']['ego_SE3_cam'], float)
    return dict(K256=w['intrinsics']['K_nano'], ego_R_cam=T[:3, :3].tolist(), ego_t_cam=T[:3, 3].tolist())


def measure(args):
    import cv2, torch, torch.nn.functional as tF
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
    torch.set_num_threads(2); cv2.setNumThreads(1)
    names = [l.strip() for l in args.names_file.read_text().splitlines() if l.strip()]
    dev = set(l.strip() for l in (args.bench / 'names-dev.txt').read_text().splitlines() if l.strip())
    if not set(names) <= dev and not (args.allow_test and (args.bench / 'instrument-frozen.txt').is_file()):
        sys.exit('refusing: names outside names-dev.txt (DEV only until protocol-v1.1 is frozen)')
    if set(names) & dev and not set(names) <= dev:
        sys.exit('refusing: a names file must not mix DEV and TEST windows')
    wdir = 'windows-dev' if set(names) <= dev else 'windows-test'
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, args.gpu_mem_gib * 2 ** 30 / total), 0)
    dev_name = 'cuda:0'
    out_dir = args.bench / OUT_SUB / 'clips'
    # a clip is measured only once its detections (the masks) exist, so every clip gets the same instrument
    jobs = [j for j in clip_list(args.bench, names, args.sources, wdir) if j[3].is_file() and j[4].is_file() and not (out_dir / j[1] / f'{j[0]}__{j[2]}.json').exists()]
    part, nparts = (int(x) for x in args.part.split('/'))
    jobs = jobs[part::nparts]
    print(f'{len(jobs)} clips to measure on {dev_name}', flush=True)
    model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).eval().to(dev_name)
    r0, r1, c0, c1 = CROP
    q: queue.Queue = queue.Queue(maxsize=3)

    def reader():
        for job in jobs:
            q.put((job, frames_of(job[3])))
        q.put(None)
    threading.Thread(target=reader, daemon=True).start()

    def wait_mem():
        while True:
            free, _ = torch.cuda.mem_get_info(0)
            if free - max(0, 1.6 * 2 ** 30 - torch.cuda.memory_reserved(0)) >= args.min_free_mib * 2 ** 20:
                return
            torch.cuda.empty_cache(); time.sleep(20)

    def prep(imgs):
        x = torch.from_numpy(np.ascontiguousarray(imgs)).permute(0, 3, 1, 2).float().div(127.5).sub(1).to(dev_name)[:, :, r0:r1, c0:c1]
        return tF.interpolate(x, scale_factor=vo.UP, mode='bicubic', align_corners=False).clamp(-1, 1)

    def raft(a_imgs, b_imgs):
        out = []
        with torch.no_grad():
            for s in range(0, len(a_imgs), args.batch):
                wait_mem()
                fl = model(prep(a_imgs[s:s + args.batch]), prep(b_imgs[s:s + args.batch]), num_flow_updates=vo.ITERS)[-1]
                fl = tF.interpolate(fl, size=(r1 - r0, c1 - c0), mode='area') / vo.UP
                full = np.full((fl.shape[0], 2, H, W), np.nan, np.float32); full[:, :, r0:r1, c0:c1] = fl.cpu().numpy()
                out.append(full)
        return np.concatenate(out)

    vv, uu = np.mgrid[0:H, 0:W]

    def fb_ok(fw, bw):
        B = vo.bilinear(np.nan_to_num(bw), (uu + np.nan_to_num(fw[0])).ravel(), (vv + np.nan_to_num(fw[1])).ravel()).reshape(2, H, W)
        e2 = ((fw + B) ** 2).sum(0); m2 = (fw ** 2).sum(0) + (B ** 2).sum(0)
        return np.isfinite(e2) & (e2 < 0.01 * m2 + 0.5)

    def rnd(th):
        return None if th is None else [round(float(x), 6) for x in th]

    def plaus(t, info, cam):
        info['_fwd'] = cam.fwd; ok = vo.plausible(t, info); info.pop('_fwd'); return ok

    t0 = time.time(); cams = {}
    for done, item in enumerate(iter(q.get, None)):
        (name, source, sample, video, det_file), fr = item
        if len(fr) != 81:
            print(f'skip {source} {name} {sample}: {len(fr)} frames', flush=True); continue
        if name not in cams:
            cams[name] = camera_of(args.bench, name, wdir)
        cam = vo.Camera(**cams[name]); grid = Grid(cam)
        dj = json.loads(det_file.read_text()) if det_file.is_file() else None
        dets = {t: d for t, d in zip(dj['frames'], dj['detections'])} if dj else {}
        p1 = [(i, i + 1) for i in range(vo.FIRST, 80)]
        fw = raft(fr[[i for i, _ in p1]], fr[[j for _, j in p1]]); bw = raft(fr[[j for _, j in p1]], fr[[i for i, _ in p1]])
        s1, th = [], None
        for k, (i, j) in enumerate(p1):
            t, info = fit_pair(cam, grid, fw[k], fb_ok(fw[k], bw[k]), dets.get(i), dets.get(j), theta0=th)
            th = t if t is not None else th
            s1.append((t, info, plaus(t, info, cam)))
        good = [k for k, (_, _, ok) in enumerate(s1) if ok]
        s1f = [s1[k][0] if s1[k][2] else (s1[min(good, key=lambda r: abs(r - k))][0] if good else np.zeros(6)) for k in range(len(p1))]
        adv1 = np.array([float(np.asarray(t[:3]) @ cam.fwd) for t in s1f])
        p4 = [(i, i + vo.STRIDE) for i in range(vo.FIRST, 80, vo.STRIDE)]
        cur = [vo.compose(s1f[i - vo.FIRST:i - vo.FIRST + vo.STRIDE]) for i, _ in p4]
        chain = [float(adv1[i - vo.FIRST:i - vo.FIRST + vo.STRIDE].sum()) for i, _ in p4]
        passes = []
        for _ in range(vo.REFINE_PASSES):
            warps, warped = [], []
            for k, (i, j) in enumerate(p4):
                Wf = vo.model_flow(cam, cur[k], shape=(H, W)); warps.append(Wf)
                warped.append(cv2.remap(np.ascontiguousarray(fr[j]), (uu + Wf[0]).astype(np.float32), (vv + Wf[1]).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE))
            warped = np.stack(warped)
            Rf = raft(fr[[i for i, _ in p4]], warped); Rb = raft(warped, fr[[i for i, _ in p4]])
            res = []
            for k, (i, j) in enumerate(p4):
                Ws = vo.bilinear(warps[k], (uu + np.nan_to_num(Rf[k][0])).ravel(), (vv + np.nan_to_num(Rf[k][1])).ravel()).reshape(2, H, W)
                t, info = fit_pair(cam, grid, Rf[k] + Ws, fb_ok(Rf[k], Rb[k]), dets.get(i), dets.get(j), theta0=cur[k])
                res.append((t, info, plaus(t, info, cam)))
            passes.append(res)
            cur = [t if ok else cur[k] for k, (t, info, ok) in enumerate(res)]
        rec = dict(name=name, source=source, sample=sample, video=str(video), detections=str(det_file) if dj else None, camera=cams[name],
                   groups=[dict(pair=list(p), chain_m=round(chain[k], 5), init_theta=rnd(vo.compose(s1f[p[0] - vo.FIRST:p[0] - vo.FIRST + vo.STRIDE])),
                                passes=[dict(theta=rnd(passes[qq][k][0]), plausible=passes[qq][k][2],
                                             advance_m=None if passes[qq][k][0] is None else round(float(np.asarray(passes[qq][k][0][:3]) @ cam.fwd), 5),
                                             **passes[qq][k][1]) for qq in range(len(passes))])
                           for k, p in enumerate(p4)],
                   stride1=[dict(pair=list(p), advance_m=None if s1[k][0] is None else round(float(np.asarray(s1[k][0][:3]) @ cam.fwd), 5),
                                 theta=rnd(s1[k][0]), plausible=s1[k][2], **s1[k][1]) for k, p in enumerate(p1)],
                   stride1_plausible_share=round(len(good) / len(p1), 3))
        vo.finalize(rec)
        rec['planar'] = planar(rec)
        f = out_dir / source / f'{name}__{sample}.json'
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_suffix('.tmp'); tmp.write_text(json.dumps(rec) + '\n'); tmp.rename(f)
        if (done + 1) % 5 == 0:
            print(f'{done + 1} clips, {time.time() - t0:.0f} s, peak {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB', flush=True)
    print(f'done in {time.time() - t0:.0f} s', flush=True)


# ------------------------------------------------------------------------------------------------------------- gate
def gt_ego(bench, name):
    gt = json.loads((bench / 'windows-dev' / name / 'gt.json').read_text())
    ego = {e['k']: e for e in gt['ego']}
    return {str(h): dict(D_m=ego[int(round(h * 10))]['xyz_m'][0], Psi_deg=math.degrees(ego[int(round(h * 10))]['yaw_rad'])) for h in HORIZONS_S}


def load_vo(bench, source, name, sample):
    f = bench / OUT_SUB / 'clips' / source / f'{name}__{sample}.json'
    return json.loads(f.read_text()) if f.is_file() else None


def gate_eval(bench, names, source, sample, h='4.0'):
    from scipy.stats import spearmanr
    rows = []
    for n in names:
        r = load_vo(bench, source, n, sample)
        if r is None:
            continue
        g = gt_ego(bench, n)
        rows.append(dict(name=n, D_vo=r['planar']['horizons'][h]['D_m'], D_gt=g[h]['D_m'], Psi_vo=r['planar']['horizons'][h]['Psi_deg'], Psi_gt=g[h]['Psi_deg'],
                         accepted_share=r['quality']['accepted_share'], travelled_vo=r['travelled_m'].get(h.rstrip('0').rstrip('.') if False else str(int(float(h))))))
    if not rows:
        return dict(n=0)
    D_vo = np.array([r['D_vo'] for r in rows]); D_gt = np.array([r['D_gt'] for r in rows])
    P_vo = np.array([r['Psi_vo'] for r in rows]); P_gt = np.array([r['Psi_gt'] for r in rows])
    tol = np.maximum(0.15 * np.abs(D_gt), 1.5)
    ratio = np.abs(D_vo - D_gt) / tol
    rho = float(spearmanr(D_vo, D_gt).correlation) if len(rows) > 2 else float('nan')
    c1 = float(np.median(ratio)); c2 = float(np.median(np.abs(P_vo - P_gt)))
    return dict(n=len(rows), horizon_s=float(h),
                criterion1_median_err_over_tol=round(c1, 4), criterion1_pass=bool(c1 <= 1.0),
                criterion2_median_abs_heading_err_deg=round(c2, 4), criterion2_pass=bool(c2 <= 3.0),
                criterion3_spearman=round(rho, 4), criterion3_pass=bool(rho >= 0.90),
                share_within_tol=round(float(np.mean(ratio <= 1)), 4), D_bias_median_m=round(float(np.median(D_vo - D_gt)), 4),
                D_abs_err_median_m=round(float(np.median(np.abs(D_vo - D_gt))), 4), D_abs_err_p90_m=round(float(np.percentile(np.abs(D_vo - D_gt), 90)), 4),
                D_rel_err_median_moving=round(float(np.median(((D_vo - D_gt) / D_gt)[D_gt > 2])), 4) if (D_gt > 2).any() else None,
                heading_bias_median_deg=round(float(np.median(P_vo - P_gt)), 4), heading_abs_err_p90_deg=round(float(np.percentile(np.abs(P_vo - P_gt), 90)), 4),
                accepted_share_median=round(float(np.median([r['accepted_share'] for r in rows])), 4), rows=rows)


def cmd_gate(args):
    from scipy.stats import spearmanr
    B = args.bench; d = B / OUT_SUB
    raw_names = [l.strip() for l in (d / 'names-ego-inter-dev.txt').read_text().splitlines() if l.strip()]
    rec_names = [l.strip() for l in (d / 'names-ego-inter-dev-gen.txt').read_text().splitlines() if l.strip()]
    res = dict(gate_file='VO-GATE.md', results={})
    for h in ('2.0', '4.0', '6.0'):
        res['results'][f'raw|{h}'] = gate_eval(B, raw_names, 'nano-raw', 'raw', h)
        res['results'][f'recon|{h}'] = gate_eval(B, rec_names, args.recon_run, 'real', h)
    r4, c4 = res['results']['raw|4.0'], res['results']['recon|4.0']
    passed = all(x.get(k, False) for x in (r4, c4) for k in ('criterion1_pass', 'criterion2_pass', 'criterion3_pass'))
    res['gate_passed'] = passed
    res['secondary'] = {f'D_{h}s': all(res['results'][f'{s}|{h}'].get('criterion1_pass', False) for s in ('raw', 'recon')) for h in ('2.0', '6.0')}
    # raw vs recon agreement on the 32 windows that have both
    both = [n for n in rec_names if load_vo(B, 'nano-raw', n, 'raw') and load_vo(B, args.recon_run, n, 'real')]
    if len(both) > 2:
        a = [load_vo(B, 'nano-raw', n, 'raw')['planar']['horizons']['4.0']['D_m'] for n in both]
        b = [load_vo(B, args.recon_run, n, 'real')['planar']['horizons']['4.0']['D_m'] for n in both]
        res['raw_vs_recon_D4'] = dict(n=len(both), spearman=round(float(spearmanr(a, b).correlation), 4), median_abs_diff_m=round(float(np.median(np.abs(np.subtract(a, b)))), 4))
    # all-window context (not gating): every DEV window measured on raw
    allraw = sorted({p.name.split('__')[0] for p in (d / 'clips' / 'nano-raw').glob('*.json')})
    res['context_all_raw_windows|4.0'] = {k: v for k, v in gate_eval(B, allraw, 'nano-raw', 'raw', '4.0').items() if k != 'rows'}
    (d / 'gate-result.json').write_text(json.dumps(res, indent=1) + '\n')
    print(json.dumps({k: ({kk: vv for kk, vv in v.items() if kk != 'rows'} if isinstance(v, dict) and 'rows' in v else v) for k, v in res['results'].items()}, indent=1))
    print(json.dumps(dict(gate_passed=passed, secondary=res['secondary'], raw_vs_recon=res.get('raw_vs_recon_D4'), context=res['context_all_raw_windows|4.0']), indent=1))


# ---------------------------------------------------------------------------------------------------------- history
def planar_history(rec):
    """Ego positions over the real history (group ends t = -2.0, -1.6, ..., 0.0) in the t0 ego frame, integrated from
    frame 0 with the same per-group advance / yaw rule as planar(); the least-squares forward speed and yaw rate."""
    R_ec = np.asarray(rec['camera']['ego_R_cam'], float)
    psi, x, y = 0.0, 0.0, 0.0
    pts = [(-2.0, 0.0, 0.0, 0.0)]
    for k, g in enumerate(rec['groups']):
        th = g['passes'][0]['theta'] if g.get('accepted_m') is not None and g['passes'][0]['theta'] is not None else g['init_theta']
        yaw = ego_yaw(np.asarray(th, float), R_ec) if th is not None else 0.0
        mid = psi + yaw / 2; adv = g['final_m']
        x += adv * math.cos(mid); y += adv * math.sin(mid); psi += yaw
        pts.append((round(-2.0 + 0.4 * (k + 1), 1), x, y, psi))
    _, X0, Y0, P0 = pts[-1]                      # the t0 pose in frame-0 coordinates
    c, s_ = math.cos(-P0), math.sin(-P0)
    traj = [dict(t=t, x=c * (px - X0) - s_ * (py - Y0), y=s_ * (px - X0) + c * (py - Y0), psi_deg=math.degrees(pp - P0)) for t, px, py, pp in pts]
    T = np.array([q['t'] for q in traj]); A = np.stack([np.ones_like(T), T], 1)
    (_, vx), *_ = np.linalg.lstsq(A, np.array([q['x'] for q in traj]), rcond=None)
    (_, wz), *_ = np.linalg.lstsq(A, np.array([q['psi_deg'] for q in traj]), rcond=None)
    return dict(trajectory=traj, ls_speed_mps=float(vx), ls_yaw_rate_dps=float(wz))


def measure_history(args):
    """VO over the real history only (frames 0 -> 20: stride-1 chain 0->1 ... 19->20, groups 0->4 ... 16->20, one
    refinement), for the constant-velocity baseline of the ego family. Separate from the gated instrument (its outputs go
    to ego-vo/clips-hist/); the gated odometry is unchanged."""
    import cv2, torch, torch.nn.functional as tF
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
    torch.set_num_threads(2); cv2.setNumThreads(1)
    names = [l.strip() for l in args.names_file.read_text().splitlines() if l.strip()]
    dev = set(l.strip() for l in (args.bench / 'names-dev.txt').read_text().splitlines() if l.strip())
    if not set(names) <= dev and not (args.allow_test and (args.bench / 'instrument-frozen.txt').is_file()):
        sys.exit('refusing: names outside names-dev.txt (DEV only until protocol-v1.1 is frozen)')
    wdir = 'windows-dev' if set(names) <= dev else 'windows-test'
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, args.gpu_mem_gib * 2 ** 30 / total), 0)
    dev_name = 'cuda:0'
    out_dir = args.bench / OUT_SUB / 'clips-hist'
    jobs = [j for j in clip_list(args.bench, names, args.sources, wdir) if j[2] in ('raw', 'real') and j[3].is_file() and j[4].is_file()
            and not (out_dir / j[1] / f'{j[0]}__{j[2]}.json').exists()]
    part, nparts = (int(x) for x in args.part.split('/'))
    jobs = jobs[part::nparts]
    print(f'{len(jobs)} history clips to measure on {dev_name}', flush=True)
    model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).eval().to(dev_name)
    vv, uu = np.mgrid[0:H, 0:W]

    def prep(imgs):
        x = torch.from_numpy(np.ascontiguousarray(imgs)).permute(0, 3, 1, 2).float().div(127.5).sub(1).to(dev_name)
        return tF.interpolate(x, scale_factor=vo.UP, mode='bicubic', align_corners=False).clamp(-1, 1)

    def raft(a_imgs, b_imgs):
        out = []
        with torch.no_grad():
            for st in range(0, len(a_imgs), args.batch):
                while True:
                    free, _ = torch.cuda.mem_get_info(0)
                    if free - max(0, 1.6 * 2 ** 30 - torch.cuda.memory_reserved(0)) >= args.min_free_mib * 2 ** 20:
                        break
                    torch.cuda.empty_cache(); time.sleep(20)
                fl = model(prep(a_imgs[st:st + args.batch]), prep(b_imgs[st:st + args.batch]), num_flow_updates=vo.ITERS)[-1]
                fl = tF.interpolate(fl, size=(H, W), mode='area') / vo.UP
                out.append(fl.cpu().numpy())
        return np.concatenate(out)

    def fb_ok(fw, bw):
        B = vo.bilinear(np.nan_to_num(bw), (uu + np.nan_to_num(fw[0])).ravel(), (vv + np.nan_to_num(fw[1])).ravel()).reshape(2, H, W)
        e2 = ((fw + B) ** 2).sum(0); m2 = (fw ** 2).sum(0) + (B ** 2).sum(0)
        return np.isfinite(e2) & (e2 < 0.01 * m2 + 0.5)

    def plaus(t, info, cam):
        info['_fwd'] = cam.fwd; ok = vo.plausible(t, info); info.pop('_fwd'); return ok

    rnd = lambda th: None if th is None else [round(float(v), 6) for v in th]
    t0 = time.time()
    for done, job in enumerate(jobs):
        name, source, sample, video, det_file = job
        fr = frames_of(video)
        if len(fr) != 81:
            continue
        cam = vo.Camera(**camera_of(args.bench, name, wdir)); grid = Grid(cam)
        dj = json.loads(det_file.read_text())
        dets = {t: d for t, d in zip(dj['frames'], dj['detections'])}
        p1 = [(i, i + 1) for i in range(0, 20)]
        fw = raft(fr[[i for i, _ in p1]], fr[[j for _, j in p1]]); bw = raft(fr[[j for _, j in p1]], fr[[i for i, _ in p1]])
        s1, th = [], None
        for k, (i, j) in enumerate(p1):
            t, info = fit_pair(cam, grid, fw[k], fb_ok(fw[k], bw[k]), dets.get(i), dets.get(j), theta0=th)
            th = t if t is not None else th
            s1.append((t, info, plaus(t, info, cam)))
        good = [k for k, (_, _, ok) in enumerate(s1) if ok]
        s1f = [s1[k][0] if s1[k][2] else (s1[min(good, key=lambda r: abs(r - k))][0] if good else np.zeros(6)) for k in range(len(p1))]
        adv1 = np.array([float(np.asarray(t[:3]) @ cam.fwd) for t in s1f])
        p4 = [(i, i + vo.STRIDE) for i in range(0, 20, vo.STRIDE)]
        cur = [vo.compose(s1f[i:i + vo.STRIDE]) for i, _ in p4]
        chain = [float(adv1[i:i + vo.STRIDE].sum()) for i, _ in p4]
        warps, warped = [], []
        for k, (i, j) in enumerate(p4):
            Wf = vo.model_flow(cam, cur[k], shape=(H, W)); warps.append(Wf)
            warped.append(cv2.remap(np.ascontiguousarray(fr[j]), (uu + Wf[0]).astype(np.float32), (vv + Wf[1]).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE))
        warped = np.stack(warped)
        Rf = raft(fr[[i for i, _ in p4]], warped); Rb = raft(warped, fr[[i for i, _ in p4]])
        res = []
        for k, (i, j) in enumerate(p4):
            Ws = vo.bilinear(warps[k], (uu + np.nan_to_num(Rf[k][0])).ravel(), (vv + np.nan_to_num(Rf[k][1])).ravel()).reshape(2, H, W)
            t, info = fit_pair(cam, grid, Rf[k] + Ws, fb_ok(Rf[k], Rb[k]), dets.get(i), dets.get(j), theta0=cur[k])
            res.append((t, info, plaus(t, info, cam)))
        rec = dict(name=name, source=source, sample=sample, video=str(video), camera=camera_of(args.bench, name, wdir),
                   groups=[dict(pair=list(p), chain_m=round(chain[k], 5), init_theta=rnd(cur[k]),
                                passes=[dict(theta=rnd(res[k][0]), plausible=res[k][2],
                                             advance_m=None if res[k][0] is None else round(float(np.asarray(res[k][0][:3]) @ cam.fwd), 5), **res[k][1])])
                           for k, p in enumerate(p4)],
                   stride1_plausible_share=round(len(good) / len(p1), 3))
        vo.finalize(rec)
        rec.pop('travelled_m', None); rec.pop('diagnostics', None)
        rec['history'] = planar_history(rec)
        f = out_dir / source / f'{name}__{sample}.json'
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_suffix('.tmp'); tmp.write_text(json.dumps(rec) + '\n'); tmp.rename(f)
        if (done + 1) % 10 == 0:
            print(f'{done + 1} clips, {time.time() - t0:.0f} s', flush=True)
    print(f'done in {time.time() - t0:.0f} s', flush=True)


def cmd_history_check(args):
    """Ego history (VO over frames 0 -> 20) against the gt poses (k = -20..0) and, as a constant-velocity forecast, against
    the gt forward displacement at 2 / 4 s and the gated odometry's own reading of the same clip."""
    from scipy.stats import spearmanr
    out = {}
    for source, sample, names_file in (('nano-raw', 'raw', 'names-ego-inter-dev.txt'), (args.recon_run, 'real', 'names-ego-inter-dev-gen.txt')):
        names = [l.strip() for l in (args.bench / OUT_SUB / names_file).read_text().splitlines() if l.strip()]
        rows = []
        for n in names:
            f = args.bench / OUT_SUB / 'clips-hist' / source / f'{n}__{sample}.json'
            if not f.is_file():
                continue
            h = json.loads(f.read_text())['history']
            gt = json.loads((args.bench / 'windows-dev' / n / 'gt.json').read_text())
            ego = {e['k']: e for e in gt['ego']}
            ks = list(range(-20, 1, 4)); T = np.array(ks) / 10.0; A = np.stack([np.ones_like(T), T], 1)
            (_, gv), *_ = np.linalg.lstsq(A, np.array([ego[k]['xyz_m'][0] for k in ks]), rcond=None)
            (_, gw), *_ = np.linalg.lstsq(A, np.degrees(np.array([ego[k]['yaw_rad'] for k in ks])), rcond=None)
            fut = load_vo(args.bench, source, n, sample)
            rows.append(dict(name=n, v_vo=h['ls_speed_mps'], v_gt=float(gv), w_vo=h['ls_yaw_rate_dps'], w_gt=float(gw),
                             D4_gt=ego[40]['xyz_m'][0], D2_gt=ego[20]['xyz_m'][0], D4_vo=None if fut is None else fut['planar']['horizons']['4.0']['D_m']))
        if not rows:
            continue
        v_vo = np.array([r['v_vo'] for r in rows]); v_gt = np.array([r['v_gt'] for r in rows])
        w_vo = np.array([r['w_vo'] for r in rows]); w_gt = np.array([r['w_gt'] for r in rows])
        D4 = np.array([r['D4_gt'] for r in rows])
        res = dict(n=len(rows), speed_err_median_abs=float(np.median(np.abs(v_vo - v_gt))), speed_err_p90_abs=float(np.percentile(np.abs(v_vo - v_gt), 90)),
                   speed_bias=float(np.median(v_vo - v_gt)), speed_spearman=float(spearmanr(v_vo, v_gt).correlation),
                   yaw_rate_err_median_abs=float(np.median(np.abs(w_vo - w_gt))), yaw_rate_err_p90_abs=float(np.percentile(np.abs(w_vo - w_gt), 90)),
                   cv4_vs_gt_median_abs_A=float(np.median(np.abs(4 * v_vo - D4))), cv4_vs_gt_median_abs_B=float(np.median(np.abs(4 * v_gt - D4))))
        fr = [r for r in rows if r['D4_vo'] is not None]
        if fr:
            res['cv4_vs_vo_reading_median_abs_A'] = float(np.median([abs(4 * r['v_vo'] - r['D4_vo']) for r in fr]))
            res['cv4_vs_vo_reading_median_abs_B'] = float(np.median([abs(4 * r['v_gt'] - r['D4_vo']) for r in fr]))
        out[f'{source}|{sample}'] = res
    (args.bench / OUT_SUB / 'history-check.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('measure')
    p.add_argument('--bench', type=Path, required=True); p.add_argument('--names-file', type=Path, required=True)
    p.add_argument('--sources', nargs='+', required=True); p.add_argument('--batch', type=int, default=4)
    p.add_argument('--gpu-mem-gib', type=float, default=2.5); p.add_argument('--min-free-mib', type=float, default=1500)
    p.add_argument('--part', default='0/1')
    p.add_argument('--allow-test', action='store_true', help='TEST windows; refused unless BENCH/instrument-frozen.txt exists')
    p = sub.add_parser('gate')
    p.add_argument('--bench', type=Path, required=True); p.add_argument('--recon-run', default='dev-recon')
    p = sub.add_parser('history')
    p.add_argument('--bench', type=Path, required=True); p.add_argument('--names-file', type=Path, required=True)
    p.add_argument('--sources', nargs='+', required=True); p.add_argument('--batch', type=int, default=4)
    p.add_argument('--gpu-mem-gib', type=float, default=2.5); p.add_argument('--min-free-mib', type=float, default=1500)
    p.add_argument('--part', default='0/1'); p.add_argument('--allow-test', action='store_true')
    p = sub.add_parser('history-check')
    p.add_argument('--bench', type=Path, required=True); p.add_argument('--recon-run', default='dev-recon')
    args = ap.parse_args()
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    {'measure': measure, 'gate': cmd_gate, 'history': measure_history, 'history-check': cmd_history_check}[args.cmd](args)


if __name__ == '__main__':
    main()
