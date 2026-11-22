#!/usr/bin/env python3
"""Ego forward motion read from road-surface optical flow, the same way on real, reconstructed and generated 256x320 clips
of the world-model calibration study (2026-11-01; replaces the LK/Farneback attempt in measure_ego_motion.py).

Geometry. Road pixels 3-35 m ahead (rows 150-266, outside detected vehicles: score >= 0.3, boxes padded 4 px, in both
frames) are back-projected onto the flat road of the window's camera: AV2 ring_front_center intrinsics scaled to 256x320
(pinhole: adding the AV2 radial terms raised the flow residuals on DEV), mounting rotation and height from
egovehicle_SE3_sensor (official devkit loader), road 0.33 m below the ego origin (median LiDAR ground height at the 116 DEV
anchors), i.e. a camera ~1.74 m above the road. (The lead-gap instrument's d = 412.8 / (y - 161.4) was fitted to box
bottoms against rear-face depth; for road texture it reads ~10 % short.)

Flow. RAFT-large (torchvision DEFAULT weights) on the frame without hood and side bars (rows 8-272, cols 12-244), input
upsampled x2, 20 updates, forward and backward; forward-backward check |F + B|^2 < 0.01 (|F|^2 + |B|^2) + 0.5.

Fit. Per frame pair a robust (Cauchy IRLS Gauss-Newton) fit of the camera's 6-DoF motion to the flow endpoints, plane
fixed. A fit is plausible if |roll| <= 1 deg, residual scale <= 2 px, >= 80 points, advance > -0.5 m.

Two time scales, because the video tokenizer packs 4 frames into one latent: decoded fine road texture does not move
consistently inside a latent group, so consecutive-frame flow under-reads reconstructed / generated clips (~15 % on
DEV), while 4-frame pairs aligned at the anchor (16->20->24->...->80) read them right but lose the near road at speed.
  1. stride-1 chain 16->17->...->80 (plausible fits; gaps filled from the nearest plausible pair);
  2. per 4-frame group, the four stride-1 motions are composed into an initial motion; frame j is warped by it, RAFT is
     run on (i, warped j) and back, the residual flow is composed with the warp and the motion refitted (two passes);
  3. a group's advance is the refined stride-4 value if plausible with >= 200 points, otherwise the chain's value times
     the clip's own ratio refined / chain (median over its supported groups; 1 if fewer than 3).
The advance is the camera displacement along the ego's forward axis; distance travelled 0..T sums the groups after the
anchor (T = 1 s interpolated between 0.8 and 1.2 s). Quality fields per clip: share of supported groups, ratio.

  measure_ego_motion_raft.py measure --study S --names-file F --sources real-raw real-recon dev-g6-s0-3 --out DIR
  measure_ego_motion_raft.py report  --study S --names-file F --out DIR [--gen-run dev-g6-s0-3]

`measure` (GPU 0, ~2 GB with the default batch) writes one JSON per clip into DIR/clips/<source>/ and is resumable
(--part i/n splits the clip list between processes); `report` validates real-raw / real-recon against
instrument/gt-ego-dev.json and, with --gen-run, writes the exploratory ego / lead decomposition of the 4 s gap
(lead advance = gap(4) - gap(0) + ego advance).
"""
from __future__ import annotations

import argparse, json, os, queue, subprocess, threading, time
from pathlib import Path

import numpy as np

GROUND_BELOW_EGO_M = 0.33          # road surface below the AV2 ego-vehicle origin (median LiDAR ground z at the DEV anchors)
CROP = (8, 272, 12, 244)           # RAFT input rows / cols (no hood, no side bars)
UP, ITERS = 2, 20                  # input upsampling, RAFT refinement steps
STRIDE = 4                         # frames per group: one tokenizer latent frame, aligned at the anchor (frame 20)
FIRST = 16                         # first frame used: one group of history before the anchor
MIN_POINTS_S4 = 80                 # support needed to trust a refined stride-4 group
REFINE_PASSES = 1                  # warped refinements per stride-4 group (a second pass drifted on decoded texture on DEV)
REJECT_BAND = 1.5                  # a refined group whose refined / chain ratio is off the clip's median ratio by more is rejected
RULE = 'reject'                    # 'reject' (default), 'refined' (no rejection) or 'hybrid' (gaps from the ratio-scaled chain)
ROLL_MAX_DEG, SIGMA_MAX_PX, ADV_MIN_M = 1.0, 2.0, -0.5
ROI_ROWS, ROI_COLS, TGT_ROW_MAX = (150, 266), (14, 241), 268
DIST_M = (3.0, 35.0)               # ground distance of the fitted pixels
DET_SCORE, DET_PAD = 0.3, 4
MIN_POINTS = 80
HORIZONS = (1, 2, 4, 6)
AV2_ROOT = Path(os.environ.get('DOB_ROOT', '.'), 'datasets/av2_sensor')


# ----------------------------------------------------------------------------------------------------------- geometry
def rodrigues(w):
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3)
    k = np.asarray(w) / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


class Camera:
    """Pinhole 256x320 camera of a window with the road plane below it (camera coordinates: x right, y down, z forward)."""

    def __init__(self, K256, ego_R_cam, ego_t_cam):
        K = np.asarray(K256, float)
        self.fx, self.fy, self.cx, self.cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        R = np.asarray(ego_R_cam, float)
        self.n_up = R.T @ np.array([0.0, 0.0, 1.0])       # ego up axis in camera coordinates
        self.fwd = R.T @ np.array([1.0, 0.0, 0.0])        # ego forward axis in camera coordinates
        self.h = float(ego_t_cam[2]) + GROUND_BELOW_EGO_M  # camera height above the road

    def rays(self, u, v):
        return np.stack([(u - self.cx) / self.fx, (v - self.cy) / self.fy, np.ones_like(u)], 1)

    def ground(self, ray):
        den = ray @ self.n_up
        ok = den < -1e-3
        return ray * np.where(ok, -self.h / np.where(ok, den, -1.0), np.nan)[:, None]


def camera_for(study: Path, name: str, windows_dir: str = 'windows-dev'):
    """The window's camera from the AV2 calibration (official devkit loader) and its 256x320 intrinsics."""
    from av2.geometry.camera.pinhole_camera import PinholeCamera
    lead = json.loads((study / windows_dir / name / 'lead.json').read_text())
    cam = PinholeCamera.from_feather(AV2_ROOT / lead['split'] / lead['log'], 'ring_front_center')
    return dict(K256=lead['intrinsics_256x320'], ego_R_cam=cam.ego_SE3_cam.rotation.tolist(), ego_t_cam=cam.ego_SE3_cam.translation.tolist())


def project(P, th):
    Q = (P - th[:3]) @ rodrigues(th[3:6]).T
    return Q[:, :2] / Q[:, 2:3], Q[:, 2]


def fit_motion(P, tgt, f, theta0=None, iters=20, sigma_floor=0.15, kc=2.0):
    """Robust 6-DoF camera motion (C, w): X_j = R(w) (X_i - C). P ground points (frame i), tgt normalised targets."""
    th = np.zeros(6) if theta0 is None else np.array(theta0, float)
    eps = np.array([1e-4] * 3 + [1e-6] * 3)
    sig = sigma_floor
    for _ in range(iters):
        pr, z = project(P, th)
        r = (pr - tgt) * f
        ok = np.isfinite(r).all(1) & (z > 0.3)
        rn = np.linalg.norm(np.nan_to_num(r), axis=1)
        sig = max(float(np.median(rn[ok])) / 1.1774, sigma_floor) if ok.sum() > 10 else sigma_floor
        w = 1.0 / (1.0 + (rn / (kc * sig)) ** 2); w[~ok] = 0
        J = np.zeros((len(P), 2, 6))
        for k in range(6):
            d = np.zeros(6); d[k] = eps[k]
            J[:, :, k] = (project(P, th + d)[0] - project(P, th - d)[0]) * f / (2 * eps[k])
        Jo, ro, wo = np.nan_to_num(J[ok]), r[ok], w[ok]
        A = np.einsum('n,nck,ncl->kl', wo, Jo, Jo); b = np.einsum('n,nck,nc->k', wo, Jo, ro)
        step = np.linalg.solve(A + 1e-9 * np.eye(6), -b)
        th += step
        if np.abs(step[:3]).max() < 1e-5 and np.abs(step[3:]).max() < 1e-7:
            break
    pr, _ = project(P, th)
    rr = np.linalg.norm((pr - tgt) * f, axis=1)
    return th, dict(sigma_px=round(float(sig), 4), median_residual_px=round(float(np.nanmedian(rr)), 4))


def model_flow(cam: Camera, th, shape=(320, 256)):
    """Flow of every pixel under the fitted motion: ground plane below the horizon, rotation only above it."""
    vv, uu = np.mgrid[0:shape[0], 0:shape[1]]
    uu = uu.ravel().astype(float); vv = vv.ravel().astype(float)
    ray = cam.rays(uu, vv)
    den = ray @ cam.n_up; g = den < -0.02
    X = np.where(g[:, None], ray * (-cam.h / np.where(g, den, -1.0))[:, None] - th[:3], ray)
    Q = X @ rodrigues(th[3:6]).T
    W = np.stack([cam.fx * Q[:, 0] / Q[:, 2] + cam.cx - uu, cam.fy * Q[:, 1] / Q[:, 2] + cam.cy - vv])
    return np.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0).reshape(2, *shape).astype(np.float32)


def bilinear(img, x, y):
    C, H, W = img.shape
    ok = (x >= 0) & (x <= W - 1) & (y >= 0) & (y <= H - 1)
    xc = np.clip(x, 0, W - 1.001); yc = np.clip(y, 0, H - 1.001)
    x0 = np.floor(xc).astype(int); y0 = np.floor(yc).astype(int); dx = xc - x0; dy = yc - y0
    out = img[:, y0, x0] * (1 - dx) * (1 - dy) + img[:, y0, x0 + 1] * dx * (1 - dy) + img[:, y0 + 1, x0] * (1 - dx) * dy + img[:, y0 + 1, x0 + 1] * dx * dy
    out[:, ~ok] = np.nan
    return out


def box_mask(dets, shape=(320, 256)):
    m = np.zeros(shape, bool)
    for d in dets or []:
        if d[4] < DET_SCORE or d[1] > 265:   # a box starting in the hood zone is the ego hood
            continue
        x1, y1, x2, y2 = d[:4]
        m[max(0, int(y1) - DET_PAD):min(shape[0], int(np.ceil(y2)) + DET_PAD), max(0, int(x1) - DET_PAD):min(shape[1], int(np.ceil(x2)) + DET_PAD)] = True
    return m


class Grid:
    """The fitted road pixels of a camera (fixed per window)."""

    def __init__(self, cam: Camera, step=2):
        vv, uu = np.mgrid[ROI_ROWS[0]:ROI_ROWS[1] + 1:step, ROI_COLS[0]:ROI_COLS[1] + 1:step]
        self.u = uu.ravel().astype(float); self.v = vv.ravel().astype(float)
        self.iu = self.u.astype(int); self.iv = self.v.astype(int)
        self.P = cam.ground(cam.rays(self.u, self.v))
        dist = self.P @ cam.fwd
        self.base = np.isfinite(dist) & (dist >= DIST_M[0]) & (dist <= DIST_M[1])


def fit_pair(cam: Camera, grid: Grid, flow, fb_ok, dets_i, dets_j, theta0=None):
    """flow (2,320,256) i->j (nan outside the crop); fb_ok (320,256) bool. Returns (theta, info) or (None, info)."""
    F = flow[:, grid.iv, grid.iu].T
    tu, tv = grid.u + F[:, 0], grid.v + F[:, 1]
    ok = grid.base & np.isfinite(F).all(1) & fb_ok[grid.iv, grid.iu] & ~box_mask(dets_i)[grid.iv, grid.iu]
    ok &= (tu >= ROI_COLS[0]) & (tu <= ROI_COLS[1]) & (tv <= TGT_ROW_MAX) & (tv >= ROI_ROWS[0] - 10)
    ok &= ~box_mask(dets_j)[np.clip(np.round(np.nan_to_num(tv)).astype(int), 0, 319), np.clip(np.round(np.nan_to_num(tu)).astype(int), 0, 255)]
    n = int(ok.sum())
    if n < MIN_POINTS:
        return None, dict(points=n)
    tgt = np.stack([(tu[ok] - cam.cx) / cam.fx, (tv[ok] - cam.cy) / cam.fy], 1)
    th, info = fit_motion(grid.P[ok], tgt, cam.fy, theta0=theta0)
    return th, dict(points=n, **info)


def plausible(th, info):
    return bool(th is not None and abs(np.degrees(th[5])) <= ROLL_MAX_DEG and info.get('sigma_px', 9.0) <= SIGMA_MAX_PX
                and info.get('points', 0) >= MIN_POINTS and float(th[:3] @ info['_fwd']) > ADV_MIN_M)


def compose(thetas):
    """Chain motions X_{k+1} = R_k (X_k - C_k) into one (C, w) over the whole span (C in the first frame's coordinates)."""
    from scipy.spatial.transform import Rotation
    R = np.eye(3); C = np.zeros(3)
    for t in thetas:
        C = C + R.T @ np.asarray(t[:3]); R = rodrigues(t[3:6]) @ R
    return np.concatenate([C, Rotation.from_matrix(R).as_rotvec()])


def travelled_from_groups(adv_groups):
    """Advances of the 15 groups after the anchor (frames 20->24 ... 76->80) -> distance travelled at the horizons."""
    cum = np.concatenate([[0.0], np.cumsum(adv_groups)]); t = np.arange(len(cum)) * STRIDE / 10.0
    return {str(h): round(float(np.interp(h, t, cum)), 4) for h in HORIZONS}


# ------------------------------------------------------------------------------------------------------------- measure
def frames_of(path):
    raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-threads', '1', '-i', str(path), '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'])
    return np.frombuffer(raw, np.uint8).reshape(-1, 320, 256, 3)


def clip_list(study: Path, names, sources, windows_dir='windows-dev', recon_run='recon-dev'):
    det = study / 'detections-every1'
    for name in names:
        for source in sources:
            if source == 'real-raw':
                yield name, 'real-raw', study / windows_dir / name / 'real.mp4', det / 'real-raw' / f'{name}.json'
            elif source == 'real-recon':
                yield name, 'real-recon', study / 'runs' / recon_run / 'native' / f'{name}-recon' / 'vision.mp4', det / 'real-recon' / f'{name}.json'
            else:
                for d in sorted((study / 'runs' / source / 'native').glob(f'{name}-s*')):
                    yield name, f'{source}-{d.name.rsplit("-", 1)[1]}', d / 'vision.mp4', det / source / f'{d.name}.json'


def measure(args):
    import cv2, torch, torch.nn.functional as tF
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
    torch.set_num_threads(2); cv2.setNumThreads(1)
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.gpu_mem_gib / (torch.cuda.get_device_properties(0).total_memory / 2 ** 30), 0)
    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    names = [l.strip() for l in args.names_file.read_text().splitlines() if l.strip()]
    out_dir = args.out / 'clips'
    jobs = [j for j in clip_list(args.study, names, args.sources, args.windows_dir, args.recon_run)
            if j[2].is_file() and not (out_dir / j[1].rsplit('-s', 1)[0] / f'{j[0]}__{j[1]}.json').exists()]
    part, nparts = (int(x) for x in args.part.split('/'))
    jobs = jobs[part::nparts]
    print(f'{len(jobs)} clips to measure on {dev}', flush=True)
    cams = {}
    model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).eval().to(dev)
    r0, r1, c0, c1 = CROP
    q: queue.Queue = queue.Queue(maxsize=3)

    def reader():
        for job in jobs:
            q.put((job, frames_of(job[2])))
        q.put(None)
    threading.Thread(target=reader, daemon=True).start()

    def prep(imgs):
        x = torch.from_numpy(np.ascontiguousarray(imgs)).permute(0, 3, 1, 2).float().div(127.5).sub(1).to(dev)[:, :, r0:r1, c0:c1]
        return tF.interpolate(x, scale_factor=UP, mode='bicubic', align_corners=False).clamp(-1, 1)

    def raft(a_imgs, b_imgs):
        out = []
        with torch.no_grad():
            for s in range(0, len(a_imgs), args.batch):
                fl = model(prep(a_imgs[s:s + args.batch]), prep(b_imgs[s:s + args.batch]), num_flow_updates=ITERS)[-1]
                fl = tF.interpolate(fl, size=(r1 - r0, c1 - c0), mode='area') / UP
                full = np.full((fl.shape[0], 2, 320, 256), np.nan, np.float32); full[:, :, r0:r1, c0:c1] = fl.cpu().numpy()
                out.append(full)
        return np.concatenate(out)

    vv, uu = np.mgrid[0:320, 0:256]

    def fb_ok(fw, bw):
        B = bilinear(np.nan_to_num(bw), (uu + np.nan_to_num(fw[0])).ravel(), (vv + np.nan_to_num(fw[1])).ravel()).reshape(2, 320, 256)
        e2 = ((fw + B) ** 2).sum(0); m2 = (fw ** 2).sum(0) + (B ** 2).sum(0)
        return np.isfinite(e2) & (e2 < 0.01 * m2 + 0.5)

    def rnd(th):
        return None if th is None else [round(float(x), 6) for x in th]

    t0 = time.time()
    for done, item in enumerate(iter(q.get, None)):
        (name, key, video, det_file), fr = item
        if len(fr) != 81:
            print(f'skip {key} {name}: {len(fr)} frames', flush=True); continue
        if name not in cams:
            cams[name] = camera_for(args.study, name, args.windows_dir)
        cam = Camera(**cams[name]); grid = Grid(cam)
        dj = json.loads(det_file.read_text()) if det_file.is_file() else None
        dets = {t: d for t, d in zip(dj['ticks'], dj['detections'])} if dj else {}
        # 1. stride-1 chain
        p1 = [(i, i + 1) for i in range(FIRST, 80)]
        fw = raft(fr[[i for i, _ in p1]], fr[[j for _, j in p1]]); bw = raft(fr[[j for _, j in p1]], fr[[i for i, _ in p1]])
        s1, th = [], None
        for k, (i, j) in enumerate(p1):
            t, info = fit_pair(cam, grid, fw[k], fb_ok(fw[k], bw[k]), dets.get(i), dets.get(j), theta0=th)
            th = t if t is not None else th
            info['_fwd'] = cam.fwd; ok = plausible(t, info); info.pop('_fwd')
            s1.append((t, info, ok))
        good = [k for k, (_, _, ok) in enumerate(s1) if ok]
        s1f = [s1[k][0] if s1[k][2] else (s1[min(good, key=lambda r: abs(r - k))][0] if good else np.zeros(6)) for k in range(len(p1))]
        adv1 = np.array([float(np.asarray(t[:3]) @ cam.fwd) for t in s1f])
        # 2. stride-4 groups initialised by the composed chain, refined twice on warped frames
        p4 = [(i, i + STRIDE) for i in range(FIRST, 80, STRIDE)]
        cur = [compose(s1f[i - FIRST:i - FIRST + STRIDE]) for i, _ in p4]
        chain = [float(adv1[i - FIRST:i - FIRST + STRIDE].sum()) for i, _ in p4]
        passes = []
        for _ in range(REFINE_PASSES):
            warps, warped = [], []
            for k, (i, j) in enumerate(p4):
                W = model_flow(cam, cur[k]); warps.append(W)
                warped.append(cv2.remap(np.ascontiguousarray(fr[j]), (uu + W[0]).astype(np.float32), (vv + W[1]).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE))
            warped = np.stack(warped)
            R = raft(fr[[i for i, _ in p4]], warped); Rb = raft(warped, fr[[i for i, _ in p4]])
            res = []
            for k, (i, j) in enumerate(p4):
                Ws = bilinear(warps[k], (uu + np.nan_to_num(R[k][0])).ravel(), (vv + np.nan_to_num(R[k][1])).ravel()).reshape(2, 320, 256)
                t, info = fit_pair(cam, grid, R[k] + Ws, fb_ok(R[k], Rb[k]), dets.get(i), dets.get(j), theta0=cur[k])
                info['_fwd'] = cam.fwd; ok = plausible(t, info); info.pop('_fwd')
                res.append((t, info, ok))
            passes.append(res)
            cur = [t if ok else cur[k] for k, (t, info, ok) in enumerate(res)]
        rec = dict(name=name, source=key, video=str(video), detections=str(det_file) if dj else None, camera=cams[name],
                   groups=[dict(pair=list(p), chain_m=round(chain[k], 5), init_theta=rnd(compose(s1f[p[0] - FIRST:p[0] - FIRST + STRIDE])),
                                passes=[dict(theta=rnd(passes[q][k][0]), plausible=passes[q][k][2],
                                             advance_m=None if passes[q][k][0] is None else round(float(np.asarray(passes[q][k][0][:3]) @ cam.fwd), 5),
                                             **passes[q][k][1]) for q in range(len(passes))])
                           for k, p in enumerate(p4)],
                   stride1=[dict(pair=list(p), advance_m=None if s1[k][0] is None else round(float(np.asarray(s1[k][0][:3]) @ cam.fwd), 5),
                                 theta=rnd(s1[k][0]), plausible=s1[k][2], **s1[k][1]) for k, p in enumerate(p1)],
                   stride1_plausible_share=round(len(good) / len(p1), 3))
        finalize(rec)
        f = out_dir / key.rsplit('-s', 1)[0] / f'{name}__{key}.json'
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_suffix('.tmp'); tmp.write_text(json.dumps(rec) + '\n'); tmp.rename(f)
        if (done + 1) % 10 == 0:
            peak = torch.cuda.max_memory_allocated() / 2 ** 30 if dev != 'cpu' else 0
            print(f'{done + 1} clips, {time.time() - t0:.0f} s, peak {peak:.2f} GiB', flush=True)
    print(f'done in {time.time() - t0:.0f} s', flush=True)


def finalize(rec, rule=None, min_points=None):
    """Distance travelled from a clip record. A stride-4 group is accepted if its first warped refinement is plausible
    with >= MIN_POINTS_S4 points and (RULE 'reject') its refined / chain ratio is within REJECT_BAND of the clip's median
    ratio (needs >= 3 groups with a chain advance > 0.3 m). Groups not accepted are interpolated from the accepted ones
    ('reject', 'refined') or take the chain times the median ratio ('hybrid'). A clip without any accepted group after the
    anchor falls back to the chain times the median ratio (or the chain alone) and is flagged in quality['rule']."""
    rule = rule or RULE; min_points = MIN_POINTS_S4 if min_points is None else min_points
    groups = rec['groups']; chain = np.array([g['chain_m'] for g in groups])
    first = [g['passes'][0] for g in groups]
    s4 = [x['advance_m'] if x['plausible'] and x.get('points', 0) >= min_points else None for x in first]
    both = [k for k in range(len(groups)) if s4[k] is not None and chain[k] > 0.3]
    rmed = float(np.median([s4[k] / chain[k] for k in both])) if len(both) >= 3 else None
    rejected = 0
    if rule == 'reject' and rmed is not None and rmed > 0:
        keep = []
        for k, v in enumerate(s4):
            bad = v is not None and chain[k] > 0.3 and (v <= 0 or abs(np.log(v / chain[k] / rmed)) > np.log(REJECT_BAND))
            rejected += int(bad); keep.append(None if bad else v)
        s4 = keep
    if not any(v is not None for v in s4[1:]):
        final = chain * (rmed if rmed else 1.0); how = 'chain (no accepted group)'
    elif rule == 'hybrid':
        final = np.array([s4[k] if s4[k] is not None else (rmed or 1.0) * chain[k] for k in range(len(groups))]); how = 'hybrid'
    else:
        final = _fill_none(s4); how = rule
    after = slice(1, None)  # group 0 is the history group 16->20
    for k, g in enumerate(groups):
        g['accepted_m'] = None if s4[k] is None else round(float(s4[k]), 5); g['final_m'] = round(float(final[k]), 5)
    rec['travelled_m'] = travelled_from_groups(np.asarray(final)[after])
    rec['quality'] = dict(rule=how, min_points=min_points, accepted_share=round(float(np.mean([v is not None for v in s4[after]])), 3),
                          rejected_groups=rejected, ratio_refined_over_chain=None if rmed is None else round(rmed, 4), ratio_groups=len(both),
                          stride1_plausible_share=rec.get('stride1_plausible_share'))
    rec['diagnostics'] = dict(chain_only_travelled_m=travelled_from_groups(chain[after]))
    return rec


def recombine(args):
    """Re-apply finalize() (current RULE / MIN_POINTS_S4) to every clip record under OUT/clips (no re-measurement)."""
    n = 0
    for f in sorted((args.out / 'clips').glob('*/*.json')):
        rec = json.loads(f.read_text())
        rec.setdefault('stride1_plausible_share', (rec.get('quality') or {}).get('stride1_plausible_share'))
        for g in rec['groups']:   # records written before advance_m was stored per pass
            for x in g['passes']:
                if 'advance_m' not in x:
                    x['advance_m'] = None if x['theta'] is None else round(float(np.asarray(x['theta'][:3]) @ Camera(**rec['camera']).fwd), 5)
        finalize(rec)
        tmp = f.with_suffix('.tmp'); tmp.write_text(json.dumps(rec) + '\n'); tmp.rename(f); n += 1
    print(f'recombined {n} clip records with rule={RULE}, min_points={MIN_POINTS_S4}')


def _fill_none(vals):
    a = np.array([np.nan if v is None else v for v in vals], float); good = np.isfinite(a)
    if good.any() and not good.all():
        a[~good] = np.interp(np.where(~good)[0], np.where(good)[0], a[good])
    return a


# -------------------------------------------------------------------------------------------------------------- report
def load_clips(out: Path):
    clips = {}
    for f in (out / 'clips').glob('*/*.json'):
        r = json.loads(f.read_text()); clips[(r['name'], r['source'])] = r
    return clips


def spearman(a, b):
    from scipy.stats import spearmanr
    a, b = np.asarray(a, float), np.asarray(b, float); ok = np.isfinite(a) & np.isfinite(b)
    return float(spearmanr(a[ok], b[ok]).correlation), int(ok.sum())


def validation(clips, names, gt, which='travelled_m'):
    """Distance travelled vs the AV2 poses for real-raw / real-recon at each horizon.
    which: travelled_m (the instrument) or chain (stride-1 chain only)."""
    def est_of(n, src, h):
        rec = clips.get((n, src))
        if rec is None:
            return np.nan
        tr = {'travelled_m': rec['travelled_m'], 'chain': rec['diagnostics']['chain_only_travelled_m']}[which]
        return np.nan if tr is None or tr.get(str(h)) is None else tr[str(h)]
    rows = {}
    for h in HORIZONS:
        g = np.array([gt[n]['travelled_m'][str(h)] for n in names])
        est = {}
        for src in ('real-raw', 'real-recon'):
            e = np.array([est_of(n, src, h) for n in names], float); est[src] = e
            fin = np.isfinite(e); mv = fin & (g > 2.0)
            rel = e[mv] / g[mv] - 1
            rho_all, n_all = spearman(e, g); rho_mv, _ = spearman(e[mv], g[mv])
            rows[(h, src)] = dict(n=n_all, n_moving=int(mv.sum()), median_rel=float(np.median(rel)), median_abs_rel=float(np.median(np.abs(rel))),
                                  p90_abs_rel=float(np.percentile(np.abs(rel), 90)), within_10pct=float(np.mean(np.abs(rel) <= 0.1)),
                                  spearman_all=rho_all, spearman_moving=rho_mv, median_abs_err_m=float(np.median(np.abs(e[fin] - g[fin]))))
        rho, n = spearman(est['real-raw'], est['real-recon'])
        mv = np.isfinite(est['real-raw']) & np.isfinite(est['real-recon']) & (g > 2)
        rows[(h, 'raw-vs-recon')] = dict(n=n, spearman=rho, median_recon_over_raw=float(np.median(est['real-recon'][mv] / est['real-raw'][mv])))
    return rows


def report(args):
    S = args.study
    names = [l.strip() for l in args.names_file.read_text().splitlines() if l.strip()]
    gt = json.loads((S / 'instrument' / 'gt-ego-dev.json').read_text())
    clips = load_clips(args.out)
    val = validation(clips, names, gt); val_c = validation(clips, names, gt, which='chain')
    acc = dict(raw_median_abs_rel=val[(4, 'real-raw')]['median_abs_rel'] <= 0.10, raw_spearman=val[(4, 'real-raw')]['spearman_all'] >= 0.90,
               recon_median_abs_rel=val[(4, 'real-recon')]['median_abs_rel'] <= 0.10, recon_spearman=val[(4, 'real-recon')]['spearman_all'] >= 0.90,
               raw_vs_recon_spearman=val[(4, 'raw-vs-recon')]['spearman'] >= 0.95)
    acc['passed'] = all(acc.values())
    result = dict(validation={f'{h}s|{s}': v for (h, s), v in val.items()}, acceptance_4s=acc,
                  validation_chain_only={f'{h}s|{s}': v for (h, s), v in val_c.items()})
    # per-window table for the record
    per = {}
    for n in names:
        per[n] = dict(gt_m=gt[n]['travelled_m'], **{src: clips[(n, src)]['travelled_m'] for src in ('real-raw', 'real-recon') if (n, src) in clips})
    gen = args.gen_run
    # measurement quality per source (comparable between reconstructed real and generated clips?)
    qual = {}
    for src in sorted({k[1].rsplit('-s', 1)[0] if k[1].startswith(gen or '#') else k[1] for k in clips}):
        recs = [r for (n, k), r in clips.items() if (k == src or (gen and k.startswith(gen) and src == gen)) and n in names]
        sup = np.array([r['quality']['accepted_share'] for r in recs]); rat = np.array([np.nan if r['quality']['ratio_refined_over_chain'] is None else r['quality']['ratio_refined_over_chain'] for r in recs])
        qual[src] = dict(clips=len(recs), accepted_share_median=float(np.median(sup)), accepted_share_below_half=float(np.mean(sup < 0.5)),
                         ratio_refined_over_chain_median=float(np.nanmedian(rat)), chain_fallback=int(sum(r['quality']['rule'].startswith('chain') for r in recs)),
                         stride1_plausible_share_median=float(np.median([r['quality']['stride1_plausible_share'] or np.nan for r in recs])))
    result['measurement_quality'] = qual
    if gen:
        result['decomposition'] = decomposition(S, names, clips, gen, gt, args.out)
    (args.out / 'results.json').write_text(json.dumps(result, indent=1) + '\n')
    (args.out / 'per-window-real.json').write_text(json.dumps(per) + '\n')
    print(json.dumps(dict(acceptance_4s=acc, **{k: result['validation'][k] for k in ('4s|real-raw', '4s|real-recon', '4s|raw-vs-recon')}), indent=1))
    print(json.dumps(qual, indent=1))
    if gen:
        print(json.dumps(result['decomposition']['summary'], indent=1))


def anchor_gap(S: Path, run_dir: str, name: str, clip_key: str):
    """Ground-range reading at t = 0 of a clip: median over the three frozen trackers (v3, v5, v8)."""
    vals = []
    for t in ('v3', 'v5', 'v8'):
        m = _measure_cache(S / run_dir / f'measure-{t}.json')
        tr = m['windows'].get(name, {}).get('clips', {}).get(clip_key, {}).get('track', [])
        r0 = next((r for r in tr if abs(r.get('t', -1)) < 1e-9 and 'reading_m' in r), None)
        if r0 is not None:
            vals.append(r0['reading_m'])
    return float(np.median(vals)) if vals else None


def followed_at(S: Path, run_dir: str, name: str, clip_key: str, t=4.0):
    """How many of the three frozen trackers follow the lead (state 'followed') at time t in this clip."""
    k = 0
    for tr_name in ('v3', 'v5', 'v8'):
        tr = _measure_cache(S / run_dir / f'measure-{tr_name}.json')['windows'].get(name, {}).get('clips', {}).get(clip_key, {}).get('track', [])
        r = next((x for x in tr if abs(x.get('t', -1) - t) < 1e-9), None)
        k += int(r is not None and r.get('state') == 'followed')
    return k


_MEASURES = {}


def _measure_cache(path):
    if path not in _MEASURES:
        _MEASURES[path] = json.loads(path.read_text())
    return _MEASURES[path]


def decomposition(S: Path, names, clips, gen, gt, out: Path, horizon=4):
    """Exploratory DEV decomposition of the 4 s gap: ego advance and lead advance = gap(4) - gap(0) + ego advance."""
    rng = np.random.default_rng(2026)
    windows = {w['name']: w for w in json.loads((S / f'results-{gen}' / 'primary' / 'windows.json').read_text())}
    rows = []
    for n in names:
        w = windows.get(n)
        if w is None:
            continue
        g0_real = anchor_gap(S, f'results-{gen}', n, 'real-recon')
        g0_gen = [anchor_gap(S, f'results-{gen}', n, f'{gen}-s{k}') for k in range(4)]
        ego_real = (clips.get((n, 'real-recon')) or {}).get('travelled_m', {}).get(str(horizon))
        ego_raw = (clips.get((n, 'real-raw')) or {}).get('travelled_m', {}).get(str(horizon))
        ego_gen = [(clips.get((n, f'{gen}-s{k}')) or {}).get('travelled_m', {}).get(str(horizon)) for k in range(4)]
        read_gen = [((clips.get((n, f'{gen}-s{k}')) or {}).get('quality') or {}).get('accepted_share') for k in range(4)]
        fol = [followed_at(S, f'results-{gen}', n, 'real-recon', horizon)] + [followed_at(S, f'results-{gen}', n, f'{gen}-s{k}', horizon) for k in range(4)]
        rows.append(dict(name=n, gap4_real=w['real'], gap4_gen=w['generated'], gap0_real=g0_real, gap0_gen=g0_gen, trackers_following_at_4s=fol,
                         ego_accel_mps2=(w.get('covariates') or {}).get('ego_accel_mps2'), ego_speed_mps=(w.get('covariates') or {}).get('ego_speed_mps'),
                         ego_real=ego_real, ego_raw=ego_raw, ego_gt=gt[n]['travelled_m'][str(horizon)], ego_gen=ego_gen, read_gen=read_gen,
                         read_real=((clips.get((n, 'real-recon')) or {}).get('quality') or {}).get('accepted_share')))
    (out / f'decomposition-{gen}-windows.json').write_text(json.dumps(rows, indent=1) + '\n')

    def rank_stats(real, gens, label):
        """PIT-style rank of the real value among the K generated values (randomised ties) + mean gen - real."""
        ok = [i for i in range(len(real)) if real[i] is not None and all(g is not None for g in gens[i])]
        r = np.array([real[i] for i in ok], float); G = np.array([gens[i] for i in ok], float)
        below = (G < r[:, None]).sum(1); ties = (G == r[:, None]).sum(1)
        rank = below + np.array([rng.integers(0, t + 1) for t in ties])
        K = G.shape[1]
        pit = (rank + 0.5) / (K + 1)
        diff = G.mean(1) - r
        boot = [np.mean(diff[rng.integers(0, len(diff), len(diff))]) for _ in range(4000)]
        bootp = [np.mean(pit[rng.integers(0, len(pit), len(pit))]) - 0.5 for _ in range(4000)]
        return dict(quantity=label, n=len(ok), rank_counts=np.bincount(rank, minlength=K + 1).tolist(), mean_pit_minus_half=float(pit.mean() - 0.5),
                    mean_pit_ci95=[float(np.percentile(bootp, 2.5)), float(np.percentile(bootp, 97.5))], share_real_above_all=float(np.mean(rank == K)),
                    share_real_below_all=float(np.mean(rank == 0)), mean_gen_minus_real_m=float(diff.mean()),
                    mean_gen_minus_real_ci95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))], median_gen_minus_real_m=float(np.median(diff)),
                    mean_real_m=float(r.mean()), mean_gen_m=float(G.mean()))

    def lead(gap4, gap0, ego):
        return None if gap4 is None or gap0 is None or ego is None else gap4 - gap0 + ego

    out_s = {}
    acc = lambda r: r['ego_accel_mps2'] if r['ego_accel_mps2'] is not None else 0.0
    for subset, keep in (('all', lambda r: True),
                         ('lead followed at 4 s in all five clips (>= 2 of 3 trackers)', lambda r: min(r['trackers_following_at_4s']) >= 2),
                         ('stratum: braking ego at the anchor (accel < -0.5 m/s2)', lambda r: acc(r) < -0.5),
                         ('stratum: steady ego (|accel| <= 0.5 m/s2)', lambda r: abs(acc(r)) <= 0.5),
                         ('stratum: accelerating ego (accel > 0.5 m/s2)', lambda r: acc(r) > 0.5),
                         ('accepted stride-4 groups >= 0.5 in all five clips', lambda r: (r['read_real'] or 0) >= 0.5 and all((x or 0) >= 0.5 for x in r['read_gen']))):
        R = [r for r in rows if keep(r)]
        ego_real = [r['ego_real'] for r in R]; ego_gen = [r['ego_gen'] for r in R]
        lead_real = [lead(r['gap4_real'], r['gap0_real'], r['ego_real']) for r in R]
        lead_gen = [[lead(r['gap4_gen'][k], r['gap0_gen'][k], r['ego_gen'][k]) for k in range(4)] for r in R]
        gap_change_real = [None if r['gap0_real'] is None else r['gap4_real'] - r['gap0_real'] for r in R]
        gap_change_gen = [[None if r['gap0_gen'][k] is None else r['gap4_gen'][k] - r['gap0_gen'][k] for k in range(4)] for r in R]
        out_s[subset] = dict(windows=len(R), ego_advance=rank_stats(ego_real, ego_gen, 'ego advance 0-4 s (m)'),
                             lead_advance=rank_stats(lead_real, lead_gen, 'lead advance 0-4 s = gap(4) - gap(0) + ego (m)'),
                             gap_change=rank_stats(gap_change_real, gap_change_gen, 'gap(4) - gap(0) (m)'),
                             gap_at_4s=rank_stats([r['gap4_real'] for r in R], [r['gap4_gen'] for r in R], 'gap(4) (m), frozen reading'))
    # the same with the real side's ego advance from the poses instead of the instrument (asymmetric check)
    R = rows
    lead_real_gt = [lead(r['gap4_real'], r['gap0_real'], r['ego_gt']) for r in R]
    lead_gen = [[lead(r['gap4_gen'][k], r['gap0_gen'][k], r['ego_gen'][k]) for k in range(4)] for r in R]
    out_s['check: real ego advance from the AV2 poses'] = dict(ego_advance=rank_stats([r['ego_gt'] for r in R], [r['ego_gen'] for r in R], 'ego advance (real = poses)'),
                                                              lead_advance=rank_stats(lead_real_gt, lead_gen, 'lead advance (real ego = poses)'))
    return dict(summary=out_s, gen_run=gen, horizon_s=horizon)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    for c in ('measure', 'report', 'recombine'):
        p = sub.add_parser(c)
        p.add_argument('--study', type=Path, required=True); p.add_argument('--names-file', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
        p.add_argument('--windows-dir', default='windows-dev'); p.add_argument('--recon-run', default='recon-dev')
        if c == 'recombine':
            continue
        if c == 'measure':
            p.add_argument('--sources', nargs='+', required=True); p.add_argument('--batch', type=int, default=4); p.add_argument('--gpu-mem-gib', type=float, default=3.0)
            p.add_argument('--part', default='0/1', help='i/n: measure every n-th clip starting at i (parallel processes)')
        else:
            p.add_argument('--gen-run', default=None)
    args = ap.parse_args()
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    {'measure': measure, 'report': report, 'recombine': recombine}[args.cmd](args)


if __name__ == '__main__':
    main()
