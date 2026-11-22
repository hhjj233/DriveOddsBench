#!/usr/bin/env python3
"""Band-only ego odometry for the shared canvas instrument (canvas-v1): bench-v1's VO port
(measure_bench_ego_vo.py, VO v2 of measure_ego_motion_raft.py; both imported read only) on a model's canvas clips at
its frame rate, for models without a Nano frame (Orbis 2, LTX-2.5, Epona).

Geometry (mode 'band'): the 512x288 canvas clip (rows >= the spec's valid rows set to 0) resized to 320x180 with
cv2.INTER_AREA, i.e. the Nano pixel density (0.3303226 x 0.625 = 0.2064516 = the Nano scale); camera K_band x 0.625
(pixel-centre convention); the frozen Nano rows moved by the band offset 0.2064516 x (768 - r0) (-6 rows for the bench
band): road pixels rows 54-179, flow targets rows 44-179, columns 2-317; with valid canvas rows [0, v) (Epona: 216) the
RAFT input is cut at the last multiple of 4 below 0.625 v (132 rows; RAFT needs sizes divisible by 8 after the x2
upsampling) and both row ranges end at its last row (131). Masks: the canvas detections (score >= 0.3) x 0.625, padded
4 px. Everything else is the frozen code: RAFT large x2 / 20 updates, forward-backward check, robust 6-DoF ground-plane
fit, plausibility, one warped refinement per 0.4 s group, finalize(), planar odometry.
Mode 'nano' runs the same loop on nano/real.mp4 with the frozen Nano geometry (the regression against the frozen VO
records; regress mode only).

Frame rate: groups are 0.4 s at every rate (the frozen finalize / planar / history timing, STRIDE 4 at 10 Hz):
  10 Hz  stride-1 pairs 16->17 ... 79->80 (0.1 s), groups 16->20, 20->24 ... 76->80 (4 frames)
   5 Hz  stride-1 pairs  8->9  ... 39->40 (0.2 s), groups  8->10, 10->12 ... 38->40 (2 frames)
History (constant-velocity baseline inputs): frames 0 -> t0, the same pairs and groups, one refinement.

  ego_vo_band.py measure|history --bench B --names-file F --model M --source NAME --from KIND[:ARG] --det-root DIR
                                 --out VO_DIR [--mode band|nano]
  ego_vo_band.py gate --bench B --out VO_DIR --raw-source RAW --recon-source REF
  ego_vo_band.py history-check --bench B --out VO_DIR --raw-source RAW --recon-source REF
Records: VO_DIR/clips/<source>/<window>__<sample>.json and VO_DIR/clips-hist/... (the frozen schema). measure / history:
DEV by default, TEST only with --allow-test once the model's frozen protocol file exists; gate / history-check: DEV only.
"""
from __future__ import annotations

import argparse, json, math, os, queue, sys, threading, time, types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import ci_core as ci  # noqa: E402
import detect_canvas as dc  # noqa: E402  (clip sources and canvas frame loading)
import measure_ego_motion_raft as vo  # noqa: E402  (frozen VO v2, read only)
import measure_bench_ego_vo as ego  # noqa: E402  (frozen VO port, read only)

VO_SCALE = 0.625
RAFT_WEIGHTS = Path(os.environ.get('DOB_ROOT', '.'), 'cache/torch/hub/checkpoints/raft_large_C_T_SKHT_V2-ff5fadd5.pth')
RAFT_SHA256 = 'ff5fadd56d26b40647388883af1547351ea17868b765c05b27231e72dd16a322'   # instrument-frozen.txt


class VOGeom:
    """Frame geometry of one mode / spec (rows, columns, crop, masks)."""

    def __init__(self, mode, spec):
        self.mode = mode
        if mode == 'nano':
            if spec['step'] != 1:
                raise SystemExit('nano mode is the 10 Hz regression')
            self.H, self.W = ego.H, ego.W
            self.crop = ego.CROP
            self.roi_rows, self.roi_cols, self.tgt_rows = ego.ROI_ROWS, ego.ROI_COLS, (ego.TGT_ROW_MIN, ego.TGT_ROW_MAX)
        else:
            self.H, self.W = int(round(mb_canvas_h() * VO_SCALE)), int(round(512 * VO_SCALE))      # 180, 320
            vrows = int(math.floor(spec['valid_canvas_rows'][1] * VO_SCALE + 1e-9))
            crop_end = vrows - vrows % 4
            off = ci.S * VO_SCALE * (768 - spec['band_rows'][0])
            last = crop_end - 1
            self.crop = (0, crop_end, 0, self.W)
            self.roi_rows = (int(round(ego.ROI_ROWS[0] + off)), min(int(round(ego.ROI_ROWS[1] + off)), last))
            self.tgt_rows = (int(round(ego.TGT_ROW_MIN + off)), min(int(round(ego.TGT_ROW_MAX + off)), last))
            self.roi_cols = ego.ROI_COLS
        self.spec = spec

    def describe(self):
        return dict(mode=self.mode, frame_hw=[self.H, self.W], crop=list(self.crop), roi_rows=list(self.roi_rows), roi_cols=list(self.roi_cols),
                    target_rows=list(self.tgt_rows), dist_m=list(ego.DIST_M), det_score=ego.DET_SCORE, det_pad=ego.DET_PAD, scale=VO_SCALE if self.mode == 'band' else None)

    def K(self, window):
        if self.mode == 'nano':
            return window['intrinsics']['K_nano']
        K = ci.K_band(window, self.spec['band_rows'][0])
        a = VO_SCALE
        return [[a * K[0, 0], 0.0, a * (K[0, 2] + 0.5) - 0.5], [0.0, a * K[1, 1], a * (K[1, 2] + 0.5) - 0.5], [0.0, 0.0, 1.0]]

    def box_mask(self, dets):
        if self.mode == 'nano':
            return ego.box_mask(dets)
        m = np.zeros((self.H, self.W), bool); a = VO_SCALE
        for d in dets or []:
            if d[4] < ego.DET_SCORE:
                continue
            x1, y1, x2, y2 = d[0] * a, d[1] * a, d[2] * a, d[3] * a
            m[max(0, int(y1) - ego.DET_PAD):min(self.H, int(np.ceil(y2)) + ego.DET_PAD), max(0, int(x1) - ego.DET_PAD):min(self.W, int(np.ceil(x2)) + ego.DET_PAD)] = True
        return m

    def frames(self, job):
        if self.mode == 'nano':
            return ego.frames_of(job['video'])
        import cv2
        fr = dc.frames_of(job)
        return np.stack([cv2.resize(np.ascontiguousarray(f), (self.W, self.H), interpolation=cv2.INTER_AREA) for f in fr])


def mb_canvas_h():
    return 288


class Grid:
    """measure_bench_ego_vo.Grid with the mode's rows."""

    def __init__(self, cam, g: VOGeom, step=2):
        vv, uu = np.mgrid[g.roi_rows[0]:g.roi_rows[1] + 1:step, g.roi_cols[0]:g.roi_cols[1] + 1:step]
        self.u = uu.ravel().astype(float); self.v = vv.ravel().astype(float)
        self.iu = self.u.astype(int); self.iv = self.v.astype(int)
        self.P = cam.ground(cam.rays(self.u, self.v))
        dist = self.P @ cam.fwd
        self.base = np.isfinite(dist) & (dist >= ego.DIST_M[0]) & (dist <= ego.DIST_M[1])


def fit_pair(cam, grid, g: VOGeom, flow, fb_ok, dets_i, dets_j, theta0=None):
    """measure_bench_ego_vo.fit_pair with the mode's rows and masks."""
    F = flow[:, grid.iv, grid.iu].T
    tu, tv = grid.u + F[:, 0], grid.v + F[:, 1]
    ok = grid.base & np.isfinite(F).all(1) & fb_ok[grid.iv, grid.iu] & ~g.box_mask(dets_i)[grid.iv, grid.iu]
    ok &= (tu >= g.roi_cols[0]) & (tu <= g.roi_cols[1]) & (tv <= g.tgt_rows[1]) & (tv >= g.tgt_rows[0])
    ok &= ~g.box_mask(dets_j)[np.clip(np.round(np.nan_to_num(tv)).astype(int), 0, g.H - 1), np.clip(np.round(np.nan_to_num(tu)).astype(int), 0, g.W - 1)]
    n = int(ok.sum())
    if n < vo.MIN_POINTS:
        return None, dict(points=n)
    tgt = np.stack([(tu[ok] - cam.cx) / cam.fx, (tv[ok] - cam.cy) / cam.fy], 1)
    th, info = vo.fit_motion(grid.P[ok], tgt, cam.fy, theta0=theta0)
    return th, dict(points=n, **info)


def sample_key(s):
    return 'real' if s == 'recon' else ('raw' if s == 'raw' else s.lstrip('s'))


def jobs_of(args, spec, names):
    a = types.SimpleNamespace(bench=args.bench, out=args.det_root, source=args.source, from_=args.from_, windows_dir=args.windows_dir)
    if args.mode == 'nano':
        for w in names:
            yield dict(video=args.bench / args.windows_dir / w / 'nano' / 'real.mp4', out=args.det_root / args.source / f'{w}.json', window=w, sample='raw')
        return
    yield from dc.jobs_for(a, spec, names)


def run(args, history=False):
    import cv2, torch, torch.nn.functional as tF
    from torchvision.models.optical_flow import raft_large
    torch.set_num_threads(2); cv2.setNumThreads(1)
    spec = ci.load_spec(args.model); g = VOGeom(args.mode, spec)
    names = ci.names_from(args.names_file); split = ci.split_of(args.bench, names, spec, args.allow_test)
    args.windows_dir = ci.windows_dir(split)
    sub = 'clips-hist' if history else 'clips'
    out_dir = args.out / sub
    jobs = []
    for j in jobs_of(args, spec, names):
        f = out_dir / args.source / f'{j["window"]}__{sample_key(j["sample"])}.json'
        if history and sample_key(j['sample']) not in ('raw', 'real'):
            continue
        if Path(j['video']).is_file() and Path(j['out']).is_file() and not f.exists():
            jobs.append((j, f))
    part, nparts = (int(x) for x in args.part.split('/'))
    jobs = jobs[part::nparts]
    print(json.dumps(dict(clips=len(jobs), source=args.source, history=history, geometry=g.describe())), flush=True)
    if not jobs:
        return
    if ci.sha256_file(RAFT_WEIGHTS) != RAFT_SHA256:
        sys.exit('RAFT weights differ from instrument-frozen.txt')
    free, total = torch.cuda.mem_get_info(0)
    if free / 2 ** 20 < args.min_free_mib + 3072:
        sys.exit(f'GPU free {free / 2 ** 20:.0f} MiB: not enough to keep {args.min_free_mib:.0f} MiB free')
    torch.cuda.set_per_process_memory_fraction(min(1.0, args.gpu_mem_gib * 2 ** 30 / total), 0)
    dev_name = 'cuda:0'
    model = raft_large(weights=None, progress=False)
    model.load_state_dict(torch.load(RAFT_WEIGHTS, map_location='cpu', weights_only=True)); model.eval().to(dev_name)
    H, W = g.H, g.W; r0, r1, c0, c1 = g.crop
    q: queue.Queue = queue.Queue(maxsize=3)

    def reader():
        for j, f in jobs:
            try:
                q.put(((j, f), g.frames(j)))
            except Exception as e:
                q.put(((j, f), e))
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

    rnd = lambda th: None if th is None else [round(float(x), 6) for x in th]

    def plaus(t, info, cam):
        info['_fwd'] = cam.fwd; ok = vo.plausible(t, info); info.pop('_fwd'); return ok

    G = 4 // spec['step']                         # frames per 0.4 s group
    t0f, n = spec['t0_frame'], spec['n_frames']
    first, last = (0, t0f) if history else (t0f - G, n - 1)
    t_start = time.time(); script_sha = ci.sha256_file(__file__)
    for done, item in enumerate(iter(q.get, None)):
        (job, fout), fr = item
        if isinstance(fr, Exception) or len(fr) != n:
            print(f'skip {job["window"]} {job["sample"]}: {fr if isinstance(fr, Exception) else len(fr)}', flush=True); continue
        w = job['window']
        win = json.loads((args.bench / args.windows_dir / w / 'window.json').read_text())
        T = np.asarray(win['intrinsics']['ego_SE3_cam'], float)
        camd = dict(K256=g.K(win), ego_R_cam=T[:3, :3].tolist(), ego_t_cam=T[:3, 3].tolist())
        cam = vo.Camera(**camd); grid = Grid(cam, g)
        dj = json.loads(Path(job['out']).read_text())
        dets = {t: d for t, d in zip(dj['frames'], dj['detections'])}
        p1 = [(i, i + 1) for i in range(first, last)]
        fw = raft(fr[[i for i, _ in p1]], fr[[j for _, j in p1]]); bw = raft(fr[[j for _, j in p1]], fr[[i for i, _ in p1]])
        s1, th = [], None
        for k, (i, j) in enumerate(p1):
            t, info = fit_pair(cam, grid, g, fw[k], fb_ok(fw[k], bw[k]), dets.get(i), dets.get(j), theta0=th)
            th = t if t is not None else th
            s1.append((t, info, plaus(t, info, cam)))
        good = [k for k, (_, _, ok) in enumerate(s1) if ok]
        s1f = [s1[k][0] if s1[k][2] else (s1[min(good, key=lambda r: abs(r - k))][0] if good else np.zeros(6)) for k in range(len(p1))]
        adv1 = np.array([float(np.asarray(t[:3]) @ cam.fwd) for t in s1f])
        p4 = [(i, i + G) for i in range(first, last, G)]
        cur = [vo.compose(s1f[i - first:i - first + G]) for i, _ in p4]
        chain = [float(adv1[i - first:i - first + G].sum()) for i, _ in p4]
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
                t, info = fit_pair(cam, grid, g, Rf[k] + Ws, fb_ok(Rf[k], Rb[k]), dets.get(i), dets.get(j), theta0=cur[k])
                res.append((t, info, plaus(t, info, cam)))
            passes.append(res)
            cur = [t if ok else cur[k] for k, (t, info, ok) in enumerate(res)]
        rec = dict(name=w, source=args.source, sample=sample_key(job['sample']), video=str(job['video']), detections=str(job['out']), camera=camd,
                   groups=[dict(pair=list(p), chain_m=round(chain[k], 5), init_theta=rnd(vo.compose(s1f[p[0] - first:p[0] - first + G])),
                                passes=[dict(theta=rnd(passes[qq][k][0]), plausible=passes[qq][k][2],
                                             advance_m=None if passes[qq][k][0] is None else round(float(np.asarray(passes[qq][k][0][:3]) @ cam.fwd), 5),
                                             **passes[qq][k][1]) for qq in range(len(passes))])
                           for k, p in enumerate(p4)],
                   stride1=[dict(pair=list(p), advance_m=None if s1[k][0] is None else round(float(np.asarray(s1[k][0][:3]) @ cam.fwd), 5),
                                 theta=rnd(s1[k][0]), plausible=s1[k][2], **s1[k][1]) for k, p in enumerate(p1)],
                   stride1_plausible_share=round(len(good) / len(p1), 3))
        if history:
            rec.pop('stride1')
        vo.finalize(rec)
        if history:
            rec.pop('travelled_m', None); rec.pop('diagnostics', None)
            rec['history'] = ego.planar_history(rec)
        else:
            rec['planar'] = ego.planar(rec)
        rec['canvas_instrument'] = dict(model=spec['model'], fps=spec['fps'], geometry=g.describe(), group_frames=G, script_sha256=script_sha, batch=args.batch)
        fout.parent.mkdir(parents=True, exist_ok=True)
        tmp = fout.with_suffix('.tmp'); tmp.write_text(json.dumps(rec) + '\n'); tmp.rename(fout)
        if (done + 1) % 5 == 0:
            print(f'{done + 1} clips, {time.time() - t_start:.0f} s, peak {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB', flush=True)
    print(f'done in {time.time() - t_start:.0f} s', flush=True)


# ------------------------------------------------------------------------------------------------------------- gate
def load_vo(vdir, source, name, sample, sub='clips'):
    f = vdir / sub / source / f'{name}__{sample}.json'
    return json.loads(f.read_text()) if f.is_file() else None


def gate_eval(bench, vdir, names, source, sample, h='4.0'):
    """measure_bench_ego_vo.gate_eval on this instrument's records (criteria of VO-GATE.md, unchanged)."""
    from scipy.stats import spearmanr
    rows = []
    for n in names:
        r = load_vo(vdir, source, n, sample)
        if r is None:
            continue
        gg = ego.gt_ego(bench, n)
        rows.append(dict(name=n, D_vo=r['planar']['horizons'][h]['D_m'], D_gt=gg[h]['D_m'], Psi_vo=r['planar']['horizons'][h]['Psi_deg'], Psi_gt=gg[h]['Psi_deg'],
                         accepted_share=r['quality']['accepted_share'], travelled_vo=r['travelled_m'].get(str(int(float(h))))))
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
    """measure_bench_ego_vo.cmd_gate for this instrument (raw: the 108 ego_inter DEV windows; reference: the 32 with
    generations); pass = criteria 1-3 at 4.0 s on raw AND on the reference (VO-GATE.md)."""
    from scipy.stats import spearmanr
    B = args.bench; d = B / ego.OUT_SUB
    raw_names = ci.names_from(d / 'names-ego-inter-dev.txt'); rec_names = ci.names_from(d / 'names-ego-inter-dev-gen.txt')
    res = dict(gate_file=str(d / 'VO-GATE.md'), gate_sha256=ci.sha256_file(d / 'VO-GATE.md'), raw_source=args.raw_source, recon_source=args.recon_source, results={})
    for h in ('2.0', '4.0', '6.0'):
        res['results'][f'raw|{h}'] = gate_eval(B, args.out, raw_names, args.raw_source, 'raw', h)
        if args.recon_source:
            res['results'][f'recon|{h}'] = gate_eval(B, args.out, rec_names, args.recon_source, 'real', h)
    r4 = res['results']['raw|4.0']; c4 = res['results'].get('recon|4.0', {})
    crit = ('criterion1_pass', 'criterion2_pass', 'criterion3_pass')
    res['raw_passed'] = all(r4.get(k, False) for k in crit)
    res['recon_passed'] = all(c4.get(k, False) for k in crit) if args.recon_source else None
    res['gate_passed'] = bool(res['raw_passed'] and res['recon_passed'])
    res['secondary'] = {f'D_{h}s': all(res['results'].get(f'{s}|{h}', {}).get('criterion1_pass', False) for s in ('raw', 'recon')) for h in ('2.0', '6.0')}
    if args.recon_source:
        both = [n for n in rec_names if load_vo(args.out, args.raw_source, n, 'raw') and load_vo(args.out, args.recon_source, n, 'real')]
        if len(both) > 2:
            a = [load_vo(args.out, args.raw_source, n, 'raw')['planar']['horizons']['4.0']['D_m'] for n in both]
            b = [load_vo(args.out, args.recon_source, n, 'real')['planar']['horizons']['4.0']['D_m'] for n in both]
            res['raw_vs_recon_D4'] = dict(n=len(both), spearman=round(float(spearmanr(a, b).correlation), 4), median_abs_diff_m=round(float(np.median(np.abs(np.subtract(a, b)))), 4))
    res['script_sha256'] = ci.sha256_file(__file__)
    (args.out / f'gate-result-{args.raw_source}.json').write_text(json.dumps(res, indent=1) + '\n')
    print(json.dumps({k: ({kk: vv for kk, vv in v.items() if kk != 'rows'} if isinstance(v, dict) and 'rows' in v else v) for k, v in res['results'].items()}, indent=1))
    print(json.dumps(dict(raw_passed=res['raw_passed'], recon_passed=res['recon_passed'], gate_passed=res['gate_passed'], secondary=res['secondary'],
                          raw_vs_recon=res.get('raw_vs_recon_D4')), indent=1))


def cmd_history_check(args):
    """measure_bench_ego_vo.cmd_history_check on this instrument's records."""
    from scipy.stats import spearmanr
    out = {}
    d = args.bench / ego.OUT_SUB
    for source, sample, names_file in ((args.raw_source, 'raw', 'names-ego-inter-dev.txt'), (args.recon_source, 'real', 'names-ego-inter-dev-gen.txt')):
        if not source:
            continue
        rows = []
        for n in ci.names_from(d / names_file):
            hrec = load_vo(args.out, source, n, sample, 'clips-hist')
            if hrec is None:
                continue
            h = hrec['history']
            gt = json.loads((args.bench / 'windows-dev' / n / 'gt.json').read_text())
            egp = {e['k']: e for e in gt['ego']}
            ks = list(range(-20, 1, 4)); T = np.array(ks) / 10.0; A = np.stack([np.ones_like(T), T], 1)
            (_, gv), *_ = np.linalg.lstsq(A, np.array([egp[k]['xyz_m'][0] for k in ks]), rcond=None)
            (_, gw), *_ = np.linalg.lstsq(A, np.degrees(np.array([egp[k]['yaw_rad'] for k in ks])), rcond=None)
            fut = load_vo(args.out, source, n, sample)
            rows.append(dict(name=n, v_vo=h['ls_speed_mps'], v_gt=float(gv), w_vo=h['ls_yaw_rate_dps'], w_gt=float(gw),
                             D4_gt=egp[40]['xyz_m'][0], D4_vo=None if fut is None else fut['planar']['horizons']['4.0']['D_m']))
        if not rows:
            continue
        v_vo = np.array([r['v_vo'] for r in rows]); v_gt = np.array([r['v_gt'] for r in rows])
        w_vo = np.array([r['w_vo'] for r in rows]); w_gt = np.array([r['w_gt'] for r in rows]); D4 = np.array([r['D4_gt'] for r in rows])
        res = dict(n=len(rows), speed_err_median_abs=float(np.median(np.abs(v_vo - v_gt))), speed_err_p90_abs=float(np.percentile(np.abs(v_vo - v_gt), 90)),
                   speed_bias=float(np.median(v_vo - v_gt)), speed_spearman=float(spearmanr(v_vo, v_gt).correlation),
                   yaw_rate_err_median_abs=float(np.median(np.abs(w_vo - w_gt))), yaw_rate_err_p90_abs=float(np.percentile(np.abs(w_vo - w_gt), 90)),
                   cv4_vs_gt_median_abs_A=float(np.median(np.abs(4 * v_vo - D4))), cv4_vs_gt_median_abs_B=float(np.median(np.abs(4 * v_gt - D4))))
        fr = [r for r in rows if r['D4_vo'] is not None]
        if fr:
            res['cv4_vs_vo_reading_median_abs_A'] = float(np.median([abs(4 * r['v_vo'] - r['D4_vo']) for r in fr]))
        out[f'{source}|{sample}'] = res
    (args.out / f'history-check-{args.raw_source}.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    for c in ('measure', 'history'):
        p = sub.add_parser(c)
        p.add_argument('--bench', type=Path, required=True); p.add_argument('--names-file', type=Path, required=True)
        p.add_argument('--model', required=True); p.add_argument('--source', required=True); p.add_argument('--from', dest='from_', default='')
        p.add_argument('--det-root', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
        p.add_argument('--mode', choices=('band', 'nano'), default='band')
        p.add_argument('--batch', type=int, default=4); p.add_argument('--gpu-mem-gib', type=float, default=2.5)
        p.add_argument('--min-free-mib', type=float, default=6144); p.add_argument('--part', default='0/1')
        p.add_argument('--allow-test', action='store_true', help="TEST windows; refused unless the spec's test_freeze_file exists in the bench")
    for c in ('gate', 'history-check'):
        p = sub.add_parser(c)
        p.add_argument('--bench', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
        p.add_argument('--raw-source', required=True); p.add_argument('--recon-source', default=None)
    args = ap.parse_args()
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    if args.cmd in ('measure', 'history'):
        run(args, history=args.cmd == 'history')
    else:
        {'gate': cmd_gate, 'history-check': cmd_history_check}[args.cmd](args)


if __name__ == '__main__':
    main()
