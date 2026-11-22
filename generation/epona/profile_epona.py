#!/usr/bin/env python3
"""Speed profile of the Epona adapter (DEV only; research server, the allowed GPU under the shared lock; see run_dev_local.sh).
Weights and Epona checkout: EPONA_MODELS_DIR and EPONA_SRC (default $EPONA_ENV/src-git), from the environment or the
git-ignored models/epona/local.env (template local.env.example); profile_local.sh sets the GPU, the lock and the guards.

  profile_epona.py --input-root DIR --window W --out DIR --plan micro,breakdown,prof,batch,options
                   [--batches 1,4,8,16] [--ref-dir DIR] [--leave-free-mib 6144] [--max-gpu-mib 31000]

Plans (each writes <out>/<plan>.json; nothing changes the released sampling: 100 steps, same settings):
  micro      per-call time of STT (one evaluate), TrajDiT and VisDiT (one network evaluation each) on the window's real
             inputs at batch 1, 2, 4, 8, 16 (synchronised timers), their FLOPs (nn.Linear and scaled-dot-product
             attention counted by hooks) and the achieved TFLOPS
  breakdown  one batch-1 sample on the adapter's own path (model.rollout, the released step_eval) with synchronised
             timers around STT, every TrajDiT / VisDiT evaluation, the two Euler loops, the DCAE calls; then the same
             sample without timers, with nvidia-smi dmon (SM utilisation) running
  prof       torch.profiler over one generated frame at batch 1 and at the largest batch: summed kernel time vs wall time
  batch      rollout_batch (per-sample generators) for each batch size: per-sample time, peak memory, outputs compared
             with batch-1 outputs of the released path (sample 0..R-1, made here or read from --ref-dir), and a rerun of
             the first batch size > 1 with the same layout (bit identity)
  options    DCAE encode/decode with cudnn.benchmark and with TF32 matmul (speed and pixel differences), torch.compile of
             the TrajDiT / VisDiT forward (speed, differences, determinism), and one 50-step sample (information only)
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_epona_bench as rb  # noqa: E402

LOCAL_ENV = Path(os.environ.get('WCB_LOCAL_ENV') or HERE / 'local.env')


def local_setting(key: str, path: Path | None = None) -> str | None:
    """A machine-specific setting: the environment variable KEY, else KEY=VALUE in models/epona/local.env (git-ignored;
    template local.env.example; WCB_LOCAL_ENV=FILE points elsewhere), else None. The rules of local_env.sh: lines are
    KEY=VALUE with optional 'export ' and quotes, '#' starts a comment, the first line for a key counts."""
    if os.environ.get(key):
        return os.environ[key]
    try:
        lines = Path(path or LOCAL_ENV).read_text().splitlines()
    except OSError:
        return None
    for line in lines:
        line = line.split('#', 1)[0].strip()
        if line.startswith('export '):
            line = line[len('export '):].strip()
        k, eq, v = line.partition('=')
        if eq and k.strip() == key:
            return v.strip().strip('"').strip("'") or None
    return None


def epona_paths() -> tuple[Path, Path]:
    """(weights directory, Epona checkout): EPONA_MODELS_DIR and EPONA_SRC (default $EPONA_ENV/src-git)."""
    models, src, env = local_setting('EPONA_MODELS_DIR'), local_setting('EPONA_SRC'), local_setting('EPONA_ENV')
    src = src or (str(Path(env) / 'src-git') if env else None)
    if not models or not src:
        raise SystemExit('EPONA_MODELS_DIR and EPONA_ENV (or EPONA_SRC) are not set: put them into models/epona/local.env '
                         '(template local.env.example) or the environment')
    return Path(models), Path(src)


def sha(a) -> str:
    return rb.sha256_bytes(a.tobytes())


class Sync:
    """Synchronised wall-clock accumulator."""

    def __init__(self, torch):
        self.torch, self.t = torch, {}
        self.n = {}

    @contextlib.contextmanager
    def __call__(self, key):
        self.torch.cuda.synchronize()
        t0 = time.perf_counter()
        yield
        self.torch.cuda.synchronize()
        self.t[key] = self.t.get(key, 0.0) + time.perf_counter() - t0
        self.n[key] = self.n.get(key, 0) + 1


@contextlib.contextmanager
def count_flops(torch, module):
    """FLOPs of nn.Linear (2 M K N) and scaled_dot_product_attention (4 B H Lq Lk D) inside module."""
    box = dict(linear=0, attention=0)
    hooks = []

    def lin_hook(mod, inp, out):
        x = inp[0]
        box['linear'] += 2 * (x.numel() // mod.in_features) * mod.in_features * mod.out_features
    for sub in module.modules():
        if isinstance(sub, torch.nn.Linear):
            hooks.append(sub.register_forward_hook(lin_hook))
    F = torch.nn.functional
    orig = F.scaled_dot_product_attention

    def sdpa(q, k, v, *a, **kw):
        b = q.numel() // (q.shape[-2] * q.shape[-1])
        box['attention'] += 4 * b * q.shape[-2] * k.shape[-2] * q.shape[-1]
        return orig(q, k, v, *a, **kw)
    F.scaled_dot_product_attention = sdpa
    try:
        yield box
    finally:
        F.scaled_dot_product_attention = orig
        for h in hooks:
            h.remove()


class Dmon:
    """nvidia-smi dmon -s u on one GPU, 1 s samples; returns the SM utilisation samples between start() and stop()."""

    def __init__(self, gpu_index: str, path: Path):
        self.gpu, self.path, self.proc = gpu_index, path, None

    def start(self):
        self.f = open(self.path, 'w')
        self.proc = subprocess.Popen(['nvidia-smi', 'dmon', '-s', 'u', '-i', self.gpu, '-d', '1'], stdout=self.f, stderr=subprocess.DEVNULL)
        time.sleep(1.5)

    def stop(self):
        self.proc.terminate()
        self.proc.wait()
        self.f.close()
        sm = []
        for line in self.path.read_text().splitlines():
            parts = line.split()
            if parts and parts[0].isdigit() and len(parts) >= 2 and parts[1].isdigit():
                sm.append(int(parts[1]))
        return dict(samples=len(sm), sm_mean=round(sum(sm) / len(sm), 1) if sm else None, sm_median=sorted(sm)[len(sm) // 2] if sm else None)


class Pmon:
    """nvidia-smi pmon -s u on one GPU, 1 s samples: SM share of this process vs the other processes (GPU 1 is shared
    with other users' jobs that do not take the lock)."""

    def __init__(self, gpu_index: str, path: Path):
        self.gpu, self.path, self.proc, self.me = gpu_index, path, None, os.getpid()

    def start(self):
        self.f = open(self.path, 'w')
        self.proc = subprocess.Popen(['nvidia-smi', 'pmon', '-i', self.gpu, '-s', 'u', '-d', '1'], stdout=self.f, stderr=subprocess.DEVNULL)
        time.sleep(1.5)

    def stop(self):
        self.proc.terminate()
        self.proc.wait()
        self.f.close()
        mine, others = [], []
        for line in self.path.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5 or not parts[0].isdigit():
                continue
            pid, typ, sm = parts[1], parts[2], parts[3]
            if typ != 'C':
                continue
            v = int(sm) if sm.isdigit() else 0
            (mine if pid == str(self.me) else others).append((pid, v))
        other_pids = sorted({p for p, _ in others})
        n_samples = max(1, len(mine))
        return dict(samples=len(mine), my_sm_mean=round(sum(v for _, v in mine) / n_samples, 1),
                    others_sm_sum_mean=round(sum(v for _, v in others) / n_samples, 1), other_pids=other_pids)


def load_window(input_root: Path, window: str, np):
    wdir = input_root / 'windows' / 'dev' / window
    hist, _ = rb.decode_rgb(wdir / 'history.mp4', 2)
    mats = rb.load_poses(wdir / 'past_poses.json', np)
    return hist, mats


def seeds_for(window: str, ks):
    import hashlib
    return [int.from_bytes(hashlib.sha256(f'bench-v1|{window}|{k}'.encode()).digest()[:4], 'big') & 0x7FFFFFFF for k in ks]


# ----------------------------------------------------------------------------------------------------------------------
def plan_micro(E, hist, mats, out: Path, batches, gpu_index: str = '1'):
    torch, np, m = E.torch, E.np, E.model
    res = dict(tokens={}, per_batch={})
    with torch.no_grad():
        img = E.normalize(hist[rb.N_HIST - rb.CF:]).unsqueeze(0).cuda()
        lat1 = E.tokenizer.encode_to_z(img[:, :rb.CF])
        pose1, yaw1 = E.condition_poses(mats)
    for B in batches:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        row = {}
        pm = Pmon(gpu_index, out / f'pmon-micro-b{B}.txt')
        pm.start()
        try:
            with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                lat = lat1.repeat(B, 1, 1, 1)
                pt = E.poses_to_indices(pose1.repeat(B, 1, 1), m.pose_x_vocab_size, m.pose_y_vocab_size)
                yt = E.yaws_to_indices(yaw1.repeat(B, 1, 1), m.yaw_vocab_size)
                # STT
                with count_flops(torch, m.model) as fl:
                    stt, pose_emb = m.model.evaluate(lat, pt, yt, sample_last=True)
                row['stt_flops'] = fl['linear'] + fl['attention']
                row['stt_flops_split'] = dict(fl)
                if B == batches[0]:
                    res['tokens']['stt'] = dict(per_frame=int(m.total_token_size), frames=rb.CF, total=int(m.total_token_size) * rb.CF,
                                                img=int(m.img_token_size), pose_yaw=int(m.total_token_size - m.img_token_size))
                reps = 3 if B >= 8 else 5
                for _ in range(2):
                    m.model.evaluate(lat, pt, yt, sample_last=True)
                torch.cuda.synchronize(); t0 = time.perf_counter()
                for _ in range(reps):
                    m.model.evaluate(lat, pt, yt, sample_last=True)
                torch.cuda.synchronize(); row['stt_s'] = (time.perf_counter() - t0) / reps
                # TrajDiT and VisDiT, one evaluation each, real shapes
                img_ids, cond_ids, traj_ids = E.prepare_ids(B, m.h, m.w, m.total_token_size, m.traj_len)
                g = torch.Generator().manual_seed(0)
                xt = torch.randn(B, m.traj_len, m.traj_token_size, generator=g).to(stt)
                xi = torch.randn(B, m.img_token_size, m.vae_emb_dim, generator=g).to(stt)
                tv = torch.full((B,), 0.5, dtype=stt.dtype, device=stt.device)
                with count_flops(torch, m.traj_dit) as fl:
                    m.traj_dit(traj=xt, traj_ids=traj_ids, cond=stt, cond_ids=cond_ids, timesteps=tv)
                row['traj_flops'] = fl['linear'] + fl['attention']
                with count_flops(torch, m.dit) as fl:
                    m.dit(img=xi, img_ids=img_ids, cond=stt, cond_ids=cond_ids, timesteps=tv, y=pose_emb)
                row['vis_flops'] = fl['linear'] + fl['attention']
                row['vis_flops_split'] = dict(fl)
                if B == batches[0]:
                    res['tokens']['vis_dit'] = dict(img=int(xi.shape[1]), cond=int(stt.shape[1]), total=int(xi.shape[1] + stt.shape[1]))
                    res['tokens']['traj_dit'] = dict(traj=int(xt.shape[1]), cond=int(stt.shape[1]), total=int(xt.shape[1] + stt.shape[1]))
                for key, fn, reps in (('traj', lambda: m.traj_dit(traj=xt, traj_ids=traj_ids, cond=stt, cond_ids=cond_ids, timesteps=tv), 50),
                                      ('vis', lambda: m.dit(img=xi, img_ids=img_ids, cond=stt, cond_ids=cond_ids, timesteps=tv, y=pose_emb), 20)):
                    for _ in range(3):
                        fn()
                    torch.cuda.synchronize(); t0 = time.perf_counter()
                    for _ in range(reps):
                        fn()
                    torch.cuda.synchronize(); row[f'{key}_s'] = (time.perf_counter() - t0) / reps
                    # one call at a time from an idle GPU: the CPU time to issue it vs the time until it has run
                    issue, done = [], []
                    for _ in range(reps):
                        torch.cuda.synchronize(); t0 = time.perf_counter()
                        fn()
                        issue.append(time.perf_counter() - t0)
                        torch.cuda.synchronize(); done.append(time.perf_counter() - t0)
                    row[f'{key}_cpu_issue_s'] = sum(issue) / reps
                    row[f'{key}_isolated_s'] = sum(done) / reps
            for key in ('stt', 'traj', 'vis'):
                row[f'{key}_tflops'] = round(row[f'{key}_flops'] / row[f'{key}_s'] / 1e12, 1)
                row[f'{key}_ms_per_sample'] = round(row[f'{key}_s'] / B * 1e3, 3)
            row['frame_model_s_per_sample'] = round((row['stt_s'] + 100 * row['traj_s'] + 100 * row['vis_s']) / B, 3)
            row['peak_allocated_gib'] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        except torch.cuda.OutOfMemoryError as exc:
            row = dict(error=f'OOM: {exc}'[:300])
        row['pmon'] = pm.stop()
        res['per_batch'][B] = row
        print(json.dumps({'micro': B, **{k: v for k, v in row.items() if not isinstance(v, dict)}}), flush=True)
    (out / ('micro.json' if not (out / 'micro.json').exists() else f'micro-{time.strftime("%H%M%S")}.json')).write_text(json.dumps(res, indent=1) + '\n')
    return res


def plan_breakdown(E, hist, mats, out: Path, window: str, gpu_index: str):
    torch, m, tok = E.torch, E.model, E.tokenizer
    seed = seeds_for(window, [0])[0]
    S = Sync(torch)
    orig = dict(evaluate=m.model.evaluate, traj_sample=m.traj_dit.sample, dit_sample=m.dit.sample, z2i=tok.z_to_image, e2z=tok.encode_to_z,
                step_eval=m.step_eval)

    def wrap(key, fn):
        def w(*a, **kw):
            with S(key):
                return fn(*a, **kw)
        return w
    hooks = []
    for key, mod in (('traj_forward', m.traj_dit), ('vis_forward', m.dit)):
        box = {}

        def pre(mod_, inp, key=key, box=box):
            torch.cuda.synchronize(); box['t'] = time.perf_counter()

        def post(mod_, inp, outp, key=key, box=box):
            torch.cuda.synchronize(); S.t[key] = S.t.get(key, 0.0) + time.perf_counter() - box['t']; S.n[key] = S.n.get(key, 0) + 1
        hooks += [mod.register_forward_pre_hook(pre), mod.register_forward_hook(post)]
    m.model.evaluate = wrap('stt', orig['evaluate'])
    m.traj_dit.sample = wrap('traj_loop', orig['traj_sample'])
    m.dit.sample = wrap('vis_loop', orig['dit_sample'])
    tok.z_to_image = wrap('dcae_decode', orig['z2i'])
    tok.encode_to_z = wrap('dcae_encode', orig['e2z'])
    m.step_eval = wrap('frame', orig['step_eval'])
    t0 = time.perf_counter()
    try:
        fut, ctx, traj, cond, counts, timing = E.rollout(hist, mats, seed)
    finally:
        for h in hooks:
            h.remove()
        for obj, name in ((m.model, 'evaluate'), (m.traj_dit, 'sample'), (m.dit, 'sample'), (tok, 'z_to_image'), (tok, 'encode_to_z'), (m, 'step_eval')):
            delattr(obj, name)   # drop the instance-level wrapper; the class method is used again
    total_timed = time.perf_counter() - t0
    t = S.t
    per_frame = {k: round(v / 30, 4) for k, v in t.items()}
    res = dict(seed=seed, timed_run_s=round(total_timed, 2), totals_s={k: round(v, 3) for k, v in t.items()}, calls=S.n, per_frame_s=per_frame,
               derived=dict(traj_euler_overhead_s=round(t['traj_loop'] - t['traj_forward'], 3), vis_euler_overhead_s=round(t['vis_loop'] - t['vis_forward'], 3),
                            frame_other_s=round(t['frame'] - t['stt'] - t['traj_loop'] - t['vis_loop'], 3),
                            outside_frames_s=round(total_timed - t['frame'], 3),
                            traj_ms_per_eval=round(t['traj_forward'] / S.n['traj_forward'] * 1e3, 3), vis_ms_per_eval=round(t['vis_forward'] / S.n['vis_forward'] * 1e3, 3),
                            stt_ms_per_call=round(t['stt'] / S.n['stt'] * 1e3, 2)),
               counts=counts, pred_traj_sha=sha(traj), future_sha=sha(fut))
    # the same sample without timers, with dmon; also checks that the timers did not change the numbers
    dm = Dmon(gpu_index, out / 'dmon-breakdown.txt')
    dm.start()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    fut2, _, traj2, _, _, timing2 = E.rollout(hist, mats, seed)
    torch.cuda.synchronize(); plain = time.perf_counter() - t0
    res['plain_run'] = dict(seconds=round(plain, 2), steps_mean_s=round(sum(timing2['steps']) / 30, 3), decode_s=timing2['decode'],
                            encode_context_s=timing2['encode_context'], dmon=dm.stop(), identical_to_timed=bool((fut2 == fut).all() and (traj2 == traj).all()))
    (out / 'breakdown.json').write_text(json.dumps(res, indent=1) + '\n')
    print(json.dumps(dict(breakdown=res['derived'], plain=res['plain_run'])), flush=True)
    return res


def plan_prof(E, hist, mats, out: Path, window: str, batches):
    torch, m = E.torch, E.model
    from torch.profiler import profile, ProfilerActivity
    res = {}
    for B in sorted({batches[0], batches[-1]}):
        seeds = seeds_for(window, range(B))
        gens = [torch.Generator().manual_seed(s) for s in seeds]
        with torch.no_grad():
            img = E.normalize(hist[rb.N_HIST - rb.CF:]).unsqueeze(0).cuda()
            lat = E.tokenizer.encode_to_z(img[:, :rb.CF]).repeat(B, 1, 1, 1)
            pose1, yaw1 = E.condition_poses(mats)
            pose, yaw = pose1.repeat(B, 1, 1), yaw1.repeat(B, 1, 1)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                E.step_eval_gen(lat, pose, yaw, [torch.Generator().manual_seed(1) for _ in range(B)])   # warm-up
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                t0 = time.perf_counter()
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    E.step_eval_gen(lat, pose, yaw, gens)
                torch.cuda.synchronize()
                wall = time.perf_counter() - t0
        ka = prof.key_averages()
        dev = [e for e in ka if 'cuda' in str(getattr(e, 'device_type', '')).lower()]   # kernels, memcpy, memset (no CPU ops: no double count)
        kern = sum(e.self_cuda_time_total for e in dev) / 1e6
        top = sorted(dev, key=lambda e: -e.self_cuda_time_total)[:12]
        n_kernels = sum(e.count for e in dev)
        res[B] = dict(wall_s=round(wall, 3), kernel_time_s=round(kern, 3), gpu_busy_fraction=round(kern / wall, 3), kernel_launches=int(n_kernels),
                      top_kernels=[dict(name=e.key[:90], calls=e.count, cuda_s=round(e.self_cuda_time_total / 1e6, 4)) for e in top],
                      note='one generated frame (STT + 100 TrajDiT + 100 VisDiT evaluations) under the profiler (wall includes profiler overhead)')
        print(json.dumps({'prof': B, 'wall_s': res[B]['wall_s'], 'kernel_s': res[B]['kernel_time_s'], 'busy': res[B]['gpu_busy_fraction']}), flush=True)
    (out / 'prof.json').write_text(json.dumps(res, indent=1) + '\n')
    return res


def throughput(E, hist, mats, seeds, n_frames: int = 6):
    """rollout_batch's loop for the first n_frames generated frames (one decode group of 6 per row): seconds per frame
    (all rows) and the decode time, synchronised."""
    torch, tok, rearrange = E.torch, E.tokenizer, E.rearrange
    B = len(seeds)
    gens = [torch.Generator().manual_seed(int(s)) for s in seeds]
    with torch.no_grad():
        img = E.normalize(hist[rb.N_HIST - rb.CF:]).unsqueeze(0).cuda()
        start_latents = tok.encode_to_z(img[:, :rb.CF]).repeat(B, 1, 1, 1)
        pose1, yaw1 = E.condition_poses(mats)
        pose, yaw = pose1.repeat(B, 1, 1), yaw1.repeat(B, 1, 1)
        patch = []
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(n_frames):
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                predict_traj, predict_latents = E.step_eval_gen(start_latents, pose, yaw, gens)
            predict_pose, predict_yaw = predict_traj[:, 0:1, 0:2], predict_traj[:, 0:1, 2:3]
            pose = torch.cat((pose[:, 1:rb.CF, ...], predict_pose, predict_pose), dim=1)
            yaw = torch.cat((yaw[:, 1:rb.CF, ...], predict_yaw, predict_yaw), dim=1)
            patch.append(predict_latents)
            predict_latents = rearrange(predict_latents, '(b f) h w c -> b f (h w) c', f=1)
            start_latents = torch.cat((start_latents[:, 1:rb.CF, ...], predict_latents), dim=1)
        torch.cuda.synchronize(); t1 = time.perf_counter()
        for b in range(B):
            tok.z_to_image(torch.cat([p[b:b + 1] for p in patch], dim=0), is_video=False).cpu()
        torch.cuda.synchronize(); t2 = time.perf_counter()
    return dict(frame_s=(t1 - t0) / n_frames, frame_s_per_sample=(t1 - t0) / n_frames / B, decode_group_s_per_sample=(t2 - t1) / B)


def plan_tput(E, hist, mats, out: Path, window: str, batches, gpu_index: str, rounds: int = 2):
    """Per-sample throughput at each batch size over the first 6 frames, in interleaved rounds (GPU 1 is shared with
    jobs outside the lock, so each round records their SM share; the least-contended round is the estimate)."""
    torch = E.torch
    res = dict(window=window, rounds=[])
    for r in range(rounds):
        order = batches if r % 2 == 0 else list(reversed(batches))
        for B in order:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            pm = Pmon(gpu_index, out / f'pmon-tput-r{r}-b{B}.txt')
            pm.start()
            try:
                row = throughput(E, hist, mats, seeds_for(window, range(B)), n_frames=3)
            except torch.cuda.OutOfMemoryError as exc:
                row = dict(error=f'OOM: {exc}'[:200])
                torch.cuda.empty_cache()
            row.update(round=r, batch=B, pmon=pm.stop(), peak_allocated_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                       peak_reserved_gib=round(torch.cuda.max_memory_reserved() / 2**30, 2))
            res['rounds'].append(row)
            print(json.dumps({'tput': B, **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}}), flush=True)
            (out / 'tput.json').write_text(json.dumps(res, indent=1) + '\n')
    return res


def native_ref(ref_dir: Path, window: str, k: int, np):
    """A batch-1 reference made by the released path: native.mp4 frames 11..40 and pred_traj.npy, if present."""
    for d in sorted(ref_dir.glob(f'*/native/{window}-s{k}')):
        if (d / 'native.mp4').is_file() and (d / 'pred_traj.npy').is_file():
            fr, _ = rb.decode_rgb(d / 'native.mp4', 2)
            return fr[rb.N_HIST:], np.load(d / 'pred_traj.npy'), str(d)
    return None


def plan_batch(E, hist, mats, out: Path, window: str, batches, ref_dirs, n_ref: int, gpu_index: str):
    torch, np = E.torch, E.np
    res = dict(window=window, refs={}, runs=[])
    refs = {}
    # batch-1 references on the released path (model.rollout), from earlier runs or made here
    for k in range(n_ref):
        got = None
        for rd in ref_dirs:
            got = got or native_ref(rd, window, k, np)
        if got is None:
            seed = seeds_for(window, [k])[0]
            torch.cuda.synchronize(); t0 = time.perf_counter()
            fut, _, traj, _, _, _ = E.rollout(hist, mats, seed)
            torch.cuda.synchronize()
            got = (fut, traj, f'made here ({time.perf_counter() - t0:.1f} s)')
        refs[k] = got
        res['refs'][k] = got[2]
    plan = [b for b in batches]
    first_multi = next((b for b in batches if b > 1), None)
    if first_multi:
        plan.append(f'{first_multi}-rerun')
    stored = {}
    for item in plan:
        B = int(str(item).split('-')[0])
        seeds = seeds_for(window, range(B))
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        dm = Pmon(gpu_index, out / f'pmon-batch-{item}.txt')
        dm.start()
        try:
            torch.cuda.synchronize(); t0 = time.perf_counter()
            fut, ctx, trajs, cond, counts, timing = E.rollout_batch(hist, mats, seeds)
            torch.cuda.synchronize(); wall = time.perf_counter() - t0
        except torch.cuda.OutOfMemoryError as exc:
            dm.stop()
            res['runs'].append(dict(batch=item, error=f'OOM: {exc}'[:300]))
            print(json.dumps(dict(batch=item, error='OOM')), flush=True)
            torch.cuda.empty_cache()
            continue
        util = dm.stop()
        row = dict(batch=item, seconds=round(wall, 2), per_sample_s=round(wall / B, 2), steps_mean_s=round(sum(timing['steps']) / 30, 3),
                   decode_s=timing['decode'], encode_context_s=timing['encode_context'], counts=counts, pmon=util,
                   peak_allocated_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2), peak_reserved_gib=round(torch.cuda.max_memory_reserved() / 2**30, 2),
                   frames_sha=[sha(f) for f in fut], traj_sha=[sha(t) for t in trajs])
        cmp = {}
        for k, (rf, rt, _) in refs.items():
            if k >= B:
                continue
            d0 = np.abs(fut[k][0].astype(np.int16) - rf[0])
            dl = np.abs(fut[k][-1].astype(np.int16) - rf[-1])
            dt = np.abs(trajs[k] - rt)
            cmp[k] = dict(identical=bool(np.array_equal(fut[k], rf) and np.array_equal(trajs[k], rt)),
                          first_frame=dict(max=int(d0.max()), mean=round(float(d0.mean()), 4), share_px_diff=round(float((d0.max(-1) > 0).mean()), 4)),
                          last_frame=dict(max=int(dl.max()), mean=round(float(dl.mean()), 3)),
                          traj_first_step=dict(max=round(float(dt[0].max()), 5), mean=round(float(dt[0].mean()), 6)),
                          traj_all=dict(max=round(float(dt.max()), 4), mean=round(float(dt.mean()), 5)))
        row['vs_batch1_released_path'] = cmp
        if str(item).endswith('rerun'):
            prev = stored[B]
            row['rerun_identical'] = bool(all(np.array_equal(a, b) for a, b in zip(prev[0], fut)) and all(np.array_equal(a, b) for a, b in zip(prev[1], trajs)))
        stored.setdefault(B, (fut, trajs))
        if B == 1:
            row['batch1_gen_equals_released'] = cmp.get(0, {}).get('identical')
        res['runs'].append(row)
        np.savez(out / f'batch-{item}-first-last.npz', first=np.stack([f[0] for f in fut]), last=np.stack([f[-1] for f in fut]), traj=np.stack(trajs))
        print(json.dumps({k: v for k, v in row.items() if k not in ('frames_sha', 'traj_sha')}), flush=True)
        (out / 'batch.json').write_text(json.dumps(res, indent=1) + '\n')
        del fut, trajs
    (out / 'batch.json').write_text(json.dumps(res, indent=1) + '\n')
    return res


def plan_options(E, hist, mats, out: Path, window: str, input_root: Path, batches):
    torch, np, m, tok = E.torch, E.np, E.model, E.tokenizer
    res = {}
    real, _ = rb.decode_rgb(input_root / 'windows' / 'dev' / window / 'real.mp4', 2)
    cmap = rb.canvas_map([797, 1669])

    def dcae_once():
        torch.cuda.synchronize(); t0 = time.perf_counter()
        head = E.roundtrip(real[:rb.N_HIST])
        z = E.encode_frames(real[rb.N_HIST:])
        tail = E.decode_groups(z)
        torch.cuda.synchronize()
        return np.concatenate([head, tail]), time.perf_counter() - t0
    base, _ = dcae_once()                       # warm-up
    base, tb = dcae_once()
    rows = dict(baseline=dict(seconds=round(tb, 2)))
    for name, setup, undo in (('cudnn_benchmark', lambda: setattr(torch.backends.cudnn, 'benchmark', True), lambda: setattr(torch.backends.cudnn, 'benchmark', False)),
                              ('tf32_matmul', lambda: setattr(torch.backends.cuda.matmul, 'allow_tf32', True), lambda: setattr(torch.backends.cuda.matmul, 'allow_tf32', False))):
        setup()
        try:
            dcae_once()
            fr, tt = dcae_once()
            fr2, tt2 = dcae_once()
        finally:
            undo()
        d = np.abs(fr.astype(np.int16) - base)
        rows[name] = dict(seconds=round(min(tt, tt2), 2), max_abs_vs_baseline=int(d.max()), mean_abs=round(float(d.mean()), 5),
                          rerun_identical=bool(np.array_equal(fr, fr2)))
    res['dcae_41_frames'] = rows
    print(json.dumps(dict(dcae=rows)), flush=True)
    # TF32 inside the rollout: one generated frame (step_eval) with and without, same noise
    with torch.no_grad():
        img = E.normalize(hist[rb.N_HIST - rb.CF:]).unsqueeze(0).cuda()
        lat = tok.encode_to_z(img[:, :rb.CF])
        pose, yaw = E.condition_poses(mats)

        def one_frame():
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                return E.step_eval_gen(lat, pose, yaw, [torch.Generator().manual_seed(7)])
        tr0, la0 = one_frame()
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            tr1, la1 = one_frame()
        finally:
            torch.backends.cuda.matmul.allow_tf32 = False
        res['tf32_matmul_rollout_frame'] = dict(latents_identical=bool(torch.equal(la0, la1)), traj_identical=bool(torch.equal(tr0, tr1)),
                                                latent_max_abs=float((la0.float() - la1.float()).abs().max()))
    print(json.dumps(dict(tf32_frame=res['tf32_matmul_rollout_frame'])), flush=True)
    # torch.compile of the TrajDiT and VisDiT forwards
    try:
        import torch._inductor.config as ic
        ic.compile_threads = 2
        res['compile'] = compile_test(E, lat, pose, yaw, batches)
    except Exception as exc:  # noqa: BLE001
        res['compile'] = dict(error=f'{type(exc).__name__}: {exc}'[:600])
    print(json.dumps(dict(compile=res['compile'])), flush=True)
    (out / 'options.json').write_text(json.dumps(res, indent=1) + '\n')
    # 50 steps, one sample, information only
    seed = seeds_for(window, [0])[0]
    m.args.num_sampling_steps = 50
    try:
        torch.cuda.synchronize(); t0 = time.perf_counter()
        fut, _, traj, _, counts, timing = E.rollout(hist, mats, seed)
        torch.cuda.synchronize()
        res['steps50_one_sample'] = dict(seconds=round(time.perf_counter() - t0, 2), steps_mean_s=round(sum(timing['steps']) / 30, 3), counts=counts,
                                         note='information only; the benchmark keeps 100 steps')
    finally:
        m.args.num_sampling_steps = 100
    print(json.dumps(dict(steps50=res['steps50_one_sample'])), flush=True)
    (out / 'options.json').write_text(json.dumps(res, indent=1) + '\n')
    return res


def compile_test(E, lat, pose, yaw, batches):
    torch, m = E.torch, E.model
    out = {}
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        pt = E.poses_to_indices(pose, m.pose_x_vocab_size, m.pose_y_vocab_size)
        yt = E.yaws_to_indices(yaw, m.yaw_vocab_size)
        stt1, pe1 = m.model.evaluate(lat, pt, yt, sample_last=True)
    for mode in ('default', 'reduce-overhead'):
        for B in sorted({batches[0], batches[-1]}):
            key = f'{mode}-b{B}'
            try:
                img_ids, cond_ids, traj_ids = E.prepare_ids(B, m.h, m.w, m.total_token_size, m.traj_len)
                g = torch.Generator().manual_seed(3)
                stt, pe = stt1.repeat(B, 1, 1), pe1.repeat(B, 1)
                xt = torch.randn(B, m.traj_len, m.traj_token_size, generator=g).to(stt)
                xi = torch.randn(B, m.img_token_size, m.vae_emb_dim, generator=g).to(stt)
                tv = torch.full((B,), 0.5, dtype=stt.dtype, device=stt.device)
                row = {}
                for name, mod, kwargs, reps in (('traj', m.traj_dit, dict(traj=xt, traj_ids=traj_ids, cond=stt, cond_ids=cond_ids, timesteps=tv), 50),
                                                ('vis', m.dit, dict(img=xi, img_ids=img_ids, cond=stt, cond_ids=cond_ids, timesteps=tv, y=pe), 20)):
                    torch._dynamo.reset()
                    fc = torch.compile(mod.forward, mode=mode)
                    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        ref = mod(**kwargs)
                        t0 = time.perf_counter()
                        a = fc(**kwargs).clone()
                        torch.cuda.synchronize()
                        comp_s = time.perf_counter() - t0
                        b = fc(**kwargs).clone()
                        for _ in range(3):
                            fc(**kwargs)
                        torch.cuda.synchronize(); t0 = time.perf_counter()
                        for _ in range(reps):
                            fc(**kwargs)
                        torch.cuda.synchronize(); tc = (time.perf_counter() - t0) / reps
                        torch.cuda.synchronize(); t0 = time.perf_counter()
                        for _ in range(reps):
                            mod(**kwargs)
                        torch.cuda.synchronize(); te = (time.perf_counter() - t0) / reps
                    row[name] = dict(compile_s=round(comp_s, 1), eager_ms=round(te * 1e3, 3), compiled_ms=round(tc * 1e3, 3), speedup=round(te / tc, 2),
                                     max_abs_vs_eager=float((a.float() - ref.float()).abs().max()), identical_to_eager=bool(torch.equal(a, ref)),
                                     deterministic=bool(torch.equal(a, b)))
                out[key] = row
            except Exception as exc:  # noqa: BLE001
                out[key] = dict(error=f'{type(exc).__name__}: {exc}'[:500])
            print(json.dumps({key: out[key]}), flush=True)
    torch._dynamo.reset()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input-root', type=Path, required=True)
    ap.add_argument('--window', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--plan', default='micro,breakdown,prof')
    ap.add_argument('--batches', default='1,4,8,16')
    ap.add_argument('--ref-dir', type=Path, nargs='*', default=[])
    ap.add_argument('--n-ref', type=int, default=4)
    ap.add_argument('--leave-free-mib', type=int, default=6144)
    ap.add_argument('--max-gpu-mib', type=int, default=31000)
    ap.add_argument('--expect-gpu-uuid', default=None)
    ap.add_argument('--min-host-avail-gib', type=float, default=10.0)
    args = ap.parse_args()
    models_dir, epona_src = epona_paths()
    args.out.mkdir(parents=True, exist_ok=True)
    if rb.mem_available_gib() < args.min_host_avail_gib:
        raise SystemExit(f'host MemAvailable {rb.mem_available_gib():.1f} GiB < {args.min_host_avail_gib}')
    import numpy as np
    import torch
    torch.set_num_threads(2)
    gpu = rb.gpu_info(torch)
    if args.expect_gpu_uuid and gpu.get('uuid') != args.expect_gpu_uuid:
        raise SystemExit(f'visible GPU {gpu.get("uuid")} is not the expected {args.expect_gpu_uuid}')
    free, total = torch.cuda.mem_get_info()
    cap = int(min(args.max_gpu_mib, free / 2**20 - args.leave_free_mib - 1024))
    torch.cuda.set_per_process_memory_fraction(min(1.0, cap * 2**20 / total), 0)
    E = rb.Epona(models_dir, epona_src, 'size', False, 2)
    hist, mats = load_window(args.input_root, args.window, np)
    batches = [int(b) for b in args.batches.split(',')]
    meta = dict(gpu=gpu, cap_mib=cap, free_at_start_mib=round(free / 2**20), software=rb.software(E), window=args.window, plan=args.plan,
                host_memory_after_load=rb.host_memory())
    (args.out / f'meta-{args.plan.replace(",", "-")}.json').write_text(json.dumps(meta, indent=1) + '\n')
    sel = (os.environ.get('CUDA_VISIBLE_DEVICES') or '0').split(',')[0]
    for p in args.plan.split(','):
        t0 = time.time()
        if p == 'micro':
            plan_micro(E, hist, mats, args.out, batches, sel)
        elif p == 'breakdown':
            plan_breakdown(E, hist, mats, args.out, args.window, sel)
        elif p == 'prof':
            plan_prof(E, hist, mats, args.out, args.window, batches)
        elif p == 'batch':
            plan_batch(E, hist, mats, args.out, args.window, batches, args.ref_dir, args.n_ref, sel)
        elif p == 'tput':
            plan_tput(E, hist, mats, args.out, args.window, batches, sel)
        elif p == 'options':
            plan_options(E, hist, mats, args.out, args.window, args.input_root, batches)
        else:
            raise SystemExit(f'unknown plan {p}')
        print(json.dumps(dict(plan=p, seconds=round(time.time() - t0, 1), host=rb.host_memory())), flush=True)


if __name__ == '__main__':
    main()
