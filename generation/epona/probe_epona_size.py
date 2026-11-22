#!/usr/bin/env python3
"""Feasibility probe: Epona at a reduced model-frame size (DEV only; a development tool, not part of any run).

The released code derives every size from the config's image_size: the latent grid h x w = image_size / 32,
img_token_size = h * w, the STT's 2D RoPE table (compute_axial_cis over latent_size), the VisDiT's (row, col) RoPE ids
(prepare_ids), the DCAE temporal blocks' token count, and the Flux schedule shift (get_schedule over img_token_size).
No learned spatial position embedding exists (the STT's only learned position table is its per-frame time embedding),
so the pinned weights load unchanged at another size. This probe loads them with image_size overridden (process-local
override of run_epona_bench's config loader and checks; run_epona_bench.py itself is not changed), renders DEV windows'
band at that size with the bench's render rule, and writes run_epona_bench.py's clip layout: 41 frames, 5 fps, canvas
512x288; frames 0-10 the per-frame DCAE round trip of the history, frames 11-40 generated (rollout(): the released
step_eval at batch 1, the demo's seeding) or, for the reference, the real frames encoded one at a time and decoded in
groups of 6.

  probe_epona_size.py render --size 512x256 --band 579 1451 --windows W1,W2 --input-root INPUTS_V1 --bench-root BENCH --out DIR
  probe_epona_size.py gpu    --size 512x256 --band 579 1451 --windows W1,W2 --input-root INPUTS_V1 --out DIR --tag main
                             --stages recon,cont,tput --expect-gpu-uuid UUID [--seeds-k 0,1] [--tput-batches 1,16]
  probe_epona_size.py gpu    ... --tag rerun --stages cont --seeds-k 0          (second process: determinism)
  probe_epona_size.py check  --size 512x256 --band 579 1451 --windows W1,W2 --out DIR --bench-root BENCH [--native-root DIR ...]

Outputs: DIR/inputs/<w>/{history,real}.mp4 at the probe size; DIR/<tag>/native/<w>-{recon,s<k>}/{vision.mp4 (canvas),
model.mp4 (probe size), pred_traj.npy, sample_outputs.json}; DIR/<tag>/meta.json; DIR/check.json and contact sheets.
Weights and checkout: EPONA_MODELS_DIR / EPONA_SRC (or EPONA_ENV) from the environment or models/epona/local.env.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_epona_bench as rb  # noqa: E402
import profile_epona as pe  # noqa: E402  (epona_paths, seeds_for, Pmon)

CF, N_HIST, N_FUT, GROUP = rb.CF, rb.N_HIST, rb.N_FUT, rb.GROUP
TICKS = rb.TICKS                       # 41 ticks, k = -20..60 step 2
SRC_BAND = (797, 1669)                 # the measurement band = the canvas
CW, CH = rb.CW, rb.CH                  # 512 x 288


def parse_size(s: str):
    w, h = (int(v) for v in s.lower().split('x'))
    if w % 32 or h % 32:
        raise SystemExit(f'size {s}: width and height must be multiples of 32 (DCAE f32)')
    return w, h


def probe_cmap(band, H: int) -> dict:
    """run_epona_bench.canvas_map for a model frame of H rows: model rows covering source rows [797, min(b, 1669))."""
    a, b = int(band[0]), int(band[1])
    if b - a != 872 or a > 797 or ((797 - a) * H) % 872:
        raise SystemExit(f'band {band} at {H} rows: needs b - a = 872, a <= 797 and (797 - a) * H / 872 an integer')
    top = (797 - a) * H // 872
    rows_c = round((min(b, 1669) - 797) * CH / 872)
    return dict(model_band_rows=[a, b], model_rows=[top, H], canvas_rows=[0, rows_c])


def stt_rope_layout(h: int, w: int) -> dict:
    """Where the STT's 2D RoPE (compute_axial_cis(end_x = latent_size[0] = h, end_y = w); token t -> (t % h, t // h)) puts
    the row-major latent tokens (t = r * w + c): consistent iff every vertical neighbour pair has the same offset."""
    pos = [((r * w + c) % h, (r * w + c) // h) for r in range(h) for c in range(w)]
    vert = sorted({(pos[(r + 1) * w + c][0] - pos[r * w + c][0], pos[(r + 1) * w + c][1] - pos[r * w + c][1]) for r in range(h - 1) for c in range(w)})
    horiz = sorted({(pos[r * w + c + 1][0] - pos[r * w + c][0], pos[r * w + c + 1][1] - pos[r * w + c][1]) for r in range(h) for c in range(w - 1)})
    return dict(grid_hw=[h, w], rope_x_range=[min(p[0] for p in pos), max(p[0] for p in pos)], rope_y_range=[min(p[1] for p in pos), max(p[1] for p in pos)],
                vertical_neighbour_offsets=[list(v) for v in vert], horizontal_neighbour_offsets=[list(v) for v in horiz],
                consistent=len(vert) == 1 and w % h == 0)


# ----------------------------------------------------------------------------------------------------------------------
class SizedEpona(rb.Epona):
    """rb.Epona with the config's image_size overridden (everything else as pinned; the weights are not touched)."""

    def __init__(self, *args, size_hw, **kw):
        self.size_hw = [int(size_hw[0]), int(size_hw[1])]
        orig_load, orig_checked = rb.load_config, dict(rb.DEF['config_checked'])
        rb.load_config = lambda path, overrides: orig_load(path, dict(overrides, image_size=tuple(self.size_hw)))
        rb.DEF['config_checked'] = dict(orig_checked, image_size=self.size_hw)
        try:
            super().__init__(*args, **kw)
        finally:
            rb.load_config, rb.DEF['config_checked'] = orig_load, orig_checked

    def _read_back_settings(self) -> dict:
        m, cfg, torch, vae = self.model, self.cfg, self.torch, self.tokenizer.vae
        npar = lambda mod: int(sum(p.numel() for p in mod.parameters()))  # noqa: E731
        att = m.model.causal_time_space_blocks[0].space_block.attn
        got = dict(world_model=type(m).__name__, params_m=dict(stt=round(npar(m.model) / 1e6, 1), vis_dit=round(npar(m.dit) / 1e6, 1),
                   traj_dit=round(npar(m.traj_dit) / 1e6, 1), dcae=round(npar(vae) / 1e6, 1)), param_dtype=str(next(m.dit.parameters()).dtype),
                   condition_frames=int(m.condition_frames), traj_len=int(m.traj_len), latent_hw=[self.lat_h, self.lat_w],
                   img_token_size=int(m.img_token_size), total_token_size=int(m.total_token_size), num_sampling_steps=int(cfg.num_sampling_steps),
                   diffusion_model_type=cfg.diffusion_model_type, downsample_fps=int(cfg.downsample_fps), image_size=list(cfg.image_size),
                   decoder_temporal=bool(vae.add_decoder_temporal), encoder_temporal=bool(vae.add_encoder_temporal),
                   dcae_temporal_frames=int(getattr(vae, 'condition_frames', 0)), dcae_space_token_size=int(vae.token_size),
                   dcae_space_rope_shape=list(vae.freqs_cis_space_img.shape), stt_rope_grid=list(att.patch_size),
                   stt_rope_table_shape=list(att.freqs_cis_singlescale.shape), stt_time_emb_shape=list(m.model.time_emb.shape),
                   stt_mask_space_shape=list(m.model.mask_space.shape), checkpoint_keys=self.ckpt_keys,
                   tf32_matmul=bool(torch.backends.cuda.matmul.allow_tf32), cudnn_deterministic=bool(torch.backends.cudnn.deterministic),
                   cudnn_benchmark=bool(torch.backends.cudnn.benchmark), config_override=dict(image_size=self.size_hw))
        h, w = self.size_hw[0] // 32, self.size_hw[1] // 32
        expect = dict(condition_frames=CF, traj_len=15, latent_hw=[h, w], img_token_size=h * w, num_sampling_steps=100, diffusion_model_type='flow',
                      downsample_fps=5, image_size=self.size_hw, decoder_temporal=True, encoder_temporal=False, dcae_temporal_frames=6,
                      dcae_space_token_size=h * w, stt_rope_grid=[h, w])
        bad = {k: (got[k], v) for k, v in expect.items() if got[k] != v}
        if bad:
            raise SystemExit(f'the probe model does not run the expected settings: {bad}')
        return got


# ----------------------------------------------------------------------------------------------------------------------
def cmd_render(a):
    import cv2
    import numpy as np
    W, H = parse_size(a.size)
    rep = {}
    for w in a.windows.split(','):
        build = json.loads((a.input_root / 'windows' / 'dev' / w / 'build.json').read_text())
        if build['split'] != 'dev':
            raise SystemExit(f'{w}: not a DEV window')
        paths = build['ticks']['frame_paths']
        frames, src = [], {}
        for k in TICKS:
            p = paths[str(k)]
            if p not in src:
                img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
                src[p] = cv2.resize(img[a.band[0]:a.band[1]], (W, H), interpolation=cv2.INTER_AREA)
            frames.append(src[p])
        frames = np.stack(frames)
        out = a.out / 'inputs' / w
        out.mkdir(parents=True, exist_ok=True)
        r = {}
        for name, clip in (('history.mp4', frames[:N_HIST]), ('real.mp4', frames)):
            rb.write_lossless(clip, out / name, 2)
            back, _ = rb.decode_rgb(out / name, 2)
            r[name] = dict(frames=len(clip), identical=bool(np.array_equal(back, clip)), sha256=rb.sha256_file(out / name))
        if list(a.band) == list(SRC_BAND) and (W, H) == (CW, CH):   # the canvas itself: must equal the bench canvas
            ref, _ = rb.decode_rgb(a.bench_root / 'windows-dev' / w / 'canvas' / 'real.mp4', 2)
            r['equals_bench_canvas_even_ticks'] = bool(all(np.array_equal(frames[i], ref[k + 20]) for i, k in enumerate(TICKS)))
        rep[w] = r
    (a.out / 'inputs' / 'render.json').write_text(json.dumps(dict(size=[W, H], band=list(a.band), windows=rep), indent=1) + '\n')
    print(json.dumps(rep))


def write_clip(E, out_dir: Path, clip_m, cmap, rec: dict, traj=None):
    import numpy as np
    out_dir.mkdir(parents=True, exist_ok=True)
    canvas = E.to_canvas(clip_m, cmap)
    rb.write_lossless(canvas, out_dir / 'vision.mp4', 2, preset='ultrafast')
    rb.write_lossless(clip_m, out_dir / 'model.mp4', 2, preset='ultrafast')
    back, _ = rb.decode_rgb(out_dir / 'vision.mp4', 2)
    if not np.array_equal(back, canvas):
        raise RuntimeError(f'{out_dir}: vision.mp4 does not decode to the written frames')
    if traj is not None:
        np.save(out_dir / 'pred_traj.npy', traj)
    rec.update(video_sha256=rb.sha256_file(out_dir / 'vision.mp4'), model_video_sha256=rb.sha256_file(out_dir / 'model.mp4'),
               prefix11_canvas_sha256=rb.sha256_bytes(np.ascontiguousarray(canvas[:N_HIST]).tobytes()), canvas_map=cmap,
               pred_traj_sha256=rb.sha256_file(out_dir / 'pred_traj.npy') if traj is not None else None)
    (out_dir / 'sample_outputs.json').write_text(json.dumps(rec, indent=1) + '\n')


def tput(E, hist, mats, seeds, n_frames: int) -> dict:
    """Per-step time of step_eval_gen (the batched path; rows seeded per sample) for n_frames autoregressive steps, no
    decoding; synchronised timers; the first step is reported apart (warm-up)."""
    torch, B = E.torch, len(seeds)
    img = E.normalize(hist[N_HIST - CF:]).unsqueeze(0).cuda()
    pose1, yaw1 = E.condition_poses(mats)
    gens = [torch.Generator().manual_seed(int(s)) for s in seeds]
    torch.cuda.reset_peak_memory_stats()
    steps = []
    with torch.no_grad():
        start1 = E.tokenizer.encode_to_z(img[:, :CF])
        start, pose, yaw = start1.repeat(B, 1, 1, 1), pose1.repeat(B, 1, 1), yaw1.repeat(B, 1, 1)
        for _ in range(n_frames):
            torch.cuda.synchronize()
            ts = time.time()
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                pt, pl = E.step_eval_gen(start, pose, yaw, gens)
            pp, py = pt[:, 0:1, 0:2], pt[:, 0:1, 2:3]
            pose = torch.cat((pose[:, 1:CF], pp, pp), dim=1)
            yaw = torch.cat((yaw[:, 1:CF], py, py), dim=1)
            start = torch.cat((start[:, 1:CF], E.rearrange(pl, '(b f) h w c -> b f (h w) c', f=1)), dim=1)
            torch.cuda.synchronize()
            steps.append(round(time.time() - ts, 3))
    later = steps[1:] or steps
    return dict(batch=B, frames=n_frames, step_s=steps, step_mean_s_after_first=round(sum(later) / len(later), 3),
                per_sample_per_frame_s=round(sum(later) / len(later) / B, 4),
                peak_alloc_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2), peak_reserved_gib=round(torch.cuda.max_memory_reserved() / 2**30, 2))


def cmd_gpu(a):
    import numpy as np
    import torch
    if rb.mem_available_gib() < a.min_host_avail_gib:
        raise SystemExit(f'host MemAvailable {rb.mem_available_gib():.1f} GiB < {a.min_host_avail_gib}')
    W, H = parse_size(a.size)
    cmap = probe_cmap(a.band, H)
    models_dir, src = (a.models_dir, a.epona_src) if a.models_dir and a.epona_src else pe.epona_paths()
    gpu = rb.gpu_info(torch)
    if a.expect_gpu_uuid and gpu.get('uuid') != a.expect_gpu_uuid:
        raise SystemExit(f'visible GPU {gpu.get("uuid")} is not the expected {a.expect_gpu_uuid}')
    free, total = torch.cuda.mem_get_info()
    cap = int(min(a.max_gpu_mib, free / 2**20 - a.leave_free_mib - 1024))
    if cap < a.min_cap_mib:
        raise SystemExit(f'GPU cap {cap} MiB < {a.min_cap_mib} MiB (free {free / 2**20:.0f} MiB, leave {a.leave_free_mib})')
    torch.cuda.set_per_process_memory_fraction(min(1.0, cap * 2**20 / total), 0)
    out = a.out / a.tag
    (out / 'native').mkdir(parents=True, exist_ok=True)
    sel = (os.environ.get('CUDA_VISIBLE_DEVICES') or '0').split(',')[0]
    pmon = pe.Pmon(sel, out / 'pmon.txt')
    pmon.start()
    import atexit
    atexit.register(lambda: pmon.proc.poll() is None and pmon.proc.terminate())   # no orphaned nvidia-smi on an error
    t0 = time.time()
    E = SizedEpona(Path(models_dir), Path(src), 'size', False, 2, size_hw=(H, W))
    from models.modules.sampling import get_lin_function
    meta = dict(schema='wm-calib-bench.epona.size_probe.v1', tag=a.tag, size_wh=[W, H], band=list(a.band), canvas_map=cmap, gpu=gpu, cap_mib=cap,
                free_at_start_mib=round(free / 2**20), load_s=round(time.time() - t0, 1), settings=E.settings, software=rb.software(E),
                stt_rope_layout=stt_rope_layout(H // 32, W // 32), stt_rope_layout_native=stt_rope_layout(16, 32),
                flux_schedule_mu=dict(this=round(get_lin_function(y1=0.5, y2=1.15)((H // 32) * (W // 32)), 4), native=round(get_lin_function(y1=0.5, y2=1.15)(512), 4)),
                host_memory_after_load=rb.host_memory(), stages=a.stages, seeds_k=a.seeds_k)
    (out / 'meta.json').write_text(json.dumps(meta, indent=1) + '\n')
    print(json.dumps(dict(loaded=True, load_s=meta['load_s'], latent_hw=E.settings['latent_hw'], stt_rope=meta['stt_rope_layout']['consistent'])), flush=True)
    stages = a.stages.split(',')
    ks = [int(k) for k in a.seeds_k.split(',')] if a.seeds_k else []
    results = []
    for w in (a.windows.split(',') if ('recon' in stages or 'cont' in stages) else []):
        hist, _ = rb.decode_rgb(a.out / 'inputs' / w / 'history.mp4', 2)
        real, _ = rb.decode_rgb(a.out / 'inputs' / w / 'real.mp4', 2)
        mats = rb.load_poses(a.input_root / 'windows' / 'dev' / w / 'past_poses.json', np)
        if hist.shape != (N_HIST, H, W, 3) or real.shape != (len(TICKS), H, W, 3):
            raise SystemExit(f'{w}: inputs are {hist.shape} / {real.shape}, expected {H}x{W}')
        t = time.time()
        prefix = E.roundtrip(hist)
        t_prefix = round(time.time() - t, 2)
        base = dict(window=w, size_wh=[W, H], band=list(a.band), tag=a.tag, prefix_s=t_prefix, gpu=dict(name=gpu.get('name'), uuid=gpu.get('uuid')))
        if 'recon' in stages:
            torch.cuda.reset_peak_memory_stats()
            t = time.time()
            fut = E.decode_groups(E.encode_frames(real[N_HIST:]))
            clip = np.concatenate([prefix, fut])
            rec = dict(base, name=f'{w}-recon', mode='recon', seed=0, seconds_model=round(time.time() - t, 2),
                       peak_alloc_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2))
            write_clip(E, out / 'native' / f'{w}-recon', clip, cmap, rec)
            results.append(dict(name=rec['name'], s=rec['seconds_model']))
            print(json.dumps(results[-1]), flush=True)
        if 'cont' in stages:
            for k, seed in zip(ks, pe.seeds_for(w, ks)):
                torch.cuda.reset_peak_memory_stats()
                t = time.time()
                frames, context, traj, cond_record, counts, timing = E.rollout(hist, mats, seed, decode_context=True)
                secs = round(time.time() - t, 2)
                clip = np.concatenate([prefix, frames])
                ctx = np.abs(context.astype(np.int16) - prefix[1:].astype(np.int16))
                rec = dict(base, name=f'{w}-s{k}', mode='continue', k=k, seed=int(seed), seconds_model=secs, timing=timing, counts=counts,
                           conditioning=cond_record, context_vs_prefix=dict(max=int(ctx.max()), mean=round(float(ctx.mean()), 4)),
                           peak_alloc_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                           peak_reserved_gib=round(torch.cuda.max_memory_reserved() / 2**30, 2))
                write_clip(E, out / 'native' / f'{w}-s{k}', clip, cmap, rec, traj)
                results.append(dict(name=rec['name'], s=secs, per_frame_s=round(sum(timing['steps']) / len(timing['steps']), 3), counts=counts))
                print(json.dumps(results[-1]), flush=True)
    if 'tput' in stages:
        w = a.tput_window or a.windows.split(',')[0]
        hist, _ = rb.decode_rgb(a.out / 'inputs' / w / 'history.mp4', 2)
        mats = rb.load_poses(a.input_root / 'windows' / 'dev' / w / 'past_poses.json', np)
        tp = []
        for B in [int(b) for b in a.tput_batches.split(',')]:
            try:
                tp.append(dict(window=w, **tput(E, hist, mats, pe.seeds_for(w, range(B)), a.tput_frames)))
            except torch.cuda.OutOfMemoryError as e:  # noqa: PERF203
                tp.append(dict(window=w, batch=B, error=f'out of memory under the cap ({cap} MiB): {str(e)[:200]}'))
                torch.cuda.empty_cache()
            print(json.dumps(tp[-1]), flush=True)
        (out / 'tput.json').write_text(json.dumps(tp, indent=1) + '\n')
    meta.update(results=results, seconds_total=round(time.time() - t0, 1), pmon=pmon.stop(), host_memory_end=rb.host_memory())
    (out / 'meta.json').write_text(json.dumps(meta, indent=1) + '\n')
    print(json.dumps(dict(done=True, seconds_total=meta['seconds_total'], pmon=meta['pmon'])), flush=True)


# ----------------------------------------------------------------------------------------------------------------------
def framing(c, rows: int) -> dict:
    """Vertical content shift (phase correlation of the covered canvas rows, grey) between t0 (frame 10) and later frames."""
    import cv2
    import numpy as np

    def dy(i, j):
        ga = cv2.cvtColor(c[i][:rows], cv2.COLOR_RGB2GRAY).astype(np.float32)
        gb = cv2.cvtColor(c[j][:rows], cv2.COLOR_RGB2GRAY).astype(np.float32)
        (sx, sy), resp = cv2.phaseCorrelate(ga, gb, cv2.createHanningWindow((ga.shape[1], ga.shape[0]), cv2.CV_32F))
        return round(float(sy), 1)
    return dict(dy_t0_to_0p2s=dy(10, 11), dy_t0_to_1s=dy(10, 15), dy_t0_to_2s=dy(10, 20), dy_hist_last_step=dy(9, 10))


def cmd_check(a):
    import cv2
    import numpy as np
    import check_epona_smoke as cs
    W, H = parse_size(a.size)
    rows = probe_cmap(a.band, H)['canvas_rows'][1]
    rep = dict(size=[W, H], band=list(a.band), windows={})
    sheet_rows = []
    for w in a.windows.split(','):
        main = a.out / 'main' / 'native'
        clips = {n: rb.decode_rgb(main / f'{w}-{n}' / 'vision.mp4', 2)[0] for n in ('recon', 's0', 's1') if (main / f'{w}-{n}').is_dir()}
        r = dict(prefix_identical=bool(all(np.array_equal(clips['recon'][:N_HIST], c[:N_HIST]) for c in clips.values())))
        rr = a.out / 'rerun' / 'native' / f'{w}-s0'
        if rr.is_dir():
            m0, m1 = json.loads((main / f'{w}-s0' / 'sample_outputs.json').read_text()), json.loads((rr / 'sample_outputs.json').read_text())
            r['rerun_s0_identical'] = dict(vision=m0['video_sha256'] == m1['video_sha256'], model=m0['model_video_sha256'] == m1['model_video_sha256'],
                                           pred_traj=m0['pred_traj_sha256'] == m1['pred_traj_sha256'])
        r['framing'] = {n: framing(c, rows) for n, c in clips.items()}
        r['mean_abs_step'] = {n: [round(float(np.abs(c[i].astype(np.int16) - c[i - 1]).mean()), 2) for i in (9, 10, 11, 12, 15)] for n, c in clips.items()}
        traj = {n: np.load(main / f'{w}-{n}' / 'pred_traj.npy') for n in ('s0', 's1') if (main / f'{w}-{n}' / 'pred_traj.npy').is_file()}
        r['own_ego_x_m'] = {n: [round(float(t[:i, 0, 0].sum()), 2) for i in (5, 10, 20, 30)] for n, t in traj.items()}  # integrated first steps
        rep['windows'][w] = r
        raw, _ = rb.decode_rgb(a.bench_root / 'windows-dev' / w / 'canvas' / 'real.mp4', 2)
        line = [np.concatenate([cs.label(cv2.resize(raw[k + 20], (256, 144), interpolation=cv2.INTER_AREA), f'{w[:8]} raw {k / 10:+.0f}s') for k in cs.SHEET_K], 1)]
        for n, c in clips.items():
            line.append(np.concatenate([cs.label(cv2.resize(c[TICKS.index(k)], (256, 144), interpolation=cv2.INTER_AREA), f'{W}x{H} {n} {k / 10:+.0f}s') for k in cs.SHEET_K], 1))
        for root in a.native_root or []:
            for n in ('s0', 's1'):
                d = Path(root) / f'{w}-{n}'
                if (d / 'vision.mp4').is_file():
                    c = rb.decode_rgb(d / 'vision.mp4', 2)[0]
                    line.append(np.concatenate([cs.label(cv2.resize(c[TICKS.index(k)], (256, 144), interpolation=cv2.INTER_AREA), f'1024x512 {n} {k / 10:+.0f}s') for k in cs.SHEET_K], 1))
        sheet_rows += line
        cv2.imwrite(str(a.out / f'sheet-{w[:8]}.png'), cv2.cvtColor(np.concatenate(line, 0), cv2.COLOR_RGB2BGR))
    (a.out / 'check.json').write_text(json.dumps(rep, indent=1) + '\n')
    print(json.dumps(rep, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=['render', 'gpu', 'check'])
    ap.add_argument('--size', required=True, help='WxH, multiples of 32')
    ap.add_argument('--band', type=int, nargs=2, required=True, help='source rows [a, b) rendered into the model frame (872 rows)')
    ap.add_argument('--windows', required=True)
    ap.add_argument('--input-root', type=Path, help='the 1024x512 input root (build.json frame paths, past_poses.json)')
    ap.add_argument('--bench-root', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--tag', default='main')
    ap.add_argument('--stages', default='recon,cont')
    ap.add_argument('--seeds-k', default='0,1')
    ap.add_argument('--tput-window')
    ap.add_argument('--tput-batches', default='1,16')
    ap.add_argument('--tput-frames', type=int, default=6)
    ap.add_argument('--models-dir', type=Path)
    ap.add_argument('--epona-src', type=Path)
    ap.add_argument('--expect-gpu-uuid')
    ap.add_argument('--leave-free-mib', type=int, default=6144)
    ap.add_argument('--max-gpu-mib', type=int, default=30000)
    ap.add_argument('--min-cap-mib', type=int, default=16000)
    ap.add_argument('--min-host-avail-gib', type=float, default=10.0)
    ap.add_argument('--native-root', action='append', help='a native/ directory with 1024x512 s0/s1 clips of the same windows (sheet)')
    a = ap.parse_args()
    if a.cmd != 'check' and a.input_root is None:
        ap.error('--input-root is required')
    {'render': cmd_render, 'gpu': cmd_gpu, 'check': cmd_check}[a.cmd](a)


if __name__ == '__main__':
    main()
