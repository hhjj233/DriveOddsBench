# DrivingWorld kit (1.4 s of 10 Hz frames + past ego poses)

This kit generated the DrivingWorld results of the paper; its protocol was frozen on DEV before any TEST clip was generated. It follows the
DriveLaW-Video kit layout. The model is the released DrivingWorld world model
(YvanYin/DrivingWorld `faceb9861582c2ecad0d82c1f97699a1a314cb62`, MIT; weights
huxiaotaostasy/DrivingWorld `949a282034a8f2116d53b194c294f4eeea8e8697`, MIT, ungated): a 1.0 B-parameter
next-state GPT (12 temporal-multimodal layers + 6 internal-frame AR layers, width 1536) over a 70 M video VQ tokenizer
(16x downsampling, codebook 16,384). One state = 1 yaw token + 2 location tokens + 512 image tokens of a 256x512 frame.
The licence was accepted before the download and the installation.

## Files

| file | role |
|---|---|
| `drivingworld_pins.json` | code commit and the git blob of every tracked file, weight revision/sizes/sha256, allowed pickle globals, all bench settings |
| `drivingworld_common.py` | hashes, guards, the 10 Hz grid, model bands and canvas mapping, pose encoding, shared seeds, lossless video I/O (no torch) |
| `build_drivingworld_inputs.py` | raw AV2 -> 10 Hz model frames (history; DEV real future) and past-only ego poses |
| `make_drivingworld_specs.py` | continuation specs (shared seed table, `--samples 0-15`) and reference specs |
| `drivingworld_rollout.py` | the next-state rollout on the released modules (own poses, per-sample generators, per-row decode) |
| `run_drivingworld_bench.py` | loads the model once, runs every spec (one batch per window), writes `<run>/native/<name>/` |
| `canvas_spec_drivingworld.json` | canvas-instrument spec (new file; instrument unchanged); `_top` = the alternative band |
| `download_drivingworld_weights.py`, `prepare_source.py`, `prepare_env.sh`, `requirements-drivingworld.{in,lock.txt}`, `lock-pyproject.toml` | pinned weights, exact source, hash-locked environment |
| `check_cpu.py`, `check_strict_load.py`, `check_smoke.py` | CPU contract tests, CPU strict loads/tokenizer round trip/pose probe, DEV smoke structural checks |

Machine paths go in the git-ignored `local.env` (template `local.env.example`); every tool takes explicit arguments.

## Model behaviour (registered settings)

These settings were fixed before the freeze; the `decisions_pending` list in `drivingworld_pins.json` records the options as they stood when the kit was written.

- Frames: the window's own tick frames (nearest 20 Hz camera frame to t0 + k x 0.1 s). 10 Hz = the canvas rate, so
  vision.mp4 frame i = tick i-20 = native frame i, no resampling and no repeated steps.
- Model band `centred` (default): the official loader's centre crop of the raw 1550x2048 frame (resize to width 512, keep
  the middle 256 rows = source rows [636, 1412)), realised on the canvas pixel grid: a 512x360 INTER_AREA render of source
  rows [579, 1669), rows [19, 275). Model rows [53, 256) are canvas rows [0, 203) exactly; canvas rows [203, 288) are
  black. Alternative `top`: canvas rows [0, 256) of the band render, rows [256, 288) black.
- Conditioning: the 15 frames at ticks -14..0 (released `condition_frames=15`; 1.4 s of the 2 s history) and their
  AV2 ego poses (exact camera-timestamp poses, released `get_meta_data` encoding: per-step (x, y) m and yaw deg, entry 0
  zero). No pose after the anchor is read; no future pose is ever given.
- Generation: 60 next states (ticks 1..60). Pose mode `released_long_term_last_step` (the TEST setting, `drivingworld_pins.json`):
  every state is given the last history pose step (x, y, yaw), as the released `tools/test_long_term_demo.py`, and no pose
  token is sampled; then 512 image tokens (released `eval_first_second` / `eval_third`, bf16 autocast as in the demos), top-k 30,
  temperature 1.0 (the released sampler settings). The registered alternative `own` samples yaw, then x, then y with the
  model's own heads and feeds the decoded pose back as the next window entry; the window's first entry is zeroed every step (demos).
- Kit additions (the release lacks them): own-pose sampling (the released inference only takes given poses; `eval_third`
  applies `pose_x_head` to the y position, so y uses the training head `pose_y_head`); one torch generator per sample
  (seed_of(window, k)); per-row `z_to_clips` decoding (the released one decodes only the last frame of a batch); no KV cache.
- Batch: one window per batch, rows = samples in ascending k. With one row and the same seed the kit's image-token loop
  equals the released `generate_gt_pose_gt_yaw` token for token (CPU test; `--equivalence-check` repeats it on the GPU).
- Prefix (canvas frames 0-20) shared by every clip of a window: the VQ round trip of the real history. Reference: every
  real future frame tokenised as the newest frame of a 15-frame window and decoded like a generated frame.

## CPU preparation

```bash
source local.env   # or set the variables by hand
"$CPU_PY" -B check_cpu.py --src "$DW_SRC"; "$DW_PY" -B check_cpu.py --src "$DW_SRC"
"$CPU_PY" -B prepare_source.py --dest "$DW_SRC"                                    # fresh machines only
"$CPU_PY" -B download_drivingworld_weights.py --dest "$DW_WEIGHTS" --approval "$DW_WORK/owner-license-approval.json"
bash prepare_env.sh "$UV" "$PYTHON310" "$NEW_VENV"
CUDA_VISIBLE_DEVICES="" "$DW_PY" -B check_strict_load.py --src "$DW_SRC" --weights "$DW_WEIGHTS" --report NEW.json \
  --input-root "$DW_WORK/inputs-dev-smoke-v1" --roundtrip 2eeaccb0-t2.5
"$CPU_PY" -B build_drivingworld_inputs.py --bench "$WCB_BENCH" --split dev --names "$DW_WORK/names-dev-smoke.txt" \
  --out "$DW_WORK/inputs-dev-smoke-v1"
"$CPU_PY" -B make_drivingworld_specs.py --bench "$WCB_BENCH" --input-root "$DW_WORK/inputs-dev-smoke-v1" --split dev \
  --names "$DW_WORK/names-dev-smoke.txt" --mode continue --run drivingworld-dev-smoke-v1 --samples 0-1 \
  --snapshot "$WCB_SNAPSHOT_DIR" --out "$DW_WORK/specs"
"$CPU_PY" -B make_drivingworld_specs.py ... --mode recon --run drivingworld-dev-smoke-v1 ...
```

## GPU smoke test

One process, both windows (batch 2 per window, then the window's reference), through a GPU wrapper:

```bash
"$DW_PY" -B run_drivingworld_bench.py --specs "$DW_WORK"/specs/drivingworld-dev-smoke-v1/*.json --bench "$WCB_BENCH" \
  --input-root "$DW_WORK/inputs-dev-smoke-v1" --output-root "$DW_WORK/outputs" --src "$DW_SRC" --weights "$DW_WEIGHTS" \
  --equivalence-check            # add --dry-run first: CPU spec/source validation only
"$CPU_PY" -B check_smoke.py --samples "$DW_WORK"/outputs/drivingworld-dev-smoke-v1/native/* --report NEW.json \
  --equivalence "$DW_WORK"/outputs/drivingworld-dev-smoke-v1/equivalence-check-*.json
```

For K = 16, `--samples 0-15`; the runner batches a window's 16 samples. Scoped DEV probes on windows outside the 60
generation windows (for example a moving subset) need `--diagnostic` in the input and spec builders (names-dev.txt; run
name `drivingworld-dev-diag-*`; refused for TEST). TEST continuation specs need
`protocol-drivingworld-frozen.txt`; TEST reference specs and TEST future inputs need
`protocol-drivingworld-stage2-frozen.txt`. Raw AV2 frames stay out of git and out of public storage.
