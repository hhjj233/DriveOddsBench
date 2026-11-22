# ReWorld-Video (Stage-1 Video DiT) kit

Kit for bench-v1, modelled on `models/drivelaw_video/` (same authors, same LTX-Video 2B 0.9.5
backbone, same 9 condition + 24 future frames per 33-frame call). It generated the ReWorld-Video results of the paper;
its protocol was frozen on DEV before any TEST clip was generated. Its settings (D1-D11) are recorded under `decisions` in
`reworld_pins.json` (`decisions_pending` is empty, and `bench.coordinator_approved` is set). GPU runs used H100 SXM GPUs.

## What is pinned

- Code: `xiaomi-research/reworld` at `7dfacf354ac13c41ab5f3f3059beb1b02b1b1d59` (Apache-2.0). `prepare_source.py`
  makes a sparse exact-commit checkout of `src/`, `configs/`, `scripts/` and the top-level build/launch files; the
  31 registered files are checked by git blob SHA1 and no other Python file may exist under `src/dualflow`.
  `DriveLaW-Act/` (Stages 2-3, NAVSIM planner) is not fetched or used.
- Weights (`download_reworld_weights.py`, owner licence approval record required before mkdir/network):
  `tz2026/ReWorld` @ `8170912332cc30a4a7aeaa923189f7018c9eb9ee`: `ReWorld-Video/dualflow_dit_best_fvd.safetensors`
  (3,847,926,960 bytes) plus config/README/licence; `Lightricks/LTX-Video-0.9.5` @
  `e58e28c39631af4d1468ee57a853764e11c1d37e`: `vae/` and `scheduler/` (what the official sampler loads) plus
  model_index/README/licence. 10 files, 6,341,835,365 bytes, every SHA256 pinned; the downloader writes `SHA256SUMS`
  and `download-verified.json`. Not downloaded: the Action DiT, the T5 text encoder/tokenizer (the vae_only DiT uses
  null text embeddings) and the LTX transformer (replaced by the ReWorld checkpoint).
- Environment: `requirements-reworld-video.lock.txt` (77 hash-locked packages, CPython 3.10, torch 2.6.0+cu124,
  diffusers 0.37.1 from PyPI, transformers 4.57.1, decord 0.6.0). `prepare_env.sh` installs it with `--no-deps`; the
  official source is not pip-installed: the runner puts `<source>/src` on `sys.path` exactly as
  `scripts/sample_dualflow_dit.py` does. bitsandbytes (pyproject dependency) is omitted: never imported on this path.

Paths are explicit CLI arguments; `local.env.example` is documentation only. Machine paths, approval records, data,
weights and outputs stay outside git.

## Registered model behaviour (see the protocol for every deviation)

Raw camera nearest timestamps (DriveLaW rule) -> full-width 2:1 crop of the TOP of the band, source rows
[797, 1572) -> `cv2.INTER_AREA` 1024x512 @ 8 Hz. The model reads the 9 real frames t0-1..t0 (2 latent frames);
no text, action, pose or navigation input. One official call = `sample_dualflow` conditioned branch: official decord
loader + VAE encode of the conditioning file, `run_vae_latent_sampling` (30 flow-match Euler steps, LTX-Video 0.9.5
scheduler, per-token timesteps, conditioning tokens re-anchored each step, self-guided sampling with the block-8 IG
head and scale 1.4, RoPE fps 8 read from the conditioning file), official `decode_video` (timestep 0, no tiling)
and the official uint8 rule `(clamp(x,0,1)*255).astype(uint8)`. Pass 2 conditions on pass 1's last 9 generated frames
(written as a lossless 8 fps file and read by the same official loader), seed + 1000003.

native65 = codec prefix[:8] + pass1[0:33] + pass2[9:33] (t0 = 16). Nearest 8 -> 10 Hz selection gives the 81-frame
canvas (<= 50 ms, exact 1/2/4/6 s, 12 repeated future steps), each frame `INTER_AREA` 2x to canvas rows [0, 256);
rows [256, 288) black (`canvas_spec_reworld.json`, valid rows [0, 256), instrument edge_bottom 252). References: the
real [-1, 3] s and [2, 6] s spans through the same official codec functions, seeded `seed_of(w,'recon')` and +1000003.

## CPU preparation

```bash
"$CPU_PY" -B "$KIT/check_cpu.py"
"$CPU_PY" -B "$KIT/prepare_source.py" --dest "$WORK/source"
"$CPU_PY" -B "$KIT/download_reworld_weights.py" --dest "$WORK/weights" --print
"$CPU_PY" -B "$KIT/download_reworld_weights.py" --dest "$WORK/weights" --license-approval "$WORK/owner-license-approval.json"
bash "$KIT/prepare_env.sh" "$UV" "$PYTHON310" "$WORK/source" "$WORK/venv"
# names-dev-smoke.txt: exactly the two registered DEV smoke windows.
"$CPU_PY" -B "$KIT/build_reworld_inputs.py" --bench "$BENCH" --split dev --names "$WORK/names-dev-smoke.txt" --out "$WORK/inputs"
"$CPU_PY" -B "$KIT/make_reworld_specs.py" --bench "$BENCH" --input-root "$WORK/inputs" --split dev --purpose smoke \
  --names "$WORK/names-dev-smoke.txt" --mode continue --run reworld-dev-smoke-v3 --samples 0-1 --snapshot "$SEED_SNAPSHOT" --out "$WORK/specs"
"$CPU_PY" -B "$KIT/make_reworld_specs.py" --bench "$BENCH" --input-root "$WORK/inputs" --split dev --purpose smoke \
  --names "$WORK/names-dev-smoke.txt" --mode recon --run reworld-dev-smoke-ref-v3 --snapshot "$SEED_SNAPSHOT" --out "$WORK/specs"
# D9 diagnostic: the 4 moving windows, built and specified against the authorised diagnostic bench view.
"$CPU_PY" -B "$KIT/build_reworld_inputs.py" --bench "$DIAG_VIEW" --split dev --names "$WORK/names-dev-tsdiag.txt" --out "$WORK/inputs-tsdiag"
"$CPU_PY" -B "$KIT/make_reworld_specs.py" --bench "$DIAG_VIEW" --input-root "$WORK/inputs-tsdiag" --split dev --purpose timescale-diagnostic \
  --names "$WORK/names-dev-tsdiag.txt" --mode continue --run reworld-dev-tsdiag-v3 --samples 0-1 --snapshot "$SEED_SNAPSHOT" --out "$WORK/specs"
CUDA_VISIBLE_DEVICES= "$WORK/venv/bin/python" -B "$KIT/check_strict_load.py" --src "$WORK/source" --weights "$WORK/weights" --report "$WORK/strict.json"
CUDA_VISIBLE_DEVICES= "$WORK/venv/bin/python" -B "$KIT/check_cpu_pipeline.py" --src "$WORK/source" --weights "$WORK/weights" \
  --bench "$BENCH" --input-root "$WORK/inputs" --window 2eeaccb0-t2.5 --snapshot "$SEED_SNAPSHOT" --out "$WORK/cpu-pipeline"
"$CPU_PY" -B "$KIT/run_reworld_bench.py" --specs "${SPECS[@]}" --bench "$BENCH" --input-root "$WORK/inputs" \
  --output-root "$WORK/outputs" --src "$WORK/source" --weights "$WORK/weights" --dry-run
```

`check_cpu_pipeline.py` runs the whole runner on the CPU at a check-only 256x128 size with 2 steps (one continuation
and one reference): a code-path check, never a sample. The input builder and spec builder need only NumPy, OpenCV and
ffmpeg; `--samples 0-15` is the shared K=16 seed table. TEST inputs are history-only unless the stage-2 record exists.

## Smoke sets

- `smoke`: the 2 registered smoke windows x seeds 0/1 (`--purpose smoke --run reworld-dev-smoke-...`) plus one
  reference each (`--mode recon --purpose smoke`), built against the registered bench (window list
  `names-dev-gen.txt`, SHA256 pinned in `bench.registered_names_sha256`).
- `timescale-diagnostic` (D9; smoke only, never part of the main DEV analysis): the DriveLaW predeclared moving subset
  (selection SHA256 `314843667f05...`) x seeds 0/1, continuations only, `--purpose timescale-diagnostic --run
  reworld-dev-tsdiag-...`, built and run with `--bench` = the authorised diagnostic bench view (its
  `names-dev-gen.txt` SHA256 is pinned). Spec builder and runner refuse other windows, seeds, run prefixes, modes or
  window lists for each purpose; main runs cannot use the view or the special prefixes.
- `check_smoke.py --set smoke|tsdiag` checks the 6 / 8 returned samples.

## GPU runs (H100 SXM)

`run_reworld_bench.py --specs ... --bench --input-root --output-root --src --weights` loads the model once and runs
many specs in one process (the worker interface of the DriveLaW kit). Every path argument must be absolute (the
environment is copied to the same absolute path on the GPU machine). The runner reads only the specs, inputs, source and weights and writes
only under `--output-root` (sample directories, their `.partial` files, `failure.json`); it never uses the working
directory. Library caches (TMPDIR, HF_HOME, CUDA_CACHE_PATH, ...) are set by the launching worker; HF_HUB_OFFLINE
defaults to 1. It requires `CUDA_VISIBLE_DEVICES`, the weight download proof, the exact lock and
`coordinator_approved`; it never overwrites a sample directory, writes `failure.json` and stops on the first error. Per sample: `prefix-native8.mp4` (17), `pass1-native8.mp4` (33),
`pass2-condition8.mp4` (9), `pass2-native8.mp4` (33), `native8.mp4` (65 @ 8 Hz, 1024x512), `vision.mp4`
(81 @ 10 Hz, 512x288, libx264rgb crf 0) and `sample_outputs.json` (purpose, seeds plus `seed_rules` with the rule and
value of every seeded call (D4), settings, official config, deviations, strict-load evidence, per-call wall times,
peak GPU memory, GPU name/UUID, interpreter and env prefix, software, attention processors).
References add `ref-seg1-input8.mp4` / `ref-seg2-input8.mp4`.
