# wm-calib-bench

Model adapters and cluster tooling for a calibration benchmark of video world models on real driving footage
(bench-v1: Argoverse 2 front-camera windows, 2.0 s history, 6.0 s real future, 10 Hz). A model continues each history
K = 16 times; a frozen instrument reads target positions on a 512 x 288 evaluation canvas; calibration is tested by
the rank of the real outcome among the 16 generated outcomes. The real reference is the real clip passed through the
same model's tokenizer, so real and generated clips share the codec.

This repository holds code only. Windows, videos, weights, containers and results stay outside git (`.gitignore`).

## Models and their cluster guides

See `HANDOFF-cluster.md` for the operating rules (TEST only after `freezes/<model>-v1.json`). Each model is an independent batch-system job set; they can be submitted at the same time and queue in parallel.

| model | code | cluster guide (Chinese) | input package (copied by hand, never committed) |
|---|---|---|---|
| Orbis 2 | `models/orbis2/` | `cluster/RUNBOOK.md` | `orbis2-inputs-v1` |
| Epona (nuplan+china, horizon-centred band, 5 Hz, batch 16) | `models/epona/` | `cluster/epona/RUNBOOK.md` | `epona-inputs-v1` |
| LTX-2.5 distilled, stage 1 only at 512x288 (`distilled-s1-512`; bf16 resident; gated weights) | `models/ltx25/` | `cluster/ltx25/RUNBOOK.md` | `ltx25d-s1-inputs-v1` (`ltx25d-inputs-v1` superseded) |
| Vista, single-frame track at 320x576 (official code; outputs resized to 512x288; kit update pending) | `models/vista/` | `cluster/vista/RUNBOOK.md` | `vista-inputs-v3` (v1, v2 superseded) |

Each package directory holds the DEV and TEST tars, `SHA256SUMS` and a README; run `sha256sum -c SHA256SUMS` after the
copy. Only the DEV tar is needed for the smoke and DEV jobs; the TEST tar is used after that model's protocol is frozen.
Local helper scripts read machine paths from a git-ignored `models/<model>/local.env` (template `local.env.example`).

```
models/orbis2/
  run_orbis2_bench.py        adapter: spec JSONs -> vision.mp4 + sample_outputs.json (continue and recon modes)
  make_orbis2_specs.py       spec generator (seed table imported from the code snapshot's make_bench_nano_specs.py)
  build_input_package.py     builds the cluster input package from the bench-v1 windows (research server only)
  orbis2_pins.json           pinned code commit, weight revision, file hashes, released defaults
  hf_configs/                the three released config.yaml files of sud0301/orbis2 (MIT), sha256-pinned
  requirements-orbis2.in     the Orbis 2 environment.yml as requirements
  requirements-orbis2.lock.txt  hash-locked resolution (Python 3.10, torch 2.13.0+cu126, as of 2026-07-20)
  package_README.md          template of the input package's README
cluster/
  orbis2.def, Dockerfile     Apptainer / Docker recipes (same contents; no weights, no data)
  build_sif.sub              interactive build job for the .sif on a cluster build node
  stage_weights.sh           one-time download of the pinned checkpoints into /staging
  orbis2_chunk.sub           batch-system submit file: one job = one input chunk, 1 GPU (type chosen by the owner)
  run_chunk.sh               job wrapper (models dir, inputs check, adapter, smoke rerun, one output tarball)
  check_smoke.py             checks of the returned dev-smoke tarball (PASS/FAIL list, contact sheet)
  collect_outputs.py         checks the returned tarballs and unpacks them into run directories
  uv-requirements.txt        uv pinned by hash (installs the lock inside the image)
  RUNBOOK.md                 step-by-step guide for the cluster runs (Chinese)
```

Not in this repository (separate packages, never committed):
- the code snapshot `orbis2-code-snapshot-v1` (read-only copies of the bench-v1 scripts: windows, Nano specs and runner,
  frozen instrument, analyses; byte-identical to the originals, with `MANIFEST.sha256` and `verify_snapshot.sh`).
  Its tar unpacks to `snapshot/` (git-ignored). Only `make_orbis2_specs.py` uses it, to import `seed_of()`; point to it
  with `--snapshot DIR` or `WCB_SNAPSHOT_DIR`. Cluster jobs do not need it;
- the input package `orbis2-inputs-v1` (windows, specs, chunk tars) and the weights.

## Orbis 2 entry

- Code `lmb-freiburg/orbis2` at commit `f8a0d68` (2026-07-20); weights `sud0301/orbis2` at revision `7163cbf` (MIT, not
  gated). Used: `L1/` (detail predictor), `L2/` (standard abstract predictor), `tok/` (tokenizer), 20.6 GB.
- Released defaults, as `evaluate/rollout_demo_v2.py` runs them: `L1/config.yaml`, EMA weights, fp16 autocast, no
  steering (the framework's all-NaN placeholder), 12 rollout steps of 5 frames (6.0 s at 10 Hz). L1 sampler: the
  configured Heun+Euler schedule (2 Heun steps of 0.4, 1 Euler step of 0.2: 5 network evaluations per step; the CLI's
  `--l1_nfe 30` is ignored). L2 sampler: Euler, 20 evaluations per call. No guidance. The Orbis 2 paper states 30 and 15.
- History: the 21-frame clip t0 - 2.0 .. t0. The model reads frames 16-20 (L1, 10 Hz) and 0, 5, 10, 15, 20 (L2, 2 Hz).
- Output `vision.mp4`: 81 frames, 512 x 288, 10 fps, libx264rgb -crf 0 (lossless). Frames 0-20 are the per-frame
  tokenizer round trip of the history (the same function as the real reference, so they equal the reference's frames
  0-20), frames 21-80 the rollout. The real reference (`recon`) is the per-frame round trip of the 81 real frames.
- Geometry: the model's 512 x 288 frame is the evaluation canvas itself (the band rows [797, 1669) of the source),
  so the canvas mapping is the identity.
- Seeds: sample k of window w uses `seed_of(w, k)` of the Cosmos3-Nano spec generator (one table for every model).

## Local use (a GPU with about 30 GB free and about 30 GB of free host memory)

```bash
python3.10 -m venv tools && tools/bin/pip install --require-hashes -r cluster/uv-requirements.txt
python3.10 -m venv --without-pip venv
tools/bin/uv pip sync --python venv/bin/python --require-hashes --index-strategy unsafe-best-match models/orbis2/requirements-orbis2.lock.txt
git clone https://github.com/lmb-freiburg/orbis2.git ORBIS2_SRC && git -C ORBIS2_SRC checkout f8a0d6898689a20df10e9f71bcc0fa2abfdb7a8c
hf download sud0301/orbis2 --revision 7163cbfbe5c48195ad7536f4d2c2b804002992b9 --local-dir MODELS --include "L1/*" "L2/*" "tok/*"
venv/bin/python models/orbis2/run_orbis2_bench.py --specs INPUT_ROOT/specs --input-root INPUT_ROOT \
    --output-root OUT --models-dir MODELS --orbis2-src ORBIS2_SRC --verify-weights sha256 --resume
```

On the cluster, follow `cluster/RUNBOOK.md`.

## Rules

- TEST windows are generated only after the model's protocol is frozen; nothing here measures a clip.
- AV2 camera frames carry no face or plate blurring: inputs and outputs stay in private storage, never in git.
- The GPU type is not fixed: the submit file states the code's needs as overridable macros, and every sample records the
  GPU, driver, CUDA and torch versions it ran with.
