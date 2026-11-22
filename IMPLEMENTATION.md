# DriveOddsBench: implementation

This document describes how every step of the benchmark is implemented, in the order in which it runs. Section numbers of the paper are given as §, and supplementary sections as App. The scripts are in `scripts/` unless a path is given. The frozen files, and the differences of their copies in this repository, are listed in [`FROZEN_MANIFEST.md`](FROZEN_MANIFEST.md).

Contents: [1 Layout](#1-layout-and-conventions) · [2 Histories](#2-histories-and-scenario-families) · [3 Canvas and timing](#3-measurement-canvas-and-timing) · [4 Generation](#4-generation) · [5 Instrument](#5-instrument) · [6 Validation](#6-validation-and-reliability-labels) · [7 Ranks and tests](#7-ranks-and-confirmatory-tests) · [8 Further analyses](#8-further-analyses) · [9 Running](#9-running-the-pipeline)

## 1. Layout and conventions

All paths are relative to one working directory, `$DOB_ROOT` (default: the current directory):

```text
$DOB_ROOT/
├── datasets/av2_sensor/                 # Argoverse 2 Sensor dataset (train/val/test logs)
├── cache/torch/hub/checkpoints/         # fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth, raft_large_C_T_SKHT_V2-ff5fadd5.pth
├── checkpoints/                         # world-model weights, one folder per model (see each kit's *_pins.json)
└── outputs/wm-calibration/bench-v1/     # windows-dev/, windows-test/, detections/, runs/, instrument/, analysis outputs
```

- A *history* (in the code: a *window*) is named `<first 8 hex digits of the SHA-256 of the AV2 log id>-t<anchor offset in s>`, e.g. `9947fb6e-t6.5`; the hash prefix is the `log_hash` column of `data/*_families.csv`.
- *Ticks* are the times t0 + 0.1 k s, k = −20 … 60. Tick 0 is t0, the last history frame. Clips are stored as 81 frames at 10 Hz; frames 0–20 are the history and frames 21–80 the 6 s future.
- *Samples* are numbered 0–15; the real future after the model's autoencoder is the sample `real` (run `recon`), and the raw real video the sample `raw`.
- Everything random has a fixed seed: the tie-breaking of the confirmatory ranks uses seed 2026 with the histories in name order, and the other streams (permutations, bootstrap, Monte Carlo) are separate generators derived from the same seed (`analyze_bench_v1.py`, `STREAM`).

## 2. Histories and scenario families

(§3.2; `make_benchmark_windows.py`, `count_scenario_families.py`, `make_wm_calibration_windows.py`; lists in `data/`.)

- **Source.** The ring front-centre camera of the AV2 Sensor dataset (1550 × 2048 portrait images at 20 Hz), its LiDAR annotations, ego poses, calibration and maps, through the official `av2` devkit.
- **Anchors.** Annotation times on a 0.5 s grid, from 2.5 s after the start of a log to 6.1 s before its end.
- **Past information only.** Every family rule reads the annotations, ego poses, map and anchor image at or before t0. On test logs the annotation table is cut at the last candidate anchor right after loading, every lookup is restricted to times ≤ t0 (asserted), and the code that would read an outcome refuses to run.
- **Families** (one target each; velocities from annotations over the last second; heading difference relative to the ego):

| Family | Target at t0 | Rule | Primary quantity |
|---|---|---|---|
| Lead | moving lead | nearest vehicle with box centre ahead (x > 0) and \|y\| < 1.8 m; rear face 5–40 m ahead; heading difference < 30°; speed > 1 m/s | gap x at 4 s |
| Brake/stop | braking or stopped lead | the lead, rear face 5–40 m; speed < 0.5 m/s, or mean acceleration over the last 0.5 s < −1 m/s² with heading difference < 30° | gap x at 4 s |
| Cut-in | adjacent vehicle, same direction | nearest vehicle with heading difference < 30°, speed > 1 m/s, 1.8 ≤ \|y\| ≤ 5.5 m, −2 ≤ x ≤ 40 m | \|lateral offset\| at 2 s |
| Oncoming | vehicle in the opposite direction | nearest vehicle with heading difference > 150°, speed > 2 m/s, 5 ≤ x ≤ 60 m, \|y\| < 15 m (rule F3 of `count_scenario_families.py`) | position x at 1 s |
| Ego | the ego vehicle at a junction | ego approaching a junction at > 2 m/s (rule F8), or ego < 0.5 m/s with an intersection lane on its forward line within 15 m | ego advance 0–4 s |

- **Visibility of the target at t0.** The projected box lies inside the measurement band, is at least 97 px tall in the source image (20 px in the 320 × 192 frame) and holds at least five LiDAR points, and a COCO detection of the right class (score ≥ 0.4) matches it with IoU ≥ 0.3 on the t0 canvas frame.
- **Splits.** DEV: 60 histories from 60 logs; TEST: 200 histories from 200 other logs; at most one history per log, so no test log enters instrument development. `data/{dev,test}_histories.txt` list them, and `data/{dev,test}_families.csv` give each history's scored families and their targets. Four further families (pedestrian, start-up, crossing, rider) are only descriptive and are not scored.

## 3. Measurement canvas and timing

(App. B.1; `make_benchmark_windows.py`.)

- **Measurement band.** Rows [797, 1669) of the 1550 × 2048 image, full width (16:9), rendered to the **512 × 288 canvas** on which every clip of every model is measured. The rows are cropped and resized with `cv2.resize` (`INTER_AREA`); boxes and intrinsics map to the canvas with the pixel-centre convention x′ = s x + 0.5 s − 0.5, s = 512/1550.
- **Ticks.** Each tick takes the nearest 20 Hz camera frame (ties to the earlier frame; |offset| ≤ 50 ms).
- **Model frames.** Each model generates at its own size and rate ([4](#4-generation)). Its kit maps the model's frames onto the canvas (`generation/<model>/canvas_spec_*.json` where present: the crop, scale and valid rows) and its frames onto the 10 Hz ticks (nearest frame).

## 4. Generation

(§3.1, §5.1, App. B.1 and Table S1; `generation/<model>/`, `scripts/make_bench_nano_specs.py`, `scripts/make_bench_edge_specs.py`, `scripts/run_cosmos3_calibration.py`, `scripts/edge/run_cosmos3_calibration.py`.)

- **Inputs.** Every model receives the 2 s of front-camera video before t0, or the part of it that it accepts. Epona and DrivingWorld also receive the past ego poses. No model receives any information from after t0, and no model receives future ego actions.
- **Settings.** Each model runs at its official native resolution and sampler settings, with distilled variants only where the developers release them:

| Model | Frame | Rate | History read | Other input | Sampler | Guidance | Calls |
|---|---|---|---|---|---|---|---|
| Orbis 2 | 512×288 | 10 Hz | 9 frames (5 at 10 Hz, 5 at 2 Hz) | – | flow matching, NFE 5 and 20 | none | 12 steps |
| Cosmos3-Nano | 320×192 | 10 Hz | 21 frames | prompt | UniPC, 35 steps | 6.0 | 1 |
| Cosmos3-Edge | 320×192 | 10 Hz | 21 frames | prompt | UniPC, 35 steps | 6.0 | 1 |
| LTX-2.5 (distilled) | 1024×576 | 20 Hz | 41 frames | prompt | Euler ancestral, 8 + 3 steps | none | 1 |
| Wan2.1-VACE (1.3B) | 832×480 | 16 Hz | 33 frames | prompt | UniPC, 50 steps | 5.0 | 2 |
| Epona | 1024×512 | 5 Hz | 10 frames (1.8 s) | past ego poses | Euler, 100 steps | none | 30 steps |
| DrivingWorld | 512×256 | 10 Hz | 15 frames | past ego poses | tokens, top-k 30, T = 1 | none | 60 steps |
| ReWorld-Video | 1024×512 | 8 Hz | 9 frames (1.0 s) | – | flow matching, 30 steps | internal 1.4 | 2 |
| DriveLaW-Video | 1280×704 | 8 Hz | 9 frames (1.0 s) | prompt | 30 steps | 3.0 | 2 |
| VaViM | 512×288 | 2 Hz | 5 frames | – | tokens, full vocabulary, T = 1 | none | 12 steps |
| Vista | 1024×576 | 10 Hz | last frame | – | Euler EDM, 50 steps | 1.0–2.5 | 3 |

  Frame: the size at which the model generates. Guidance: classifier-free guidance scale; Vista's guider rises from 1.0 at the ends of each window to 2.5 at its middle, and ReWorld-Video uses its own internal guidance. Calls: model calls per 6 s future, or autoregressive steps.
- **Two deviations from the released defaults.** VaViM's released default decodes greedily, which makes all 16 samples identical, so we sample its full temperature-1 distribution. DrivingWorld's released inference reads the next recorded ego pose, so every future step is given the last history pose step instead, as in its released long-term demo (`pose_mode` `released_long_term_last_step` in `drivingworld_pins.json`).
- **Prompt.** Models that accept text receive one fixed, neutral prompt describing a forward-facing dashcam, with no words about motion or events, the same for every history. DriveLaW-Video instead receives its released prompt template, filled per history with the ego displacement and yaw over the past 2 s and the current speed, from the past ego poses (`drivelaw_common.caption`).
- **Seeds.** Sample k of history w uses `seed_of(w, k)`, the first four bytes of `sha256("bench-v1|w|k")` (31 bits), in every model. No sample is discarded or drawn again.
- **Real reference.** The real future of each history is passed through the model's own autoencoder or tokenizer (for Cosmos3, all latents conditioned and two sampling steps with seed 0), so that the instrument reads real and generated clips with the same decoder artefacts. The raw real video is kept as a sensitivity check.
- **A kit** (`generation/<model>/`, nine models) holds the input builder (`build_*_inputs.py` or `build_input_package.py`), the spec generator (`make_*_specs.py`), the runner (`run_*_bench.py`), the canvas map where the kit has one (`canvas_spec_*.json`), the pinned code and weights with their hashes (`*_pins.json`), the environment lock files and a `package_README.md`. Its smoke and CPU checks (`check_*.py`), where present, compare each step with the released code. The two Cosmos3 models run from the sample specs of `scripts/make_bench_nano_specs.py` and `scripts/make_bench_edge_specs.py`, with `scripts/run_cosmos3_calibration.py` and `scripts/edge/run_cosmos3_calibration.py`.

## 5. Instrument

(§3.3, App. B.1; `detect_bench_v2.py`, `measure_bench_v2.py`, `measure_bench_ego_vo.py`, `measure_ego_motion_raft.py`, `derive_bench_quantities.py`, `canvas_instrument/`, `instrument_config/config-v2.json`.)

One fixed function of the video reads every clip of a history, real or generated, with the same code, parameters and failure rules.

**Detection.** torchvision's COCO Faster R-CNN v2 (`fasterrcnn_resnet50_fpn_v2`, default internal resize to 1333 × 750) runs on all 81 canvas frames of a clip. It keeps person, bicycle, car, motorcycle, bus and truck with score ≥ 0.1 (the trackers apply their own thresholds). Detections are cached per clip as JSON in canvas pixel-edge coordinates.

**Geometry.** Per history, from its `window.json`: the canvas intrinsics `K_canvas` (the source intrinsics through the band affine) and the AV2 camera extrinsics. The road is modelled as flat, 0.33 m below the AV2 ego origin. The reading point is the bottom centre of the tracked box; its ray is intersected with the road, which gives x (forward) and y (left) in the ego frame at the clip's time; |y| is the lateral offset.

**Tracking.** The target of each family is fixed at t0 by its box at t0 and followed by three trackers:

- **v3:** IoU association with flat-ground kinematic gates on range and lateral motion per frame;
- **v5:** a constant-velocity predicted box with a physical size gate and range and lateral gates;
- **v8:** the bottom centre and log height in pixels, a constant-velocity prediction and pixel gates.

Following stops when the box touches the side or bottom edge of the canvas (the exit is recorded) or after too many missed frames (the target is lost).

**Outcomes.** The reading at horizon T is the tracker's reading at T if the target is followed then; otherwise the last reading before T is carried forward (a sensitivity check treats it as missing instead). The ensemble value is the median of the three trackers (the majority for binary outcomes).

**Other canvases.** `canvas_instrument/` runs the same frozen steps on a model's own canvas geometry and frame rate (`ci_core.py`, `measure_canvas.py`, `detect_canvas.py`, `derive_canvas.py`). For models that rewrite the first frames, it also checks that a generated clip still holds the same target at t0: at least two of the three trackers must match a detection with IoU ≥ 0.5 in both the clip and its real reference. Otherwise the whole (history, family) case is dropped (`measure_canvas.py export --t0-rule window`); `--t0-rule clip` drops only that clip (the sensitivity check of App. C).

**Ego motion.** Visual odometry on the same band (`measure_ego_motion_raft.py` VO v2, ported to the band in `measure_bench_ego_vo.py` and `canvas_instrument/ego_vo_band.py`):

1. RAFT-large optical flow on road pixels (×2 upsampling, 20 updates), with detections masked (score ≥ 0.3, padded 4 px) and a forward–backward check;
2. a robust 6-DoF ground-plane fit, with one warped refinement per 0.4 s group;
3. planar odometry, which gives the forward displacement D(T) in the t0 ego frame and the heading change.

**Secondary quantities.** `derive_bench_quantities.py` derives them from the same tracks: time headway, post-encroachment time (minimum and at t0), closing speed, lateral speed 1–2 s, the time an oncoming vehicle leaves the view, ego advance 0–2 s and 0–6 s, ego speed at 2 and 4 s, and peak deceleration 0–4 s.

## 6. Validation and reliability labels

(§3.3, App. B.1 and Figs. S2–S3; `validate_bench_v2.py`, `rules_bench_v2.py`, `canvas_instrument/validate_canvas.py`, `canvas_instrument/rules_canvas.py`, `instrument_config/scorability-rules.json`.)

- **Ground truth on DEV.** The instrument's readings on the raw real video are compared with the LiDAR annotations (target families) and the AV2 ego poses (ego advance).
- **Reliability labels, per model.** Before any test clip of a model is measured, DEV fixes which of its quantities are *tested*. The right target must be read in at least 90 % of histories on both the raw and the model-reconstructed real video, and the two readings must agree (Spearman ≥ 0.8; binary outcomes: agreement ≥ 0.85, also with the annotation truth); the checks on the reconstructed video are skipped below 8 DEV window-families (`rules_bench_v2.py`, `canvas_instrument/rules_canvas.py`).
- **Odometry gate (D2).** The ego family is scored for a model only if the odometry passes at 4 s on the raw DEV video and on the model's reference clips: the median of |D(4 s) − truth| / max(0.15 × truth, 1.5 m) is at most 1, the median heading-change error at most 3°, and the Spearman correlation with the AV2 poses at least 0.90 (`measure_bench_ego_vo.py gate`, `canvas_instrument/ego_vo_band.py gate`).
- **Secondary quantities** (`derive_bench_quantities.py`, `verdict`) are labelled insufficient (fewer than 15 DEV readings), not informative (at least 80 % of the raw readings censored, or a binary event with a ground-truth prevalence outside 5–95 %), tested (Spearman ≥ 0.80 with the ground truth and between the raw and reconstructed readings; binary: agreement ≥ 0.85 and κ ≥ 0.40 with the ground truth and agreement ≥ 0.85 between raw and reconstructed; the raw–reconstructed checks need at least 8 DEV pairs) or not reliable.
- **Families not scored.** A family that fails on a model's reconstruction is not scored for that model: DrivingWorld's ego family and VaViM's cut-in, oncoming and ego families.

## 7. Ranks and confirmatory tests

(§4.1–4.2, §4.6, App. A.1–A.4; `analyze_bench_v1.py`, `analyze_wm_calibration.py`, `bench_holm_two_sided.py`.)

**Retention.** In the test, every retained (history, family) case of a primary quantity had the real reading and all 16 generated readings valid. Retention is symmetric in the 17 readings, so the uniform reference holds.

**Rank.** With L generated readings strictly below the real one and E equal to it, the rank is R = L + J, where J is uniform on {0, …, E} (seed 2026). PIT = (R + U) / (K + 1), with an independent uniform U.

**H2 (coverage).** The statistic is T, the share of a history's scored primary quantities whose real reading lies outside all samples (R ∈ {0, K}), averaged over histories. Under alignment it is 2/(K + 1) = 11.8 %. The test is one-sided against a permutation null: each permutation swaps the real video with one of the history's videos for all its quantities at once, which keeps their dependence (B = 100,000, plus-one p-value).

**H1 (skew).** The statistic is the mean PIT − ½ of the lead gap at 4 s (exact null variance 1/(12n)). It is tested two-sided with the slope component of the rank chi-square, z = Σᵢ (Rᵢ − K/2) / √(n K (K + 2)/12) (`slope_z`).

**Decision.** Holm's procedure across H2 and H1 for each model. H1 is confirmed only if its Holm-adjusted p ≤ 0.05 and the 95 % bootstrap interval of mean PIT − ½ (10,000 resamples of histories) excludes zero on the same side (`bench_holm_two_sided.py`, from the `results.json` of `analyze_bench_v1.py`).

**Secondary tests** (App. B.2; two-sided, Benjamini–Hochberg at q = 0.10 within each statistic across families; `analyze_bench_v12.py` for the secondary quantities), per quantity:

- mean PIT − ½;
- the convexity (spread) component of the rank chi-square;
- the shares of cases below and above all samples, with exact binomial tests.

A Monte Carlo chi-square test over the 17 rank bins (100,000 draws) and the slope (shift) component are reported as well.

## 8. Further analyses

**Whole trajectories** (§4.3, App. D; exploratory; `scripts/trajectory-calibration-v1*/`, frozen with their plans before any trajectory statistic was computed).

- Each trajectory is sampled on a 0.4 s grid up to its validated horizon (the oncoming position on a 0.2 s grid up to 1 s).
- The pre-rank is band depth: at every step the 17 trajectories are ranked (random ties), and a trajectory of rank r adds r(K + 2 − r).
- Under alignment the real trajectory is the least central of the 17 with probability 1/17 = 5.9 %.

**Informativeness** (§4.4, App. A.6 and B.2).

- Every model is scored with the fair CRPS, (1/K) Σ|X_k − Y| − (1/(2K(K−1))) Σ_{k≠l}|X_k − X_l|; whole trajectories use the fair energy score.
- Skill against climatology is one minus the ratio of the summed scores. Climatology draws its samples from the DEV raw readings of the same family, from disjoint logs.

**Spread ratio** (§4.2, App. A.7 and Table S2; `scripts/spread_ratio.py`; descriptive, added after all results).

- Per family it is ρ = Σ Ŝ_i / Σ Ê_i, where Ŝ is the mean distance between two samples and Ê the mean distance from a sample to the real reading (the two terms of the fair CRPS). It is then averaged over the scored families.
- Under alignment ρ = 1 for every K.
- The K′ table subsamples K′ ∈ {2, 4, 8, 16} samples, with 200 draws per case.

**Property scores** (§4.5, Fig. 3, App. E and Table S4; `scripts/make_overview_figures.py`, with the continuation breaks of `scripts/continuation_break.py`; VaViM: `scripts/continuation_break_2hz.py`). These four descriptive scores in [0, 1] are defined after the frozen analyses, and the summary is their mean:

- validity: the share of scorable cases × (1 − the share of clips that break at t0);
- coverage, from H2 and the least-central share;
- centering: 1 − 2 × the mean |PIT − ½|;
- informativeness: the mean of the CRPS and energy-score skills, clipped to [0, 1].

**Robustness checks** (§5.4, App. C):

- a positive control that uses one generated sample as the observation (K = 15);
- split-sample replication on the two halves of the 16 samples;
- the raw real video instead of the reconstructed one;
- lost targets treated as missing;
- the outside rate split by GPU type (its script is not in this repository).

**Scalar validation** (§5.4, App. F and Table S5; `scripts/validate_rank_test.py`). The rank test is checked on simulated numbers: N ∈ {64, 256, 1024}, K = 16, Gaussian models (correct, shifted, too narrow, too wide), Bernoulli(0.3) ties, 2,000 repetitions, 50,000 null histograms, seed 20261110. The script writes `data/statistical_validation.json`, `data/statistical_validation.csv` and `tables/statistical_validation.tex`.

## 9. Running the pipeline

`B` below is `$DOB_ROOT/outputs/wm-calibration/bench-v1`. The scripts process DEV by default. Test histories are processed only with `--allow-test` and once the model's frozen record exists, and the analysis also requires that every test clip has been measured.

```bash
# 1. Histories: selection and draw of the DEV and TEST lists (after --phase extend they equal the lists in data/);
#    --phase render, verify and sheets then render the clips, check the t0 boxes and draw contact sheets
python scripts/make_benchmark_windows.py --phase select --workers 12
python scripts/make_benchmark_windows.py --phase draw

# 2. Generation: inputs, specs and runs of one model (see generation/<model>/package_README.md)
python generation/orbis2/build_input_package.py ...
python generation/orbis2/make_orbis2_specs.py ...
python generation/orbis2/run_orbis2_bench.py ...

# 3. Detections on the canvas
python scripts/detect_bench_v2.py --bench $B --names-file NAMES --canvas-raw --runs RUN ...
python scripts/canvas_instrument/detect_canvas.py ...          # models with their own canvas map

# 4. Readings and ego motion
python scripts/measure_bench_v2.py measure --bench $B --names-file NAMES --config instrument_config/config-v2.json --sources canvas-raw RUN --out MEASURE
python scripts/measure_bench_v2.py export --bench $B --measure MEASURE --run RUN --real-run RECON --out outcomes.csv
python scripts/canvas_instrument/measure_canvas.py measure|export ...
python scripts/canvas_instrument/ego_vo_band.py measure --bench $B --names-file NAMES --model M --source RUN --from KIND --det-root DET --out VO
python scripts/derive_bench_quantities.py derive|export ...

# 5. Reliability labels (DEV), then the frozen analysis (TEST)
python scripts/rules_bench_v2.py --validation VALIDATION.json --gate GATE.json --recon-run dev-recon --out RULES.json
python scripts/analyze_bench_v1.py --runs outcomes-test-k0-10.csv outcomes-test-k11-15.csv outcomes-test-raw.csv \
       --windows-dir $B/windows-test --climatology CLIM.csv --rules instrument_config/scorability-rules.json --label TEST --allow-test --out ANALYSIS
python scripts/bench_holm_two_sided.py --results ANALYSIS/results.json --model M --out ANALYSIS/confirmatory-two-sided.json --md ANALYSIS/confirmatory-two-sided.md --expect-k 16
python scripts/analyze_bench_v12.py --runs ... --windows-dir $B/windows-test --label TEST --K 16 --allow-test --out ANALYSIS_V12

# 6. Whole trajectories (frozen plans in each folder; the scripts refuse to run if a listed file changed). run_ext1.py and
#    run_ext2.py also check the earlier models against the previous run, read from the run folders below, so use these paths
T1=scripts/trajectory-calibration-v1; T2=scripts/trajectory-calibration-v1-ext1
python $T1/analyze_trajectories.py --bench $B --out $T1/run-20261110-v1
python $T1/analyze_trajectory_climatology.py --bench $B --out $T1/run-20261110-v1
python $T2/run_ext1.py --bench $B --out $T2/run-20261114-v1
python scripts/trajectory-calibration-v1-ext2/run_ext2.py --bench $B --out $B/cross-model/trajectory-calibration-v1-ext2/run-20261115-v1    # all eleven models; read by step 7

# 7. Continuation breaks, property scores, figure data and the spread ratio (the last three write to $DOB_FIGURE_OUT, default figure_out/)
python scripts/continuation_break.py --help
python scripts/make_overview_figures.py
python scripts/export_supply.py
python scripts/spread_ratio.py

# 8. Scalar validation of the rank test
python scripts/validate_rank_test.py
```

Each script's docstring states its inputs, outputs and checks; most scripts also print their usage with `--help`.
