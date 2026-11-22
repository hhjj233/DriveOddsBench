# Code map

Where each part of the paper lives. § = section of the main paper, App. = appendix of the paper.

| Paper | What | Code |
|---|---|---|
| §3.2, Table 2 | Histories, anchors, past-only scenario families, DEV/TEST splits | `scripts/make_benchmark_windows.py`, `scripts/count_scenario_families.py`, `scripts/make_wm_calibration_windows.py`, `data/` |
| §3.1, §5.1, App. Table S1 | Generation of 16 futures per history; settings, seeds, real reference through the autoencoder | `generation/<model>/`, `scripts/make_bench_nano_specs.py`, `scripts/make_bench_edge_specs.py`, `scripts/run_cosmos3_calibration.py`, `scripts/edge/run_cosmos3_calibration.py` |
| §3.3, App. B.1 | Measurement canvas, detections, trackers, ground-contact geometry, outcomes | `scripts/make_benchmark_windows.py` (rendering), `scripts/detect_bench_v2.py`, `scripts/measure_bench_v2.py`, `scripts/canvas_instrument/`, `instrument_config/config-v2.json` |
| §3.3 | Ego motion by band visual odometry | `scripts/measure_ego_motion_raft.py`, `scripts/measure_bench_ego_vo.py`, `scripts/canvas_instrument/ego_vo_band.py` |
| §3.3, Fig. 7, App. B.1–B.2 | Secondary quantities (headway, PET, speeds, exit time, deceleration), their tests and the skew profile | `scripts/derive_bench_quantities.py`, `scripts/canvas_instrument/derive_canvas.py`, `scripts/analyze_bench_v12.py`, `scripts/make_overview_figures.py` |
| §3.3, Table 2 (Check), App. Figs. S2–S3 | Validation on DEV and reliability labels | `scripts/validate_bench_v2.py`, `scripts/rules_bench_v2.py`, `scripts/canvas_instrument/validate_canvas.py`, `scripts/canvas_instrument/rules_canvas.py`, `instrument_config/scorability-rules.json` |
| §4.1, App. A.2–A.3 | Ranks with random tie-breaking, symmetric retention | `scripts/analyze_wm_calibration.py` (`ranks_of`), `scripts/analyze_bench_v1.py` |
| §4.2, §4.6, Table 3, Fig. 4, App. A.4 | H2 with the permutation null, H1, Holm and the bootstrap condition | `scripts/analyze_bench_v1.py`, `scripts/bench_holm_two_sided.py` |
| §4.2, App. A.7, Table S2 | Spread ratio and its K′ table (case pool from `scripts/export_supply.py`) | `scripts/spread_ratio.py` |
| §4.3, App. D | Whole trajectories (band depth, least-central share, fair energy score, climatology), frozen with their plans | `scripts/trajectory-calibration-v1/` (`analyze_trajectories.py`, `analyze_trajectory_climatology.py`), `scripts/trajectory-calibration-v1-ext1/`, `scripts/trajectory-calibration-v1-ext2/` |
| §4.4, App. A.6 | Fair CRPS and energy score, skill against climatology | `scripts/analyze_bench_v1.py`, `scripts/analyze_bench_v12.py`, `scripts/trajectory-calibration-v1/analyze_trajectory_climatology.py` |
| Table 3 (Jump), §5.2 | Continuation breaks at t0 | `scripts/continuation_break.py`, `scripts/continuation_break_2hz.py` (VaViM) |
| §4.5, Fig. 3, App. E | Property scores | `scripts/make_overview_figures.py` (validity uses `scripts/continuation_break.py`) |
| §5.4, App. C | Positive control, split-sample replication, raw video, lost targets as missing | `scripts/analyze_bench_v1.py`, `scripts/analyze_bench_v12.py` |
| §5.4, App. F, Table S5 | Scalar validation of the rank test | `scripts/validate_rank_test.py` |
| Figures and tables | Export of the numbers behind the figures | `scripts/export_supply.py` |

Self-checks of the frozen analyses: `scripts/check_analyze_bench_v1.py`, `scripts/check_analyze_bench_v12.py` (independent helpers in `scripts/validate_calib_common.py`). Checks of the canvas instrument against the frozen one: `scripts/canvas_instrument/regress_canvas.py`; its DEV validation tables: `scripts/canvas_instrument/tables_canvas.py`.

Not yet released: the per-reading records, the generated futures and the frozen protocols, except the analysis plans of the whole-trajectory analysis, which are in `scripts/trajectory-calibration-v1*/` (see the Release TODO in the README). Not in this repository: the scripts that draw the figures and tables, among them the exact K′ curves of Fig. S6, Kendall's τ and the GPU-type split of App. C, and the selection of the histories shown in Figs. 6, S1, S4 and S5; the Cosmos3 sampler defaults read by `make_bench_nano_specs.py` and `make_bench_edge_specs.py` (`study-v1/hist5-specs/{abl-h5,recon-hist5}/defaults.json`); and the DEV ego-speed table read by `export_supply.py` (`cross-model/ego-motion-prior-v1/run-20261114-new3-v1/primary.csv`).
