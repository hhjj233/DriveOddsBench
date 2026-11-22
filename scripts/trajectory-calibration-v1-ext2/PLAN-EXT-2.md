# Trajectory-level calibration v1, extension 2: VaViM and Wan2.1-VACE (coordinator p-inter-84, 2026-11-15)

Written before any trajectory statistic of these two models is computed. Their frozen TEST analyses (H1, H2, v1.2) are done.

- The frozen v1 plan (`../trajectory-calibration-v1/PLAN.md`, `frozen.txt`), addendum 1 and extension 1 (`../trajectory-calibration-v1-ext1/PLAN-EXT-1.md`) apply unchanged. The v1 scripts are imported, not copied or edited; their guards still run.
- `run_ext2.py` adds five entries to the scripts' model table: the three of extension 1 (same paths) and two new ones, each from its frozen stage-2 TEST outputs (primary outcome CSV, measure JSONs, band-VO records):
  - VaViM: `vavim/stage2-p84/`, prefix `vavim`, as for the extension-1 models.
  - Wan2.1-VACE: `wan21-vace13b/test-v1/instrument/` (outcomes and measure files, `vace832-test-*`) and `wan21-vace13b/ego-vo/clips/vace832-test-{k16,recon}`.
- The run covers all eleven models. Each model's random stream restarts from the plan's seed, so the nine earlier models must reproduce `../trajectory-calibration-v1-ext1/run-20261114-v1` exactly (BH q values aside, since BH now spans eleven models); the run stops if they do not.
- VaViM: only lead and lead_stop are scored (its frozen labels and the O10 and D2 rulings), so its pooled statistics cover two families. Its 10 Hz canvas repeats each 2 Hz model frame (nearest), so its trajectories change every 0.5 s; this is how the frozen instrument reads it, and it is reported as is.
- Everything stays exploratory, as in the plan.
