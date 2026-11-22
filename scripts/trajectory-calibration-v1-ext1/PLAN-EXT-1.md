# Trajectory-level calibration v1, extension 1: three later models (coordinator p-inter-84, 2026-11-14)

Written before any trajectory statistic of these three models is computed. Their frozen TEST analyses (H1, H2, v1.2) are done.

- The frozen v1 plan (`../trajectory-calibration-v1/PLAN.md`, `frozen.txt`) and addendum 1 (`PLAN-ADDENDUM-1.md`, `frozen-addendum-1.txt`)
  apply unchanged. Their scripts are imported, not copied or edited; the guards of both still run.
- `run_ext1.py` only adds three entries to the scripts' model table: DriveLaW-Video, DrivingWorld and ReWorld-Video. The paths are each
  model's frozen stage-2 TEST outputs: the window-rule outcome CSV (primary), the measure JSONs, and the band-VO records written by the
  TEST instrument.
- The run covers all nine models. Each model's random stream restarts from the plan's seed, so the six earlier models must reproduce
  run-20261110-v1 exactly; the run stops if they do not. Only the BH q values change, because BH now spans nine models.
- DrivingWorld: the frozen D2 ruling makes every ego_inter case unscorable, so that family has no case. The frozen code skips an empty
  family, so DrivingWorld's pooled statistics cover four families. Its addendum-1 skill for ego_inter is undefined (no case) and is
  reported as such.
- Everything stays exploratory, as in the plan.
