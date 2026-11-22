# Trajectory-level calibration v1, addendum 1: informativeness against climatology (coordinator, 2026-11-10)

Written before any trajectory score against climatology is computed. PLAN.md and analyze_trajectories.py are unchanged;
this addendum adds one statistic in a separate script that imports the frozen functions read-only.

## Reason
The paper measures informativeness as the improvement of a proper score over the climatological forecast (the
forecast that ignores the history; it passes the pooled rank tests but carries no information about the scene).
The frozen scalar analyses already report CRPS skill against climatology. Statistic 3 of PLAN.md used constant
velocity for trajectories; this addendum adds the climatological reference for trajectories, with the same
definition as the scalar one.

## Definition
- Climatology ensemble for family f: the DEV raw curves of f, read from instrument/measure-dev/measure-nano-raw.json
  (targets) and instrument/ego-vo/clips/nano-raw (ego) with the same reading rule and grid as PLAN.md, for the DEV
  windows whose primary outcome is scorable in instrument/outcomes/outcomes-dev-raw-climatology.csv (the file the
  frozen scalar analyses use). A DEV curve enters only if it has a value at every grid time. Consistency check: at
  the primary horizon every DEV curve equals the CSV value (abs diff <= 1e-3), else the run stops.
- Per TEST case (the cases of PLAN.md): fair energy score of the 16 generated curves and of the m climatology curves
  against the real curve. Skill per family = 1 - sum ES_model / sum ES_clim, with a 95 % window-bootstrap interval
  (B = 2000, seed 2026). All five scored families (oncoming included: no history fit is needed).
- Reported as exploratory, next to the frozen scalar CRPS skill against climatology.

## Output
run-20261110-v1/addendum-1-climatology.json and .md.
