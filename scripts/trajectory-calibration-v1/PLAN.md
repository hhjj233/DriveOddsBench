# Trajectory-level calibration v1: plan (coordinator, 2026-11-10, written before any trajectory statistic is computed)

Status: pre-specified secondary analysis on TEST, labelled exploratory in the paper. It does not change H1, H2 or any
frozen analysis. No clip is generated or re-measured: every reading comes from the frozen TEST measurement files
(per-frame instrument series and band-VO trajectories) that the frozen v1 analyses already used.

## Question
The frozen analyses rank one scalar per family (for example the lead gap at 4 s). Two continuations can agree at that
time and still differ in how they get there. Is each model calibrated for the whole trajectory of the primary quantity,
and how do its centering and its uncertainty change with the horizon?

## Data
- Models: Orbis 2, Cosmos3-Nano, Cosmos3-Edge, Epona, LTX-2.5 (2 s video history), Vista (single frame), TEST, K = 16.
- Cases: exactly the (window, family) cases retained by the frozen v1 analysis for the family's primary outcome
  (the real reading and all 16 generated readings scorable in the frozen outcome CSVs). No other filter.
- Curves (same reading rule as the primary outcomes: per frame the median over the three trackers, each tracker
  carrying its last reading after it stops; abs(y) is the median over trackers of each tracker's abs(y), the rule of
  the scalar absy outcomes; ego from the band-VO planar trajectory, D(t) = forward advance):
  | family | quantity | grid (s) | reason for the horizon |
  |---|---|---|---|
  | lead, lead_stop | forward distance x(t) | 0.4, 0.8, ..., 4.0 | x_6s is not reliable for several models |
  | cutin | lateral offset abs(y(t)) | 0.4, 0.8, ..., 4.0 | absy_4s is reliable for every model |
  | oncoming | forward distance x(t) | 0.2, 0.4, ..., 1.0 | the target leaves the view; only x_1s is validated |
  | ego_inter | ego advance D(t) | 0.4, 0.8, ..., 6.0 | VO grid; ego_dx_0_6s is tested for every model |
- The 0.4 s grid is shared by the 10 Hz models and Epona (5 Hz, frames every 0.2 s).
- Shared validity rule: a case enters only if all 17 curves have a value at every grid time. Dropped cases are
  reported (expected none, because the scalar rule already requires a t0 match and carries readings forward).
- Consistency check (not a result): at the primary horizon the curve must equal the frozen CSV value (abs diff <= 1e-3)
  for every retained curve; otherwise the run stops.

## Statistics (per model, per family, and pooled over the five scored families)
Ties at any step are broken by independent uniform tags (seed 2026), as in the frozen analyses.
1. Horizon-resolved centering and coverage: at each grid time, the rank of the real value among the 16, then
   mean PIT - 1/2 with a 95 % window-bootstrap interval (B = 2000) and the share outside all 16 samples
   (calibrated 2/17). Descriptive curves.
2. Whole-trajectory rank, two pre-rank functions (Thorarinsdottir, Scheuerer and Heinz 2016), each symmetric in the
   17 curves so that, under calibration, the real curve's rank among them is uniform on 0..16:
   - band depth: rho_i = mean over t of r_it (18 - r_it), where r_it in 1..17 is curve i's rank at time t;
     R_bd = number of generated curves with smaller rho than the real curve. R_bd = 0 means the real curve is the
     least central of the 17. Statistic: share of cases with R_bd = 0 (calibrated 1/17).
   - average rank: rho_i = mean over t of r_it; R_avg as above. Statistics: share with R_avg in {0, 16}
     (calibrated 2/17) and mean PIT - 1/2 of R_avg.
   Per family: exact two-sided binomial p. Pooled: the window statistic of H2 (mean over windows of the share of the
   window's scored families that hit the event), with its exact null simulated (B = 100000) and a window bootstrap
   interval. BH at q = 0.10 within each statistic across (model, family). All labelled exploratory.
3. Whole-trajectory skill: fair energy score of the 16 curves against the real curve, and the energy score of the
   constant-velocity curve fitted to the real history (lead and lead_stop: x0 + vx t; cutin: abs(y0 + vy t);
   ego: history speed times t; oncoming: none, its history is not reliable). Skill = 1 - sum ES_model / sum ES_cv
   over cases with a history fit, with a window bootstrap interval.
4. Positive control: sample 0 as the observation against samples 1-15 (K = 15) for statistics 1 and 2; it must be
   uniform (references 1/16 and 2/16).

## Outputs
results.json, results.md and the figure data in run-<date>-v1/. The paper may show at most one figure from this
analysis in the main text; everything else goes to the supplement.
