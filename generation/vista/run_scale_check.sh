#!/usr/bin/env bash
# The DEV action-scale check of Vista T2 (vista-v1/checks/ACTION-SCALE-RULE-vista.md) in priority order; every part is
# resumable (finished samples, detections and odometry records are skipped) and stoppable (touch OUT/STOP: it stops after
# the sample in flight). start_scale_check.sh runs it in tmux; every GPU step goes through the server's GPU runner
# (WCB_GPU_RUNNER), which pauses the GPU memory hold only while that step runs. Deferred with T2 since 2026-11-04.
#   bash run_scale_check.sh OUT
#   1a. t2 s0 of the 23 moving windows (with their prefixes and references), detection, band odometry, checks: an early,
#       indicative S1 on s0 (action-scale-check.md);
#   1b. t2 s0 of the other 10 windows (stationary, start-from-rest, slow), instrument, checks;
#   2.  t2 s1 of the 33 windows, instrument, checks: the rule's verdict on s0 + s1;
#   3.  the h0 control (s0) and the time-aligned diagnostic (t2 s0, 23 moving windows), instrument, checks.
set -euo pipefail
OUT=${1:?usage: run_scale_check.sh OUT}
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/local_env.sh"; load_local_env
IN=$WCB_OUTPUTS/vista-v1/inputs
for part in dev-action-s0-moving dev-action-s0-rest dev-action-s1; do
  ACTION_SPECS=$IN/specs/$part bash "$HERE/run_dev_smoke.sh" "$OUT" action,instrument,checks
done
bash "$HERE/run_dev_smoke.sh" "$OUT" h0ctl,ta,instrument,checks
