#!/usr/bin/env bash
# Start or resume the Vista DEV action-scale check (T2; deferred since 2026-11-04) in tmux session vista-dev on the research
# server. Every GPU step goes through the server's GPU-0 runner (WCB_GPU_RUNNER in local.env, which serialises the GPU-0
# users and pauses / resumes the GPU-0 memory hold itself); this script never touches the hold. Marker at the end:
#   vista-v1/checks/scale-check.done | .stopped (clean stop by OUT/STOP) | .failed (exit code; non-zero exit)
#   bash start_scale_check.sh       (resume: the same command; finished samples, detections and odometry records are skipped)
# Stop cleanly after the sample in flight:  touch $WCB_OUTPUTS/vista-v1/dev-local/STOP
# Stop now (the sample in flight is lost):  tmux send-keys -t vista-dev C-c
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/local_env.sh"; vista_settings WCB_GPU_RUNNER
OUT=$WCB_OUTPUTS/vista-v1/dev-local
MARK=$WCB_OUTPUTS/vista-v1/checks/scale-check
if tmux has-session -t vista-dev 2>/dev/null; then echo "tmux session vista-dev exists (tmux attach -t vista-dev; or tmux kill-session -t vista-dev if its chain has ended)"; exit 1; fi
if [ -f "$OUT/STOP" ]; then echo "$OUT/STOP exists (a clean stop was requested): rm it to resume"; exit 1; fi
rm -f "$MARK.done" "$MARK.failed" "$MARK.stopped"
tmux new-session -d -s vista-dev "env PATH='$PATH' VISTA_LAYOUT=local48 bash '$HERE/run_scale_check.sh' '$OUT'; rc=\$?; \
  if [ \$rc = 0 ]; then date -Is > '$MARK.done'; elif [ \$rc = 3 ] && [ -f '$OUT/STOP' ]; then echo \"\$(date -Is) stopped by $OUT/STOP\" > '$MARK.stopped'; \
  else echo \"\$(date -Is) FAILED exit \$rc (logs: $OUT/logs)\" > '$MARK.failed'; fi; echo \"chain exit \$rc\"; exec bash"
echo "started tmux session vista-dev; logs $OUT/logs; marker $MARK.{done,stopped,failed}"
