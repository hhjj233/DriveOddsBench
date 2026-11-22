#!/usr/bin/env bash
# One local GPU job of the Wan2.1-VACE-1.3B tools through the server's GPU runner (WCB_GPU_RUNNER in local.env; it
# serialises our GPU jobs and pauses / resumes the GPU-0 memory hold itself; this script never touches the hold), with
# loud failure: a non-zero exit and a .failed marker; .done on success. Meant to run inside tmux (session wan21_vace13b-*).
#   bash run_gpu_job.sh NAME NEED_MIB MIN_HOST_AVAIL_GIB -- COMMAND [ARGS...]
# Markers and logs: $VACE_OUT/jobs/NAME.{log,done,failed}. Refuses to start when MemAvailable < MIN_HOST_AVAIL_GIB (the
# job's host peak + 4 GB) or when NAME.done exists.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/local_env.sh"; load_local_env; need_setting VACE_OUT WCB_GPU_RUNNER
NAME="${1:?usage: run_gpu_job.sh NAME NEED_MIB MIN_HOST_AVAIL_GIB -- COMMAND...}"; NEED="${2:?}"; MINAVAIL="${3:?}"; shift 3
[ "${1:-}" = -- ] && shift
J="$VACE_OUT/jobs"; mkdir -p "$J"
[ -f "$J/$NAME.done" ] && { echo "$NAME already done ($J/$NAME.done)"; exit 0; }
rm -f "$J/$NAME.failed"
avail=$(awk '/MemAvailable/ {printf "%d", $2 / 1048576}' /proc/meminfo)
if [ "$avail" -lt "$MINAVAIL" ]; then
  echo "$(date -Is) $NAME: MemAvailable $avail GiB < $MINAVAIL GiB, not starting" | tee -a "$J/$NAME.log" >&2
  echo "host memory" > "$J/$NAME.failed"; exit 6
fi
echo "$(date -Is) $NAME: NEED_MIB=$NEED MemAvailable=${avail}GiB runner=$WCB_GPU_RUNNER cmd: $*" >> "$J/$NAME.log"
NEED_MIB="$NEED" bash "$WCB_GPU_RUNNER" "$@" >> "$J/$NAME.log" 2>&1
rc=$?
echo "$(date -Is) $NAME: exit $rc" >> "$J/$NAME.log"
if [ "$rc" = 0 ]; then touch "$J/$NAME.done"; else echo "exit $rc" > "$J/$NAME.failed"; fi
exit "$rc"
