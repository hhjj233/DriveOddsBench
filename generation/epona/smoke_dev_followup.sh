#!/usr/bin/env bash
# Follow-up DEV checks after smoke_dev_local.sh (research server, the allowed GPU only, shared lock, guarded):
#   1. one moving-ego DEV window (0ed86959-t7.0, 12.8 m/s, lead + cut-in) x (recon, s0, s1): the time-scale check under motion;
#   2. check_epona_pose_sign.py on the 60 DEV generation windows (trajectory head only, three readings of the pose signs).
#   bash smoke_dev_followup.sh OUT_DIR
# Machine-specific settings as smoke_dev_local.sh (models/epona/local.env; bench files: $WCB_OUTPUTS/bench-v1).
set -euo pipefail
OUT=${1:?usage: smoke_dev_followup.sh OUT_DIR}
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/local_env.sh"; epona_settings WCB_OUTPUTS    # PY SRC MODELS LOCK GPU GPU_UUID (local_env.sh)
INPUTS=$WCB_OUTPUTS/epona-v1/inputs
BENCH=$WCB_OUTPUTS/bench-v1
export CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONNOUSERSITE=1 MALLOC_MMAP_THRESHOLD_=65536
export TRITON_CACHE_DIR=$EPONA_ENV/triton-cache
mkdir -p "$OUT/logs"
guard=("$PY" "$HERE/guarded_run.py" --min-host-avail-gib 4 --gpu-uuid "$GPU_UUID" --min-gpu-free-mib 6144)

flock "$LOCK" "${guard[@]}" --log "$OUT/logs/extra-guard.json" -- \
  "$PY" "$HERE/run_epona_bench.py" --specs "$INPUTS/specs/dev-smoke-extra" --input-root "$INPUTS" --output-root "$OUT" \
    --models-dir "$MODELS" --epona-src "$SRC" --save-native --resume --expect-gpu-uuid "$GPU_UUID" \
    --min-free-mib 26000 --leave-free-mib 6144 --max-gpu-mib 30000 --min-host-avail-gib 13 \
  > "$OUT/logs/extra.out" 2> "$OUT/logs/extra.err"

flock "$LOCK" "${guard[@]}" --log "$OUT/logs/pose-sign-guard.json" -- \
  "$PY" "$HERE/check_epona_pose_sign.py" --input-root "$INPUTS" --bench-root "$BENCH" --names "$BENCH/names-dev-gen.txt" \
    --out "$OUT/pose-sign-check.json" --models-dir "$MODELS" --epona-src "$SRC" --seeds 4 --expect-gpu-uuid "$GPU_UUID" \
    --leave-free-mib 6144 --max-gpu-mib 30000 \
  > "$OUT/logs/pose-sign.out" 2> "$OUT/logs/pose-sign.err"
date -Is > "$OUT/logs/followup-finished.txt"
