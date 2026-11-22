#!/usr/bin/env bash
# Local DEV smoke of the Epona adapter (bench-v1). Research server only; the allowed GPU only.
#
#   bash smoke_dev_local.sh OUT_DIR
#
# 1. adapter on 2 DEV windows (2eeaccb0-t2.5, 47a9fb77-t9.5) x (recon, s0, s1), with a full sha256 check of the weights
#    and native.mp4 kept for the audit;
# 2. a second process reruns 2eeaccb0-t2.5 s0 and recon with the same specs (same seeds) into OUT_DIR/rerun;
# Each GPU process runs under the shared lock (flock on WCB_GPU_LOCK, which the other bench agents also take), under
# guarded_run.py (stops its own process group if MemAvailable < 4 GiB or the GPU's free memory < 6 GiB), and with a
# per-process GPU cap that leaves at least 6 GiB of the GPU's free memory free. The checks are in check_epona_smoke.py.
# Machine-specific settings: models/epona/local.env (template local.env.example; the environment wins): EPONA_ENV,
# EPONA_MODELS_DIR, WCB_OUTPUTS (inputs: $WCB_OUTPUTS/epona-v1/inputs), WCB_GPU_LOCK, WCB_LOCAL_GPU, WCB_LOCAL_GPU_UUID.
set -euo pipefail
OUT=${1:?usage: smoke_dev_local.sh OUT_DIR}
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/local_env.sh"; epona_settings WCB_OUTPUTS    # PY SRC MODELS LOCK GPU GPU_UUID (local_env.sh)
INPUTS=$WCB_OUTPUTS/epona-v1/inputs
SPECS=$INPUTS/specs/dev-smoke
export CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONNOUSERSITE=1
export MALLOC_MMAP_THRESHOLD_=65536                   # large host buffers go back to the OS when freed (lower host peak)
export TRITON_CACHE_DIR=$EPONA_ENV/triton-cache
mkdir -p "$OUT/logs"

main_specs=()
for w in 2eeaccb0-t2.5 47a9fb77-t9.5; do
  main_specs+=("$SPECS/$w/$w-recon.json" "$SPECS/$w/$w-s0.json" "$SPECS/$w/$w-s1.json")
done
rerun_specs=("$SPECS/2eeaccb0-t2.5/2eeaccb0-t2.5-recon.json" "$SPECS/2eeaccb0-t2.5/2eeaccb0-t2.5-s0.json")

run_adapter() {  # $1 = output root, $2 = log tag, $3 = weight check, rest = specs
  local root=$1 tag=$2 verify=$3; shift 3
  flock "$LOCK" "$PY" "$HERE/guarded_run.py" --log "$OUT/logs/$tag-guard.json" --min-host-avail-gib 4 \
      --gpu-uuid "$GPU_UUID" --min-gpu-free-mib 6144 -- \
    /usr/bin/time -v "$PY" "$HERE/run_epona_bench.py" --specs "$@" --input-root "$INPUTS" --output-root "$root" \
      --models-dir "$MODELS" --epona-src "$SRC" --verify-weights "$verify" --save-native --resume \
      --expect-gpu-uuid "$GPU_UUID" --min-free-mib 26000 --leave-free-mib 6144 --max-gpu-mib 30000 --min-host-avail-gib 13 \
    > "$OUT/logs/$tag.out" 2> "$OUT/logs/$tag.err"
}

date -Is > "$OUT/logs/started.txt"
run_adapter "$OUT" main sha256 "${main_specs[@]}"
run_adapter "$OUT/rerun" rerun size "${rerun_specs[@]}"
date -Is > "$OUT/logs/finished.txt"
