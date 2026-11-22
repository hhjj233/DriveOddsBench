#!/usr/bin/env bash
# One guarded local run of the Epona adapter on DEV specs (research server; the allowed GPU only; shared lock).
#
#   bash run_dev_local.sh INPUT_ROOT OUTPUT_ROOT TAG SPEC [SPEC ...]      (SPEC: spec JSON files or directories)
#
# Machine-specific settings: models/epona/local.env (template local.env.example; the environment wins): EPONA_ENV,
# EPONA_MODELS_DIR, WCB_GPU_LOCK, WCB_LOCAL_GPU, WCB_LOCAL_GPU_UUID, optionally EPONA_SRC.
# Environment: EPONA_EXTRA (more adapter options, e.g. --batch-k 2), EPONA_NATIVE=0 (no native.mp4), EPONA_MAX_GPU_MIB.
# Takes the shared GPU lock (flock on WCB_GPU_LOCK, also used by the other bench agents), runs under guarded_run.py (stops its
# own process group if MemAvailable < 4 GiB or the GPU's free memory < 6 GiB), caps the process so that at least 6 GiB of the
# GPU memory free at start stays free, and refuses TEST specs (the adapter is not given --allow-test because there is none:
# this wrapper checks the split itself). Logs go to OUTPUT_ROOT/logs/TAG.{out,err} and TAG-guard.json.
set -euo pipefail
INPUTS=${1:?usage: run_dev_local.sh INPUT_ROOT OUTPUT_ROOT TAG SPEC...}; OUT=${2:?}; TAG=${3:?}; shift 3
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/local_env.sh"; epona_settings                # PY SRC MODELS LOCK GPU GPU_UUID (local_env.sh)
if grep -rlq '"split": "test"' "$@" 2>/dev/null; then echo "refusing: TEST specs among the inputs" >&2; exit 2; fi
export CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONNOUSERSITE=1 MALLOC_MMAP_THRESHOLD_=65536
export TRITON_CACHE_DIR=$EPONA_ENV/triton-cache
mkdir -p "$OUT/logs"
NATIVE_FLAG=--save-native; [ "${EPONA_NATIVE:-1}" = "0" ] && NATIVE_FLAG=""   # EPONA_NATIVE=0: no native.mp4
# EPONA_EXTRA: further adapter options, e.g. "--batch-k 2 --writer-preset ultrafast"; EPONA_MAX_GPU_MIB: GPU cap (default 30000)
flock "$LOCK" "$PY" "$HERE/guarded_run.py" --log "$OUT/logs/$TAG-guard.json" --min-host-avail-gib 4 \
    --gpu-uuid "$GPU_UUID" --min-gpu-free-mib 6144 -- \
  "$PY" "$HERE/run_epona_bench.py" --specs "$@" --input-root "$INPUTS" --output-root "$OUT" \
    --models-dir "$MODELS" --epona-src "$SRC" $NATIVE_FLAG --resume --expect-gpu-uuid "$GPU_UUID" \
    --min-free-mib 26000 --leave-free-mib 6144 --max-gpu-mib "${EPONA_MAX_GPU_MIB:-30000}" --min-host-avail-gib 13 ${EPONA_EXTRA:-} \
  > "$OUT/logs/$TAG.out" 2> "$OUT/logs/$TAG.err"
