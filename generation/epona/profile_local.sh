#!/usr/bin/env bash
# profile_epona.py on the research server: the allowed GPU only, shared lock, memory guards (as run_dev_local.sh). DEV only.
#   bash profile_local.sh OUT_DIR TAG [profile_epona.py arguments without --out / --expect-gpu-uuid]
# Machine-specific settings as run_dev_local.sh (models/epona/local.env, the environment wins); profile_epona.py reads
# EPONA_MODELS_DIR and EPONA_SRC (default $EPONA_ENV/src-git) the same way.
set -euo pipefail
OUT=${1:?usage: profile_local.sh OUT_DIR TAG ARGS...}; TAG=${2:?}; shift 2
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/local_env.sh"; epona_settings                # PY SRC MODELS LOCK GPU GPU_UUID (local_env.sh)
export CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONNOUSERSITE=1 MALLOC_MMAP_THRESHOLD_=65536
export TRITON_CACHE_DIR=$EPONA_ENV/triton-cache TORCHINDUCTOR_COMPILE_THREADS=2
export TORCHINDUCTOR_CACHE_DIR=$EPONA_ENV/inductor-cache
mkdir -p "$OUT/logs"
flock "$LOCK" "$PY" "$HERE/guarded_run.py" --log "$OUT/logs/$TAG-guard.json" --min-host-avail-gib 4 \
    --gpu-uuid "$GPU_UUID" --min-gpu-free-mib 6144 -- \
  "$PY" "$HERE/profile_epona.py" --out "$OUT" --expect-gpu-uuid "$GPU_UUID" "$@" \
  > "$OUT/logs/$TAG.out" 2> "$OUT/logs/$TAG.err"
