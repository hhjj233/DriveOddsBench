#!/usr/bin/env bash
# LTX-2.5 local MECHANICS smoke on the research server (GPU 1 only, serialized by the shared GPU lock). DEV windows only.
# Variant: distilled (default; fp8-cast specs, because bf16 does not fit GPU 1: the analysed clips are made in bf16 on the
# cluster H200, see cluster/ltx25/RUNBOOK.md) or dev (VARIANT=dev).
#   1. the variant's weights must be downloaded and verified (download_ltx25_weights.sh: weights-manifest.json)
#   2. prompt cache: the fixed prompt through the Gemma 4 text encoder and this variant's embeddings processor, once
#      (about 25 GiB GPU)
#   3. smoke: 2eeaccb0-t2.5 and 47a9fb77-t9.5 x (recon, s0, s1); distilled: fp8-cast, transformer per sample (built once
#      for both stages, freed before decoding), native20.mp4 and the 1024x576 decode (full1024.mp4) kept
#   4. rerun: s0 and recon of 2eeaccb0-t2.5 in a fresh process with another transformer lifecycle (distilled: the official
#      per-call build; dev: resident): same seeds, so the clips must be bit-identical
#   5. checks and contact sheet (CPU)
# Each GPU step refuses to start unless the GPU can give it the memory it needs while 6 GiB stay free, and caps its own
# allocator at (free - 6 GiB - 1 GiB); each step also checks the host's MemAvailable.
# Machine-specific settings (local.env, template local.env.example): LTX25_ENV, LTX25_MODELS_DIR, WCB_OUTPUTS, WCB_GPU_LOCK,
# WCB_LOCAL_GPU (the GPU index the server's rules allow), optionally LTX25_SRC.
#
#   [VARIANT=distilled|dev] run_dev_smoke.sh [OUT_DIR]   (default $WCB_OUTPUTS/ltx25d-dev-localsmoke-v1 or ltx25-dev-smoke-v1)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/local_env.sh"
load_local_env
need_setting LTX25_ENV LTX25_MODELS_DIR WCB_OUTPUTS WCB_GPU_LOCK WCB_LOCAL_GPU
ENV="$LTX25_ENV"
PY="$ENV/venv/bin/python"
export LTX25_SRC="${LTX25_SRC:-$ENV/src/LTX-2}"
VARIANT="${VARIANT:-distilled}"
IN="$WCB_OUTPUTS/ltx25-inputs-v1"
BENCH="$WCB_OUTPUTS/bench-v1"
WEIGHTS="$LTX25_MODELS_DIR"
LOCK="$WCB_GPU_LOCK"
GPU="$WCB_LOCAL_GPU"
if [[ "$VARIANT" == distilled ]]; then PFX=ltx25d; SMOKE=dev-localsmoke; RES_SMOKE=per-sample; RES_RERUN=per-call; FULL="--keep-native20 --keep-full-res"
else PFX=ltx25; SMOKE=dev-smoke; RES_SMOKE=per-sample; RES_RERUN=resident; FULL=--keep-native20; fi
OUT="${1:-$WCB_OUTPUTS/${PFX}-${SMOKE}-v1}"
CACHE="$OUT/prompt-cache-$VARIANT.safetensors"
KEEP_FREE_MIB=6144
export CUDA_VISIBLE_DEVICES="$GPU" OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1
mkdir -p "$OUT"
log() { echo "$(date '+%F %T') $*" | tee -a "$OUT/smoke.log"; }

"$PY" -I -c "import json,sys; m=json.load(open(sys.argv[1])); sys.exit(0 if m.get('all_ok') and m.get('variants_complete', {}).get(sys.argv[2]) else 1)" \
  "$WEIGHTS/weights-manifest.json" "$VARIANT" || { log "weights for $VARIANT not verified: run download_ltx25_weights.sh --variant $VARIANT first"; exit 1; }
if nvidia-smi -i "$GPU" -q | grep -qi "requires reset.*yes"; then log "GPU $GPU requires a reset"; exit 1; fi

gpu() {  # run a GPU step under the lock; report free memory and MemAvailable first
  log "waiting for $LOCK: $*"
  flock -w 7200 "$LOCK" bash -c '
    echo "$(date "+%F %T") lock held; GPU $WCB_LOCAL_GPU: $(nvidia-smi -i "$WCB_LOCAL_GPU" --query-gpu=memory.used,memory.free --format=csv,noheader); $(grep MemAvailable /proc/meminfo)"
    exec "$@"' _ "$@" 2>&1 | tee -a "$OUT/smoke.log"
  return "${PIPESTATUS[0]}"
}

if [[ ! -f "$CACHE" ]]; then
  gpu "$PY" "$HERE/run_ltx25_bench.py" --variant "$VARIANT" --models-dir "$WEIGHTS" --ltx-src "$LTX25_SRC" --encode-prompt "$CACHE" \
      --keep-free-mib $KEEP_FREE_MIB --min-host-avail-mib 10240
fi
gpu "$PY" "$HERE/run_ltx25_bench.py" --variant "$VARIANT" --models-dir "$WEIGHTS" --ltx-src "$LTX25_SRC" --specs "$IN/specs/$VARIANT/$SMOKE" \
    --input-root "$IN" --output-root "$OUT" --prompt-cache "$CACHE" --residency $RES_SMOKE --keep-free-mib $KEEP_FREE_MIB $FULL --resume
gpu "$PY" "$HERE/run_ltx25_bench.py" --variant "$VARIANT" --models-dir "$WEIGHTS" --ltx-src "$LTX25_SRC" --specs "$IN/specs/$VARIANT/$SMOKE-rerun" \
    --input-root "$IN" --output-root "$OUT" --prompt-cache "$CACHE" --residency $RES_RERUN --keep-free-mib $KEEP_FREE_MIB $FULL --resume \
  || log "rerun step failed (see the message above); it can be rerun with another --residency"

CUDA_VISIBLE_DEVICES="" "$PY" "$HERE/check_ltx25_smoke.py" --output-root "$OUT" --bench "$BENCH" --rerun-run "$PFX-$SMOKE-rerun" \
    --keep-free-mib $KEEP_FREE_MIB | tee -a "$OUT/smoke.log"
CUDA_VISIBLE_DEVICES="" "$PY" "$HERE/make_ltx25_contact_sheet.py" --bench "$BENCH" --output-root "$OUT" --windows 2eeaccb0-t2.5 47a9fb77-t9.5 \
    --rerun-run "$PFX-$SMOKE-rerun" --out "$OUT/contact-sheet.png" | tee -a "$OUT/smoke.log"
log "done: $OUT/smoke-check.json, $OUT/contact-sheet.png (private: AV2 frames are not blurred)"
