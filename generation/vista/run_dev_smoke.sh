#!/usr/bin/env bash
# Local DEV runs of the Vista adapter at the superseded native size 576 x 1024 (the first DEV smoke and the paused T2
# action-scale check, layout local48; their specs name no size, so the adapter reads them as native). Research server only,
# on the one GPU the server's rules allow (local.env: WCB_LOCAL_GPU, WCB_LOCAL_GPU_UUID, WCB_GPU_RUNNER or WCB_GPU_LOCK).
# The canvas size 288 x 512 (the default since 2026-11-04) runs through run_local_chunks.sh (the cluster job script, layout
# official) locally and through cluster/vista on cluster; a T2 check at the canvas size needs new specs (make_vista_specs.py).
#
#   bash run_dev_smoke.sh OUT_DIR [STAGES]   STAGES: comma list of equiv,smoke,rerun,action,h0ctl,ta,instrument,checks
#                                             (default: equiv,smoke,rerun,checks = the smoke alone)
#   equiv       GPU: --equivalence-test (2 sampling steps): the adapter's rollout vs sample_utils.do_sample, bit for bit
#   smoke       GPU: 2eeaccb0-t2.5, 47a9fb77-t9.5 x (prefix, recon, h0 s0, h0 s1, t2 s0, t2 s1); native 1024 clips and latents
#   rerun       GPU: a second process reruns 2eeaccb0-t2.5 (prefix, recon, h0 s0, t2 s0) into OUT_DIR/rerun (determinism)
#   action      GPU: the action-scale set of the Nano T2 rule (nano-t2/names-t2-scale-all.txt, 33 DEV windows) x (prefix,
#               recon, t2 s0, t2 s1); the 2 smoke windows are reused (same specs)
#   h0ctl       GPU: h0 s0 on the 33 windows (the action-free control of the same seeds)
#   ta          GPU: the time-aligned trajectory variant (dose-response diagnostic), t2 s0 on the 23 moving windows
#   instrument  GPU (small): the frozen detector (batch 4, as canvas-v1) and the band odometry on the t2 / reference / h0 / ta clips
#   checks      CPU: check_vista_smoke.py (+ contact sheets) and action_scale_check.py (the Nano T2 rule S1-S4)
# Each GPU process runs through the server's GPU runner (WCB_GPU_RUNNER: one GPU job at a time over all users; it pauses the
# GPU memory hold only while the process runs; NEED_MIB = the free memory it waits for) or, without one, under the shared
# lock (flock WCB_GPU_LOCK); this script never touches a hold itself. Inside, guarded_run.py stops its own process group if
# MemAvailable < 4 GiB or the GPU's free memory < 6 GiB, and the adapter's allocator cap leaves 6 GiB of the free memory free.
# Vista's released low_vram sampler peaks at about 39 GiB allocated (probe_vista_memory.py: the conditioner's VAE encoder on
# 25 fp32 copies of the 576 x 1024 anchor; the 14-latent fp32 decode window about 32 GiB), so a Vista process starts only
# with VISTA_MIN_FREE_MIB (default 46000) free. On a 48 GB GPU the released layout does not fit next to the 6 GB reserve (it
# OOMs above 40.4 GiB in the conditioner): the local specs pin the layout local48 (low_vram + cond_encode_chunk 1, peak about
# 35 GiB), and the equivalence test uses VISTA_LAYOUT (default local48). The official layout needs a GPU of 80 GB or more.
set -euo pipefail
OUT=${1:?usage: run_dev_smoke.sh OUT_DIR [STAGES]}
STAGES=${2:-equiv,smoke,rerun,checks}
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/local_env.sh"; vista_settings                 # PY SRC CKPT IPY LOCK GPU GPU_UUID
IN=$WCB_OUTPUTS/vista-v1/inputs
BENCH=$WCB_OUTPUTS/bench-v1
MAIN=$OUT/main
MIN_FREE=${VISTA_MIN_FREE_MIB:-46000}
export CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export MALLOC_MMAP_THRESHOLD_=65536                    # large host buffers go back to the OS when freed (lower host peak)
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128   # allocator only (less fragmentation; torch 2.0.1 has no expandable segments)
export WCB_SNAPSHOT_DIR
mkdir -p "$OUT/logs"
has() { [[ ",$STAGES," == *",$1,"* ]]; }
STOP=$OUT/STOP                                         # touch it to stop after the sample in flight (resume: run again)
stopped() { if [ -f "$STOP" ]; then echo "$(date -Is) stop file present: stopping before $1" | tee -a "$OUT/logs/stops.txt"; exit 3; fi; }

guard() {  # guard TAG CMD...: one GPU process under the lock (or the server's GPU runner) and the guard
  local tag=$1; shift
  local pre=(flock "$LOCK")
  # WCB_GPU_RUNNER (local.env): the server's GPU-0 runner, which serialises all GPU-0 users and pauses / resumes the GPU-0
  # memory hold itself (NEED_MIB = the free memory to wait for); without it, flock on WCB_GPU_LOCK
  if [ -n "${WCB_GPU_RUNNER:-}" ]; then pre=(env NEED_MIB="${NEED_MIB:-$MIN_FREE}" bash "$WCB_GPU_RUNNER"); fi
  "${pre[@]}" "$PY" "$HERE/guarded_run.py" --log "$OUT/logs/$tag-guard.json" --min-host-avail-gib 4 --pid-file "$OUT/logs/child.pid" \
      --gpu-uuid "$GPU_UUID" --min-gpu-free-mib 6144 -- /usr/bin/time -v "$@" > "$OUT/logs/$tag.out" 2> "$OUT/logs/$tag.err"
}

adapter() {  # adapter TAG OUTPUT_ROOT SPECS...
  local tag=$1 root=$2; shift 2
  guard "$tag" "$PY" "$HERE/run_vista_bench.py" --specs "$@" --input-root "$IN" --output-root "$root" --vista-src "$SRC" --ckpt "$CKPT" \
    --verify-weights size --resume --save-native --save-latents --expect-gpu-uuid "$GPU_UUID" --min-free-mib "$MIN_FREE" \
    --leave-free-mib 6144 --min-host-avail-gib 28 --stop-file "$STOP"
}

date -Is >> "$OUT/logs/started.txt"
if has equiv && [ ! -f "$OUT/equivalence-steps2.json" ]; then
  stopped equiv
  guard equiv "$PY" "$HERE/run_vista_bench.py" --equivalence-test --input-root "$IN" --window 2eeaccb0-t2.5 --vista-src "$SRC" --ckpt "$CKPT" \
    --steps 2 --size "${VISTA_SIZE:-native}" --out "$OUT/equivalence-steps2.json" --expect-gpu-uuid "$GPU_UUID" --min-free-mib "$MIN_FREE" --leave-free-mib 6144 \
    --layout "${VISTA_LAYOUT:-local48}"
fi
if has smoke; then stopped smoke; adapter smoke "$MAIN" "$IN/specs/dev-smoke"; fi
if has rerun; then stopped rerun; adapter rerun "$OUT/rerun" "$IN/specs/dev-smoke-rerun"; fi
if has action; then stopped action; adapter action "$MAIN" "${ACTION_SPECS:-$IN/specs/dev-action}"; fi   # ACTION_SPECS: a subset (s0 first)
if has h0ctl; then stopped h0ctl; adapter h0ctl "$MAIN" "$IN/specs/dev-action-h0"; fi
if has ta; then stopped ta; adapter ta "$MAIN" "$IN/specs/dev-action-ta"; fi

if has instrument; then   # one run view per clip kind (symlinks to the clip dirs), then the frozen detector and the band odometry
  CI=$WCB_SCRIPTS/canvas_instrument
  SPEC=$HERE/canvas_spec_vista.json
  NAMES=$IN/names-dev-action33.txt
  for kind in t2:vista-dev-t2-k2 ref:vista-dev-recon h0:vista-dev-h0-k1 t2ta:vista-dev-t2ta-k1; do
    tag=${kind%%:*}; run=${kind#*:}
    [ -d "$MAIN/$run/native" ] || continue
    stopped "instrument-$tag"
    v=$OUT/view-$tag/native; mkdir -p "$v"
    for d in "$MAIN/$run"/native/*; do   # finished samples only (their record says success)
      if [ -f "$d/vision.mp4" ] && grep -q '"status": "success"' "$d/sample_outputs.json" 2>/dev/null; then ln -sfn "$d" "$v/$(basename "$d")"; fi
    done
    NEED_MIB=12000 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True guard "det-$tag" "$IPY" "$CI/detect_canvas.py" --bench "$BENCH" --names-file "$NAMES" \
      --model "$SPEC" --source "vista-$tag" --from "run:$OUT/view-$tag" --out "$OUT/detections" --batch 4 --max-gib 3.5 --leave-free-mib 6144
    NEED_MIB=12000 guard "vo-$tag" "$IPY" "$CI/ego_vo_band.py" measure --bench "$BENCH" --names-file "$NAMES" --model "$SPEC" --source "vista-$tag" \
      --from "run:$OUT/view-$tag" --det-root "$OUT/detections" --out "$OUT/ego-vo" --gpu-mem-gib 2.5 --min-free-mib 6144
  done
fi

if has checks; then
  if [ -d "$OUT/rerun" ]; then RR=(--rerun-root "$OUT/rerun" --rerun-specs "$IN/specs/dev-smoke-rerun"); else RR=(); fi
  "$PY" "$HERE/check_vista_smoke.py" --out-root "$MAIN" "${RR[@]}" --specs "$IN/specs/dev-smoke" --result "$OUT/check-smoke.json" \
    --sheet "$OUT/contact-sheet-smoke.png" > "$OUT/logs/check-smoke.txt" || true
  if [ -d "$OUT/ego-vo" ]; then
    "$PY" "$HERE/check_vista_smoke.py" --out-root "$MAIN" --specs "$IN/specs/dev-action" --result "$OUT/check-action.json" \
      --sheet "$OUT/contact-sheet-action.png" > "$OUT/logs/check-action.txt" || true
    RAWVO=$BENCH/instrument/canvas-v1/ego-vo/band-10hz
    "$IPY" "$HERE/action_scale_check.py" --bench "$BENCH" --inputs "$IN" --names "$IN/names-dev-action33.txt" \
      --source "t2=$OUT/ego-vo:vista-t2:0,1" --source "ref=$OUT/ego-vo:vista-ref:real" --source "h0=$OUT/ego-vo:vista-h0:0" \
      --source "t2ta=$OUT/ego-vo:vista-t2ta:0" --source "raw=$RAWVO:canvas-raw:raw" \
      --out "$OUT/action-scale-check.json" --md "$OUT/action-scale-check.md" > "$OUT/logs/action-check.txt"
  fi
fi
date -Is >> "$OUT/logs/finished.txt"
