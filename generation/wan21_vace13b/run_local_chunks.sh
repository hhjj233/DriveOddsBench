#!/usr/bin/env bash
# Run chunks of the Wan2.1-VACE-1.3B cluster input package on the research server's GPU exactly as a cluster job runs them
# (cluster/wan21_vace13b/run_chunk.sh: the same adapter, pins, prompt cache and weights layout), with the locked environment
# of the container (VACE_PY in local.env), through the server's GPU runner (WCB_GPU_RUNNER: it serialises our GPU jobs and
# pauses / resumes the GPU-0 memory hold itself; this script never touches the hold). Each runner call does at most
# VACE_LOCAL_MAX_CONT continuations (run_chunk.sh WCB_MAX_CONTINUATIONS, default 2; prefix and reference samples are not
# counted), then the job script's checkpoint exit (85) hands
# the GPU back to the runner's queue and the chunk goes on in the next call, so other GPU users wait at most about one call.
# Outputs are the same wan21-vace13b-out-<chunk>.tar as on cluster, so cluster/wan21_vace13b/check_smoke.py and
# collect_outputs.py check them unchanged.
#
#   bash run_local_chunks.sh PACKAGE_DIR CHUNK [CHUNK ...]      e.g.  ... wan21-vace13b-512-inputs-v1 dev-smoke
# PACKAGE_DIR: the package (its outer tars <package>-{dev,test}.tar next to SHA256SUMS; the chunk tars and the prompt cache
# are read from inside them). Work dirs: $VACE_OUT/local-runs/<package>/<chunk>/; resumable (out/ is kept; a chunk whose
# wan21-vace13b-out-<chunk>.tar exists and whose run ended with status 0 is skipped). Loud failure: a non-zero exit and
# <chunk>/FAILED. Stop cleanly: touch $VACE_OUT/local-runs/STOP (all packages; the adapter stops after the sample in flight)
# or $VACE_OUT/local-runs/<package>/STOP (this package; before its next call); rm it and run again to resume.
# TEST chunks are refused unless bench-v1/protocol-wan21-vace13b-frozen.txt exists.
# Settings (environment): VACE_NEED_MIB (the GPU memory the runner waits for: the measured peak + 2000), VACE_MIN_HOST_GIB
# (the adapter's MemAvailable floor: the measured host peak + 4), VACE_LOCAL_MAX_CONT, VACE_LOCAL_SKIP_RERUN=1 (a size probe:
# the dev-smoke chunk without its rerun), WCB_GPU_RUNNER / WCB_LOCAL_GPU_UUID (override local.env to use the other GPU).
set -euo pipefail
PKGDIR=$(cd "${1:?usage: run_local_chunks.sh PACKAGE_DIR CHUNK ...}" && pwd); shift
[ $# -gt 0 ] || { echo "no chunk given" >&2; exit 2; }
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
. "$HERE/local_env.sh"; load_local_env; need_setting WCB_GPU_RUNNER VACE_PY VACE_SRC VACE_CKPT VACE_OUT WCB_OUTPUTS WCB_LOCAL_GPU_UUID
PKG=$(basename "$PKGDIR")
RUNS=$VACE_OUT/local-runs
STOP=$RUNS/STOP
NEED=${VACE_NEED_MIB:-24000}
MINHOST=${VACE_MIN_HOST_GIB:-24}
MAXC=${VACE_LOCAL_MAX_CONT:-2}
(cd "$PKGDIR" && sha256sum -c --quiet SHA256SUMS) || { echo "$PKGDIR: SHA256SUMS does not match" >&2; exit 2; }
for CH in "$@"; do
  case "$CH" in test-*) [ -f "$WCB_OUTPUTS/bench-v1/protocol-wan21-vace13b-frozen.txt" ] || { echo "$CH: TEST chunks only after the freeze" >&2; exit 2; };; esac
  [ -f "$STOP" ] && { echo "$(date -Is) stop file $STOP: not starting $CH"; exit 3; }
  W=$RUNS/$PKG/$CH
  mkdir -p "$W"
  [ -f "$RUNS/$PKG/STOP" ] && { echo "$(date -Is) stop file $RUNS/$PKG/STOP: not starting $CH"; exit 3; }
  if [ -f "$W/wan21-vace13b-out-$CH.tar" ] && [ "$(cat "$W/status" 2>/dev/null)" = 0 ]; then echo "$(date -Is) $CH done, skipped"; continue; fi
  rm -f "$W/FAILED"
  split=${CH%%-*}; [ "$CH" = dev-smoke ] && split=dev
  # the job's sandbox: the chunk tar and the prompt cache from the package, the code from this checkout, the weights layout of stage_weights.sh
  tar -xOf "$PKGDIR/$PKG-$split.tar" "$PKG/chunks/$PKG-$CH.tar" > "$W/$PKG-$CH.tar"
  tar -xOf "$PKGDIR/$PKG-$split.tar" "$PKG/prompt-cache/vace-t5-long.safetensors" > "$W/vace-t5-long.safetensors"
  tar -xOf "$PKGDIR/$PKG-$split.tar" "$PKG/prompt-cache/vace-t5-long.json" > "$W/vace-t5-long.json"
  cp "$HERE/run_vace_bench.py" "$HERE/vace_common.py" "$HERE/vace_pins.json" "$W/"
  ln -sfn "$VACE_CKPT/diffusion_pytorch_model.safetensors" "$W/diffusion_pytorch_model.safetensors"
  ln -sfn "$VACE_CKPT/Wan2.1_VAE.pth" "$W/Wan2.1_VAE.pth"
  mkdir -p "$W/.ckpt-small"
  cp "$VACE_CKPT/config.json" "$VACE_CKPT/LICENSE.txt" "$VACE_CKPT/README.md" "$W/.ckpt-small/"
  grep -E '  (LICENSE.txt|README.md|Wan2.1_VAE.pth|config.json|diffusion_pytorch_model.safetensors)$' "$VACE_CKPT/SHA256SUMS" > "$W/.ckpt-small/SHA256SUMS"
  tar -cf "$W/wan21-vace13b-ckpt-small.tar" -C "$W/.ckpt-small" config.json LICENSE.txt README.md SHA256SUMS
  call=0
  while :; do
    call=$((call + 1))
    for st in "$STOP" "$RUNS/$PKG/STOP"; do [ -f "$st" ] && { echo "$(date -Is) stop file $st: $CH paused before call $call"; exit 3; }; done
    echo "$(date -Is) $CH: call $call, waiting for the GPU runner (NEED_MIB $NEED)"
    set +e
    (cd "$W" && env NEED_MIB="$NEED" WCB_PACKAGE="$PKG" WCB_PYTHON="$VACE_PY" VACE_SRC="$VACE_SRC" WCB_CKPT_WINDOWS="${VACE_LOCAL_CKPT_WINDOWS:-2}" \
       WCB_MAX_CONTINUATIONS="$MAXC" WCB_SKIP_RERUN="${VACE_LOCAL_SKIP_RERUN:-0}" OMP_NUM_THREADS=2 \
       WCB_ADAPTER_ARGS="--gpu-uuid $WCB_LOCAL_GPU_UUID --min-host-avail-gib $MINHOST --stop-file $STOP" \
       bash "$WCB_GPU_RUNNER" bash "$REPO/cluster/wan21_vace13b/run_chunk.sh" "$CH" "${VACE_LOCAL_QUEUE:-local}" >> "$W/run.out" 2>> "$W/run.err")
    rc=$?
    set -e
    echo "$rc" > "$W/status"
    echo "$(date -Is) $CH: call $call exit $rc ($W)"
    [ "$rc" = 85 ] && continue                       # checkpoint: the next samples in a new runner call
    break
  done
  if [ "$rc" != 0 ]; then echo "exit $rc at call $call" > "$W/FAILED"; exit "$rc"; fi
done
