#!/usr/bin/env bash
# Run chunks of the Vista cluster input package on the research server's GPU exactly as a cluster job runs them
# (cluster/vista/run_chunk.sh: layout official, the package's generation size, the same adapter and pins), with the locked
# environment of the container (VISTA_PY in local.env: the venv of requirements-vista-cu118.lock.txt), through the
# server's GPU runner (WCB_GPU_RUNNER: one call per group of windows, so other GPU users interleave; it pauses and resumes
# the GPU memory hold itself, this script never touches it). Outputs are the same vista-out-<chunk>.tar as on cluster, so
# cluster/vista/check_smoke.py and cluster/vista/collect_outputs.py check them unchanged.
#
#   bash run_local_chunks.sh PACKAGE_DIR CHUNK [CHUNK ...]      e.g.  ... vista-inputs-v3 dev-smoke
#                                                                     ... vista-inputs-v3 $(cat queues/dev.txt)
# PACKAGE_DIR: the unpacked package (its outer tars vista-inputs-v3-{dev,test}.tar next to SHA256SUMS; the chunk tars are
# read from inside them). Work dirs: $WCB_OUTPUTS/vista-v1/local-runs/<package>/<chunk>/; resumable (out/ is kept; a chunk
# whose vista-out-<chunk>.tar exists and whose run ended with status 0 is skipped). One GPU-runner call covers
# VISTA_LOCAL_CKPT_WINDOWS windows (default 2, one model load each): the job script's checkpoint exit (85) hands the GPU back
# to the runner's queue and the chunk goes on in the next call, so other GPU users wait at most about one such group.
# Stop cleanly: touch $WCB_OUTPUTS/vista-v1/local-runs/STOP (the adapter stops after the sample in flight; rm it and run
# again to resume). TEST chunks are refused unless bench-v1/protocol-vista-frozen.txt exists.
set -euo pipefail
PKGDIR=$(cd "${1:?usage: run_local_chunks.sh PACKAGE_DIR CHUNK ...}" && pwd); shift
[ $# -gt 0 ] || { echo "no chunk given" >&2; exit 2; }
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
. "$HERE/local_env.sh"; vista_settings WCB_GPU_RUNNER VISTA_PY
PKG=$(basename "$PKGDIR")
RUNS=$WCB_OUTPUTS/vista-v1/local-runs
STOP=$RUNS/STOP
CAP=${VISTA_MAX_GPU_MIB:-29000}         # allocator cap: 320 x 576 peaks at 22.5 GiB allocated; without a cap the cache grew to 33 GiB
NEED=${VISTA_NEED_MIB:-32000}          # the job's GPU memory for the runner: the cap + the CUDA context + 2000 MiB
(cd "$PKGDIR" && sha256sum -c --quiet SHA256SUMS) || { echo "$PKGDIR: SHA256SUMS does not match" >&2; exit 2; }
for CH in "$@"; do
  case "$CH" in test-*) [ -f "$WCB_OUTPUTS/bench-v1/protocol-vista-frozen.txt" ] || { echo "$CH: TEST chunks only after the freeze" >&2; exit 2; };; esac
  [ -f "$STOP" ] && { echo "$(date -Is) stop file $STOP: not starting $CH"; exit 3; }
  W=$RUNS/$PKG/$CH
  mkdir -p "$W"
  if [ -f "$W/vista-out-$CH.tar" ] && [ "$(cat "$W/status" 2>/dev/null)" = 0 ]; then echo "$(date -Is) $CH done, skipped"; continue; fi
  split=${CH%%-*}; [ "$CH" = dev-smoke ] && split=dev
  tar -xOf "$PKGDIR/$PKG-$split.tar" "$PKG/chunks/$PKG-$CH.tar" > "$W/$PKG-$CH.tar"
  cp "$HERE/run_vista_bench.py" "$HERE/vista_pins.json" "$W/"
  ln -sfn "$VISTA_CKPT" "$W/vista.safetensors"
  call=0
  while :; do
    call=$((call + 1))
    echo "$(date -Is) $CH: call $call, waiting for the GPU runner (NEED_MIB $NEED)"
    set +e
    (cd "$W" && env NEED_MIB="$NEED" WCB_PACKAGE="$PKG" WCB_PYTHON="$VISTA_PY" VISTA_SRC="$SRC" WCB_CKPT_WINDOWS="${VISTA_LOCAL_CKPT_WINDOWS:-2}" \
       OMP_NUM_THREADS=4 WCB_ADAPTER_ARGS="--leave-free-mib 6144 --max-gpu-mib $CAP --min-free-mib $NEED --min-host-avail-gib 28 --expect-gpu-uuid $GPU_UUID --stop-file $STOP" \
       bash "$WCB_GPU_RUNNER" "$VISTA_PY" "$HERE/guarded_run.py" --log "$W/guard-$call.json" --min-host-avail-gib 4 --pid-file "$W/child.pid" \
         --gpu-uuid "$GPU_UUID" --min-gpu-free-mib 6144 -- bash "$REPO/cluster/vista/run_chunk.sh" "$CH" >> "$W/run.out" 2>> "$W/run.err")
    rc=$?
    set -e
    echo "$rc" > "$W/status"
    echo "$(date -Is) $CH: call $call exit $rc ($W)"
    [ "$rc" = 85 ] && [ ! -f "$STOP" ] && continue     # checkpoint: the next group of windows in a new runner call
    break
  done
  [ "$rc" = 0 ] || exit "$rc"
done
