#!/usr/bin/env bash
# Explicit large-install step.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
UV=${1:?usage: prepare_env.sh UV PYTHON310 SOURCE_CHECKOUT NEW_VENV}
PY310=${2:?}; SRC=${3:?}; VENV=${4:?}
[[ ! -e "$VENV" ]] || { echo 'Fresh environment directory required' >&2; exit 1; }
"$PY310" -B -c 'import sys; assert sys.version_info[:2]==(3,10)'
"$PY310" -B -c 'import sys;sys.path.insert(0,sys.argv[1]);import drivelaw_common as d; print(d.verify_source(sys.argv[2]));d.verify_runtime_complete(sys.argv[2])' "$HERE" "$SRC"
"$UV" --no-config venv --python "$PY310" "$VENV"
"$UV" --no-config pip install --python "$VENV/bin/python" --require-hashes --no-deps --default-index https://pypi.org/simple --torch-backend cu124 -r "$HERE/requirements-drivelaw-video.lock.txt"
"$UV" --no-config pip install --python "$VENV/bin/python" --no-deps --no-build-isolation "$SRC/DriveLaW-Video/Infer/diffusers"
"$UV" --no-config pip freeze --python "$VENV/bin/python" > "$VENV/installed-manifest.txt"
"$VENV/bin/python" -B -c 'import torch,transformers,diffusers; print(torch.__version__,transformers.__version__,diffusers.__version__)'
