#!/usr/bin/env bash
# Explicit large-install step.
# Creates a fresh CPython 3.10 environment from the exact hash lock. The official ReWorld source is
# verified but NOT installed: the runner puts <source>/src on sys.path, exactly as the shipped
# scripts/sample_dualflow_dit.py does, so no build step or extra dependency resolution happens here.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
UV=${1:?usage: prepare_env.sh UV PYTHON310 SOURCE_CHECKOUT NEW_VENV}
PY310=${2:?}; SRC=${3:?}; VENV=${4:?}
[[ ! -e "$VENV" ]] || { echo 'Fresh environment directory required' >&2; exit 1; }
"$PY310" -B -c 'import sys; assert sys.version_info[:2]==(3,10)'
"$PY310" -B -c 'import sys;sys.path.insert(0,sys.argv[1]);import reworld_common as r; print(r.verify_source(sys.argv[2]))' "$HERE" "$SRC"
"$UV" --no-config venv --python "$PY310" "$VENV"
"$UV" --no-config pip install --python "$VENV/bin/python" --require-hashes --no-deps --default-index https://pypi.org/simple --torch-backend cu124 -r "$HERE/requirements-reworld-video.lock.txt"
"$UV" --no-config pip freeze --python "$VENV/bin/python" > "$VENV/installed-manifest.txt"
"$VENV/bin/python" -B -c 'import torch,torchvision,transformers,diffusers,decord; print(torch.__version__,torchvision.__version__,transformers.__version__,diffusers.__version__,decord.__version__)'
