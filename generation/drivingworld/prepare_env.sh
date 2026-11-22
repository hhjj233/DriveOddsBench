#!/usr/bin/env bash
# Hash-locked DrivingWorld inference environment: CPython 3.10, torch 2.1.0+cu121 (self-contained CUDA 12.1 wheel, kernels
# up to sm_90, so the same environment runs the CPU checks here and the GPU jobs on H100/H200).
#   prepare_env.sh UV PYTHON310 NEW_VENV
# Every wheel is checked against requirements-drivingworld.lock.txt (--require-hashes, --no-deps); the PyTorch index
# serves torch, PyPI the rest. The installed manifest is written into the environment.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
UV=${1:?usage: prepare_env.sh UV PYTHON310 NEW_VENV}
PY310=${2:?usage: prepare_env.sh UV PYTHON310 NEW_VENV}
VENV=${3:?usage: prepare_env.sh UV PYTHON310 NEW_VENV}
[[ ! -e "$VENV" ]] || { echo 'fresh environment directory required' >&2; exit 1; }
"$PY310" -B -c 'import sys; assert sys.version_info[:2] == (3, 10), sys.version'
"$UV" --no-config venv --python "$PY310" "$VENV"
"$UV" --no-config pip install --python "$VENV/bin/python" --require-hashes --no-deps --index-strategy unsafe-best-match \
  --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu121 \
  -r "$HERE/requirements-drivingworld.lock.txt"
"$UV" --no-config pip freeze --python "$VENV/bin/python" > "$VENV/installed-manifest.txt"
"$VENV/bin/python" -B -c 'import torch, einops, numpy; print(torch.__version__, torch.version.cuda, einops.__version__, numpy.__version__)'
