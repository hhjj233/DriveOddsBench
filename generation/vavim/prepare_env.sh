#!/usr/bin/env bash
# Hash-locked VaViM inference environment: CPython 3.10 (installed inside the environment's own directory, so the whole
# tree can be copied to another machine at the same absolute path), torch 2.4.0 (PyPI CUDA 12.1 build, kernels to sm_90).
#   prepare_env.sh UV ENV_ROOT
# creates ENV_ROOT/python (uv-managed CPython 3.10.19) and ENV_ROOT/venv. Every wheel or archive is checked against
# requirements-vavim.lock.txt (--require-hashes, --no-deps); the only source build is the pinned mup archive. The installed
# manifest is written to ENV_ROOT/venv/installed-manifest.txt. The official VideoActionModel source is NOT installed: the
# runner puts the verified checkout on sys.path (the README's `pip install -e .` would add only the same package).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
UV=${1:?usage: prepare_env.sh UV ENV_ROOT}
ROOT=${2:?usage: prepare_env.sh UV ENV_ROOT}
[[ "$ROOT" = /* ]] || { echo 'absolute ENV_ROOT required' >&2; exit 1; }
[[ ! -e "$ROOT/venv" && ! -e "$ROOT/python" ]] || { echo 'fresh environment directory required' >&2; exit 1; }
export UV_PYTHON_INSTALL_DIR="$ROOT/python" UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT/uv-cache}"
"$UV" --no-config python install --no-bin 3.10.19
PY310="$ROOT/python/cpython-3.10.19-linux-x86_64-gnu/bin/python3.10"
"$PY310" -B -c 'import sys; assert sys.version_info[:3] == (3, 10, 19), sys.version'
"$UV" --no-config venv --python "$PY310" "$ROOT/venv"
"$UV" --no-config pip install --python "$ROOT/venv/bin/python" --require-hashes --no-deps \
  --index-url https://pypi.org/simple -r "$HERE/requirements-vavim.lock.txt"
"$UV" --no-config pip freeze --python "$ROOT/venv/bin/python" > "$ROOT/venv/installed-manifest.txt"
"$ROOT/venv/bin/python" -B -c 'import torch, torchvision, einops, mup, lightning, hydra, numpy; print(torch.__version__, torch.version.cuda, torchvision.__version__, einops.__version__, lightning.__version__, hydra.__version__, numpy.__version__)'
