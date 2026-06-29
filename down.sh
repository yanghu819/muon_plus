#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

mkdir -p .cache/huggingface .cache/uv .cache/pip data/fineweb10B
export XDG_CACHE_HOME="${REPO_ROOT}/.cache"
export UV_CACHE_DIR="${REPO_ROOT}/.cache/uv"
export PIP_CACHE_DIR="${REPO_ROOT}/.cache/pip"
export HF_HOME="${REPO_ROOT}/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="${REPO_ROOT}/.cache/huggingface/hub"
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

CHUNKS="${CHUNKS:-50}"
PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
  ./setup.sh
fi

"${PYTHON}" data/cached_fineweb10B.py "${CHUNKS}"
"${PYTHON}" scripts/check_fineweb.py --data-dir data/fineweb10B --chunks "${CHUNKS}"

