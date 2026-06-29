#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

mkdir -p .cache/uv .cache/pip .cache/huggingface .local artifacts models runs data/fineweb10B
export XDG_CACHE_HOME="${REPO_ROOT}/.cache"
export UV_CACHE_DIR="${REPO_ROOT}/.cache/uv"
export PIP_CACHE_DIR="${REPO_ROOT}/.cache/pip"
export HF_HOME="${REPO_ROOT}/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="${REPO_ROOT}/.cache/huggingface/hub"
export TORCH_HOME="${REPO_ROOT}/.cache/torch"
export TRITON_CACHE_DIR="${REPO_ROOT}/.cache/triton"
export CUDA_CACHE_PATH="${REPO_ROOT}/.cache/nv"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -x /opt/conda/bin/python ]]; then
    PYTHON_BIN=/opt/conda/bin/python
  else
    PYTHON_BIN="$(command -v python3)"
  fi
fi

if ! command -v uv >/dev/null 2>&1; then
  "${PYTHON_BIN}" -m pip install --prefix "${REPO_ROOT}/.local" uv
  export PATH="${REPO_ROOT}/.local/bin:${PATH}"
fi

if [[ ! -x .venv/bin/python ]]; then
  uv venv --system-site-packages --python "${PYTHON_BIN}" .venv
fi

uv pip install --python .venv/bin/python --requirements requirements.lock.txt

.venv/bin/python - <<'PY'
import importlib
import torch

missing = []
for name in ["numpy", "huggingface_hub", "kernels", "tqdm"]:
    try:
        importlib.import_module(name)
    except Exception as exc:
        missing.append(f"{name}: {exc}")

print("python", __import__("sys").version.split()[0])
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
if missing:
    raise SystemExit("missing dependencies: " + "; ".join(missing))
PY

