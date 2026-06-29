#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

MODE="${1:-smoke}"
METHOD="${2:-muon1}"

case "${METHOD}" in
  adam1) TRAIN_SCRIPT="train_gpt_adam_1.py"; LAUNCHER="python" ;;
  muon1) TRAIN_SCRIPT="train_gpt_muon_1.py"; LAUNCHER="python" ;;
  newton_muon1) TRAIN_SCRIPT="train_gpt_newton_muon_1.py"; LAUNCHER="python" ;;
  adam2) TRAIN_SCRIPT="train_gpt_adam_2.py"; LAUNCHER="torchrun" ;;
  muon2) TRAIN_SCRIPT="train_gpt_muon_2.py"; LAUNCHER="torchrun" ;;
  newton_muon2) TRAIN_SCRIPT="train_gpt_newton_muon_2.py"; LAUNCHER="torchrun" ;;
  *) echo "unknown method: ${METHOD}" >&2; exit 2 ;;
esac

mkdir -p .cache/uv .cache/pip .cache/huggingface .cache/triton .cache/nv runs artifacts models data/fineweb10B experiments
export XDG_CACHE_HOME="${REPO_ROOT}/.cache"
export UV_CACHE_DIR="${REPO_ROOT}/.cache/uv"
export PIP_CACHE_DIR="${REPO_ROOT}/.cache/pip"
export HF_HOME="${REPO_ROOT}/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="${REPO_ROOT}/.cache/huggingface/hub"
export TORCH_HOME="${REPO_ROOT}/.cache/torch"
export TRITON_CACHE_DIR="${REPO_ROOT}/.cache/triton"
export CUDA_CACHE_PATH="${REPO_ROOT}/.cache/nv"
export PYTHONUNBUFFERED=1

if [[ ! -x .venv/bin/python ]]; then
  ./setup.sh
fi

GIT_SHA="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
RUN_ID="${RUN_ID:-${STARTED_AT//[:]/-}-${METHOD}-${MODE}-${GIT_SHA:0:8}}"
RUN_DIR="${REPO_ROOT}/runs/${RUN_ID}"
SNAPSHOT_DIR="${RUN_DIR}/source_snapshot"
mkdir -p "${SNAPSHOT_DIR}" "${RUN_DIR}/data/fineweb10B"

if [[ "${MODE}" == "smoke" ]]; then
  .venv/bin/python scripts/make_tiny_fineweb.py --out-dir "${RUN_DIR}/data/fineweb10B"
elif [[ "${MODE}" == "full" ]]; then
  .venv/bin/python scripts/check_fineweb.py --data-dir data/fineweb10B --chunks "${CHUNKS:-50}"
else
  echo "unknown mode: ${MODE}" >&2
  exit 2
fi

.venv/bin/python scripts/materialize_train.py \
  --source "${TRAIN_SCRIPT}" \
  --out "${SNAPSHOT_DIR}/train.py" \
  --mode "${MODE}" \
  --method "${METHOD}" \
  --repo-root "${REPO_ROOT}" \
  --run-dir "${RUN_DIR}"
cp triton_kernels.py "${SNAPSHOT_DIR}/triton_kernels.py"

.venv/bin/python scripts/write_metadata.py \
  --run-dir "${RUN_DIR}" \
  --run-id "${RUN_ID}" \
  --method "${METHOD}" \
  --mode "${MODE}" \
  --launcher "${LAUNCHER}" \
  --source "${TRAIN_SCRIPT}" \
  --git-sha "${GIT_SHA}" \
  --started-at "${STARTED_AT}"

echo "run_id=${RUN_ID}"
echo "run_dir=${RUN_DIR}"
echo "git_sha=${GIT_SHA}"

set +e
if [[ "${LAUNCHER}" == "torchrun" ]]; then
  (
    cd "${SNAPSHOT_DIR}"
    PYTHONPATH="${SNAPSHOT_DIR}:${REPO_ROOT}" "${REPO_ROOT}/.venv/bin/python" -m torch.distributed.run --standalone --nproc_per_node=1 train.py
  ) 2>&1 | tee "${RUN_DIR}/stdout.log"
  STATUS="${PIPESTATUS[0]}"
else
  (
    cd "${SNAPSHOT_DIR}"
    PYTHONPATH="${SNAPSHOT_DIR}:${REPO_ROOT}" "${REPO_ROOT}/.venv/bin/python" train.py
  ) 2>&1 | tee "${RUN_DIR}/stdout.log"
  STATUS="${PIPESTATUS[0]}"
fi
set -e

.venv/bin/python scripts/finalize_run.py --run-dir "${RUN_DIR}" --status-code "${STATUS}"
exit "${STATUS}"
