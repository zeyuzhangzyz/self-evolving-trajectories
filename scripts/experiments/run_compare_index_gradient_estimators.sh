#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_TAG="${RUN_TAG:-direct_vs_batch_shared_k4_16_81_v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results/index_weighting_sampling}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_TAG}}"

if [[ -e "${OUTPUT_DIR}" ]]; then
    echo "Refusing to overwrite existing output directory: ${OUTPUT_DIR}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"
export CUDA_VISIBLE_DEVICES=""

COMMAND=(
    "${PYTHON_BIN}"
    "${SCRIPT_DIR}/compare_index_gradient_estimators.py"
    --k-values 4,16,81
    --draw-counts 100,1000,10000,100000
    --problem-seeds 3
    --sampling-seeds 20
    --batch-size 8
    --quiz-size 2
    --value-vocab-size 11
    --n-layer 1
    --n-head 1
    --n-embd 16
    --threads 2
    --dtype float32
    --device cpu
    --output-dir "${OUTPUT_DIR}"
)

export RUN_COMMAND_ORIGINAL="CUDA_VISIBLE_DEVICES='' ${COMMAND[*]}"
printf 'Running: %s\n' "${RUN_COMMAND_ORIGINAL}"
"${COMMAND[@]}" 2>&1 | tee "${OUTPUT_DIR}/run.log"
