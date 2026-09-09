#!/usr/bin/env bash

set -euo pipefail

NCU_BIN="${NCU_BIN:-}"
BINARY="${BINARY:-.build/cuda/llmforge_matmul}"
N="${N:-4096}"
WARMUPS="${WARMUPS:-5}"
ITERATIONS="${ITERATIONS:-20}"

if [[ -z "${NCU_BIN}" ]]; then
    echo "NCU_BIN is not set."
    exit 1
fi

if [[ ! -x "${NCU_BIN}" ]]; then
    echo "Nsight Compute not executable: ${NCU_BIN}"
    exit 1
fi

if [[ ! -x "${BINARY}" ]]; then
    echo "MatMul binary not found: ${BINARY}"
    exit 1
fi

RUN_DIR="$(
    printf \
        'artifacts/profiling/matmul/%s' \
        "$(date -u +%Y%m%dT%H%M%SZ)"
)"

mkdir -p "${RUN_DIR}"

"${NCU_BIN}" --version \
    > "${RUN_DIR}/ncu_version.txt"

git rev-parse HEAD \
    > "${RUN_DIR}/git_commit.txt"

profile_variant() {
    local variant="$1"
    local kernel_regex="$2"

    echo
    echo "=== Profiling ${variant} ==="

    "${NCU_BIN}" \
        --section SpeedOfLight \
        --section MemoryWorkloadAnalysis \
        --section Occupancy \
        --section WarpStateStats \
        --kernel-name-base function \
        --kernel-name "regex:${kernel_regex}" \
        --launch-skip 6 \
        --launch-count 1 \
        --export "${RUN_DIR}/${variant}" \
        "${BINARY}" \
        "${N}" \
        "${variant}" \
        "${WARMUPS}" \
        "${ITERATIONS}"
}

profile_variant \
    naive \
    'matmul_naive_kernel.*'

profile_variant \
    tiled16 \
    'matmul_tiled_kernel.*'

profile_variant \
    tiled32 \
    'matmul_tiled_kernel.*'

echo
echo "Reports written to: ${RUN_DIR}"