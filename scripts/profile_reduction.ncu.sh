#!/usr/bin/env bash

set -euo pipefail

NCU_BIN="${NCU_BIN:-}"
BINARY="${BINARY:-.build/cuda/llmforge_reduction}"
ELEMENTS="${ELEMENTS:-16777216}"
BLOCK_SIZE="${BLOCK_SIZE:-256}"
WARMUPS="${WARMUPS:-3}"
ITERATIONS="${ITERATIONS:-10}"

if [[ -z "${NCU_BIN}" ]]; then
    echo "NCU_BIN is not set."
    echo "Use an Nsight Compute version compatible with CUDA 12.1."
    exit 1
fi

if [[ ! -x "${NCU_BIN}" ]]; then
    echo "Nsight Compute not executable: ${NCU_BIN}"
    exit 1
fi

if [[ ! -x "${BINARY}" ]]; then
    echo "Reduction binary not found: ${BINARY}"
    exit 1
fi

RUN_DIR="$(
    printf \
        'artifacts/profiling/reduction/%s' \
        "$(date -u +%Y%m%dT%H%M%SZ)"
)"

mkdir -p "${RUN_DIR}"

profile_variant() {
    local variant="$1"
    local kernel="$2"

    echo
    echo "=== Profiling ${variant} ==="

    "${NCU_BIN}" \
        --set full \
        --kernel-name-base function \
        --kernel-name "${kernel}" \
        --launch-skip 4 \
        --launch-count 1 \
        --export "${RUN_DIR}/${variant}" \
        "${BINARY}" \
        "${ELEMENTS}" \
        "${BLOCK_SIZE}" \
        "${variant}" \
        "${WARMUPS}" \
        "${ITERATIONS}"
}

profile_variant \
    shared_interleaved \
    reduce_shared_interleaved_kernel

profile_variant \
    shared \
    reduce_shared_kernel

profile_variant \
    warp \
    reduce_warp_kernel

echo
echo "Reports written to: ${RUN_DIR}"