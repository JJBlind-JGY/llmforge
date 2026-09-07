#!/usr/bin/env bash

set -euo pipefail

CUDA_ROOT="${CUDA_ROOT:-/usr/local/cuda-12.1}"
CUDA_ARCH="${CUDA_ARCH:-89}"
CUDA_HOST_CXX="${CUDA_HOST_CXX:-$(command -v g++-11 || true)}"
BUILD_DIR="${BUILD_DIR:-.build/cuda}"
BUILD_JOBS="${BUILD_JOBS:-$(nproc)}"

if [[ ! -x "${CUDA_ROOT}/bin/nvcc" ]]; then
    echo "CUDA compiler not found: ${CUDA_ROOT}/bin/nvcc"
    exit 1
fi

if [[ -z "${CUDA_HOST_CXX}" ]]; then
    echo "No compatible CUDA host C++ compiler found."
    exit 1
fi

echo "CUDA_ROOT=${CUDA_ROOT}"
echo "CUDA_ARCH=${CUDA_ARCH}"
echo "CUDA_HOST_CXX=${CUDA_HOST_CXX}"
echo "BUILD_DIR=${BUILD_DIR}"

cmake \
    -S kernels/cuda \
    -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER="${CUDA_HOST_CXX}" \
    -DCMAKE_CUDA_COMPILER="${CUDA_ROOT}/bin/nvcc" \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}"

cmake --build \
    "${BUILD_DIR}" \
    -j "${BUILD_JOBS}"
