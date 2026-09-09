"""Triton FP32 IEEE matrix multiplication."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_AUTOTUNE_CONFIGS = [
    triton.Config(
        {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
        },
        num_warps=4,
        num_stages=4,
    ),
    triton.Config(
        {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
        },
        num_warps=4,
        num_stages=4,
    ),
    triton.Config(
        {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
        },
        num_warps=4,
        num_stages=4,
    ),
    triton.Config(
        {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
        },
        num_warps=2,
        num_stages=5,
    ),
    triton.Config(
        {
            "BLOCK_SIZE_M": 32,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
        },
        num_warps=2,
        num_stages=5,
    ),
]


@triton.autotune(configs=_AUTOTUNE_CONFIGS, key=["M", "N", "K"])
@triton.jit
def _matmul_bf16_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)

    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k_block in range(tl.cdiv(K, BLOCK_SIZE_K)):
        current_k = k_block * BLOCK_SIZE_K + offs_k

        a_mask = (offs_m[:, None] < M) & (current_k[None, :] < K)
        b_mask = (current_k[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, a_mask, other=0.0)
        b = tl.load(b_ptrs, b_mask, other=0.0)

        accumulator = tl.dot(a, b, accumulator)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def matmul_bf16_into(a: torch.Tensor, b: torch.Tensor, output: torch.Tensor) -> None:
    """Compute BF16 GEMM with FP32 accumulation."""

    if not (a.is_cuda and b.is_cuda and output.is_cuda):
        raise ValueError("All tensors must be CUDA tensors.")
    if not (a.dtype == b.dtype == output.dtype == torch.bfloat16):
        raise ValueError("BF16 kernel requires bfloat16 tensors.")

    if not (a.ndim == 2 and b.ndim == 2 and output.ndim == 2):
        raise ValueError("All tensors must be 2-D.")

    m, k = a.shape
    b_k, n = b.shape

    if k != b_k:
        raise ValueError("Incompatible matrix dimensions.")

    if output.shape != (m, n):
        raise ValueError("Output shape must be (M, N).")

    if not (a.is_contiguous() and b.is_contiguous() and output.is_contiguous()):
        raise ValueError("Current kernel requires contiguous tensors.")

    grid = lambda meta: (
        triton.cdiv(
            m,
            meta["BLOCK_SIZE_M"],
        )
        * triton.cdiv(n, meta["BLOCK_SIZE_N"]),
    )

    _matmul_bf16_kernel[grid](
        a,
        b,
        output,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        output.stride(0),
        output.stride(1),
    )


def get_best_bf16_matmul_config() -> dict[str, object] | None:
    """Return selected BF16 Triton GEMM configuration."""

    config = getattr(_matmul_bf16_kernel, "best_config", None)

    if config is None:
        return None

    return {
        **dict(config.kwargs),
        "num_warps": config.num_warps,
        "num_stages": config.num_stages,
    }


@triton.autotune(configs=_AUTOTUNE_CONFIGS, key=["M", "N", "K"])
@triton.jit
def _matmul_fp32_ieee_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)

    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k_block in range(tl.cdiv(K, BLOCK_SIZE_K)):
        current_k = k_block * BLOCK_SIZE_K + offs_k

        a_mask = (offs_m[:, None] < M) & (current_k[None, :] < K)
        b_mask = (current_k[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, a_mask, other=0.0)
        b = tl.load(b_ptrs, b_mask, other=0.0)

        accumulator = tl.dot(a, b, accumulator, input_precision="ieee")

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def matmul_fp32_ieee_into(
    a: torch.Tensor, b: torch.Tensor, output: torch.Tensor
) -> None:
    """Compute output = a @ b using Triton FP32 IEEE."""

    if not (a.is_cuda and b.is_cuda and output.is_cuda):
        raise ValueError("All tensors must be CUDA tensors.")

    if not (a.dtype == b.dtype == output.dtype == torch.float32):
        raise ValueError("FP32 IEEE kernel requires float32 tensors.")

    if a.ndim != 2 or b.ndim != 2 or output.ndim != 2:
        raise ValueError("All tensors must be 2-D.")

    m, k = a.shape
    b_k, n = b.shape

    if k != b_k:
        raise ValueError("Incompatible matrix dimensions.")

    if output.shape != (m, n):
        raise ValueError("Output tensor shape must match (m, n).")

    if not a.is_contiguous() or not b.is_contiguous() or not output.is_contiguous():
        raise ValueError("Current tensors requires contiguous tensors.")

    # IMPORTANT:
    # This is host-side Python code, so use triton.cdiv,
    # not tl.cdiv.
    #
    # The kernel reads only tl.program_id(axis=0),
    # therefore all output tiles must be flattened
    # into one 1-D program grid.
    grid = lambda meta: (
        triton.cdiv(m, meta["BLOCK_SIZE_M"]) * triton.cdiv(n, meta["BLOCK_SIZE_N"]),
    )
    _matmul_fp32_ieee_kernel[grid](
        a,
        b,
        output,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        output.stride(0),
        output.stride(1),
    )


def get_best_matmul_config() -> dict[str, object] | None:
    """Return the most recently selected autotune configuration."""

    config = getattr(_matmul_fp32_ieee_kernel, "best_config", None)

    if config is None:
        return None

    return {
        **dict(config.kwargs),
        "num_warps": config.num_warps,
        "num_stages": config.num_stages,
    }
