"""Triton vector addition kernel."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _vector_add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    program_id = tl.program_id(axis=0)
    block_start = program_id * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def vector_add_into(
    x: torch.Tensor,
    y: torch.Tensor,
    output: torch.Tensor,
    *,
    block_size: int = 1024,
) -> None:
    """Launch Triton vector addition into a preallocated tensor."""
    if not (x.is_cuda and y.is_cuda and output.is_cuda):
        raise ValueError("All tensors must be CUDA tensors.")

    if x.device != y.device or x.device != output.device:
        raise ValueError("All tensors must be on the same device.")

    if x.dtype != y.dtype or x.dtype != output.dtype:
        raise ValueError("All tensors must have the same dtype.")

    if x.numel() != y.numel() or x.numel() != output.numel():
        raise ValueError("Tensor sizes must match.")

    if not (x.is_contiguous() and y.is_contiguous() and output.is_contiguous()):
        raise ValueError("Current kernel requires contiguous tensors.")

    if block_size <= 0 or block_size & (block_size - 1):
        raise ValueError("block_size must be a power of two.")

    n_elements = output.numel()
    grid = triton.cdiv(n_elements, block_size)
    _vector_add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=block_size)
