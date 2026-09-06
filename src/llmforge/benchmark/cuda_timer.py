"""CUDA timing utilities for LLMForge GPU benchmarks."""

from __future__ import annotations

import time
from collections.abc import Callable


def _require_cuda():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for CUDA benchmarks."
            "Install LLMForge with the GPU extra."
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    return torch


def warmup(operation: Callable[[], object], iterations: int) -> None:
    """Warm up a CUDA operation before collecting measurements."""
    torch = _require_cuda()

    for _ in range(iterations):
        operation()

    torch.cuda.synchronize()


def measure_cpu_naive(operation: Callable[[], object], iterations: int) -> list[float]:
    """Measure CPU-side enqueue time without CUDA synchronization."""
    samples: list[float] = []

    for _ in range(iterations):
        start = time.perf_counter()
        operation()
        end = time.perf_counter()

        samples.append((end - start) * 1000.0)

    _require_cuda().cuda.synchronize()

    return samples


def measure_cpu_synchronized(
    operation: Callable[[], object], iterations: int
) -> list[float]:
    """Measure wall time while synchronizing after each CUDA operation."""
    torch = _require_cuda()
    samples: list[float] = []

    torch.cuda.synchronize()

    for _ in range(iterations):
        start = time.perf_counter()

        operation()
        torch.cuda.synchronize()

        end = time.perf_counter()

        samples.append((end - start) * 1000.0)

    return samples


def measure_cuda_events(
    operation: Callable[[], object], iterations: int
) -> list[float]:
    """Measure device elapsed time with CUDA events."""
    torch = _require_cuda()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]

    for start_event, end_event in zip(start_events, end_events, strict=True):
        start_event.record()
        operation()
        end_event.record()

    torch.cuda.synchronize()

    return [
        start.elapsed_time(end)
        for start, end in zip(start_events, end_events, strict=True)
    ]
