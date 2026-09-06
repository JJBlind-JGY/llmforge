"""GEMM performance model and benchmark runner."""

from __future__ import annotations

from typing import Any

from llmforge.benchmark.cuda_timer import (
    measure_cpu_naive,
    measure_cpu_synchronized,
    measure_cuda_events,
    warmup,
)
from llmforge.benchmark.statistics import summarize_ms


def gemm_flops(m: int, n: int, k: int) -> int:
    """Return the conventional GEMM FLOP count."""
    return 2 * m * n * k


def gemm_minimum_bytes(m: int, n: int, k: int, element_size_bytes: int) -> int:
    """Estimate minimum GEMM memory traffic.

    Assumes A and B are each read once and C is written once.
    """
    elements = m * k + k * n + m * n

    return element_size_bytes * elements


def gemm_arithmetic_intensity(m: int, n: int, k: int, element_size_bytes: int) -> float:
    """Return idealized GEMM arithmetic intensity."""
    return gemm_flops(m, n, k) / gemm_minimum_bytes(m, n, k, element_size_bytes)


def effective_tflops(flops: int, latency_ms: float) -> float:
    """Compute effective TFLOPS from operation count and latency."""
    seconds = latency_ms / 1000.0

    return flops / seconds / 1e12


def run_square_gemm(size: int, warmups: int, iterations: int) -> dict[str, Any]:
    """Run a square FP32 GEMM timing benchmark."""
    import torch
    from torch.utils import benchmark

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    device = torch.device("cuda:0")
    dtype = torch.float32

    torch.manual_seed(0)

    a = torch.randn(size, size, device=device, dtype=dtype)
    b = torch.randn(size, size, device=device, dtype=dtype)

    out = torch.empty_like(a)

    def operation() -> None:
        torch.mm(a, b, out=out)

    warmup(operation, warmups)

    naive_samples = measure_cpu_naive(operation=operation, iterations=iterations)
    synchronized_samples = measure_cpu_synchronized(
        operation=operation, iterations=iterations
    )
    event_samples = measure_cuda_events(operation=operation, iterations=iterations)

    torch_timer = benchmark.Timer(
        stmt="torch.mm(a, b, out=out)",
        globals={"torch": torch, "a": a, "b": b, "out": out},
    )

    measurement = torch_timer.blocked_autorange(min_run_time=1.0)

    flops = gemm_flops(size, size, size)
    element_size = torch.empty((), dtype=dtype).element_size()

    minimum_bytes = gemm_minimum_bytes(size, size, size, element_size)
    arithmetic_intensity = gemm_arithmetic_intensity(size, size, size, element_size)

    naive_stats = summarize_ms(naive_samples)
    synchronized_stats = summarize_ms(synchronized_samples)
    event_stats = summarize_ms(event_samples)

    return {
        "shape": {"m": size, "n": size, "k": size},
        "dtype": str(dtype),
        "theory": {
            "flops": flops,
            "minimum_bytes": minimum_bytes,
            "arithmetic_intensity_flops_per_byte": arithmetic_intensity,
        },
        "precision": {
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_fp32_precision": getattr(
                torch.backends.cuda.matmul, "fp32_precision", None
            ),
        },
        "timing": {
            "cpu_naive": naive_stats.to_dict(),
            "cpu_synchronized": synchronized_stats.to_dict(),
            "cuda_event": event_stats.to_dict(),
            "torch_benchmark": {"median_ms": measurement.median * 1000.0},
        },
        "performance": {
            "cuda_event_effective_tflops": effective_tflops(
                flops, event_stats.median_ms
            ),
        },
        "raw_samples_ms": {
            "cpu_naive": naive_samples,
            "cpu_synchronized": synchronized_samples,
            "cuda_event": event_samples,
        },
    }
