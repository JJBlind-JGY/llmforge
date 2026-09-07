"""Benchmark GEMM precision modes on CUDA."""

from __future__ import annotations

import argparse

import torch

from llmforge.benchmark.cuda_timer import measure_cuda_events, warmup
from llmforge.benchmark.gemm import gemm_flops
from llmforge.benchmark.statistics import summarize_ms


def run_gemmm(
    size: int, dtype: torch.dtype, iterations: int, warmups: int
) -> tuple[float, float]:
    a = torch.randn(size, size, device="cuda", dtype=dtype)
    b = torch.randn(size, size, device="cuda", dtype=dtype)
    out = torch.empty_like(a)

    def operation():
        torch.mm(a, b, out=out)

    warmup(operation=operation, iterations=warmups)
    samples = measure_cuda_events(operation=operation, iterations=iterations)
    stats = summarize_ms(samples=samples)

    flops = gemm_flops(size, size, size)
    tflops = flops / (stats.median_ms * 1e-3) / 1e12

    return stats.median_ms, tflops


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=50)

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    cases = []

    # IEEE 754 single-precision float
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    latency, tflops = run_gemmm(args.size, torch.float32, args.iterations, 20)
    cases.append(("fp32_ieee", latency, tflops))

    # TF32 single-precision float
    torch.backends.cuda.matmul.fp32_precision = "tf32"
    latency, tflops = run_gemmm(args.size, torch.float32, args.iterations, 20)
    cases.append(("tf32", latency, tflops))

    # BF16
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    latency, tflops = run_gemmm(args.size, torch.bfloat16, args.iterations, 20)
    cases.append(("bf16", latency, tflops))

    print(f"{'Mode':<16}, {'Latency(ms)':>14}, {'TFLOPS':>14}")

    print("-" * 44)

    for mode, latency, tflops in cases:
        print(f"{mode:<16}, {latency:>14.4f}, {tflops:>14.2f}")


if __name__ == "__main__":
    main()
