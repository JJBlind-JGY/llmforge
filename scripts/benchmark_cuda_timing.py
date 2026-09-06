"""Compare common CUDA timing methods on GPU matrix multiplication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils import benchmark

from llmforge.benchmark.cuda_timer import (
    measure_cpu_naive,
    measure_cpu_synchronized,
    measure_cuda_events,
    warmup,
)
from llmforge.benchmark.statistics import summarize_ms
from llmforge.environment import collect_environment


def effective_tflops(size: int, latency_ms: float) -> float:
    flops = 2 * size**3
    seconds = latency_ms / 1000.0

    return flops / seconds / 1e12


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--size", type=int, default=4096)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/benchmarks/cuda_timing.json")
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    torch.manual_seed(0)

    device = torch.device("cuda:0")

    a = torch.randn(args.size, args.size, device=device, dtype=torch.float32)
    b = torch.randn(args.size, args.size, device=device, dtype=torch.float32)

    out = torch.empty_like(a)

    def operation() -> None:
        torch.mm(a, b, out=out)

    warmup(operation, args.warmups)

    naive_samples = measure_cpu_naive(operation=operation, iterations=args.iterations)
    synchronized_samples = measure_cpu_synchronized(
        operation=operation, iterations=args.iterations
    )
    event_samples = measure_cuda_events(operation=operation, iterations=args.iterations)

    torch_timer = benchmark.Timer(
        stmt="torch.mm(a, b, out=out)",
        globals={"torch": torch, "a": a, "b": b, "out": out},
    )

    torch_measurement = torch_timer.blocked_autorange(min_run_time=1.0)

    naive_stats = summarize_ms(naive_samples)
    synchronized_stats = summarize_ms(synchronized_samples)
    event_stats = summarize_ms(event_samples)

    event_tflops = effective_tflops(args.size, event_stats.median_ms)

    result = {
        "experiment": "cuda_timing_method_comparison",
        "config": {
            "matrix_size": args.size,
            "dtype": str(a.dtype),
            "warmup": args.warmups,
            "iteration": args.iterations,
        },
        "results": {
            "cpu_naive": naive_stats.to_dict(),
            "cpu_synchronized": synchronized_stats.to_dict(),
            "cuda_event": event_stats.to_dict(),
            "torch_benchmark": {
                "median_ms": torch_measurement.median_ms * 1000.0,
            },
            "cuda_event_effective_tflops": event_tflops,
        },
        "environment": collect_environment(role="gpu_server"),
    }

    print(json.dumps(result["results"], indent=2))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"Benchmark written to: {args.output}")


if __name__ == "__main__":
    main()
