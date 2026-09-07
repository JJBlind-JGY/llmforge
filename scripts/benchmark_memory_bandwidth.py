"""Measure GPU memory bandwidth using elementwise vector addition."""

from __future__ import annotations

import argparse

import torch

from llmforge.benchmark.cuda_timer import measure_cuda_events, warmup
from llmforge.benchmark.guard import validate_formal_benchmark
from llmforge.benchmark.memory import (
    effective_bandwidth_gbps,
    vector_add_arithmetic_intensity,
    vector_add_bytes,
)
from llmforge.benchmark.statistics import summarize_ms
from llmforge.environment import collect_environment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--elements", type=int, nargs="+", default=[2**20, 2**22, 2**24, 2**26, 2**27]
    )
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)

    args = parser.parse_args()
    environment = collect_environment(role="gpu-server")
    validate_formal_benchmark(environment=environment)

    print(
        f"{'Elements':<16}, {'Size(MiB)':>12}, {'Latency(ms)':>14}, {'Bandwidth(GB/s)':>14}"
    )
    print("-" * 50)
    for elements in args.elements:
        dtype = torch.float32
        x = torch.randn(elements, device="cuda", dtype=dtype)
        y = torch.randn(elements, device="cuda", dtype=dtype)
        out = torch.empty_like(x)

        def operation(x=x, y=y, out=out) -> None:
            torch.add(x, y, out=out)

        warmup(operation=operation, iterations=args.warmups)
        sanmples = measure_cuda_events(operation=operation, iterations=args.iterations)
        stats = summarize_ms(sanmples)

        bytes_transferred = vector_add_bytes(
            elements=elements, element_size_bytes=x.element_size()
        )
        bandwidth = effective_bandwidth_gbps(
            bytes_transfered=bytes_transferred,
            latency_ms=stats.median_ms,
        )
        size_mib = elements * x.element_size() / 1024**2
        print(
            f"{elements:>12}, {size_mib:>12.1f}, {stats.median_ms:>14.4f}, {bandwidth:>12.2f}"
        )
        print(
            f"FP32 vector-add arithmetic intensity: {vector_add_arithmetic_intensity(x.element_size()):.4f} FLOP/Byte"
        )


if __name__ == "__main__":
    main()
