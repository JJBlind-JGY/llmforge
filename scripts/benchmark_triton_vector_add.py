"""Benchmark Triton VectorAdd against PyTorch."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import triton

from llmforge.benchmark.cuda_timer import measure_cuda_events, warmup
from llmforge.benchmark.experiment import create_run_directory, write_json
from llmforge.benchmark.guard import validate_formal_benchmark
from llmforge.benchmark.memory import effective_bandwidth_gbps, vector_add_bytes
from llmforge.benchmark.statistics import summarize_ms
from llmforge.environment import collect_environment
from llmforge.kernels.triton.vector_add import vector_add_into


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--elements", type=int, nargs="+", default=[2**20, 2**22, 2**24, 2**26, 2**27]
    )
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--output-root", type=Path, default=Path("artifacts/benchmarks")
    )

    args = parser.parse_args()
    environment = collect_environment("gpu-server")
    validate_formal_benchmark(environment=environment)

    results = []
    print(f"{'Elements':>12}{'Provider':>12}{'Latency(ms)':>14}{'GB/s':>12}")
    print("-" * 50)

    for elements in args.elements:
        x = torch.randn(elements, device="cuda", dtype=torch.float32)
        y = torch.randn(elements, device="cuda", dtype=torch.float32)
        triton_output = torch.empty_like(x, device="cuda")
        torch_output = torch.empty_like(x, device="cuda")

        # Trigger JIT compilation and verify correctness
        vector_add_into(x, y, triton_output, block_size=args.block_size)
        torch.add(x, y, out=torch_output)
        torch.cuda.synchronize()

        torch.testing.assert_close(triton_output, torch_output, rtol=1e-5, atol=1e-6)

        def triton_operation(x=x, y=y, triton_output=triton_output) -> None:
            vector_add_into(x, y, triton_output, block_size=args.block_size)

        def torch_operation(x=x, y=y, torch_output=torch_output) -> None:
            torch.add(x, y, out=torch_output)

        providers = {
            "triton": triton_operation,
            "torch": torch_operation,
        }

        for provider, operation in providers.items():
            warmup(operation=operation, iterations=args.warmups)
            samples = measure_cuda_events(
                operation=operation, iterations=args.iterations
            )
            stats = summarize_ms(samples)
            transferred = vector_add_bytes(
                elements=elements, element_size_bytes=x.element_size()
            )
            bandwidth = effective_bandwidth_gbps(transferred, stats.median_ms)

            result = {
                "elements": elements,
                "provider": provider,
                "median_ms": stats.median_ms,
                "bandwidth_gbps": bandwidth,
                "raw_sample_ms": samples,
            }
            results.append(result)
            print(
                f"{elements:>12}"
                f"{provider:>12}"
                f"{stats.median_ms:>14.4f}"
                f"{bandwidth:>12.2f}"
            )

    commit = environment["project"]["git"]["commit"]
    run_directory = create_run_directory(
        args.output_root,
        "triton_vector_add",
        commit,
    )

    payload = {
        "experiment": ("triton_vector_add"),
        "config": {
            "elements": args.elements,
            "block_size": (args.block_size),
            "warmups": args.warmups,
            "iterations": (args.iterations),
        },
        "software": {
            "triton": (triton.__version__),
        },
        "environment": environment,
        "results": results,
    }

    result_path = run_directory / "result.json"
    write_json(result_path, payload)
    print(f"Result written to: {result_path}")


if __name__ == "__main__":
    main()
