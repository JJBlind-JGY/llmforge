"""Benchmark native CUDA GEMM implementations."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from llmforge.benchmark.experiment import create_run_directory, write_json
from llmforge.benchmark.guard import validate_formal_benchmark
from llmforge.benchmark.native import parse_key_value_output
from llmforge.environment import collect_environment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--binary", type=Path, default=Path(".build/cuda/llmforge_matmul")
    )
    parser.add_argument("--sizes", type=int, nargs="+", default=[256, 512, 1024, 2048])
    parser.add_argument(
        "--variants", type=str, nargs="+", default=["naive", "tiled16", "tiled32"]
    )
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output-root", type=Path, default=Path("artifact/benchmarks"))

    args = parser.parse_args()
    environment = collect_environment(role="gpu-server")
    validate_formal_benchmark(environment=environment)

    if not args.binary.exists():
        raise FileNotFoundError(f"MatMul binary not found: {args.binary}")

    results = []
    print(f"{'N':>6}{'Variant':>14}{'Latency(ms)':>16}{'TFLOPS':>12}{'vs Naive':>12}")
    print("-" * 60)

    for size in args.sizes:
        size_results = []

        for variant in args.variants:
            completed = subprocess.run(
                [
                    str(args.binary),
                    str(size),
                    variant,
                    str(args.warmups),
                    str(args.iterations),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            result = parse_key_value_output(completed.stdout)
            size_results.append(result)

        naive_latency = next(
            float(result["median_ms"])
            for result in size_results
            if result["variant"] == "naive"
        )

        for result in size_results:
            latency = float(result["median_ms"])
            speedup = naive_latency / latency
            result["speedup_vs_naive"] = speedup
            results.append(result)
            print(
                f"{size:>6}"
                f"{result['variant']!s:>14}"
                f"{latency:>16.4f}"
                f"{float(result['effective_tflops']):>12.2f}"
                f"{speedup:>12.2f}x"
            )

    commit = environment["project"]["git"]["commit"]
    run_directory = create_run_directory(args.output_root, "naive_matmul", commit)

    payload = {
        "experiment": ("native_cuda_matmul"),
        "config": {
            "sizes": args.sizes,
            "variants": args.variants,
            "warmups": args.warmups,
            "iterations": args.iterations,
        },
        "environment": environment,
        "results": results,
    }

    result_path = run_directory / "result.json"
    write_json(result_path, payload)
    print()
    print(f"Result written to: {result_path}")


if __name__ == "__main__":
    main()
