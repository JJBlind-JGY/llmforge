"""Benchmark native CUDA reduction variants."""

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
        "--binary", type=Path, default=Path("./build/cuda/llmforge_reduction")
    )
    parser.add_argument("--elements", type=int, default=2**24)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--variants", type=str, nargs="+", default=["atomic", "shared"])
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument(
        "--output-root", type=Path, default=Path("artifacts/benchmarks")
    )

    args = parser.parse_args()
    environment = collect_environment(role="gpu-server")
    validate_formal_benchmark(environment=environment)

    if not args.binary.exists():
        raise FileNotFoundError(f"CUDA binary not found: {args.binary}")

    results = []

    for variant in args.variants:
        completed = subprocess.run(
            [
                str(args.binary),
                str(args.elements),
                str(args.block_size),
                variant,
                str(args.warmups),
                str(args.iterations),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        result = parse_key_value_output(completed.stdout)
        results.append(result)

    atomic_latency = float(results[0]["median_ms"])

    print(
        f"{'Variant':<12}"
        f"{'Latency(ms)':>14}"
        f"{'GB/s':>12}"
        f"{'Atomic Ops':>16}"
        f"{'Speedup':>12}"
    )

    print("-" * 66)

    for result in results:
        latency = float(result["median_ms"])

        speedup = atomic_latency / latency

        result["speedup_vs_atomic"] = speedup

        print(
            f"{result['variant']!s:<12}"
            f"{latency:>14.4f}"
            f"{float(result['useful_input_bandwidth_gbps']):>12.2f}"
            f"{int(result['global_atomic_updates']):>16}"
            f"{speedup:>12.2f}x"
        )

    commit = environment["project"]["git"]["commit"]

    run_directory = create_run_directory(
        args.output_root,
        "native_reduction",
        commit,
    )

    payload = {
        "experiment": ("native_cuda_reduction"),
        "config": {
            "elements": args.elements,
            "block_size": (args.block_size),
            "variants": args.variants,
            "warmups": args.warmups,
            "iterations": (args.iterations),
        },
        "environment": environment,
        "results": results,
    }

    result_path = run_directory / "result.json"
    write_json(result_path, payload)
    print(f"Result written to: {result_path}")


if __name__ == "__main__":
    main()
