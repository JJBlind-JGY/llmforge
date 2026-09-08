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
        "--binary", type=Path, default=Path(".build/cuda/llmforge_reduction")
    )
    parser.add_argument("--elements", type=int, default=2**24)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument(
        "--variants",
        type=str,
        nargs="+",
        default=["atomic", "shared_interleaved", "shared", "warp"],
    )
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

    latency = {str(result["variant"]): float(result["median_ms"]) for result in results}
    atomic_latency = latency["atomic"]
    shared_latency = latency["shared"]
    warp_latency = latency["warp"]
    shared_interleaved_latency = latency["shared_interleaved"]

    print(
        f"{'Variant':<20}"
        f"{'Latency(ms)':>14}"
        f"{'Useful GB/s':>14}"
        f"{'Atomic Ops':>16}"
        f"{'vs Atomic':>12}"
        f"{'vs Shared':>12}"
        f"{'vs Warp':>12}"
        f"{'vs Interleaved':>16}"
    )

    print("-" * 88)

    for result in results:
        latency = float(result["median_ms"])

        speedup_atomic = (
            atomic_latency / latency if atomic_latency is not None else None
        )
        speedup_shared = (
            shared_latency / latency if shared_latency is not None else None
        )
        speedup_warp = warp_latency / latency if warp_latency is not None else None
        speedup_shared_interleaved = (
            shared_interleaved_latency / latency
            if shared_interleaved_latency is not None
            else None
        )

        result["speedup_vs_atomic"] = speedup_atomic
        result["speedup_vs_shared"] = speedup_shared
        result["speedup_vs_warp"] = speedup_warp
        result["speedup_vs_shared_interleaved"] = speedup_shared_interleaved

        print(
            f"{result['variant']!s:<12}"
            f"{latency:>14.4f}"
            f"{float(result['useful_input_bandwidth_gbps']):>12.2f}"
            f"{int(result['global_atomic_updates']):>16}"
            f"{speedup_atomic:>12.2f}x"
            f"{speedup_shared:>12.2f}x"
            f"{speedup_warp:>12.2f}x"
            f"{speedup_shared_interleaved:>12.2f}x"
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
