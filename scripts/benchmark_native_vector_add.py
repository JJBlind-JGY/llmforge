"""Benchmark the native CUDA VectorAdd implementation."""

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
        "--binary", type=Path, default=Path(".build/cuda/llmforge_vector_add")
    )
    parser.add_argument("--elements", type=int, default=2**27)
    parser.add_argument(
        "--blocks", type=int, nargs="+", default=[64, 128, 256, 512, 1024]
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("artifacts/benchmarks")
    )

    args = parser.parse_args()
    environment = collect_environment(role="gpu-server")
    validate_formal_benchmark(environment=environment)

    if not args.binary.exists():
        raise FileNotFoundError(f"CUDA binary not found: {args.binary}")

    results = []
    for block_size in args.blocks:
        completed = subprocess.run(
            [
                str(args.binary),
                str(args.elements),
                str(block_size),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        result = parse_key_value_output(completed.stdout)

        results.append(result)

        print(
            f"block={block_size:>4} "
            f"occupancy="
            f"{float(result['theoretical_occupancy_pct']):>6.1f}% "
            f"latency="
            f"{float(result['median_ms']):>8.4f} ms "
            f"bandwidth="
            f"{float(result['bandwidth_gbps']):>8.2f} GB/s"
        )

    commit = environment["project"]["git"]["commit"]
    run_directory = create_run_directory(
        args.output_root,
        "native_vector_add",
        commit,
    )

    payload = {
        "experiment": ("native_cuda_vector_add_block_sweep"),
        "config": {
            "elements": args.elements,
            "block_sizes": args.blocks,
            "binary": str(args.binary),
        },
        "environment": environment,
        "results": results,
    }

    write_json(run_directory / "result.json", payload)

    print(f"Result written to {run_directory / 'result.json'}")


if __name__ == "__main__":
    main()
