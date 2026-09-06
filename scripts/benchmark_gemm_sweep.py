"""Run a reproducible square GEMM shape sweep."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from llmforge.benchmark.experiment import create_run_directory, write_json
from llmforge.benchmark.gemm import run_square_gemm
from llmforge.environment import collect_environment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--size", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192]
    )
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument(
        "--output-root", type=Path, default=Path("artifacts/benchmarks")
    )

    args = parser.parse_args()
    environment = collect_environment(role="gpu-server")
    git_commit = environment["project"]["git"]["commit"]
    run_directory = create_run_directory(args.output_root, "gemm_sweep", git_commit)
    conda_prefix = os.environ.get("CONDA_PREFIX", None)

    if conda_prefix:
        print(
            "WARNING: Conda environment is activated. Formal benchmark runs should use a clean shell."
        )

    results = []
    for size in args.size:
        print(f"Running GEMM shape {size}x{size}x{size}")
        result = run_square_gemm(size, args.warmups, args.iterations)
        results.append(result)

        median = result["timing"]["cuda_event"]["median_ms"]
        tflops = result["performance"]["cuda_event_effective_tflops"]
        print(f"N={size}, median={median:.4f} ms, tflops={tflops:.2f} TFOPS")

        payload = {
            "experiment": "square_gemm_shape_sweep",
            "config": {
                "sizes": args.sizes,
                "warmups": args.warmups,
                "iterations": args.iterations,
            },
            "runtime": {
                "cuda_visible_devices": (os.environ.get("CUDA_VISIBLE_DEVICES")),
                "conda_active": (conda_prefix is not None),
            },
            "environment": environment,
            "results": results,
        }
        result_path = run_directory / "result.json"
        write_json(result_path, payload)
        print(
            json.dumps(
                {
                    "run_directory": str(run_directory),
                    "result": str(result_path),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
