"""Benchmark Triton FP32 IEEE MatMul against PyTorch."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import triton

from llmforge.benchmark.cuda_timer import measure_cuda_events, warmup
from llmforge.benchmark.experiment import create_run_directory, write_json
from llmforge.benchmark.guard import validate_formal_benchmark
from llmforge.benchmark.statistics import summarize_ms
from llmforge.environment import collect_environment
from llmforge.kernels.triton.matmul import (
    get_best_bf16_matmul_config,
    get_best_matmul_config,
    matmul_bf16_into,
    matmul_fp32_ieee_into,
)


def gemm_tflops(n: int, latency_ms: float) -> float:
    flops = 2.0 * n**3
    return flops / (latency_ms * 1e-3) / 1e12


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=[512, 1024, 2048, 4096])
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument(
        "--precision", choices=["fp32_ieee", "bf16"], default="fp32_ieee"
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("artifacts/benchmarks")
    )

    args = parser.parse_args()
    environment = collect_environment(role="gpu-server")
    validate_formal_benchmark(environment=environment)

    if args.precision == "fp32_ieee":
        dtype = torch.float32
        matmul_fn = matmul_fp32_ieee_into
        get_config_fn = get_best_matmul_config
        torch.backends.cuda.matmul.fp32_precision = "ieee"
    else:  # bf16
        dtype = torch.bfloat16
        matmul_fn = matmul_bf16_into
        get_config_fn = get_best_bf16_matmul_config
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

    torch.manual_seed(0)

    results = []
    print(f"{'N':>6}{'Provider':>18}{'Latency(ms)':>16}{'TFLOPS':>12}")
    print("-" * 52)

    for n in args.sizes:
        a = torch.randn((n, n), device="cuda", dtype=dtype)
        b = torch.randn((n, n), device="cuda", dtype=dtype)
        triton_output = torch.empty((n, n), device="cuda", dtype=dtype)
        torch_output = torch.empty_like(triton_output)

        # First Triton call triggers JIT/autotuning.
        matmul_fn(a, b, triton_output)
        torch.mm(a, b, out=torch_output)
        torch.cuda.synchronize()
        torch.testing.assert_close(triton_output, torch_output, rtol=1e-3, atol=1e-2)

        best_config = get_config_fn()

        providers = {
            "triton": lambda a=a, b=b, triton_output=triton_output: matmul_fn(
                a, b, triton_output
            ),
            "torch": lambda a=a, b=b, torch_output=torch_output: torch.mm(
                a, b, out=torch_output
            ),
        }

        for provider, operation in providers.items():
            warmup(operation, args.warmups)
            samples = measure_cuda_events(operation, args.iterations)
            stats = summarize_ms(samples)
            tflops = gemm_tflops(n, stats.median_ms)

            result = {
                "n": n,
                "provider": provider,
                "median_ms": stats.median_ms,
                "effective_tflops": tflops,
                "raw_samples_ms": samples,
            }

            if provider == "triton":
                result["autotune_config"] = best_config

            results.append(result)
            print(f"{n:>6}{provider:>18}{stats.median_ms:>16.4f}{tflops:>12.2f}")
        print(f"  best_triton_config={best_config}")

    commit = environment["project"]["git"]["commit"]

    run_directory = create_run_directory(
        args.output_root,
        f"triton_matmul_{args.precision}",
        commit,
    )
    result_path = run_directory / "result.json"
    write_json(
        result_path,
        {
            "experiment": (f"triton_matmul_{args.precision}"),
            "config": {
                "sizes": args.sizes,
                "warmups": args.warmups,
                "iterations": (args.iterations),
                "precision": (args.precision),
            },
            "software": {
                "triton": (triton.__version__),
            },
            "environment": environment,
            "results": results,
        },
    )
    print(f"Result written to: {result_path}")


if __name__ == "__main__":
    main()
