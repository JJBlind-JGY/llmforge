"""Benchmark prefill and decode behavior of the educational decoder."""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import torch

from llmforge.benchmark.cuda_timer import (
    measure_cuda_events,
    warmup,
)
from llmforge.benchmark.experiment import (
    create_run_directory,
    write_json,
)
from llmforge.benchmark.guard import (
    validate_formal_benchmark,
)
from llmforge.benchmark.statistics import (
    summarize_ms,
)
from llmforge.environment import (
    collect_environment,
)
from llmforge.model_execution.mini_decoder import (
    MiniDecoderConfig,
    MiniDecoderLM,
    generate_cached,
)


def cache_storage_bytes(
    past_key_values,
) -> int:
    """Return physical bytes represented by KV tensors."""

    total = 0

    for key, value in past_key_values:
        total += key.numel() * key.element_size()

        total += value.numel() * value.element_size()

    return total


def synchronized_wall_time_ms(
    operation,
) -> float:
    """Measure end-to-end CUDA work including Python orchestration."""

    torch.cuda.synchronize()

    start = time.perf_counter()

    operation()

    torch.cuda.synchronize()

    end = time.perf_counter()

    return (end - start) * 1000.0


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt-lengths",
        type=int,
        nargs="+",
        default=[
            128,
            512,
            2048,
            4096,
        ],
    )

    parser.add_argument(
        "--output-lengths",
        type=int,
        nargs="+",
        default=[
            1,
            8,
            32,
        ],
    )

    parser.add_argument(
        "--output-sweep-prompt",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--warmups",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--generation-iterations",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/benchmarks"),
    )

    args = parser.parse_args()

    environment = collect_environment(role="gpu-server")

    validate_formal_benchmark(environment)

    torch.manual_seed(0)

    device = torch.device("cuda")

    dtype = torch.bfloat16

    required_sequence_length = max(
        max(args.prompt_lengths),
        args.output_sweep_prompt + max(args.output_lengths),
    )

    config = MiniDecoderConfig(
        vocab_size=1024,
        hidden_size=512,
        intermediate_size=1024,
        num_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,
        max_sequence_length=max(
            8192,
            required_sequence_length,
        ),
    )

    model = MiniDecoderLM(config).to(
        device=device,
        dtype=dtype,
    )

    model.eval()

    prompt_results = []

    print()
    print("=== Prompt / Context Sweep ===")
    print(f"{'Prompt':>8}{'Prefill(ms)':>16}{'Decode(ms)':>16}{'KV(MiB)':>12}")
    print("-" * 52)

    for prompt_length in args.prompt_lengths:
        prompt = torch.randint(
            low=0,
            high=config.vocab_size,
            size=(
                1,
                prompt_length,
            ),
            device=device,
        )

        def prefill_operation() -> None:
            logits, _ = model.forward_with_cache(prompt)

            torch.argmax(
                logits[:, -1, :],
                dim=-1,
                keepdim=True,
            )

        warmup(
            prefill_operation,
            args.warmups,
        )

        prefill_samples = measure_cuda_events(
            prefill_operation,
            args.iterations,
        )

        prefill_stats = summarize_ms(prefill_samples)

        with torch.inference_mode():
            logits, base_cache = model.forward_with_cache(prompt)

            next_token = torch.argmax(
                logits[:, -1, :],
                dim=-1,
                keepdim=True,
            )

        torch.cuda.synchronize()

        kv_bytes = cache_storage_bytes(base_cache)

        def decode_operation() -> None:
            logits, _ = model.forward_with_cache(
                next_token,
                past_key_values=(base_cache),
            )

            torch.argmax(
                logits[:, -1, :],
                dim=-1,
                keepdim=True,
            )

        warmup(
            decode_operation,
            args.warmups,
        )

        decode_samples = measure_cuda_events(
            decode_operation,
            args.iterations,
        )

        decode_stats = summarize_ms(decode_samples)

        result = {
            "prompt_length": (prompt_length),
            "prefill_median_ms": (prefill_stats.median_ms),
            "decode_step_median_ms": (decode_stats.median_ms),
            "kv_cache_bytes": (kv_bytes),
            "prefill_samples_ms": (prefill_samples),
            "decode_samples_ms": (decode_samples),
        }

        prompt_results.append(result)

        print(
            f"{prompt_length:>8}"
            f"{prefill_stats.median_ms:>16.4f}"
            f"{decode_stats.median_ms:>16.4f}"
            f"{kv_bytes / 2**20:>12.3f}"
        )

    print()
    print("=== Output Length Sweep ===")
    print(f"{'Output':>8}{'E2E(ms)':>14}{'Tok/s':>14}")
    print("-" * 36)

    output_prompt = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(
            1,
            args.output_sweep_prompt,
        ),
        device=device,
    )

    output_results = []

    for output_length in args.output_lengths:

        def generation_operation() -> None:
            generate_cached(
                model,
                output_prompt,
                max_new_tokens=(output_length),
            )

        for _ in range(args.warmups):
            generation_operation()

        torch.cuda.synchronize()

        samples = []

        for _ in range(args.generation_iterations):
            samples.append(synchronized_wall_time_ms(generation_operation))

        median_ms = statistics.median(samples)

        tokens_per_second = output_length / (median_ms / 1000.0)

        output_results.append(
            {
                "prompt_length": (args.output_sweep_prompt),
                "output_length": (output_length),
                "e2e_median_ms": (median_ms),
                "output_tokens_per_second": (tokens_per_second),
                "samples_ms": samples,
            }
        )

        print(f"{output_length:>8}{median_ms:>14.4f}{tokens_per_second:>14.2f}")

    commit = environment["project"]["git"]["commit"]

    run_directory = create_run_directory(
        args.output_root,
        "prefill_decode",
        commit,
    )

    result_path = run_directory / "result.json"

    write_json(
        result_path,
        {
            "experiment": ("prefill_decode"),
            "config": {
                "model": {
                    "vocab_size": (config.vocab_size),
                    "hidden_size": (config.hidden_size),
                    "intermediate_size": (config.intermediate_size),
                    "num_layers": (config.num_layers),
                    "num_attention_heads": (config.num_attention_heads),
                    "num_key_value_heads": (config.num_key_value_heads),
                },
                "dtype": "bfloat16",
                "prompt_lengths": (args.prompt_lengths),
                "output_lengths": (args.output_lengths),
                "output_sweep_prompt": (args.output_sweep_prompt),
                "warmups": (args.warmups),
                "iterations": (args.iterations),
            },
            "environment": (environment),
            "prompt_results": (prompt_results),
            "output_results": (output_results),
        },
    )

    print()
    print(f"Result written to: {result_path}")


if __name__ == "__main__":
    main()
