"""Benchmark Hugging Face generation on one GPU."""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)

from llmforge.benchmark.experiment import (
    create_run_directory,
    write_json,
)
from llmforge.benchmark.guard import (
    validate_formal_benchmark,
)
from llmforge.benchmark.serving import (
    average_decode_latency_ms,
    static_decode_slot_stats,
    static_prompt_padding_stats,
)
from llmforge.environment import (
    collect_environment,
)


def bytes_to_gib(value: int) -> float:
    return value / 2**30


def make_prompt(
    *,
    length: int,
    vocab_size: int,
    device: torch.device,
    seed: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:
    """Create deterministic ordinary token IDs."""

    generator = torch.Generator()
    generator.manual_seed(seed + length)

    # Avoid the high token-id region where
    # model-specific special tokens live.
    upper_bound = min(
        vocab_size,
        10_000,
    )

    input_ids = torch.randint(
        low=1_000,
        high=upper_bound,
        size=(1, length),
        generator=generator,
        dtype=torch.long,
    ).to(device)

    attention_mask = torch.ones(
        (1, length),
        dtype=torch.long,
        device=device,
    )

    return (
        input_ids,
        attention_mask,
    )


def make_generation_config(
    *,
    output_tokens: int,
    pad_token_id: int,
    cache_implementation: str,
) -> GenerationConfig:
    return GenerationConfig(
        max_new_tokens=output_tokens,
        min_new_tokens=output_tokens,
        do_sample=False,
        num_beams=1,
        use_cache=True,
        cache_implementation=(cache_implementation),
        # Performance benchmark:
        # disable EOS stopping so every case
        # executes exactly output_tokens steps.
        eos_token_id=None,
        pad_token_id=pad_token_id,
    )


@torch.inference_mode()
def run_generation(
    *,
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    generation_config: GenerationConfig,
) -> tuple[
    float,
    int,
]:
    """Return synchronized wall time and incremental peak allocation."""

    torch.cuda.synchronize()

    baseline_allocated = torch.cuda.memory_allocated()

    torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()

    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        generation_config=(generation_config),
    )

    torch.cuda.synchronize()

    elapsed_ms = (time.perf_counter() - start) * 1000.0

    peak_allocated = torch.cuda.max_memory_allocated()

    peak_incremental = max(
        0,
        peak_allocated - baseline_allocated,
    )

    del output

    return (
        elapsed_ms,
        peak_incremental,
    )


def benchmark_case(
    *,
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    generation_config: GenerationConfig,
    warmups: int,
    iterations: int,
) -> dict:
    for _ in range(warmups):
        run_generation(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=(generation_config),
        )

    samples_ms = []
    peak_incremental_bytes = 0

    for _ in range(iterations):
        elapsed_ms, peak_bytes = run_generation(
            model=model,
            input_ids=input_ids,
            attention_mask=(attention_mask),
            generation_config=(generation_config),
        )

        samples_ms.append(elapsed_ms)

        peak_incremental_bytes = max(
            peak_incremental_bytes,
            peak_bytes,
        )

    return {
        "samples_ms": samples_ms,
        "median_ms": statistics.median(samples_ms),
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
        "peak_incremental_bytes": (peak_incremental_bytes),
    }


def make_left_padded_batch(
    *,
    sequences: list[torch.Tensor],
    pad_token_id: int,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:
    batch_size = len(sequences)

    max_length = max(sequence.numel() for sequence in sequences)

    input_ids = torch.full(
        (
            batch_size,
            max_length,
        ),
        fill_value=pad_token_id,
        dtype=torch.long,
        device=device,
    )

    attention_mask = torch.zeros(
        (
            batch_size,
            max_length,
        ),
        dtype=torch.long,
        device=device,
    )

    for row, sequence in enumerate(sequences):
        length = sequence.numel()

        input_ids[row, -length:] = sequence.to(device)

        attention_mask[row, -length:] = 1

    return (
        input_ids,
        attention_mask,
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-8B",
    )

    parser.add_argument(
        "--revision",
        required=True,
    )

    parser.add_argument(
        "--prompt-lengths",
        nargs="+",
        type=int,
        default=[
            128,
            512,
            2048,
            4096,
        ],
    )

    parser.add_argument(
        "--output-lengths",
        nargs="+",
        type=int,
        default=[
            1,
            8,
            32,
        ],
    )

    parser.add_argument(
        "--mixed-prompt-lengths",
        nargs="+",
        type=int,
        default=[
            128,
            512,
            1024,
            2048,
        ],
    )

    parser.add_argument(
        "--mixed-output-length",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--warmups",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--cache-implementation",
        default="dynamic",
        choices=[
            "dynamic",
        ],
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/benchmarks"),
    )

    args = parser.parse_args()

    environment = collect_environment(role="gpu-server")

    validate_formal_benchmark(environment)

    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one visible GPU.")

    device = torch.device("cuda:0")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
    )

    pad_token_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )

    if pad_token_id is None:
        raise RuntimeError("A pad token id is required.")

    print("=== Loading Model ===")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device)

    model.eval()

    torch.cuda.synchronize()

    model_allocated_bytes = torch.cuda.memory_allocated()

    print(f"model_allocated_gib={bytes_to_gib(model_allocated_bytes):.3f}")

    # Stabilize runtime state before formal cases.
    stabilization_ids, stabilization_mask = make_prompt(
        length=512,
        vocab_size=(model.config.vocab_size),
        device=device,
        seed=args.seed,
    )

    stabilization_config = make_generation_config(
        output_tokens=4,
        pad_token_id=pad_token_id,
        cache_implementation=(args.cache_implementation),
    )

    for _ in range(2):
        run_generation(
            model=model,
            input_ids=stabilization_ids,
            attention_mask=(stabilization_mask),
            generation_config=(stabilization_config),
        )

    print()
    print("=== Single Request Shape Sweep ===")

    print(f"{'Prompt':>8}{'Output':>8}{'E2E(ms)':>12}{'OutTok/s':>12}{'PeakΔGiB':>12}")

    print("-" * 52)

    single_results = []

    # Long shapes first to reduce small-workload
    # device-state sensitivity.
    execution_prompt_lengths = sorted(
        args.prompt_lengths,
        reverse=True,
    )

    for prompt_length in execution_prompt_lengths:
        input_ids, attention_mask = make_prompt(
            length=prompt_length,
            vocab_size=(model.config.vocab_size),
            device=device,
            seed=args.seed,
        )

        for output_length in args.output_lengths:
            generation_config = make_generation_config(
                output_tokens=(output_length),
                pad_token_id=(pad_token_id),
                cache_implementation=(args.cache_implementation),
            )

            result = benchmark_case(
                model=model,
                input_ids=input_ids,
                attention_mask=(attention_mask),
                generation_config=(generation_config),
                warmups=args.warmups,
                iterations=args.iterations,
            )

            median_ms = result["median_ms"]

            output_tps = output_length / (median_ms / 1000.0)

            record = {
                "prompt_length": (prompt_length),
                "output_length": (output_length),
                **result,
                "output_tokens_per_second": (output_tps),
            }

            single_results.append(record)

    # Add first-token/decode proxies.
    by_prompt: dict[
        int,
        list[dict],
    ] = {}

    for record in single_results:
        by_prompt.setdefault(
            record["prompt_length"],
            [],
        ).append(record)

    for prompt_length, records in by_prompt.items():
        first_record = next(
            record for record in records if record["output_length"] == 1
        )

        first_token_proxy_ms = first_record["median_ms"]

        for record in records:
            record["first_token_proxy_ms"] = first_token_proxy_ms

            decode_latency = average_decode_latency_ms(
                e2e_ms=(record["median_ms"]),
                first_token_proxy_ms=(first_token_proxy_ms),
                output_tokens=(record["output_length"]),
            )

            record["average_decode_latency_ms"] = decode_latency

            record["decode_tokens_per_second"] = (
                None if decode_latency is None else 1000.0 / decode_latency
            )

    for record in sorted(
        single_results,
        key=lambda item: (
            item["prompt_length"],
            item["output_length"],
        ),
    ):
        print(
            f"{record['prompt_length']:>8}"
            f"{record['output_length']:>8}"
            f"{record['median_ms']:>12.2f}"
            f"{record['output_tokens_per_second']:>12.2f}"
            f"{bytes_to_gib(record['peak_incremental_bytes']):>12.3f}"
        )

    print()
    print("=== Derived Decode Proxy ===")

    print(
        f"{'Prompt':>8}"
        f"{'Output':>8}"
        f"{'First(ms)':>12}"
        f"{'Decode(ms)':>12}"
        f"{'DecodeTok/s':>14}"
    )

    print("-" * 56)

    for record in sorted(
        single_results,
        key=lambda item: (
            item["prompt_length"],
            item["output_length"],
        ),
    ):
        decode_latency = record["average_decode_latency_ms"]

        decode_tps = record["decode_tokens_per_second"]

        print(
            f"{record['prompt_length']:>8}"
            f"{record['output_length']:>8}"
            f"{record['first_token_proxy_ms']:>12.2f}"
            f"{'-' if decode_latency is None else f'{decode_latency:.2f}':>12}"
            f"{'-' if decode_tps is None else f'{decode_tps:.2f}':>14}"
        )

    print()
    print("=== Mixed-Length Static Batch ===")

    padding_stats = static_prompt_padding_stats(args.mixed_prompt_lengths)

    print(f"prompt_lengths={tuple(args.mixed_prompt_lengths)}")

    print(f"useful_prompt_tokens={padding_stats.useful_tokens}")

    print(f"padded_prompt_slots={padding_stats.executed_slots}")

    print(f"prompt_slot_utilization={padding_stats.utilization * 100:.2f}%")

    print(f"prompt_compute_amplification={padding_stats.amplification:.2f}x")

    mixed_sequences = []

    for prompt_length in args.mixed_prompt_lengths:
        sequence, _ = make_prompt(
            length=prompt_length,
            vocab_size=(model.config.vocab_size),
            device=torch.device("cpu"),
            seed=args.seed,
        )

        mixed_sequences.append(sequence[0].cpu())

    batch_ids, batch_mask = make_left_padded_batch(
        sequences=mixed_sequences,
        pad_token_id=pad_token_id,
        device=device,
    )

    mixed_config = make_generation_config(
        output_tokens=(args.mixed_output_length),
        pad_token_id=pad_token_id,
        cache_implementation=(args.cache_implementation),
    )

    static_batch_result = benchmark_case(
        model=model,
        input_ids=batch_ids,
        attention_mask=batch_mask,
        generation_config=mixed_config,
        warmups=args.warmups,
        iterations=args.iterations,
    )

    batch_size = len(args.mixed_prompt_lengths)

    useful_output_tokens = batch_size * args.mixed_output_length

    static_output_tps = useful_output_tokens / (
        static_batch_result["median_ms"] / 1000.0
    )

    print(f"static_batch_e2e_ms={static_batch_result['median_ms']:.2f}")

    print(f"static_batch_output_tok_s={static_output_tps:.2f}")

    print(
        "static_batch_peak_delta_gib="
        f"{bytes_to_gib(static_batch_result['peak_incremental_bytes']):.3f}"
    )

    print()
    print("=== Static Decode Tail Model ===")

    tail_output_lengths = [
        4,
        8,
        16,
        32,
    ]

    tail_stats = static_decode_slot_stats(tail_output_lengths)

    print(f"desired_output_lengths={tuple(tail_output_lengths)}")

    print(f"useful_decode_tokens={tail_stats.useful_tokens}")

    print(f"static_decode_slots={tail_stats.executed_slots}")

    print(f"decode_slot_utilization={tail_stats.utilization * 100:.2f}%")

    print(f"decode_slot_amplification={tail_stats.amplification:.2f}x")

    commit = environment["project"]["git"]["commit"]

    run_directory = create_run_directory(
        args.output_root,
        "hf_generation",
        commit,
    )

    result_path = run_directory / "result.json"

    write_json(
        result_path,
        {
            "experiment": ("hf_generation_baseline"),
            "engine": {
                "name": "transformers",
                "version": (transformers.__version__),
                "attention_backend": ("sdpa"),
                "cache_implementation": (args.cache_implementation),
            },
            "model": {
                "id": args.model,
                "revision": (args.revision),
                "dtype": "bfloat16",
            },
            "workload": {
                "prompt_lengths": (args.prompt_lengths),
                "output_lengths": (args.output_lengths),
                "mixed_prompt_lengths": (args.mixed_prompt_lengths),
                "mixed_output_length": (args.mixed_output_length),
                "seed": args.seed,
            },
            "environment": environment,
            "model_allocated_bytes": (model_allocated_bytes),
            "single_results": (single_results),
            "static_batch": {
                "padding": {
                    "useful_tokens": (padding_stats.useful_tokens),
                    "executed_slots": (padding_stats.executed_slots),
                    "wasted_slots": (padding_stats.wasted_slots),
                    "utilization": (padding_stats.utilization),
                    "amplification": (padding_stats.amplification),
                },
                "benchmark": (static_batch_result),
                "output_tokens_per_second": (static_output_tps),
            },
            "static_decode_tail_model": {
                "output_lengths": (tail_output_lengths),
                "useful_tokens": (tail_stats.useful_tokens),
                "executed_slots": (tail_stats.executed_slots),
                "utilization": (tail_stats.utilization),
                "amplification": (tail_stats.amplification),
            },
        },
    )

    print()
    print(f"Result written to: {result_path}")


if __name__ == "__main__":
    main()
