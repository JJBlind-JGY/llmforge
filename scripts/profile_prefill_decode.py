"""Profile prefill and decode operator composition."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.profiler import (
    ProfilerActivity,
    profile,
    record_function,
)

from llmforge.benchmark.experiment import (
    create_run_directory,
    write_json,
)
from llmforge.benchmark.guard import (
    validate_formal_benchmark,
)
from llmforge.environment import (
    collect_environment,
)
from llmforge.model_execution.mini_decoder import (
    MiniDecoderConfig,
    MiniDecoderLM,
)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=[
            "prefill",
            "decode",
        ],
        required=True,
    )

    parser.add_argument(
        "--context-length",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--warmups",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--row-limit",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/profiling"),
    )

    args = parser.parse_args()

    environment = collect_environment(role="gpu-server")

    validate_formal_benchmark(environment)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    torch.manual_seed(0)

    device = torch.device("cuda")

    config = MiniDecoderConfig(
        vocab_size=1024,
        hidden_size=512,
        intermediate_size=1024,
        num_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,
        max_sequence_length=8192,
    )

    model = MiniDecoderLM(config).to(
        device=device,
        dtype=torch.bfloat16,
    )

    model.eval()

    prompt = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(
            1,
            args.context_length,
        ),
        device=device,
    )

    if args.mode == "prefill":

        @torch.inference_mode()
        def operation() -> None:
            model.forward_with_cache(prompt)

        label = f"prefill_s{args.context_length}"

    else:
        with torch.inference_mode():
            logits, base_cache = model.forward_with_cache(prompt)

            next_token = torch.argmax(
                logits[:, -1, :],
                dim=-1,
                keepdim=True,
            )

        @torch.inference_mode()
        def operation() -> None:
            model.forward_with_cache(
                next_token,
                past_key_values=(base_cache),
            )

        label = f"decode_s{args.context_length}"

    # Warm up outside the profiler.
    for _ in range(args.warmups):
        operation()

    torch.cuda.synchronize()

    with profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_flops=True,
        with_stack=False,
    ) as prof:
        for _ in range(args.iterations):
            with record_function(label):
                operation()

            prof.step()

    torch.cuda.synchronize()

    cuda_table = prof.key_averages(group_by_input_shape=True).table(
        sort_by=("self_cuda_time_total"),
        row_limit=args.row_limit,
    )

    cpu_table = prof.key_averages(group_by_input_shape=True).table(
        sort_by=("self_cpu_time_total"),
        row_limit=args.row_limit,
    )

    print()
    print("=== CUDA Operator Table ===")
    print(cuda_table)

    print()
    print("=== CPU Operator Table ===")
    print(cpu_table)

    commit = environment["project"]["git"]["commit"]

    run_directory = create_run_directory(
        args.output_root,
        (f"prefill_decode_{args.mode}_s{args.context_length}"),
        commit,
    )

    cuda_table_path = run_directory / "cuda_table.txt"

    cpu_table_path = run_directory / "cpu_table.txt"

    trace_path = run_directory / "trace.json"

    metadata_path = run_directory / "metadata.json"

    cuda_table_path.write_text(
        cuda_table,
        encoding="utf-8",
    )

    cpu_table_path.write_text(
        cpu_table,
        encoding="utf-8",
    )

    prof.export_chrome_trace(str(trace_path))

    write_json(
        metadata_path,
        {
            "experiment": ("prefill_decode_profile"),
            "mode": args.mode,
            "context_length": (args.context_length),
            "iterations": (args.iterations),
            "dtype": "bfloat16",
            "environment": (environment),
        },
    )

    print()
    print(f"Artifacts written to: {run_directory}")


if __name__ == "__main__":
    main()
