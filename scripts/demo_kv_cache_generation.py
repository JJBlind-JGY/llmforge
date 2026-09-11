"""Compare naive and KV-cached autoregressive generation."""

from __future__ import annotations

import argparse

import torch

from llmforge.model_execution.mini_decoder import (
    MiniDecoderConfig,
    MiniDecoderLM,
    generate_cached,
    generate_naive,
)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt-length",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--new-tokens",
        type=int,
        default=4,
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    torch.manual_seed(0)

    device = torch.device("cuda")

    config = MiniDecoderConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_sequence_length=2048,
    )

    # Float32 here deliberately prioritizes
    # correctness comparison over throughput.
    model = MiniDecoderLM(config).to(
        device=device,
        dtype=torch.float32,
    )

    model.eval()

    prompt = torch.randint(
        0,
        config.vocab_size,
        (
            1,
            args.prompt_length,
        ),
        device=device,
    )

    naive_output, naive_stats = generate_naive(
        model,
        prompt,
        max_new_tokens=(args.new_tokens),
    )

    cached_output, cached_stats = generate_cached(
        model,
        prompt,
        max_new_tokens=(args.new_tokens),
    )

    print(f"outputs_equal={torch.equal(naive_output, cached_output)}")

    print(f"naive_input_lengths={naive_stats.sequence_lengths}")

    print(f"cached_input_lengths={cached_stats.model_input_lengths}")

    print(f"cache_lengths={cached_stats.cache_lengths_after_forward}")

    print(f"naive_token_evaluations={naive_stats.model_token_evaluations}")

    print(f"cached_token_evaluations={cached_stats.model_token_evaluations}")

    print(
        "evaluation_reduction="
        f"{naive_stats.model_token_evaluations / cached_stats.model_token_evaluations:.2f}x"
    )


if __name__ == "__main__":
    main()
