"""Demonstrate naive autoregressive generation."""

from __future__ import annotations

import argparse

import torch

from llmforge.model_execution.mini_decoder import (
    MiniDecoderConfig,
    MiniDecoderLM,
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
        raise RuntimeError("CUDA is required for this demo.")

    device = torch.device("cuda")

    torch.manual_seed(0)

    config = MiniDecoderConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_sequence_length=2048,
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
            args.prompt_length,
        ),
        device=device,
    )

    output, stats = generate_naive(
        model,
        prompt,
        max_new_tokens=(args.new_tokens),
    )

    print(f"prompt_length={args.prompt_length}")

    print(f"new_tokens={args.new_tokens}")

    print(f"final_length={output.shape[1]}")

    print(f"forward_calls={stats.forward_calls}")

    print(f"sequence_lengths={stats.sequence_lengths}")

    print(f"model_token_evaluations={stats.model_token_evaluations}")

    unique_sequence_tokens = args.prompt_length + args.new_tokens

    print(f"final_sequence_tokens={unique_sequence_tokens}")

    print(
        "evaluation_to_final_token_ratio="
        f"{stats.model_token_evaluations / unique_sequence_tokens:.2f}x"
    )


if __name__ == "__main__":
    main()
