import torch

from llmforge.model_execution.mini_decoder import (
    MiniDecoderConfig,
    MiniDecoderLM,
    generate_naive,
)


def tiny_config() -> MiniDecoderConfig:
    return MiniDecoderConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_sequence_length=32,
    )


def test_forward_shape() -> None:
    torch.manual_seed(0)

    model = MiniDecoderLM(tiny_config())

    input_ids = torch.tensor(
        [
            [1, 2, 3, 4],
            [4, 3, 2, 1],
        ],
        dtype=torch.long,
    )

    logits = model(input_ids)

    assert logits.shape == (
        2,
        4,
        64,
    )


def test_naive_generation_shape() -> None:
    torch.manual_seed(0)

    model = MiniDecoderLM(tiny_config())

    model.eval()

    input_ids = torch.tensor(
        [[1, 2, 3, 4]],
        dtype=torch.long,
    )

    output, stats = generate_naive(
        model,
        input_ids,
        max_new_tokens=3,
    )

    assert output.shape == (
        1,
        7,
    )

    assert stats.forward_calls == 3

    assert stats.sequence_lengths == (
        4,
        5,
        6,
    )

    assert stats.model_token_evaluations == 15


def test_naive_generation_batch_accounting() -> None:
    torch.manual_seed(0)

    model = MiniDecoderLM(tiny_config())

    model.eval()

    input_ids = torch.tensor(
        [
            [1, 2, 3, 4],
            [4, 3, 2, 1],
        ],
        dtype=torch.long,
    )

    _, stats = generate_naive(
        model,
        input_ids,
        max_new_tokens=3,
    )

    # B × (4 + 5 + 6)
    assert stats.model_token_evaluations == 30
