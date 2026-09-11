import torch

from llmforge.model_execution.mini_decoder import (
    MiniDecoderConfig,
    MiniDecoderLM,
    generate_cached,
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


def test_kv_cache_shape() -> None:
    torch.manual_seed(0)

    config = tiny_config()

    model = MiniDecoderLM(config)

    input_ids = torch.tensor(
        [[1, 2, 3, 4]],
        dtype=torch.long,
    )

    _, cache = model.forward_with_cache(input_ids)

    assert len(cache) == 2

    key, value = cache[0]

    assert key.shape == (
        1,
        2,
        4,
        8,
    )

    assert value.shape == key.shape


def test_cached_decode_matches_full_forward() -> None:
    torch.manual_seed(0)

    model = MiniDecoderLM(tiny_config())

    model.eval()

    prompt = torch.tensor(
        [[1, 2, 3, 4]],
        dtype=torch.long,
    )

    prefill_logits, cache = model.forward_with_cache(prompt)

    next_token = torch.argmax(
        prefill_logits[:, -1, :],
        dim=-1,
        keepdim=True,
    )

    full_sequence = torch.cat(
        (
            prompt,
            next_token,
        ),
        dim=1,
    )

    full_logits = model(full_sequence)

    cached_logits, new_cache = model.forward_with_cache(
        next_token,
        past_key_values=cache,
    )

    torch.testing.assert_close(
        cached_logits[:, -1, :],
        full_logits[:, -1, :],
        rtol=1e-5,
        atol=1e-6,
    )

    assert new_cache[0][0].shape[2] == 5


def test_cached_generation_matches_naive() -> None:
    torch.manual_seed(0)

    model = MiniDecoderLM(tiny_config())

    model.eval()

    prompt = torch.tensor(
        [[1, 2, 3, 4]],
        dtype=torch.long,
    )

    naive_output, naive_stats = generate_naive(
        model,
        prompt,
        max_new_tokens=3,
    )

    cached_output, cached_stats = generate_cached(
        model,
        prompt,
        max_new_tokens=3,
    )

    assert torch.equal(
        cached_output,
        naive_output,
    )

    assert naive_stats.model_token_evaluations == 15

    assert cached_stats.model_input_lengths == (4, 1, 1)

    assert cached_stats.cache_lengths_after_forward == (4, 5, 6)

    assert cached_stats.model_token_evaluations == 6
