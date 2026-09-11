from llmforge.resource_model.transformer import (
    TransformerConfig,
    decode_step_flops,
    kv_cache_bytes,
    kv_cache_bytes_per_token,
    naive_attention_score_bytes,
    parameter_count,
    prefill_flops,
)


def tiny_config() -> TransformerConfig:
    return TransformerConfig(
        hidden_size=8,
        intermediate_size=16,
        num_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        vocab_size=32,
    )


def test_parameter_count() -> None:
    assert parameter_count(tiny_config()) == 1448


def test_kv_cache_bytes_per_token() -> None:
    assert (
        kv_cache_bytes_per_token(
            tiny_config(),
            bytes_per_element=2,
        )
        == 32
    )


def test_kv_cache_bytes() -> None:
    assert (
        kv_cache_bytes(
            tiny_config(),
            batch_size=2,
            sequence_length=3,
            bytes_per_element=2,
        )
        == 192
    )


def test_naive_attention_score_bytes() -> None:
    assert (
        naive_attention_score_bytes(
            tiny_config(),
            batch_size=2,
            sequence_length=3,
            bytes_per_element=2,
        )
        == 72
    )


def test_prefill_flops() -> None:
    assert (
        prefill_flops(
            tiny_config(),
            batch_size=1,
            sequence_length=3,
        )
        == 7488
    )


def test_decode_step_flops() -> None:
    assert (
        decode_step_flops(
            tiny_config(),
            batch_size=1,
            context_length=3,
        )
        == 2496
    )
