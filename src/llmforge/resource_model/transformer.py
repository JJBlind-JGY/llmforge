"""Resource accounting for Llama-style decoder-only Transformer."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TransformerConfig:
    """Minimal architecture for resource accounting."""

    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int
    tie_word_embeddings: bool = True

    def __post_init__(self) -> None:
        values = {
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_layers": self.num_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "vocab_size": self.vocab_size,
        }

        for name, value in values.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive.")

        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads.")

        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads."
            )

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def kv_width(self) -> int:
        return self.num_key_value_heads * self.head_dim


def parameter_count(
    config: TransformerConfig,
) -> int:
    """Count parameters in a bias-free Llama-style model."""

    d = config.hidden_size
    d_ff = config.intermediate_size
    d_kv = config.kv_width

    embedding = config.vocab_size * d

    attention_per_layer = d * d + d * d_kv + d * d_kv + d * d

    # gate_proj + up_proj + down_proj
    mlp_per_layer = 3 * d * d_ff

    # attention RMSNorm + MLP RMSNorm
    norm_per_layer = 2 * d

    total = (
        embedding
        + config.num_layers * (attention_per_layer + mlp_per_layer + norm_per_layer)
        + d  # final RMSNorm
    )

    if not config.tie_word_embeddings:
        total += config.vocab_size * d

    return total


def parameter_memory_bytes(
    config: TransformerConfig,
    *,
    bytes_per_weight: int,
) -> int:
    """First-order parameter storage estimate."""

    if bytes_per_weight <= 0:
        raise ValueError("bytes_per_weight must be positive.")

    return parameter_count(config) * bytes_per_weight


def kv_cache_bytes_per_token(
    config: TransformerConfig,
    *,
    bytes_per_element: int,
) -> int:
    """KV bytes for one token across all Transformer layers."""

    if bytes_per_element <= 0:
        raise ValueError("bytes_per_element must be positive.")

    return (
        config.num_layers
        * 2  # K + V
        * config.num_key_value_heads
        * config.head_dim
        * bytes_per_element
    )


def kv_cache_bytes(
    config: TransformerConfig,
    *,
    batch_size: int,
    sequence_length: int,
    bytes_per_element: int,
) -> int:
    """Total KV cache for a batch of sequences."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive.")

    return (
        batch_size
        * sequence_length
        * kv_cache_bytes_per_token(
            config,
            bytes_per_element=bytes_per_element,
        )
    )


def naive_attention_score_bytes(
    config: TransformerConfig,
    *,
    batch_size: int,
    sequence_length: int,
    bytes_per_element: int,
) -> int:
    """Size of a materialized [B, H, S, S] attention-score tensor."""

    return (
        batch_size
        * config.num_attention_heads
        * sequence_length
        * sequence_length
        * bytes_per_element
    )


def linear_flops_per_token_per_layer(
    config: TransformerConfig,
) -> int:
    """Major projection FLOPs for one token in one layer."""

    d = config.hidden_size
    d_ff = config.intermediate_size
    d_kv = config.kv_width

    weight_elements = (
        # Q and O
        2 * d * d
        # K and V
        + 2 * d * d_kv
        # gate / up / down
        + 3 * d * d_ff
    )

    # Multiply-add = 2 FLOPs.
    return 2 * weight_elements


def prefill_flops(
    config: TransformerConfig,
    *,
    batch_size: int,
    sequence_length: int,
) -> int:
    """Approximate Transformer-block FLOPs for prefill.

    Excludes embedding, RMSNorm, RoPE, softmax elementwise operations,
    sampling, and LM-head projection.
    """

    linear = sequence_length * linear_flops_per_token_per_layer(config)

    attention = 4 * sequence_length * sequence_length * config.hidden_size

    return batch_size * config.num_layers * (linear + attention)


def decode_step_flops(
    config: TransformerConfig,
    *,
    batch_size: int,
    context_length: int,
) -> int:
    """Approximate Transformer-block FLOPs for one decode step."""

    if context_length <= 0:
        raise ValueError("context_length must be positive.")

    linear = linear_flops_per_token_per_layer(config)

    attention = 4 * context_length * config.hidden_size

    return batch_size * config.num_layers * (linear + attention)
