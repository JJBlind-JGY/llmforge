"""Simplified decoder-only Transformer execution path."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

KeyValueCache = tuple[
    torch.Tensor,
    torch.Tensor,
]

PastKeyValues = tuple[
    KeyValueCache,
    ...,
]


@dataclass(frozen=True)
class MiniDecoderConfig:
    """Configuration for the educational decoder-only model."""

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    max_sequence_length: int

    def __post_init__(self) -> None:
        values = {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_layers": self.num_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "max_sequence_length": self.max_sequence_length,
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

    @property
    def query_heads_per_kv_head(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        input_dtype = x.dtype

        x_float = x.float()

        variance = x_float.pow(2).mean(
            dim=-1,
            keepdim=True,
        )

        normalized = x_float * torch.rsqrt(variance + self.eps)

        return normalized.to(input_dtype) * self.weight


class CausalSelfAttention(nn.Module):
    """Simplified causal grouped-query self-attention."""

    def __init__(
        self,
        config: MiniDecoderConfig,
    ) -> None:
        super().__init__()

        self.num_attention_heads = config.num_attention_heads

        self.num_key_value_heads = config.num_key_value_heads

        self.head_dim = config.head_dim

        self.kv_repeat = config.query_heads_per_kv_head

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=False,
        )

        self.k_proj = nn.Linear(
            config.hidden_size,
            config.kv_width,
            bias=False,
        )

        self.v_proj = nn.Linear(
            config.hidden_size,
            config.kv_width,
            bias=False,
        )

        self.o_proj = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: torch.Tensor,
        *,
        past_key_value: KeyValueCache | None = None,
        use_cache: bool = False,
    ) -> tuple[
        torch.Tensor,
        KeyValueCache | None,
    ]:
        batch_size, sequence_length, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(
            batch_size,
            sequence_length,
            self.num_attention_heads,
            self.head_dim,
        ).transpose(1, 2)

        k = k.view(
            batch_size,
            sequence_length,
            self.num_key_value_heads,
            self.head_dim,
        ).transpose(1, 2)

        v = v.view(
            batch_size,
            sequence_length,
            self.num_key_value_heads,
            self.head_dim,
        ).transpose(1, 2)

        if past_key_value is not None:
            past_key, past_value = past_key_value

            if past_key.shape != past_value.shape:
                raise ValueError("Past key/value shapes must match.")

            if past_key.shape[0] != batch_size:
                raise ValueError("Past KV batch size mismatch.")

            if past_key.shape[1] != self.num_key_value_heads:
                raise ValueError("Past KV head count mismatch.")

            if past_key.shape[-1] != self.head_dim:
                raise ValueError("Past KV head dimension mismatch.")

            k = torch.cat(
                (
                    past_key,
                    k,
                ),
                dim=2,
            )

            v = torch.cat(
                (
                    past_value,
                    v,
                ),
                dim=2,
            )

        present_key_value = (k, v) if use_cache else None

        k_for_attention = k.repeat_interleave(
            self.kv_repeat,
            dim=1,
        )

        v_for_attention = v.repeat_interleave(
            self.kv_repeat,
            dim=1,
        )

        scores = torch.matmul(
            q,
            k_for_attention.transpose(
                -2,
                -1,
            ),
        )

        scores = scores / math.sqrt(self.head_dim)

        scores = scores.masked_fill(
            causal_mask,
            torch.finfo(scores.dtype).min,
        )

        probabilities = torch.softmax(
            scores,
            dim=-1,
        )

        context = torch.matmul(
            probabilities,
            v_for_attention,
        )

        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(
                batch_size,
                sequence_length,
                -1,
            )
        )

        output = self.o_proj(context)

        return (
            output,
            present_key_value,
        )


class SwiGLUMLP(nn.Module):
    """Simplified Llama-style SwiGLU MLP."""

    def __init__(
        self,
        config: MiniDecoderConfig,
    ) -> None:
        super().__init__()

        self.gate_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
        )

        self.up_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
        )

        self.down_proj = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))

        up = self.up_proj(x)

        return self.down_proj(gate * up)


class DecoderBlock(nn.Module):
    """Pre-norm decoder block."""

    def __init__(
        self,
        config: MiniDecoderConfig,
    ) -> None:
        super().__init__()

        self.attention_norm = RMSNorm(config.hidden_size)

        self.attention = CausalSelfAttention(config)

        self.mlp_norm = RMSNorm(config.hidden_size)

        self.mlp = SwiGLUMLP(config)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: torch.Tensor,
        *,
        past_key_value: KeyValueCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, KeyValueCache | None]:
        attention_output, present = self.attention(
            self.attention_norm(x),
            causal_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )

        x = x + attention_output
        x = x + self.mlp(self.mlp_norm(x))

        return x, present


def _past_length(
    past_key_values: PastKeyValues | None,
) -> int:
    """Return the number of cached tokens across all layers."""
    if past_key_values is None:
        return 0

    if not past_key_values:
        raise ValueError("past_key_values cannot be empty.")

    first_key, _ = past_key_values[0]
    return first_key.shape[2]  # 序列维度


def _build_causal_mask(
    *,
    query_length: int,
    past_length: int,
    device: torch.device,
) -> torch.Tensor:
    """Build causal mask supporting both Prefill and Decode.

    Prefill: past_length=0, query_length=S  → [1, 1, S, S]
    Decode:  past_length=S, query_length=1  → [1, 1, 1, S+1]
    """
    total_length = past_length + query_length

    query_positions = past_length + torch.arange(query_length, device=device)
    key_positions = torch.arange(total_length, device=device)

    mask = key_positions[None, :] > query_positions[:, None]

    return mask[None, None, :, :]


class MiniDecoderLM(nn.Module):
    """Educational decoder-only language model."""

    def __init__(
        self,
        config: MiniDecoderConfig,
    ) -> None:
        super().__init__()

        self.config = config

        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
        )

        self.position_embedding = nn.Embedding(
            config.max_sequence_length,
            config.hidden_size,
        )

        self.layers = nn.ModuleList(
            [DecoderBlock(config) for _ in range(config.num_layers)]
        )

        self.final_norm = RMSNorm(config.hidden_size)

        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )

        # Tie input embedding and LM-head weights.
        self.lm_head.weight = self.token_embedding.weight

    def _forward_impl(
        self,
        input_ids: torch.Tensor,
        *,
        past_key_values: PastKeyValues | None,
        use_cache: bool,
    ) -> tuple[torch.Tensor, PastKeyValues | None]:
        """Unified forward path supporting Prefill and Incremental Decode."""

        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [B, S].")

        _, sequence_length = input_ids.shape

        if sequence_length <= 0:
            raise ValueError("sequence length must be positive.")

        # 校验 cache 层数
        if past_key_values is not None and len(past_key_values) != len(self.layers):
            raise ValueError("past_key_values must contain one entry per layer.")

        past_length = _past_length(past_key_values)
        total_length = past_length + sequence_length

        if total_length > self.config.max_sequence_length:
            raise ValueError("sequence length exceeds max_sequence_length.")

        # ⚠️ 关键：positions 必须从 past_length 开始，而不是 0
        positions = torch.arange(
            past_length,
            total_length,
            device=input_ids.device,
        )

        x = (
            self.token_embedding(input_ids)
            + self.position_embedding(positions)[None, :, :]
        )

        # ⚠️ 关键：使用通用 mask 函数
        causal_mask = _build_causal_mask(
            query_length=sequence_length,
            past_length=past_length,
            device=input_ids.device,
        )

        presents: list[KeyValueCache] = []

        for layer_index, layer in enumerate(self.layers):
            layer_past = (
                None if past_key_values is None else past_key_values[layer_index]
            )

            x, present = layer(
                x,
                causal_mask,
                past_key_value=layer_past,
                use_cache=use_cache,
            )

            if use_cache:
                if present is None:
                    raise RuntimeError("Cache requested but layer returned no cache.")
                presents.append(present)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        return (
            logits,
            tuple(presents) if use_cache else None,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Full-sequence forward (used by naive generation)."""
        logits, _ = self._forward_impl(
            input_ids,
            past_key_values=None,
            use_cache=False,
        )
        return logits

    def forward_with_cache(
        self,
        input_ids: torch.Tensor,
        *,
        past_key_values: PastKeyValues | None = None,
    ) -> tuple[torch.Tensor, PastKeyValues]:
        """Forward with KV cache for Prefill or Incremental Decode."""
        logits, new_cache = self._forward_impl(
            input_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )

        if new_cache is None:
            raise RuntimeError("Expected KV cache.")

        return logits, new_cache


@dataclass(frozen=True)
class NaiveGenerationStats:
    """Execution statistics for naive autoregressive generation."""

    forward_calls: int
    sequence_lengths: tuple[int, ...]
    model_token_evaluations: int


@torch.inference_mode()
def generate_naive(
    model: MiniDecoderLM,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
) -> tuple[
    torch.Tensor,
    NaiveGenerationStats,
]:
    """Greedy generation without a KV cache."""

    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive.")

    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [B, S].")

    batch_size, prompt_length = input_ids.shape

    if prompt_length + max_new_tokens > model.config.max_sequence_length:
        raise ValueError("prompt + generated tokens exceed max_sequence_length.")

    generated = input_ids

    sequence_lengths: list[int] = []

    for _ in range(max_new_tokens):
        current_length = generated.shape[1]

        sequence_lengths.append(current_length)

        logits = model(generated)

        next_token = torch.argmax(
            logits[:, -1, :],
            dim=-1,
            keepdim=True,
        )

        generated = torch.cat(
            (
                generated,
                next_token,
            ),
            dim=1,
        )

    token_evaluations = batch_size * sum(sequence_lengths)

    stats = NaiveGenerationStats(
        forward_calls=max_new_tokens,
        sequence_lengths=tuple(sequence_lengths),
        model_token_evaluations=(token_evaluations),
    )

    return generated, stats


def kv_cache_storage_bytes(
    past_key_values: PastKeyValues,
) -> int:
    """Return actual tensor storage of the educational KV cache."""
    total = 0
    for key, value in past_key_values:
        total += key.numel() * key.element_size()
        total += value.numel() * value.element_size()
    return total


@dataclass(frozen=True)
class CachedGenerationStats:
    """Execution statistics for KV-cached generation."""

    forward_calls: int
    model_input_lengths: tuple[int, ...]
    cache_lengths_after_forward: tuple[int, ...]
    model_token_evaluations: int


@torch.inference_mode()
def generate_cached(
    model: MiniDecoderLM,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
) -> tuple[torch.Tensor, CachedGenerationStats]:
    """Greedy generation with a per-layer KV cache."""

    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive.")

    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [B, S].")

    batch_size, prompt_length = input_ids.shape

    if prompt_length + max_new_tokens > model.config.max_sequence_length:
        raise ValueError("prompt + generated tokens exceed max_sequence_length.")

    generated = input_ids
    model_input_lengths: list[int] = []
    cache_lengths: list[int] = []

    # ---------- Prefill ----------
    logits, past_key_values = model.forward_with_cache(input_ids)
    model_input_lengths.append(prompt_length)
    cache_lengths.append(past_key_values[0][0].shape[2])

    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    generated = torch.cat((generated, next_token), dim=1)

    # ---------- Incremental Decode ----------
    for _ in range(1, max_new_tokens):
        logits, past_key_values = model.forward_with_cache(
            next_token,
            past_key_values=past_key_values,
        )
        model_input_lengths.append(1)
        cache_lengths.append(past_key_values[0][0].shape[2])

        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        generated = torch.cat((generated, next_token), dim=1)

    token_evaluations = batch_size * sum(model_input_lengths)

    stats = CachedGenerationStats(
        forward_calls=max_new_tokens,
        model_input_lengths=tuple(model_input_lengths),
        cache_lengths_after_forward=tuple(cache_lengths),
        model_token_evaluations=token_evaluations,
    )

    return generated, stats
