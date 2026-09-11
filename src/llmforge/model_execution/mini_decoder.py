"""Simplified decoder-only Transformer execution path."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


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
    ) -> torch.Tensor:
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

        # Educational GQA implementation:
        # materialize repeated K/V heads.
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

        return self.o_proj(context)


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
    ) -> torch.Tensor:
        x = x + self.attention(
            self.attention_norm(x),
            causal_mask,
        )

        x = x + self.mlp(self.mlp_norm(x))

        return x


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

    def forward(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [B, S].")

        batch_size, sequence_length = input_ids.shape

        if sequence_length <= 0:
            raise ValueError("sequence length must be positive.")

        if sequence_length > self.config.max_sequence_length:
            raise ValueError("sequence length exceeds max_sequence_length.")

        positions = torch.arange(
            sequence_length,
            device=input_ids.device,
        )

        x = (
            self.token_embedding(input_ids)
            + self.position_embedding(positions)[None, :, :]
        )

        causal_mask = torch.triu(
            torch.ones(
                (
                    sequence_length,
                    sequence_length,
                ),
                dtype=torch.bool,
                device=input_ids.device,
            ),
            diagonal=1,
        )[None, None, :, :]

        for layer in self.layers:
            x = layer(
                x,
                causal_mask,
            )

        x = self.final_norm(x)

        return self.lm_head(x)


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
