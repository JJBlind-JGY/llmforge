"""Pure serving benchmark metric helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class SlotUtilization:
    """Useful-token utilization of a static batch."""

    useful_tokens: int
    executed_slots: int
    wasted_slots: int
    utilization: float
    amplification: float


def static_prompt_padding_stats(
    prompt_lengths: Sequence[int],
) -> SlotUtilization:
    """Compute prompt-slot utilization for padded static batching."""

    if not prompt_lengths:
        raise ValueError("prompt_lengths must not be empty.")

    if any(length <= 0 for length in prompt_lengths):
        raise ValueError("prompt lengths must be positive.")

    useful_tokens = sum(prompt_lengths)

    executed_slots = len(prompt_lengths) * max(prompt_lengths)

    wasted_slots = executed_slots - useful_tokens

    return SlotUtilization(
        useful_tokens=useful_tokens,
        executed_slots=executed_slots,
        wasted_slots=wasted_slots,
        utilization=(useful_tokens / executed_slots),
        amplification=(executed_slots / useful_tokens),
    )


def static_decode_slot_stats(
    output_lengths: Sequence[int],
) -> SlotUtilization:
    """Model synchronized static-batch decode slot utilization."""

    if not output_lengths:
        raise ValueError("output_lengths must not be empty.")

    if any(length <= 0 for length in output_lengths):
        raise ValueError("output lengths must be positive.")

    useful_tokens = sum(output_lengths)

    executed_slots = len(output_lengths) * max(output_lengths)

    wasted_slots = executed_slots - useful_tokens

    return SlotUtilization(
        useful_tokens=useful_tokens,
        executed_slots=executed_slots,
        wasted_slots=wasted_slots,
        utilization=(useful_tokens / executed_slots),
        amplification=(executed_slots / useful_tokens),
    )


def average_decode_latency_ms(
    *,
    e2e_ms: float,
    first_token_proxy_ms: float,
    output_tokens: int,
) -> float | None:
    """Estimate average subsequent-token latency.

    This is a local runtime proxy, not serving TPOT.
    """

    if output_tokens <= 0:
        raise ValueError("output_tokens must be positive.")

    if output_tokens == 1:
        return None

    decode_duration = e2e_ms - first_token_proxy_ms

    if decode_duration <= 0:
        return None

    return decode_duration / (output_tokens - 1)
