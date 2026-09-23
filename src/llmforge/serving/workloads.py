from __future__ import annotations

import random
from collections.abc import Sequence

from llmforge.serving.schema import RequestSpec


def _tokens(
    length: int, *, seed: int, vocab_size: int, low: int, high: int
) -> tuple[int, ...]:
    if length <= 0:
        raise ValueError("length must be positive")
    if not (0 <= low < high <= vocab_size):
        raise ValueError("invalid token range")
    rng = random.Random(seed)
    return tuple(rng.randrange(low, high) for _ in range(length))


def fixed_workload(
    *,
    num_requests: int,
    prompt_tokens: int,
    output_tokens: int,
    seed: int,
    vocab_size: int,
    token_low: int = 1000,
    token_high: int = 10000,
    request_prefix: str = "fixed",
) -> list[RequestSpec]:
    if num_requests <= 0 or output_tokens <= 0:
        raise ValueError("num_requests and output_tokens must be positive")
    return [
        RequestSpec(
            request_id=f"{request_prefix}-{i:04d}",
            prompt_token_ids=_tokens(
                prompt_tokens,
                seed=seed + i * 1009,
                vocab_size=vocab_size,
                low=token_low,
                high=token_high,
            ),
            output_tokens=output_tokens,
            metadata={"workload": "fixed"},
        )
        for i in range(num_requests)
    ]


def mixed_workload(
    *,
    num_requests: int,
    shapes: Sequence[tuple[int, int]],
    seed: int,
    vocab_size: int,
    token_low: int = 1000,
    token_high: int = 10000,
    request_prefix: str = "mixed",
) -> list[RequestSpec]:
    if num_requests <= 0 or not shapes:
        raise ValueError("num_requests must be positive and shapes non-empty")
    requests = []
    for i in range(num_requests):
        p, o = shapes[i % len(shapes)]
        requests.append(
            RequestSpec(
                request_id=f"{request_prefix}-{i:04d}",
                prompt_token_ids=_tokens(
                    p,
                    seed=seed + i * 1009,
                    vocab_size=vocab_size,
                    low=token_low,
                    high=token_high,
                ),
                output_tokens=o,
                metadata={"workload": "mixed", "shape": [p, o]},
            )
        )
    return requests


def shared_prefix_workload(
    *,
    num_requests: int,
    prompt_tokens: int,
    output_tokens: int,
    shared_prefix_tokens: int,
    seed: int,
    vocab_size: int,
    token_low: int = 1000,
    token_high: int = 10000,
    request_prefix: str = "prefix",
) -> list[RequestSpec]:
    if not 0 <= shared_prefix_tokens <= prompt_tokens:
        raise ValueError("shared_prefix_tokens out of range")
    prefix = (
        _tokens(
            shared_prefix_tokens,
            seed=seed,
            vocab_size=vocab_size,
            low=token_low,
            high=token_high,
        )
        if shared_prefix_tokens
        else ()
    )
    requests = []
    suffix_len = prompt_tokens - shared_prefix_tokens
    for i in range(num_requests):
        suffix = (
            _tokens(
                suffix_len,
                seed=seed + 10000 + i * 1009,
                vocab_size=vocab_size,
                low=token_low,
                high=token_high,
            )
            if suffix_len
            else ()
        )
        requests.append(
            RequestSpec(
                request_id=f"{request_prefix}-{i:04d}",
                prompt_token_ids=prefix + suffix,
                output_tokens=output_tokens,
                metadata={
                    "workload": "shared_prefix",
                    "shared_prefix_tokens": shared_prefix_tokens,
                    "shared_prefix_ratio": shared_prefix_tokens / prompt_tokens,
                },
            )
        )
    return requests


def with_poisson_arrivals(
    requests: Sequence[RequestSpec], *, request_rate: float, seed: int
) -> list[RequestSpec]:
    if request_rate <= 0:
        raise ValueError("request_rate must be positive")
    rng = random.Random(seed)
    offset = 0.0
    result = []
    for i, req in enumerate(requests):
        if i:
            offset += rng.expovariate(request_rate)
        result.append(
            RequestSpec(
                request_id=req.request_id,
                prompt_token_ids=req.prompt_token_ids,
                output_tokens=req.output_tokens,
                arrival_offset_s=offset,
                metadata={
                    **req.metadata,
                    "arrival_process": "poisson",
                    "request_rate": request_rate,
                },
            )
        )
    return result


def poisson_workload(
    *,
    num_requests: int,
    prompt_tokens: int,
    output_tokens: int,
    request_rate: float,
    seed: int,
    vocab_size: int,
    token_low: int = 1000,
    token_high: int = 10000,
    request_prefix: str = "poisson",
) -> list[RequestSpec]:
    return with_poisson_arrivals(
        fixed_workload(
            num_requests=num_requests,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            seed=seed,
            vocab_size=vocab_size,
            token_low=token_low,
            token_high=token_high,
            request_prefix=request_prefix,
        ),
        request_rate=request_rate,
        seed=seed + 1,
    )


def burst_workload(
    *,
    num_bursts: int,
    burst_size: int,
    prompt_tokens: int,
    output_tokens: int,
    inter_burst_s: float,
    seed: int,
    vocab_size: int,
    token_low: int = 1000,
    token_high: int = 10000,
    request_prefix: str = "burst",
) -> list[RequestSpec]:
    if num_bursts <= 0 or burst_size <= 0 or inter_burst_s < 0:
        raise ValueError("invalid burst parameters")
    result = []
    for b in range(num_bursts):
        for j in range(burst_size):
            i = b * burst_size + j
            result.append(
                RequestSpec(
                    request_id=f"{request_prefix}-{b:03d}-{j:03d}",
                    prompt_token_ids=_tokens(
                        prompt_tokens,
                        seed=seed + i * 1009,
                        vocab_size=vocab_size,
                        low=token_low,
                        high=token_high,
                    ),
                    output_tokens=output_tokens,
                    arrival_offset_s=b * inter_burst_s,
                    metadata={"workload": "burst", "burst_index": b},
                )
            )
    return result
