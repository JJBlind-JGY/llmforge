"""Scheduler/KV observations independent of a particular vLLM class layout."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SchedulerStepObservation:
    step_id: int
    running_requests: int
    waiting_requests: int
    scheduled_requests: int
    scheduled_tokens: int
    scheduled_prompt_tokens: int
    scheduled_output_tokens: int
    preempted_requests: int
    kv_usage_ratio: float
    schedule_duration_us: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RequestScheduleObservation:
    step_id: int
    request_id: str
    scheduled_tokens: int
    scheduled_prompt_tokens: int
    scheduled_output_tokens: int
    num_prompt_tokens: int | None
    num_computed_tokens_after_schedule: int | None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class KVObservation:
    step_id: int
    usage_ratio: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.usage_ratio <= 1.0:
            raise ValueError("usage_ratio must be in [0, 1].")

    def to_dict(self) -> dict:
        return asdict(self)


def split_scheduled_tokens(
    *,
    scheduled_tokens: int,
    num_prompt_tokens: int | None,
    num_computed_tokens_after_schedule: int | None,
) -> tuple[int, int]:
    if scheduled_tokens < 0:
        raise ValueError("scheduled_tokens must be non-negative.")

    if num_prompt_tokens is None or num_computed_tokens_after_schedule is None:
        return 0, scheduled_tokens

    before = max(0, num_computed_tokens_after_schedule - scheduled_tokens)
    after = num_computed_tokens_after_schedule

    prompt_before = min(before, num_prompt_tokens)
    prompt_after = min(after, num_prompt_tokens)

    prompt_tokens = max(0, prompt_after - prompt_before)
    output_tokens = max(0, scheduled_tokens - prompt_tokens)
    return prompt_tokens, output_tokens
