from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from llmforge.runtime.observations import (
    RequestScheduleObservation,
    split_scheduled_tokens,
)


def extract_request_schedule_observations(
    *,
    step_id: int,
    num_scheduled_tokens: Mapping[str, int],
    requests: Mapping[str, Any],
) -> list[RequestScheduleObservation]:
    observations = []

    for request_id, scheduled_tokens in num_scheduled_tokens.items():
        request = requests.get(request_id)

        num_prompt_tokens = (
            getattr(request, "num_prompt_tokens", None) if request is not None else None
        )

        num_computed_after = (
            getattr(request, "num_computed_tokens", None)
            if request is not None
            else None
        )

        prompt_tokens, output_tokens = split_scheduled_tokens(
            scheduled_tokens=int(scheduled_tokens),
            num_prompt_tokens=num_prompt_tokens,
            num_computed_tokens_after_schedule=num_computed_after,
        )

        observations.append(
            RequestScheduleObservation(
                step_id=step_id,
                request_id=request_id,
                scheduled_tokens=int(scheduled_tokens),
                scheduled_prompt_tokens=prompt_tokens,
                scheduled_output_tokens=output_tokens,
                num_prompt_tokens=num_prompt_tokens,
                num_computed_tokens_after_schedule=num_computed_after,
            )
        )

    return observations
