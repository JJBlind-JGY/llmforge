from types import SimpleNamespace

from llmforge.runtime.integrations.vllm.helpers import (
    extract_request_schedule_observations,
)


def test_extract_request_schedule_observations() -> None:
    requests = {
        "prefill": SimpleNamespace(
            num_prompt_tokens=128,
            num_computed_tokens=64,
        ),
        "decode": SimpleNamespace(
            num_prompt_tokens=128,
            num_computed_tokens=160,
        ),
    }

    observations = extract_request_schedule_observations(
        step_id=7,
        num_scheduled_tokens={
            "prefill": 32,
            "decode": 1,
        },
        requests=requests,
    )

    by_id = {observation.request_id: observation for observation in observations}

    assert by_id["prefill"].scheduled_prompt_tokens == 32
    assert by_id["prefill"].scheduled_output_tokens == 0
    assert by_id["decode"].scheduled_prompt_tokens == 0
    assert by_id["decode"].scheduled_output_tokens == 1
