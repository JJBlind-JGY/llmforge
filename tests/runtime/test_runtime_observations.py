from llmforge.runtime.observations import KVObservation, split_scheduled_tokens


def test_prefill_only_schedule() -> None:
    prompt, output = split_scheduled_tokens(
        scheduled_tokens=512,
        num_prompt_tokens=2048,
        num_computed_tokens_after_schedule=1024,
    )
    assert prompt == 512
    assert output == 0


def test_schedule_crosses_prompt_boundary() -> None:
    prompt, output = split_scheduled_tokens(
        scheduled_tokens=8,
        num_prompt_tokens=128,
        num_computed_tokens_after_schedule=132,
    )
    assert prompt == 4
    assert output == 4


def test_decode_only_schedule() -> None:
    prompt, output = split_scheduled_tokens(
        scheduled_tokens=1,
        num_prompt_tokens=128,
        num_computed_tokens_after_schedule=160,
    )
    assert prompt == 0
    assert output == 1


def test_kv_usage_validation() -> None:
    assert KVObservation(step_id=1, usage_ratio=0.75).usage_ratio == 0.75
