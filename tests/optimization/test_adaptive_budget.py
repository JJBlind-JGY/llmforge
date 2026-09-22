from types import SimpleNamespace

from llmforge.optimization.candidates.adaptive_budget import (
    AdaptiveBudgetConfig,
    SchedulerPressure,
    choose_budget,
    count_request_phases,
)


def test_phase_counting() -> None:
    requests = [
        SimpleNamespace(
            num_prompt_tokens=128,
            num_computed_tokens=64,
        ),
        SimpleNamespace(
            num_prompt_tokens=128,
            num_computed_tokens=128,
        ),
        SimpleNamespace(
            num_prompt_tokens=128,
            num_computed_tokens=140,
        ),
    ]

    prefill, decode = count_request_phases(requests)

    assert prefill == 1
    assert decode == 2


def test_policy_keeps_baseline_without_mixed_work() -> None:
    decision = choose_budget(
        config=(AdaptiveBudgetConfig()),
        base_budget=2048,
        pressure=SchedulerPressure(
            running_prefill=0,
            running_decode=8,
            waiting_prefill=0,
            waiting_decode=0,
            kv_usage_ratio=0.5,
        ),
    )

    assert decision.applied_budget == 2048

    assert not decision.changed


def test_policy_reduces_budget_under_decode_pressure() -> None:
    decision = choose_budget(
        config=AdaptiveBudgetConfig(
            min_budget=256,
            prefill_quantum=256,
            decode_pressure=4,
            kv_pressure=0.9,
        ),
        base_budget=2048,
        pressure=SchedulerPressure(
            running_prefill=1,
            running_decode=8,
            waiting_prefill=2,
            waiting_decode=0,
            kv_usage_ratio=0.5,
        ),
    )

    assert decision.applied_budget == 264

    assert decision.changed
    assert decision.reason == "decode_pressure"


def test_policy_caps_at_base_budget() -> None:
    decision = choose_budget(
        config=AdaptiveBudgetConfig(
            min_budget=256,
            prefill_quantum=4096,
            decode_pressure=1,
            kv_pressure=0.9,
        ),
        base_budget=1024,
        pressure=SchedulerPressure(
            running_prefill=1,
            running_decode=2,
            waiting_prefill=1,
            waiting_decode=0,
            kv_usage_ratio=0.5,
        ),
    )

    assert decision.applied_budget == 1024
