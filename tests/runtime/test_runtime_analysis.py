from llmforge.runtime.analysis import summarize_runtime_trace
from llmforge.runtime.events import RuntimeEvent, RuntimeEventKind


def event(
    *,
    kind: RuntimeEventKind,
    t_ms: int,
    request_id: str | None = None,
    step_id: int | None = None,
    payload: dict | None = None,
) -> RuntimeEvent:
    ns = t_ms * 1_000_000
    return RuntimeEvent(1, kind, ns, ns, 1, 1, step_id, request_id, payload or {})


def test_runtime_summary() -> None:
    events = [
        event(kind=RuntimeEventKind.REQUEST_ENQUEUED, t_ms=0, request_id="r1"),
        event(
            kind=RuntimeEventKind.REQUEST_SCHEDULED,
            t_ms=10,
            request_id="r1",
            step_id=1,
            payload={"scheduled_prompt_tokens": 128, "scheduled_output_tokens": 0},
        ),
        event(
            kind=RuntimeEventKind.SCHEDULER_STEP,
            t_ms=10,
            step_id=1,
            payload={"schedule_duration_us": 100.0, "kv_usage_ratio": 0.25},
        ),
        event(
            kind=RuntimeEventKind.REQUEST_SCHEDULED,
            t_ms=20,
            request_id="r1",
            step_id=2,
            payload={"scheduled_prompt_tokens": 0, "scheduled_output_tokens": 1},
        ),
        event(kind=RuntimeEventKind.REQUEST_FINISHED, t_ms=30, request_id="r1"),
    ]
    summary = summarize_runtime_trace(events)
    request = summary.requests[0]
    assert request.queue_wait_ms == 10.0
    assert request.lifetime_ms == 30.0
    assert request.scheduled_steps == 2
    assert request.scheduled_prompt_tokens == 128
    assert request.scheduled_output_tokens == 1
    assert summary.scheduler_steps == 1
    assert summary.kv_usage_ratio_max == 0.25
