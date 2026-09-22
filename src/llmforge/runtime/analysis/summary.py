from __future__ import annotations
from collections import defaultdict
from dataclasses import asdict, dataclass
from llmforge.runtime.events import RuntimeEvent, RuntimeEventKind
from .stats import distribution


@dataclass(frozen=True)
class RequestLifecycleSummary:
    request_id: str
    queue_wait_ms: float | None
    lifetime_ms: float | None
    scheduled_steps: int
    scheduled_prompt_tokens: int
    scheduled_output_tokens: int
    preemptions: int
    finished: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RuntimeTraceSummary:
    requests: tuple[RequestLifecycleSummary, ...]
    scheduler_steps: int
    scheduler_duration_us: dict
    queue_wait_ms: dict
    request_lifetime_ms: dict
    kv_usage_ratio_max: float | None
    preemptions_total: int
    scheduled_prompt_tokens: int
    scheduled_output_tokens: int
    trace_drop_events: int

    def to_dict(self) -> dict:
        return {
            "requests": [r.to_dict() for r in self.requests],
            "scheduler_steps": self.scheduler_steps,
            "scheduler_duration_us": self.scheduler_duration_us,
            "queue_wait_ms": self.queue_wait_ms,
            "request_lifetime_ms": self.request_lifetime_ms,
            "kv_usage_ratio_max": self.kv_usage_ratio_max,
            "preemptions_total": self.preemptions_total,
            "scheduled_prompt_tokens": self.scheduled_prompt_tokens,
            "scheduled_output_tokens": self.scheduled_output_tokens,
            "trace_drop_events": self.trace_drop_events,
        }


def summarize_runtime_trace(events: list[RuntimeEvent]) -> RuntimeTraceSummary:
    by_request: dict[str, list[RuntimeEvent]] = defaultdict(list)
    scheduler_events = []
    trace_drop_events = 0
    for event in events:
        if event.request_id is not None:
            by_request[event.request_id].append(event)
        if event.kind == RuntimeEventKind.SCHEDULER_STEP:
            scheduler_events.append(event)
        if event.kind == RuntimeEventKind.TRACE_DROPPED:
            trace_drop_events += int(event.payload.get("dropped_events", 1))

    requests = []
    for request_id, request_events in by_request.items():
        ordered = sorted(request_events, key=lambda e: e.monotonic_ns)
        enqueue = next(
            (e for e in ordered if e.kind == RuntimeEventKind.REQUEST_ENQUEUED), None
        )
        schedules = [e for e in ordered if e.kind == RuntimeEventKind.REQUEST_SCHEDULED]
        first_schedule = schedules[0] if schedules else None
        finish = next(
            (
                e
                for e in reversed(ordered)
                if e.kind == RuntimeEventKind.REQUEST_FINISHED
            ),
            None,
        )
        queue_wait_ms = (
            None
            if enqueue is None or first_schedule is None
            else (first_schedule.monotonic_ns - enqueue.monotonic_ns) / 1e6
        )
        lifetime_ms = (
            None
            if enqueue is None or finish is None
            else (finish.monotonic_ns - enqueue.monotonic_ns) / 1e6
        )
        requests.append(
            RequestLifecycleSummary(
                request_id=request_id,
                queue_wait_ms=queue_wait_ms,
                lifetime_ms=lifetime_ms,
                scheduled_steps=len(schedules),
                scheduled_prompt_tokens=sum(
                    int(e.payload.get("scheduled_prompt_tokens", 0)) for e in schedules
                ),
                scheduled_output_tokens=sum(
                    int(e.payload.get("scheduled_output_tokens", 0)) for e in schedules
                ),
                preemptions=sum(
                    e.kind == RuntimeEventKind.REQUEST_PREEMPTED for e in ordered
                ),
                finished=finish is not None,
            )
        )
    requests.sort(key=lambda r: r.request_id)
    scheduler_durations = [
        float(e.payload.get("schedule_duration_us", 0.0)) for e in scheduler_events
    ]
    kv_usage = [
        float(e.payload["kv_usage_ratio"])
        for e in scheduler_events
        if "kv_usage_ratio" in e.payload
    ]
    return RuntimeTraceSummary(
        requests=tuple(requests),
        scheduler_steps=len(scheduler_events),
        scheduler_duration_us=distribution(scheduler_durations),
        queue_wait_ms=distribution(
            [r.queue_wait_ms for r in requests if r.queue_wait_ms is not None]
        ),
        request_lifetime_ms=distribution(
            [r.lifetime_ms for r in requests if r.lifetime_ms is not None]
        ),
        kv_usage_ratio_max=max(kv_usage) if kv_usage else None,
        preemptions_total=sum(r.preemptions for r in requests),
        scheduled_prompt_tokens=sum(r.scheduled_prompt_tokens for r in requests),
        scheduled_output_tokens=sum(r.scheduled_output_tokens for r in requests),
        trace_drop_events=trace_drop_events,
    )
