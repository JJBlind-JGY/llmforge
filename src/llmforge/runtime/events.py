"""Version-stable runtime event schema."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class RuntimeEventKind(StrEnum):
    REQUEST_ENQUEUED = "request_enqueued"
    REQUEST_SCHEDULED = "request_scheduled"
    SCHEDULER_STEP = "scheduler_step"
    REQUEST_PREEMPTED = "request_preempted"
    MODEL_OUTPUT_PROCESSED = "model_output_processed"
    REQUEST_FINISHED = "request_finished"
    TRACE_DROPPED = "trace_dropped"


@dataclass(frozen=True)
class ClockStamp:
    wall_time_ns: int
    monotonic_ns: int


def clock_now() -> ClockStamp:
    return ClockStamp(
        wall_time_ns=time.time_ns(),
        monotonic_ns=time.perf_counter_ns(),
    )


@dataclass(frozen=True)
class RuntimeEvent:
    schema_version: int
    kind: RuntimeEventKind
    wall_time_ns: int
    monotonic_ns: int
    process_id: int
    thread_id: int
    step_id: int | None = None
    request_id: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["kind"] = self.kind.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RuntimeEvent":
        return cls(
            schema_version=int(data["schema_version"]),
            kind=RuntimeEventKind(data["kind"]),
            wall_time_ns=int(data["wall_time_ns"]),
            monotonic_ns=int(data["monotonic_ns"]),
            process_id=int(data["process_id"]),
            thread_id=int(data["thread_id"]),
            step_id=data.get("step_id"),
            request_id=data.get("request_id"),
            payload=dict(data.get("payload", {})),
        )
