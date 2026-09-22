"""Request-correlated application spans."""

from __future__ import annotations

import secrets
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from typing import Any, Iterator, Protocol

from llmforge.telemetry.context import (
    current_request_context,
    new_trace_id,
)


_CURRENT_SPAN_ID: ContextVar[str | None] = ContextVar(
    "llmforge_current_span_id",
    default=None,
)


@dataclass(frozen=True)
class SpanRecord:
    schema_version: int
    trace_id: str
    span_id: str
    parent_span_id: str | None
    name: str
    start_wall_time_ns: int
    end_wall_time_ns: int
    start_monotonic_ns: int
    end_monotonic_ns: int
    status: str
    attributes: dict[str, Any]

    @property
    def duration_ms(
        self,
    ) -> float:
        return (self.end_monotonic_ns - self.start_monotonic_ns) / 1e6

    def to_dict(self) -> dict:
        data = asdict(self)
        data["duration_ms"] = self.duration_ms
        return data


class SpanExporter(Protocol):
    def export(
        self,
        span: SpanRecord,
    ) -> bool: ...

    def flush(self) -> None: ...

    def close(self) -> None: ...


class SpanHandle:
    def __init__(
        self,
        *,
        name: str,
        trace_id: str,
        span_id: str,
        parent_span_id: str | None,
        attributes: dict[
            str,
            Any,
        ]
        | None,
    ) -> None:
        self.name = name
        self.trace_id = trace_id
        self.span_id = span_id
        self.parent_span_id = parent_span_id
        self.attributes = dict(attributes or {})
        self.status = "ok"

        self.start_wall_time_ns = time.time_ns()

        self.start_monotonic_ns = time.perf_counter_ns()

    def set_attribute(
        self,
        key: str,
        value: Any,
    ) -> None:
        self.attributes[key] = value

    def set_status(
        self,
        status: str,
    ) -> None:
        self.status = status

    def finish(
        self,
    ) -> SpanRecord:
        return SpanRecord(
            schema_version=1,
            trace_id=self.trace_id,
            span_id=self.span_id,
            parent_span_id=(self.parent_span_id),
            name=self.name,
            start_wall_time_ns=(self.start_wall_time_ns),
            end_wall_time_ns=(time.time_ns()),
            start_monotonic_ns=(self.start_monotonic_ns),
            end_monotonic_ns=(time.perf_counter_ns()),
            status=self.status,
            attributes=dict(self.attributes),
        )


class SpanManager:
    def __init__(
        self,
        exporter: SpanExporter,
    ) -> None:
        self.exporter = exporter

    @contextmanager
    def start_span(
        self,
        name: str,
        *,
        attributes: dict[
            str,
            Any,
        ]
        | None = None,
    ) -> Iterator[SpanHandle]:
        request_context = current_request_context()

        trace_id = request_context.trace_id if request_context else new_trace_id()

        parent_span_id = _CURRENT_SPAN_ID.get()

        if parent_span_id is None and request_context is not None:
            parent_span_id = request_context.parent_span_id

        handle = SpanHandle(
            name=name,
            trace_id=trace_id,
            span_id=(secrets.token_hex(8)),
            parent_span_id=(parent_span_id),
            attributes=attributes,
        )

        token = _CURRENT_SPAN_ID.set(handle.span_id)

        try:
            yield handle
        except BaseException as exc:
            handle.set_status("error")

            handle.set_attribute(
                "exception.type",
                type(exc).__name__,
            )

            handle.set_attribute(
                "exception.message",
                str(exc),
            )

            raise
        finally:
            _CURRENT_SPAN_ID.reset(token)

            self.exporter.export(handle.finish())

    def flush(self) -> None:
        self.exporter.flush()

    def close(self) -> None:
        self.exporter.close()
