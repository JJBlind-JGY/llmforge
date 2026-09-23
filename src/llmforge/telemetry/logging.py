"""Structured JSON logging with request/trace correlation."""

from __future__ import annotations

import json
import logging
import time
from datetime import UTC, datetime
from typing import Any

from .context import (
    current_request_context,
)

_RESERVED_RECORD_KEYS = {
    "name",
    "msg",
    "args",
    "levelname",
    "levelno",
    "pathname",
    "filename",
    "module",
    "exc_info",
    "exc_text",
    "stack_info",
    "lineno",
    "funcName",
    "created",
    "msecs",
    "relativeCreated",
    "thread",
    "threadName",
    "processName",
    "process",
    "message",
    "taskName",
}


class RequestContextFilter(logging.Filter):
    def filter(
        self,
        record: logging.LogRecord,
    ) -> bool:
        context = current_request_context()

        record.request_id = context.request_id if context else None

        record.trace_id = context.trace_id if context else None

        return True


class JsonFormatter(logging.Formatter):
    def format(
        self,
        record: logging.LogRecord,
    ) -> str:
        timestamp = datetime.fromtimestamp(
            record.created,
            tz=UTC,
        ).isoformat()

        payload: dict[
            str,
            Any,
        ] = {
            "timestamp": timestamp,
            "timestamp_ns": (time.time_ns()),
            "level": record.levelname,
            "logger": record.name,
            "message": (record.getMessage()),
            "request_id": getattr(
                record,
                "request_id",
                None,
            ),
            "trace_id": getattr(
                record,
                "trace_id",
                None,
            ),
        }

        component = getattr(
            record,
            "component",
            None,
        )

        if component is not None:
            payload["component"] = component

        extras = {}

        for key, value in record.__dict__.items():
            if key in _RESERVED_RECORD_KEYS or key in {
                "request_id",
                "trace_id",
                "component",
            }:
                continue

            try:
                json.dumps(value)
                extras[key] = value
            except (
                TypeError,
                ValueError,
            ):
                extras[key] = repr(value)

        if extras:
            payload["fields"] = extras

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        )


def configure_json_logging(
    *,
    level: str = "INFO",
) -> None:
    numeric_level = getattr(
        logging,
        level.upper(),
        None,
    )

    if not isinstance(
        numeric_level,
        int,
    ):
        raise ValueError(f"Unsupported log level: {level!r}.")

    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())

    context_filter = RequestContextFilter()

    handler.addFilter(context_filter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(numeric_level)
