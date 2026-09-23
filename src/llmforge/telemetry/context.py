"""Request/trace context propagation via contextvars."""

from __future__ import annotations

import secrets
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass


@dataclass(frozen=True)
class RequestContext:
    request_id: str
    trace_id: str
    parent_span_id: str | None = None


_CURRENT_CONTEXT: ContextVar[RequestContext | None] = ContextVar(
    "llmforge_request_context",
    default=None,
)


def new_request_id() -> str:
    return "req_" + secrets.token_hex(8)


def new_trace_id() -> str:
    # W3C trace-id is 16 bytes / 32 hex characters.
    return secrets.token_hex(16)


def current_request_context() -> RequestContext | None:
    return _CURRENT_CONTEXT.get()


@contextmanager
def bind_request_context(
    *,
    request_id: str | None = None,
    trace_id: str | None = None,
    parent_span_id: str | None = None,
) -> Iterator[RequestContext]:
    context = RequestContext(
        request_id=request_id or new_request_id(),
        trace_id=trace_id or new_trace_id(),
        parent_span_id=parent_span_id,
    )

    token: Token[RequestContext | None] = _CURRENT_CONTEXT.set(context)

    try:
        yield context
    finally:
        _CURRENT_CONTEXT.reset(token)
