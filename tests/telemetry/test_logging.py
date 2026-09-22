import json
import logging

from llmforge.telemetry.context import (
    bind_request_context,
)
from llmforge.telemetry.logging import (
    JsonFormatter,
    RequestContextFilter,
)


def test_json_log_has_request_context() -> None:
    record = logging.LogRecord(
        name="llmforge.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello",
        args=(),
        exc_info=None,
    )

    request_filter = RequestContextFilter()

    with bind_request_context(
        request_id="req-1",
        trace_id="1" * 32,
    ):
        assert request_filter.filter(record)

        payload = json.loads(JsonFormatter().format(record))

    assert payload["request_id"] == "req-1"

    assert payload["trace_id"] == "1" * 32

    assert payload["message"] == "hello"
