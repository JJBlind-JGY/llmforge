from llmforge.telemetry.context import (
    bind_request_context,
    current_request_context,
)


def test_request_context_is_scoped() -> None:
    assert current_request_context() is None

    with bind_request_context(
        request_id="req-test",
        trace_id="0" * 32,
    ) as context:
        assert current_request_context() == context

    assert current_request_context() is None
