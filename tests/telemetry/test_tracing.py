import json
from pathlib import Path

from llmforge.telemetry.context import (
    bind_request_context,
)
from llmforge.telemetry.tracing import (
    BufferedJsonlSpanExporter,
    SpanManager,
)


def test_nested_spans_share_trace_and_parent(
    tmp_path: Path,
) -> None:
    path = tmp_path / "spans.jsonl"

    with BufferedJsonlSpanExporter(
        path,
        queue_size=64,
        flush_every=1,
    ) as exporter:
        manager = SpanManager(exporter)

        with bind_request_context(
            request_id="req-1",
            trace_id="a" * 32,
        ), manager.start_span("outer") as outer:
            with manager.start_span("inner") as inner:
                inner.set_attribute(
                    "tokens",
                    32,
                )

            assert inner.parent_span_id == outer.span_id

        manager.flush()

    records = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
    ]

    assert len(records) == 2
    assert {record["trace_id"] for record in records} == {"a" * 32}

    by_name = {record["name"]: record for record in records}

    assert by_name["inner"]["parent_span_id"] == by_name["outer"]["span_id"]
