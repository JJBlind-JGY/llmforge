from types import SimpleNamespace

from llmforge.telemetry.adapters import (
    observe_runtime_summary,
    observe_serving_result,
)
from llmforge.telemetry.metrics import (
    create_default_registry,
)


def test_serving_adapter_keeps_client_metrics() -> None:
    registry = create_default_registry()

    result = SimpleNamespace(
        prompt_tokens=512,
        output_tokens=32,
        ttft_ms=20.0,
        tpot_ms=10.0,
        e2e_ms=330.0,
        success=True,
    )

    observe_serving_result(
        registry,
        result,
    )

    snapshot = registry.snapshot()

    names = {item["name"] for item in snapshot["histograms"]}

    assert "llmforge_ttft_seconds" in names

    assert "llmforge_e2e_seconds" in names


def test_runtime_adapter_records_queue_wait() -> None:
    registry = create_default_registry()

    summary = SimpleNamespace(
        requests=(SimpleNamespace(queue_wait_ms=12.0),),
        kv_usage_ratio_max=0.75,
    )

    observe_runtime_summary(
        registry,
        summary,
    )

    snapshot = registry.snapshot()

    queue_histogram = next(
        item
        for item in snapshot["histograms"]
        if item["name"] == "llmforge_queue_wait_seconds"
    )

    assert queue_histogram["count"] == 1
