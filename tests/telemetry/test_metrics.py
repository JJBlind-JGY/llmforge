import pytest

from llmforge.telemetry.metrics import (
    MetricKind,
    MetricRegistry,
    MetricSpec,
    create_default_registry,
)


def test_histogram_is_bounded_and_cumulative() -> None:
    registry = MetricRegistry()

    registry.register(
        MetricSpec(
            name="latency_seconds",
            kind=MetricKind.HISTOGRAM,
            help="Latency.",
            label_names=("source",),
            buckets=(
                0.1,
                1.0,
            ),
        )
    )

    for value in (
        0.05,
        0.5,
        2.0,
    ):
        registry.observe_histogram(
            "latency_seconds",
            value,
            labels={
                "source": "test",
            },
        )

    snapshot = registry.snapshot()

    histogram = snapshot["histograms"][0]

    assert histogram["bucket_counts"] == [1, 2]

    assert histogram["count"] == 3

    text = registry.prometheus_text()

    assert 'latency_seconds_bucket{source="test",le="0.1"} 1' in text

    assert 'latency_seconds_bucket{source="test",le="1.0"} 2' in text

    assert 'latency_seconds_bucket{source="test",le="+Inf"} 3' in text


def test_counter_rejects_negative_delta() -> None:
    registry = create_default_registry()

    with pytest.raises(ValueError):
        registry.inc_counter(
            "llmforge_request_total",
            -1,
            labels={
                "source": "client",
                "status": "ok",
            },
        )


def test_default_registry_has_m6_metrics() -> None:
    snapshot = create_default_registry().snapshot()

    names = set(snapshot["specs"])

    required = {
        "llmforge_request_total",
        "llmforge_request_running",
        "llmforge_request_waiting",
        "llmforge_queue_wait_seconds",
        "llmforge_ttft_seconds",
        "llmforge_tpot_seconds",
        "llmforge_e2e_seconds",
        "llmforge_input_tokens",
        "llmforge_output_tokens",
        "llmforge_kv_usage_ratio",
        "llmforge_prefix_cache_hit_total",
        "llmforge_gpu_utilization_ratio",
        "llmforge_gpu_memory_bytes",
    }

    assert required <= names
