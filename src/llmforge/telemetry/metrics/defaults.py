"""Normalized metric vocabulary required by the M6 syllabus."""

from __future__ import annotations

from .registry import MetricRegistry
from .schema import (
    MetricKind,
    MetricSpec,
)

_LATENCY_BUCKETS_S = (
    0.001,
    0.002,
    0.005,
    0.010,
    0.020,
    0.050,
    0.100,
    0.200,
    0.500,
    1.0,
    2.0,
    5.0,
    10.0,
    30.0,
    60.0,
)

_TOKEN_BUCKETS = (
    1.0,
    8.0,
    16.0,
    32.0,
    64.0,
    128.0,
    256.0,
    512.0,
    1024.0,
    2048.0,
    4096.0,
    8192.0,
    16384.0,
    32768.0,
)


def create_default_registry() -> MetricRegistry:
    registry = MetricRegistry()

    specs = (
        MetricSpec(
            name=("llmforge_request_total"),
            kind=MetricKind.COUNTER,
            help=("Total requests observed by LLMForge."),
            label_names=(
                "source",
                "status",
            ),
        ),
        MetricSpec(
            name=("llmforge_request_running"),
            kind=MetricKind.GAUGE,
            help=("Requests currently running."),
            label_names=("source",),
        ),
        MetricSpec(
            name=("llmforge_request_waiting"),
            kind=MetricKind.GAUGE,
            help=("Requests currently waiting."),
            label_names=("source",),
        ),
        MetricSpec(
            name=("llmforge_queue_wait_seconds"),
            kind=MetricKind.HISTOGRAM,
            help=("Request queue wait time."),
            label_names=("source",),
            buckets=_LATENCY_BUCKETS_S,
        ),
        MetricSpec(
            name=("llmforge_ttft_seconds"),
            kind=MetricKind.HISTOGRAM,
            help=("Time to first token."),
            label_names=("source",),
            buckets=_LATENCY_BUCKETS_S,
        ),
        MetricSpec(
            name=("llmforge_tpot_seconds"),
            kind=MetricKind.HISTOGRAM,
            help=("Request-level time per output token."),
            label_names=("source",),
            buckets=_LATENCY_BUCKETS_S,
        ),
        MetricSpec(
            name=("llmforge_e2e_seconds"),
            kind=MetricKind.HISTOGRAM,
            help=("End-to-end request latency."),
            label_names=("source",),
            buckets=_LATENCY_BUCKETS_S,
        ),
        MetricSpec(
            name=("llmforge_input_tokens"),
            kind=MetricKind.HISTOGRAM,
            help=("Input tokens per request."),
            label_names=("source",),
            buckets=_TOKEN_BUCKETS,
        ),
        MetricSpec(
            name=("llmforge_output_tokens"),
            kind=MetricKind.HISTOGRAM,
            help=("Output tokens per request."),
            label_names=("source",),
            buckets=_TOKEN_BUCKETS,
        ),
        MetricSpec(
            name=("llmforge_input_tokens_total"),
            kind=MetricKind.COUNTER,
            help=("Total input tokens."),
            label_names=("source",),
        ),
        MetricSpec(
            name=("llmforge_output_tokens_total"),
            kind=MetricKind.COUNTER,
            help=("Total output tokens."),
            label_names=("source",),
        ),
        MetricSpec(
            name=("llmforge_kv_usage_ratio"),
            kind=MetricKind.GAUGE,
            help=("Fraction of KV cache capacity currently used."),
            label_names=("source",),
        ),
        MetricSpec(
            name=("llmforge_prefix_cache_query_total"),
            kind=MetricKind.COUNTER,
            help=("Total prefix-cache queries."),
            label_names=("source",),
        ),
        MetricSpec(
            name=("llmforge_prefix_cache_hit_total"),
            kind=MetricKind.COUNTER,
            help=("Total prefix-cache hits."),
            label_names=("source",),
        ),
        MetricSpec(
            name=("llmforge_gpu_utilization_ratio"),
            kind=MetricKind.GAUGE,
            help=("GPU utilization as a 0-1 ratio."),
            label_names=("gpu",),
        ),
        MetricSpec(
            name=("llmforge_gpu_memory_bytes"),
            kind=MetricKind.GAUGE,
            help=("GPU memory currently used in bytes."),
            label_names=("gpu",),
        ),
        MetricSpec(
            name=("llmforge_gpu_memory_total_bytes"),
            kind=MetricKind.GAUGE,
            help=("Total GPU memory in bytes."),
            label_names=("gpu",),
        ),
        MetricSpec(
            name=("llmforge_gpu_power_watts"),
            kind=MetricKind.GAUGE,
            help=("GPU power draw in watts."),
            label_names=("gpu",),
        ),
        MetricSpec(
            name=("llmforge_observer_error_total"),
            kind=MetricKind.COUNTER,
            help=("Observability collector errors."),
            label_names=("collector",),
        ),
    )

    for spec in specs:
        registry.register(spec)

    return registry
