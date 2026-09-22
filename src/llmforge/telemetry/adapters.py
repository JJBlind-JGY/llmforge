"""Cross-milestone adapters into the normalized M6 metric vocabulary."""

from __future__ import annotations

from typing import Protocol

from .metrics import MetricRegistry


class ServingResultLike(Protocol):
    prompt_tokens: int
    output_tokens: int
    ttft_ms: float | None
    tpot_ms: float | None
    e2e_ms: float
    success: bool


class RuntimeRequestLike(Protocol):
    queue_wait_ms: float | None


class RuntimeSummaryLike(Protocol):
    requests: tuple
    kv_usage_ratio_max: float | None


class GpuSampleLike(Protocol):
    index: int
    utilization_gpu_percent: float
    memory_used_mib: float
    memory_total_mib: float
    power_draw_w: float


def observe_serving_result(
    registry: MetricRegistry,
    result: ServingResultLike,
) -> None:
    """Record M3 client-observed request metrics.

    Client semaphore wait is intentionally not mapped to queue_wait: M6 keeps
    client scheduling delay separate from server/runtime queueing.
    """

    status = "success" if result.success else "error"

    registry.inc_counter(
        "llmforge_request_total",
        labels={
            "source": "client",
            "status": status,
        },
    )

    registry.observe_histogram(
        "llmforge_input_tokens",
        float(result.prompt_tokens),
        labels={
            "source": "client",
        },
    )

    registry.observe_histogram(
        "llmforge_output_tokens",
        float(result.output_tokens),
        labels={
            "source": "client",
        },
    )

    registry.inc_counter(
        "llmforge_input_tokens_total",
        float(result.prompt_tokens),
        labels={
            "source": "client",
        },
    )

    registry.inc_counter(
        "llmforge_output_tokens_total",
        float(result.output_tokens),
        labels={
            "source": "client",
        },
    )

    if result.ttft_ms is not None:
        registry.observe_histogram(
            "llmforge_ttft_seconds",
            result.ttft_ms / 1000.0,
            labels={
                "source": "client",
            },
        )

    if result.tpot_ms is not None:
        registry.observe_histogram(
            "llmforge_tpot_seconds",
            result.tpot_ms / 1000.0,
            labels={
                "source": "client",
            },
        )

    registry.observe_histogram(
        "llmforge_e2e_seconds",
        result.e2e_ms / 1000.0,
        labels={
            "source": "client",
        },
    )


def observe_runtime_summary(
    registry: MetricRegistry,
    summary: RuntimeSummaryLike,
) -> None:
    """Record M4 engine-side queue/KV observations."""

    for request in summary.requests:
        queue_wait_ms = getattr(
            request,
            "queue_wait_ms",
            None,
        )

        if queue_wait_ms is not None:
            registry.observe_histogram(
                "llmforge_queue_wait_seconds",
                float(queue_wait_ms) / 1000.0,
                labels={
                    "source": "runtime",
                },
            )

    if summary.kv_usage_ratio_max is not None:
        registry.set_gauge(
            "llmforge_kv_usage_ratio",
            float(summary.kv_usage_ratio_max),
            labels={
                "source": "runtime",
            },
        )


def observe_gpu_sample(
    registry: MetricRegistry,
    sample: GpuSampleLike,
) -> None:
    labels = {
        "gpu": str(sample.index),
    }

    registry.set_gauge(
        "llmforge_gpu_utilization_ratio",
        float(sample.utilization_gpu_percent) / 100.0,
        labels=labels,
    )

    registry.set_gauge(
        "llmforge_gpu_memory_bytes",
        float(sample.memory_used_mib) * 1024.0 * 1024.0,
        labels=labels,
    )

    registry.set_gauge(
        "llmforge_gpu_memory_total_bytes",
        float(sample.memory_total_mib) * 1024.0 * 1024.0,
        labels=labels,
    )

    registry.set_gauge(
        "llmforge_gpu_power_watts",
        float(sample.power_draw_w),
        labels=labels,
    )
