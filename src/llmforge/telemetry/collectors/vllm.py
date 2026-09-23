"""Read a small stable subset of vLLM's Prometheus endpoint."""

from __future__ import annotations

import re
import urllib.request
from dataclasses import dataclass

from llmforge.telemetry.metrics import (
    MetricRegistry,
)

_SAMPLE_RE = re.compile(
    r"^([a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"(?:\{[^}]*\})?\s+"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)"
    r"(?:[eE][-+]?\d+)?)$"
)


def parse_prometheus_scalars(
    text: str,
) -> dict[
    str,
    list[float],
]:
    values: dict[
        str,
        list[float],
    ] = {}

    for raw_line in text.splitlines():
        line = raw_line.strip()

        if not line or line.startswith("#"):
            continue

        match = _SAMPLE_RE.fullmatch(line)

        if match is None:
            continue

        name = match.group(1)
        value = float(match.group(2))

        values.setdefault(
            name,
            [],
        ).append(value)

    return values


@dataclass(frozen=True)
class VLLMMetricSnapshot:
    running: float | None
    waiting: float | None
    kv_usage_ratio: float | None
    prefix_queries: float | None
    prefix_hits: float | None
    prompt_tokens_total: float | None
    generation_tokens_total: float | None
    request_success_total: float | None


def _sum(
    values: dict[
        str,
        list[float],
    ],
    name: str,
) -> float | None:
    samples = values.get(name)

    if not samples:
        return None

    return sum(samples)


def _max(
    values: dict[
        str,
        list[float],
    ],
    name: str,
) -> float | None:
    samples = values.get(name)

    if not samples:
        return None

    return max(samples)


def snapshot_from_prometheus(
    text: str,
) -> VLLMMetricSnapshot:
    values = parse_prometheus_scalars(text)

    return VLLMMetricSnapshot(
        running=_sum(
            values,
            "vllm:num_requests_running",
        ),
        waiting=_sum(
            values,
            "vllm:num_requests_waiting",
        ),
        kv_usage_ratio=_max(
            values,
            "vllm:kv_cache_usage_perc",
        ),
        prefix_queries=_sum(
            values,
            "vllm:prefix_cache_queries",
        ),
        prefix_hits=_sum(
            values,
            "vllm:prefix_cache_hits",
        ),
        prompt_tokens_total=_sum(
            values,
            "vllm:prompt_tokens_total",
        ),
        generation_tokens_total=_sum(
            values,
            "vllm:generation_tokens_total",
        ),
        request_success_total=_sum(
            values,
            "vllm:request_success_total",
        ),
    )


class VLLMMetricsCollector:
    name = "vllm"

    def __init__(
        self,
        *,
        base_url: str,
        timeout_s: float,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def _fetch(self) -> str:
        with urllib.request.urlopen(
            self.base_url + "/metrics",
            timeout=self.timeout_s,
        ) as response:
            return response.read().decode(
                "utf-8",
                errors="replace",
            )

    def collect(
        self,
        registry: MetricRegistry,
    ) -> None:
        snapshot = snapshot_from_prometheus(self._fetch())

        if snapshot.running is not None:
            registry.set_gauge(
                "llmforge_request_running",
                snapshot.running,
                labels={
                    "source": "vllm",
                },
            )

        if snapshot.waiting is not None:
            registry.set_gauge(
                "llmforge_request_waiting",
                snapshot.waiting,
                labels={
                    "source": "vllm",
                },
            )

        if snapshot.kv_usage_ratio is not None:
            registry.set_gauge(
                "llmforge_kv_usage_ratio",
                snapshot.kv_usage_ratio,
                labels={
                    "source": "vllm",
                },
            )

        if snapshot.prefix_queries is not None:
            registry.set_counter_absolute(
                "llmforge_prefix_cache_query_total",
                snapshot.prefix_queries,
                labels={
                    "source": "vllm",
                },
            )

        if snapshot.prefix_hits is not None:
            registry.set_counter_absolute(
                "llmforge_prefix_cache_hit_total",
                snapshot.prefix_hits,
                labels={
                    "source": "vllm",
                },
            )

        if snapshot.prompt_tokens_total is not None:
            registry.set_counter_absolute(
                "llmforge_input_tokens_total",
                snapshot.prompt_tokens_total,
                labels={
                    "source": "vllm",
                },
            )

        if snapshot.generation_tokens_total is not None:
            registry.set_counter_absolute(
                "llmforge_output_tokens_total",
                snapshot.generation_tokens_total,
                labels={
                    "source": "vllm",
                },
            )

        if snapshot.request_success_total is not None:
            registry.set_counter_absolute(
                "llmforge_request_total",
                snapshot.request_success_total,
                labels={
                    "source": "vllm",
                    "status": "success",
                },
            )
