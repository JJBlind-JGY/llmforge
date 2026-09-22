"""Normalize M3 serving artifacts for M5 scaling comparisons."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ServingRun:
    profile: str
    devices: int
    request_throughput_rps: float
    output_throughput_tok_s: float
    total_throughput_tok_s: float
    ttft_p50_ms: float | None
    ttft_p95_ms: float | None
    ttft_p99_ms: float | None
    tpot_p50_ms: float | None
    tpot_p95_ms: float | None
    tpot_p99_ms: float | None
    e2e_p50_ms: float | None
    e2e_p95_ms: float | None
    e2e_p99_ms: float | None
    artifact_path: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ScalingComparison:
    profile: str
    devices: int
    request_throughput_speedup: float
    output_throughput_speedup: float
    output_throughput_efficiency: float
    ttft_p50_ratio: float | None
    ttft_p99_ratio: float | None
    tpot_p50_ratio: float | None
    tpot_p99_ratio: float | None
    e2e_p50_ratio: float | None
    e2e_p99_ratio: float | None

    def to_dict(self) -> dict:
        return asdict(self)


def _distribution_value(
    summary: dict[str, Any],
    metric: str,
    percentile: str,
) -> float | None:
    distribution = summary.get(metric)

    if not isinstance(
        distribution,
        dict,
    ):
        return None

    value = distribution.get(percentile)

    return None if value is None else float(value)


def load_m3_serving_run(
    *,
    path: Path,
    profile: str,
    devices: int,
) -> ServingRun:
    payload = json.loads(path.read_text(encoding="utf-8"))

    summary = payload["summary"]

    return ServingRun(
        profile=profile,
        devices=devices,
        request_throughput_rps=float(summary["request_throughput_rps"]),
        output_throughput_tok_s=float(summary["output_throughput_tok_s"]),
        total_throughput_tok_s=float(summary["total_throughput_tok_s"]),
        ttft_p50_ms=_distribution_value(
            summary,
            "ttft_ms",
            "p50",
        ),
        ttft_p95_ms=_distribution_value(
            summary,
            "ttft_ms",
            "p95",
        ),
        ttft_p99_ms=_distribution_value(
            summary,
            "ttft_ms",
            "p99",
        ),
        tpot_p50_ms=_distribution_value(
            summary,
            "tpot_ms",
            "p50",
        ),
        tpot_p95_ms=_distribution_value(
            summary,
            "tpot_ms",
            "p95",
        ),
        tpot_p99_ms=_distribution_value(
            summary,
            "tpot_ms",
            "p99",
        ),
        e2e_p50_ms=_distribution_value(
            summary,
            "e2e_ms",
            "p50",
        ),
        e2e_p95_ms=_distribution_value(
            summary,
            "e2e_ms",
            "p95",
        ),
        e2e_p99_ms=_distribution_value(
            summary,
            "e2e_ms",
            "p99",
        ),
        artifact_path=str(path),
    )


def _ratio(
    candidate: float | None,
    baseline: float | None,
) -> float | None:
    if candidate is None or baseline is None or baseline == 0:
        return None

    return candidate / baseline


def compare_to_baseline(
    *,
    baseline: ServingRun,
    candidate: ServingRun,
) -> ScalingComparison:
    if baseline.devices <= 0:
        raise ValueError("baseline.devices must be positive.")

    if candidate.devices <= 0:
        raise ValueError("candidate.devices must be positive.")

    request_speedup = candidate.request_throughput_rps / baseline.request_throughput_rps

    output_speedup = (
        candidate.output_throughput_tok_s / baseline.output_throughput_tok_s
    )

    ideal_scale = candidate.devices / baseline.devices

    return ScalingComparison(
        profile=candidate.profile,
        devices=candidate.devices,
        request_throughput_speedup=(request_speedup),
        output_throughput_speedup=(output_speedup),
        output_throughput_efficiency=(output_speedup / ideal_scale),
        ttft_p50_ratio=_ratio(
            candidate.ttft_p50_ms,
            baseline.ttft_p50_ms,
        ),
        ttft_p99_ratio=_ratio(
            candidate.ttft_p99_ms,
            baseline.ttft_p99_ms,
        ),
        tpot_p50_ratio=_ratio(
            candidate.tpot_p50_ms,
            baseline.tpot_p50_ms,
        ),
        tpot_p99_ratio=_ratio(
            candidate.tpot_p99_ms,
            baseline.tpot_p99_ms,
        ),
        e2e_p50_ratio=_ratio(
            candidate.e2e_p50_ms,
            baseline.e2e_p50_ms,
        ),
        e2e_p99_ratio=_ratio(
            candidate.e2e_p99_ms,
            baseline.e2e_p99_ms,
        ),
    )
