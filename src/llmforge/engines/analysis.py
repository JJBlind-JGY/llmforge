"""Normalize M3 serving artifacts across engines and test qualitative generality."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median


@dataclass(frozen=True)
class EngineRunSummary:
    engine: str
    artifact: str
    request_throughput_rps: float
    output_throughput_tok_s: float
    ttft_p50_ms: float
    ttft_p99_ms: float
    tpot_p50_ms: float
    tpot_p99_ms: float
    e2e_p50_ms: float
    e2e_p99_ms: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class EngineAggregate:
    engine: str
    runs: int
    request_throughput_rps: float
    output_throughput_tok_s: float
    ttft_p50_ms: float
    ttft_p99_ms: float
    tpot_p50_ms: float
    tpot_p99_ms: float
    e2e_p50_ms: float
    e2e_p99_ms: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class CrossEngineComparison:
    left: EngineAggregate
    right: EngineAggregate
    ratios_right_over_left: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
            "ratios_right_over_left": dict(self.ratios_right_over_left),
        }


def _dist(summary: dict, name: str, percentile: str) -> float:
    return float(summary[name][percentile])


def load_m3_engine_run(
    *,
    path: Path,
    engine: str,
) -> EngineRunSummary:
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload["summary"]

    return EngineRunSummary(
        engine=engine,
        artifact=str(path),
        request_throughput_rps=float(summary["request_throughput_rps"]),
        output_throughput_tok_s=float(summary["output_throughput_tok_s"]),
        ttft_p50_ms=_dist(summary, "ttft_ms", "p50"),
        ttft_p99_ms=_dist(summary, "ttft_ms", "p99"),
        tpot_p50_ms=_dist(summary, "tpot_ms", "p50"),
        tpot_p99_ms=_dist(summary, "tpot_ms", "p99"),
        e2e_p50_ms=_dist(summary, "e2e_ms", "p50"),
        e2e_p99_ms=_dist(summary, "e2e_ms", "p99"),
    )


def aggregate_runs(
    runs: list[EngineRunSummary],
) -> EngineAggregate:
    if len(runs) < 3:
        raise ValueError("At least three runs per engine are required.")

    engines = {run.engine for run in runs}
    if len(engines) != 1:
        raise ValueError("All runs must belong to one engine.")

    def med(field: str) -> float:
        return float(median(getattr(run, field) for run in runs))

    return EngineAggregate(
        engine=runs[0].engine,
        runs=len(runs),
        request_throughput_rps=med("request_throughput_rps"),
        output_throughput_tok_s=med("output_throughput_tok_s"),
        ttft_p50_ms=med("ttft_p50_ms"),
        ttft_p99_ms=med("ttft_p99_ms"),
        tpot_p50_ms=med("tpot_p50_ms"),
        tpot_p99_ms=med("tpot_p99_ms"),
        e2e_p50_ms=med("e2e_p50_ms"),
        e2e_p99_ms=med("e2e_p99_ms"),
    )


def compare_aggregates(
    left: EngineAggregate,
    right: EngineAggregate,
) -> CrossEngineComparison:
    fields = (
        "request_throughput_rps",
        "output_throughput_tok_s",
        "ttft_p50_ms",
        "ttft_p99_ms",
        "tpot_p50_ms",
        "tpot_p99_ms",
        "e2e_p50_ms",
        "e2e_p99_ms",
    )

    ratios = {}

    for field in fields:
        base = float(getattr(left, field))
        candidate = float(getattr(right, field))

        if base == 0:
            ratios[field] = float("inf") if candidate != 0 else 1.0
        else:
            ratios[field] = candidate / base

    return CrossEngineComparison(
        left=left,
        right=right,
        ratios_right_over_left=ratios,
    )


def qualitative_direction(
    *,
    baseline: float,
    stressed: float,
    metric_direction: str,
    tolerance_fraction: float = 0.02,
) -> str:
    if baseline == 0:
        return "unknown"

    relative = stressed / baseline - 1.0

    if abs(relative) <= tolerance_fraction:
        return "stable"

    if metric_direction == "lower_is_better":
        return "degraded" if relative > 0 else "improved"

    if metric_direction == "higher_is_better":
        return "improved" if relative > 0 else "degraded"

    raise ValueError("Unsupported metric_direction.")


def cross_engine_generality(
    *,
    left_baseline: float,
    left_stressed: float,
    right_baseline: float,
    right_stressed: float,
    metric_direction: str,
    tolerance_fraction: float = 0.02,
) -> dict:
    left = qualitative_direction(
        baseline=left_baseline,
        stressed=left_stressed,
        metric_direction=metric_direction,
        tolerance_fraction=tolerance_fraction,
    )

    right = qualitative_direction(
        baseline=right_baseline,
        stressed=right_stressed,
        metric_direction=metric_direction,
        tolerance_fraction=tolerance_fraction,
    )

    return {
        "left_direction": left,
        "right_direction": right,
        "same_direction": left == right,
        "claim": (
            "cross_engine_direction_consistent"
            if left == right and left != "stable"
            else "engine_specific_or_inconclusive"
        ),
    }
