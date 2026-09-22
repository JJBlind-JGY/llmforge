from __future__ import annotations

from math import ceil
from statistics import mean

from .schema import LatencySummary


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("values must not be empty.")
    if not 0 <= q <= 100:
        raise ValueError("q must be in [0, 100].")

    ordered = sorted(values)

    if q == 0:
        return ordered[0]

    rank = ceil(q / 100.0 * len(ordered))
    return ordered[
        min(
            max(rank - 1, 0),
            len(ordered) - 1,
        )
    ]


def summarize_latency(values: list[float]) -> LatencySummary:
    if not values:
        raise ValueError("values must not be empty.")

    return LatencySummary(
        count=len(values),
        mean_ms=mean(values),
        p50_ms=percentile(values, 50),
        p95_ms=percentile(values, 95),
        p99_ms=percentile(values, 99),
        min_ms=min(values),
        max_ms=max(values),
    )
