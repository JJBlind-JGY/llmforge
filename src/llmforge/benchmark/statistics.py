"""Statistics utilities for reproducible performance benchmarks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean, median


@dataclass(frozen=True)
class TimingStats:
    count: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def percentile(samples: list[float], q: float) -> float:
    """Compute a percentile using linear interpolation."""
    if not samples:
        raise ValueError("samples must not be empty")

    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be between 0 and 1")

    ordered = sorted(samples)

    if len(ordered) == 1:
        return ordered[0]

    position = q * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)

    weight = position - lower

    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_ms(samples: list[float]) -> TimingStats:
    """Summarize millisecond timing samples."""
    if not samples:
        raise ValueError("samples must not be empty")

    return TimingStats(
        count=len(samples),
        mean_ms=mean(samples),
        median_ms=median(samples),
        p95_ms=percentile(samples=samples, q=0.95),
        p99_ms=percentile(samples=samples, q=0.99),
        min_ms=min(samples),
        max_ms=max(samples),
    )
