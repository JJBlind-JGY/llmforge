from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


class CollectiveKind(StrEnum):
    ALL_REDUCE = "all_reduce"
    ALL_GATHER = "all_gather"
    REDUCE_SCATTER = "reduce_scatter"


@dataclass(frozen=True)
class LatencySummary:
    count: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class CollectiveRun:
    collective: CollectiveKind
    world_size: int
    dtype: str
    payload_bytes: int
    per_rank_input_bytes: int
    per_rank_output_bytes: int
    warmups: int
    iterations: int
    local_rank_samples_ms: tuple[tuple[float, ...], ...]
    max_rank_samples_ms: tuple[float, ...]
    latency: LatencySummary
    algorithm_bandwidth_gbps: float
    bus_bandwidth_gbps: float
    metadata: dict[str, Any]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["collective"] = self.collective.value
        return data
