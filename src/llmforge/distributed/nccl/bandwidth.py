"""NCCL-tests-compatible algorithm/bus bandwidth normalization."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .schema import CollectiveKind


@dataclass(frozen=True)
class BandwidthResult:
    algorithm_bandwidth_gbps: float
    bus_bandwidth_gbps: float
    correction_factor: float

    def to_dict(self) -> dict:
        return asdict(self)


def _bus_correction_factor(
    collective: CollectiveKind,
    world_size: int,
) -> float:
    if world_size <= 0:
        raise ValueError("world_size must be positive.")

    if world_size == 1:
        return 1.0

    if collective == CollectiveKind.ALL_REDUCE:
        return 2.0 * (world_size - 1) / world_size

    if collective in {
        CollectiveKind.ALL_GATHER,
        CollectiveKind.REDUCE_SCATTER,
    }:
        return (world_size - 1) / world_size

    raise ValueError(f"Unsupported collective: {collective}")


def calculate_collective_bandwidth(
    *,
    collective: CollectiveKind,
    payload_bytes: int,
    latency_ms: float,
    world_size: int,
) -> BandwidthResult:
    """Calculate NCCL-tests style algBW and busBW.

    `payload_bytes` follows NCCL-tests' S semantics:
    - AllReduce: per-rank input/output tensor size.
    - AllGather: full gathered output size on each rank.
    - ReduceScatter: full reduce-scatter input size on each rank.
    """

    if payload_bytes <= 0:
        raise ValueError("payload_bytes must be positive.")
    if latency_ms <= 0:
        raise ValueError("latency_ms must be positive.")

    seconds = latency_ms / 1000.0

    alg_bw = payload_bytes / seconds / 1e9

    factor = _bus_correction_factor(
        collective,
        world_size,
    )

    return BandwidthResult(
        algorithm_bandwidth_gbps=alg_bw,
        bus_bandwidth_gbps=alg_bw * factor,
        correction_factor=factor,
    )
