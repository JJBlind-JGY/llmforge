import pytest

from llmforge.distributed.nccl.bandwidth import (
    calculate_collective_bandwidth,
)
from llmforge.distributed.nccl.schema import CollectiveKind


def test_allreduce_bus_bandwidth_factor() -> None:
    result = calculate_collective_bandwidth(
        collective=CollectiveKind.ALL_REDUCE,
        payload_bytes=1_000_000_000,
        latency_ms=1000.0,
        world_size=4,
    )

    assert result.algorithm_bandwidth_gbps == pytest.approx(1.0)
    assert result.correction_factor == pytest.approx(1.5)
    assert result.bus_bandwidth_gbps == pytest.approx(1.5)


@pytest.mark.parametrize(
    "collective",
    [
        CollectiveKind.ALL_GATHER,
        CollectiveKind.REDUCE_SCATTER,
    ],
)
def test_gather_scatter_bus_factor(
    collective: CollectiveKind,
) -> None:
    result = calculate_collective_bandwidth(
        collective=collective,
        payload_bytes=1_000_000_000,
        latency_ms=1000.0,
        world_size=4,
    )

    assert result.correction_factor == pytest.approx(0.75)
