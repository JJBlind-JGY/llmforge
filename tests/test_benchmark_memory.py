import pytest

from llmforge.benchmark.memory import (
    effective_bandwidth_gbps,
    vector_add_arithmetic_intensity,
    vector_add_bytes,
)


def test_vector_add_bytes_fp32() -> None:
    assert vector_add_bytes(1024, 4) == 12288


def test_vector_add_arithmetic_intensity_fp32() -> None:
    assert vector_add_arithmetic_intensity(4) == 1.0 / 12.0


def test_effective_bandwidth() -> None:
    bandwidth = effective_bandwidth_gbps(
        bytes_transfered=1_000_000_000,
        latency_ms=1000.0,
    )

    assert bandwidth == pytest.approx(1.0)
