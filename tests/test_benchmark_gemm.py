import pytest

from llmforge.benchmark.gemm import (
    effective_tflops,
    gemm_arithmetic_intensity,
    gemm_flops,
    gemm_minimum_bytes,
)


def test_gemm_flops() -> None:
    assert gemm_flops(2, 3, 4) == 48


def test_gemm_minimum_bytes() -> None:
    assert gemm_minimum_bytes(2, 3, 4, element_size_bytes=4) == 104


def test_square_fp32_arithmetic_intensity() -> None:
    intensity = gemm_arithmetic_intensity(4096, 4096, 4096, element_size_bytes=4)
    assert intensity == pytest.approx(4096 / 6)


def test_effective_tflops() -> None:
    value = effective_tflops(flops=1_000_000_000_000, latency_ms=1000.0)

    assert value == pytest.approx(1.0)
