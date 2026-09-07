"""Memory-bandwidth performance models."""

from __future__ import annotations


def vector_add_bytes(
    elements: int,
    element_size_bytes: int,
) -> int:
    """Bytes transferred by out = x + y

    Reads x and y and writes out.
    """
    return elements * element_size_bytes * 3


def effective_bandwidth_gbps(
    bytes_transfered: int,
    latency_ms: float,
) -> float:
    """Return effective bandwidth in decimal GB/s."""
    seconds = latency_ms / 1000.0

    return bytes_transfered / seconds / 1e9


def vector_add_arithmetic_intensity(
    element_size_bytes: int,
) -> float:
    """Arithmetic intensity of elementwise addition."""
    return 1.0 / (3 * element_size_bytes)
