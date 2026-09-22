"""Bounded metrics registry with Prometheus exposition."""

from .defaults import create_default_registry
from .registry import MetricRegistry
from .schema import (
    MetricKind,
    MetricSpec,
)

__all__ = [
    "MetricKind",
    "MetricRegistry",
    "MetricSpec",
    "create_default_registry",
]
