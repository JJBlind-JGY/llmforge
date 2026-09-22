from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum


_METRIC_NAME = re.compile(r"^[a-zA-Z_:][a-zA-Z0-9_:]*$")

_LABEL_NAME = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


class MetricKind(StrEnum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"


@dataclass(frozen=True)
class MetricSpec:
    name: str
    kind: MetricKind
    help: str
    label_names: tuple[str, ...] = ()
    buckets: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if not _METRIC_NAME.fullmatch(self.name):
            raise ValueError(f"Invalid metric name: {self.name!r}.")

        if not self.help:
            raise ValueError("Metric help must not be empty.")

        if len(self.label_names) != len(set(self.label_names)):
            raise ValueError("Metric label names must be unique.")

        for label in self.label_names:
            if not _LABEL_NAME.fullmatch(label):
                raise ValueError(f"Invalid label name: {label!r}.")

        if self.kind == MetricKind.HISTOGRAM:
            if not self.buckets:
                raise ValueError("Histogram buckets must not be empty.")

            if tuple(sorted(self.buckets)) != self.buckets:
                raise ValueError("Histogram buckets must be sorted.")

            if len(set(self.buckets)) != len(self.buckets):
                raise ValueError("Histogram buckets must be unique.")

        elif self.buckets:
            raise ValueError("Only histograms may declare buckets.")
