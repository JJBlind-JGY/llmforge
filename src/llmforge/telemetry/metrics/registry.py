"""Thread-safe fixed-memory metrics registry.

Histograms keep only bucket counts, sum, and count. They do not retain every
sample, so memory usage remains bounded during long-running services.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Iterable

from .schema import (
    MetricKind,
    MetricSpec,
)


LabelValues = tuple[tuple[str, str], ...]


@dataclass
class _HistogramState:
    bucket_counts: list[int]
    count: int = 0
    total: float = 0.0


def _escape_label(
    value: str,
) -> str:
    return (
        value.replace(
            "\\",
            "\\\\",
        )
        .replace(
            "\n",
            "\\n",
        )
        .replace(
            '"',
            '\\"',
        )
    )


def _format_number(
    value: float,
) -> str:
    if math.isinf(value):
        return "+Inf" if value > 0 else "-Inf"

    return repr(float(value))


class MetricRegistry:
    def __init__(self) -> None:
        self._lock = threading.RLock()

        self._specs: dict[
            str,
            MetricSpec,
        ] = {}

        self._counters: dict[
            tuple[
                str,
                LabelValues,
            ],
            float,
        ] = {}

        self._gauges: dict[
            tuple[
                str,
                LabelValues,
            ],
            float,
        ] = {}

        self._histograms: dict[
            tuple[
                str,
                LabelValues,
            ],
            _HistogramState,
        ] = {}

    def register(
        self,
        spec: MetricSpec,
    ) -> None:
        with self._lock:
            existing = self._specs.get(spec.name)

            if existing is not None and existing != spec:
                raise ValueError(
                    f"Metric already registered with a different spec: {spec.name}."
                )

            self._specs[spec.name] = spec

    def _labels(
        self,
        spec: MetricSpec,
        labels: dict[
            str,
            str,
        ]
        | None,
    ) -> LabelValues:
        labels = labels or {}

        expected = set(spec.label_names)

        actual = set(labels)

        if actual != expected:
            raise ValueError(
                f"Labels for {spec.name} "
                f"must be {sorted(expected)}, "
                f"got {sorted(actual)}."
            )

        return tuple(
            (
                name,
                str(labels[name]),
            )
            for name in spec.label_names
        )

    def _spec(
        self,
        name: str,
        kind: MetricKind,
    ) -> MetricSpec:
        try:
            spec = self._specs[name]
        except KeyError as exc:
            raise KeyError(f"Metric {name!r} is not registered.") from exc

        if spec.kind != kind:
            raise TypeError(f"Metric {name!r} is {spec.kind.value}, not {kind.value}.")

        return spec

    def inc_counter(
        self,
        name: str,
        value: float = 1.0,
        *,
        labels: dict[
            str,
            str,
        ]
        | None = None,
    ) -> None:
        if value < 0:
            raise ValueError("Counters cannot decrease.")

        with self._lock:
            spec = self._spec(
                name,
                MetricKind.COUNTER,
            )

            key = (
                name,
                self._labels(
                    spec,
                    labels,
                ),
            )

            self._counters[key] = (
                self._counters.get(
                    key,
                    0.0,
                )
                + value
            )

    def set_counter_absolute(
        self,
        name: str,
        value: float,
        *,
        labels: dict[
            str,
            str,
        ]
        | None = None,
    ) -> None:
        """Set a scraped monotonic counter without manufacturing deltas."""

        if value < 0:
            raise ValueError("Counters cannot be negative.")

        with self._lock:
            spec = self._spec(
                name,
                MetricKind.COUNTER,
            )

            key = (
                name,
                self._labels(
                    spec,
                    labels,
                ),
            )

            previous = self._counters.get(key)

            if previous is not None and value < previous:
                # Upstream process restarted. Prometheus counters are allowed to
                # reset across process lifetimes, so adopt the new baseline.
                self._counters[key] = value
                return

            self._counters[key] = value

    def set_gauge(
        self,
        name: str,
        value: float,
        *,
        labels: dict[
            str,
            str,
        ]
        | None = None,
    ) -> None:
        with self._lock:
            spec = self._spec(
                name,
                MetricKind.GAUGE,
            )

            key = (
                name,
                self._labels(
                    spec,
                    labels,
                ),
            )

            self._gauges[key] = value

    def observe_histogram(
        self,
        name: str,
        value: float,
        *,
        labels: dict[
            str,
            str,
        ]
        | None = None,
    ) -> None:
        with self._lock:
            spec = self._spec(
                name,
                MetricKind.HISTOGRAM,
            )

            key = (
                name,
                self._labels(
                    spec,
                    labels,
                ),
            )

            state = self._histograms.get(key)

            if state is None:
                state = _HistogramState(bucket_counts=[0 for _ in spec.buckets])

                self._histograms[key] = state

            for index, bound in enumerate(spec.buckets):
                if value <= bound:
                    state.bucket_counts[index] += 1

            state.count += 1
            state.total += value

    def snapshot(
        self,
    ) -> dict:
        with self._lock:
            return {
                "specs": {
                    name: {
                        "kind": (spec.kind.value),
                        "help": spec.help,
                        "label_names": list(spec.label_names),
                        "buckets": list(spec.buckets),
                    }
                    for name, spec in self._specs.items()
                },
                "counters": [
                    {
                        "name": name,
                        "labels": dict(labels),
                        "value": value,
                    }
                    for (
                        name,
                        labels,
                    ), value in self._counters.items()
                ],
                "gauges": [
                    {
                        "name": name,
                        "labels": dict(labels),
                        "value": value,
                    }
                    for (
                        name,
                        labels,
                    ), value in self._gauges.items()
                ],
                "histograms": [
                    {
                        "name": name,
                        "labels": dict(labels),
                        "bucket_counts": list(state.bucket_counts),
                        "count": state.count,
                        "sum": state.total,
                    }
                    for (
                        name,
                        labels,
                    ), state in self._histograms.items()
                ],
            }

    @staticmethod
    def _render_labels(
        labels: Iterable[
            tuple[
                str,
                str,
            ]
        ],
        *,
        extra: tuple[
            str,
            str,
        ]
        | None = None,
    ) -> str:
        values = list(labels)

        if extra is not None:
            values.append(extra)

        if not values:
            return ""

        rendered = ",".join(
            f'{name}="{_escape_label(value)}"' for name, value in values
        )

        return "{" + rendered + "}"

    def prometheus_text(
        self,
    ) -> str:
        lines: list[str] = []

        with self._lock:
            for name, spec in sorted(self._specs.items()):
                help_text = spec.help.replace(
                    "\\",
                    "\\\\",
                ).replace(
                    "\n",
                    "\\n",
                )

                lines.append(f"# HELP {name} {help_text}")

                lines.append(f"# TYPE {name} {spec.kind.value}")

                if spec.kind == MetricKind.COUNTER:
                    for (
                        metric_name,
                        labels,
                    ), value in sorted(self._counters.items()):
                        if metric_name != name:
                            continue

                        lines.append(
                            f"{name}"
                            f"{self._render_labels(labels)} "
                            f"{_format_number(value)}"
                        )

                elif spec.kind == MetricKind.GAUGE:
                    for (
                        metric_name,
                        labels,
                    ), value in sorted(self._gauges.items()):
                        if metric_name != name:
                            continue

                        lines.append(
                            f"{name}"
                            f"{self._render_labels(labels)} "
                            f"{_format_number(value)}"
                        )

                else:
                    for (
                        metric_name,
                        labels,
                    ), state in sorted(self._histograms.items()):
                        if metric_name != name:
                            continue

                        for (
                            bound,
                            count,
                        ) in zip(
                            spec.buckets,
                            state.bucket_counts,
                        ):
                            lines.append(
                                f"{name}_bucket"
                                f"{self._render_labels(labels, extra=('le', _format_number(bound)))} "
                                f"{count}"
                            )

                        lines.append(
                            f"{name}_bucket"
                            f"{self._render_labels(labels, extra=('le', '+Inf'))} "
                            f"{state.count}"
                        )

                        lines.append(
                            f"{name}_sum"
                            f"{self._render_labels(labels)} "
                            f"{_format_number(state.total)}"
                        )

                        lines.append(
                            f"{name}_count{self._render_labels(labels)} {state.count}"
                        )

        return "\n".join(lines) + ("\n" if lines else "")
