"""Extract collective timing context from Chrome/PyTorch profiler traces."""

from __future__ import annotations

import gzip
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

_COLLECTIVE_TOKENS = (
    "nccl",
    "all_reduce",
    "allreduce",
    "all_gather",
    "allgather",
    "reduce_scatter",
    "reducescatter",
)


@dataclass(frozen=True)
class CommunicationProfileSummary:
    matched_events: int
    total_duration_us: float
    by_name_us: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "matched_events": self.matched_events,
            "total_duration_us": self.total_duration_us,
            "by_name_us": dict(self.by_name_us),
        }


def _read_json(path: Path) -> dict:
    if path.suffix == ".gz":
        with gzip.open(
            path,
            "rt",
            encoding="utf-8",
        ) as handle:
            return json.load(handle)

    return json.loads(path.read_text(encoding="utf-8"))


def summarize_communication_trace(
    path: Path,
) -> CommunicationProfileSummary:
    """Sum duration of collective-like complete events in one profiler trace.

    PyTorch/Chrome trace `dur` values are microseconds. The summary is per trace
    file/rank and is not automatically interpreted as end-to-end serving time.
    """

    payload = _read_json(path)

    trace_events = payload.get("traceEvents", [])

    by_name: dict[
        str,
        float,
    ] = defaultdict(float)

    matched = 0
    total = 0.0

    for event in trace_events:
        name = str(
            event.get(
                "name",
                "",
            )
        )

        lowered = name.lower()

        if not any(token in lowered for token in _COLLECTIVE_TOKENS):
            continue

        if event.get("ph") != "X":
            continue

        duration = event.get("dur")

        if duration is None:
            continue

        duration_us = float(duration)

        matched += 1
        total += duration_us
        by_name[name] += duration_us

    return CommunicationProfileSummary(
        matched_events=matched,
        total_duration_us=total,
        by_name_us=dict(
            sorted(
                by_name.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        ),
    )
