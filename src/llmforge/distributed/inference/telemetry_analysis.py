"""Summarize nvidia-smi samples without keeping an unbounded in-memory service state."""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean


@dataclass(frozen=True)
class GpuTelemetrySummary:
    gpu_index: int
    samples: int
    utilization_mean_percent: float
    utilization_max_percent: float
    memory_used_max_mib: float
    memory_usage_max_ratio: float
    power_mean_w: float
    power_max_w: float

    def to_dict(self) -> dict:
        return {
            "gpu_index": self.gpu_index,
            "samples": self.samples,
            "utilization_mean_percent": self.utilization_mean_percent,
            "utilization_max_percent": self.utilization_max_percent,
            "memory_used_max_mib": self.memory_used_max_mib,
            "memory_usage_max_ratio": self.memory_usage_max_ratio,
            "power_mean_w": self.power_mean_w,
            "power_max_w": self.power_max_w,
        }


def summarize_gpu_telemetry(
    path: Path,
) -> list[GpuTelemetrySummary]:
    rows: dict[
        int,
        list[dict[str, float]],
    ] = defaultdict(list)

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        for row in reader:
            index = int(row["index"])

            rows[index].append(
                {
                    "util": float(row["utilization_gpu_percent"]),
                    "memory": float(row["memory_used_mib"]),
                    "memory_ratio": float(row["memory_usage_ratio"]),
                    "power": float(row["power_draw_w"]),
                }
            )

    summaries = []

    for index, samples in sorted(rows.items()):
        summaries.append(
            GpuTelemetrySummary(
                gpu_index=index,
                samples=len(samples),
                utilization_mean_percent=mean(sample["util"] for sample in samples),
                utilization_max_percent=max(sample["util"] for sample in samples),
                memory_used_max_mib=max(sample["memory"] for sample in samples),
                memory_usage_max_ratio=max(
                    sample["memory_ratio"] for sample in samples
                ),
                power_mean_w=mean(sample["power"] for sample in samples),
                power_max_w=max(sample["power"] for sample in samples),
            )
        )

    return summaries
