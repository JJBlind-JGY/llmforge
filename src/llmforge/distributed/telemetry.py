"""GPU telemetry parsing for serving/distributed experiments."""

from __future__ import annotations

import csv
import io
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class GpuTelemetrySample:
    timestamp: str
    index: int
    utilization_gpu_percent: float
    memory_used_mib: float
    memory_total_mib: float
    power_draw_w: float

    @property
    def memory_usage_ratio(
        self,
    ) -> float:
        if self.memory_total_mib <= 0:
            return 0.0
        return self.memory_used_mib / self.memory_total_mib

    def to_dict(self) -> dict:
        data = asdict(self)
        data["memory_usage_ratio"] = self.memory_usage_ratio
        return data


def parse_nvidia_smi_sample(
    text: str,
) -> list[GpuTelemetrySample]:
    samples = []

    reader = csv.reader(io.StringIO(text))

    for row in reader:
        if not row:
            continue

        if len(row) != 6:
            raise ValueError("Expected six telemetry columns.")

        samples.append(
            GpuTelemetrySample(
                timestamp=row[0].strip(),
                index=int(row[1].strip()),
                utilization_gpu_percent=float(row[2].strip()),
                memory_used_mib=float(row[3].strip()),
                memory_total_mib=float(row[4].strip()),
                power_draw_w=float(row[5].strip()),
            )
        )

    return samples
