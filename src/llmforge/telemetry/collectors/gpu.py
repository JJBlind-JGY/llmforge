"""Live GPU collector reusing the M5 nvidia-smi parser."""

from __future__ import annotations

import subprocess

from llmforge.distributed.telemetry import (
    parse_nvidia_smi_sample,
)
from llmforge.telemetry.adapters import (
    observe_gpu_sample,
)
from llmforge.telemetry.metrics import (
    MetricRegistry,
)

_QUERY = "timestamp,index,utilization.gpu,memory.used,memory.total,power.draw"


class GpuMetricsCollector:
    name = "gpu"

    def __init__(
        self,
        *,
        gpu_indices: tuple[int, ...] = (),
    ) -> None:
        self.gpu_indices = set(gpu_indices)

    def collect(
        self,
        registry: MetricRegistry,
    ) -> None:
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                f"--query-gpu={_QUERY}",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )

        for sample in parse_nvidia_smi_sample(raw):
            if self.gpu_indices and sample.index not in self.gpu_indices:
                continue

            observe_gpu_sample(
                registry,
                sample,
            )
