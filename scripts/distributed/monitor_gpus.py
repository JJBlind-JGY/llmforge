#!/usr/bin/env python3
"""Poll nvidia-smi while a serving experiment is running."""

from __future__ import annotations

import argparse
import csv
import subprocess
import time
from pathlib import Path

from llmforge.distributed.telemetry import (
    parse_nvidia_smi_sample,
)

QUERY = "timestamp,index,utilization.gpu,memory.used,memory.total,power.draw"


def capture_once() -> str:
    return subprocess.check_output(
        [
            "nvidia-smi",
            f"--query-gpu={QUERY}",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--interval-s",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--duration-s",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--gpus",
        default="",
        help=("Optional physical GPU filter, e.g. 0,1. Empty records all GPUs."),
    )

    args = parser.parse_args()

    if args.interval_s <= 0:
        raise ValueError("interval_s must be positive.")

    if args.duration_s <= 0:
        raise ValueError("duration_s must be positive.")

    selected = {int(value) for value in args.gpus.split(",") if value.strip()}

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    deadline = time.monotonic() + args.duration_s

    with args.output.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "timestamp",
                "index",
                "utilization_gpu_percent",
                "memory_used_mib",
                "memory_total_mib",
                "memory_usage_ratio",
                "power_draw_w",
            ],
        )

        writer.writeheader()

        while time.monotonic() < deadline:
            started = time.monotonic()

            for sample in parse_nvidia_smi_sample(capture_once()):
                if selected and sample.index not in selected:
                    continue

                writer.writerow(sample.to_dict())

            handle.flush()

            remaining = args.interval_s - (time.monotonic() - started)

            if remaining > 0:
                time.sleep(remaining)

    print(f"GPU telemetry written to: {args.output}")


if __name__ == "__main__":
    main()
