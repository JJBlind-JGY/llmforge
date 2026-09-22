#!/usr/bin/env python3
"""Launch a repeatable collective sweep through torch.distributed.run."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def parse_gpu_list(
    raw: str,
) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in raw.split(",") if value.strip())

    if not values:
        raise ValueError("At least one GPU is required.")

    if len(set(values)) != len(values):
        raise ValueError("GPU indices must be unique.")

    return values


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/distributed/m5_collective_sweep.json"),
    )

    parser.add_argument(
        "--gpus",
        required=True,
        help="Physical GPUs, e.g. 0,1",
    )

    parser.add_argument(
        "--placement-label",
        required=True,
        help=("Artifact label such as same_numa_01 or cross_numa_02."),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/distributed/nccl"),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))

    gpus = parse_gpu_list(args.gpus)

    world_size = len(gpus)

    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in gpus)

    for collective in config["collectives"]:
        for message_mib in config["message_mib"]:
            output = (
                args.output_root
                / args.placement_label
                / (f"{collective}_{message_mib}mib.json")
            )

            command = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node",
                str(world_size),
                ("scripts/distributed/benchmark_collective.py"),
                "--collective",
                collective,
                "--message-mib",
                str(message_mib),
                "--dtype",
                str(config["dtype"]),
                "--warmups",
                str(config["warmups"]),
                "--iterations",
                str(config["iterations"]),
                "--output",
                str(output),
            ]

            print(
                "\n$ " + " ".join(command),
                flush=True,
            )

            print(
                "  CUDA_VISIBLE_DEVICES=" + environment["CUDA_VISIBLE_DEVICES"],
                flush=True,
            )

            if not args.dry_run:
                subprocess.run(
                    command,
                    check=True,
                    env=environment,
                )


if __name__ == "__main__":
    main()
