#!/usr/bin/env python3
"""Capture inventory, topology, and candidate placements into one artifact."""

from __future__ import annotations

import argparse
import json
import subprocess
from itertools import combinations
from pathlib import Path

from llmforge.distributed.topology import (
    classify_placement,
    parse_gpu_inventory_csv,
    parse_nvidia_smi_topology,
)


def run(*args: str) -> str:
    return subprocess.check_output(
        args,
        text=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/distributed/topology.json"),
    )

    parser.add_argument(
        "--max-placement-size",
        type=int,
        default=4,
    )

    args = parser.parse_args()

    inventory_raw = run(
        "nvidia-smi",
        "--query-gpu=index,name,uuid,pci.bus_id,memory.total",
        "--format=csv,noheader,nounits",
    )

    topology_raw = run(
        "nvidia-smi",
        "topo",
        "-m",
    )

    inventory = parse_gpu_inventory_csv(inventory_raw)

    snapshot = parse_nvidia_smi_topology(
        topology_raw,
        inventory=inventory,
    )

    indices = tuple(device.index for device in snapshot.devices)

    placements = []

    for size in range(
        1,
        min(
            args.max_placement_size,
            len(indices),
        )
        + 1,
    ):
        for placement in combinations(
            indices,
            size,
        ):
            placements.append(
                classify_placement(
                    snapshot,
                    placement,
                ).to_dict()
            )

    payload = {
        "snapshot": snapshot.to_dict(),
        "placements": placements,
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Topology written to: {args.output}")


if __name__ == "__main__":
    main()
