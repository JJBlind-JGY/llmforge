#!/usr/bin/env python3
"""Print vLLM profiler arguments for a short M5 communication trace."""

from __future__ import annotations

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-dir",
        required=True,
        help=("Absolute directory visible to vLLM workers."),
    )

    parser.add_argument(
        "--active-iterations",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    config = {
        "profiler": "torch",
        "torch_profiler_dir": (args.output_dir),
        "torch_profiler_with_stack": False,
        "torch_profiler_record_shapes": False,
        "torch_profiler_with_memory": False,
        "active_iterations": (args.active_iterations),
        "ignore_frontend": True,
    }

    print("--profiler-config " + repr(json.dumps(config)))


if __name__ == "__main__":
    main()
