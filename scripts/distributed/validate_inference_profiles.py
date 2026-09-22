#!/usr/bin/env python3
"""Validate M5 TP/DP/PP profiles before reserving GPUs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llmforge.distributed.inference import (
    ModelParallelShape,
    ParallelMode,
    ParallelProfile,
    validate_parallel_profile,
)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/distributed/m5_inference_profiles.json"),
    )

    args = parser.parse_args()

    payload = json.loads(args.config.read_text(encoding="utf-8"))

    model = ModelParallelShape(**payload["model_shape"])

    output = []

    for raw in payload["profiles"]:
        profile = ParallelProfile(
            name=raw["name"],
            mode=ParallelMode(raw["mode"]),
            gpu_indices=tuple(raw["gpu_indices"]),
            tensor_parallel_size=int(
                raw.get(
                    "tensor_parallel_size",
                    1,
                )
            ),
            pipeline_parallel_size=int(
                raw.get(
                    "pipeline_parallel_size",
                    1,
                )
            ),
            data_parallel_size=int(
                raw.get(
                    "data_parallel_size",
                    1,
                )
            ),
            optional=bool(
                raw.get(
                    "optional",
                    False,
                )
            ),
            note=str(
                raw.get(
                    "note",
                    "",
                )
            ),
        )

        validation = validate_parallel_profile(
            profile=profile,
            model=model,
        )

        output.append(
            {
                "profile": (profile.to_dict()),
                "validation": (validation.to_dict()),
            }
        )

    print(
        json.dumps(
            output,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
