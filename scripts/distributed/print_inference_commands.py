#!/usr/bin/env python3
"""Print pinned vLLM commands for valid M5 inference profiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llmforge.distributed.inference import (
    ModelParallelShape,
    ParallelMode,
    ParallelProfile,
    build_vllm_server_command,
    validate_parallel_profile,
)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base-server-config",
        type=Path,
        default=Path("configs/serving/qwen3_8b_vllm_baseline.json"),
    )

    parser.add_argument(
        "--profiles",
        type=Path,
        default=Path("configs/distributed/m5_inference_profiles.json"),
    )

    parser.add_argument(
        "--hf-home",
        required=True,
    )

    parser.add_argument(
        "--profile",
        action="append",
        default=[],
        help=("Optional profile name filter; may be repeated."),
    )

    args = parser.parse_args()

    base = json.loads(args.base_server_config.read_text(encoding="utf-8"))

    payload = json.loads(args.profiles.read_text(encoding="utf-8"))

    model = ModelParallelShape(**payload["model_shape"])

    selected = set(args.profile)

    for raw in payload["profiles"]:
        if selected and raw["name"] not in selected:
            continue

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

        print("\n# " + profile.name)

        print("# valid=" + str(validation.valid))

        if validation.reasons:
            for reason in validation.reasons:
                print("# reason: " + reason)

        if not validation.valid:
            continue

        print(
            build_vllm_server_command(
                base=base,
                profile=profile,
                hf_home=args.hf_home,
            )
        )


if __name__ == "__main__":
    main()
