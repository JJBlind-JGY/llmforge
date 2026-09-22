#!/usr/bin/env python3
"""Print the pinned vLLM command for one frozen M7 experiment arm."""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path

from llmforge.optimization.experiment import (
    load_experiment,
)


def add_bool_flag(
    command: list[str],
    *,
    name: str,
    enabled: bool,
) -> None:
    command.append(f"--{name}" if enabled else f"--no-{name}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--arm",
        choices=[
            "baseline",
            "candidate",
        ],
        required=True,
    )

    parser.add_argument(
        "--gpus",
        required=True,
        help=("Physical CUDA_VISIBLE_DEVICES value, e.g. 3 or 0,1."),
    )

    parser.add_argument(
        "--hf-home",
        required=True,
    )

    parser.add_argument(
        "--host",
        default="127.0.0.1",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
    )

    args = parser.parse_args()

    experiment = load_experiment(args.experiment)

    arm = getattr(
        experiment,
        args.arm,
    )

    config = arm.server_config

    command = [
        "vllm",
        "serve",
        str(config["model"]),
        "--revision",
        str(config["revision"]),
        "--tokenizer-revision",
        str(config["revision"]),
        "--served-model-name",
        str(
            config.get(
                "served_model_name",
                "qwen3-8b",
            )
        ),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--dtype",
        str(
            config.get(
                "dtype",
                "bfloat16",
            )
        ),
        "--kv-cache-dtype",
        str(
            config.get(
                "kv_cache_dtype",
                "bfloat16",
            )
        ),
        "--gpu-memory-utilization",
        str(
            config.get(
                "gpu_memory_utilization",
                0.90,
            )
        ),
        "--model-impl",
        str(
            config.get(
                "model_impl",
                "vllm",
            )
        ),
        "--load-format",
        str(
            config.get(
                "load_format",
                "safetensors",
            )
        ),
        "--max-model-len",
        str(
            config.get(
                "max_model_len",
                8192,
            )
        ),
        "--max-num-batched-tokens",
        str(
            config.get(
                "max_num_batched_tokens",
                2048,
            )
        ),
        "--generation-config",
        str(
            config.get(
                "generation_config",
                "vllm",
            )
        ),
        "--seed",
        str(
            config.get(
                "seed",
                0,
            )
        ),
        "--default-chat-template-kwargs",
        (
            '{"enable_thinking":'
            + (
                "true"
                if config.get(
                    "enable_thinking",
                    False,
                )
                else "false"
            )
            + "}"
        ),
    ]

    add_bool_flag(
        command,
        name="enable-chunked-prefill",
        enabled=bool(
            config.get(
                "enable_chunked_prefill",
                True,
            )
        ),
    )

    add_bool_flag(
        command,
        name="enable-prefix-caching",
        enabled=bool(
            config.get(
                "enable_prefix_caching",
                False,
            )
        ),
    )

    if arm.scheduler_class:
        command.extend(
            [
                "--scheduler-cls",
                arm.scheduler_class,
            ]
        )

    environment = {
        "CUDA_VISIBLE_DEVICES": (args.gpus),
        "HF_HOME": args.hf_home,
        "HF_HUB_OFFLINE": "1",
        **arm.environment,
    }

    prefix = " ".join(
        f"{key}={shlex.quote(value)}" for key, value in environment.items()
    )

    print(prefix + " " + shlex.join(command))


if __name__ == "__main__":
    main()
