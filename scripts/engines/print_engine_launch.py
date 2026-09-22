#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

from llmforge.engines import create_engine_adapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--engine", choices=["vllm", "sglang"], required=True)
    parser.add_argument("--hf-home", required=True)
    args = parser.parse_args()

    raw = json.loads(args.config.read_text(encoding="utf-8"))
    common = {
        "model": raw["model"],
        "revision": raw["revision"],
        "served_model_name": raw["served_model_name"],
        "dtype": raw["dtype"],
        "max_model_len": raw["max_model_len"],
        "gpu_memory_utilization": raw["gpu_memory_utilization"],
        "offline": raw.get("offline", True),
        "hf_home": args.hf_home,
    }
    common.update(raw["profiles"][args.engine])

    spec = create_engine_adapter(args.engine).build_launch_spec(common)

    prefix = " ".join(
        f"{key}={shlex.quote(value)}"
        for key, value in spec.environment.items()
        if value != ""
    )

    print(prefix + " " + shlex.join(spec.command))
    print()
    print(json.dumps(spec.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
