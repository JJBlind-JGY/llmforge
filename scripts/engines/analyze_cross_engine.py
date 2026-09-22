#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from llmforge.engines.analysis import (
    aggregate_runs,
    compare_aggregates,
    load_m3_engine_run,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))

    vllm = aggregate_runs(
        [
            load_m3_engine_run(path=Path(path), engine="vllm")
            for path in manifest["vllm_results"]
        ]
    )

    sglang = aggregate_runs(
        [
            load_m3_engine_run(path=Path(path), engine="sglang")
            for path in manifest["sglang_results"]
        ]
    )

    comparison = compare_aggregates(vllm, sglang)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(comparison.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Cross-engine analysis written to: {args.output}")


if __name__ == "__main__":
    main()
