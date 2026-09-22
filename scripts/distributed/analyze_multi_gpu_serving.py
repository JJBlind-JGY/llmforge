#!/usr/bin/env python3
"""Build a normalized M5 serving comparison from M3 result artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llmforge.distributed.inference.analysis import (
    compare_to_baseline,
    load_m3_serving_run,
)
from llmforge.distributed.inference.telemetry_analysis import (
    summarize_gpu_telemetry,
)


def load_entry(
    raw: dict,
) -> dict:
    run = load_m3_serving_run(
        path=Path(raw["serving_result"]),
        profile=str(raw["profile"]),
        devices=int(raw["devices"]),
    )

    telemetry_path = raw.get("gpu_telemetry")

    telemetry = []

    if telemetry_path and Path(telemetry_path).exists():
        telemetry = [
            summary.to_dict()
            for summary in summarize_gpu_telemetry(Path(telemetry_path))
        ]

    return {
        "run": run,
        "telemetry": telemetry,
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))

    baseline_entry = load_entry(manifest["baseline"])

    baseline = baseline_entry["run"]

    candidates = [load_entry(raw) for raw in manifest["candidates"]]

    output = {
        "baseline": {
            "serving": (baseline.to_dict()),
            "gpu_telemetry": (baseline_entry["telemetry"]),
        },
        "candidates": [],
        "communication_context": (
            manifest.get(
                "communication_context",
                {},
            )
        ),
    }

    for entry in candidates:
        run = entry["run"]

        output["candidates"].append(
            {
                "serving": (run.to_dict()),
                "scaling": (
                    compare_to_baseline(
                        baseline=baseline,
                        candidate=run,
                    ).to_dict()
                ),
                "gpu_telemetry": (entry["telemetry"]),
            }
        )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(
            output,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Analysis written to: {args.output}")


if __name__ == "__main__":
    main()
