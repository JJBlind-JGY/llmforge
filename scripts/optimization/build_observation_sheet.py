#!/usr/bin/env python3
"""Build one machine-readable M7 observation sheet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llmforge.optimization.observations import (
    build_observation_sheet,
)


def optional_path(
    value: str | None,
) -> Path | None:
    if not value:
        return None
    return Path(value)


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

    raw = json.loads(args.manifest.read_text(encoding="utf-8"))

    sheet = build_observation_sheet(
        observation_id=raw["observation_id"],
        workload_id=raw["workload_id"],
        hypothesis=raw["hypothesis"],
        serving_paths=[Path(value) for value in raw["serving_results"]],
        runtime_path=optional_path(raw.get("runtime_summary")),
        distributed_path=optional_path(raw.get("distributed_analysis")),
        telemetry_path=optional_path(raw.get("telemetry_status")),
        profiler_artifacts=tuple(
            raw.get(
                "profiler_artifacts",
                (),
            )
        ),
        notes=tuple(
            raw.get(
                "notes",
                (),
            )
        ),
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(
            sheet.to_dict(),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Observation sheet written to: {args.output}")


if __name__ == "__main__":
    main()
