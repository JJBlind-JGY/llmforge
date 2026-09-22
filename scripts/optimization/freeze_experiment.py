#!/usr/bin/env python3
"""Validate and freeze one M7 experiment specification."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from llmforge.optimization.experiment import (
    load_experiment,
)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    experiment = load_experiment(args.experiment)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    frozen = args.output_dir / "experiment.json"

    frozen.write_text(
        json.dumps(
            experiment.to_dict(),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    workload = Path(experiment.workload_config)

    if workload.exists():
        shutil.copy2(
            workload,
            args.output_dir / "workload.json",
        )

    print(f"Frozen experiment written to: {frozen}")

    print(f"baseline_fingerprint={experiment.baseline.fingerprint}")

    print(f"candidate_fingerprint={experiment.candidate.fingerprint}")


if __name__ == "__main__":
    main()
