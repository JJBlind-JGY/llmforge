#!/usr/bin/env python3
"""Analyze repeated M7 baseline/candidate serving artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llmforge.optimization.analysis import (
    analyze_experiment,
)
from llmforge.optimization.experiment import (
    load_experiment,
)


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

    experiment = load_experiment(Path(manifest["experiment"]))

    selection = json.loads(Path(manifest["selection_gate"]).read_text(encoding="utf-8"))

    correctness = json.loads(Path(manifest["correctness"]).read_text(encoding="utf-8"))

    analysis = analyze_experiment(
        baseline_paths=[Path(value) for value in manifest["baseline_results"]],
        candidate_paths=[Path(value) for value in manifest["candidate_results"]],
        objective_metric=(experiment.objective_metric),
        objective_direction=(experiment.objective_direction),
        guardrail_limits=(experiment.guardrails),
        correctness_passed=bool(
            correctness.get(
                "passed",
                False,
            )
        ),
        selection_gate_passed=bool(
            selection.get(
                "selection_gate",
                {},
            ).get(
                "eligible",
                False,
            )
        ),
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(
            analysis.to_dict(),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"claim_ready={analysis.claim_ready}")

    print(f"Analysis written to: {args.output}")


if __name__ == "__main__":
    main()
