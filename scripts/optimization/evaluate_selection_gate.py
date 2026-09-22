#!/usr/bin/env python3
"""Evaluate the eight-condition M7 candidate-selection gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llmforge.optimization import (
    candidate_from_dict,
    evaluate_selection_gate,
)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--candidate",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    candidate = candidate_from_dict(
        json.loads(args.candidate.read_text(encoding="utf-8"))
    )

    result = evaluate_selection_gate(candidate)

    payload = {
        "candidate": (candidate.to_dict()),
        "selection_gate": (result.to_dict()),
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"eligible={result.eligible}")

    print(f"Selection result written to: {args.output}")


if __name__ == "__main__":
    main()
