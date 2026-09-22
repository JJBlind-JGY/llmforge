#!/usr/bin/env python3
"""Print M7 candidate classifications before implementation/experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llmforge.optimization.catalog import (
    load_candidate_catalog,
)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("configs/optimization/m7_candidate_catalog.json"),
    )

    args = parser.parse_args()

    entries = load_candidate_catalog(args.catalog)

    payload = {
        "candidates": [entry.to_dict() for entry in entries],
        "main_candidate_ids": [
            entry.candidate_id
            for entry in entries
            if (entry.classification.value == "main_candidate")
        ],
        "not_headline_ids": [
            entry.candidate_id
            for entry in entries
            if (entry.classification.value != "main_candidate")
        ],
    }

    print(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
