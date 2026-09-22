#!/usr/bin/env python3
"""Compare deterministic baseline/candidate correctness artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llmforge.optimization.correctness import (
    compare_correctness,
    load_correctness_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--baseline",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--candidate",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--allow-response-difference",
        action="store_true",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    result = compare_correctness(
        load_correctness_jsonl(args.baseline),
        load_correctness_jsonl(args.candidate),
        require_exact_response=(not args.allow_response_difference),
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(
            result.to_dict(),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Correctness passed={result.passed}")

    if not result.passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
