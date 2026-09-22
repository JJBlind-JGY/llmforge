#!/usr/bin/env python3
"""Summarize adaptive-budget policy decisions."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "trace",
        type=Path,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    rows = [
        json.loads(
            line
        )
        for line in args.trace.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    changed = [
        row
        for row in rows
        if row.get(
            "changed"
        )
    ]

    reasons = Counter(
        str(
            row.get(
                "reason",
                "unknown",
            )
        )
        for row in rows
    )

    payload = {
        "steps": len(rows),
        "changed_steps": len(
            changed
        ),
        "changed_fraction": (
            len(changed)
            / len(rows)
            if rows
            else 0.0
        ),
        "reason_counts": dict(
            reasons
        ),
        "mean_base_budget": (
            mean(
                float(
                    row[
                        "base_budget"
                    ]
                )
                for row in rows
            )
            if rows
            else None
        ),
        "mean_applied_budget": (
            mean(
                float(
                    row[
                        "applied_budget"
                    ]
                )
                for row in rows
            )
            if rows
            else None
        ),
        "max_kv_usage_ratio": (
            max(
                float(
                    row[
                        "pressure"
                    ][
                        "kv_usage_ratio"
                    ]
                )
                for row in rows
            )
            if rows
            else None
        ),
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

    print(
        f"Decision summary written to: "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()
