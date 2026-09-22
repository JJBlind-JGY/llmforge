#!/usr/bin/env python3
"""Render the final M7 Markdown trade-off report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llmforge.optimization.report import (
    render_optimization_report,
)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "analysis",
        type=Path,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/optimization_report.md"),
    )

    args = parser.parse_args()

    analysis = json.loads(args.analysis.read_text(encoding="utf-8"))

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        render_optimization_report(analysis),
        encoding="utf-8",
    )

    print(f"Report written to: {args.output}")


if __name__ == "__main__":
    main()
