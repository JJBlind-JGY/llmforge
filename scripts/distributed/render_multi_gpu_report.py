#!/usr/bin/env python3
"""Render the M5 Markdown report from normalized analysis JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llmforge.distributed.report import (
    render_multi_gpu_report,
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
        default=Path("reports/multi_gpu_inference.md"),
    )

    args = parser.parse_args()

    payload = json.loads(args.analysis.read_text(encoding="utf-8"))

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        render_multi_gpu_report(payload),
        encoding="utf-8",
    )

    print(f"M5 report written to: {args.output}")


if __name__ == "__main__":
    main()
