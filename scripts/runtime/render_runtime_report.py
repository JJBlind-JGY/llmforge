#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from llmforge.runtime.report import render_request_table, render_runtime_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports/runtime_trace.md"))
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    rendered = (
        render_runtime_report(summary)
        + "\n## Request details\n\n"
        + render_request_table(summary)
        + "\n"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(f"Runtime report written to: {args.output}")


if __name__ == "__main__":
    main()
