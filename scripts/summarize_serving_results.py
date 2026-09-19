#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from llmforge.serving.report import render_report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("results", nargs="+", type=Path)
    p.add_argument(
        "--output", type=Path, default=Path("reports/vllm_baseline.generated.md")
    )
    args = p.parse_args()

    artifacts = []
    for path in args.results:
        data = json.loads(path.read_text(encoding="utf-8"))
        artifacts.append((data.get("case_name", path.parent.name), data))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_report(artifacts), encoding="utf-8")
    print(f"Report written to: {args.output}")


if __name__ == "__main__":
    main()
