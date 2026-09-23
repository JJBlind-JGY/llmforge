#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from llmforge.runtime.analysis import summarize_runtime_trace
from llmforge.runtime.trace import JsonlTraceReader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize_runtime_trace(JsonlTraceReader(args.trace).read_all())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Runtime summary written to: {args.output}")


if __name__ == "__main__":
    main()
