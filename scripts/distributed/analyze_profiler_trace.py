#!/usr/bin/env python3
"""Summarize NCCL/collective events from one or more profiler traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llmforge.distributed.profiling import (
    summarize_communication_trace,
)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "traces",
        nargs="+",
        type=Path,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    summaries = []

    for trace in args.traces:
        summary = summarize_communication_trace(trace)

        summaries.append(
            {
                "trace": str(trace),
                "summary": (summary.to_dict()),
            }
        )

    payload = {
        "traces": summaries,
        "note": (
            "Communication duration is extracted "
            "from profiler events and must not be "
            "equated with client E2E latency."
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

    print(f"Profiler summary written to: {args.output}")


if __name__ == "__main__":
    main()
