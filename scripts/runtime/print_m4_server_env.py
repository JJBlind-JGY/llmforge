#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trace-path", default="artifacts/runtime/vllm_runtime_{pid}.jsonl"
    )
    parser.add_argument(
        "--mode", choices=["off", "sync", "buffered"], default="buffered"
    )
    args = parser.parse_args()
    print("export LLMFORGE_RUNTIME_TRACE_MODE=" + shlex.quote(args.mode))
    print("export LLMFORGE_RUNTIME_TRACE_PATH=" + shlex.quote(args.trace_path))
    print(
        "--scheduler-cls llmforge.runtime.integrations.vllm.tracing_scheduler.TracingScheduler"
    )


if __name__ == "__main__":
    main()
