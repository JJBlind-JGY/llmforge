#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from llmforge.engines import create_engine_adapter
from llmforge.engines.probe import probe_engine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["vllm", "sglang"], required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--timeout-s", type=float, default=3.0)
    args = parser.parse_args()

    result = probe_engine(
        adapter=create_engine_adapter(args.engine),
        base_url=args.base_url,
        timeout_s=args.timeout_s,
    )

    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))

    if not result.ready:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
