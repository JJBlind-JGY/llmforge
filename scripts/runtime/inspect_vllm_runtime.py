#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from llmforge.runtime.integrations.vllm import discover_vllm_runtime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rendered = json.dumps(
        discover_vllm_runtime().to_dict(),
        indent=2,
        sort_keys=True,
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")

    print(rendered)


if __name__ == "__main__":
    main()
