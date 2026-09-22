#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from llmforge.engines.correctness import (
    EngineCorrectnessRecord,
    compare_engine_correctness,
)


def load(path: Path) -> dict[str, EngineCorrectnessRecord]:
    records = {}

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue

            raw = json.loads(line)

            record = EngineCorrectnessRecord(
                case_id=str(raw["case_id"]),
                engine=str(raw["engine"]),
                success=bool(raw["success"]),
                response_sha256=str(raw.get("response_sha256", "")),
                prompt_tokens=(
                    None
                    if raw.get("prompt_tokens") is None
                    else int(raw["prompt_tokens"])
                ),
                completion_tokens=(
                    None
                    if raw.get("completion_tokens") is None
                    else int(raw["completion_tokens"])
                ),
                finish_reason=raw.get("finish_reason"),
                error=raw.get("error"),
            )

            records[record.case_id] = record

    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--exact-text", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = compare_engine_correctness(
        load(args.left),
        load(args.right),
        require_exact_text=args.exact_text,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"cross_engine_correctness_passed={result.passed}")

    if not result.passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
