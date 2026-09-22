#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from llmforge.release import audit_release


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/release/audit.json"),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
    )
    args = parser.parse_args()

    result = audit_release(args.repo_root)

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(
            result.to_dict(),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            result.to_dict(),
            indent=2,
            sort_keys=True,
        )
    )

    if args.strict and not result.release_ready:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
