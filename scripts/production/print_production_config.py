#!/usr/bin/env python3
"""Render the effective M6 config after file + environment overrides."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llmforge.production import (
    load_production_config,
)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/production/m6_observability.json"),
    )

    args = parser.parse_args()

    config = load_production_config(args.config)

    print(
        json.dumps(
            config.to_dict(),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
