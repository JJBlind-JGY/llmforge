"""Benchmark experiment artifact utilities."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def create_run_directory(
    root: Path, experiment_name: str, git_commit: str | None
) -> Path:
    """Create a unique benchmark run directory."""
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    commit = git_commit[:7] if git_commit else "nogit"
    run_directory = root / experiment_name / f"{timestamp}_{commit}"

    run_directory.mkdir(parents=True, exist_ok=True)

    return run_directory


def write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    """Write a JSON benchmark artifact."""
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
