"""Validation rules for formal benchmark runs."""

from __future__ import annotations

import os
from typing import Any


class BenchmarkEnvironmentError(RuntimeError):
    """Raised when a formal benchmark run environment is invalid."""


def validate_formal_benchmark(
    environment: dict[str, Any], require_single_gpu: bool = True
) -> None:
    """Validate conditions required for a formal benchmark."""
    problems: list[str] = []
    git_metadata = environment["project"]["git"]

    if git_metadata["dirty"]:
        problems.append("Git worktree is dirty.")

    if os.environ.get("CUDA_PREFIX"):
        problems.append("A Conda environment is active.")

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")

    if require_single_gpu:
        if visible_devices is None:
            problems.append("CUDA_VISIBLE_DEVICES is not explicitly set.")
        elif "," in visible_devices:
            problems.append("Single-GPU benchmark exposes multiple GPUs.")

    if problems:
        message = "Formal benchmark validation failed:\n-" + "\n-".join(problems)
        raise BenchmarkEnvironmentError(message)
