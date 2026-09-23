"""LLMForge Infra: LLM inference infrastructure and performance engineering."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as distribution_version

try:
    __version__ = distribution_version("llmforge")
except PackageNotFoundError:
    # Source-tree fallback when the package has not been installed.
    __version__ = "0+unknown"


__all__ = ["__version__"]
