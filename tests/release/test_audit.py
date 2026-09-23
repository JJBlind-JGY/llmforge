import json
from pathlib import Path

from llmforge.release import audit_release

RELEASE_PATHS = (
    "README.md",
    "DESIGN.md",
    "pyproject.toml",
    "docs/index.md",
    "docs/getting_started.md",
    "docs/learning_path.md",
    "docs/hardware_tiers.md",
    "docs/compatibility.md",
    "docs/architecture.md",
    "docs/benchmark_methodology.md",
    "docs/reproduction.md",
    "reports/gpu_kernel_performance.md",
    "reports/transformer_runtime.md",
    "examples/README.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "CODE_OF_CONDUCT.md",
    "LICENSE",
    "CHANGELOG.md",
    "CITATION.cff",
    ".github/workflows/ci-cpu.yml",
)

PORTFOLIO_REPORTS = (
    "reports/vllm_baseline.md",
    "reports/runtime_trace.md",
    "reports/multi_gpu_inference.md",
    "reports/production_readiness.md",
    "reports/final_optimization.md",
    "reports/cross_engine_validation.md",
)


def write(path: Path, content: str = "complete\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def create_release_surface(root: Path) -> None:
    for relative in RELEASE_PATHS:
        if relative in {
            "README.md",
            "pyproject.toml",
        }:
            continue

        write(root / relative)

    write(
        root / "README.md",
        """
# LLMForge Infra
## Why LLMForge?
## Architecture
## Validated Results
## Project Status
## Quick Start
## Explore LLMForge
## Non-Goals
""".strip()
        + "\n",
    )

    write(
        root / "pyproject.toml",
        """
[project]
name = "llmforge"
version = "0.1.0a1"
readme = "README.md"
license = "Apache-2.0"

[project.urls]
Repository = "https://github.com/JJBlind-JGY/llmforge"
""".strip()
        + "\n",
    )


def test_release_can_be_ready_before_portfolio_is_complete(
    tmp_path: Path,
) -> None:
    create_release_surface(tmp_path)

    result = audit_release(tmp_path)

    assert result.release_ready
    assert not result.portfolio_complete


def test_release_gate_fails_when_public_surface_is_missing(
    tmp_path: Path,
) -> None:
    create_release_surface(tmp_path)
    (tmp_path / "docs" / "learning_path.md").unlink()

    result = audit_release(tmp_path)

    assert not result.release_ready


def test_portfolio_requires_completed_reports_and_upstream_attempt(
    tmp_path: Path,
) -> None:
    create_release_surface(tmp_path)

    for relative in PORTFOLIO_REPORTS:
        write(
            tmp_path / relative,
            "Measured experiment report.\n",
        )

    result = audit_release(tmp_path)

    assert result.release_ready
    assert not result.portfolio_complete

    write(
        tmp_path / "docs" / "upstream_contribution.json",
        json.dumps(
            {
                "status": "issue",
                "url": ("https://github.com/example/project/issues/1"),
            }
        ),
    )

    result = audit_release(tmp_path)

    assert result.portfolio_complete


def test_pending_marker_blocks_portfolio_completion(
    tmp_path: Path,
) -> None:
    create_release_surface(tmp_path)

    for relative in PORTFOLIO_REPORTS:
        write(
            tmp_path / relative,
            "Measured experiment report.\n",
        )

    write(
        tmp_path / "reports" / "final_optimization.md",
        "PENDING final result\n",
    )

    write(
        tmp_path / "docs" / "upstream_contribution.json",
        json.dumps(
            {
                "status": "pull_request",
                "url": ("https://github.com/example/project/pull/1"),
            }
        ),
    )

    result = audit_release(tmp_path)

    assert result.release_ready
    assert not result.portfolio_complete
