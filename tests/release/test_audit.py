import json
from pathlib import Path

from llmforge.release import audit_release

REQUIRED = (
    "README.md",
    "DESIGN.md",
    "docs/architecture.md",
    "docs/benchmark_methodology.md",
    "docs/reproduction.md",
    "docs/vllm_request_lifecycle.md",
    "docs/cross_engine_validation.md",
    "docs/system_radar.md",
    "reports/gpu_kernel_performance.md",
    "reports/vllm_baseline.md",
    "reports/multi_gpu_inference.md",
    "reports/final_optimization.md",
    "CONTRIBUTING.md",
    "LICENSE",
)


def test_release_audit_passes_complete_repo(
    tmp_path: Path,
) -> None:
    for relative in REQUIRED:
        path = tmp_path / relative
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        content = "complete\n"

        if relative == "README.md":
            content = (
                "What is LLMForge\n"
                "Architecture\n"
                "Key Results\n"
                "Supported Environment\n"
                "Quick Start\n"
                "Reproduce\n"
            )

        if relative == "reports/final_optimization.md":
            content = "Measured final optimization report."

        path.write_text(
            content,
            encoding="utf-8",
        )

    upstream = tmp_path / "docs" / "upstream_contribution.json"

    upstream.write_text(
        json.dumps(
            {
                "status": "issue",
                "url": ("https://github.com/example/project/issues/1"),
            }
        ),
        encoding="utf-8",
    )

    result = audit_release(tmp_path)

    assert result.release_ready


def test_release_audit_blocks_missing_upstream_attempt(
    tmp_path: Path,
) -> None:
    result = audit_release(tmp_path)

    assert not result.release_ready
