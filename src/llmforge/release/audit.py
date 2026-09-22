"""Repository release audit for the M8 exit criteria."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


_REQUIRED_PATHS = (
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


@dataclass(frozen=True)
class AuditCheck:
    name: str
    passed: bool
    detail: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ReleaseAudit:
    release_ready: bool
    checks: tuple[AuditCheck, ...]

    def to_dict(self) -> dict:
        return {
            "release_ready": self.release_ready,
            "checks": [check.to_dict() for check in self.checks],
        }


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def audit_release(
    repo_root: Path,
) -> ReleaseAudit:
    checks: list[AuditCheck] = []

    for relative in _REQUIRED_PATHS:
        path = repo_root / relative
        checks.append(
            AuditCheck(
                name=f"path:{relative}",
                passed=path.is_file(),
                detail=("present" if path.is_file() else "missing"),
            )
        )

    upstream = repo_root / "docs" / "upstream_contribution.json"

    if not upstream.is_file():
        checks.append(
            AuditCheck(
                name="upstream_contribution",
                passed=False,
                detail=("docs/upstream_contribution.json is missing."),
            )
        )
    else:
        try:
            payload = _read_json(upstream)

            status = str(
                payload.get(
                    "status",
                    "",
                )
            )

            url = str(
                payload.get(
                    "url",
                    "",
                )
            )

            valid_status = status in {
                "issue",
                "patch",
                "pull_request",
            }

            checks.append(
                AuditCheck(
                    name="upstream_contribution",
                    passed=(valid_status and url.startswith(("http://", "https://"))),
                    detail=(f"status={status!r}, url={url!r}"),
                )
            )

        except (
            json.JSONDecodeError,
            OSError,
        ) as exc:
            checks.append(
                AuditCheck(
                    name="upstream_contribution",
                    passed=False,
                    detail=(f"invalid metadata: {exc}"),
                )
            )

    final_report = repo_root / "reports" / "final_optimization.md"

    if final_report.is_file():
        text = final_report.read_text(encoding="utf-8")

        placeholders = (
            "PENDING",
            "TODO",
            "<measured result>",
            "<specific workload>",
        )

        found = [token for token in placeholders if token in text]

        checks.append(
            AuditCheck(
                name="final_optimization_no_placeholders",
                passed=not found,
                detail=(
                    "no blocked placeholders"
                    if not found
                    else ("found: " + ", ".join(found))
                ),
            )
        )

    readme = repo_root / "README.md"

    if readme.is_file():
        text = readme.read_text(encoding="utf-8").lower()

        required_sections = (
            "what is llmforge",
            "architecture",
            "key results",
            "supported environment",
            "quick start",
            "reproduce",
        )

        missing = [item for item in required_sections if item not in text]

        checks.append(
            AuditCheck(
                name="readme_60_second_experience",
                passed=not missing,
                detail=(
                    "all required concepts found"
                    if not missing
                    else "missing concepts: " + ", ".join(missing)
                ),
            )
        )

    return ReleaseAudit(
        release_ready=all(check.passed for check in checks),
        checks=tuple(checks),
    )
