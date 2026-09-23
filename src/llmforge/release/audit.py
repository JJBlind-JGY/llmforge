"""Repository release and portfolio readiness audit."""

from __future__ import annotations

import json
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

_RELEASE_REQUIRED_PATHS = (
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

_PORTFOLIO_REQUIRED_REPORTS = (
    "reports/vllm_baseline.md",
    "reports/runtime_trace.md",
    "reports/multi_gpu_inference.md",
    "reports/production_readiness.md",
    "reports/final_optimization.md",
    "reports/cross_engine_validation.md",
)

_BLOCKED_REPORT_MARKERS = (
    "PENDING",
    "TODO",
    "<measured result>",
    "<specific workload>",
)


@dataclass(frozen=True)
class AuditCheck:
    name: str
    gate: str
    passed: bool
    detail: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ReleaseAudit:
    release_ready: bool
    portfolio_complete: bool
    checks: tuple[AuditCheck, ...]

    def to_dict(self) -> dict:
        return {
            "release_ready": self.release_ready,
            "portfolio_complete": self.portfolio_complete,
            "checks": [check.to_dict() for check in self.checks],
        }


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _check_path(
    *,
    repo_root: Path,
    relative: str,
    gate: str,
) -> AuditCheck:
    path = repo_root / relative
    return AuditCheck(
        name=f"path:{relative}",
        gate=gate,
        passed=path.is_file(),
        detail="present" if path.is_file() else "missing",
    )


def _check_pyproject(repo_root: Path) -> AuditCheck:
    path = repo_root / "pyproject.toml"

    if not path.is_file():
        return AuditCheck(
            name="package_metadata",
            gate="release",
            passed=False,
            detail="pyproject.toml is missing.",
        )

    try:
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
        project = payload["project"]
    except (OSError, tomllib.TOMLDecodeError, KeyError) as exc:
        return AuditCheck(
            name="package_metadata",
            gate="release",
            passed=False,
            detail=f"invalid pyproject metadata: {exc}",
        )

    required = {
        "name": "llmforge-infra",
        "readme": "README.md",
        "license": "Apache-2.0",
    }

    mismatches = []

    for key, expected in required.items():
        actual = project.get(key)

        if actual != expected:
            mismatches.append(f"{key}={actual!r}, expected {expected!r}")

    urls = project.get("urls", {})

    if not isinstance(urls, dict) or not urls.get("Repository"):
        mismatches.append("project.urls.Repository is missing")

    return AuditCheck(
        name="package_metadata",
        gate="release",
        passed=not mismatches,
        detail=(
            "package metadata is release-ready"
            if not mismatches
            else "; ".join(mismatches)
        ),
    )


def _check_readme(repo_root: Path) -> AuditCheck:
    path = repo_root / "README.md"

    if not path.is_file():
        return AuditCheck(
            name="readme_public_entry",
            gate="release",
            passed=False,
            detail="README.md is missing.",
        )

    text = path.read_text(encoding="utf-8").lower()

    required_fragments = (
        "why llmforge",
        "architecture",
        "validated results",
        "project status",
        "quick start",
        "explore llmforge",
        "non-goals",
    )

    missing = [fragment for fragment in required_fragments if fragment not in text]

    return AuditCheck(
        name="readme_public_entry",
        gate="release",
        passed=not missing,
        detail=(
            "public README entry points are present"
            if not missing
            else "missing concepts: " + ", ".join(missing)
        ),
    )


def _check_legacy_layout(repo_root: Path) -> AuditCheck:
    legacy_reports = repo_root / "docs" / "reports"

    root_engineering_notes = list((repo_root / "docs").glob("M*_ENGINEERING_NOTES.md"))

    problems = []

    if legacy_reports.exists():
        problems.append("docs/reports still exists")

    if root_engineering_notes:
        problems.append("engineering notes still live at docs/ root")

    return AuditCheck(
        name="public_document_layout",
        gate="release",
        passed=not problems,
        detail=(
            "public document layout is consolidated"
            if not problems
            else "; ".join(problems)
        ),
    )


def _check_upstream_contribution(
    repo_root: Path,
) -> AuditCheck:
    path = repo_root / "docs" / "upstream_contribution.json"

    if not path.is_file():
        return AuditCheck(
            name="upstream_contribution",
            gate="portfolio",
            passed=False,
            detail="docs/upstream_contribution.json is missing.",
        )

    try:
        payload = _read_json(path)
    except (json.JSONDecodeError, OSError) as exc:
        return AuditCheck(
            name="upstream_contribution",
            gate="portfolio",
            passed=False,
            detail=f"invalid metadata: {exc}",
        )

    status = str(payload.get("status", ""))
    url = str(payload.get("url", ""))

    valid_status = status in {
        "issue",
        "patch",
        "pull_request",
    }

    valid_url = url.startswith(("http://", "https://"))

    return AuditCheck(
        name="upstream_contribution",
        gate="portfolio",
        passed=valid_status and valid_url,
        detail=f"status={status!r}, url={url!r}",
    )


def _check_completed_report(
    *,
    repo_root: Path,
    relative: str,
) -> tuple[AuditCheck, AuditCheck]:
    path = repo_root / relative

    exists = AuditCheck(
        name=f"path:{relative}",
        gate="portfolio",
        passed=path.is_file(),
        detail="present" if path.is_file() else "missing",
    )

    if not path.is_file():
        completed = AuditCheck(
            name=f"completed:{relative}",
            gate="portfolio",
            passed=False,
            detail="report is missing",
        )
        return exists, completed

    text = path.read_text(encoding="utf-8")

    found = [marker for marker in _BLOCKED_REPORT_MARKERS if marker in text]

    completed = AuditCheck(
        name=f"completed:{relative}",
        gate="portfolio",
        passed=not found,
        detail=(
            "no blocked placeholder markers"
            if not found
            else "found: " + ", ".join(found)
        ),
    )

    return exists, completed


def audit_release(
    repo_root: Path,
) -> ReleaseAudit:
    checks: list[AuditCheck] = []

    for relative in _RELEASE_REQUIRED_PATHS:
        checks.append(
            _check_path(
                repo_root=repo_root,
                relative=relative,
                gate="release",
            )
        )

    checks.append(_check_pyproject(repo_root))
    checks.append(_check_readme(repo_root))
    checks.append(_check_legacy_layout(repo_root))

    for relative in _PORTFOLIO_REQUIRED_REPORTS:
        exists, completed = _check_completed_report(
            repo_root=repo_root,
            relative=relative,
        )
        checks.extend((exists, completed))

    checks.append(_check_upstream_contribution(repo_root))

    release_checks = [check for check in checks if check.gate == "release"]

    portfolio_checks = [check for check in checks if check.gate == "portfolio"]

    release_ready = all(check.passed for check in release_checks)

    portfolio_complete = release_ready and all(
        check.passed for check in portfolio_checks
    )

    return ReleaseAudit(
        release_ready=release_ready,
        portfolio_complete=portfolio_complete,
        checks=tuple(checks),
    )
