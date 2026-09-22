"""Liveness/readiness checks for the M6 control plane."""

from __future__ import annotations

import json
import subprocess
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Protocol


class HealthStatus(StrEnum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    required: bool
    detail: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class HealthReport:
    status: HealthStatus
    ready: bool
    checks: tuple[CheckResult, ...]
    consecutive_failures: dict[
        str,
        int,
    ]

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "ready": self.ready,
            "checks": [check.to_dict() for check in self.checks],
            "consecutive_failures": dict(self.consecutive_failures),
        }

    def to_json(self) -> bytes:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
        ).encode("utf-8")


class HealthCheck(Protocol):
    name: str
    required: bool

    def check(
        self,
    ) -> CheckResult: ...


class UpstreamHealthCheck:
    name = "upstream_vllm"

    def __init__(
        self,
        *,
        base_url: str,
        timeout_s: float,
        required: bool,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.required = required

    def check(
        self,
    ) -> CheckResult:
        try:
            with urllib.request.urlopen(
                self.base_url + "/health",
                timeout=self.timeout_s,
            ) as response:
                status = int(response.status)

            ok = 200 <= status < 300

            detail = f"HTTP {status}"

        except (
            urllib.error.URLError,
            TimeoutError,
            OSError,
        ) as exc:
            ok = False
            detail = f"{type(exc).__name__}: {exc}"

        return CheckResult(
            name=self.name,
            ok=ok,
            required=self.required,
            detail=detail,
        )


class GpuHealthCheck:
    name = "gpu"

    def __init__(
        self,
        *,
        gpu_indices: tuple[int, ...],
        required: bool,
    ) -> None:
        self.gpu_indices = gpu_indices
        self.required = required

    def check(
        self,
    ) -> CheckResult:
        try:
            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    ("--query-gpu=index"),
                    ("--format=csv,noheader,nounits"),
                ],
                text=True,
                timeout=3,
            )

            available = {
                int(line.strip()) for line in output.splitlines() if line.strip()
            }

            missing = set(self.gpu_indices) - available

            ok = not missing

            detail = (
                "available"
                if ok
                else (
                    "missing GPUs: " + ",".join(str(index) for index in sorted(missing))
                )
            )

        except (
            subprocess.SubprocessError,
            OSError,
            ValueError,
        ) as exc:
            ok = False
            detail = f"{type(exc).__name__}: {exc}"

        return CheckResult(
            name=self.name,
            ok=ok,
            required=self.required,
            detail=detail,
        )


class HealthEvaluator:
    def __init__(
        self,
        *,
        checks: list[HealthCheck],
        failure_threshold: int,
    ) -> None:
        if failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive.")

        self.checks = list(checks)

        self.failure_threshold = failure_threshold

        self._failures = {check.name: 0 for check in self.checks}

    def evaluate(
        self,
    ) -> HealthReport:
        results = []

        for check in self.checks:
            result = check.check()
            results.append(result)

            if result.ok:
                self._failures[result.name] = 0
            else:
                self._failures[result.name] = (
                    self._failures.get(
                        result.name,
                        0,
                    )
                    + 1
                )

        required_failures = [
            result for result in results if (result.required and not result.ok)
        ]

        optional_failures = [
            result for result in results if (not result.required and not result.ok)
        ]

        ready = not required_failures

        if required_failures:
            threshold_exceeded = any(
                self._failures[result.name] >= self.failure_threshold
                for result in required_failures
            )

            status = (
                HealthStatus.UNHEALTHY if threshold_exceeded else HealthStatus.DEGRADED
            )

        elif optional_failures:
            status = HealthStatus.DEGRADED

        else:
            status = HealthStatus.HEALTHY

        return HealthReport(
            status=status,
            ready=ready,
            checks=tuple(results),
            consecutive_failures=dict(self._failures),
        )
