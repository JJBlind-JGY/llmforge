"""The eight-condition M7 optimization-selection gate."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .contract import (
    OptimizationCandidate,
)


@dataclass(frozen=True)
class GateCheck:
    criterion: str
    passed: bool
    detail: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SelectionGateResult:
    candidate_id: str
    eligible: bool
    checks: tuple[GateCheck, ...]

    def to_dict(self) -> dict:
        return {
            "candidate_id": (self.candidate_id),
            "eligible": self.eligible,
            "checks": [check.to_dict() for check in self.checks],
        }


def evaluate_selection_gate(
    candidate: OptimizationCandidate,
    *,
    min_reproduction_runs: int = 3,
) -> SelectionGateResult:
    evidence = candidate.evidence

    checks = (
        GateCheck(
            criterion=("stable_real_workload"),
            passed=(len(evidence.reproduction_artifacts) >= min_reproduction_runs),
            detail=(
                "Requires repeated raw baseline artifacts from the same real workload."
            ),
        ),
        GateCheck(
            criterion=("profiler_or_telemetry"),
            passed=bool(evidence.profiler_artifacts),
            detail=("Requires profiler/runtime/telemetry evidence."),
        ),
        GateCheck(
            criterion=("mechanism_explained"),
            passed=bool(evidence.mechanism_note.strip()),
            detail=("Requires a concrete systems mechanism, not correlation only."),
        ),
        GateCheck(
            criterion=("not_parameter_only"),
            passed=(not evidence.parameter_only),
            detail=(
                "Changing one existing default parameter is not a main M7 optimization."
            ),
        ),
        GateCheck(
            criterion=("baseline_frozen"),
            passed=bool(evidence.baseline_artifact),
            detail=("Requires a frozen baseline artifact/config."),
        ),
        GateCheck(
            criterion=("correctness_and_regression"),
            passed=(
                bool(evidence.correctness_plan.strip())
                and bool(evidence.regression_metrics)
            ),
            detail=("Requires correctness and performance guardrails."),
        ),
        GateCheck(
            criterion=("hardware_feasible"),
            passed=(evidence.hardware_feasible),
            detail=("Must be credible on the available hardware."),
        ),
        GateCheck(
            criterion=("community_relevance"),
            passed=bool(evidence.community_evidence),
            detail=("Requires a current upstream issue/doc/paper/source reference."),
        ),
    )

    return SelectionGateResult(
        candidate_id=(candidate.candidate_id),
        eligible=all(check.passed for check in checks),
        checks=checks,
    )
