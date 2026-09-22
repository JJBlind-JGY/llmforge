"""M7 candidate/evidence contracts.

A candidate may exist in code before it is selected, but it must not become the
headline optimization until the original syllabus selection gate is satisfied.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class OptimizationArea(StrEnum):
    SCHEDULER_FAIRNESS = "scheduler_fairness"
    PREFILL_DECODE = "prefill_decode_interference"
    KV_PRESSURE = "kv_cache_pressure"
    CPU_RUNTIME = "cpu_runtime_overhead"
    GPU_KERNEL = "gpu_kernel"
    MULTI_GPU = "multi_gpu_communication"


class CandidateState(StrEnum):
    DRAFT = "draft"
    EVIDENCE_PENDING = "evidence_pending"
    EVIDENCE_READY = "evidence_ready"
    SELECTED = "selected"
    IMPLEMENTED = "implemented"
    VALIDATED = "validated"
    REJECTED = "rejected"


@dataclass(frozen=True)
class SelectionEvidence:
    reproduction_artifacts: tuple[str, ...] = ()
    profiler_artifacts: tuple[str, ...] = ()
    mechanism_note: str = ""
    baseline_artifact: str | None = None
    correctness_plan: str = ""
    regression_metrics: tuple[str, ...] = ()
    hardware_feasible: bool = False
    community_evidence: tuple[str, ...] = ()
    upstream_overlap: str = ""
    parameter_only: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class OptimizationCandidate:
    candidate_id: str
    title: str
    area: OptimizationArea
    problem: str
    mechanism: str
    changed_component: str
    expected_benefit: str
    expected_tradeoffs: tuple[str, ...]
    state: CandidateState = CandidateState.DRAFT
    evidence: SelectionEvidence = field(default_factory=SelectionEvidence)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["area"] = self.area.value
        data["state"] = self.state.value
        return data


def candidate_from_dict(
    raw: dict[str, Any],
) -> OptimizationCandidate:
    evidence_raw = dict(
        raw.get(
            "evidence",
            {},
        )
    )

    evidence = SelectionEvidence(
        reproduction_artifacts=tuple(
            evidence_raw.get(
                "reproduction_artifacts",
                (),
            )
        ),
        profiler_artifacts=tuple(
            evidence_raw.get(
                "profiler_artifacts",
                (),
            )
        ),
        mechanism_note=str(
            evidence_raw.get(
                "mechanism_note",
                "",
            )
        ),
        baseline_artifact=(evidence_raw.get("baseline_artifact")),
        correctness_plan=str(
            evidence_raw.get(
                "correctness_plan",
                "",
            )
        ),
        regression_metrics=tuple(
            evidence_raw.get(
                "regression_metrics",
                (),
            )
        ),
        hardware_feasible=bool(
            evidence_raw.get(
                "hardware_feasible",
                False,
            )
        ),
        community_evidence=tuple(
            evidence_raw.get(
                "community_evidence",
                (),
            )
        ),
        upstream_overlap=str(
            evidence_raw.get(
                "upstream_overlap",
                "",
            )
        ),
        parameter_only=bool(
            evidence_raw.get(
                "parameter_only",
                False,
            )
        ),
    )

    return OptimizationCandidate(
        candidate_id=str(raw["candidate_id"]),
        title=str(raw["title"]),
        area=OptimizationArea(raw["area"]),
        problem=str(raw["problem"]),
        mechanism=str(raw["mechanism"]),
        changed_component=str(raw["changed_component"]),
        expected_benefit=str(raw["expected_benefit"]),
        expected_tradeoffs=tuple(
            raw.get(
                "expected_tradeoffs",
                (),
            )
        ),
        state=CandidateState(
            raw.get(
                "state",
                "evidence_pending",
            )
        ),
        evidence=evidence,
        metadata=dict(
            raw.get(
                "metadata",
                {},
            )
        ),
    )
