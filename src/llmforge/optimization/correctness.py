"""Correctness artifacts and comparison for scheduler/runtime optimizations."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class CorrectnessRecord:
    case_id: str
    prompt_sha256: str
    response_sha256: str
    prompt_tokens: int | None
    completion_tokens: int | None
    finish_reason: str | None
    success: bool
    error: str | None = None

    @classmethod
    def from_dict(
        cls,
        raw: dict,
    ) -> "CorrectnessRecord":
        return cls(
            case_id=str(raw["case_id"]),
            prompt_sha256=str(raw["prompt_sha256"]),
            response_sha256=str(
                raw.get(
                    "response_sha256",
                    "",
                )
            ),
            prompt_tokens=(
                int(raw["prompt_tokens"])
                if raw.get("prompt_tokens") is not None
                else None
            ),
            completion_tokens=(
                int(raw["completion_tokens"])
                if raw.get("completion_tokens") is not None
                else None
            ),
            finish_reason=raw.get("finish_reason"),
            success=bool(raw["success"]),
            error=raw.get("error"),
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class CorrectnessMismatch:
    case_id: str
    field: str
    baseline: object
    candidate: object

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class CorrectnessComparison:
    passed: bool
    baseline_cases: int
    candidate_cases: int
    mismatches: tuple[CorrectnessMismatch, ...]

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "baseline_cases": (self.baseline_cases),
            "candidate_cases": (self.candidate_cases),
            "mismatches": [item.to_dict() for item in self.mismatches],
        }


def load_correctness_jsonl(
    path: Path,
) -> dict[
    str,
    CorrectnessRecord,
]:
    records: dict[
        str,
        CorrectnessRecord,
    ] = {}

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line_number, line in enumerate(
            handle,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            record = CorrectnessRecord.from_dict(json.loads(line))

            if record.case_id in records:
                raise ValueError(
                    "Duplicate correctness "
                    f"case_id={record.case_id!r} "
                    f"at line {line_number}."
                )

            records[record.case_id] = record

    return records


def compare_correctness(
    baseline: dict[
        str,
        CorrectnessRecord,
    ],
    candidate: dict[
        str,
        CorrectnessRecord,
    ],
    *,
    require_exact_response: bool = True,
) -> CorrectnessComparison:
    mismatches = []

    all_case_ids = sorted(set(baseline) | set(candidate))

    for case_id in all_case_ids:
        left = baseline.get(case_id)

        right = candidate.get(case_id)

        if left is None:
            mismatches.append(
                CorrectnessMismatch(
                    case_id=case_id,
                    field="missing_baseline",
                    baseline=None,
                    candidate=(right.to_dict() if right else None),
                )
            )
            continue

        if right is None:
            mismatches.append(
                CorrectnessMismatch(
                    case_id=case_id,
                    field="missing_candidate",
                    baseline=(left.to_dict()),
                    candidate=None,
                )
            )
            continue

        if not left.success:
            mismatches.append(
                CorrectnessMismatch(
                    case_id=case_id,
                    field="baseline_success",
                    baseline=False,
                    candidate=right.success,
                )
            )
            continue

        fields = [
            "prompt_sha256",
            "prompt_tokens",
            "completion_tokens",
            "finish_reason",
            "success",
        ]

        if require_exact_response:
            fields.append("response_sha256")

        for field in fields:
            left_value = getattr(
                left,
                field,
            )

            right_value = getattr(
                right,
                field,
            )

            if left_value != right_value:
                mismatches.append(
                    CorrectnessMismatch(
                        case_id=case_id,
                        field=field,
                        baseline=left_value,
                        candidate=right_value,
                    )
                )

    return CorrectnessComparison(
        passed=not mismatches,
        baseline_cases=len(baseline),
        candidate_cases=len(candidate),
        mismatches=tuple(mismatches),
    )
