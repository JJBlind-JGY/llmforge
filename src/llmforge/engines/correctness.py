"""Cross-engine correctness comparison.

The first M8 comparison uses deterministic OpenAI-compatible chat requests and
compares semantic/result-shape invariants. Exact text equality is optional
because two correct serving engines can expose tiny floating-point-dependent
generation differences even under greedy decoding.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class EngineCorrectnessRecord:
    case_id: str
    engine: str
    success: bool
    response_sha256: str
    prompt_tokens: int | None
    completion_tokens: int | None
    finish_reason: str | None
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class EngineCorrectnessMismatch:
    case_id: str
    field: str
    left: object
    right: object

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class EngineCorrectnessComparison:
    passed: bool
    exact_text: bool
    mismatches: tuple[EngineCorrectnessMismatch, ...]

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "exact_text": self.exact_text,
            "mismatches": [item.to_dict() for item in self.mismatches],
        }


def compare_engine_correctness(
    left: dict[str, EngineCorrectnessRecord],
    right: dict[str, EngineCorrectnessRecord],
    *,
    require_exact_text: bool = False,
) -> EngineCorrectnessComparison:
    mismatches = []

    case_ids = sorted(set(left) | set(right))

    for case_id in case_ids:
        a = left.get(case_id)
        b = right.get(case_id)

        if a is None or b is None:
            mismatches.append(
                EngineCorrectnessMismatch(
                    case_id=case_id,
                    field="missing_case",
                    left=None if a is None else a.to_dict(),
                    right=None if b is None else b.to_dict(),
                )
            )
            continue

        for field in (
            "success",
            "prompt_tokens",
            "completion_tokens",
            "finish_reason",
        ):
            if getattr(a, field) != getattr(b, field):
                mismatches.append(
                    EngineCorrectnessMismatch(
                        case_id=case_id,
                        field=field,
                        left=getattr(a, field),
                        right=getattr(b, field),
                    )
                )

        if require_exact_text and a.response_sha256 != b.response_sha256:
            mismatches.append(
                EngineCorrectnessMismatch(
                    case_id=case_id,
                    field="response_sha256",
                    left=a.response_sha256,
                    right=b.response_sha256,
                )
            )

    return EngineCorrectnessComparison(
        passed=not mismatches,
        exact_text=require_exact_text,
        mismatches=tuple(mismatches),
    )
