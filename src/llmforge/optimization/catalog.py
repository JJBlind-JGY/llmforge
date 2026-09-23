"""M7 candidate catalog with explicit upstream-overlap classification."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class CandidateClass(StrEnum):
    MAIN_CANDIDATE = "main_candidate"
    BASELINE_CONTROL = "baseline_control"
    UPSTREAM_EQUIVALENT = "upstream_equivalent"
    SUPPORTING_SYSTEM = "supporting_system"


@dataclass(frozen=True)
class CandidateCatalogEntry:
    candidate_id: str
    title: str
    area: str
    classification: CandidateClass
    implementation: str | None
    current_upstream_controls: tuple[str, ...]
    why_not_parameter_only: str
    evidence_needed: tuple[str, ...]
    notes: tuple[str, ...]

    @classmethod
    def from_dict(
        cls,
        raw: dict,
    ) -> CandidateCatalogEntry:
        return cls(
            candidate_id=str(raw["candidate_id"]),
            title=str(raw["title"]),
            area=str(raw["area"]),
            classification=(CandidateClass(raw["classification"])),
            implementation=raw.get("implementation"),
            current_upstream_controls=tuple(
                raw.get(
                    "current_upstream_controls",
                    (),
                )
            ),
            why_not_parameter_only=str(
                raw.get(
                    "why_not_parameter_only",
                    "",
                )
            ),
            evidence_needed=tuple(
                raw.get(
                    "evidence_needed",
                    (),
                )
            ),
            notes=tuple(
                raw.get(
                    "notes",
                    (),
                )
            ),
        )

    def to_dict(self) -> dict:
        return {
            "candidate_id": (self.candidate_id),
            "title": self.title,
            "area": self.area,
            "classification": (self.classification.value),
            "implementation": (self.implementation),
            "current_upstream_controls": list(self.current_upstream_controls),
            "why_not_parameter_only": (self.why_not_parameter_only),
            "evidence_needed": list(self.evidence_needed),
            "notes": list(self.notes),
        }


def load_candidate_catalog(
    path: Path,
) -> tuple[CandidateCatalogEntry, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))

    entries = tuple(
        CandidateCatalogEntry.from_dict(raw) for raw in payload["candidates"]
    )

    ids = [entry.candidate_id for entry in entries]

    if len(ids) != len(set(ids)):
        raise ValueError("Candidate IDs must be unique.")

    return entries
