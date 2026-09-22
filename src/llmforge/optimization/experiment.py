"""Frozen baseline/candidate experiment contract for M7."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def stable_hash(
    payload: dict[str, Any],
) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ExperimentArm:
    name: str
    server_config: dict[str, Any]
    environment: dict[str, str]
    scheduler_class: str | None

    @property
    def fingerprint(
        self,
    ) -> str:
        return stable_hash(
            {
                "server_config": (self.server_config),
                "environment": (self.environment),
                "scheduler_class": (self.scheduler_class),
            }
        )

    def to_dict(self) -> dict:
        data = asdict(self)
        data["fingerprint"] = self.fingerprint
        return data


@dataclass(frozen=True)
class OptimizationExperiment:
    schema_version: int
    experiment_id: str
    candidate_id: str
    workload_id: str
    workload_config: str
    repetitions: int
    warmup_requests: int
    baseline: ExperimentArm
    candidate: ExperimentArm
    objective_metric: str
    objective_direction: str
    guardrails: dict[str, float]
    notes: tuple[str, ...]

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError("Unsupported experiment schema.")

        if self.repetitions < 3:
            raise ValueError("M7 requires at least three repetitions per arm.")

        if self.warmup_requests < 0:
            raise ValueError("warmup_requests must be non-negative.")

        if self.objective_direction not in {
            "minimize",
            "maximize",
        }:
            raise ValueError("objective_direction must be minimize or maximize.")

        if self.baseline.server_config.get("model") != self.candidate.server_config.get(
            "model"
        ):
            raise ValueError("Baseline/candidate model must match.")

        if self.baseline.server_config.get(
            "revision"
        ) != self.candidate.server_config.get("revision"):
            raise ValueError("Baseline/candidate model revision must match.")

        if self.baseline.server_config.get("dtype") != self.candidate.server_config.get(
            "dtype"
        ):
            raise ValueError("Baseline/candidate dtype must match.")

        for metric, limit in self.guardrails.items():
            if limit < 0:
                raise ValueError(f"Guardrail limits must be non-negative: {metric}.")

    def to_dict(self) -> dict:
        return {
            "schema_version": (self.schema_version),
            "experiment_id": (self.experiment_id),
            "candidate_id": (self.candidate_id),
            "workload_id": (self.workload_id),
            "workload_config": (self.workload_config),
            "repetitions": (self.repetitions),
            "warmup_requests": (self.warmup_requests),
            "baseline": (self.baseline.to_dict()),
            "candidate": (self.candidate.to_dict()),
            "objective_metric": (self.objective_metric),
            "objective_direction": (self.objective_direction),
            "guardrails": dict(self.guardrails),
            "notes": list(self.notes),
        }


def _arm(
    raw: dict,
) -> ExperimentArm:
    return ExperimentArm(
        name=str(raw["name"]),
        server_config=dict(raw["server_config"]),
        environment={
            str(key): str(value)
            for key, value in raw.get(
                "environment",
                {},
            ).items()
        },
        scheduler_class=raw.get("scheduler_class"),
    )


def load_experiment(
    path: Path,
) -> OptimizationExperiment:
    raw = json.loads(path.read_text(encoding="utf-8"))

    experiment = OptimizationExperiment(
        schema_version=int(
            raw.get(
                "schema_version",
                1,
            )
        ),
        experiment_id=str(raw["experiment_id"]),
        candidate_id=str(raw["candidate_id"]),
        workload_id=str(raw["workload_id"]),
        workload_config=str(raw["workload_config"]),
        repetitions=int(raw["repetitions"]),
        warmup_requests=int(
            raw.get(
                "warmup_requests",
                0,
            )
        ),
        baseline=_arm(raw["baseline"]),
        candidate=_arm(raw["candidate"]),
        objective_metric=str(raw["objective_metric"]),
        objective_direction=str(raw["objective_direction"]),
        guardrails={
            str(key): float(value)
            for key, value in raw.get(
                "guardrails",
                {},
            ).items()
        },
        notes=tuple(
            raw.get(
                "notes",
                (),
            )
        ),
    )

    experiment.validate()
    return experiment
