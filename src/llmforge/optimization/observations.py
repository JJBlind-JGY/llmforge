"""Build one M7 observation sheet from M3/M4/M5/M6 evidence."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def _nested(
    mapping: dict[str, Any],
    *keys: str,
) -> Any:
    current: Any = mapping

    for key in keys:
        if not isinstance(
            current,
            dict,
        ):
            return None

        current = current.get(key)

    return current


def _float_or_none(
    value: Any,
) -> float | None:
    if value is None:
        return None
    return float(value)


@dataclass(frozen=True)
class ServingObservation:
    artifact: str
    request_throughput_rps: float | None
    output_throughput_tok_s: float | None
    ttft_p50_ms: float | None
    ttft_p99_ms: float | None
    tpot_p50_ms: float | None
    tpot_p99_ms: float | None
    e2e_p50_ms: float | None
    e2e_p99_ms: float | None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RuntimeObservation:
    artifact: str
    scheduler_steps: int | None
    scheduler_p99_us: float | None
    queue_wait_p99_ms: float | None
    kv_usage_ratio_max: float | None
    preemptions_total: int | None
    scheduled_prompt_tokens: int | None
    scheduled_output_tokens: int | None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class DistributedObservation:
    artifact: str
    candidate_profiles: tuple[str, ...]
    best_output_speedup_observed: float | None
    worst_ttft_p99_ratio_observed: float | None

    def to_dict(self) -> dict:
        data = asdict(self)
        data["candidate_profiles"] = list(self.candidate_profiles)
        return data


@dataclass(frozen=True)
class TelemetryObservation:
    artifact: str
    request_running: float | None
    request_waiting: float | None
    kv_usage_ratio: float | None
    observer_errors: float | None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ObservationSheet:
    schema_version: int
    observation_id: str
    workload_id: str
    hypothesis: str
    serving: tuple[ServingObservation, ...]
    runtime: RuntimeObservation | None
    distributed: DistributedObservation | None
    telemetry: TelemetryObservation | None
    profiler_artifacts: tuple[str, ...]
    notes: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "schema_version": (self.schema_version),
            "observation_id": (self.observation_id),
            "workload_id": (self.workload_id),
            "hypothesis": (self.hypothesis),
            "serving": [item.to_dict() for item in self.serving],
            "runtime": (self.runtime.to_dict() if self.runtime else None),
            "distributed": (self.distributed.to_dict() if self.distributed else None),
            "telemetry": (self.telemetry.to_dict() if self.telemetry else None),
            "profiler_artifacts": list(self.profiler_artifacts),
            "notes": list(self.notes),
        }


def load_serving_observation(
    path: Path,
) -> ServingObservation:
    payload = json.loads(path.read_text(encoding="utf-8"))

    summary = payload["summary"]

    return ServingObservation(
        artifact=str(path),
        request_throughput_rps=(_float_or_none(summary.get("request_throughput_rps"))),
        output_throughput_tok_s=(
            _float_or_none(summary.get("output_throughput_tok_s"))
        ),
        ttft_p50_ms=(
            _float_or_none(
                _nested(
                    summary,
                    "ttft_ms",
                    "p50",
                )
            )
        ),
        ttft_p99_ms=(
            _float_or_none(
                _nested(
                    summary,
                    "ttft_ms",
                    "p99",
                )
            )
        ),
        tpot_p50_ms=(
            _float_or_none(
                _nested(
                    summary,
                    "tpot_ms",
                    "p50",
                )
            )
        ),
        tpot_p99_ms=(
            _float_or_none(
                _nested(
                    summary,
                    "tpot_ms",
                    "p99",
                )
            )
        ),
        e2e_p50_ms=(
            _float_or_none(
                _nested(
                    summary,
                    "e2e_ms",
                    "p50",
                )
            )
        ),
        e2e_p99_ms=(
            _float_or_none(
                _nested(
                    summary,
                    "e2e_ms",
                    "p99",
                )
            )
        ),
    )


def load_runtime_observation(
    path: Path,
) -> RuntimeObservation:
    payload = json.loads(path.read_text(encoding="utf-8"))

    return RuntimeObservation(
        artifact=str(path),
        scheduler_steps=(
            int(payload["scheduler_steps"])
            if payload.get("scheduler_steps") is not None
            else None
        ),
        scheduler_p99_us=(
            _float_or_none(
                _nested(
                    payload,
                    "scheduler_duration_us",
                    "p99",
                )
            )
        ),
        queue_wait_p99_ms=(
            _float_or_none(
                _nested(
                    payload,
                    "queue_wait_ms",
                    "p99",
                )
            )
        ),
        kv_usage_ratio_max=(_float_or_none(payload.get("kv_usage_ratio_max"))),
        preemptions_total=(
            int(payload["preemptions_total"])
            if payload.get("preemptions_total") is not None
            else None
        ),
        scheduled_prompt_tokens=(
            int(payload["scheduled_prompt_tokens"])
            if payload.get("scheduled_prompt_tokens") is not None
            else None
        ),
        scheduled_output_tokens=(
            int(payload["scheduled_output_tokens"])
            if payload.get("scheduled_output_tokens") is not None
            else None
        ),
    )


def load_distributed_observation(
    path: Path,
) -> DistributedObservation:
    payload = json.loads(path.read_text(encoding="utf-8"))

    candidates = payload.get(
        "candidates",
        [],
    )

    profiles = tuple(str(item["serving"]["profile"]) for item in candidates)

    speedups = [
        float(item["scaling"]["output_throughput_speedup"])
        for item in candidates
        if (
            item.get(
                "scaling",
                {},
            ).get("output_throughput_speedup")
            is not None
        )
    ]

    ttft_ratios = [
        float(item["scaling"]["ttft_p99_ratio"])
        for item in candidates
        if (
            item.get(
                "scaling",
                {},
            ).get("ttft_p99_ratio")
            is not None
        )
    ]

    return DistributedObservation(
        artifact=str(path),
        candidate_profiles=profiles,
        best_output_speedup_observed=(max(speedups) if speedups else None),
        worst_ttft_p99_ratio_observed=(max(ttft_ratios) if ttft_ratios else None),
    )


def _metric_value(
    snapshot: dict,
    name: str,
) -> float | None:
    metrics = snapshot.get(
        "metrics",
        snapshot,
    )

    gauges = metrics.get(
        "gauges",
        [],
    )

    counters = metrics.get(
        "counters",
        [],
    )

    values = [
        float(item["value"])
        for item in [
            *gauges,
            *counters,
        ]
        if item.get("name") == name
    ]

    if not values:
        return None

    return max(values)


def load_telemetry_observation(
    path: Path,
) -> TelemetryObservation:
    payload = json.loads(path.read_text(encoding="utf-8"))

    return TelemetryObservation(
        artifact=str(path),
        request_running=_metric_value(
            payload,
            "llmforge_request_running",
        ),
        request_waiting=_metric_value(
            payload,
            "llmforge_request_waiting",
        ),
        kv_usage_ratio=_metric_value(
            payload,
            "llmforge_kv_usage_ratio",
        ),
        observer_errors=_metric_value(
            payload,
            "llmforge_observer_error_total",
        ),
    )


def build_observation_sheet(
    *,
    observation_id: str,
    workload_id: str,
    hypothesis: str,
    serving_paths: list[Path],
    runtime_path: Path | None = None,
    distributed_path: Path | None = None,
    telemetry_path: Path | None = None,
    profiler_artifacts: tuple[str, ...] = (),
    notes: tuple[str, ...] = (),
) -> ObservationSheet:
    if not observation_id:
        raise ValueError("observation_id must not be empty.")

    if not workload_id:
        raise ValueError("workload_id must not be empty.")

    if not serving_paths:
        raise ValueError("At least one M3 serving artifact is required.")

    return ObservationSheet(
        schema_version=1,
        observation_id=observation_id,
        workload_id=workload_id,
        hypothesis=hypothesis,
        serving=tuple(load_serving_observation(path) for path in serving_paths),
        runtime=(load_runtime_observation(runtime_path) if runtime_path else None),
        distributed=(
            load_distributed_observation(distributed_path) if distributed_path else None
        ),
        telemetry=(
            load_telemetry_observation(telemetry_path) if telemetry_path else None
        ),
        profiler_artifacts=(profiler_artifacts),
        notes=notes,
    )
