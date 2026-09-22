"""Cross-engine workload fairness contract."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class CrossEngineWorkload:
    workload_id: str
    source_workload: str
    client: str
    endpoint: str
    streaming: bool
    prompt_length_mode: str
    output_length_mode: str
    request_rate: str
    max_concurrency: int
    repetitions: int
    warmup_requests: int
    prefix_reuse: bool
    notes: tuple[str, ...]

    def validate(self) -> None:
        if self.endpoint not in {
            "/v1/chat/completions",
            "/v1/completions",
        }:
            raise ValueError("Cross-engine M8 requires an OpenAI-compatible endpoint.")

        if self.max_concurrency <= 0:
            raise ValueError("max_concurrency must be positive.")

        if self.repetitions < 3:
            raise ValueError("At least three repetitions are required.")

        if self.warmup_requests < 0:
            raise ValueError("warmup_requests must be non-negative.")

    def to_dict(self) -> dict:
        return asdict(self)


def load_cross_engine_workload(
    path: Path,
) -> CrossEngineWorkload:
    raw = json.loads(path.read_text(encoding="utf-8"))

    workload = CrossEngineWorkload(
        workload_id=str(raw["workload_id"]),
        source_workload=str(raw["source_workload"]),
        client=str(raw.get("client", "llmforge_m3")),
        endpoint=str(
            raw.get(
                "endpoint",
                "/v1/chat/completions",
            )
        ),
        streaming=bool(raw.get("streaming", True)),
        prompt_length_mode=str(raw["prompt_length_mode"]),
        output_length_mode=str(raw["output_length_mode"]),
        request_rate=str(raw["request_rate"]),
        max_concurrency=int(raw["max_concurrency"]),
        repetitions=int(raw.get("repetitions", 5)),
        warmup_requests=int(raw.get("warmup_requests", 8)),
        prefix_reuse=bool(raw.get("prefix_reuse", False)),
        notes=tuple(raw.get("notes", ())),
    )

    workload.validate()
    return workload
