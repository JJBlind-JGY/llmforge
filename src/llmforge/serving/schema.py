from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass(frozen=True)
class RequestSpec:
    request_id: str
    prompt_token_ids: tuple[int, ...]
    output_tokens: int
    arrival_offset_s: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)


@dataclass
class RequestResult:
    request_id: str
    prompt_tokens: int
    requested_output_tokens: int
    output_tokens: int
    scheduled_offset_s: float
    client_queue_ms: float
    ttft_ms: float | None
    e2e_ms: float
    tpot_ms: float | None
    itl_ms: list[float]
    success: bool
    error: str | None = None
    status_code: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
