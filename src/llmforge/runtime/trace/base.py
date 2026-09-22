from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from llmforge.runtime.events import RuntimeEvent


@dataclass(frozen=True)
class TraceRecorderStats:
    submitted_events: int
    written_events: int
    dropped_events: int


class TraceRecorder(Protocol):
    def emit(self, event: RuntimeEvent) -> bool: ...
    def flush(self) -> None: ...
    def close(self) -> None: ...
    def stats(self) -> TraceRecorderStats: ...
