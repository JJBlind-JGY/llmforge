from __future__ import annotations

import json
from pathlib import Path
from typing import Self

from llmforge.runtime.events import RuntimeEvent

from .base import TraceRecorderStats


class SyncJsonlTraceRecorder:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8", buffering=1)
        self._submitted = 0
        self._written = 0
        self._closed = False

    def emit(self, event: RuntimeEvent) -> bool:
        if self._closed:
            raise RuntimeError("trace recorder is closed.")
        self._submitted += 1
        self._handle.write(
            json.dumps(event.to_dict(), sort_keys=True, separators=(",", ":")) + "\n"
        )
        self._written += 1
        return True

    def flush(self) -> None:
        if not self._closed:
            self._handle.flush()

    def close(self) -> None:
        if self._closed:
            return
        self._handle.flush()
        self._handle.close()
        self._closed = True

    def stats(self) -> TraceRecorderStats:
        return TraceRecorderStats(self._submitted, self._written, 0)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
