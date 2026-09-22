from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

TraceMode = Literal["off", "sync", "buffered"]


@dataclass(frozen=True)
class VLLMTraceConfig:
    mode: TraceMode
    path: Path
    queue_size: int
    flush_every: int
    drop_on_full: bool

    @classmethod
    def from_environment(cls) -> "VLLMTraceConfig":
        mode = os.environ.get(
            "LLMFORGE_RUNTIME_TRACE_MODE",
            "buffered",
        ).lower()

        if mode not in {"off", "sync", "buffered"}:
            raise ValueError(
                "LLMFORGE_RUNTIME_TRACE_MODE must be off, sync, or buffered."
            )

        raw_path = os.environ.get(
            "LLMFORGE_RUNTIME_TRACE_PATH",
            "artifacts/runtime/vllm_runtime_{pid}.jsonl",
        )

        return cls(
            mode=mode,
            path=Path(raw_path.format(pid=os.getpid())),
            queue_size=int(
                os.environ.get(
                    "LLMFORGE_RUNTIME_TRACE_QUEUE_SIZE",
                    "65536",
                )
            ),
            flush_every=int(
                os.environ.get(
                    "LLMFORGE_RUNTIME_TRACE_FLUSH_EVERY",
                    "256",
                )
            ),
            drop_on_full=(
                os.environ.get(
                    "LLMFORGE_RUNTIME_TRACE_DROP_ON_FULL",
                    "1",
                )
                != "0"
            ),
        )
