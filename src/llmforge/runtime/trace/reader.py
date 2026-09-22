from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from llmforge.runtime.events import RuntimeEvent


class JsonlTraceReader:
    def __init__(self, path: Path) -> None:
        self.path = path

    def __iter__(self) -> Iterator[RuntimeEvent]:
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid runtime trace JSON at line {line_number}."
                    ) from exc
                yield RuntimeEvent.from_dict(payload)

    def read_all(self) -> list[RuntimeEvent]:
        return list(self)
