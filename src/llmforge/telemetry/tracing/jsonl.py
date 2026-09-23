"""Bounded background JSONL span exporter."""

from __future__ import annotations

import json
import queue
import threading
from dataclasses import dataclass
from pathlib import Path

from .span import SpanRecord

_STOP = object()


@dataclass(frozen=True)
class SpanExporterStats:
    submitted: int
    written: int
    dropped: int


class BufferedJsonlSpanExporter:
    def __init__(
        self,
        path: Path,
        *,
        queue_size: int = 8192,
        flush_every: int = 128,
    ) -> None:
        if queue_size <= 0:
            raise ValueError("queue_size must be positive.")

        if flush_every <= 0:
            raise ValueError("flush_every must be positive.")

        self.path = path
        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._queue: queue.Queue[SpanRecord | object] = queue.Queue(maxsize=queue_size)

        self._flush_every = flush_every

        self._lock = threading.Lock()

        self._submitted = 0
        self._written = 0
        self._dropped = 0
        self._closed = False
        self._error: BaseException | None = None

        self._thread = threading.Thread(
            target=self._writer,
            name=("llmforge-span-exporter"),
            daemon=True,
        )

        self._thread.start()

    def export(
        self,
        span: SpanRecord,
    ) -> bool:
        if self._closed:
            raise RuntimeError("span exporter is closed.")

        self._raise_error()

        with self._lock:
            self._submitted += 1

        try:
            self._queue.put_nowait(span)
            return True
        except queue.Full:
            with self._lock:
                self._dropped += 1
            return False

    def flush(self) -> None:
        if self._closed:
            return

        self._queue.join()
        self._raise_error()

    def close(self) -> None:
        if self._closed:
            return

        self._queue.join()
        self._queue.put(_STOP)
        self._thread.join()
        self._closed = True
        self._raise_error()

    def stats(
        self,
    ) -> SpanExporterStats:
        with self._lock:
            return SpanExporterStats(
                submitted=(self._submitted),
                written=(self._written),
                dropped=(self._dropped),
            )

    def _writer(self) -> None:
        try:
            with self.path.open(
                "a",
                encoding="utf-8",
                buffering=1024 * 1024,
            ) as handle:
                pending = 0

                while True:
                    item = self._queue.get()

                    try:
                        if item is _STOP:
                            handle.flush()
                            return

                        assert isinstance(
                            item,
                            SpanRecord,
                        )

                        handle.write(
                            json.dumps(
                                item.to_dict(),
                                sort_keys=True,
                                separators=(",", ":"),
                            )
                            + "\n"
                        )

                        with self._lock:
                            self._written += 1

                        pending += 1

                        if pending >= self._flush_every:
                            handle.flush()
                            pending = 0
                    finally:
                        self._queue.task_done()

        except BaseException as exc:
            self._error = exc

            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
                else:
                    self._queue.task_done()

    def _raise_error(self) -> None:
        if self._error is not None:
            raise RuntimeError("span exporter failed.") from self._error

    def __enter__(
        self,
    ) -> BufferedJsonlSpanExporter:
        return self

    def __exit__(
        self,
        exc_type,
        exc,
        traceback,
    ) -> None:
        self.close()
