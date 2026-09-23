from __future__ import annotations

import json
import queue
import threading
from pathlib import Path

from llmforge.runtime.events import RuntimeEvent

from .base import TraceRecorderStats

_STOP = object()


class BufferedJsonlTraceRecorder:
    """Background JSONL recorder that avoids file I/O on the scheduler thread."""

    def __init__(
        self,
        path: Path,
        *,
        queue_size: int = 65_536,
        flush_every: int = 256,
        drop_on_full: bool = True,
    ) -> None:
        if queue_size <= 0:
            raise ValueError("queue_size must be positive.")
        if flush_every <= 0:
            raise ValueError("flush_every must be positive.")

        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._queue: queue.Queue[RuntimeEvent | object] = queue.Queue(
            maxsize=queue_size
        )
        self._flush_every = flush_every
        self._drop_on_full = drop_on_full
        self._lock = threading.Lock()
        self._submitted = 0
        self._written = 0
        self._dropped = 0
        self._closed = False
        self._worker_error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._writer_main,
            name="llmforge-runtime-trace",
            daemon=True,
        )
        self._thread.start()

    def emit(self, event: RuntimeEvent) -> bool:
        if self._closed:
            raise RuntimeError("trace recorder is closed.")
        self._raise_worker_error()

        with self._lock:
            self._submitted += 1

        try:
            if self._drop_on_full:
                self._queue.put_nowait(event)
            else:
                self._queue.put(event)
            return True
        except queue.Full:
            with self._lock:
                self._dropped += 1
            return False

    def flush(self) -> None:
        if self._closed:
            return
        self._queue.join()
        self._raise_worker_error()

    def close(self) -> None:
        if self._closed:
            return
        self._queue.join()
        self._queue.put(_STOP)
        self._thread.join()
        self._closed = True
        self._raise_worker_error()

    def stats(self) -> TraceRecorderStats:
        with self._lock:
            return TraceRecorderStats(self._submitted, self._written, self._dropped)

    def _writer_main(self) -> None:
        try:
            with self.path.open("a", encoding="utf-8", buffering=1024 * 1024) as handle:
                pending = 0
                while True:
                    item = self._queue.get()
                    try:
                        if item is _STOP:
                            handle.flush()
                            return
                        assert isinstance(item, RuntimeEvent)
                        handle.write(
                            json.dumps(
                                item.to_dict(),
                                sort_keys=True,
                                separators=(",", ":"),
                            )
                            + "\n"
                        )
                        pending += 1
                        with self._lock:
                            self._written += 1
                        if pending >= self._flush_every:
                            handle.flush()
                            pending = 0
                    finally:
                        self._queue.task_done()
        except BaseException as exc:
            self._worker_error = exc
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
                else:
                    self._queue.task_done()

    def _raise_worker_error(self) -> None:
        if self._worker_error is not None:
            raise RuntimeError("runtime trace writer failed.") from self._worker_error

    def __enter__(self) -> BufferedJsonlTraceRecorder:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
