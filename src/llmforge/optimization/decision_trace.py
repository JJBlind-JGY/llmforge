"""Bounded machine-readable trace for M7 policy decisions."""

from __future__ import annotations

import json
import queue
import threading
from pathlib import Path

_STOP = object()


class BufferedDecisionRecorder:
    def __init__(
        self,
        path: Path,
        *,
        queue_size: int = 8192,
        flush_every: int = 128,
    ) -> None:
        self.path = path
        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._queue: queue.Queue[dict | object] = queue.Queue(maxsize=queue_size)

        self._flush_every = flush_every

        self._closed = False
        self._submitted = 0
        self._written = 0
        self._dropped = 0
        self._error: BaseException | None = None

        self._lock = threading.Lock()

        self._thread = threading.Thread(
            target=self._writer,
            name=("llmforge-m7-decision-trace"),
            daemon=True,
        )

        self._thread.start()

    def emit(
        self,
        payload: dict,
    ) -> bool:
        if self._closed:
            raise RuntimeError("decision recorder is closed.")

        self._raise_error()

        with self._lock:
            self._submitted += 1

        try:
            self._queue.put_nowait(dict(payload))
            return True
        except queue.Full:
            with self._lock:
                self._dropped += 1
            return False

    def close(self) -> None:
        if self._closed:
            return

        self._queue.join()
        self._queue.put(_STOP)
        self._thread.join()
        self._closed = True
        self._raise_error()

    def stats(self) -> dict:
        with self._lock:
            return {
                "submitted": (self._submitted),
                "written": (self._written),
                "dropped": (self._dropped),
            }

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
                            dict,
                        )

                        handle.write(
                            json.dumps(
                                item,
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

        except BaseException as exc:  # noqa: BLE001 — worker thread top-level guard
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
            raise RuntimeError("decision trace writer failed.") from self._error
