"""Periodic collector runner with failure isolation."""

from __future__ import annotations

import logging
import threading
import time
from typing import Protocol

from llmforge.telemetry.metrics import (
    MetricRegistry,
)

_LOGGER = logging.getLogger(__name__)


class Collector(Protocol):
    name: str

    def collect(
        self,
        registry: MetricRegistry,
    ) -> None: ...


class PeriodicCollectorRunner:
    def __init__(
        self,
        *,
        registry: MetricRegistry,
        collectors: list[Collector],
        interval_s: float,
    ) -> None:
        if interval_s <= 0:
            raise ValueError("interval_s must be positive.")

        self.registry = registry
        self.collectors = list(collectors)
        self.interval_s = interval_s

        self._stop = threading.Event()

        self._thread = threading.Thread(
            target=self._run,
            name=("llmforge-observability-sampler"),
            daemon=True,
        )

        self._started = False

    def start(self) -> None:
        if self._started:
            return

        self._started = True
        self._thread.start()

    def stop(
        self,
        *,
        timeout_s: float = 5.0,
    ) -> None:
        if not self._started:
            return

        self._stop.set()
        self._thread.join(timeout=timeout_s)

    def collect_once(self) -> None:
        for collector in self.collectors:
            try:
                collector.collect(self.registry)
            except Exception:
                self.registry.inc_counter(
                    "llmforge_observer_error_total",
                    labels={
                        "collector": (collector.name),
                    },
                )

                _LOGGER.exception(
                    "observability collector failed",
                    extra={"component": (collector.name)},
                )

    def _run(self) -> None:
        next_tick = time.monotonic()

        while not self._stop.is_set():
            self.collect_once()

            next_tick += self.interval_s

            delay = next_tick - time.monotonic()

            if delay <= 0:
                next_tick = time.monotonic()
                continue

            self._stop.wait(delay)
