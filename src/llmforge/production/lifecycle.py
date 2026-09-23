"""Timeout, cancellation, and graceful-shutdown primitives."""

from __future__ import annotations

import asyncio
import signal
import threading
from collections.abc import Awaitable


class CancellationToken:
    def __init__(self) -> None:
        self._event = asyncio.Event()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def cancel(self) -> None:
        self._event.set()

    async def wait(self) -> None:
        await self._event.wait()

    def raise_if_cancelled(
        self,
    ) -> None:
        if self.cancelled:
            raise asyncio.CancelledError


async def run_with_timeout[T](
    awaitable: Awaitable[T],
    *,
    timeout_s: float,
) -> T:
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive.")

    async with asyncio.timeout(timeout_s):
        return await awaitable


class GracefulShutdown:
    """Process-wide shutdown signal for scripts/services."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._installed = False

    @property
    def requested(self) -> bool:
        return self._event.is_set()

    def request(self) -> None:
        self._event.set()

    def wait(
        self,
        timeout_s: float | None = None,
    ) -> bool:
        return self._event.wait(timeout=timeout_s)

    def install_signal_handlers(
        self,
    ) -> None:
        if self._installed:
            return

        def handler(
            signum,
            frame,
        ) -> None:
            del signum
            del frame
            self.request()

        signal.signal(
            signal.SIGINT,
            handler,
        )

        if hasattr(
            signal,
            "SIGTERM",
        ):
            signal.signal(
                signal.SIGTERM,
                handler,
            )

        self._installed = True
