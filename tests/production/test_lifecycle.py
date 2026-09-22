import asyncio

import pytest

from llmforge.production.lifecycle import (
    CancellationToken,
    run_with_timeout,
)


def test_cancellation_token() -> None:
    async def scenario() -> None:
        token = CancellationToken()

        assert not token.cancelled

        token.cancel()

        assert token.cancelled

        with pytest.raises(asyncio.CancelledError):
            token.raise_if_cancelled()

    asyncio.run(scenario())


def test_run_with_timeout() -> None:
    async def slow() -> None:
        await asyncio.sleep(0.05)

    async def scenario() -> None:
        with pytest.raises(TimeoutError):
            await run_with_timeout(
                slow(),
                timeout_s=0.001,
            )

    asyncio.run(scenario())
