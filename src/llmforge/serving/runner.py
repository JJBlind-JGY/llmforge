from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from dataclasses import dataclass

from .client import VLLMOpenAIClient
from .metrics import BenchmarkSummary, summarize_results
from .schema import RequestResult, RequestSpec


@dataclass(frozen=True)
class RunResult:
    results: list[RequestResult]
    summary: BenchmarkSummary
    wall_time_s: float


async def _sleep_until(target: float) -> None:
    delay = target - time.perf_counter()
    if delay > 0:
        await asyncio.sleep(delay)


async def run_workload_async(
    *,
    client: VLLMOpenAIClient,
    requests: Sequence[RequestSpec],
    max_in_flight: int,
    ttft_slo_ms: float | None = None,
    tpot_slo_ms: float | None = None,
    e2e_slo_ms: float | None = None,
) -> RunResult:
    if not requests or max_in_flight <= 0:
        raise ValueError("non-empty requests and positive max_in_flight required")

    sem = asyncio.Semaphore(max_in_flight)
    start = time.perf_counter()

    async with client:
        await client.health()

        async def execute(req: RequestSpec) -> RequestResult:
            due = start + req.arrival_offset_s
            await _sleep_until(due)
            async with sem:
                return await client.run_request(req, scheduled_due_perf=due)

        results = list(
            await asyncio.gather(
                *(asyncio.create_task(execute(req)) for req in requests)
            )
        )

    wall = time.perf_counter() - start
    summary = summarize_results(
        results,
        wall_time_s=wall,
        ttft_slo_ms=ttft_slo_ms,
        tpot_slo_ms=tpot_slo_ms,
        e2e_slo_ms=e2e_slo_ms,
    )
    return RunResult(results=results, summary=summary, wall_time_s=wall)


def run_workload(**kwargs) -> RunResult:
    return asyncio.run(run_workload_async(**kwargs))
