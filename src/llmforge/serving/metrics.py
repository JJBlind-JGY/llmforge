from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from math import ceil
from statistics import mean

from llmforge.serving.schema import RequestResult


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if not 0 <= q <= 100:
        raise ValueError("q must be in [0, 100]")
    ordered = sorted(values)
    if q == 0:
        return ordered[0]
    idx = min(ceil(q / 100 * len(ordered)) - 1, len(ordered) - 1)
    return ordered[idx]


@dataclass(frozen=True)
class DistributionSummary:
    count: int
    mean: float | None
    p50: float | None
    p95: float | None
    p99: float | None


@dataclass(frozen=True)
class BenchmarkSummary:
    requests_total: int
    requests_succeeded: int
    requests_failed: int
    wall_time_s: float
    request_throughput_rps: float
    output_throughput_tok_s: float
    total_throughput_tok_s: float
    ttft_ms: DistributionSummary
    tpot_ms: DistributionSummary
    itl_ms: DistributionSummary
    e2e_ms: DistributionSummary
    client_queue_ms: DistributionSummary
    goodput_rps: float | None = None
    good_requests: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def summarize_distribution(values: Iterable[float]) -> DistributionSummary:
    xs = list(values)
    return DistributionSummary(
        count=len(xs),
        mean=mean(xs) if xs else None,
        p50=percentile(xs, 50),
        p95=percentile(xs, 95),
        p99=percentile(xs, 99),
    )


def summarize_results(
    results: list[RequestResult],
    *,
    wall_time_s: float,
    ttft_slo_ms: float | None = None,
    tpot_slo_ms: float | None = None,
    e2e_slo_ms: float | None = None,
) -> BenchmarkSummary:
    if wall_time_s <= 0:
        raise ValueError("wall_time_s must be positive")

    ok = [r for r in results if r.success]
    out_tokens = sum(r.output_tokens for r in ok)
    in_tokens = sum(r.prompt_tokens for r in ok)
    itls = [x for r in ok for x in r.itl_ms]

    use_slo = any(x is not None for x in (ttft_slo_ms, tpot_slo_ms, e2e_slo_ms))
    good = None
    goodput = None
    if use_slo:

        def meets(r: RequestResult) -> bool:
            if not r.success:
                return False
            if ttft_slo_ms is not None and (
                r.ttft_ms is None or r.ttft_ms > ttft_slo_ms
            ):
                return False
            if tpot_slo_ms is not None and (
                r.tpot_ms is None or r.tpot_ms > tpot_slo_ms
            ):
                return False
            if e2e_slo_ms is not None and r.e2e_ms > e2e_slo_ms:
                return False
            return True

        good = sum(meets(r) for r in results)
        goodput = good / wall_time_s

    return BenchmarkSummary(
        requests_total=len(results),
        requests_succeeded=len(ok),
        requests_failed=len(results) - len(ok),
        wall_time_s=wall_time_s,
        request_throughput_rps=len(ok) / wall_time_s,
        output_throughput_tok_s=out_tokens / wall_time_s,
        total_throughput_tok_s=(in_tokens + out_tokens) / wall_time_s,
        ttft_ms=summarize_distribution(r.ttft_ms for r in ok if r.ttft_ms is not None),
        tpot_ms=summarize_distribution(r.tpot_ms for r in ok if r.tpot_ms is not None),
        itl_ms=summarize_distribution(itls),
        e2e_ms=summarize_distribution(r.e2e_ms for r in ok),
        client_queue_ms=summarize_distribution(r.client_queue_ms for r in results),
        goodput_rps=goodput,
        good_requests=good,
    )
