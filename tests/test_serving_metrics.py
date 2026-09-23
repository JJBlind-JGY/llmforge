import pytest

from llmforge.serving.metrics import percentile, summarize_results
from llmforge.serving.schema import RequestResult


def result(i, ttft, tpot, e2e):
    return RequestResult(
        request_id=str(i),
        prompt_tokens=100,
        requested_output_tokens=3,
        output_tokens=3,
        scheduled_offset_s=0,
        client_queue_ms=0,
        ttft_ms=ttft,
        e2e_ms=e2e,
        tpot_ms=tpot,
        itl_ms=[tpot, tpot],
        success=True,
    )


def test_percentile():
    xs = list(range(1, 101))
    assert percentile(xs, 50) == 50
    assert percentile(xs, 95) == 95
    assert percentile(xs, 99) == 99


def test_summary_and_goodput():
    s = summarize_results(
        [result(1, 10, 5, 20), result(2, 20, 10, 40)],
        wall_time_s=1.0,
        ttft_slo_ms=25,
        tpot_slo_ms=11,
    )
    assert s.requests_succeeded == 2
    assert s.output_throughput_tok_s == pytest.approx(6.0)
    assert s.good_requests == 2
