import json
from pathlib import Path

import pytest

from llmforge.distributed.inference.analysis import (
    compare_to_baseline,
    load_m3_serving_run,
)


def write_result(
    path: Path,
    *,
    request_throughput: float,
    output_throughput: float,
    ttft_p50: float,
    tpot_p50: float,
) -> None:
    payload = {
        "summary": {
            "request_throughput_rps": request_throughput,
            "output_throughput_tok_s": output_throughput,
            "total_throughput_tok_s": output_throughput + 10.0,
            "ttft_ms": {
                "p50": ttft_p50,
                "p95": ttft_p50 * 1.5,
                "p99": ttft_p50 * 2.0,
            },
            "tpot_ms": {
                "p50": tpot_p50,
                "p95": tpot_p50 * 1.5,
                "p99": tpot_p50 * 2.0,
            },
            "e2e_ms": {
                "p50": 100.0,
                "p95": 150.0,
                "p99": 200.0,
            },
        }
    }

    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def test_load_and_compare_m3_results(
    tmp_path: Path,
) -> None:
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"

    write_result(
        baseline_path,
        request_throughput=10.0,
        output_throughput=100.0,
        ttft_p50=20.0,
        tpot_p50=10.0,
    )

    write_result(
        candidate_path,
        request_throughput=16.0,
        output_throughput=160.0,
        ttft_p50=25.0,
        tpot_p50=8.0,
    )

    baseline = load_m3_serving_run(
        path=baseline_path,
        profile="single",
        devices=1,
    )

    candidate = load_m3_serving_run(
        path=candidate_path,
        profile="tp2",
        devices=2,
    )

    comparison = compare_to_baseline(
        baseline=baseline,
        candidate=candidate,
    )

    assert comparison.output_throughput_speedup == pytest.approx(1.6)

    assert comparison.output_throughput_efficiency == pytest.approx(0.8)

    assert comparison.ttft_p50_ratio == pytest.approx(1.25)

    assert comparison.tpot_p50_ratio == pytest.approx(0.8)
