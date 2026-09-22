import json
from pathlib import Path

import pytest

from llmforge.optimization.analysis import (
    analyze_experiment,
)


def write_result(
    path: Path,
    *,
    throughput: float,
    ttft_p99: float,
    tpot_p99: float,
) -> None:
    path.write_text(
        json.dumps(
            {
                "summary": {
                    "request_throughput_rps": 10.0,
                    "output_throughput_tok_s": throughput,
                    "total_throughput_tok_s": throughput + 5.0,
                    "ttft_ms": {
                        "p50": 20.0,
                        "p95": 40.0,
                        "p99": ttft_p99,
                    },
                    "tpot_ms": {
                        "p50": 10.0,
                        "p95": 20.0,
                        "p99": tpot_p99,
                    },
                    "e2e_ms": {
                        "p50": 200.0,
                        "p95": 400.0,
                        "p99": 500.0,
                    },
                    "error_rate": 0.0,
                }
            }
        ),
        encoding="utf-8",
    )


def test_analysis_reports_tradeoff(
    tmp_path: Path,
) -> None:
    baseline = []
    candidate = []

    for index in range(3):
        left = tmp_path / f"b{index}.json"
        right = tmp_path / f"c{index}.json"

        write_result(
            left,
            throughput=100.0,
            ttft_p99=100.0,
            tpot_p99=40.0,
        )

        write_result(
            right,
            throughput=97.0,
            ttft_p99=105.0,
            tpot_p99=30.0,
        )

        baseline.append(left)

        candidate.append(right)

    result = analyze_experiment(
        baseline_paths=baseline,
        candidate_paths=candidate,
        objective_metric="tpot_p99_ms",
        objective_direction="minimize",
        guardrail_limits={
            "output_throughput_tok_s": 0.05,
            "ttft_p99_ms": 0.10,
            "error_rate": 0.0,
        },
        correctness_passed=True,
        selection_gate_passed=True,
    )

    assert result.claim_ready

    metric = next(item for item in result.metrics if (item.metric == "tpot_p99_ms"))

    assert metric.improvement_fraction == pytest.approx(0.25)

    throughput_gate = next(
        item for item in result.guardrails if (item.metric == "output_throughput_tok_s")
    )

    assert throughput_gate.passed


def test_analysis_blocks_claim_when_correctness_fails(
    tmp_path: Path,
) -> None:
    paths = []

    for index in range(3):
        path = tmp_path / f"r{index}.json"

        write_result(
            path,
            throughput=100.0,
            ttft_p99=100.0,
            tpot_p99=40.0,
        )

        paths.append(path)

    result = analyze_experiment(
        baseline_paths=paths,
        candidate_paths=paths,
        objective_metric="tpot_p99_ms",
        objective_direction="minimize",
        guardrail_limits={},
        correctness_passed=False,
        selection_gate_passed=True,
    )

    assert not result.claim_ready


def test_zero_error_rate_guardrail_fails_on_new_errors(
    tmp_path: Path,
) -> None:
    baseline = []
    candidate = []

    for index in range(3):
        left = tmp_path / f"zero_b{index}.json"
        right = tmp_path / f"zero_c{index}.json"

        write_result(
            left,
            throughput=100.0,
            ttft_p99=100.0,
            tpot_p99=40.0,
        )

        payload = json.loads(left.read_text(encoding="utf-8"))
        payload["summary"]["error_rate"] = 0.0
        left.write_text(
            json.dumps(payload),
            encoding="utf-8",
        )

        write_result(
            right,
            throughput=100.0,
            ttft_p99=100.0,
            tpot_p99=35.0,
        )

        payload = json.loads(right.read_text(encoding="utf-8"))
        payload["summary"]["error_rate"] = 0.01
        right.write_text(
            json.dumps(payload),
            encoding="utf-8",
        )

        baseline.append(left)
        candidate.append(right)

    result = analyze_experiment(
        baseline_paths=baseline,
        candidate_paths=candidate,
        objective_metric="tpot_p99_ms",
        objective_direction="minimize",
        guardrail_limits={
            "error_rate": 0.0,
        },
        correctness_passed=True,
        selection_gate_passed=True,
    )

    assert not result.claim_ready

    error_gate = next(item for item in result.guardrails if item.metric == "error_rate")

    assert not error_gate.passed
