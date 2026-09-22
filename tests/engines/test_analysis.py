import json
from pathlib import Path

import pytest

from llmforge.engines.analysis import (
    aggregate_runs,
    compare_aggregates,
    cross_engine_generality,
    load_m3_engine_run,
)


def write_result(path: Path, scale: float) -> None:
    path.write_text(
        json.dumps(
            {
                "summary": {
                    "request_throughput_rps": 10.0 * scale,
                    "output_throughput_tok_s": 100.0 * scale,
                    "ttft_ms": {"p50": 20.0 / scale, "p99": 40.0 / scale},
                    "tpot_ms": {"p50": 10.0 / scale, "p99": 20.0 / scale},
                    "e2e_ms": {"p50": 100.0 / scale, "p99": 200.0 / scale},
                }
            }
        ),
        encoding="utf-8",
    )


def test_cross_engine_aggregate(tmp_path: Path) -> None:
    left = []
    right = []

    for i in range(3):
        a = tmp_path / f"a{i}.json"
        b = tmp_path / f"b{i}.json"
        write_result(a, 1.0)
        write_result(b, 2.0)
        left.append(load_m3_engine_run(path=a, engine="vllm"))
        right.append(load_m3_engine_run(path=b, engine="sglang"))

    comparison = compare_aggregates(
        aggregate_runs(left),
        aggregate_runs(right),
    )

    assert comparison.ratios_right_over_left[
        "output_throughput_tok_s"
    ] == pytest.approx(2.0)


def test_generality_requires_same_direction() -> None:
    result = cross_engine_generality(
        left_baseline=20.0,
        left_stressed=30.0,
        right_baseline=25.0,
        right_stressed=40.0,
        metric_direction="lower_is_better",
    )

    assert result["same_direction"]
    assert result["claim"] == "cross_engine_direction_consistent"
