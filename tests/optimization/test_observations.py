import json
from pathlib import Path

from llmforge.optimization.observations import (
    build_observation_sheet,
)


def test_build_observation_sheet(
    tmp_path: Path,
) -> None:
    serving = tmp_path / "serving.json"

    serving.write_text(
        json.dumps(
            {
                "summary": {
                    "request_throughput_rps": 10.0,
                    "output_throughput_tok_s": 100.0,
                    "ttft_ms": {
                        "p50": 20.0,
                        "p99": 100.0,
                    },
                    "tpot_ms": {
                        "p50": 10.0,
                        "p99": 30.0,
                    },
                    "e2e_ms": {
                        "p50": 200.0,
                        "p99": 500.0,
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    runtime = tmp_path / "runtime.json"

    runtime.write_text(
        json.dumps(
            {
                "scheduler_steps": 10,
                "scheduler_duration_us": {
                    "p99": 150.0,
                },
                "queue_wait_ms": {
                    "p99": 80.0,
                },
                "kv_usage_ratio_max": 0.8,
                "preemptions_total": 2,
                "scheduled_prompt_tokens": 1000,
                "scheduled_output_tokens": 100,
            }
        ),
        encoding="utf-8",
    )

    sheet = build_observation_sheet(
        observation_id="obs",
        workload_id="workload",
        hypothesis="hypothesis",
        serving_paths=[serving],
        runtime_path=runtime,
    )

    assert sheet.serving[0].ttft_p99_ms == 100.0

    assert sheet.runtime is not None

    assert sheet.runtime.preemptions_total == 2
