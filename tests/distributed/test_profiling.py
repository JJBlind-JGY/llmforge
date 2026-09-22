import json
from pathlib import Path

from llmforge.distributed.profiling import (
    summarize_communication_trace,
)


def test_profile_extracts_collectives(
    tmp_path: Path,
) -> None:
    trace = tmp_path / "trace.json"

    trace.write_text(
        json.dumps(
            {
                "traceEvents": [
                    {
                        "name": "ncclAllReduceKernel",
                        "ph": "X",
                        "dur": 10.0,
                    },
                    {
                        "name": "aten::matmul",
                        "ph": "X",
                        "dur": 20.0,
                    },
                    {
                        "name": "all_gather_into_tensor",
                        "ph": "X",
                        "dur": 5.0,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    summary = summarize_communication_trace(trace)

    assert summary.matched_events == 2
    assert summary.total_duration_us == 15.0
    assert summary.by_name_us["ncclAllReduceKernel"] == 10.0
