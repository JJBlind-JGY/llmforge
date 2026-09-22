import json
from pathlib import Path

import pytest

from llmforge.engines.workload_contract import load_cross_engine_workload


def test_cross_engine_workload_requires_repeats(tmp_path: Path) -> None:
    path = tmp_path / "workload.json"
    path.write_text(
        json.dumps(
            {
                "workload_id": "x",
                "source_workload": "m3.json",
                "prompt_length_mode": "fixed",
                "output_length_mode": "fixed",
                "request_rate": "inf",
                "max_concurrency": 4,
                "repetitions": 2,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_cross_engine_workload(path)
