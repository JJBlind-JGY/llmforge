import json
from pathlib import Path

import pytest

from llmforge.optimization.experiment import (
    load_experiment,
)


def write_experiment(
    path: Path,
    *,
    repetitions: int = 5,
) -> None:
    payload = {
        "experiment_id": "exp",
        "candidate_id": "candidate",
        "workload_id": "workload",
        "workload_config": "workload.json",
        "repetitions": repetitions,
        "warmup_requests": 4,
        "objective_metric": "tpot_p99_ms",
        "objective_direction": "minimize",
        "guardrails": {
            "output_throughput_tok_s": 0.05,
        },
        "baseline": {
            "name": "baseline",
            "server_config": {
                "model": "model",
                "revision": "revision",
                "dtype": "bfloat16",
            },
            "environment": {},
            "scheduler_class": None,
        },
        "candidate": {
            "name": "candidate",
            "server_config": {
                "model": "model",
                "revision": "revision",
                "dtype": "bfloat16",
            },
            "environment": {
                "X": "1",
            },
            "scheduler_class": "pkg.Scheduler",
        },
    }

    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def test_experiment_freezes_distinct_arms(
    tmp_path: Path,
) -> None:
    path = tmp_path / "exp.json"
    write_experiment(path)

    experiment = load_experiment(path)

    assert experiment.baseline.fingerprint != experiment.candidate.fingerprint

    assert experiment.repetitions == 5


def test_experiment_requires_repeats(
    tmp_path: Path,
) -> None:
    path = tmp_path / "exp.json"
    write_experiment(
        path,
        repetitions=2,
    )

    with pytest.raises(ValueError):
        load_experiment(path)
