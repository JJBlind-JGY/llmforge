import json
from pathlib import Path

from llmforge.production.config import (
    load_production_config,
)


def test_config_precedence(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.json"

    path.write_text(
        json.dumps(
            {
                "service": {
                    "port": 9000,
                },
                "observability": {
                    "gpu_indices": [1, 2],
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_production_config(
        path,
        environment={
            "LLMFORGE_CONTROL_PORT": "9200",
            "LLMFORGE_GPU_INDICES": "2,3",
        },
    )

    assert config.service.port == 9200
    assert config.observability.gpu_indices == (2, 3)


def test_default_config_is_valid() -> None:
    config = load_production_config(environment={})

    assert config.schema_version == 1
    assert config.service.port == 9108
    assert config.observability.tracing.mode == "jsonl"
