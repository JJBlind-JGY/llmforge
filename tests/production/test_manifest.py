from pathlib import Path

from llmforge.production.manifest import (
    build_run_manifest,
    stable_json_hash,
)


def test_config_hash_is_order_independent() -> None:
    assert stable_json_hash(
        {
            "a": 1,
            "b": 2,
        }
    ) == stable_json_hash(
        {
            "b": 2,
            "a": 1,
        }
    )


def test_manifest_records_selected_environment(
    tmp_path: Path,
) -> None:
    manifest = build_run_manifest(
        run_id="run-1",
        command=[
            "python",
            "-V",
        ],
        cwd=tmp_path,
        config={
            "x": 1,
        },
        environment={
            "CUDA_VISIBLE_DEVICES": ("0,1"),
            "HF_HOME": "/tmp/hf",
            "PASSWORD": "secret",
        },
    )

    selected = manifest.environment.selected_environment

    assert selected["CUDA_VISIBLE_DEVICES"] == "0,1"

    assert "PASSWORD" not in selected
    assert manifest.config_sha256 is not None
