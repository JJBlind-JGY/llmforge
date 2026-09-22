import pytest

from llmforge.engines import create_engine_adapter, supported_engines


def test_supported_engines() -> None:
    assert supported_engines() == ("vllm", "sglang")


def test_registry_rejects_unknown_engine() -> None:
    with pytest.raises(ValueError):
        create_engine_adapter("unknown")
