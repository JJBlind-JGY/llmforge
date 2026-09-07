import pytest

from llmforge.benchmark.guard import (
    BenchmarkEnvironmentError,
    validate_formal_benchmark,
)


def test_guard_rejects_dirty_git(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CUDA_PREFIX", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    environment = {
        "project": {
            "git": {
                "dirty": True,
            }
        }
    }

    with pytest.raises(BenchmarkEnvironmentError):
        validate_formal_benchmark(environment=environment)


def test_guard_rejects_conda_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_PREFIX", "/opt/conda/envs/base")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    environment = {
        "project": {
            "git": {
                "dirty": False,
            }
        }
    }

    with pytest.raises(BenchmarkEnvironmentError):
        validate_formal_benchmark(environment=environment)


def test_guard_accepts_single_gpu_and_clean(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CUDA_PREFIX", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    environment = {
        "project": {
            "git": {
                "dirty": False,
            }
        }
    }

    validate_formal_benchmark(environment=environment)
