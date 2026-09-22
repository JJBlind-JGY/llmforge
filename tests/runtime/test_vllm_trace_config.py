from llmforge.runtime.integrations.vllm.config import VLLMTraceConfig


def test_trace_config_from_environment(monkeypatch) -> None:
    monkeypatch.setenv(
        "LLMFORGE_RUNTIME_TRACE_MODE",
        "sync",
    )
    monkeypatch.setenv(
        "LLMFORGE_RUNTIME_TRACE_PATH",
        "trace_{pid}.jsonl",
    )

    config = VLLMTraceConfig.from_environment()

    assert config.mode == "sync"
    assert config.path.name.startswith("trace_")
    assert config.path.suffix == ".jsonl"
