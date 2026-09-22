from llmforge.engines.correctness import (
    EngineCorrectnessRecord,
    compare_engine_correctness,
)


def record(engine: str, response_hash: str) -> EngineCorrectnessRecord:
    return EngineCorrectnessRecord(
        case_id="case",
        engine=engine,
        success=True,
        response_sha256=response_hash,
        prompt_tokens=10,
        completion_tokens=5,
        finish_reason="stop",
    )


def test_semantic_contract_allows_text_difference() -> None:
    result = compare_engine_correctness(
        {"case": record("vllm", "aaa")},
        {"case": record("sglang", "bbb")},
        require_exact_text=False,
    )

    assert result.passed


def test_exact_mode_catches_text_difference() -> None:
    result = compare_engine_correctness(
        {"case": record("vllm", "aaa")},
        {"case": record("sglang", "bbb")},
        require_exact_text=True,
    )

    assert not result.passed
