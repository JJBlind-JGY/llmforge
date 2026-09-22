from llmforge.optimization.correctness import (
    CorrectnessRecord,
    compare_correctness,
)


def record(
    case_id: str,
    response: str,
) -> CorrectnessRecord:
    return CorrectnessRecord(
        case_id=case_id,
        prompt_sha256="prompt",
        response_sha256=response,
        prompt_tokens=10,
        completion_tokens=4,
        finish_reason="stop",
        success=True,
    )


def test_exact_correctness_passes() -> None:
    baseline = {
        "a": record(
            "a",
            "hash",
        )
    }

    candidate = {
        "a": record(
            "a",
            "hash",
        )
    }

    result = compare_correctness(
        baseline,
        candidate,
    )

    assert result.passed


def test_response_mismatch_fails() -> None:
    result = compare_correctness(
        {
            "a": record(
                "a",
                "left",
            )
        },
        {
            "a": record(
                "a",
                "right",
            )
        },
    )

    assert not result.passed
    assert result.mismatches[0].field == "response_sha256"
