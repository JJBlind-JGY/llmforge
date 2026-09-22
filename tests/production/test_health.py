from llmforge.production.health import (
    CheckResult,
    HealthEvaluator,
    HealthStatus,
)


class FakeCheck:
    def __init__(
        self,
        *,
        name: str,
        ok: bool,
        required: bool,
    ) -> None:
        self.name = name
        self.ok = ok
        self.required = required

    def check(
        self,
    ) -> CheckResult:
        return CheckResult(
            name=self.name,
            ok=self.ok,
            required=self.required,
            detail="fake",
        )


def test_required_failure_blocks_readiness() -> None:
    evaluator = HealthEvaluator(
        checks=[
            FakeCheck(
                name="upstream",
                ok=False,
                required=True,
            ),
        ],
        failure_threshold=2,
    )

    first = evaluator.evaluate()

    assert not first.ready
    assert first.status == HealthStatus.DEGRADED

    second = evaluator.evaluate()

    assert not second.ready
    assert second.status == HealthStatus.UNHEALTHY


def test_optional_failure_keeps_readiness() -> None:
    evaluator = HealthEvaluator(
        checks=[
            FakeCheck(
                name="gpu",
                ok=False,
                required=False,
            ),
        ],
        failure_threshold=1,
    )

    report = evaluator.evaluate()

    assert report.ready
    assert report.status == HealthStatus.DEGRADED
