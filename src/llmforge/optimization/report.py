"""Render M7 trade-offs without hiding regressions."""

from __future__ import annotations

from typing import Any


def _percent(
    value: float | None,
) -> str:
    if value is None:
        return "-"
    return f"{value * 100:+.2f}%"


def _number(
    value: float | None,
) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"


def render_optimization_report(
    analysis: dict[str, Any],
) -> str:
    lines = [
        "# M7 Optimization Report",
        "",
        (f"> claim_ready = `{analysis['claim_ready']}`"),
        "",
        "## Evidence gates",
        "",
        (f"- Selection gate passed: {analysis['selection_gate_passed']}"),
        (f"- Correctness passed: {analysis['correctness_passed']}"),
        (f"- Baseline repetitions: {analysis['baseline_runs']}"),
        (f"- Candidate repetitions: {analysis['candidate_runs']}"),
        "",
        "## Objective",
        "",
        (f"- Metric: `{analysis['objective_metric']}`"),
        (f"- Direction: `{analysis['objective_direction']}`"),
        "",
        "## Full trade-off table",
        "",
        (
            "| Metric | Direction | Baseline median | "
            "Candidate median | Relative change | "
            "Improvement |"
        ),
        ("| --- | --- | ---: | ---: | ---: | ---: |"),
    ]

    for metric in analysis["metrics"]:
        lines.append(
            "| "
            f"{metric['metric']} | "
            f"{metric['direction']} | "
            f"{_number(metric['baseline_median'])} | "
            f"{_number(metric['candidate_median'])} | "
            f"{_percent(metric['relative_change'])} | "
            f"{_percent(metric['improvement_fraction'])} |"
        )

    lines.extend(
        [
            "",
            "## Regression guardrails",
            "",
            ("| Metric | Allowed regression | Observed regression | Pass |"),
            "| --- | ---: | ---: | --- |",
        ]
    )

    for gate in analysis["guardrails"]:
        lines.append(
            "| "
            f"{gate['metric']} | "
            f"{_percent(gate['allowed_relative_regression'])} | "
            f"{_percent(gate['observed_relative_regression'])} | "
            f"{gate['passed']} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "Do not write a performance claim here unless "
                "`claim_ready=true` and the raw artifacts, profiler "
                "evidence, correctness result, and workload are linked."
            ),
            "",
            "## Negative results",
            "",
            (
                "Record failed hypotheses and regressions. A rejected "
                "candidate is still valid project evidence when its "
                "failure mechanism is explained."
            ),
            "",
        ]
    )

    return "\n".join(lines)
