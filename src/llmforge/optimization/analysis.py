"""Repeated-run M7 comparison and regression-gate logic."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any

_METRIC_DIRECTIONS = {
    "request_throughput_rps": "maximize",
    "output_throughput_tok_s": "maximize",
    "total_throughput_tok_s": "maximize",
    "ttft_p50_ms": "minimize",
    "ttft_p95_ms": "minimize",
    "ttft_p99_ms": "minimize",
    "tpot_p50_ms": "minimize",
    "tpot_p95_ms": "minimize",
    "tpot_p99_ms": "minimize",
    "e2e_p50_ms": "minimize",
    "e2e_p95_ms": "minimize",
    "e2e_p99_ms": "minimize",
    "error_rate": "minimize",
}


def _nested(
    mapping: dict[str, Any],
    *keys: str,
) -> Any:
    current: Any = mapping

    for key in keys:
        if not isinstance(
            current,
            dict,
        ):
            return None
        current = current.get(key)

    return current


def _metric_from_summary(
    summary: dict[str, Any],
    metric: str,
) -> float | None:
    direct = summary.get(metric)

    if direct is not None:
        return float(direct)

    aliases = {
        "ttft_p50_ms": (
            "ttft_ms",
            "p50",
        ),
        "ttft_p95_ms": (
            "ttft_ms",
            "p95",
        ),
        "ttft_p99_ms": (
            "ttft_ms",
            "p99",
        ),
        "tpot_p50_ms": (
            "tpot_ms",
            "p50",
        ),
        "tpot_p95_ms": (
            "tpot_ms",
            "p95",
        ),
        "tpot_p99_ms": (
            "tpot_ms",
            "p99",
        ),
        "e2e_p50_ms": (
            "e2e_ms",
            "p50",
        ),
        "e2e_p95_ms": (
            "e2e_ms",
            "p95",
        ),
        "e2e_p99_ms": (
            "e2e_ms",
            "p99",
        ),
    }

    if metric in aliases:
        value = _nested(
            summary,
            *aliases[metric],
        )

        return None if value is None else float(value)

    if metric == "error_rate":
        if summary.get("error_rate") is not None:
            return float(summary["error_rate"])

        failed = summary.get("failed_requests")

        total = summary.get("total_requests")

        if failed is not None and total:
            return float(failed) / float(total)

    return None


def load_run_metrics(
    path: Path,
) -> dict[
    str,
    float,
]:
    payload = json.loads(path.read_text(encoding="utf-8"))

    summary = payload["summary"]

    values = {}

    for metric in _METRIC_DIRECTIONS:
        value = _metric_from_summary(
            summary,
            metric,
        )

        if value is not None:
            values[metric] = value

    return values


@dataclass(frozen=True)
class MetricComparison:
    metric: str
    direction: str
    baseline_values: tuple[float, ...]
    candidate_values: tuple[float, ...]
    baseline_median: float
    candidate_median: float
    relative_change: float
    improvement_fraction: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class GuardrailResult:
    metric: str
    allowed_relative_regression: float
    observed_relative_regression: float | None
    passed: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class OptimizationAnalysis:
    baseline_runs: int
    candidate_runs: int
    objective_metric: str
    objective_direction: str
    metrics: tuple[MetricComparison, ...]
    guardrails: tuple[GuardrailResult, ...]
    correctness_passed: bool
    selection_gate_passed: bool
    claim_ready: bool

    def to_dict(self) -> dict:
        return {
            "baseline_runs": (self.baseline_runs),
            "candidate_runs": (self.candidate_runs),
            "objective_metric": (self.objective_metric),
            "objective_direction": (self.objective_direction),
            "metrics": [metric.to_dict() for metric in self.metrics],
            "guardrails": [guardrail.to_dict() for guardrail in self.guardrails],
            "correctness_passed": (self.correctness_passed),
            "selection_gate_passed": (self.selection_gate_passed),
            "claim_ready": (self.claim_ready),
        }


def compare_metric(
    *,
    metric: str,
    baseline_values: list[float],
    candidate_values: list[float],
    direction: str,
) -> MetricComparison:
    if not baseline_values:
        raise ValueError(f"No baseline values for {metric}.")

    if not candidate_values:
        raise ValueError(f"No candidate values for {metric}.")

    baseline = float(median(baseline_values))

    candidate = float(median(candidate_values))

    if direction not in {
        "maximize",
        "minimize",
    }:
        raise ValueError("direction must be maximize or minimize.")

    if baseline == 0:
        if candidate == 0:
            relative_change = 0.0
            improvement = 0.0
        elif direction == "minimize":
            relative_change = float("inf")
            improvement = float("-inf")
        else:
            relative_change = float("inf")
            improvement = float("inf")
    else:
        relative_change = candidate / baseline - 1.0

        if direction == "maximize":
            improvement = (candidate - baseline) / abs(baseline)
        else:
            improvement = (baseline - candidate) / abs(baseline)

    return MetricComparison(
        metric=metric,
        direction=direction,
        baseline_values=tuple(baseline_values),
        candidate_values=tuple(candidate_values),
        baseline_median=baseline,
        candidate_median=candidate,
        relative_change=(relative_change),
        improvement_fraction=(improvement),
    )


def evaluate_guardrail(
    *,
    comparison: MetricComparison,
    allowed_relative_regression: float,
) -> GuardrailResult:
    regression = max(
        0.0,
        -comparison.improvement_fraction,
    )

    return GuardrailResult(
        metric=comparison.metric,
        allowed_relative_regression=(allowed_relative_regression),
        observed_relative_regression=(regression),
        passed=(regression <= allowed_relative_regression),
    )


def analyze_experiment(
    *,
    baseline_paths: list[Path],
    candidate_paths: list[Path],
    objective_metric: str,
    objective_direction: str,
    guardrail_limits: dict[
        str,
        float,
    ],
    correctness_passed: bool,
    selection_gate_passed: bool,
) -> OptimizationAnalysis:
    if len(baseline_paths) < 3:
        raise ValueError("At least three baseline runs are required.")

    if len(candidate_paths) < 3:
        raise ValueError("At least three candidate runs are required.")

    baseline_runs = [load_run_metrics(path) for path in baseline_paths]

    candidate_runs = [load_run_metrics(path) for path in candidate_paths]

    requested_metrics = {
        objective_metric,
        *guardrail_limits.keys(),
        "request_throughput_rps",
        "output_throughput_tok_s",
        "ttft_p99_ms",
        "tpot_p99_ms",
        "e2e_p99_ms",
    }

    comparisons = []

    for metric in sorted(requested_metrics):
        direction = (
            objective_direction
            if metric == objective_metric
            else _METRIC_DIRECTIONS.get(metric)
        )

        if direction is None:
            continue

        baseline_values = [run[metric] for run in baseline_runs if metric in run]

        candidate_values = [run[metric] for run in candidate_runs if metric in run]

        if len(baseline_values) != len(baseline_runs) or len(candidate_values) != len(
            candidate_runs
        ):
            continue

        comparisons.append(
            compare_metric(
                metric=metric,
                baseline_values=(baseline_values),
                candidate_values=(candidate_values),
                direction=direction,
            )
        )

    comparison_by_name = {item.metric: item for item in comparisons}

    guardrails = []

    for metric, limit in guardrail_limits.items():
        comparison = comparison_by_name.get(metric)

        if comparison is None:
            guardrails.append(
                GuardrailResult(
                    metric=metric,
                    allowed_relative_regression=(limit),
                    observed_relative_regression=None,
                    passed=False,
                )
            )

            continue

        guardrails.append(
            evaluate_guardrail(
                comparison=comparison,
                allowed_relative_regression=(limit),
            )
        )

    objective = comparison_by_name.get(objective_metric)

    objective_available = objective is not None

    guardrails_passed = all(result.passed for result in guardrails)

    claim_ready = (
        selection_gate_passed
        and correctness_passed
        and objective_available
        and guardrails_passed
    )

    return OptimizationAnalysis(
        baseline_runs=len(baseline_paths),
        candidate_runs=len(candidate_paths),
        objective_metric=(objective_metric),
        objective_direction=(objective_direction),
        metrics=tuple(comparisons),
        guardrails=tuple(guardrails),
        correctness_passed=(correctness_passed),
        selection_gate_passed=(selection_gate_passed),
        claim_ready=claim_ready,
    )
