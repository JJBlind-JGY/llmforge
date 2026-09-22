from llmforge.optimization.report import (
    render_optimization_report,
)


def test_report_keeps_regressions_visible() -> None:
    analysis = {
        "claim_ready": True,
        "selection_gate_passed": True,
        "correctness_passed": True,
        "baseline_runs": 5,
        "candidate_runs": 5,
        "objective_metric": "tpot_p99_ms",
        "objective_direction": "minimize",
        "metrics": [
            {
                "metric": "tpot_p99_ms",
                "direction": "minimize",
                "baseline_median": 40.0,
                "candidate_median": 30.0,
                "relative_change": -0.25,
                "improvement_fraction": 0.25,
            },
            {
                "metric": "output_throughput_tok_s",
                "direction": "maximize",
                "baseline_median": 100.0,
                "candidate_median": 97.0,
                "relative_change": -0.03,
                "improvement_fraction": -0.03,
            },
        ],
        "guardrails": [
            {
                "metric": "output_throughput_tok_s",
                "allowed_relative_regression": 0.05,
                "observed_relative_regression": 0.03,
                "passed": True,
            }
        ],
    }

    report = render_optimization_report(analysis)

    assert "+25.00%" in report
    assert "-3.00%" in report
    assert "Negative results" in report
