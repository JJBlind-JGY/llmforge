from llmforge.runtime.report import render_runtime_report


def test_runtime_report_contains_metrics() -> None:
    distribution = {
        "count": 1,
        "mean": 1.0,
        "p50": 1.0,
        "p95": 1.0,
        "p99": 1.0,
        "max": 1.0,
    }
    summary = {
        "requests": [],
        "scheduler_steps": 1,
        "scheduler_duration_us": distribution,
        "queue_wait_ms": distribution,
        "request_lifetime_ms": distribution,
        "kv_usage_ratio_max": 0.5,
        "preemptions_total": 0,
        "scheduled_prompt_tokens": 128,
        "scheduled_output_tokens": 1,
        "trace_drop_events": 0,
    }
    report = render_runtime_report(summary)
    assert "Scheduler steps: 1" in report
    assert "Scheduled prompt tokens: 128" in report
    assert "Queue wait (ms)" in report
