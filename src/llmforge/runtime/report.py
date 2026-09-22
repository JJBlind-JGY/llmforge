from __future__ import annotations
from typing import Any


def _fmt(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def render_runtime_report(
    summary: dict[str, Any], *, title: str = "LLMForge M4 Runtime Trace"
) -> str:
    lines = [
        f"# {title}",
        "",
        "## Runtime overview",
        "",
        f"- Scheduler steps: {summary['scheduler_steps']}",
        f"- Scheduled prompt tokens: {summary['scheduled_prompt_tokens']}",
        f"- Scheduled output tokens: {summary['scheduled_output_tokens']}",
        f"- Preemptions: {summary['preemptions_total']}",
        f"- Peak observed KV usage: {_fmt(summary['kv_usage_ratio_max'])}",
        f"- Trace drop events: {summary['trace_drop_events']}",
        "",
        "## Distributions",
        "",
        "| Metric | Mean | P50 | P95 | P99 | Max |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, dist in (
        ("Scheduler duration (us)", summary["scheduler_duration_us"]),
        ("Queue wait (ms)", summary["queue_wait_ms"]),
        ("Request lifetime (ms)", summary["request_lifetime_ms"]),
    ):
        lines.append(
            "| "
            + f"{label} | {_fmt(dist['mean'])} | {_fmt(dist['p50'])} | "
            + f"{_fmt(dist['p95'])} | {_fmt(dist['p99'])} | {_fmt(dist['max'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "> Fill this section only after the real trace and matching M3 client benchmark are available.",
            "",
        ]
    )
    return "\n".join(lines)


def render_request_table(summary: dict[str, Any]) -> str:
    lines = [
        "| Request | Queue wait ms | Lifetime ms | Steps | Prompt tokens | Output tokens | Preemptions | Finished |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for request in summary["requests"]:
        lines.append(
            "| "
            + f"{request['request_id']} | {_fmt(request['queue_wait_ms'])} | {_fmt(request['lifetime_ms'])} | "
            + f"{request['scheduled_steps']} | {request['scheduled_prompt_tokens']} | {request['scheduled_output_tokens']} | "
            + f"{request['preemptions']} | {request['finished']} |"
        )
    return "\n".join(lines)
