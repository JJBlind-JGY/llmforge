from __future__ import annotations
from typing import Any


def _fmt(value: float | None, digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def render_case_markdown(case_name: str, artifact: dict[str, Any]) -> str:
    s = artifact["summary"]
    rows = []
    for key, label in (
        ("ttft_ms", "TTFT (ms)"),
        ("tpot_ms", "TPOT (ms)"),
        ("itl_ms", "ITL (ms)"),
        ("e2e_ms", "E2E (ms)"),
        ("client_queue_ms", "Client queue (ms)"),
    ):
        d = s[key]
        rows.append(
            f"| {label} | {_fmt(d['mean'])} | {_fmt(d['p50'])} | "
            f"{_fmt(d['p95'])} | {_fmt(d['p99'])} |"
        )

    return "\n".join(
        [
            f"## {case_name}",
            "",
            f"- Workload: `{artifact['workload']['kind']}`",
            f"- Requests: {s['requests_total']}",
            f"- Success: {s['requests_succeeded']}",
            f"- Request throughput: {_fmt(s['request_throughput_rps'])} req/s",
            f"- Output throughput: {_fmt(s['output_throughput_tok_s'])} tok/s",
            "",
            "| Metric | Mean | P50 | P95 | P99 |",
            "| --- | ---: | ---: | ---: | ---: |",
            *rows,
            "",
        ]
    )


def render_report(artifacts: list[tuple[str, dict[str, Any]]]) -> str:
    chunks = [
        "# LLMForge M3 Serving Benchmark Results",
        "",
        "> Generated from raw artifacts. Add causal interpretation only after "
        "checking the actual server configuration, GPU state, and profiler/telemetry evidence.",
        "",
    ]
    for name, artifact in artifacts:
        chunks.append(render_case_markdown(name, artifact))
    return "\n".join(chunks)
