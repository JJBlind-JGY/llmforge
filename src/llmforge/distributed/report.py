"""Render M5 multi-GPU analysis to Markdown."""

from __future__ import annotations

from typing import Any


def _fmt(
    value: float | None,
    digits: int = 2,
) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def render_multi_gpu_report(
    analysis: dict[str, Any],
) -> str:
    baseline = analysis["baseline"]["serving"]

    lines = [
        "# M5 Multi-GPU Inference Report",
        "",
        (
            "> Generated from real M3 serving artifacts. "
            "Interpretation must be added only after "
            "the matching topology/NCCL/profiler evidence exists."
        ),
        "",
        "## Baseline",
        "",
        f"- Profile: `{baseline['profile']}`",
        f"- Devices: {baseline['devices']}",
        (f"- Output throughput: {_fmt(baseline['output_throughput_tok_s'])} tok/s"),
        (
            "- TTFT P50/P99: "
            f"{_fmt(baseline['ttft_p50_ms'])} / "
            f"{_fmt(baseline['ttft_p99_ms'])} ms"
        ),
        (
            "- TPOT P50/P99: "
            f"{_fmt(baseline['tpot_p50_ms'])} / "
            f"{_fmt(baseline['tpot_p99_ms'])} ms"
        ),
        "",
        "## Scaling comparison",
        "",
        (
            "| Profile | GPUs | Output tok/s | Speedup | "
            "Efficiency | TTFT P50 ratio | TTFT P99 ratio | "
            "TPOT P50 ratio | TPOT P99 ratio |"
        ),
        ("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"),
    ]

    for entry in analysis["candidates"]:
        serving = entry["serving"]
        scaling = entry["scaling"]

        lines.append(
            "| "
            f"{serving['profile']} | "
            f"{serving['devices']} | "
            f"{_fmt(serving['output_throughput_tok_s'])} | "
            f"{_fmt(scaling['output_throughput_speedup'])}x | "
            f"{_fmt(scaling['output_throughput_efficiency'])} | "
            f"{_fmt(scaling['ttft_p50_ratio'])}x | "
            f"{_fmt(scaling['ttft_p99_ratio'])}x | "
            f"{_fmt(scaling['tpot_p50_ratio'])}x | "
            f"{_fmt(scaling['tpot_p99_ratio'])}x |"
        )

    lines.extend(
        [
            "",
            "## GPU telemetry",
            "",
        ]
    )

    for entry in [
        analysis["baseline"],
        *analysis["candidates"],
    ]:
        serving = entry["serving"]

        lines.append(f"### {serving['profile']}")
        lines.append("")

        telemetry = entry.get(
            "gpu_telemetry",
            [],
        )

        if not telemetry:
            lines.append("_No telemetry artifact attached._")
            lines.append("")
            continue

        lines.extend(
            [
                (
                    "| GPU | Mean util % | Max util % | "
                    "Max memory MiB | Max memory ratio | "
                    "Mean power W | Max power W |"
                ),
                ("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |"),
            ]
        )

        for gpu in telemetry:
            lines.append(
                "| "
                f"{gpu['gpu_index']} | "
                f"{_fmt(gpu['utilization_mean_percent'])} | "
                f"{_fmt(gpu['utilization_max_percent'])} | "
                f"{_fmt(gpu['memory_used_max_mib'])} | "
                f"{_fmt(gpu['memory_usage_max_ratio'])} | "
                f"{_fmt(gpu['power_mean_w'])} | "
                f"{_fmt(gpu['power_max_w'])} |"
            )

        lines.append("")

    lines.extend(
        [
            "## Communication evidence",
            "",
            (
                "Collective microbenchmarks characterize topology "
                "and available communication behavior. They are "
                "**not** model-level communication time."
            ),
            "",
            (
                "Model-level communication time must come from the "
                "matching short profiler trace."
            ),
            "",
            "## Interpretation",
            "",
            "- Why did TP change TTFT?",
            "- Why did TP change TPOT?",
            "- Did throughput scale with device count?",
            "- Did same-NUMA and cross-NUMA TP2 differ?",
            "- Did two replicas scale throughput better than TP?",
            "- Is the observed bottleneck compute, memory, or communication?",
            "",
            "## Conclusion",
            "",
            "_Pending real experiment evidence._",
            "",
        ]
    )

    return "\n".join(lines)
