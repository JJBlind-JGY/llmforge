from llmforge.distributed.report import (
    render_multi_gpu_report,
)


def test_report_contains_scaling_table() -> None:
    run = {
        "profile": "single",
        "devices": 1,
        "output_throughput_tok_s": 100.0,
        "ttft_p50_ms": 20.0,
        "ttft_p99_ms": 40.0,
        "tpot_p50_ms": 10.0,
        "tpot_p99_ms": 20.0,
    }

    analysis = {
        "baseline": {
            "serving": run,
            "gpu_telemetry": [],
        },
        "candidates": [
            {
                "serving": {
                    **run,
                    "profile": "tp2",
                    "devices": 2,
                    "output_throughput_tok_s": 160.0,
                },
                "scaling": {
                    "output_throughput_speedup": 1.6,
                    "output_throughput_efficiency": 0.8,
                    "ttft_p50_ratio": 1.2,
                    "ttft_p99_ratio": 1.3,
                    "tpot_p50_ratio": 0.9,
                    "tpot_p99_ratio": 1.0,
                },
                "gpu_telemetry": [],
            }
        ],
    }

    report = render_multi_gpu_report(analysis)

    assert "tp2" in report
    assert "1.60x" in report
    assert "0.80" in report
