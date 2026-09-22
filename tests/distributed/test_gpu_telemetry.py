from llmforge.distributed.telemetry import (
    parse_nvidia_smi_sample,
)


def test_parse_gpu_telemetry() -> None:
    raw = "2026/09/22 12:00:00.000, 0, 75, 12000, 24564, 350.5\n"

    sample = parse_nvidia_smi_sample(raw)[0]

    assert sample.index == 0
    assert sample.utilization_gpu_percent == 75.0
    assert sample.memory_used_mib == 12000.0
    assert 0.48 < sample.memory_usage_ratio < 0.50
    assert sample.power_draw_w == 350.5
