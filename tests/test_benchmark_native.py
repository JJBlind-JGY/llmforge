from llmforge.benchmark.native import parse_key_value_output


def test_parse_key_value_output() -> None:
    output = """
block_size=256
median_ms=1.92614
bandwidth_gbps=836.185
correct=true
device_name=NVIDIA GeForce RTX 4090
"""
    result = parse_key_value_output(output)
    assert result["block_size"] == 256
    assert result["median_ms"] == 1.92614
    assert result["bandwidth_gbps"] == 836.185
    assert result["correct"] is True
    assert result["device_name"] == "NVIDIA GeForce RTX 4090"
