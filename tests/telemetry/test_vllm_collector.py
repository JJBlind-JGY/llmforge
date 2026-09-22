from llmforge.telemetry.collectors.vllm import (
    parse_prometheus_scalars,
    snapshot_from_prometheus,
)


RAW = """
# HELP vllm:num_requests_running x
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="qwen3"} 3.0
vllm:num_requests_waiting{model_name="qwen3"} 2.0
vllm:kv_cache_usage_perc{model_name="qwen3"} 0.75
vllm:prefix_cache_queries{model_name="qwen3"} 100.0
vllm:prefix_cache_hits{model_name="qwen3"} 60.0
vllm:prompt_tokens_total{model_name="qwen3"} 1000.0
vllm:generation_tokens_total{model_name="qwen3"} 200.0
vllm:request_success_total{finish_reason="stop",model_name="qwen3"} 12.0
vllm:request_success_total{finish_reason="length",model_name="qwen3"} 2.0
vllm:time_to_first_token_seconds_bucket{le="0.1"} 5
"""


def test_prometheus_scalar_parser() -> None:
    values = parse_prometheus_scalars(RAW)

    assert values["vllm:num_requests_running"] == [3.0]


def test_vllm_snapshot_aggregates_finish_reasons() -> None:
    snapshot = snapshot_from_prometheus(RAW)

    assert snapshot.running == 3.0
    assert snapshot.waiting == 2.0
    assert snapshot.kv_usage_ratio == 0.75
    assert snapshot.request_success_total == 14.0
