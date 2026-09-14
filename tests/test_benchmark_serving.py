import pytest

from llmforge.benchmark.serving import (
    average_decode_latency_ms,
    static_decode_slot_stats,
    static_prompt_padding_stats,
)


def test_static_prompt_padding_stats() -> None:
    stats = static_prompt_padding_stats(
        [
            128,
            512,
            1024,
            2048,
        ]
    )

    assert stats.useful_tokens == 3712
    assert stats.executed_slots == 8192
    assert stats.wasted_slots == 4480

    assert stats.utilization == pytest.approx(3712 / 8192)

    assert stats.amplification == pytest.approx(8192 / 3712)


def test_static_decode_slot_stats() -> None:
    stats = static_decode_slot_stats(
        [
            4,
            8,
            16,
            32,
        ]
    )

    assert stats.useful_tokens == 60
    assert stats.executed_slots == 128
    assert stats.wasted_slots == 68

    assert stats.utilization == pytest.approx(60 / 128)


def test_average_decode_latency_ms() -> None:
    latency = average_decode_latency_ms(
        e2e_ms=100.0,
        first_token_proxy_ms=20.0,
        output_tokens=5,
    )

    assert latency == pytest.approx(20.0)


def test_one_output_token_has_no_decode_average() -> None:
    assert (
        average_decode_latency_ms(
            e2e_ms=20.0,
            first_token_proxy_ms=20.0,
            output_tokens=1,
        )
        is None
    )
