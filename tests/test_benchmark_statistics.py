import pytest

from llmforge.benchmark.statistics import percentile, summarize_ms


def test_percentile() -> None:
    samples = [1.0, 2.0, 3.0, 4.0, 5.0]

    assert percentile(samples=samples, q=0.0) == 1.0
    assert percentile(samples=samples, q=0.5) == 3.0
    assert percentile(samples=samples, q=1.0) == 5.0


def test_percentile_rejects_empty_samples() -> None:
    with pytest.raises(ValueError):
        percentile([], 0.5)


def test_summarize_ms() -> None:
    stats = summarize_ms([1.0, 2.0, 3.0])

    assert stats.count == 3
    assert stats.mean_ms == 2.0
    assert stats.median_ms == 2.0
    assert stats.min_ms == 1.0
    assert stats.max_ms == 3.0
