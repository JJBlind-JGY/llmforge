from llmforge.serving.workloads import (
    fixed_workload,
    shared_prefix_workload,
    poisson_workload,
    burst_workload,
)

VOCAB = 151936


def test_fixed_exact_length_and_unique():
    r = fixed_workload(
        num_requests=2,
        prompt_tokens=512,
        output_tokens=32,
        seed=0,
        vocab_size=VOCAB,
    )
    assert all(x.prompt_tokens == 512 for x in r)
    assert r[0].prompt_token_ids != r[1].prompt_token_ids


def test_shared_prefix():
    r = shared_prefix_workload(
        num_requests=3,
        prompt_tokens=1024,
        output_tokens=16,
        shared_prefix_tokens=768,
        seed=0,
        vocab_size=VOCAB,
    )
    prefix = r[0].prompt_token_ids[:768]
    assert all(x.prompt_token_ids[:768] == prefix for x in r)
    assert r[0].prompt_token_ids[768:] != r[1].prompt_token_ids[768:]


def test_poisson_monotonic():
    r = poisson_workload(
        num_requests=10,
        prompt_tokens=128,
        output_tokens=8,
        request_rate=4,
        seed=0,
        vocab_size=VOCAB,
    )
    offsets = [x.arrival_offset_s for x in r]
    assert offsets == sorted(offsets)
    assert offsets[-1] > 0


def test_burst_groups():
    r = burst_workload(
        num_bursts=2,
        burst_size=3,
        prompt_tokens=128,
        output_tokens=8,
        inter_burst_s=2,
        seed=0,
        vocab_size=VOCAB,
    )
    assert [x.arrival_offset_s for x in r] == [0, 0, 0, 2, 2, 2]
