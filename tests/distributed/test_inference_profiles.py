from llmforge.distributed.inference import (
    ModelParallelShape,
    ParallelMode,
    ParallelProfile,
    validate_parallel_profile,
)

MODEL = ModelParallelShape(
    num_attention_heads=32,
    num_key_value_heads=8,
    num_hidden_layers=36,
)


def test_tp2_is_valid_for_qwen3_8b() -> None:
    result = validate_parallel_profile(
        profile=ParallelProfile(
            name="tp2",
            mode=ParallelMode.TENSOR_PARALLEL,
            gpu_indices=(0, 1),
            tensor_parallel_size=2,
        ),
        model=MODEL,
    )

    assert result.valid


def test_tp4_is_valid_for_qwen3_8b() -> None:
    result = validate_parallel_profile(
        profile=ParallelProfile(
            name="tp4",
            mode=ParallelMode.TENSOR_PARALLEL,
            gpu_indices=(0, 1, 2, 3),
            tensor_parallel_size=4,
        ),
        model=MODEL,
    )

    assert result.valid


def test_tp3_is_rejected_for_qwen3_8b() -> None:
    result = validate_parallel_profile(
        profile=ParallelProfile(
            name="tp3",
            mode=ParallelMode.TENSOR_PARALLEL,
            gpu_indices=(0, 1, 2),
            tensor_parallel_size=3,
        ),
        model=MODEL,
    )

    assert not result.valid
    assert any("attention_heads" in reason for reason in result.reasons)


def test_dp2_requires_two_visible_gpus() -> None:
    result = validate_parallel_profile(
        profile=ParallelProfile(
            name="dp2",
            mode=ParallelMode.DATA_PARALLEL,
            gpu_indices=(0, 1),
            data_parallel_size=2,
        ),
        model=MODEL,
    )

    assert result.valid
