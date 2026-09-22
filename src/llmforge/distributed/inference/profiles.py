from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum


class ParallelMode(StrEnum):
    SINGLE = "single"
    TENSOR_PARALLEL = "tensor_parallel"
    DATA_PARALLEL = "data_parallel"
    PIPELINE_PARALLEL = "pipeline_parallel"


@dataclass(frozen=True)
class ModelParallelShape:
    num_attention_heads: int
    num_key_value_heads: int
    num_hidden_layers: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ParallelProfile:
    name: str
    mode: ParallelMode
    gpu_indices: tuple[int, ...]
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1
    optional: bool = False
    note: str = ""

    @property
    def world_size(self) -> int:
        return (
            self.tensor_parallel_size
            * self.pipeline_parallel_size
            * self.data_parallel_size
        )

    def to_dict(self) -> dict:
        data = asdict(self)
        data["mode"] = self.mode.value
        data["gpu_indices"] = list(self.gpu_indices)
        data["world_size"] = self.world_size
        return data


@dataclass(frozen=True)
class ProfileValidation:
    valid: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "reasons": list(self.reasons),
        }


def _tp_kv_compatible(
    *,
    kv_heads: int,
    tp_size: int,
) -> bool:
    if kv_heads >= tp_size:
        return kv_heads % tp_size == 0

    return tp_size % kv_heads == 0


def validate_parallel_profile(
    *,
    profile: ParallelProfile,
    model: ModelParallelShape,
) -> ProfileValidation:
    reasons = []

    if profile.tensor_parallel_size <= 0:
        reasons.append("tensor_parallel_size must be positive.")

    if profile.pipeline_parallel_size <= 0:
        reasons.append("pipeline_parallel_size must be positive.")

    if profile.data_parallel_size <= 0:
        reasons.append("data_parallel_size must be positive.")

    if len(profile.gpu_indices) != profile.world_size:
        reasons.append("Number of visible GPUs must equal TP × PP × DP world size.")

    if model.num_attention_heads % profile.tensor_parallel_size != 0:
        reasons.append("num_attention_heads must be divisible by tensor_parallel_size.")

    if not _tp_kv_compatible(
        kv_heads=model.num_key_value_heads,
        tp_size=profile.tensor_parallel_size,
    ):
        reasons.append(
            "KV heads cannot be evenly sharded or "
            "replicated for this tensor-parallel size."
        )

    if (
        profile.mode == ParallelMode.PIPELINE_PARALLEL
        and model.num_hidden_layers < profile.pipeline_parallel_size
    ):
        reasons.append("pipeline_parallel_size exceeds model layers.")

    return ProfileValidation(
        valid=not reasons,
        reasons=tuple(reasons),
    )
