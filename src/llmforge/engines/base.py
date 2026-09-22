"""Version-conscious engine adapter contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


class EngineCapability(StrEnum):
    OPENAI_CHAT = "openai_chat"
    STREAMING = "streaming"
    PREFIX_CACHE = "prefix_cache"
    CHUNKED_PREFILL = "chunked_prefill"
    TENSOR_PARALLEL = "tensor_parallel"
    DATA_PARALLEL = "data_parallel"
    PD_DISAGGREGATION = "pd_disaggregation"
    PROMETHEUS_METRICS = "prometheus_metrics"


@dataclass(frozen=True)
class EngineLaunchSpec:
    engine: str
    command: tuple[str, ...]
    environment: dict[str, str]
    base_url: str
    served_model_name: str
    capabilities: tuple[EngineCapability, ...]
    metadata: dict[str, Any]

    def to_dict(self) -> dict:
        return {
            "engine": self.engine,
            "command": list(self.command),
            "environment": dict(self.environment),
            "base_url": self.base_url,
            "served_model_name": self.served_model_name,
            "capabilities": [item.value for item in self.capabilities],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class EngineProbeResult:
    engine: str
    ready: bool
    models: tuple[str, ...]
    health_status: int | None
    model_status: int | None
    error: str | None

    def to_dict(self) -> dict:
        return asdict(self)


class EngineAdapter(ABC):
    name: str

    @abstractmethod
    def build_launch_spec(self, config: dict[str, Any]) -> EngineLaunchSpec:
        raise NotImplementedError

    @abstractmethod
    def capabilities(self) -> tuple[EngineCapability, ...]:
        raise NotImplementedError

    @abstractmethod
    def health_paths(self) -> tuple[str, ...]:
        raise NotImplementedError
