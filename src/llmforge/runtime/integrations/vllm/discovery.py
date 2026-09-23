from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import inspect
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class VLLMComponent:
    logical_name: str
    module_name: str | None
    symbol_name: str | None
    source_path: str | None
    available: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class VLLMRuntimeDiscovery:
    version: str | None
    components: tuple[VLLMComponent, ...]
    scheduler_capabilities: dict[str, bool]

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "components": [component.to_dict() for component in self.components],
            "scheduler_capabilities": dict(self.scheduler_capabilities),
        }


_COMPONENT_CANDIDATES = {
    "scheduler_config": (("vllm.config.scheduler", "SchedulerConfig"),),
    "scheduler": (("vllm.v1.core.sched.scheduler", "Scheduler"),),
    "request": (("vllm.v1.request", "Request"),),
    "scheduler_output": (("vllm.v1.core.sched.output", "SchedulerOutput"),),
    "kv_cache_manager": (("vllm.v1.core.kv_cache_manager", "KVCacheManager"),),
    "model_runner": (
        ("vllm.v1.worker.gpu.model_runner", "GPUModelRunner"),
        ("vllm.v1.worker.gpu_model_runner", "GPUModelRunner"),
    ),
    "input_batch": (("vllm.v1.worker.gpu_input_batch", "InputBatch"),),
}


def resolve_first_symbol(
    candidates: Iterable[tuple[str, str]],
) -> VLLMComponent | None:
    for module_name, symbol_name in candidates:
        try:
            spec = importlib.util.find_spec(module_name)
        except (ImportError, ModuleNotFoundError):
            spec = None

        if spec is None:
            continue

        module = importlib.import_module(module_name)
        symbol = getattr(module, symbol_name, None)
        if symbol is None:
            continue

        source_path = inspect.getsourcefile(symbol) or spec.origin
        if source_path is not None:
            source_path = str(Path(source_path).resolve())

        return VLLMComponent(
            logical_name="",
            module_name=module_name,
            symbol_name=symbol_name,
            source_path=source_path,
            available=True,
        )

    return None


def discover_vllm_runtime() -> VLLMRuntimeDiscovery:
    try:
        version = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        version = None

    components = []
    for logical_name, candidates in _COMPONENT_CANDIDATES.items():
        resolved = resolve_first_symbol(candidates)
        if resolved is None:
            components.append(VLLMComponent(logical_name, None, None, None, False))
        else:
            components.append(
                VLLMComponent(
                    logical_name,
                    resolved.module_name,
                    resolved.symbol_name,
                    resolved.source_path,
                    True,
                )
            )

    capabilities = {
        "get_request_counts": False,
        "get_kv_cache_usage": False,
        "schedule": False,
        "add_request": False,
        "update_from_output": False,
        "finish_requests": False,
    }

    scheduler = next(
        component for component in components if component.logical_name == "scheduler"
    )

    if scheduler.available and scheduler.module_name and scheduler.symbol_name:
        module = importlib.import_module(scheduler.module_name)
        scheduler_cls = getattr(module, scheduler.symbol_name)
        for name in capabilities:
            capabilities[name] = callable(getattr(scheduler_cls, name, None))

    return VLLMRuntimeDiscovery(
        version=version,
        components=tuple(components),
        scheduler_capabilities=capabilities,
    )
