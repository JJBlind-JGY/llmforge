from __future__ import annotations

from .base import EngineAdapter


def supported_engines() -> tuple[str, ...]:
    return ("vllm", "sglang")


def create_engine_adapter(name: str) -> EngineAdapter:
    normalized = name.lower()

    if normalized == "vllm":
        from .vllm import VLLMAdapter

        return VLLMAdapter()

    if normalized == "sglang":
        from .sglang import SGLangAdapter

        return SGLangAdapter()

    raise ValueError(
        f"Unsupported engine {name!r}; expected one of {supported_engines()}."
    )
