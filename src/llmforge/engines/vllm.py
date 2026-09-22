from __future__ import annotations

from typing import Any

from .base import EngineAdapter, EngineCapability, EngineLaunchSpec


class VLLMAdapter(EngineAdapter):
    name = "vllm"

    def capabilities(self) -> tuple[EngineCapability, ...]:
        return (
            EngineCapability.OPENAI_CHAT,
            EngineCapability.STREAMING,
            EngineCapability.PREFIX_CACHE,
            EngineCapability.CHUNKED_PREFILL,
            EngineCapability.TENSOR_PARALLEL,
            EngineCapability.DATA_PARALLEL,
            EngineCapability.PROMETHEUS_METRICS,
        )

    def health_paths(self) -> tuple[str, ...]:
        return ("/health", "/v1/models")

    def build_launch_spec(self, config: dict[str, Any]) -> EngineLaunchSpec:
        model = str(config["model"])
        revision = str(config["revision"])
        served_model_name = str(config.get("served_model_name", "qwen3-8b"))
        host = str(config.get("host", "127.0.0.1"))
        port = int(config.get("port", 8000))
        tp = int(config.get("tensor_parallel_size", 1))
        dp = int(config.get("data_parallel_size", 1))

        command = [
            "vllm",
            "serve",
            model,
            "--revision",
            revision,
            "--tokenizer-revision",
            revision,
            "--served-model-name",
            served_model_name,
            "--host",
            host,
            "--port",
            str(port),
            "--dtype",
            str(config.get("dtype", "bfloat16")),
            "--kv-cache-dtype",
            str(config.get("kv_cache_dtype", "bfloat16")),
            "--max-model-len",
            str(config.get("max_model_len", 8192)),
            "--gpu-memory-utilization",
            str(config.get("gpu_memory_utilization", 0.90)),
            "--tensor-parallel-size",
            str(tp),
            "--data-parallel-size",
            str(dp),
            "--generation-config",
            str(config.get("generation_config", "vllm")),
            "--seed",
            str(config.get("seed", 0)),
        ]

        command.append(
            "--enable-prefix-caching"
            if bool(config.get("enable_prefix_caching", False))
            else "--no-enable-prefix-caching"
        )

        command.append(
            "--enable-chunked-prefill"
            if bool(config.get("enable_chunked_prefill", True))
            else "--no-enable-chunked-prefill"
        )

        command.extend(
            [
                "--default-chat-template-kwargs",
                '{"enable_thinking":false}',
            ]
        )

        environment = {
            "CUDA_VISIBLE_DEVICES": str(config.get("cuda_visible_devices", "0")),
            "HF_HOME": str(config.get("hf_home", "")),
            "HF_HUB_OFFLINE": str(int(bool(config.get("offline", True)))),
        }

        return EngineLaunchSpec(
            engine=self.name,
            command=tuple(command),
            environment=environment,
            base_url=f"http://{host}:{port}",
            served_model_name=served_model_name,
            capabilities=self.capabilities(),
            metadata={
                "revision": revision,
                "tensor_parallel_size": tp,
                "data_parallel_size": dp,
            },
        )
