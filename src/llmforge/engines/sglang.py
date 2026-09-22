from __future__ import annotations

from typing import Any

from .base import EngineAdapter, EngineCapability, EngineLaunchSpec


class SGLangAdapter(EngineAdapter):
    name = "sglang"

    def capabilities(self) -> tuple[EngineCapability, ...]:
        return (
            EngineCapability.OPENAI_CHAT,
            EngineCapability.STREAMING,
            EngineCapability.PREFIX_CACHE,
            EngineCapability.TENSOR_PARALLEL,
            EngineCapability.PD_DISAGGREGATION,
        )

    def health_paths(self) -> tuple[str, ...]:
        # /v1/models is the cross-engine compatibility probe. /health may vary
        # across SGLang versions, so M8 does not make it the sole readiness gate.
        return ("/v1/models",)

    def build_launch_spec(self, config: dict[str, Any]) -> EngineLaunchSpec:
        model = str(config["model"])
        served_model_name = str(config.get("served_model_name", "qwen3-8b"))
        host = str(config.get("host", "127.0.0.1"))
        port = int(config.get("port", 30000))
        tp = int(config.get("tensor_parallel_size", 1))

        command = [
            "python",
            "-m",
            "sglang.launch_server",
            "--model-path",
            model,
            "--served-model-name",
            served_model_name,
            "--revision",
            str(config["revision"]),
            "--dtype",
            str(config.get("dtype", "bfloat16")),
            "--host",
            host,
            "--port",
            str(port),
            "--tp-size",
            str(tp),
            "--mem-fraction-static",
            str(config.get("mem_fraction_static", 0.80)),
            "--random-seed",
            str(config.get("seed", 0)),
        ]

        if bool(config.get("disable_radix_cache", False)):
            command.append("--disable-radix-cache")

        if config.get("context_length") is not None:
            command.extend(
                [
                    "--context-length",
                    str(int(config["context_length"])),
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
                "revision": config.get("revision"),
                "tensor_parallel_size": tp,
                "radix_cache_disabled": bool(config.get("disable_radix_cache", False)),
            },
        )
