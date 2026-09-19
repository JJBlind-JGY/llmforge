from __future__ import annotations
import json
import shlex
from dataclasses import dataclass


@dataclass(frozen=True)
class VLLMServerConfig:
    model: str
    revision: str
    served_model_name: str
    host: str = "127.0.0.1"
    port: int = 8000
    dtype: str = "bfloat16"
    kv_cache_dtype: str = "bfloat16"
    tensor_parallel_size: int = 1
    max_model_len: int = 8192
    gpu_memory_utilization: float = 0.90
    model_impl: str = "vllm"
    load_format: str = "safetensors"
    generation_config: str = "vllm"
    seed: int = 0
    enable_prefix_caching: bool = False
    enable_chunked_prefill: bool = True
    enable_thinking: bool = False

    def command(self) -> list[str]:
        cmd = [
            "vllm",
            "serve",
            self.model,
            "--revision",
            self.revision,
            "--tokenizer-revision",
            self.revision,
            "--served-model-name",
            self.served_model_name,
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--dtype",
            self.dtype,
            "--kv-cache-dtype",
            self.kv_cache_dtype,
            "--tensor-parallel-size",
            str(self.tensor_parallel_size),
            "--max-model-len",
            str(self.max_model_len),
            "--gpu-memory-utilization",
            str(self.gpu_memory_utilization),
            "--model-impl",
            self.model_impl,
            "--load-format",
            self.load_format,
            "--generation-config",
            self.generation_config,
            "--seed",
            str(self.seed),
            "--enable-prefix-caching"
            if self.enable_prefix_caching
            else "--no-enable-prefix-caching",
            "--enable-chunked-prefill"
            if self.enable_chunked_prefill
            else "--no-enable-chunked-prefill",
            "--default-chat-template-kwargs",
            json.dumps({"enable_thinking": self.enable_thinking}),
        ]
        return cmd

    def shell_command(
        self,
        *,
        gpu: int,
        hf_home: str,
        disable_flashinfer_sampler: bool = True,
        offline: bool = True,
    ) -> str:
        env = {"CUDA_VISIBLE_DEVICES": str(gpu), "HF_HOME": hf_home}
        if disable_flashinfer_sampler:
            env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
        if offline:
            env["HF_HUB_OFFLINE"] = "1"
        prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
        return f"{prefix} {shlex.join(self.command())}"
