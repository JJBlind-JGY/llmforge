"""Build reproducible vLLM single/TP/DP/PP server commands."""

from __future__ import annotations

import json
import shlex
from typing import Any

from .profiles import ParallelProfile


def build_vllm_server_command(
    *,
    base: dict[str, Any],
    profile: ParallelProfile,
    hf_home: str,
    offline: bool = True,
    disable_flashinfer_sampler: bool = True,
) -> str:
    command = [
        "vllm",
        "serve",
        str(base["model"]),
        "--revision",
        str(base["revision"]),
        "--tokenizer-revision",
        str(base["revision"]),
        "--served-model-name",
        str(base["served_model_name"]),
        "--host",
        str(base.get("host", "127.0.0.1")),
        "--port",
        str(base.get("port", 8000)),
        "--dtype",
        str(base.get("dtype", "bfloat16")),
        "--kv-cache-dtype",
        str(base.get("kv_cache_dtype", "bfloat16")),
        "--tensor-parallel-size",
        str(profile.tensor_parallel_size),
        "--pipeline-parallel-size",
        str(profile.pipeline_parallel_size),
        "--data-parallel-size",
        str(profile.data_parallel_size),
        "--max-model-len",
        str(base.get("max_model_len", 8192)),
        "--gpu-memory-utilization",
        str(base.get("gpu_memory_utilization", 0.90)),
        "--model-impl",
        str(base.get("model_impl", "vllm")),
        "--load-format",
        str(base.get("load_format", "safetensors")),
        "--generation-config",
        str(base.get("generation_config", "vllm")),
        "--seed",
        str(base.get("seed", 0)),
    ]

    command.append(
        "--enable-prefix-caching"
        if base.get("enable_prefix_caching", False)
        else "--no-enable-prefix-caching"
    )

    command.append(
        "--enable-chunked-prefill"
        if base.get("enable_chunked_prefill", True)
        else "--no-enable-chunked-prefill"
    )

    command.extend(
        [
            "--default-chat-template-kwargs",
            json.dumps(
                {
                    "enable_thinking": bool(
                        base.get(
                            "enable_thinking",
                            False,
                        )
                    )
                }
            ),
        ]
    )

    env = {
        "CUDA_VISIBLE_DEVICES": ",".join(str(index) for index in profile.gpu_indices),
        "HF_HOME": hf_home,
    }

    if offline:
        env["HF_HUB_OFFLINE"] = "1"

    if disable_flashinfer_sampler:
        env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

    env_prefix = " ".join(f"{name}={shlex.quote(value)}" for name, value in env.items())

    return env_prefix + " " + shlex.join(command)
