from llmforge.engines import create_engine_adapter


BASE = {
    "model": "Qwen/Qwen3-8B",
    "revision": "abc",
    "served_model_name": "qwen3-8b",
    "cuda_visible_devices": "3",
}


def test_vllm_adapter_builds_openai_server() -> None:
    spec = create_engine_adapter("vllm").build_launch_spec(BASE)
    command = list(spec.command)

    assert command[:2] == ["vllm", "serve"]
    assert "--served-model-name" in command
    assert spec.base_url.endswith(":8000")


def test_sglang_adapter_builds_server() -> None:
    spec = create_engine_adapter("sglang").build_launch_spec(BASE)
    command = list(spec.command)

    assert command[:3] == ["python", "-m", "sglang.launch_server"]
    assert "--model-path" in command
    assert spec.base_url.endswith(":30000")
