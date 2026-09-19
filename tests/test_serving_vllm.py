from llmforge.serving.vllm import VLLMServerConfig


def test_baseline_flags():
    c = VLLMServerConfig(
        model="Qwen/Qwen3-8B", revision="r", served_model_name="qwen3-8b"
    )
    cmd = c.command()
    assert "--no-enable-prefix-caching" in cmd
    assert "--enable-chunked-prefill" in cmd
    shell = c.shell_command(gpu=3, hf_home="/tmp/hf")
    assert "CUDA_VISIBLE_DEVICES=3" in shell
    assert "VLLM_USE_FLASHINFER_SAMPLER=0" in shell
