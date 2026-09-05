from llmforge.environment import (
    _parse_gpu_query,
    _parse_nvidia_smi_banner,
    _strip_ansi,
    collect_environment,
)


def test_strip_ansi() -> None:
    text = "\x1b[4mGPU0\tGPU1\x1b[0m"

    assert _strip_ansi(text) == "GPU0\tGPU1"


def test_parse_nvidia_smi_banner() -> None:
    output = """
    NVIDIA-SMI 580.126.09
    Driver Version: 580.126.09
    CUDA Version: 13.0
    """

    driver, cuda = _parse_nvidia_smi_banner(output)

    assert driver == "580.126.09"
    assert cuda == "13.0"


def test_parse_gpu_query() -> None:
    output = (
        "NVIDIA GeForce RTX 4090, 24564, 00000000:31:00.0\n"
        "NVIDIA GeForce RTX 4090, 24564, 00000000:4B:00.0"
    )

    devices = _parse_gpu_query(output)

    assert len(devices) == 2
    assert devices[0]["index"] == 0
    assert devices[0]["name"] == "NVIDIA GeForce RTX 4090"
    assert devices[0]["memory_total_mib"] == 24564
    assert devices[0]["pci_bus_id"] == "00000000:31:00.0"


def test_collect_environment_has_core_sections() -> None:
    fingerprint = collect_environment(role="local-dev")

    assert fingerprint["schema_version"] == "2"
    assert fingerprint["role"] == "local-dev"
    assert "project" in fingerprint
    assert "host" in fingerprint
    assert "python" in fingerprint
    assert "gpu" in fingerprint
    assert "cuda_toolkit" in fingerprint
    assert "build_toolchain" in fingerprint
    assert "pytorch" in fingerprint
