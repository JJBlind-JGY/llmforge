"""Environment fingerprint collection for reproducible experiments."""

from __future__ import annotations

import argparse
import csv
import ctypes
import json
import os
import platform
import re
import subprocess
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path
from typing import Any

from llmforge import __version__


def _run_command(args: list[str], timeout: float = 5.0) -> str | None:
    """Run a command and return stripped stdout when successful."""
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None

    if result.returncode != 0:
        return None

    return result.stdout.strip()


def _total_memory_bytes() -> int | None:
    """Return total physical system memory using only the standard library."""
    if os.name == "nt":

        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatusEx()
        status.dwLength = ctypes.sizeof(status)

        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullTotalPhys)

        return None

    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        physical_pages = os.sysconf("SC_PHYS_PAGES")
        return int(page_size * physical_pages)
    except (AttributeError, OSError, ValueError):
        return None


def _cpu_model() -> str | None:
    """Return a human-readable CPU model when available."""
    if os.name == "nt":
        return os.environ.get("PROCESSOR_IDENTIFIER") or platform.processor() or None

    lscpu_output = _run_command(["lscpu"])
    if lscpu_output:
        match = re.search(r"^Model name:\s*(.+)$", lscpu_output, re.MULTILINE)
        if match:
            return match.group(1).strip()

    return platform.processor() or None


def _parse_nvidia_smi_banner(text: str) -> tuple[str | None, str | None]:
    """Parse driver and driver-supported CUDA versions from nvidia-smi output."""
    driver_match = re.search(r"Driver Version:\s*([0-9.]+)", text)
    cuda_match = re.search(r"CUDA Version:\s*([0-9.]+)", text)

    driver_version = driver_match.group(1) if driver_match else None
    cuda_version = cuda_match.group(1) if cuda_match else None

    return driver_version, cuda_version


def _parse_gpu_query(text: str) -> list[dict[str, Any]]:
    """Parse CSV output produced by nvidia-smi --query-gpu."""
    devices: list[dict[str, Any]] = []

    for row in csv.reader(StringIO(text)):
        if len(row) != 4:
            continue

        index, name, memory_mib, pci_bus_id = (value.strip() for value in row)

        try:
            memory_value = int(float(memory_mib))
        except ValueError:
            memory_value = None

        devices.append(
            {
                "index": int(index),
                "name": name,
                "memory_total_mib": memory_value,
                "pci_bus_id": pci_bus_id,
            }
        )

    return devices


def _collect_git_metadata() -> dict[str, Any]:
    commit = _run_command(["git", "rev-parse", "HEAD"])
    branch = _run_command(["git", "branch", "--show-current"])
    status = _run_command(["git", "status", "--porcelain"])

    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(status) if status is not None else None,
    }


def _collect_gpu_metadata() -> dict[str, Any]:
    banner = _run_command(["nvidia-smi"])

    if banner is None:
        return {
            "nvidia_smi_available": False,
            "driver_version": None,
            "driver_supported_cuda_version": None,
            "devices": [],
            "topology": None,
        }

    driver_version, driver_cuda_version = _parse_nvidia_smi_banner(banner)

    query = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,pci.bus_id",
            "--format=csv,noheadr,nounits",
        ]
    )

    topology = _run_command(["nvidia-smi", "topo", "-m"])

    return {
        "nvidia_smi_available": True,
        "driver_version": driver_version,
        "driver_supported_cuda_version": driver_cuda_version,
        "devices": _parse_gpu_query(query) if query else [],
        "topology": topology,
    }


def _collect_cuda_toolkit() -> dict[str, Any]:
    nvcc_output = _run_command(["nvcc", "--version"])

    if nvcc_output is None:
        return {
            "nvcc_available": False,
            "version": None,
        }

    match = re.search(r"release\s+([0-9.]+)", nvcc_output)

    return {
        "nvcc_available": True,
        "version": match.group(1) if match else None,
    }


def _collect_pytorch() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {
            "installed": False,
            "version": None,
            "cuda_runtime_version": None,
            "cuda_available": False,
            "device_count": 0,
            "devices": [],
        }

    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0

    return {
        "installed": True,
        "version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "cuda_available": cuda_available,
        "device_count": device_count,
        "devices": [torch.cuda.get_device_name(index) for index in range(device_count)],
    }


def collect_environment(role: str = "other") -> dict[str, Any]:
    """Collect a privacy-aware environment fingerprint."""
    return {
        "schema_version": "1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "role": role,
        "project": {
            "name": "llmforge",
            "version": __version__,
            "git": _collect_git_metadata(),
        },
        "host": {
            "os": platform.system(),
            "os_release": platform.release(),
            "architecture": platform.machine(),
            "cpu_model": _cpu_model(),
            "logical_cpu_count": os.cpu_count(),
            "memory_total_bytes": _total_memory_bytes(),
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "tools": {
            "git": _run_command(["git", "--version"]),
            "uv": _run_command(["uv", "--version"]),
        },
        "gpu": _collect_gpu_metadata(),
        "cuda_toolkit": _collect_cuda_toolkit(),
        "pytorch": _collect_pytorch(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect an LLMForge experiment environment fingerprint."
    )
    parser.add_argument(
        "--role",
        choices=["local-dev", "gpu-server", "other"],
        default="other",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    args = parser.parse_args()

    fingerprint = collect_environment(role=args.role)
    payload = json.dumps(fingerprint, indent=2, ensure_ascii=False)

    if args.output is None:
        print(payload)
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload + "\n", encoding="utf-8")

    print(f"Environment fingerprint written to: {args.output}")


if __name__ == "__main__":
    main()
