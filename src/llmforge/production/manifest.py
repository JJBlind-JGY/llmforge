"""Reproducibility manifests for long-running benchmark/service commands."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Mapping


_SAFE_ENV_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "CUDA_HOME",
    "HF_HOME",
    "HF_HUB_OFFLINE",
    "NCCL_DEBUG",
    "NCCL_SOCKET_IFNAME",
    "OMP_NUM_THREADS",
    "TOKENIZERS_PARALLELISM",
    "VLLM_USE_FLASHINFER_SAMPLER",
    "LLMFORGE_LOG_LEVEL",
    "LLMFORGE_TRACING_MODE",
)


def _git_output(
    *args: str,
) -> str | None:
    try:
        return subprocess.check_output(
            [
                "git",
                *args,
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (
        subprocess.SubprocessError,
        OSError,
    ):
        return None


def _git_dirty() -> bool | None:
    try:
        result = subprocess.run(
            [
                "git",
                "diff",
                "--quiet",
            ],
            check=False,
            stderr=subprocess.DEVNULL,
        )

        staged = subprocess.run(
            [
                "git",
                "diff",
                "--cached",
                "--quiet",
            ],
            check=False,
            stderr=subprocess.DEVNULL,
        )

        return result.returncode != 0 or staged.returncode != 0

    except OSError:
        return None


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def stable_json_hash(
    value: dict,
) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class EnvironmentFingerprint:
    hostname: str
    platform: str
    python_version: str
    git_commit: str | None
    git_branch: str | None
    git_dirty: bool | None
    selected_environment: dict[
        str,
        str,
    ]

    def to_dict(self) -> dict:
        return asdict(self)


def capture_environment_fingerprint(
    *,
    environment: Mapping[
        str,
        str,
    ]
    | None = None,
) -> EnvironmentFingerprint:
    environment = os.environ if environment is None else environment

    selected = {key: environment[key] for key in _SAFE_ENV_KEYS if key in environment}

    return EnvironmentFingerprint(
        hostname=platform.node(),
        platform=platform.platform(),
        python_version=(
            sys.version.replace(
                "\n",
                " ",
            )
        ),
        git_commit=_git_output(
            "rev-parse",
            "HEAD",
        ),
        git_branch=_git_output(
            "rev-parse",
            "--abbrev-ref",
            "HEAD",
        ),
        git_dirty=_git_dirty(),
        selected_environment=selected,
    )


@dataclass(frozen=True)
class RunManifest:
    schema_version: int
    run_id: str
    command: tuple[str, ...]
    cwd: str
    started_at: str
    config_sha256: str | None
    config: dict | None
    environment: EnvironmentFingerprint

    def to_dict(self) -> dict:
        data = asdict(self)
        data["command"] = list(self.command)
        return data


@dataclass(frozen=True)
class RunCompletion:
    run_id: str
    finished_at: str
    return_code: int | None
    timed_out: bool
    interrupted: bool
    log_path: str

    def to_dict(self) -> dict:
        return asdict(self)


def build_run_manifest(
    *,
    run_id: str,
    command: list[str],
    cwd: Path,
    config: dict | None = None,
    environment: Mapping[
        str,
        str,
    ]
    | None = None,
) -> RunManifest:
    return RunManifest(
        schema_version=1,
        run_id=run_id,
        command=tuple(command),
        cwd=str(cwd.resolve()),
        started_at=utc_now(),
        config_sha256=(stable_json_hash(config) if config is not None else None),
        config=config,
        environment=(capture_environment_fingerprint(environment=environment)),
    )


def write_json(
    path: Path,
    value: dict,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
