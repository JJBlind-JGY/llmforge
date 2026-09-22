"""Validated M6 production configuration.

Precedence is:

defaults < JSON config file < LLMFORGE_* environment variables

Keeping the contract explicit makes benchmark/service launches reproducible and
prevents hidden shell state from silently changing experiments.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()

    if normalized in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return True

    if normalized in {
        "0",
        "false",
        "no",
        "off",
    }:
        return False

    raise ValueError(f"Cannot parse boolean value {value!r}.")


def _parse_gpu_indices(
    value: str,
) -> tuple[int, ...]:
    if not value.strip():
        return ()

    indices = tuple(int(item.strip()) for item in value.split(",") if item.strip())

    if len(indices) != len(set(indices)):
        raise ValueError("GPU indices must be unique.")

    if any(index < 0 for index in indices):
        raise ValueError("GPU indices must be non-negative.")

    return indices


@dataclass(frozen=True)
class ServiceConfig:
    name: str = "llmforge"
    host: str = "127.0.0.1"
    port: int = 9108
    upstream_base_url: str = "http://127.0.0.1:8000"
    request_timeout_s: float = 180.0
    shutdown_grace_s: float = 10.0

    def validate(self) -> None:
        if not self.name:
            raise ValueError("service.name must not be empty.")

        if not 1 <= self.port <= 65535:
            raise ValueError("service.port must be in [1, 65535].")

        if self.request_timeout_s <= 0:
            raise ValueError("service.request_timeout_s must be positive.")

        if self.shutdown_grace_s <= 0:
            raise ValueError("service.shutdown_grace_s must be positive.")


@dataclass(frozen=True)
class HealthConfig:
    upstream_required: bool = True
    gpu_required: bool = True
    upstream_timeout_s: float = 2.0
    failure_threshold: int = 3

    def validate(self) -> None:
        if self.upstream_timeout_s <= 0:
            raise ValueError("health.upstream_timeout_s must be positive.")

        if self.failure_threshold <= 0:
            raise ValueError("health.failure_threshold must be positive.")


@dataclass(frozen=True)
class TracingConfig:
    mode: str = "jsonl"
    jsonl_path: str = "artifacts/telemetry/spans.jsonl"
    queue_size: int = 8192
    flush_every: int = 128
    otlp_endpoint: str | None = None

    def validate(self) -> None:
        if self.mode not in {
            "off",
            "jsonl",
            "otlp",
        }:
            raise ValueError("tracing.mode must be off, jsonl, or otlp.")

        if self.queue_size <= 0:
            raise ValueError("tracing.queue_size must be positive.")

        if self.flush_every <= 0:
            raise ValueError("tracing.flush_every must be positive.")

        if self.mode == "otlp" and not self.otlp_endpoint:
            raise ValueError("tracing.otlp_endpoint is required in OTLP mode.")


@dataclass(frozen=True)
class ObservabilityConfig:
    sample_interval_s: float = 1.0
    gpu_indices: tuple[int, ...] = ()
    scrape_vllm_metrics: bool = True
    scrape_gpu_metrics: bool = True
    log_level: str = "INFO"
    tracing: TracingConfig = field(default_factory=TracingConfig)

    def validate(self) -> None:
        if self.sample_interval_s <= 0:
            raise ValueError("observability.sample_interval_s must be positive.")

        normalized = self.log_level.upper()

        if normalized not in {
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
            "CRITICAL",
        }:
            raise ValueError(f"Unsupported log level: {self.log_level!r}.")

        if len(self.gpu_indices) != len(set(self.gpu_indices)):
            raise ValueError("observability.gpu_indices must be unique.")

        self.tracing.validate()


@dataclass(frozen=True)
class ProductionConfig:
    schema_version: int = 1
    service: ServiceConfig = field(default_factory=ServiceConfig)
    health: HealthConfig = field(default_factory=HealthConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError(
                f"Unsupported production schema_version={self.schema_version}."
            )

        self.service.validate()
        self.health.validate()
        self.observability.validate()

    def to_dict(self) -> dict:
        return asdict(self)


def _from_mapping(
    mapping: Mapping,
) -> ProductionConfig:
    service_raw = dict(
        mapping.get(
            "service",
            {},
        )
    )

    health_raw = dict(
        mapping.get(
            "health",
            {},
        )
    )

    observability_raw = dict(
        mapping.get(
            "observability",
            {},
        )
    )

    tracing_raw = dict(
        observability_raw.pop(
            "tracing",
            {},
        )
    )

    if "gpu_indices" in observability_raw:
        observability_raw["gpu_indices"] = tuple(observability_raw["gpu_indices"])

    config = ProductionConfig(
        schema_version=int(
            mapping.get(
                "schema_version",
                1,
            )
        ),
        service=ServiceConfig(**service_raw),
        health=HealthConfig(**health_raw),
        observability=ObservabilityConfig(
            **observability_raw,
            tracing=TracingConfig(**tracing_raw),
        ),
    )

    config.validate()
    return config


def _apply_environment(
    mapping: dict,
    environment: Mapping[str, str],
) -> dict:
    result = json.loads(json.dumps(mapping))

    service = result.setdefault(
        "service",
        {},
    )

    health = result.setdefault(
        "health",
        {},
    )

    observability = result.setdefault(
        "observability",
        {},
    )

    tracing = observability.setdefault(
        "tracing",
        {},
    )

    env_map = {
        "LLMFORGE_SERVICE_NAME": (
            service,
            "name",
            str,
        ),
        "LLMFORGE_CONTROL_HOST": (
            service,
            "host",
            str,
        ),
        "LLMFORGE_CONTROL_PORT": (
            service,
            "port",
            int,
        ),
        "LLMFORGE_UPSTREAM_BASE_URL": (
            service,
            "upstream_base_url",
            str,
        ),
        "LLMFORGE_REQUEST_TIMEOUT_S": (
            service,
            "request_timeout_s",
            float,
        ),
        "LLMFORGE_SHUTDOWN_GRACE_S": (
            service,
            "shutdown_grace_s",
            float,
        ),
        "LLMFORGE_HEALTH_UPSTREAM_REQUIRED": (
            health,
            "upstream_required",
            _parse_bool,
        ),
        "LLMFORGE_HEALTH_GPU_REQUIRED": (
            health,
            "gpu_required",
            _parse_bool,
        ),
        "LLMFORGE_HEALTH_UPSTREAM_TIMEOUT_S": (
            health,
            "upstream_timeout_s",
            float,
        ),
        "LLMFORGE_OBSERVABILITY_INTERVAL_S": (
            observability,
            "sample_interval_s",
            float,
        ),
        "LLMFORGE_GPU_INDICES": (
            observability,
            "gpu_indices",
            lambda value: list(_parse_gpu_indices(value)),
        ),
        "LLMFORGE_SCRAPE_VLLM_METRICS": (
            observability,
            "scrape_vllm_metrics",
            _parse_bool,
        ),
        "LLMFORGE_SCRAPE_GPU_METRICS": (
            observability,
            "scrape_gpu_metrics",
            _parse_bool,
        ),
        "LLMFORGE_LOG_LEVEL": (
            observability,
            "log_level",
            str,
        ),
        "LLMFORGE_TRACING_MODE": (
            tracing,
            "mode",
            str,
        ),
        "LLMFORGE_TRACE_JSONL_PATH": (
            tracing,
            "jsonl_path",
            str,
        ),
        "LLMFORGE_TRACE_QUEUE_SIZE": (
            tracing,
            "queue_size",
            int,
        ),
        "LLMFORGE_TRACE_FLUSH_EVERY": (
            tracing,
            "flush_every",
            int,
        ),
        "OTEL_EXPORTER_OTLP_ENDPOINT": (
            tracing,
            "otlp_endpoint",
            str,
        ),
    }

    for name, (
        target,
        key,
        parser,
    ) in env_map.items():
        if name not in environment:
            continue

        target[key] = parser(environment[name])

    return result


def load_production_config(
    path: Path | None = None,
    *,
    environment: Mapping[
        str,
        str,
    ]
    | None = None,
) -> ProductionConfig:
    base = ProductionConfig().to_dict()

    if path is not None:
        file_payload = json.loads(path.read_text(encoding="utf-8"))

        def merge(
            left: dict,
            right: dict,
        ) -> dict:
            result = dict(left)

            for key, value in right.items():
                if isinstance(value, dict) and isinstance(
                    result.get(key),
                    dict,
                ):
                    result[key] = merge(
                        result[key],
                        value,
                    )
                else:
                    result[key] = value

            return result

        base = merge(
            base,
            file_payload,
        )

    environment = os.environ if environment is None else environment

    base = _apply_environment(
        base,
        environment,
    )

    return _from_mapping(base)
