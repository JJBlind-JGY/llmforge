#!/usr/bin/env python3
"""Run the M6 observability sidecar around an existing vLLM server."""

from __future__ import annotations

import argparse
import logging
import threading
from pathlib import Path

from llmforge.production import (
    load_production_config,
)
from llmforge.production.control_plane import (
    ObservabilityControlPlane,
)
from llmforge.production.health import (
    GpuHealthCheck,
    HealthEvaluator,
    UpstreamHealthCheck,
)
from llmforge.production.lifecycle import (
    GracefulShutdown,
)
from llmforge.telemetry.collectors.gpu import (
    GpuMetricsCollector,
)
from llmforge.telemetry.collectors.sampler import (
    PeriodicCollectorRunner,
)
from llmforge.telemetry.collectors.vllm import (
    VLLMMetricsCollector,
)
from llmforge.telemetry.logging import (
    configure_json_logging,
)
from llmforge.telemetry.metrics import (
    create_default_registry,
)

_LOGGER = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/production/m6_observability.json"),
    )

    args = parser.parse_args()

    config = load_production_config(args.config)

    configure_json_logging(level=(config.observability.log_level))

    registry = create_default_registry()

    collectors = []

    if config.observability.scrape_vllm_metrics:
        collectors.append(
            VLLMMetricsCollector(
                base_url=(config.service.upstream_base_url),
                timeout_s=(config.health.upstream_timeout_s),
            )
        )

    if config.observability.scrape_gpu_metrics:
        collectors.append(
            GpuMetricsCollector(gpu_indices=(config.observability.gpu_indices))
        )

    sampler = PeriodicCollectorRunner(
        registry=registry,
        collectors=collectors,
        interval_s=(config.observability.sample_interval_s),
    )

    checks = [
        UpstreamHealthCheck(
            base_url=(config.service.upstream_base_url),
            timeout_s=(config.health.upstream_timeout_s),
            required=(config.health.upstream_required),
        ),
    ]

    if config.health.gpu_required or config.observability.scrape_gpu_metrics:
        checks.append(
            GpuHealthCheck(
                gpu_indices=(config.observability.gpu_indices),
                required=(config.health.gpu_required),
            )
        )

    health = HealthEvaluator(
        checks=checks,
        failure_threshold=(config.health.failure_threshold),
    )

    plane = ObservabilityControlPlane(
        host=(config.service.host),
        port=(config.service.port),
        registry=registry,
        health=health,
        metadata={
            "name": (config.service.name),
            "upstream": (config.service.upstream_base_url),
        },
    )

    shutdown = GracefulShutdown()
    shutdown.install_signal_handlers()

    server_thread = threading.Thread(
        target=plane.serve_forever,
        name="llmforge-control-plane",
        daemon=True,
    )

    sampler.start()
    sampler.collect_once()

    server_thread.start()

    _LOGGER.info(
        "observability sidecar started",
        extra={
            "component": "control_plane",
            "host": (config.service.host),
            "port": (config.service.port),
        },
    )

    try:
        while not shutdown.wait(0.5):
            pass
    finally:
        _LOGGER.info(
            "observability sidecar stopping",
            extra={
                "component": ("control_plane"),
            },
        )

        plane.shutdown()
        sampler.stop(timeout_s=(config.service.shutdown_grace_s))

        server_thread.join(timeout=(config.service.shutdown_grace_s))


if __name__ == "__main__":
    main()
