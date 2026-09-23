"""Small M6 control plane exposing liveness, readiness, and normalized metrics."""

from __future__ import annotations

import json
from http.server import (
    BaseHTTPRequestHandler,
    ThreadingHTTPServer,
)

from llmforge.telemetry.metrics import (
    MetricRegistry,
)

from .health import (
    HealthEvaluator,
)


class ObservabilityControlPlane:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        registry: MetricRegistry,
        health: HealthEvaluator,
        metadata: dict | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.registry = registry
        self.health = health
        self.metadata = dict(metadata or {})

        registry_ref = registry
        health_ref = health
        metadata_ref = self.metadata

        class Handler(BaseHTTPRequestHandler):
            def _write_json(
                self,
                status: int,
                payload: dict,
            ) -> None:
                body = json.dumps(
                    payload,
                    sort_keys=True,
                ).encode("utf-8")

                self.send_response(status)

                self.send_header(
                    "Content-Type",
                    "application/json",
                )

                self.send_header(
                    "Content-Length",
                    str(len(body)),
                )

                self.end_headers()
                self.wfile.write(body)

            def do_GET(
                self,
            ) -> None:
                if self.path == "/health/live":
                    self._write_json(
                        200,
                        {
                            "status": "alive",
                        },
                    )
                    return

                if self.path == "/health/ready":
                    report = health_ref.evaluate()

                    self._write_json(
                        (200 if report.ready else 503),
                        report.to_dict(),
                    )
                    return

                if self.path == "/metrics":
                    body = registry_ref.prometheus_text().encode("utf-8")

                    self.send_response(200)

                    self.send_header(
                        "Content-Type",
                        ("text/plain; version=0.0.4; charset=utf-8"),
                    )

                    self.send_header(
                        "Content-Length",
                        str(len(body)),
                    )

                    self.end_headers()
                    self.wfile.write(body)
                    return

                if self.path == "/status":
                    self._write_json(
                        200,
                        {
                            "service": (metadata_ref),
                            "metrics": (registry_ref.snapshot()),
                        },
                    )
                    return

                self._write_json(
                    404,
                    {
                        "error": "not_found",
                    },
                )

            def log_message(
                self,
                format: str,
                *args,
            ) -> None:
                del format
                del args

        self._server = ThreadingHTTPServer(
            (
                self.host,
                self.port,
            ),
            Handler,
        )

    def serve_forever(
        self,
    ) -> None:
        self._server.serve_forever(poll_interval=0.5)

    def shutdown(
        self,
    ) -> None:
        self._server.shutdown()
        self._server.server_close()
