"""Optional OpenTelemetry OTLP/HTTP bridge.

This module imports OpenTelemetry lazily. The core LLMForge M6 package therefore
remains usable without the optional observability extra.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OpenTelemetryHandle:
    tracer_provider: object
    tracer: object

    def shutdown(self) -> None:
        shutdown = getattr(
            self.tracer_provider,
            "shutdown",
            None,
        )

        if callable(shutdown):
            shutdown()


def configure_otlp_tracing(
    *,
    service_name: str,
    endpoint: str,
) -> OpenTelemetryHandle:
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import (
            SERVICE_NAME,
            Resource,
        )
        from opentelemetry.sdk.trace import (
            TracerProvider,
        )
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
        )
    except ImportError as exc:
        raise RuntimeError(
            "OTLP tracing requires the optional "
            "OpenTelemetry packages. See "
            "M6_PYPROJECT_MERGE.md."
        ) from exc

    resource = Resource.create(
        {
            SERVICE_NAME: service_name,
        }
    )

    provider = TracerProvider(resource=resource)

    exporter = OTLPSpanExporter(endpoint=endpoint)

    provider.add_span_processor(BatchSpanProcessor(exporter))

    trace.set_tracer_provider(provider)

    tracer = trace.get_tracer(service_name)

    return OpenTelemetryHandle(
        tracer_provider=provider,
        tracer=tracer,
    )
