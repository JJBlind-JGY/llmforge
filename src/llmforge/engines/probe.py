"""Engine-agnostic readiness/model probe."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from .base import EngineAdapter, EngineProbeResult


def _get_json(
    url: str,
    *,
    timeout_s: float,
) -> tuple[int, dict]:
    with urllib.request.urlopen(
        url,
        timeout=timeout_s,
    ) as response:
        return (
            int(response.status),
            json.loads(response.read().decode("utf-8")),
        )


def _get_status(
    url: str,
    *,
    timeout_s: float,
) -> int:
    with urllib.request.urlopen(
        url,
        timeout=timeout_s,
    ) as response:
        return int(response.status)


def probe_engine(
    *,
    adapter: EngineAdapter,
    base_url: str,
    timeout_s: float = 3.0,
) -> EngineProbeResult:
    health_status = None
    model_status = None
    models: tuple[str, ...] = ()
    errors = []

    for path in adapter.health_paths():
        try:
            if path == "/health":
                health_status = _get_status(
                    base_url.rstrip("/") + path,
                    timeout_s=timeout_s,
                )

            elif path == "/v1/models":
                status, payload = _get_json(
                    base_url.rstrip("/") + path,
                    timeout_s=timeout_s,
                )
                model_status = status
                data = payload.get("data", [])
                models = tuple(
                    str(item["id"]) for item in data if item.get("id") is not None
                )

            else:
                # 未知 path：保守起见按 JSON probe 处理，或按 adapter 契约扩展
                status, _ = _get_json(
                    base_url.rstrip("/") + path,
                    timeout_s=timeout_s,
                )

        except (
            urllib.error.URLError,
            TimeoutError,
            OSError,
            json.JSONDecodeError,
        ) as exc:
            errors.append(f"{path}: {type(exc).__name__}: {exc}")

    ready = model_status is not None and 200 <= model_status < 300 and bool(models)

    if health_status is not None:
        ready = ready and (200 <= health_status < 300)

    return EngineProbeResult(
        engine=adapter.name,
        ready=ready,
        models=models,
        health_status=health_status,
        model_status=model_status,
        error=("; ".join(errors) if errors else None),
    )
