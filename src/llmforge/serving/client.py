from __future__ import annotations

import json
import time
from dataclasses import dataclass

from .schema import RequestResult, RequestSpec


@dataclass(frozen=True)
class ClientConfig:
    base_url: str
    model: str
    endpoint: str = "/v1/completions"
    timeout_s: float = 180.0


class VLLMOpenAIClient:
    def __init__(self, config: ClientConfig) -> None:
        self.config = config
        self._client = None

    @staticmethod
    def _httpx():
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError(
                "Serving benchmark requires httpx; install the serving extra."
            ) from exc
        return httpx

    async def __aenter__(self):
        httpx = self._httpx()
        self._client = httpx.AsyncClient(
            base_url=self.config.base_url,
            timeout=self.config.timeout_s,
        )
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def health(self) -> None:
        if self._client is None:
            raise RuntimeError("client is not open")
        response = await self._client.get("/health")
        response.raise_for_status()

    async def run_request(
        self, request: RequestSpec, *, scheduled_due_perf: float
    ) -> RequestResult:
        if self._client is None:
            raise RuntimeError("client is not open")

        dispatch = time.perf_counter()
        arrivals: list[float] = []
        status_code = None
        payload = {
            "model": self.config.model,
            "prompt": list(request.prompt_token_ids),
            "max_tokens": request.output_tokens,
            "min_tokens": request.output_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "stream": True,
            "stream_interval": 1,
            "ignore_eos": True,
            "add_special_tokens": False,
            "skip_special_tokens": False,
            "return_token_ids": True,
            "request_id": request.request_id,
        }

        try:
            async with self._client.stream(
                "POST", self.config.endpoint, json=payload
            ) as response:
                status_code = response.status_code
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    if raw == "[DONE]":
                        break
                    event = json.loads(raw)
                    for choice in event.get("choices", []):
                        token_ids = choice.get("token_ids") or []
                        if token_ids:
                            now = time.perf_counter()
                            arrivals.extend([now] * len(token_ids))

            end = time.perf_counter()
            if len(arrivals) != request.output_tokens:
                raise RuntimeError(
                    f"expected {request.output_tokens} output tokens, "
                    f"observed {len(arrivals)}"
                )

            ttft = (arrivals[0] - dispatch) * 1000.0
            e2e = (end - dispatch) * 1000.0
            tpot = (e2e - ttft) / (len(arrivals) - 1) if len(arrivals) > 1 else None
            itls = [(b - a) * 1000.0 for a, b in zip(arrivals, arrivals[1:])]

            return RequestResult(
                request_id=request.request_id,
                prompt_tokens=request.prompt_tokens,
                requested_output_tokens=request.output_tokens,
                output_tokens=len(arrivals),
                scheduled_offset_s=request.arrival_offset_s,
                client_queue_ms=max(0.0, (dispatch - scheduled_due_perf) * 1000.0),
                ttft_ms=ttft,
                e2e_ms=e2e,
                tpot_ms=tpot,
                itl_ms=itls,
                success=True,
                status_code=status_code,
                metadata=request.metadata,
            )
        except Exception as exc:
            end = time.perf_counter()
            return RequestResult(
                request_id=request.request_id,
                prompt_tokens=request.prompt_tokens,
                requested_output_tokens=request.output_tokens,
                output_tokens=len(arrivals),
                scheduled_offset_s=request.arrival_offset_s,
                client_queue_ms=max(0.0, (dispatch - scheduled_due_perf) * 1000.0),
                ttft_ms=None,
                e2e_ms=(end - dispatch) * 1000.0,
                tpot_ms=None,
                itl_ms=[],
                success=False,
                error=f"{type(exc).__name__}: {exc}",
                status_code=status_code,
                metadata=request.metadata,
            )
