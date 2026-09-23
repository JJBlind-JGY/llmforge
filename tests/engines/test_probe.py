from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from llmforge.engines.probe import probe_engine


def _make_response(*, status: int, body: bytes) -> MagicMock:
    resp = MagicMock()
    resp.status = status
    resp.read.return_value = body
    resp.__enter__ = lambda self: self
    resp.__exit__ = lambda self, *a: False
    return resp


class _FakeAdapter:
    name = "fake"

    def health_paths(self):
        return ("/health", "/v1/models")


def test_probe_health_empty_body_ok():
    def fake_urlopen(url, timeout=None):
        if url.endswith("/health"):
            return _make_response(status=200, body=b"")
        if url.endswith("/v1/models"):
            return _make_response(
                status=200,
                body=json.dumps({"data": [{"id": "qwen3-8b"}]}).encode(),
            )
        raise AssertionError(url)

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        result = probe_engine(
            adapter=_FakeAdapter(),
            base_url="http://127.0.0.1:8000",
        )

    assert result.ready
    assert result.health_status == 200
    assert result.model_status == 200
    assert result.models == ("qwen3-8b",)
    assert result.error is None
