from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from typing import Any, Dict

import pytest

import client


class _Handler(BaseHTTPRequestHandler):
    responses: Dict[str, Any] = {}

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: ARG002
        return

    def _send(self, code: int, body: Any) -> None:
        raw = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send(200, {"status": "ok"})
            return
        if self.path == "/ready":
            self._send(
                200,
                {"status": "ready", "models": ["random_forest"], "contract": "test"},
            )
            return
        self._send(404, {"detail": "nope"})

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        if self.path.startswith("/predict"):
            self._send(
                200,
                {
                    "prediction": "Benign",
                    "confidence": 0.91,
                    "class_probabilities": {"Benign": 0.91},
                    "model_used": payload.get("_unused", "random_forest"),
                },
            )
            return
        self._send(404, {"detail": "nope"})


@pytest.fixture()
def api_url(monkeypatch: pytest.MonkeyPatch) -> str:
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    monkeypatch.setenv("IOT_ANOMALY_FULL_CIC_URL", url)
    monkeypatch.setenv("IOT_ANOMALY_OVERLAP_URL", url)
    yield url
    server.shutdown()


def test_health_ok(api_url: str) -> None:
    result = client.health("full_cic")
    assert result["ok"] is True
    assert result["health"]["status"] == "ok"
    assert result["ready"]["status"] == "ready"
    assert result["url"] == api_url


def test_predict_ok(api_url: str) -> None:
    result = client.predict(
        "overlap",
        {
            "Protocol": 6,
            "Flow Duration": 5000,
            "Total Fwd Packet": 1,
            "Total Bwd packets": 1,
            "Total Length of Fwd Packet": 0,
            "Total Length of Bwd Packet": 0,
        },
        model_name="xgboost",
    )
    assert result["ok"] is True
    assert result["prediction"] == "Benign"
    assert "model_name=xgboost" in result["url"]


def test_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IOT_ANOMALY_FULL_CIC_URL", "http://127.0.0.1:9")
    result = client.health("full_cic")
    assert result["ok"] is False
    assert result["status"] == 0
