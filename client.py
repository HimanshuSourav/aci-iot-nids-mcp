"""Thin HTTP client for the canonical FastAPI serve paths.

This module is the only place the MCP server talks to models. It does not
load joblib artifacts. Missing backends surface as ApiError, not crashes.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Mapping, Optional

from schemas import TIERS

DEFAULT_TIMEOUT_S = float(os.environ.get("IOT_ANOMALY_HTTP_TIMEOUT", "10"))


class ApiError(Exception):
    """HTTP or connectivity failure talking to a serve API."""

    def __init__(self, status: int, detail: str, url: str = "") -> None:
        self.status = status
        self.detail = detail
        self.url = url
        where = f" ({url})" if url else ""
        super().__init__(f"HTTP {status}: {detail}{where}")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ok": False,
            "error": self.detail,
            "status": self.status,
            "url": self.url,
        }


def url_for(tier: str) -> str:
    info = TIERS[tier]
    return os.environ.get(info["default_url_env"], info["default_url"]).rstrip("/")


def request_json(
    method: str,
    url: str,
    body: Optional[Mapping[str, Any]] = None,
    timeout: float = DEFAULT_TIMEOUT_S,
) -> Dict[str, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(dict(body)).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(raw)
            detail = parsed.get("detail", raw)
        except json.JSONDecodeError:
            detail = raw or exc.reason
        raise ApiError(exc.code, str(detail), url) from exc
    except urllib.error.URLError as exc:
        raise ApiError(0, f"API unreachable: {exc.reason}", url) from exc


def health(tier: str) -> Dict[str, Any]:
    base = url_for(tier)
    try:
        live = request_json("GET", f"{base}/health")
        try:
            ready = request_json("GET", f"{base}/ready")
        except ApiError as exc:
            ready = exc.as_dict()
        return {"ok": True, "tier": tier, "url": base, "health": live, "ready": ready}
    except ApiError as exc:
        payload = exc.as_dict()
        payload.update({"tier": tier})
        return payload


def predict(
    tier: str,
    features: Mapping[str, Any],
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    if tier not in TIERS:
        raise ValueError(f"Unknown tier '{tier}'. Known: {sorted(TIERS)}")
    base = url_for(tier)
    url = f"{base}/predict"
    if model_name:
        url += "?" + urllib.parse.urlencode({"model_name": model_name})
    try:
        result = request_json("POST", url, body=features)
    except ApiError as exc:
        payload = exc.as_dict()
        payload.update({"tier": tier, "model_name": model_name})
        return payload
    result = dict(result)
    result.setdefault("ok", True)
    result["tier"] = tier
    result["url"] = url
    return result
