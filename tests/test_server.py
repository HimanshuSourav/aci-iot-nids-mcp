from __future__ import annotations

from typing import Any

import pytest
from mcp import Client

from schemas import FULL_CIC_EXAMPLE, OVERLAP_REQUIRED
from server import mcp


def _payload(result: Any) -> dict[str, Any]:
    body = result.structured_content
    if isinstance(body, dict) and "result" in body and isinstance(body["result"], dict):
        return body["result"]
    return body


@pytest.fixture()
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_list_tools() -> None:
    async with Client(mcp) as c:
        tools = await c.list_tools()
        names = {t.name for t in tools.tools}
        assert {
            "list_tiers",
            "describe_schema",
            "health",
            "predict_full_cic",
            "predict_overlap",
        } <= names


@pytest.mark.anyio
async def test_describe_schema_overlap() -> None:
    async with Client(mcp) as c:
        result = await c.call_tool("describe_schema", {"tier": "overlap"})
        body = _payload(result)
        assert body["name"] == "overlap"
        assert body["required_fields"] == OVERLAP_REQUIRED


@pytest.mark.anyio
async def test_describe_schema_unknown() -> None:
    async with Client(mcp) as c:
        result = await c.call_tool("describe_schema", {"tier": "wifi"})
        assert _payload(result)["ok"] is False


@pytest.mark.anyio
async def test_predict_full_cic_unreachable() -> None:
    async with Client(mcp) as c:
        result = await c.call_tool(
            "predict_full_cic",
            {"features": FULL_CIC_EXAMPLE},
        )
        body = _payload(result)
        if body.get("ok") is False:
            assert "error" in body
        else:
            assert "prediction" in body


@pytest.mark.anyio
async def test_resources() -> None:
    async with Client(mcp) as c:
        listed = await c.list_resources()
        uris = {str(r.uri) for r in listed.resources}
        assert "iot-anomaly://tiers" in uris
