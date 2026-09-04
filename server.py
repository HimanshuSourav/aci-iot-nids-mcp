#!/usr/bin/env python3
"""IoT anomaly MCP server.

Stdio MCP host (Cursor, Claude Desktop, MCP Inspector) → tools that POST
to the canonical FastAPI apps. No joblib load, no second preprocessor.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from mcp.server import MCPServer

import client
from schemas import TIERS, schema_for

mcp = MCPServer(
    "iot-anomaly",
    instructions=(
        "Classify ACI-IoT-style network flows via two existing FastAPI apps. "
        "Call list_tiers first. Use predict_full_cic only when the client has "
        "CICFlowMeter-style columns; use predict_overlap for conntrack / "
        "OpenWrt R7000 fields. Do not invent missing CIC features. This "
        "server proxies HTTP — start the APIs or tools return an error dict."
    ),
)


@mcp.tool()
def list_tiers() -> Dict[str, Any]:
    """Describe the two inference tiers and which predict tool to call."""
    return {
        "tiers": [schema_for(name) for name in TIERS],
        "note": (
            "full_cic needs the cloud-full-cic API. overlap needs the "
            "r7000-overlap API. Neither is started by this MCP server."
        ),
    }


@mcp.tool()
def describe_schema(tier: str) -> Dict[str, Any]:
    """Return required/optional fields for full_cic or overlap."""
    try:
        return schema_for(tier)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}


@mcp.tool()
def health(tier: str = "all") -> Dict[str, Any]:
    """Ping /health and /ready on one tier (full_cic, overlap) or all."""
    if tier in ("all", "*", ""):
        return {name: client.health(name) for name in TIERS}
    if tier not in TIERS:
        return {"ok": False, "error": f"Unknown tier '{tier}'. Known: {sorted(TIERS)}"}
    return client.health(tier)


@mcp.tool()
def predict_full_cic(
    features: Dict[str, Any],
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Predict using the cloud-full-cic FastAPI /predict (76-feat bundle).

    Args:
        features: Flow record. Minimum keys: Total Fwd Packet, Total Bwd
            packets, Total Length of Fwd Packet, Total Length of Bwd Packet,
            Flow Duration. Send the full bundle feature list when you have it.
        model_name: Optional model key (default random_forest).
    """
    return client.predict("full_cic", features, model_name=model_name)


@mcp.tool()
def predict_overlap(
    features: Dict[str, Any],
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Predict using the r7000-overlap FastAPI /predict (conntrack fields).

    Args:
        features: Conntrack-emittable flow. Required: Protocol, Flow Duration,
            Total Fwd Packet, Total Bwd packets, Total Length of Fwd Packet,
            Total Length of Bwd Packet. Rates optional.
        model_name: Optional model key (default xgboost).
    """
    return client.predict("overlap", features, model_name=model_name)


@mcp.resource("iot-anomaly://tiers")
def resource_tiers() -> Dict[str, Any]:
    """Static description of both inference tiers."""
    return list_tiers()


@mcp.resource("iot-anomaly://schema/{tier}")
def resource_schema(tier: str) -> Dict[str, Any]:
    """Feature contract for a single tier."""
    return describe_schema(tier)


@mcp.prompt()
def classify_conntrack_flow() -> str:
    """Agent instructions for an OpenWrt / conntrack record."""
    return (
        "You have a conntrack-derived flow. Call describe_schema with "
        "tier='overlap', then predict_overlap with only those fields. "
        "Do not call predict_full_cic. If health(overlap) is not ready, "
        "tell the user to start r7000-overlap with: "
        "uvicorn serve:app --port 8001"
    )


if __name__ == "__main__":
    mcp.run()
