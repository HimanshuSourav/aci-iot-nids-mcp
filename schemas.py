"""Feature contracts the MCP tools expose to agents.

These describe the *canonical* FastAPI payloads. The MCP server does not
invent columns or impute a 76-feature matrix from five fields.
"""

from __future__ import annotations

from typing import Any, Dict, List

FULL_CIC_REQUIRED: List[str] = [
    "Total Fwd Packet",
    "Total Bwd packets",
    "Total Length of Fwd Packet",
    "Total Length of Bwd Packet",
    "Flow Duration",
]

FULL_CIC_NOTES = (
    "cloud-full-cic /predict accepts these five as the documented minimum, "
    "plus any extra ACI/CIC columns the loaded bundle lists in "
    "input_feature_names. Accuracy numbers (~99.9% / macro-F1 ~0.91) are only "
    "valid when the client sends the full honest-protocol feature set. "
    "Default model: random_forest. Default URL: IOT_ANOMALY_FULL_CIC_URL "
    "(http://127.0.0.1:8000)."
)

OVERLAP_REQUIRED: List[str] = [
    "Protocol",
    "Flow Duration",
    "Total Fwd Packet",
    "Total Bwd packets",
    "Total Length of Fwd Packet",
    "Total Length of Bwd Packet",
]

OVERLAP_OPTIONAL: List[str] = [
    "Flow Bytes/s",
    "Flow Packets/s",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Down/Up Ratio",
    "Average Packet Size",
    "Connection Type",
]

OVERLAP_NOTES = (
    "r7000-overlap /predict: conntrack-emittable fields only. Rates are "
    "recomputed from core counters when omitted. Holdout XGBoost is ~86.5% "
    "on ACI overlap columns — not a live-router number. Default model: "
    "xgboost. Default URL: IOT_ANOMALY_OVERLAP_URL (http://127.0.0.1:8001)."
)

TIERS: Dict[str, Dict[str, Any]] = {
    "full_cic": {
        "name": "full_cic",
        "role": "Cloud / lab — full CICFlowMeter feature set",
        "repo": "cloud-full-cic (GitHub: anomaly-detection-docker)",
        "serve": "uvicorn deploy_api:app --port 8000",
        "default_url_env": "IOT_ANOMALY_FULL_CIC_URL",
        "default_url": "http://127.0.0.1:8000",
        "default_model": "random_forest",
        "required_fields": FULL_CIC_REQUIRED,
        "optional_fields": [],
        "notes": FULL_CIC_NOTES,
        "predict_tool": "predict_full_cic",
    },
    "overlap": {
        "name": "overlap",
        "role": "Edge-feasible — OpenWrt R7000 conntrack overlap",
        "repo": "r7000-overlap",
        "serve": "uvicorn serve:app --port 8001",
        "default_url_env": "IOT_ANOMALY_OVERLAP_URL",
        "default_url": "http://127.0.0.1:8001",
        "default_model": "xgboost",
        "required_fields": OVERLAP_REQUIRED,
        "optional_fields": OVERLAP_OPTIONAL,
        "notes": OVERLAP_NOTES,
        "predict_tool": "predict_overlap",
    },
}

FULL_CIC_EXAMPLE: Dict[str, float] = {
    "Total Fwd Packet": 10.0,
    "Total Bwd packets": 8.0,
    "Total Length of Fwd Packet": 1500.0,
    "Total Length of Bwd Packet": 1200.0,
    "Flow Duration": 1000000.0,
}

OVERLAP_EXAMPLE: Dict[str, Any] = {
    "Protocol": 6,
    "Flow Duration": 379933,
    "Total Fwd Packet": 11,
    "Total Bwd packets": 11,
    "Total Length of Fwd Packet": 720.0,
    "Total Length of Bwd Packet": 6169.0,
    "Connection Type": "wired",
}


def schema_for(tier: str) -> Dict[str, Any]:
    key = tier.replace("-", "_")
    if key not in TIERS:
        known = sorted(TIERS)
        raise ValueError(f"Unknown tier '{tier}'. Known: {known}")
    info = dict(TIERS[key])
    info["example"] = FULL_CIC_EXAMPLE if key == "full_cic" else OVERLAP_EXAMPLE
    return info
