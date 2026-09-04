# IoT Anomaly MCP

Real [Model Context Protocol](https://modelcontextprotocol.io) server for the
ACI-IoT anomaly thread. An agent calls tools; this process **HTTP-proxies**
the two canonical FastAPI apps. It does **not** load joblib models.

```
Cursor / other MCP host
        │ stdio
        ▼
   server.py (this repo)
        │ HTTP
        ├── IOT_ANOMALY_FULL_CIC_URL  →  cloud-full-cic deploy_api :8000
        └── IOT_ANOMALY_OVERLAP_URL   →  r7000-overlap serve.py    :8001
```

The 2025 FastAPI files that claimed `mcp_version` are in [`legacy/`](legacy/)
(ISS-09). Do not serve them.

## Tools

| Tool | Purpose |
|------|---------|
| `list_tiers` | Cloud full-CIC vs R7000 overlap, URLs, default models |
| `describe_schema` | Required / optional fields for one tier |
| `health` | `/health` + `/ready` (one tier or `all`) |
| `predict_full_cic` | POST `cloud-full-cic` `/predict` |
| `predict_overlap` | POST `r7000-overlap` `/predict` |

Resources: `iot-anomaly://tiers`, `iot-anomaly://schema/{tier}`.

## Setup

```bash
cd /home/hsourav/ml/cloud-full-cic/mcp
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/pytest
```

Start the APIs in other terminals (from their repos):

```bash
# full CIC (honest bundle)
cd /home/hsourav/ml/cloud-full-cic
# use that project's venv / image
uvicorn deploy_api:app --host 127.0.0.1 --port 8000

# overlap
cd /home/hsourav/ml/r7000-overlap
.venv/bin/uvicorn serve:app --host 127.0.0.1 --port 8001
```

## Cursor

Copy [`cursor.mcp.json.example`](cursor.mcp.json.example) into your Cursor
MCP settings (merge the `iot-anomaly` block). Paths assume this machine.

```bash
.venv/bin/python server.py   # stdio; silence is correct
```

Env:

| Variable | Default |
|----------|---------|
| `IOT_ANOMALY_FULL_CIC_URL` | `http://127.0.0.1:8000` |
| `IOT_ANOMALY_OVERLAP_URL` | `http://127.0.0.1:8001` |
| `IOT_ANOMALY_HTTP_TIMEOUT` | `10` |

If an API is down, predict/health tools return `{ok: false, ...}` instead of
crashing the MCP session.

## What this is not

- Not a replacement for Docker `deploy_api:app`
- Not on-device TinyML (`r7000-overlap/tinyml/`)
- Not in-process inference (optional later; keep it out of the scaffold)
