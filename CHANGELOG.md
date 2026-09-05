# Changelog

## 0.1.0 — 2026-09-04

Replace the 2025 FastAPI “MCP-compliant” wrapper with a real stdio
[Model Context Protocol](https://modelcontextprotocol.io) server.

### Why

The old `deploy_fastapi.py` / `mcp_wrapper.py` were a second inference
stack (different schema, leftover `models/*.joblib`). ISS-09 isolated
this repo so it would not dual-maintain preprocessors. Agents still
needed a way to call the hardened APIs.

### What agents get

| Tool | Action |
|------|--------|
| `list_tiers` | full_cic vs overlap |
| `describe_schema` | required / optional JSON fields |
| `health` | `/health` + `/ready` |
| `predict_full_cic` | POST cloud-full-cic `:8000/predict` |
| `predict_overlap` | POST edge-r7000 `:8001/predict` |

Resources: `iot-anomaly://tiers`, `iot-anomaly://schema/{tier}`.

The server does **not** load joblib. Swap models by replacing the
FastAPI bundle (`CURRENT`); swap JSON fields in `schemas.py` + the
API; add invoke backends behind the same tool names.

### Layout

- `server.py`, `client.py`, `schemas.py` — MCP surface
- `legacy/` — parked FastAPI wrapper (not production)
- `cursor.mcp.json.example` — Cursor host snippet


### Not in this commit

Live Cursor MCP config, Docker entrypoint changes, in-process
inference, TinyML. Those stay follow-ups.
