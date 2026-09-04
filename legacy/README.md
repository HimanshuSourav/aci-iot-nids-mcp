# Legacy FastAPI wrapper (not MCP)

These files are the original 2025 “MCP-compliant” experiment. They are a
**FastAPI inference app**, not a [Model Context Protocol](https://modelcontextprotocol.io)
server. They also diverge from the canonical `cloud-full-cic` serve path
(`deploy_api.py` + `model_bundle.py`) — see ISS-09.

Do **not** point Cursor, Docker, or CI at this directory.

The real MCP scaffold lives one level up (`server.py`).
