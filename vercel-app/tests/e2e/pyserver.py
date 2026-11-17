"""E2E support server: serves the six Vercel Python endpoints on fixed local ports.

Playwright's proxy (tests/e2e/proxy.mjs) forwards /api/* to these ports, so the
browser tests exercise the real serverless handler code end-to-end.

Ports: score 8600, explain 8601, analyze 8602, recommend 8603, simulate 8604, analyst 8605.
"""
import sys
import threading
from http.server import ThreadingHTTPServer

sys.path.insert(0, "api")
sys.path.insert(0, ".")
import score, explain, analyze, recommend, simulate, analyst  # noqa: E402

ENDPOINTS = {
    "/api/score": (8600, score.handler),
    "/api/explain": (8601, explain.handler),
    "/api/analyze": (8602, analyze.handler),
    "/api/recommend": (8603, recommend.handler),
    "/api/simulate": (8604, simulate.handler),
    "/api/analyst": (8605, analyst.handler),
}

servers = []
for path, (port, handler_cls) in ENDPOINTS.items():
    server = ThreadingHTTPServer(("127.0.0.1", port), handler_cls)
    servers.append(server)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"{path} -> 127.0.0.1:{port}", flush=True)
print("READY", flush=True)

try:
    threading.Event().wait()
except KeyboardInterrupt:
    for server in servers:
        server.shutdown()
