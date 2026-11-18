// E2E support proxy: one HTTP entry point for Playwright.
// /api/<name>  -> the matching Python endpoint server port
// everything else (pages, static assets) -> the Next.js server on :3000
import http from "http";

const PY_PORTS = {
  "/api/score": 8600,
  "/api/explain": 8601,
  "/api/analyze": 8602,
  "/api/recommend": 8603,
  "/api/simulate": 8604,
  "/api/analyst": 8605,
  "/api/predict": 8606,
};
const NEXT_PORT = 3000;
const LISTEN_PORT = 3100;

http
  .createServer((req, res) => {
    const path = (req.url || "/").split("?")[0];
    const targetPort = PY_PORTS[path] ?? NEXT_PORT;
    const headers = { ...req.headers, host: `127.0.0.1:${targetPort}` };
    const upstream = http.request(
      { host: "127.0.0.1", port: targetPort, method: req.method, path: req.url, headers },
      (upstreamRes) => {
        res.writeHead(upstreamRes.statusCode || 502, upstreamRes.headers);
        upstreamRes.pipe(res);
      },
    );
    upstream.on("error", (error) => {
      res.writeHead(502, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: `upstream ${targetPort} unavailable: ${error.message}` }));
    });
    if (req.method === "GET" || req.method === "HEAD") upstream.end();
    else req.pipe(upstream);
  })
  .listen(LISTEN_PORT, "127.0.0.1", () => console.log(`proxy ready on 127.0.0.1:${LISTEN_PORT}`));
