import { defineConfig, devices } from "@playwright/test";
import path from "path";

// E2E for the C.R.A.M. production app.
// Three servers are orchestrated:
//   1. the Python endpoint servers (the real Vercel handler code),
//   2. the built Next.js app,
//   3. a small proxy so the browser sees one origin with /api/* routed to Python.
const pythonReady = `${path.join("..", "venv", "Scripts", "python.exe")} tests/e2e/pyserver.py`;

export default defineConfig({
  testDir: "./tests/e2e",
  timeout: 90_000,
  expect: { timeout: 20_000 },
  fullyParallel: false,
  retries: process.env.CI ? 2 : 0,
  reporter: [["list"]],
  use: {
    baseURL: "http://127.0.0.1:3100",
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  webServer: [
    {
      command: pythonReady,
      url: "http://127.0.0.1:8600/api/score",
      reuseExistingServer: true,
      timeout: 60_000,
    },
    {
      command: "npm run build && npm run start",
      port: 3000,
      reuseExistingServer: true,
      timeout: 240_000,
    },
    {
      command: "node tests/e2e/proxy.mjs",
      url: "http://127.0.0.1:3100/api/score",
      reuseExistingServer: true,
      timeout: 30_000,
    },
  ],
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
});
