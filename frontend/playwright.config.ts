/**
 * Browser end-to-end tests (Phase 7): the Phase 6 real-browser walkthrough, automated.
 *
 *   npm run e2e              operator flows against the real API (demo profile, detector on CPU)
 *   E2E_LIVE=1 npm run e2e   + the live pipeline on CUDA: monitoring, MUTED, alarm, evidence, ack
 *
 * Two web servers: tools/e2e_stack.py (the API on :18100 in a scratch data dir, plus a loopback control
 * endpoint on :8199 that crashes / restarts it) and `vite preview` of the production build on :5373,
 * which proxies /api there. [Phase 9b] Not :8000: on a machine with Life Guard installed, the real
 * LifeGuardApi service owns :8000 and the stack refused to start.
 * Chromium comes from build/ms-playwright (approved download, Phase 7), never from the user profile.
 */
import { defineConfig, devices } from "@playwright/test";
import path from "node:path";

import { ADMIN, CONTROL_URL } from "./e2e/fixtures";

const root = path.resolve(import.meta.dirname, "..");
process.env["PLAYWRIGHT_BROWSERS_PATH"] ??= path.join(root, "build", "ms-playwright");
const live = process.env["E2E_LIVE"] === "1";
const python = process.env["E2E_PYTHON"] ?? path.join(root, ".venv", "Scripts", "python.exe");
const API_PORT = 18100;

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false,
  workers: 1, // one shared stack: specs run in file order and share its database
  forbidOnly: true,
  retries: 0,
  timeout: 300_000, // an API restart re-measures capacity: ~70 s on the CPU provider
  expect: { timeout: 15_000 },
  outputDir: path.join(root, "build", "e2e-results"),
  reporter: [["list"], ["junit", { outputFile: path.join(root, "ci-artifacts", "playwright.xml") }]],
  use: {
    baseURL: "http://localhost:5373",
    screenshot: "on",
    trace: "retain-on-failure",
    video: "off",
    launchOptions: { args: ["--autoplay-policy=no-user-gesture-required"] },
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"], viewport: { width: 1440, height: 900 } } }],
  webServer: [
    {
      command: `"${python}" "${path.join(root, "tools", "e2e_stack.py")}" --api-port ${String(API_PORT)} --control-port 8199${live ? " --live" : ""}`,
      url: `${CONTROL_URL}/status`,
      timeout: 300_000,
      reuseExistingServer: false,
      stdout: "pipe",
      env: { E2E_ADMIN_USER: ADMIN.username, E2E_ADMIN_PASSWORD: ADMIN.password },
    },
    {
      // the production bundle (what Caddy serves), via vite preview: it inherits server.proxy for /api
      command: "npx vite build --logLevel warn && npx vite preview --port 5373 --strictPort",
      url: "http://localhost:5373",
      timeout: 120_000,
      reuseExistingServer: false,
      env: { LIFEGUARD_DEV_API_PORT: String(API_PORT) },
    },
  ],
});
