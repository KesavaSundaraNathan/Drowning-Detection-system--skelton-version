import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

export default defineConfig({
  plugins: [react(), tailwindcss()], // [Phase 9b] Tailwind v4 restores the v2 look
  build: {
    sourcemap: false,
    manifest: true, // tools/check_bundle.mjs reads it: initial chunk <= 250 KB gzipped (CI)
    chunkSizeWarningLimit: 600,
  },
  server: {
    // development only: the API (uvicorn) on :8100 (rtmpose clone) (LIFEGUARD_DEV_API_PORT overrides it - the E2E stack uses
    // :18000 because an installed LifeGuardApi service owns :8000); production serves both via Caddy
    proxy: {
      "/api": { target: `http://127.0.0.1:${process.env["LIFEGUARD_DEV_API_PORT"] ?? "8100"}`, ws: true, changeOrigin: false },
    },
  },
  test: {
    environment: "jsdom",
    // https origin: the UI's __Host- CSRF cookie (Secure) must be settable in tests as in production
    environmentOptions: { jsdom: { url: "https://lifeguard.test/" } },
    setupFiles: ["./src/setupTests.ts"],
    include: ["src/**/*.test.{ts,tsx}"],
    restoreMocks: true,
  },
});
