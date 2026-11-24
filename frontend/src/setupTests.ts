import "@testing-library/jest-dom/vitest";

import { cleanup, configure } from "@testing-library/react";
import { afterEach, vi } from "vitest";

// The 1 s default is tight for the lazily loaded route chunks on a shared CI runner (App.test.tsx's Settings
// redirect passed locally and timed out in CI); a passing wait still returns at once.
configure({ asyncUtilTimeout: 5000 });

// vitest runs without globals, so Testing Library's automatic cleanup is not registered
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});
