/** Shared helpers for the browser end-to-end specs. The admin credentials are test values for the
 * throwaway data dir tools/e2e_stack.py creates per run; they exist nowhere else. */
import { readFileSync } from "node:fs";

import { test as base, expect, type Page } from "@playwright/test";

/** Every spec uses this `test`: an uncaught error in the page fails the test that caused it. */
export const test = base.extend<{ pageErrors: undefined }>({
  pageErrors: [
    async ({ page }, use) => {
      const errors: string[] = [];
      page.on("pageerror", (e) => errors.push(`${e.name}: ${e.message}
${e.stack ?? ""}`));
      await use(undefined);
      expect(errors, "uncaught errors in the page").toEqual([]);
    },
    { auto: true },
  ],
});
export { expect };

/** [Phase 6] The test-only login lives in one tracked file, shared with tools/e2e_stack.py. */
const FIXTURE = JSON.parse(readFileSync(new URL("./admin.fixture.json", import.meta.url), "utf-8")) as {
  username: string;
  password: string;
};
export const ADMIN = { username: FIXTURE.username, password: FIXTURE.password } as const;
export const CONTROL_URL = "http://127.0.0.1:8199";
export const LIVE = process.env["E2E_LIVE"] === "1";
/** The RTSP test camera tools/e2e_stack.py publishes (MediaMTX + FFmpeg looping the eval clip). */
export const CAMERA_URL = "rtsp://127.0.0.1:18554/cam0";

/** Crash ("kill") or restart ("start") the API process of the stack. */
export async function api(action: "kill" | "start"): Promise<void> {
  const r = await fetch(`${CONTROL_URL}/api/${action}`, { method: "POST" });
  expect(r.status, await r.text()).toBe(200);
}

export async function login(page: Page): Promise<void> {
  await page.goto("/");
  await page.getByLabel("Username").fill(ADMIN.username);
  await page.getByLabel("Password").fill(ADMIN.password);
  await page.getByRole("button", { name: "Log in" }).click();
  await expect(page.getByTestId("arm-audio-gate")).toBeVisible();
  await page.getByRole("button", { name: "Arm audio and start monitoring" }).click();
  await expect(page.getByTestId("arm-audio-gate")).toBeHidden();
}

/** Click past the NOT MONITORING full-screen overlay if it is up right now (cheap: no forced wait). It comes
 * back every COLLAPSE_MS (2 min) without E2E_LIVE, since no camera can start on the CPU detector, so any test
 * running longer than that must clear it again before its next click - `go()` does this on every navigation. */
export async function dismissOverlayIfUp(page: Page): Promise<void> {
  const overlay = page.getByTestId("not-monitoring");
  if (await overlay.isVisible()) {
    await page.getByRole("button", { name: /I am supervising the pool/ }).click();
    await expect(overlay).toBeHidden();
  }
}

/**
 * Without E2E_LIVE no camera can start (the CPU detector is below the admission rate), so NOT MONITORING is
 * (correctly) up and covers the controls. Do what the operator manual says an operator does: confirm direct
 * supervision, which shrinks it to the red bar for 2 min. Waits out the overlay's own startup race (it needs
 * up to WS_GRACE_MS + one status poll to appear after the page loads), unlike `dismissOverlayIfUp`.
 */
export async function supervise(page: Page): Promise<void> {
  await page.waitForTimeout(6000); // the overlay may need up to WS_GRACE_MS + one status poll to appear
  await dismissOverlayIfUp(page);
}

/** The pages that live under Settings (opened from the Settings page's own section list). */
const SETTINGS_SECTIONS = ["Zones", "Pool calibration", "Schedules", "System health", "Users", "Audit log"];

/** Navigate with the nav bar and wait for the (lazily loaded) page's own heading before touching it.
 * Settings sections are reached through the Settings nav entry, then the section list. Clears the NOT
 * MONITORING overlay first if it has come back (long specs can outlast its 2-minute collapse). */
export async function go(page: Page, link: string, heading: string = link): Promise<void> {
  await dismissOverlayIfUp(page);
  if (SETTINGS_SECTIONS.includes(link)) {
    await page.getByRole("navigation").getByRole("link", { name: "Settings", exact: true }).click();
    await page.getByRole("group", { name: "Settings sections" }).getByRole("link", { name: link, exact: true }).click();
  } else {
    await page.getByRole("navigation").getByRole("link", { name: link, exact: true }).click();
  }
  await expect(page.getByRole("heading", { level: 1, name: heading, exact: true })).toBeVisible();
}

/** Click at fractions (0-1) of an element's box - for the SVG zone / calibration editors. */
export async function clickAt(page: Page, testLabel: string, pts: readonly (readonly [number, number])[]): Promise<void> {
  const area = page.getByRole("img", { name: testLabel });
  const box = await area.boundingBox();
  if (box === null) throw new Error(`${testLabel} has no box`);
  for (const [x, y] of pts) await area.click({ position: { x: x * box.width, y: y * box.height } });
}

export async function shot(page: Page, name: string): Promise<void> {
  await page.screenshot({ path: test.info().outputPath(`${name}.png`), fullPage: true });
}
