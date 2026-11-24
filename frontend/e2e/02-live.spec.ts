/**
 * Live pipeline walkthrough (E2E_LIVE=1, CI `-Gpu`): the real detector on CUDA analyses the
 * `web_lane_pool_1080p` file camera registered by 01-operator. Replaces Phase 6 section 3.1 steps 2-6
 * (incl. the zone drawn on the live frame and MUTED), 7 (camera overlay) and 8 (camera MONITORED again).
 */
import type { Page } from "@playwright/test";

import { api, clickAt, expect, go, LIVE, login, shot, test } from "./fixtures";

test.describe.configure({ mode: "serial" });
test.skip(!LIVE, "set E2E_LIVE=1 (CI -Gpu): needs the GPU live runtime");

/** Draw "Lap lanes" over the pool on the camera's current (live) frame, optionally with a schedule. */
async function drawLapLanes(page: Page, schedule: string | null): Promise<void> {
  await go(page, "Zones");
  const frame = page.getByRole("img", { name: "Current camera frame" });
  await expect.poll(() => frame.evaluate((el) => (el as HTMLImageElement).naturalWidth), { timeout: 30_000 }).toBeGreaterThan(0);
  await clickAt(page, "Zone drawing area", [[0.05, 0.25], [0.95, 0.25], [0.95, 0.95], [0.05, 0.95]]);
  await page.getByLabel("Name").fill("Lap lanes");
  if (schedule !== null) await page.getByLabel(/^Schedule/).selectOption({ label: schedule });
  await page.getByRole("button", { name: "Save zone" }).click();
  await expect(page.locator(".zone-list")).toContainText(`Lap lanes (WATER`);
}

async function liveCameraId(page: Page): Promise<string> {
  const tile = page.locator('[data-testid^="camera-"]:not([data-testid^="camera-overlay-"])').first();
  await expect(tile).toBeVisible({ timeout: 120_000 });
  const id = await tile.getAttribute("data-testid");
  if (id === null) throw new Error("camera tile without id");
  return id.slice("camera-".length);
}

test("start monitoring: the annotated Model View streams and the camera is MONITORED", async ({ page }) => {
  await login(page);
  await go(page, "Cameras");
  const row = page.getByRole("row", { name: /Main pool/ });
  await expect(row.getByRole("cell", { name: /^(running|stopped)( - \d+ people)?$/ })).toBeVisible({ timeout: 60_000 });
  // 01-operator restarted the API: enabled cameras resume by themselves on startup (no click needed then)
  const start = row.getByRole("button", { name: "Start monitoring" });
  if (await start.isVisible()) await start.click();
  await expect(row.getByRole("cell", { name: "running" })).toBeVisible({ timeout: 120_000 });
  const id = await liveCameraId(page);
  const tile = page.getByTestId(`camera-${id}`);
  await expect(tile.getByText("MONITORED", { exact: true })).toBeVisible({ timeout: 60_000 });
  const img = tile.getByRole("img", { name: "Live view of Main pool" });
  await expect.poll(() => img.evaluate((el) => (el as HTMLImageElement).naturalWidth), { timeout: 60_000 }).toBeGreaterThan(0);
  await shot(page, "02-monitored");
});

test("with a WATER zone drawn on the live frame, an audible alarm escalates, shows its evidence (snapshot + WebM) and leaves the list on acknowledge", async ({ page }) => {
  await login(page);
  await drawLapLanes(page, null);
  await page.getByRole("navigation").getByRole("link", { name: "Dashboard", exact: true }).click();
  await page.getByRole("button", { name: /^Active alarms/ }).click(); // the full list is under the alarm clock
  await expect(page.getByRole("region", { name: "Active alarms" })).toBeVisible();
  const alarm = page.locator('article[data-testid^="alarm-"]:not(.alarm-fault)').first();
  await expect(alarm).toBeVisible({ timeout: 180_000 });
  const alarmId = await alarm.getAttribute("data-testid");
  await expect(alarm.locator('[data-testid^="escalated-"]')).toBeVisible({ timeout: 60_000 });
  await shot(page, "04-alarm-escalated");

  await expect(alarm.getByRole("button", { name: "Evidence" })).toBeVisible({ timeout: 90_000 });
  // the recording is written after the post-roll: reopen the viewer until it is there (or fail)
  const video = page.getByTestId("recording");
  const deadline = Date.now() + 120_000;
  for (;;) {
    await alarm.getByRole("button", { name: "Evidence" }).click();
    const snap = page.getByRole("dialog").getByRole("img", { name: /snapshot/ });
    await expect.poll(() => snap.evaluate((el) => (el as HTMLImageElement).naturalWidth), { timeout: 30_000 }).toBeGreaterThan(0);
    if ((await video.count()) > 0 || Date.now() > deadline) break;
    await page.getByRole("dialog").getByRole("button", { name: "Close" }).click();
    await page.waitForTimeout(5_000);
  }
  // WebM / VP8 decodes in Chromium: it must reach HAVE_CURRENT_DATA and play
  await expect(video).toHaveCount(1);
  await expect.poll(() => video.evaluate((v) => (v as HTMLVideoElement).readyState), { timeout: 30_000 }).toBeGreaterThanOrEqual(2);
  await video.evaluate((v) => (v as HTMLVideoElement).play());
  await expect.poll(() => video.evaluate((v) => (v as HTMLVideoElement).currentTime), { timeout: 15_000 }).toBeGreaterThan(0);
  await shot(page, "05-evidence");
  await page.getByRole("dialog").getByRole("button", { name: "Close" }).click();

  await alarm.getByRole("button", { name: "Acknowledge" }).click();
  await expect(page.locator(`[data-testid="${String(alarmId)}"]`)).toBeHidden({ timeout: 30_000 });
});

test("the same zone with an off-hours schedule is MUTED within seconds, without a camera restart", async ({ page }) => {
  await login(page);
  await go(page, "Zones");
  page.once("dialog", (d) => void d.accept());
  await page.locator(".zone-list li", { hasText: "Lap lanes" }).getByRole("button", { name: "Delete" }).click();
  await expect(page.locator(".zone-list li", { hasText: "Lap lanes" })).toHaveCount(0);
  await drawLapLanes(page, "Off hours");
  // the schedule's window is 12 h away: ordinary alerts in the zone are silent; submersion alerts and faults still sound
  await expect(page.getByTestId("muted-banner")).toContainText("Lap lanes", { timeout: 60_000 });
  await expect(page.getByTestId("muted-banner")).toContainText("Submersion / sudden blackout alerts and faults always sound");
  await shot(page, "03-04-zone-on-live-frame-muted");
});

test("killing the API shows NOT MONITORING; after restart the camera is MONITORED again", async ({ page }) => {
  await login(page);
  const id = await liveCameraId(page);
  await api("kill");
  try {
    await expect(page.getByTestId("not-monitoring")).toBeVisible({ timeout: 20_000 });
    await shot(page, "06-live-not-monitoring");
  } finally {
    await api("start");
  }
  await expect(page.getByTestId("not-monitoring")).toBeHidden({ timeout: 180_000 });
  await expect(page.getByTestId(`camera-${id}`).getByText("MONITORED", { exact: true })).toBeVisible({ timeout: 120_000 });
  await shot(page, "07-live-recovered");

  page.on("dialog", (d) => void d.accept());
  await go(page, "Cameras");
  await page.getByRole("button", { name: "Stop monitoring" }).first().click();
  await expect(page.getByRole("row", { name: /Main pool/ }).getByRole("cell", { name: "stopped" })).toBeVisible({ timeout: 60_000 });
});
