/**
 * Operator walkthrough against the real API (demo profile). Runs on any machine: without E2E_LIVE=1 the
 * live runtime runs its detector on the CPU and no camera is started, so nothing here needs the GPU. Replaces Phase 6 section 3.1 steps 1, 3, 7, 8
 * and adds the Phase 7 pool calibration screen.
 */
import { ADMIN, api, CAMERA_URL, clickAt, expect, go, login, shot, test } from "./fixtures";

test.describe.configure({ mode: "serial" });

/** A one-hour schedule window 12 h from now: never active while the suite runs, so its zone is MUTED. */
export function offHours(now = new Date()): { from: string; to: string } {
  const h = (now.getHours() + 12) % 24;
  const pad = (n: number) => String(n).padStart(2, "0");
  return { from: `${pad(h)}:00`, to: `${pad(h)}:59` };
}

test("login refuses a wrong password, then requires arming audio", async ({ page }) => {
  await page.goto("/");
  await page.getByLabel("Username").fill(ADMIN.username);
  await page.getByLabel("Password").fill("not-the-password-1234");
  await page.getByRole("button", { name: "Log in" }).click();
  await expect(page.getByRole("alert")).toBeVisible();
  await expect(page.getByTestId("arm-audio-gate")).toBeHidden();

  await login(page);
  await expect(page.getByText(/ASSIST \/ PILOT/)).toHaveCount(0);
  await expect(page.getByRole("navigation").getByRole("link", { name: "Settings", exact: true })).toBeVisible();
  await shot(page, "01-armed-dashboard");
});

test("admin registers an RTSP camera, a schedule, and draws and deletes a scheduled zone", async ({ page }) => {
  await login(page);
  await go(page, "Cameras");
  await page.getByLabel("Name").fill("Main pool");
  await page.getByLabel(/RTSP URL/).fill(CAMERA_URL);
  await page.getByRole("button", { name: "Add", exact: true }).click();
  await expect(page.getByRole("cell", { name: "Main pool" })).toBeVisible();

  const { from, to } = offHours();
  await go(page, "Schedules");
  await page.getByLabel("Name").fill("Off hours");
  for (const d of ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]) await page.getByLabel(d, { exact: true }).check();
  await page.getByLabel("From").fill(from);
  await page.getByLabel("To").fill(to);
  await page.getByRole("button", { name: "Create" }).click();
  await expect(page.getByText("Off hours")).toBeVisible();

  await go(page, "Zones");
  await clickAt(page, "Zone drawing area", [[0.05, 0.25], [0.95, 0.25], [0.95, 0.95], [0.05, 0.95]]);
  await page.getByLabel("Name").fill("Lap lanes");
  await page.getByLabel(/^Schedule/).selectOption({ label: "Off hours" });
  await page.getByRole("button", { name: "Save zone" }).click();
  await expect(page.locator(".zone-list")).toContainText("Lap lanes (WATER");
  await shot(page, "03-zone-with-schedule");
  // deleted again: the live walkthrough starts unzoned (first alarm audible) and draws it on the live frame
  page.once("dialog", (d) => void d.accept());
  await page.locator(".zone-list li", { hasText: "Lap lanes" }).getByRole("button", { name: "Delete" }).click();
  await expect(page.locator(".zone-list li", { hasText: "Lap lanes" })).toHaveCount(0);
});

test("supervisor calibrates the pool; the projected outline and summary come back from the server", async ({ page }) => {
  await login(page);
  await go(page, "Pool calibration");
  await expect(page.getByText("Main pool: not calibrated")).toBeVisible();
  await page.getByLabel("Pool length (m)").fill("25");
  await page.getByLabel("Pool width (m)").fill("12.5");
  // 3 points on one line first: the server refuses it with a reason
  await clickAt(page, "Calibration area", [[0.1, 0.5], [0.3, 0.5], [0.6, 0.5], [0.9, 0.5]]);
  await page.getByRole("button", { name: "Save calibration" }).click();
  await expect(page.getByRole("alert")).toContainText("invalid_calibration");
  await page.getByRole("button", { name: "Clear points" }).click();

  await clickAt(page, "Calibration area", [[0.2, 0.3], [0.8, 0.3], [0.95, 0.9], [0.05, 0.9]]);
  await page.getByLabel(/Edge band/).fill("0.8");
  await page.getByRole("button", { name: "Save calibration" }).click();
  await expect(page.getByText(/Calibrated with 4 points; pool 25 x 12.5 m/)).toBeVisible();
  await expect(page.getByTestId("calibrated-outline")).toBeVisible();
  await expect(page.getByText("Main pool: calibrated (25 x 12.5 m)")).toBeVisible();
  await expect(page.getByText("0 point(s) placed")).toBeVisible(); // the form resets after a save
  await shot(page, "09-pool-calibration");

  await page.reload();
  await page.getByRole("button", { name: "Arm audio and start monitoring" }).click();
  await expect(page.getByText(/edge band 0.8 m/)).toBeVisible(); // persisted server-side

  // the corners above are made up: remove them again (the live walkthrough needs the real geometry)
  page.once("dialog", (d) => void d.accept());
  await page.getByRole("button", { name: "Remove calibration" }).click();
  await expect(page.getByText("Main pool: not calibrated")).toBeVisible();

  await go(page, "Audit log");
  await expect(page.getByRole("cell", { name: "camera.calibration.set" }).first()).toBeVisible();
  await expect(page.getByRole("cell", { name: "camera.calibration.delete" }).first()).toBeVisible();
});

test("NOT MONITORING covers the screen when the API dies and clears when it is back", async ({ page }) => {
  await login(page);
  await expect(page.getByTestId("not-monitoring")).toBeHidden();
  await api("kill");
  try {
    // the feed must be down > 5 s before the overlay: allow for that plus the reconnect probe
    await expect(page.getByTestId("not-monitoring")).toBeVisible({ timeout: 20_000 });
    await shot(page, "06-not-monitoring");
  } finally {
    await api("start");
  }
  await expect(page.getByTestId("not-monitoring")).toBeHidden({ timeout: 180_000 });
  await expect(page.getByTestId("arm-audio-gate")).toBeHidden(); // no reload needed: the app recovered by itself
  await shot(page, "07-recovered");
});
