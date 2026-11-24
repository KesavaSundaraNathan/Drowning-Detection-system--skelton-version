/**
 * Phase 11 WP2: the client draws a zone ONCE on the pool plan and every calibrated camera of the pool uses it. The
 * test first registers and calibrates two cameras ("Deep end NW", "Deep end SW", disabled) on the 25 x 12.5 m pool
 * plan [Phase 5: this setup moved here from the removed cross-camera incident test]. Pool-plan zones are geometry
 * only; the cameras stay independent. Drawn with the mouse on the plan, previewed on both cameras before saving, saved, then the server's
 * own projection of the stored zone is checked for both cameras.
 */
import type { APIRequestContext, Page } from "@playwright/test";

import { CAMERA_URL, clickAt, expect, go, login, test } from "./fixtures";

const CAMS = ["Deep end NW", "Deep end SW"] as const;

async function csrf(page: Page): Promise<string> {
  const c = (await page.context().cookies()).find((k) => k.name === "__Host-lifeguard_csrf");
  if (!c) throw new Error("no CSRF cookie after login");
  return c.value;
}

async function calibratedCamera(req: APIRequestContext, token: string, name: string): Promise<void> {
  const headers = { "X-CSRF-Token": token };
  const r = await req.post("/api/v1/cameras", { headers, data: { name, source_url: CAMERA_URL, is_enabled: false } });
  expect(r.status(), await r.text()).toBe(201);
  const id = ((await r.json()) as { id: string }).id;
  const cal = await req.put(`/api/v1/cameras/${id}/calibration`, {
    headers,
    data: {
      image_points: [[0.2, 0.3], [0.8, 0.3], [0.95, 0.9], [0.05, 0.9]],
      pool_points: [[0, 0], [25, 0], [25, 12.5], [0, 12.5]], // the same pool plan for both cameras
    },
  });
  expect(cal.status(), await cal.text()).toBe(200);
}

// the plan shows the pool (25 x 12.5 m) plus 1.5 m of deck on every side: fractions of the 28 x 15.5 m box
const atMetres = (x: number, y: number) => [(x + 1.5) / 28, (y + 1.5) / 15.5] as const;
const TEACHING = [atMetres(18, 9.5), atMetres(24, 9.5), atMetres(24, 12), atMetres(18, 12)];

test("a supervisor draws the teaching platform once on the pool plan; both calibrated cameras get it", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 2000 }); // the whole editor in one evidence screenshot
  await login(page);
  const token = await csrf(page);
  for (const name of CAMS) await calibratedCamera(page.request, token, name);
  // the cameras query was cached at login, before the API calls above: reload (and re-arm audio, as after any reload)
  await page.reload();
  await page.getByRole("button", { name: "Arm audio and start monitoring" }).click();
  await expect(page.getByTestId("arm-audio-gate")).toBeHidden();
  await go(page, "Pool calibration");
  const section = page.getByRole("region", { name: "Pool zones" });
  await expect(section).toContainText("25 x 12.5 m");
  await clickAt(page, "Pool plan", TEACHING);
  await expect(page.getByTestId("plan-status")).toHaveText("4 corner(s), 15.0 m²");
  await expect(page.getByLabel("Corner 1 x (m)")).toHaveValue("18");
  await expect(page.getByLabel("Corner 3 y (m)")).toHaveValue("12");

  // live preview before saving: the zone as each calibrated camera will see it
  const previews = section.locator("figure.pool-preview");
  await expect(previews).toHaveCount(2);
  await expect(previews.filter({ hasText: "Deep end NW: zone in view" })).toHaveCount(1);
  await expect(previews.filter({ hasText: "Deep end SW: zone in view" })).toHaveCount(1);
  await expect(previews.first().locator("polygon")).toHaveCount(1);
  await section.screenshot({ path: test.info().outputPath("14a-pool-zone-preview.png") });

  await section.getByLabel("Name").fill("Teaching platform");
  await section.getByLabel("Type").selectOption("SHALLOW");
  await section.getByRole("button", { name: "Save pool zone" }).click();
  await expect(section.getByText(/Teaching platform \(SHALLOW, 4 corners, 15\.0 m²/)).toBeVisible();
  await expect(page.getByTestId("plan-status")).toHaveText("0 corner(s) - at least 3"); // the form resets
  await section.screenshot({ path: test.info().outputPath("14-pool-zone-editor.png") });

  // the stored zone, projected by the server into every calibrated camera of the pool
  const zones = (await (await page.request.get("/api/v1/pool-zones")).json()) as { id: number; name: string }[];
  const zone = zones.find((z) => z.name === "Teaching platform");
  expect(zone).toBeDefined();
  const proj = (await (await page.request.get(`/api/v1/pool-zones/${String(zone?.id)}/projections`)).json()) as {
    camera_name: string;
    polygon: number[][] | null;
  }[];
  expect(proj.map((p) => p.camera_name).sort()).toEqual(["Deep end NW", "Deep end SW"]);
  for (const p of proj) expect(p.polygon?.length ?? 0).toBeGreaterThanOrEqual(4);

  // the audit log records who drew it
  await go(page, "Audit log");
  await expect(page.getByRole("cell", { name: "zone.create" }).first()).toBeVisible();
});
