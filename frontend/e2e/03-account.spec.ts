/**
 * Phase 9 against the real API: the system-health verdict on the Dashboard and System health, and the password
 * lifecycle an operator depends on - create, change my own (My account), admin reset - each proven by signing
 * in again. The test user's passwords are test values for this throwaway data dir only.
 */
import type { Page } from "@playwright/test";
import { expect, go, login, shot, supervise, test } from "./fixtures";

test.describe.configure({ mode: "serial" });

const GUARD = { username: "e2e.guard", first: "Guard-First-Pass-2026!", second: "Guard-Second-Pass-2026!", reset: "Guard-Reset-Pass-2026!" };
const VERDICT = /All systems OK|DEGRADED|STATUS UNKNOWN|ACTION REQUIRED/;

async function signIn(page: Page, username: string, password: string): Promise<void> {
  await page.goto("/");
  await page.getByLabel("Username").fill(username);
  await page.getByLabel("Password").fill(password);
  await page.getByRole("button", { name: "Log in" }).click();
}

async function signInArmed(page: Page, username: string, password: string): Promise<void> {
  await signIn(page, username, password);
  await page.getByRole("button", { name: "Arm audio and start monitoring" }).click();
  await expect(page.getByTestId("arm-audio-gate")).toBeHidden();
  await supervise(page);
}

async function logOut(page: Page): Promise<void> {
  await page.getByRole("button", { name: "Log out" }).click();
  await expect(page.getByRole("button", { name: "Log in" })).toBeVisible();
}

test("System health shows one health verdict with its checks", async ({ page }) => {
  await login(page);
  await supervise(page);
  // [v4] the summary strip was removed from the Dashboard at the owner's request; the verdict lives on System health
  await expect(page.getByRole("status", { name: "System health" })).toHaveCount(0);
  await go(page, "System health");
  const overall = page.getByRole("region", { name: "Overall status" });
  await expect(overall).toContainText(VERDICT);
  const checks = page.getByRole("table", { name: "Health checks" });
  for (const label of ["Server connection", "Live monitoring", "Supervisor service", "Alert delivery", "System faults"]) {
    await expect(checks.getByRole("cell", { name: label, exact: true })).toBeVisible();
  }
  await shot(page, "03-system-health");

  // [Phase 9b] v2's theme switch: dark is applied to <html>, survives a reload, and switches back
  const dark = page.getByRole("switch", { name: "Dark theme" });
  await dark.click();
  await expect(page.locator("html")).toHaveClass(/\bdark\b/);
  await page.reload();
  await page.getByRole("button", { name: "Arm audio and start monitoring" }).click();
  await supervise(page);
  await expect(page.locator("html")).toHaveClass(/\bdark\b/);
  await shot(page, "03-system-health-dark");
  await go(page, "Dashboard", "Dashboard");
  await shot(page, "03-dashboard-dark");
  await page.getByRole("switch", { name: "Dark theme" }).click();
  await expect(page.locator("html")).not.toHaveClass(/\bdark\b/);
});

test("a user changes their own password; the old one stops working, the new one works", async ({ page }) => {
  await login(page);
  await supervise(page);
  await go(page, "Users");
  const create = page.locator("form", { has: page.getByRole("heading", { name: "New user" }) });
  await create.getByLabel("Username").fill(GUARD.username);
  await create.getByLabel(/^Password/).fill(GUARD.first);
  await create.getByLabel("Role").selectOption("lifeguard");
  await create.getByRole("button", { name: "Create" }).click();
  await expect(page.getByRole("cell", { name: GUARD.username })).toBeVisible();
  await logOut(page);

  await signInArmed(page, GUARD.username, GUARD.first);
  await page.getByRole("link", { name: `${GUARD.username} (lifeguard)` }).click();
  await expect(page.getByRole("heading", { level: 1, name: "My account" })).toBeVisible();
  const form = page.getByRole("form", { name: "Change password" });
  await form.getByLabel("Current password").fill("not-my-password-2026");
  await form.getByLabel(/^New password/).fill(GUARD.second);
  await form.getByLabel("Repeat new password").fill(GUARD.second);
  await form.getByRole("button", { name: "Change password" }).click();
  await expect(form.getByRole("alert")).toHaveText("The current password is wrong.");
  await form.getByLabel("Current password").fill(GUARD.first);
  await form.getByRole("button", { name: "Change password" }).click();
  await expect(form.getByRole("status")).toContainText("Password changed");
  await shot(page, "03-my-account-changed");
  await logOut(page);

  await signIn(page, GUARD.username, GUARD.first);
  await expect(page.getByRole("alert")).toBeVisible(); // old password refused
  await signInArmed(page, GUARD.username, GUARD.second);
  await logOut(page);
});

test("an admin resets a user's password and the user signs in with it", async ({ page }) => {
  await login(page);
  await supervise(page);
  await go(page, "Users");
  const reset = page.getByRole("form", { name: "Reset a password" });
  await reset.getByLabel("User").selectOption({ label: `${GUARD.username} (lifeguard)` });
  await reset.getByLabel(/^New password/).fill(GUARD.reset);
  await reset.getByRole("button", { name: "Reset password" }).click();
  await expect(reset.getByRole("status")).toContainText("Password reset");
  await shot(page, "03-admin-reset");
  await logOut(page);

  await signIn(page, GUARD.username, GUARD.second);
  await expect(page.getByRole("alert")).toBeVisible();
  await signInArmed(page, GUARD.username, GUARD.reset);
  await expect(page.getByRole("heading", { level: 1, name: "Dashboard" })).toBeVisible();
});
