import { fireEvent, screen, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { Role, User } from "../api/types";
import { AuthContext } from "../authModel";
import { HealthStrip } from "../components/HealthStrip";
import { mockFetch, renderWith, resetStore } from "../test/utils";
import AccountPage from "./AccountPage";
import UsersPage from "./UsersPage";

function asUser(role: Role, children: ReactNode) {
  const user: User = { id: 1, username: "site.admin", role, is_active: true, created_at: "2026-10-08T00:00:00Z" };
  const value = { user, loading: false, login: vi.fn(), logout: vi.fn() };
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

function type(label: RegExp, value: string) {
  fireEvent.change(screen.getByLabelText(label), { target: { value } });
}

describe("AccountPage (change my password)", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("sends current + new password and confirms", async () => {
    const calls = mockFetch({ "POST /api/v1/auth/password": () => ({ status: 204 }) });
    renderWith(asUser("lifeguard", <AccountPage />));
    type(/^current password/i, "Old-Password-2026!");
    type(/^new password/i, "New-Password-2026!");
    type(/^repeat new password/i, "New-Password-2026!");
    fireEvent.click(screen.getByRole("button", { name: "Change password" }));
    expect(await screen.findByRole("status")).toHaveTextContent(/password changed/i);
    const call = calls.find((c) => c.url.endsWith("/auth/password"));
    expect(call?.body).toEqual({ current_password: "Old-Password-2026!", new_password: "New-Password-2026!" });
  });

  it.each([
    ["too short", "short", "short", /at least 12/],
    ["mismatch", "New-Password-2026!", "New-Password-2027!", /do not match/],
    ["unchanged", "Old-Password-2026!", "Old-Password-2026!", /must differ/],
  ])("refuses %s before calling the server", (_n, next, repeat, msg) => {
    const calls = mockFetch({});
    renderWith(asUser("admin", <AccountPage />));
    type(/^current password/i, "Old-Password-2026!");
    type(/^new password/i, next);
    type(/^repeat new password/i, repeat);
    fireEvent.click(screen.getByRole("button", { name: "Change password" }));
    expect(screen.getByRole("alert")).toHaveTextContent(msg);
    expect(calls).toHaveLength(0);
  });

  it("explains a wrong current password and a policy rejection", async () => {
    mockFetch({ "POST /api/v1/auth/password": () => ({ status: 403, body: { detail: "invalid_current_password" } }) });
    renderWith(asUser("admin", <AccountPage />));
    type(/^current password/i, "Wrong-Password-2026!");
    type(/^new password/i, "New-Password-2026!");
    type(/^repeat new password/i, "New-Password-2026!");
    fireEvent.click(screen.getByRole("button", { name: "Change password" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("The current password is wrong.");
  });
});

describe("UsersPage (admin resets another account)", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("resets the chosen user's password", async () => {
    const calls = mockFetch({
      "GET /api/v1/users": () => ({
        body: [
          { id: 1, username: "site.admin", role: "admin", is_active: true, created_at: "2026-10-08T00:00:00Z" },
          { id: 7, username: "deck.lead", role: "supervisor", is_active: true, created_at: "2026-10-08T00:00:00Z" },
        ],
      }),
      "POST /api/v1/users/7/password": () => ({ status: 204 }),
    });
    renderWith(asUser("admin", <UsersPage />));
    await screen.findByRole("option", { name: "deck.lead (supervisor)" });
    fireEvent.change(screen.getByLabelText(/^user$/i), { target: { value: "7" } });
    const form = screen.getByRole("form", { name: "Reset a password" });
    fireEvent.change(form.querySelector("input[type=password]") as HTMLInputElement, { target: { value: "Reset-Password-2026!" } });
    fireEvent.click(screen.getByRole("button", { name: "Reset password" }));
    await waitFor(() => {
      expect(calls.some((c) => c.method === "POST" && c.url.endsWith("/users/7/password"))).toBe(true);
    });
    expect(calls.find((c) => c.url.endsWith("/users/7/password"))?.body).toEqual({ new_password: "Reset-Password-2026!" });
    expect(await screen.findByText(/all of that user's sessions were signed out/i)).toBeInTheDocument();
  });
});

describe("HealthStrip (dashboard)", () => {
  beforeEach(() => {
    resetStore();
  });
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("says ACTION REQUIRED and why when the supervisor is down", async () => {
    mockFetch({
      "GET /api/v1/live/status": () => ({
        body: { enabled: true, running: true, provider: "cuda", supervisor_down: true, cameras: [], not_started: {}, scheduler: null },
      }),
      "GET /api/v1/alerts": () => ({ body: [] }),
    });
    renderWith(asUser("lifeguard", <HealthStrip />));
    const strip = await screen.findByRole("status", { name: "System health" });
    await waitFor(() => {
      expect(strip).toHaveTextContent("ACTION REQUIRED");
    });
    expect(strip).toHaveTextContent(/Supervisor service: DOWN/);
    expect(screen.queryByRole("link", { name: "Details" })).toBeNull(); // lifeguards have no System page
  });

  it("closes to a one-line chip that still names the level, and opens again", async () => {
    sessionStorage.clear();
    mockFetch({
      "GET /api/v1/live/status": () => ({
        body: { enabled: true, running: true, provider: "cuda", supervisor_down: true, cameras: [], not_started: {}, scheduler: null },
      }),
      "GET /api/v1/alerts": () => ({ body: [] }),
    });
    renderWith(asUser("supervisor", <HealthStrip />));
    // every status answered first: a problem that appears AFTER closing re-opens the strip (by design)
    await waitFor(() => {
      expect(screen.getByRole("status", { name: "System health" })).toHaveTextContent(/Supervisor service: DOWN/);
    });
    fireEvent.click(screen.getByRole("button", { name: "Close system health message" }));
    const strip = screen.getByRole("status", { name: "System health" });
    expect(strip).toHaveTextContent("ACTION REQUIRED"); // never invisible: the level stays on screen
    expect(strip).toHaveTextContent(/problems? hidden/);
    expect(strip).not.toHaveTextContent(/Supervisor service: DOWN/);
    fireEvent.click(screen.getByRole("button", { name: "Show" }));
    expect(screen.getByRole("status", { name: "System health" })).toHaveTextContent(/Supervisor service: DOWN/);
  });

  it("is never green while the status is unknown", async () => {
    mockFetch({ "GET /api/v1/live/status": () => ({ status: 500, body: { detail: "boom" } }), "GET /api/v1/alerts": () => ({ body: [] }) });
    renderWith(asUser("supervisor", <HealthStrip />));
    const strip = await screen.findByRole("status", { name: "System health" });
    await waitFor(() => {
      expect(strip).toHaveTextContent("STATUS UNKNOWN");
    });
    expect(strip).not.toHaveTextContent("All systems OK");
  });
});
