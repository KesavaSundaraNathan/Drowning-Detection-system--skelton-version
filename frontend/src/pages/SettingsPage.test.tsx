import { screen, waitFor } from "@testing-library/react";
import { Route, Routes } from "react-router-dom";
import { describe, expect, it } from "vitest";

import { AuthProvider } from "../auth";
import { mockFetch, renderWith } from "../test/utils";
import SettingsPage, { SettingsIndex } from "./SettingsPage";
import { SETTINGS_TABS } from "./settingsTabs";

const me = (role: string) => ({ id: 1, username: "pat", role, is_active: true, created_at: "" });

function tree() {
  return (
    <AuthProvider>
      <Routes>
        <Route path="/settings" element={<SettingsPage />}>
          <Route index element={<SettingsIndex />} />
          <Route path="zones" element={<p>zones page</p>} />
          <Route path="users" element={<p>users page</p>} />
        </Route>
      </Routes>
    </AuthProvider>
  );
}

describe("Settings", () => {
  it("lists the six sections the old side nav had", () => {
    expect(SETTINGS_TABS.map((t) => t.label)).toEqual(["Zones", "Pool calibration", "Schedules", "System health", "Users", "Audit log"]);
  });

  it("a supervisor sees the supervisor sections, not Users / Audit log, and lands on Zones", async () => {
    mockFetch({ "GET /api/v1/auth/me": () => ({ body: me("supervisor") }) });
    renderWith(tree(), undefined, "/settings");
    expect(await screen.findByText("zones page")).toBeInTheDocument();
    const group = screen.getByRole("group", { name: "Settings sections" });
    await waitFor(() => { expect(group).toHaveTextContent("System health"); });
    expect(group).not.toHaveTextContent("Users");
    expect(group).not.toHaveTextContent("Audit log");
  });

  it("an admin also sees Users and Audit log", async () => {
    mockFetch({ "GET /api/v1/auth/me": () => ({ body: me("admin") }) });
    renderWith(tree(), undefined, "/settings/users");
    expect(await screen.findByText("users page")).toBeInTheDocument();
    await waitFor(() => { expect(screen.getByRole("group", { name: "Settings sections" })).toHaveTextContent("Audit log"); });
  });
});
