import { fireEvent, screen, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { CSRF_COOKIE } from "../api/client";
import type { AlertEvent, Role, User } from "../api/types";
import { AuthContext } from "../authModel";
import { mockFetch, renderWith, resetStore } from "../test/utils";
import { EventCenter } from "./EventCenter";

function asUser(role: Role, children: ReactNode) {
  const user: User = { id: 1, username: "pool.staff", role, is_active: true, created_at: "2026-10-08T00:00:00Z" };
  return <AuthContext.Provider value={{ user, loading: false, login: vi.fn(), logout: vi.fn() }}>{children}</AuthContext.Provider>;
}

function makeEvent(over: Partial<AlertEvent> = {}): AlertEvent {
  return {
    id: 5,
    camera_id: "c1",
    track_id: 3,
    opened_at: new Date(Date.now() - 20_000).toISOString(),
    updated_at: new Date().toISOString(),
    closed_at: null,
    severity: "critical",
    scenario: "DROWNING",
    title: "Drowning Alert",
    alert_reason: "Submersion / sudden blackout",
    status: "open",
    alert_count: 2,
    first_alert_id: 11,
    last_alert_id: 12,
    acknowledged_by: null,
    acknowledged_at: null,
    resolved_by: null,
    resolved_at: null,
    ...over,
  };
}

describe("EventCenter (Phase 5)", () => {
  beforeEach(() => {
    resetStore();
    document.cookie = `${CSRF_COOKIE}=csrf-xyz; path=/; Secure`;
  });
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("lists open events by severity and acknowledges one event", async () => {
    let acked = false;
    const calls = mockFetch({
      "GET /api/v1/events": () => ({
        body: [makeEvent({ status: acked ? "acknowledged" : "open" }), makeEvent({ id: 6, alert_reason: "Active drowning", alert_count: 1 })],
      }),
      "POST /api/v1/events/5/acknowledge": () => {
        acked = true;
        return { body: makeEvent({ status: "acknowledged" }) };
      },
    });
    renderWith(asUser("lifeguard", <EventCenter />));
    const row = await screen.findByTestId("event-5");
    expect(row).toHaveTextContent("Drowning Alert · Submersion / sudden blackout");
    expect(row).toHaveTextContent("2 alerts");
    expect(screen.getByTestId("event-6")).toHaveTextContent("Drowning Alert · Active drowning");
    expect(screen.queryByLabelText("Resolution note for event 5")).toBeNull(); // lifeguards cannot resolve
    fireEvent.click(row.querySelector("button.btn-ack") as HTMLButtonElement);
    await waitFor(() => {
      expect(calls.some((c) => c.method === "POST" && c.url === "/api/v1/events/5/acknowledge")).toBe(true);
    });
    await waitFor(() => {
      expect(screen.getByTestId("event-5")).toHaveTextContent("acknowledged");
    });
  });

  it("lets a supervisor resolve an event with a note", async () => {
    const calls = mockFetch({
      "GET /api/v1/events": () => ({ body: [makeEvent()] }),
      "POST /api/v1/events/5/resolve": () => ({ body: makeEvent({ status: "resolved" }) }),
    });
    renderWith(asUser("supervisor", <EventCenter />));
    const note = await screen.findByLabelText("Resolution note for event 5");
    const resolve = screen.getByRole("button", { name: "Resolve" });
    expect(resolve).toBeDisabled(); // a note is required
    fireEvent.change(note, { target: { value: "rescued" } });
    fireEvent.click(resolve);
    await waitFor(() => {
      const post = calls.find((c) => c.method === "POST");
      expect(post?.url).toBe("/api/v1/events/5/resolve");
    });
  });

  it("says when nothing is open", async () => {
    mockFetch({ "GET /api/v1/events": () => ({ body: [] }) });
    renderWith(asUser("lifeguard", <EventCenter />));
    expect(await screen.findByTestId("no-events")).toBeInTheDocument();
  });
});
