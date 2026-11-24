import { fireEvent, screen, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { liveStore } from "../state/liveStore";
import { makeAlert } from "../test/fixtures";
import { mockFetch, renderWith, resetStore } from "../test/utils";
import DashboardPage from "./DashboardPage";

const camera = { camera_id: "c1", name: "Main pool", monitoring: "MONITORED" as const, stalled: false, lagging: false, muted_zones: [] };

describe("Dashboard alarm clock and alarm stack", () => {
  beforeEach(() => {
    resetStore();
    liveStore.set({ cameras: [camera] });
  });
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("stacks the alarms on the page and lists every one under the alarm clock", async () => {
    mockFetch({
      "GET /api/v1/alerts": () => ({
        body: [makeAlert({ id: 5 }), makeAlert({ id: 6, scenario: "FAULT", fault_code: "CAMERA_STALLED" }), makeAlert({ id: 7 })],
      }),
    });
    renderWith(<DashboardPage />);
    const bell = await screen.findByRole("button", { name: "Active alarms: 3" });
    expect(screen.getByTestId("alarm-bell-badge")).toHaveTextContent("3");
    // the page: only the oldest alarm (with its Acknowledge), the rest behind it, no list and no captions
    const stack = screen.getByRole("region", { name: "Alarm stack" });
    expect(within(stack).getByTestId("alarm-5")).toBeInTheDocument();
    expect(within(stack).getByRole("button", { name: "Acknowledge" })).toBeInTheDocument();
    expect(screen.queryByTestId("alarm-6")).toBeNull();
    expect(stack).toHaveTextContent("+2 more alarms waiting");
    expect(screen.queryByRole("region", { name: "Active alarms" })).toBeNull();
    // the stack's button opens the list under the alarm clock
    fireEvent.click(within(stack).getByRole("button", { name: "Open alert list (3)" }));
    const list = screen.getByRole("region", { name: "Active alarms" });
    expect(within(list).getByTestId("alarm-5")).toBeInTheDocument();
    expect(within(list).getByTestId("alarm-6")).toHaveTextContent("FAULT CAMERA_STALLED");
    expect(within(list).getByTestId("alarm-7")).toBeInTheDocument();
    expect(screen.queryByRole("region", { name: "Alarm stack" })).toBeNull(); // never two copies of a card
    expect(bell).toHaveAttribute("aria-expanded", "true");
    fireEvent.keyDown(window, { key: "Escape" });
    expect(screen.queryByRole("region", { name: "Active alarms" })).toBeNull();
  });

  it("shows nothing about alarms on the page when none is waiting; the caption is under the alarm clock", async () => {
    mockFetch({ "GET /api/v1/alerts": () => ({ body: [] }) });
    renderWith(<DashboardPage />);
    const bell = await screen.findByRole("button", { name: "Active alarms: 0" });
    expect(screen.queryByTestId("alarm-bell-badge")).toBeNull();
    expect(screen.queryByText(/No unacknowledged alarms/)).toBeNull();
    expect(screen.queryByRole("region", { name: "Alarm stack" })).toBeNull();
    fireEvent.click(bell);
    expect(await screen.findByTestId("no-alarms")).toHaveTextContent("No unacknowledged alarms.");
    fireEvent.click(screen.getByRole("button", { name: "Close alarm list" }));
    expect(screen.queryByTestId("no-alarms")).toBeNull();
  });

  it("puts the cameras in a two-column grid", () => {
    mockFetch({ "GET /api/v1/alerts": () => ({ body: [] }) });
    renderWith(<DashboardPage />);
    expect(screen.getByRole("region", { name: "Cameras" })).toHaveClass("dash-cams");
  });
});
