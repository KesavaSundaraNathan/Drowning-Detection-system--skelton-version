import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";

import { api } from "../api/endpoints";
import type { Alert } from "../api/types";
import { ongoingSeconds } from "../alarmModel";
import { VisualAlarm } from "./VisualAlarm";

const alarm = {
  id: 7, created_at: new Date().toISOString(), camera_id: "cam-1", scenario: "DROWNING", alert_reason: "Active drowning",
  priority: "Critical", severity: "critical", risk_score: 90, zone_name: "Pool", reason: "Drowning Alert: Active drowning — x [ONGOING 8 s]", status: "pending",
  audible: true, escalation_level: 0, incident_id: null, global_id: null,
} as unknown as Alert;

function show(alerts: Alert[]) {
  vi.spyOn(api, "alerts").mockResolvedValue(alerts);
  vi.spyOn(api, "cameras").mockResolvedValue([]);
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <MemoryRouter>
        <VisualAlarm />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

describe("VisualAlarm", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("reads the ongoing duration from the alert reason", () => {
    expect(ongoingSeconds("Active Drowning: x [ONGOING 8 s]")).toBe(8);
    expect(ongoingSeconds("Active Drowning: x")).toBeNull();
  });

  it("shows a flashing frame and a bar with Acknowledge while a drowning alarm is unacknowledged", async () => {
    show([alarm]);
    const bar = await screen.findByTestId("visual-alarm-bar");
    expect(bar).toHaveTextContent("DROWNING ALARM");
    expect(bar).toHaveTextContent("ONGOING 8 s");
    expect(screen.getByRole("button", { name: "Acknowledge" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "View camera" })).toHaveAttribute("href", "/cameras/cam-1");
    // the frame must never take clicks from the live feed under it
    expect(screen.getByTestId("visual-alarm-frame")).toHaveAttribute("aria-hidden", "true");
  });

  it("labels an alarm whose camera was deleted as a removed camera, with no camera link", async () => {
    show([{ ...alarm, camera_id: null }]);
    const bar = await screen.findByTestId("visual-alarm-bar");
    expect(bar).toHaveTextContent("removed camera");
    expect(screen.queryByRole("link", { name: "View camera" })).toBeNull();
  });

  it("shows nothing without alarms, and nothing for a FAULT", async () => {
    show([{ ...alarm, scenario: "FAULT" }]);
    await waitFor(() => {
      expect(api.alerts).toHaveBeenCalled();
    });
    expect(screen.queryByTestId("visual-alarm-bar")).toBeNull();
  });
});
