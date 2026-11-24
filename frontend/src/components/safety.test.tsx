import { act, fireEvent, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { Alert } from "../api/types";
import type { ToneEngine } from "../audio/tones";
import { CSRF_COOKIE } from "../api/client";
import { liveStore } from "../state/liveStore";
import { makeAlert } from "../test/fixtures";
import { mockFetch, renderWith, resetStore } from "../test/utils";
import { ActiveAlarms } from "./alarms";
import { ArmAudioGate } from "./ArmAudioGate";
import { MutedBanner } from "./Banners";
import { CameraTile } from "./CameraTile";
import { MediaViewer } from "./MediaViewer";
import { NotMonitoringOverlay } from "./NotMonitoringOverlay";
import { SafetyAudio } from "./SafetyAudio";

const camera = { camera_id: "c1", name: "Main pool", monitoring: "MONITORED" as const, stalled: false, lagging: false, muted_zones: [] };

function fakeEngine(ok = true): ToneEngine & { calls: string[]; armCalls: () => number } {
  const calls: string[] = [];
  const arm = vi.fn(() => Promise.resolve(ok));
  return {
    calls,
    armCalls: () => arm.mock.calls.length,
    playing: null,
    arm,
    play: (t) => calls.push(`play:${t}`),
    stop: () => calls.push("stop"),
  };
}

beforeEach(() => {
  resetStore();
});
afterEach(() => {
  document.cookie = `${CSRF_COOKIE}=; expires=Thu, 01 Jan 1970 00:00:00 GMT; path=/; Secure`;
});

describe("NOT MONITORING overlay", () => {
  it("covers the screen with every reason and can only be collapsed for a while", async () => {
    liveStore.set({ ws: "open", liveEnabled: true, cameras: [{ ...camera, monitoring: "NOT_MONITORED" }] });
    renderWith(<NotMonitoringOverlay />);
    const overlay = screen.getByTestId("not-monitoring");
    expect(overlay).toHaveTextContent("NOT MONITORING");
    expect(overlay).toHaveTextContent("Main pool: not analysed");
    fireEvent.click(screen.getByRole("button", { name: /show the controls/ }));
    expect(screen.queryByTestId("not-monitoring")).toBeNull();
    expect(screen.getByTestId("not-monitoring-bar")).toHaveTextContent("Main pool");
    // the two minutes are over (a real past time, not 0) - on the next clock tick it is back full screen
    act(() => {
      liveStore.set({ overlayCollapsedUntil: Date.now() - 500 });
    });
    expect(await screen.findByTestId("not-monitoring", {}, { timeout: 2500 })).toBeInTheDocument();
  });

  it("stays away while everything is monitored", () => {
    liveStore.set({ ws: "open", liveEnabled: true, cameras: [camera] });
    renderWith(<NotMonitoringOverlay />);
    expect(screen.queryByTestId("not-monitoring")).toBeNull();
    expect(screen.queryByTestId("not-monitoring-bar")).toBeNull();
  });

  it("appears when the feed has been down for more than 5 s", () => {
    liveStore.set({ ws: "down", wsDownSince: Date.now() - 6000, liveEnabled: true, cameras: [camera] });
    renderWith(<NotMonitoringOverlay />);
    expect(screen.getByTestId("not-monitoring")).toHaveTextContent(/No connection/);
  });
});

describe("banners", () => {
  it("MUTED lists muted zones and states that submersion alerts and faults still sound", () => {
    liveStore.set({ cameras: [{ ...camera, muted_zones: ["Lap lanes"] }] });
    renderWith(<MutedBanner />);
    const b = screen.getByTestId("muted-banner");
    expect(b).toHaveTextContent("MUTED");
    expect(b).toHaveTextContent("Lap lanes (Main pool)");
    expect(b).toHaveTextContent("Submersion / sudden blackout alerts and faults always sound");
  });

  it("MUTED is absent when no zone is muted", () => {
    liveStore.set({ cameras: [camera] });
    renderWith(<MutedBanner />);
    expect(screen.queryByTestId("muted-banner")).toBeNull();
  });
});

describe("arm-audio gate", () => {
  it("shows monitoring only after audio is armed", async () => {
    const engine = fakeEngine();
    renderWith(
      <ArmAudioGate engine={engine}>
        <p>monitoring</p>
      </ArmAudioGate>,
    );
    expect(screen.queryByText("monitoring")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: /Arm audio/ }));
    expect(await screen.findByText("monitoring")).toBeInTheDocument();
    expect(engine.armCalls()).toBe(1);
  });

  it("keeps the gate closed when the browser refuses audio", async () => {
    renderWith(
      <ArmAudioGate engine={fakeEngine(false)}>
        <p>monitoring</p>
      </ArmAudioGate>,
    );
    fireEvent.click(screen.getByRole("button", { name: /Arm audio/ }));
    expect(await screen.findByRole("alert")).toHaveTextContent(/did not allow audio/);
    expect(screen.queryByText("monitoring")).toBeNull();
  });
});

describe("active alarms", () => {
  it("lists unacknowledged alarms with escalation state and acknowledges with CSRF", async () => {
    liveStore.set({ cameras: [{ ...camera, camera_id: "cam-1" }] });
    document.cookie = `${CSRF_COOKIE}=csrf-xyz; path=/; Secure`;
    let acked = false;
    const calls = mockFetch({
      "GET /api/v1/alerts": () => ({
        body: acked
          ? []
          : [
              makeAlert({ id: 5, escalation_level: 2, escalated_at: new Date().toISOString(), snapshot_available: true }),
              makeAlert({ id: 6, scenario: "FAULT", fault_code: "CAMERA_STALLED", priority: "Critical" }),
            ],
      }),
      "POST /api/v1/alerts/5/acknowledge": () => {
        acked = true;
        return { body: makeAlert({ id: 5, status: "acknowledged" }) };
      },
    });
    renderWith(<ActiveAlarms />);
    expect(await screen.findByTestId("alarm-5")).toHaveTextContent("Drowning Alert · Active drowning");
    expect(screen.getByTestId("escalated-5")).toHaveTextContent("ESCALATED x2");
    expect(screen.getByTestId("alarm-6")).toHaveTextContent("FAULT CAMERA_STALLED");
    expect(calls[0]?.url).toBe("/api/v1/alerts?status=pending&audible=true&limit=200");
    expect(screen.getByTestId("alarm-5")).toHaveTextContent("Main pool"); // camera NAME, not its id
    const card = screen.getByTestId("alarm-5");
    fireEvent.click(card.querySelector("button.btn-ack") as HTMLButtonElement);
    await waitFor(() => {
      expect(screen.getByTestId("no-alarms")).toBeInTheDocument();
    });
    const post = calls.find((c) => c.method === "POST");
    expect(post?.url).toBe("/api/v1/alerts/5/acknowledge");
    expect(post?.headers["X-CSRF-Token"]).toBe("csrf-xyz");
  });
});

describe("one alarm card per event (Phase 5)", () => {
  it("groups one event's alerts; one click acknowledges that event only; another camera keeps its own card", async () => {
    let acked = false;
    const e1a = makeAlert({ id: 11, camera_id: "c1", event_id: 5 });
    const e1b = makeAlert({ id: 12, camera_id: "c1", alert_reason: "Submersion / sudden blackout", event_id: 5 });
    const e2 = makeAlert({ id: 13, camera_id: "c2", event_id: 6 }); // another camera: independent, its own card
    const solo = makeAlert({ id: 14, camera_id: "c3" }); // no event: its own card
    liveStore.set({ cameras: [camera, { ...camera, camera_id: "c2", name: "Deep end cam" }] });
    document.cookie = `${CSRF_COOKIE}=csrf-xyz; path=/; Secure`;
    const calls = mockFetch({
      "GET /api/v1/alerts": () => ({ body: acked ? [e2, solo] : [e1b, e2, solo, e1a] }),
      "POST /api/v1/alerts/11/acknowledge": () => {
        acked = true;
        return { body: { ...e1a, status: "acknowledged" } };
      },
    });
    renderWith(<ActiveAlarms />);
    const card = await screen.findByTestId("alarm-11");
    expect(screen.queryByTestId("alarm-12")).toBeNull(); // in event 5's card
    expect(screen.getByTestId("event-count-11")).toHaveTextContent("2 alerts in this event");
    expect(screen.getByTestId("linked-12")).toHaveTextContent("Drowning Alert · Submersion / sudden blackout");
    expect(screen.getByTestId("alarm-13")).toBeInTheDocument();
    expect(screen.getByTestId("alarm-14")).toBeInTheDocument();
    const button = card.querySelector("button.btn-ack") as HTMLButtonElement;
    expect(button).toHaveTextContent("Acknowledge event (2 alerts)");
    fireEvent.click(button);
    await waitFor(() => {
      expect(screen.queryByTestId("alarm-11")).toBeNull();
    });
    expect(screen.getByTestId("alarm-13")).toBeInTheDocument(); // the other camera's event stays pending
    expect(calls.filter((c) => c.method === "POST").map((c) => c.url)).toEqual(["/api/v1/alerts/11/acknowledge"]);
  });
});

describe("version skew", () => {
  it("alerts from an API without event fields are never merged into one card", async () => {
    const old = (id: number) => {
      const a: Partial<Alert> = makeAlert({ id });
      delete a.event_id;
      return a;
    };
    mockFetch({ "GET /api/v1/alerts": () => ({ body: [old(21), old(22)] }) });
    renderWith(<ActiveAlarms />);
    expect(await screen.findByTestId("alarm-21")).toBeInTheDocument();
    expect(screen.getByTestId("alarm-22")).toBeInTheDocument();
    expect(screen.queryByTestId("event-count-21")).toBeNull();
  });
});

describe("evidence viewer", () => {
  it("plays the WebM recording in a <video> next to the snapshot", () => {
    renderWith(<MediaViewer alert={makeAlert({ id: 9, snapshot_available: true, recording_available: true })} onClose={() => undefined} />);
    expect(screen.getByRole("img", { name: "Alert 9 snapshot" })).toHaveAttribute("src", "/api/v1/alerts/9/snapshot");
    const video = screen.getByTestId("recording");
    expect(video.tagName).toBe("VIDEO");
    expect(video).toHaveAttribute("src", "/api/v1/alerts/9/recording");
    expect(video).toHaveAttribute("controls");
  });

  it("says so when there is no recording", () => {
    renderWith(<MediaViewer alert={makeAlert({ id: 9 })} onClose={() => undefined} />);
    expect(screen.getByText(/No recording/)).toBeInTheDocument();
  });
});

describe("camera tile", () => {
  it("puts NOT MONITORING over the video when the camera is not analysed or the feed is down", () => {
    mockFetch({ "GET /api/v1/alerts": () => ({ body: [] }) }); // the tile asks for its alarm ring
    const first = renderWith(<CameraTile camera={camera} notMonitored={false} feedDown={false} />);
    expect(screen.getByRole("img", { name: "Live view of Main pool" })).toHaveAttribute("data-stream", "/api/v1/live/cameras/c1/mjpeg?overlay=client");
    expect(screen.queryByTestId("camera-overlay-c1")).toBeNull();
    first.unmount();
    renderWith(<CameraTile camera={{ ...camera, stalled: true }} notMonitored feedDown={false} />);
    expect(screen.getByTestId("camera-overlay-c1")).toHaveTextContent("stream stalled");
  });
});

describe("safety audio", () => {
  it("plays the alarm tone for an unacknowledged drowning alarm once audio is armed", async () => {
    liveStore.set({ audioArmed: true, ws: "open", liveEnabled: true, cameras: [camera] });
    mockFetch({ "GET /api/v1/alerts": () => ({ body: [makeAlert({ id: 1, alert_reason: "Drowning" })] }) });
    const engine = fakeEngine();
    renderWith(<SafetyAudio engine={engine} />);
    await waitFor(() => {
      expect(engine.calls).toContain("play:alarm");
    });
  });

  it("[Stage 4] plays the same alarm tone whatever the reason or risk (no per-type tiers)", async () => {
    liveStore.set({ audioArmed: true, ws: "open", liveEnabled: true, cameras: [camera] });
    mockFetch({ "GET /api/v1/alerts": () => ({ body: [makeAlert({ id: 1, alert_reason: "Active drowning", risk_score: 70, severity: "critical" })] }) });
    const engine = fakeEngine();
    renderWith(<SafetyAudio engine={engine} />);
    await waitFor(() => {
      expect(engine.calls).toContain("play:alarm");
    });
  });

  it("plays the fault tone while the NOT MONITORING overlay is up", async () => {
    liveStore.set({ audioArmed: true, ws: "open", liveEnabled: false, cameras: [] });
    mockFetch({ "GET /api/v1/alerts": () => ({ body: [] }) });
    const engine = fakeEngine();
    renderWith(<SafetyAudio engine={engine} />);
    await waitFor(() => {
      expect(engine.calls).toContain("play:fault");
    });
  });
});
