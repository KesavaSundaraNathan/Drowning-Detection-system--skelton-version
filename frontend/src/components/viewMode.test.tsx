import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { ToneEngine } from "../audio/tones";
import { liveStore } from "../state/liveStore";
import { makeAlert } from "../test/fixtures";
import { mockFetch, renderWith, resetStore } from "../test/utils";
import { CameraWall } from "./CameraWall";
import { BoxOverlay } from "./liveFrames";
import { takeParts } from "./mjpegReader";
import { SafetyAudio } from "./SafetyAudio";
import { boxesFor, parseBoxes, readViewMode, type FrameBoxes, type ViewMode } from "./viewMode";
import { VisualAlarm } from "./VisualAlarm";

const cam = { camera_id: "cam-1", name: "Main pool", monitoring: "MONITORED" as const, stalled: false, lagging: false, muted_zones: [] };
const BOXES: FrameBoxes = {
  w: 960,
  h: 540,
  t: [
    [3, 100, 100, 200, 300, 0, 1], // alerting
    [5, 400, 100, 450, 200, 0, 0], // tracked
    [-1, 600, 100, 650, 200, 1, 0], // tentative
  ],
  c: [[8, 700, 100, 750, 200, 0]],
};
const MODES: ViewMode[] = ["full", "clean", "alerts"];

beforeEach(() => {
  localStorage.clear();
  resetStore();
});
afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("view modes: what each mode draws", () => {
  it("Full draws every box, Clean none, Alerts only the alerting person's red box", () => {
    expect(boxesFor("full", BOXES)).toBe(BOXES);
    expect(boxesFor("clean", BOXES)?.t).toEqual([]);
    expect(boxesFor("clean", BOXES)?.c).toEqual([]);
    expect(boxesFor("alerts", BOXES)?.t.map((t) => t[0])).toEqual([3]);
    expect(boxesFor("alerts", BOXES)?.c).toEqual([]);
  });

  it("renders the boxes as an overlay: red + ALERT for the alerting person", () => {
    const { container, rerender } = render(<BoxOverlay boxes={BOXES} mode="full" />);
    expect(container.querySelectorAll("[data-box]")).toHaveLength(4);
    expect(container.querySelector('[data-box="alert"]')).toHaveTextContent("#3 ALERT");
    rerender(<BoxOverlay boxes={BOXES} mode="alerts" />);
    expect(Array.from(container.querySelectorAll("[data-box]")).map((g) => g.getAttribute("data-box"))).toEqual(["alert"]);
    rerender(<BoxOverlay boxes={BOXES} mode="clean" />);
    expect(container.querySelector("[data-box]")).toBeNull();
  });

  it("parses the server's boxes header and rejects junk", () => {
    expect(parseBoxes(JSON.stringify(BOXES))?.t).toHaveLength(3);
    expect(parseBoxes("not json")).toBeNull();
    expect(parseBoxes(null)).toBeNull();
  });
});

describe("view modes: remembered per camera, default Full", () => {
  it("defaults to Full and remembers the choice per camera", () => {
    expect(readViewMode("cam-1")).toBe("full");
    mockFetch({ "GET /api/v1/alerts": () => ({ body: [] }) });
    renderWith(<CameraWall cameras={[cam, { ...cam, camera_id: "cam-2", name: "Deep end" }]} notMonitored={() => false} feedDown={false} canControl onStop={vi.fn()} />);
    const group = within(screen.getByTestId("camera-cam-1")).getByRole("group", { name: "View mode" });
    expect(within(group).getByRole("button", { name: "Full" })).toHaveAttribute("aria-pressed", "true");
    fireEvent.click(within(group).getByRole("button", { name: "Clean" }));
    expect(within(group).getByRole("button", { name: "Clean" })).toHaveAttribute("aria-pressed", "true");
    expect(localStorage.getItem("lifeguard.viewMode.cam-1")).toBe("clean");
    expect(readViewMode("cam-2")).toBe("full");
  });

  it("falls back to Full when storage is unavailable", () => {
    vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
      throw new Error("blocked");
    });
    expect(readViewMode("cam-1")).toBe("full");
  });
});

describe("a view mode never hides or delays an alert", () => {
  function engine(): ToneEngine & { calls: string[] } {
    const calls: string[] = [];
    return { calls, playing: null, arm: vi.fn(() => Promise.resolve(true)), play: (t) => calls.push(`play:${t}`), stop: () => calls.push("stop") };
  }

  it.each(MODES)("mode %s: red ring, alarm bar and alarm sound all come up for an alert on the camera", async (mode) => {
    localStorage.setItem("lifeguard.viewMode.cam-1", mode);
    liveStore.set({ audioArmed: true, ws: "open", liveEnabled: true, cameras: [cam] });
    mockFetch({ "GET /api/v1/alerts": () => ({ body: [makeAlert({ id: 1, camera_id: "cam-1" })] }), "GET /api/v1/cameras": () => ({ body: [] }) });
    const tones = engine();
    renderWith(
      <>
        <VisualAlarm />
        <SafetyAudio engine={tones} />
        <CameraWall cameras={[cam]} notMonitored={() => false} feedDown={false} canControl onStop={vi.fn()} />
      </>,
    );
    // all three on the same alerts response: none of them waits for a frame, a box or the mode
    await waitFor(() => {
      expect(screen.getByTestId("visual-alarm-bar")).toBeInTheDocument();
    });
    expect(screen.getByTestId("camera-cam-1")).toHaveClass("cam-alarm");
    expect(tones.calls).toContain("play:alarm");
  });

  it("the red box of the alerting person stays in Full and Alerts only (Clean shows the ring, bar and sound)", () => {
    for (const mode of ["full", "alerts"] as const) {
      const { container, unmount } = render(<BoxOverlay boxes={BOXES} mode={mode} />);
      expect(container.querySelector('[data-box="alert"]')).not.toBeNull();
      unmount();
    }
  });

  it("switching mode while an alarm is up keeps the alarm on screen", async () => {
    liveStore.set({ audioArmed: true, ws: "open", liveEnabled: true, cameras: [cam] });
    mockFetch({ "GET /api/v1/alerts": () => ({ body: [makeAlert({ id: 1, camera_id: "cam-1" })] }), "GET /api/v1/cameras": () => ({ body: [] }) });
    renderWith(
      <>
        <VisualAlarm />
        <CameraWall cameras={[cam]} notMonitored={() => false} feedDown={false} canControl onStop={vi.fn()} />
      </>,
    );
    await waitFor(() => {
      expect(screen.getByTestId("camera-cam-1")).toHaveClass("cam-alarm");
    });
    for (const label of ["Clean", "Alerts only", "Full"]) {
      act(() => {
        fireEvent.click(screen.getByRole("button", { name: label }));
      });
      expect(screen.getByTestId("camera-cam-1")).toHaveClass("cam-alarm");
      expect(screen.getByTestId("visual-alarm-bar")).toBeInTheDocument();
    }
  });
});

describe("multipart reader", () => {
  it("splits parts, reads the boxes header and keeps an incomplete part for later", () => {
    const enc = new TextEncoder();
    const jpg = new Uint8Array([0xff, 0xd8, 1, 2, 3, 0xff, 0xd9]);
    const part = (boxes: string) =>
      new Uint8Array([
        ...enc.encode(`--lifeguardframe\r\nContent-Type: image/jpeg\r\nContent-Length: ${String(jpg.length)}\r\nX-Lifeguard-Boxes: ${boxes}\r\n\r\n`),
        ...jpg,
        ...enc.encode("\r\n"),
      ]);
    const a = part('{"w":2,"h":2,"t":[],"c":[]}');
    const b = part('{"w":4,"h":4,"t":[],"c":[]}');
    const all = new Uint8Array([...a, ...b.slice(0, 20)]);
    const { parts, rest } = takeParts(all);
    expect(parts).toHaveLength(1);
    expect(Array.from(parts[0]?.jpeg ?? [])).toEqual(Array.from(jpg));
    expect(parseBoxes(parts[0]?.headers["x-lifeguard-boxes"] ?? null)?.w).toBe(2);
    const again = takeParts(new Uint8Array([...rest, ...b.slice(20)]));
    expect(again.parts).toHaveLength(1);
    expect(parseBoxes(again.parts[0]?.headers["x-lifeguard-boxes"] ?? null)?.w).toBe(4);
  });
});
