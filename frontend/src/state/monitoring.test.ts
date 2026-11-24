import { describe, expect, it } from "vitest";

import type { LiveStatus, LiveSummaryCamera } from "../api/types";
import { initialState, type LiveState } from "./liveStore";
import { mutedZones, summaryFromStatus, verdict, WS_GRACE_MS } from "./monitoring";

const cam = (over: Partial<LiveSummaryCamera> = {}): LiveSummaryCamera => ({
  camera_id: "c1",
  name: "Main pool",
  monitoring: "MONITORED",
  stalled: false,
  lagging: false,
  muted_zones: [],
  ...over,
});
const state = (over: Partial<LiveState> = {}): LiveState => ({ ...initialState, ws: "open", liveEnabled: true, cameras: [cam()], ...over });

describe("NOT MONITORING verdict", () => {
  it("is quiet when everything is monitored", () => {
    expect(verdict(state(), 0).notMonitoring).toBe(false);
  });

  it("waits WS_GRACE_MS before a lost connection counts", () => {
    const s = state({ ws: "down", wsDownSince: 1000 });
    expect(verdict(s, 1000 + WS_GRACE_MS).notMonitoring).toBe(false);
    const v = verdict(s, 1000 + WS_GRACE_MS + 1);
    expect(v.notMonitoring).toBe(true);
    expect(v.reasons[0]).toMatch(/No connection/);
  });

  it("flags a NOT_MONITORED or stalled camera and names it", () => {
    const v = verdict(state({ cameras: [cam(), cam({ camera_id: "c2", name: "Kids pool", stalled: true })] }), 0);
    expect(v.notMonitoring).toBe(true);
    expect([...v.cameras]).toEqual(["c2"]);
    expect(v.reasons).toEqual(["Kids pool: no video (stream stalled)"]);
    expect(verdict(state({ cameras: [cam({ monitoring: "NOT_MONITORED" })] }), 0).reasons[0]).toMatch(/not analysed/);
  });

  it("flags cameras that could not start and a disabled live runtime", () => {
    const v = verdict(state({ notStarted: ["c9"], liveEnabled: false }), 0);
    expect(v.cameras.has("c9")).toBe(true);
    expect(v.reasons).toHaveLength(2);
  });

  it("lists muted zones per camera", () => {
    expect(mutedZones(state({ cameras: [cam({ muted_zones: ["Lap lanes"] })] }))).toEqual([
      { camera: "Main pool", zone: "Lap lanes" },
    ]);
  });

  it("derives the summary from /live/status (stalled => NOT_MONITORED, MUTED zones)", () => {
    const st = {
      enabled: true,
      cameras: [
        {
          camera_id: "c1",
          name: "Main",
          monitoring: "MONITORED",
          stalled: true,
          lagging: false,
          lag_s: 0,
          analysed_fraction: 1,
          zones: [
            { zone: "A", zone_type: "WATER", state: "MUTED", schedule_id: "1" },
            { zone: "B", zone_type: "WATER", state: "MONITORED", schedule_id: null },
          ],
          capture: { fps: 25, hw_accel: 2, frames_read: 1, reconnects: 0, last_frame_age_s: 9 },
          analysis: { analysed: 1, proc_ms: 20 },
        },
      ],
    } satisfies LiveStatus;
    expect(summaryFromStatus(st)).toEqual([cam({ camera_id: "c1", name: "Main", monitoring: "NOT_MONITORED", stalled: true, muted_zones: ["A"] })]);
  });

  it("a status without cameras (runtime off) maps to no cameras instead of throwing", () => {
    expect(summaryFromStatus({ enabled: false })).toEqual([]);
  });
});
