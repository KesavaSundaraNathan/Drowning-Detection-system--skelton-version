import { describe, expect, it } from "vitest";
import type { ChannelStatus, DeliveryStatus, LiveCamera, LiveStatus } from "../api/types";
import { systemHealth, worst, type HealthInput } from "./health";

function cam(over: Partial<LiveCamera> = {}): LiveCamera {
  return {
    camera_id: "cam-1",
    name: "Deep end",
    monitoring: "MONITORED",
    stalled: false,
    lagging: false,
    lag_s: 0.2,
    analysed_fraction: 1,
    zones: [],
    capture: { fps: 25, hw_accel: 0, frames_read: 100, reconnects: 0, last_frame_age_s: 0.04 },
    analysis: { analysed: 100, proc_ms: 15 },
    ...over,
  };
}

function liveOk(over: Partial<LiveStatus> = {}): LiveStatus {
  return {
    enabled: true,
    running: true,
    provider: "cuda",
    supervisor_down: false,
    error: null,
    cameras: [cam()],
    not_started: {},
    scheduler: { mean_batch: 1, rounds: 10, stalled: false, last_round_ms: 15 },
    ...over,
  };
}

function channel(down = false): ChannelStatus {
  return { pending: 0, oldest_pending_s: null, consecutive_failures: 0, down, sent: 3, last_success_at: null, last_error: null };
}

function deliveryOk(over: Partial<DeliveryStatus> = {}): DeliveryStatus {
  return {
    channels: { ntfy: channel(), smtp: channel() },
    dispatcher_alive: true,
    escalator_alive: true,
    escalations: 0,
    oldest_pending_s: null,
    channels_down: [],
    escalation_tiers: 2,
    self_test: { enabled: true, time: "07:00", last: null },
    ...over,
  };
}

function input(over: Partial<HealthInput> = {}): HealthInput {
  return {
    wsDown: false,
    live: { data: liveOk(), error: false },
    delivery: { data: deliveryOk(), error: false },
    openFaults: { data: 0, error: false },
    ...over,
  };
}

function levelOf(i: HealthInput, key: string): string | undefined {
  return systemHealth(i).checks.find((c) => c.key === key)?.level;
}

describe("systemHealth", () => {
  it("is ok when every source is healthy", () => {
    const h = systemHealth(input());
    expect(h.level).toBe("ok");
    expect(h.problems).toEqual([]);
  });

  it("ranks unknown above degraded (a silent source never looks healthy)", () => {
    expect(worst(["ok", "degraded", "unknown"])).toBe("unknown");
    expect(worst(["unknown", "critical"])).toBe("critical");
    const h = systemHealth(input({ live: { error: true } }));
    expect(h.level).toBe("unknown");
    expect(h.problems[0]?.detail).toMatch(/request failed/);
  });

  it("waiting for the first status is unknown, not ok", () => {
    expect(systemHealth(input({ live: { error: false } })).level).toBe("unknown");
    expect(levelOf(input({ openFaults: { error: false } }), "faults")).toBe("unknown");
  });

  it.each<[string, Partial<HealthInput>, string]>([
    ["feed lost", { wsDown: true }, "feed"],
    ["live runtime off", { live: { data: { enabled: false, cameras: [] }, error: false } }, "live"],
    ["runtime start error", { live: { data: liveOk({ error: "CUDA provider unavailable" }), error: false } }, "live"],
    ["runtime not running", { live: { data: liveOk({ running: false }), error: false } }, "live"],
    ["scheduler stalled", { live: { data: liveOk({ scheduler: { mean_batch: 1, rounds: 9, stalled: true, last_round_ms: 0 } }), error: false } }, "scheduler"],
    ["supervisor down", { live: { data: liveOk({ supervisor_down: true }), error: false } }, "supervisor"],
    ["camera stalled", { live: { data: liveOk({ cameras: [cam({ stalled: true })] }), error: false } }, "cam:cam-1"],
    ["camera not analysed", { live: { data: liveOk({ cameras: [cam({ monitoring: "NOT_MONITORED" })] }), error: false } }, "cam:cam-1"],
    ["camera not started", { live: { data: liveOk({ not_started: { "cam-2": "capacity" } }), error: false } }, "cam:cam-2"],
    ["open fault", { openFaults: { data: 2, error: false } }, "faults"],
    ["no channel configured", { delivery: { data: deliveryOk({ channels: {} }), error: false } }, "delivery"],
    ["every channel down", { delivery: { data: deliveryOk({ channels: { ntfy: channel(true) } }), error: false } }, "delivery"],
    ["dispatcher dead", { delivery: { data: deliveryOk({ dispatcher_alive: false }), error: false } }, "delivery-workers"],
  ])("%s is critical", (_name, over, key) => {
    const h = systemHealth(input(over));
    expect(h.level).toBe("critical");
    expect(h.checks.find((c) => c.key === key)?.level).toBe("critical");
    expect(h.problems[0]?.level).toBe("critical");
  });

  it.each<[string, Partial<HealthInput>, string]>([
    ["camera lagging", { live: { data: liveOk({ cameras: [cam({ lagging: true, lag_s: 3.2 })] }), error: false } }, "cam:cam-1"],
    ["no camera running", { live: { data: liveOk({ cameras: [] }), error: false } }, "cameras"],
    ["not on the GPU", { live: { data: liveOk({ provider: "cpu" }), error: false } }, "provider"],
    ["one of two channels down", { delivery: { data: deliveryOk({ channels: { ntfy: channel(true), smtp: channel() } }), error: false } }, "delivery"],
    ["alerts queued", { delivery: { data: deliveryOk({ oldest_pending_s: 120 }), error: false } }, "delivery-queue"],
  ])("%s is degraded", (_name, over, key) => {
    const h = systemHealth(input(over));
    expect(h.level).toBe("degraded");
    expect(h.checks.find((c) => c.key === key)?.level).toBe("degraded");
  });

  it("judges the provider that is RUNNING, not the requested one ('auto')", () => {
    // [v4] the site's API ran on TensorRT but showed "auto - not on the GPU" (the requested value was tested)
    const onGpu = systemHealth(input({ live: { data: liveOk({ provider: "auto", active_provider: "tensorrt" }), error: false } }));
    expect(onGpu.checks.find((c) => c.key === "provider")?.level).toBe("ok");
    const fellBack = systemHealth(input({ live: { data: liveOk({ provider: "auto", active_provider: "cpu" }), error: false } }));
    expect(fellBack.checks.find((c) => c.key === "provider")?.detail).toBe("cpu - not on the GPU, capacity is reduced");
  });

  it("skips delivery for roles that cannot read it", () => {
    const h = systemHealth(input({ delivery: undefined }));
    expect(h.checks.some((c) => c.key.startsWith("delivery"))).toBe(false);
    expect(h.level).toBe("ok");
  });

  it("counts monitored cameras including the ones that never started", () => {
    const h = systemHealth(input({ live: { data: liveOk({ not_started: { "cam-2": "invalid_source" } }), error: false } }));
    expect(h.checks.find((c) => c.key === "cameras")?.detail).toBe("1 of 2 monitored");
    expect(h.checks.find((c) => c.key === "cam:cam-2")?.detail).toMatch(/invalid video source/);
    expect(h.checks.find((c) => c.key === "cam:cam-2")?.label).toBe("Camera cam-2");
    const named = systemHealth({ ...input({ live: { data: liveOk({ not_started: { "cam-2": "capacity" } }), error: false } }), cameraNames: { "cam-2": "Shallow end" } });
    expect(named.checks.find((c) => c.key === "cam:cam-2")?.label).toBe("Shallow end");
  });

  it("orders problems worst first", () => {
    const h = systemHealth(
      input({ live: { data: liveOk({ provider: "cpu", supervisor_down: true }), error: false }, delivery: { error: true } }),
    );
    expect(h.problems.map((p) => p.level)).toEqual(["critical", "unknown", "degraded"]);
  });
});
