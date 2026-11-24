import { QueryClient } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { liveStore } from "../state/liveStore";
import { resetStore } from "../test/utils";
import { BACKOFF_MS, HEARTBEAT_TIMEOUT_MS, LiveFeed } from "./feed";

class FakeSocket {
  static all: FakeSocket[] = [];
  onopen: (() => void) | null = null;
  onmessage: ((ev: { data: unknown }) => void) | null = null;
  onclose: ((ev: { code: number }) => void) | null = null;
  closed = false;
  constructor(readonly url: string) {
    FakeSocket.all.push(this);
  }
  close() {
    this.closed = true;
  }
  open() {
    this.onopen?.();
  }
  send(ev: object) {
    this.onmessage?.({ data: JSON.stringify(ev) });
  }
  drop(code = 1006) {
    this.onclose?.({ code });
  }
}

function setup(onAuthLost = vi.fn()) {
  const qc = new QueryClient();
  const invalidate = vi.spyOn(qc, "invalidateQueries");
  const feed = new LiveFeed(qc, (u) => new FakeSocket(u) as unknown as WebSocket, () => "ws://test/api/v1/ws", onAuthLost);
  return { qc, invalidate, feed, onAuthLost };
}

describe("LiveFeed", () => {
  beforeEach(() => {
    FakeSocket.all = [];
    resetStore();
    vi.useFakeTimers();
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  it("marks the feed open and refetches what may have changed", () => {
    const { feed, invalidate } = setup();
    feed.start();
    expect(liveStore.get().ws).toBe("connecting");
    FakeSocket.all[0]?.open();
    expect(liveStore.get().ws).toBe("open");
    expect(invalidate).toHaveBeenCalledWith({ queryKey: ["alerts"] });
  });

  it("applies live.summary and invalidates alerts on alert events", () => {
    const { feed, invalidate } = setup();
    feed.start();
    const ws = FakeSocket.all[0];
    ws?.open();
    invalidate.mockClear();
    ws?.send({ type: "live.summary", cameras: [{ camera_id: "c1", name: "A", monitoring: "MONITORED", stalled: false, lagging: false, muted_zones: ["Z"] }], not_started: ["c2"] });
    expect(liveStore.get().cameras?.[0]?.muted_zones).toEqual(["Z"]);
    expect(liveStore.get().notStarted).toEqual(["c2"]);
    ws?.send({ type: "alert.escalated", alert: { id: 7 } });
    expect(invalidate).toHaveBeenCalledWith({ queryKey: ["alerts"] });
    expect(invalidate).toHaveBeenCalledWith({ queryKey: ["alert", 7] });
    ws?.send({ type: "outbox.sent" }); // unknown types are ignored
    feed.handle("not json");
  });

  it("records since when it is down and reconnects with backoff", () => {
    const { feed } = setup();
    vi.setSystemTime(10_000);
    feed.start();
    FakeSocket.all[0]?.open();
    FakeSocket.all[0]?.drop();
    expect(liveStore.get().ws).toBe("down");
    expect(liveStore.get().wsDownSince).toBe(10_000);
    vi.advanceTimersByTime(BACKOFF_MS[0] - 1);
    expect(FakeSocket.all).toHaveLength(1);
    vi.advanceTimersByTime(1);
    expect(FakeSocket.all).toHaveLength(2);
    FakeSocket.all[1]?.drop(); // still down: wsDownSince keeps the FIRST failure time
    expect(liveStore.get().wsDownSince).toBe(10_000);
    vi.advanceTimersByTime(BACKOFF_MS[1]);
    expect(FakeSocket.all).toHaveLength(3);
    FakeSocket.all[2]?.open();
    expect(liveStore.get().wsDownSince).toBeNull();
  });

  it("stops and reports a lost session on close code 4401", () => {
    const { feed, onAuthLost } = setup();
    feed.start();
    FakeSocket.all[0]?.drop(4401);
    vi.advanceTimersByTime(60_000);
    expect(onAuthLost).toHaveBeenCalledOnce();
    expect(FakeSocket.all).toHaveLength(1); // no reconnect without a session
  });

  it("a silent socket (server dead behind a proxy, no close event) counts as down from when it went quiet", () => {
    const { feed } = setup();
    vi.setSystemTime(50_000);
    feed.start();
    const first = FakeSocket.all[0];
    first?.open();
    vi.advanceTimersByTime(1_000);
    first?.send({ type: "heartbeat", server_time: "x" }); // last sign of life at 51_000
    vi.advanceTimersByTime(HEARTBEAT_TIMEOUT_MS - 100);
    expect(liveStore.get().ws).toBe("open"); // not yet: beats may be late
    vi.advanceTimersByTime(700);
    expect(liveStore.get().ws).toBe("down");
    expect(liveStore.get().wsDownSince).toBe(51_000);
    expect(first?.closed).toBe(true);
    first?.drop(); // a late close event of the abandoned socket changes nothing
    expect(liveStore.get().wsDownSince).toBe(51_000);
    vi.advanceTimersByTime(BACKOFF_MS[0]);
    expect(FakeSocket.all).toHaveLength(2); // reconnecting
  });

  it("heartbeats keep the feed open; a socket that never opens is abandoned too", () => {
    const { feed } = setup();
    feed.start();
    FakeSocket.all[0]?.open();
    for (let i = 0; i < 10; i++) {
      vi.advanceTimersByTime(2_000);
      FakeSocket.all[0]?.send({ type: "heartbeat", server_time: "x" });
    }
    expect(liveStore.get().ws).toBe("open");
    expect(FakeSocket.all).toHaveLength(1);
    FakeSocket.all[0]?.drop();
    vi.advanceTimersByTime(BACKOFF_MS[0]);
    expect(FakeSocket.all).toHaveLength(2); // never opens (upgrade hangs at the proxy)
    vi.advanceTimersByTime(HEARTBEAT_TIMEOUT_MS + 500);
    expect(FakeSocket.all[1]?.closed).toBe(true);
    expect(liveStore.get().ws).toBe("down");
  });

  it("stop() ends the watchdog", () => {
    const { feed } = setup();
    feed.start();
    feed.stop();
    vi.advanceTimersByTime(60_000);
    expect(FakeSocket.all).toHaveLength(1);
  });
});
