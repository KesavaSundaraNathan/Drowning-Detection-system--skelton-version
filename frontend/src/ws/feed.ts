/**
 * Real-time feed: GET /api/v1/ws (session cookie + Origin check on the server).
 *
 * On every alert / fault event the relevant TanStack Query caches are invalidated - the database
 * stays the source of truth, so a missed event costs at most one refetch. live.summary replaces the
 * camera state in the store. The socket reconnects with capped exponential backoff; while it is
 * down the store records since when, which drives the NOT MONITORING overlay after 5 s.
 *
 * [v3] Phase 7: a server that dies behind a proxy, or a half-open TCP connection, never fires
 * `onclose` (found by the Playwright walkthrough: the UI showed "live" for minutes with the API
 * dead). The server sends a heartbeat every 2 s; HEARTBEAT_TIMEOUT_MS without any message, or
 * without the socket opening, counts as a lost feed from the moment it went silent.
 */
import type { QueryClient } from "@tanstack/react-query";

import { media } from "../api/endpoints";
import type { LiveSummaryCamera } from "../api/types";
import { liveStore } from "../state/liveStore";

export const BACKOFF_MS = [500, 1000, 2000, 4000, 8000] as const;
/** 3 missed server heartbeats (ws_heartbeat_s = 2 s). */
export const HEARTBEAT_TIMEOUT_MS = 6000;
const WATCHDOG_TICK_MS = 500;

interface FeedEvent {
  type: string;
  cameras?: LiveSummaryCamera[];
  not_started?: string[];
  alert?: { id: number };
}

export type SocketFactory = (url: string) => WebSocket;

export class LiveFeed {
  private ws: WebSocket | null = null;
  private attempts = 0;
  private timer: ReturnType<typeof setTimeout> | null = null;
  private watchdog: ReturnType<typeof setInterval> | null = null;
  private lastActivity = 0;
  private stopped = false;

  constructor(
    private readonly queryClient: QueryClient,
    private readonly factory: SocketFactory = (u) => new WebSocket(u),
    private readonly url: () => string = media.ws,
    private readonly onAuthLost: () => void = () => undefined,
  ) {}

  start(): void {
    this.stopped = false;
    this.watchdog ??= setInterval(() => {
      this.checkSilence();
    }, WATCHDOG_TICK_MS);
    this.connect();
  }

  stop(): void {
    this.stopped = true;
    if (this.timer !== null) clearTimeout(this.timer);
    if (this.watchdog !== null) clearInterval(this.watchdog);
    this.watchdog = null;
    this.ws?.close();
    this.ws = null;
  }

  /** No message (or no open) for HEARTBEAT_TIMEOUT_MS: drop the socket without waiting for a close
   * handshake that may never come, mark the feed down since it went silent, reconnect. */
  private checkSilence(): void {
    const ws = this.ws;
    if (ws === null || Date.now() - this.lastActivity <= HEARTBEAT_TIMEOUT_MS) return;
    ws.onopen = null;
    ws.onmessage = null;
    ws.onclose = null;
    ws.close();
    this.ws = null;
    const s = liveStore.get();
    liveStore.set({ ws: "down", wsDownSince: s.wsDownSince ?? this.lastActivity });
    this.schedule();
  }

  private connect(): void {
    if (this.stopped) return;
    if (liveStore.get().ws !== "down") liveStore.set({ ws: "connecting" });
    const ws = this.factory(this.url());
    this.ws = ws;
    this.lastActivity = Date.now();
    ws.onopen = () => {
      this.lastActivity = Date.now();
      this.attempts = 0;
      liveStore.set({ ws: "open", wsDownSince: null });
      // anything may have happened while we were away
      void this.queryClient.invalidateQueries({ queryKey: ["alerts"] });
      void this.queryClient.invalidateQueries({ queryKey: ["live"] });
    };
    ws.onmessage = (ev: MessageEvent) => {
      this.lastActivity = Date.now();
      this.handle(ev.data);
    };
    ws.onclose = (ev: CloseEvent) => {
      this.ws = null;
      const s = liveStore.get();
      liveStore.set({ ws: "down", wsDownSince: s.wsDownSince ?? Date.now() });
      if (ev.code === 4401) {
        this.onAuthLost(); // session ended (logout elsewhere / revoked / expired)
        return;
      }
      this.schedule();
    };
  }

  private schedule(): void {
    if (this.stopped) return;
    const delay = BACKOFF_MS[Math.min(this.attempts, BACKOFF_MS.length - 1)] ?? 8000;
    this.attempts += 1;
    this.timer = setTimeout(() => {
      this.connect();
    }, delay);
  }

  handle(raw: unknown): void {
    if (typeof raw !== "string") return;
    let ev: FeedEvent;
    try {
      ev = JSON.parse(raw) as FeedEvent;
    } catch {
      return;
    }
    switch (ev.type) {
      case "live.summary":
        liveStore.set({ cameras: ev.cameras ?? [], notStarted: ev.not_started ?? [], liveEnabled: true });
        break;
      case "alert.created":
      case "alert.updated":
      case "alert.escalated":
      case "fault.cleared":
        void this.queryClient.invalidateQueries({ queryKey: ["alerts"] });
        if (ev.alert) void this.queryClient.invalidateQueries({ queryKey: ["alert", ev.alert.id] });
        void this.queryClient.invalidateQueries({ queryKey: ["events"] });
        break;
      case "event.opened": // [Phase 5] backend alerts/events.py
      case "event.updated":
      case "event.closed":
        void this.queryClient.invalidateQueries({ queryKey: ["events"] });
        void this.queryClient.invalidateQueries({ queryKey: ["alerts"] });
        break;
      default:
        break;
    }
  }
}
