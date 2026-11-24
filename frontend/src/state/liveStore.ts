/**
 * The one small client store (plan Phase 6): WebSocket connection state, the latest live summary,
 * and whether audio has been armed. Server data (alerts, cameras, ...) lives in TanStack Query.
 */
import { useSyncExternalStore } from "react";

import type { LiveSummaryCamera } from "../api/types";

export type WsState = "connecting" | "open" | "down";

export interface LiveState {
  ws: WsState;
  /** epoch ms when the socket last went down (null while open) */
  wsDownSince: number | null;
  /** null until the first summary / status arrived */
  cameras: LiveSummaryCamera[] | null;
  notStarted: string[];
  liveEnabled: boolean | null;
  audioArmed: boolean;
  /** NOT MONITORING overlay collapsed by the operator until this epoch ms */
  overlayCollapsedUntil: number;
}

export const initialState: LiveState = {
  ws: "connecting",
  wsDownSince: null,
  cameras: null,
  notStarted: [],
  liveEnabled: null,
  audioArmed: false,
  overlayCollapsedUntil: 0,
};

type Listener = () => void;

export class Store<S> {
  private state: S;
  private readonly listeners = new Set<Listener>();

  constructor(initial: S) {
    this.state = initial;
  }

  get = (): S => this.state;

  set(patch: Partial<S>): void {
    this.state = { ...this.state, ...patch };
    for (const l of this.listeners) l();
  }

  subscribe = (l: Listener): (() => void) => {
    this.listeners.add(l);
    return () => {
      this.listeners.delete(l);
    };
  };
}

export const liveStore = new Store<LiveState>(initialState);

/** Subscribe to the store; `select` runs on the (stable) state snapshot, so it may build new arrays. */
export function useLive<T>(select: (s: LiveState) => T): T {
  const state = useSyncExternalStore(liveStore.subscribe, liveStore.get);
  return select(state);
}
