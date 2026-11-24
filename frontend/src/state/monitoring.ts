/**
 * When is the operator NOT being protected? (plan Phase 6 safety UX)
 *
 * The full-screen NOT MONITORING overlay (+ fault tone) shows when ANY of:
 *   * the real-time feed (WebSocket) has been down for more than WS_GRACE_MS (5 s);
 *   * the live runtime is disabled;
 *   * a camera is NOT_MONITORED or stalled, or an enabled camera could not be started.
 * A stale screen must never look healthy: every reason is listed for the operator.
 */
import type { LiveStatus, LiveSummaryCamera } from "../api/types";
import type { LiveState } from "./liveStore";

export const WS_GRACE_MS = 5000;

export interface MonitoringVerdict {
  notMonitoring: boolean;
  reasons: string[];
  /** camera ids that are not monitored (for the per-camera video overlay) */
  cameras: Set<string>;
}

export function verdict(s: LiveState, now: number): MonitoringVerdict {
  const reasons: string[] = [];
  const cameras = new Set<string>();
  if (s.ws === "down" && s.wsDownSince !== null && now - s.wsDownSince > WS_GRACE_MS) {
    reasons.push(`No connection to the Life Guard server for ${String(Math.round((now - s.wsDownSince) / 1000))} s`);
  }
  if (s.liveEnabled === false) {
    reasons.push("Live monitoring is switched OFF on the server");
  }
  for (const c of s.cameras ?? []) {
    if (c.monitoring === "NOT_MONITORED" || c.stalled) {
      cameras.add(c.camera_id);
      reasons.push(`${c.name}: ${c.stalled ? "no video (stream stalled)" : "not analysed"}`);
    }
  }
  for (const id of s.notStarted) {
    cameras.add(id);
    reasons.push(`Camera ${id}: not started (capacity or invalid source)`);
  }
  return { notMonitoring: reasons.length > 0, reasons, cameras };
}

export interface MutedZone {
  camera: string;
  zone: string;
}

export function mutedZones(s: LiveState): MutedZone[] {
  const out: MutedZone[] = [];
  for (const c of s.cameras ?? []) {
    for (const z of c.muted_zones) out.push({ camera: c.name, zone: z });
  }
  return out;
}

export function summaryFromStatus(st: LiveStatus): LiveSummaryCamera[] {
  // defensive: a status without cameras (runtime off, older server) must not blank the safety UI
  return (st.cameras ?? []).map((c) => ({
    camera_id: c.camera_id,
    name: c.name,
    monitoring: c.stalled ? "NOT_MONITORED" : c.monitoring,
    stalled: c.stalled,
    lagging: c.lagging,
    muted_zones: c.zones.filter((z) => z.state === "MUTED").map((z) => z.zone),
  }));
}
