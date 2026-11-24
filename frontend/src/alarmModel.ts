import { useQuery } from "@tanstack/react-query";

import { api } from "./api/endpoints";
import type { Alert } from "./api/types";
import type { Tone } from "./audio/tones";

export const ACTIVE_ALARMS_KEY = ["alerts", "active"] as const;

/** Pending AUDIBLE alerts = alarms that still sound / escalate until somebody acknowledges them. */
export function useActiveAlarms() {
  return useQuery({
    queryKey: ACTIVE_ALARMS_KEY,
    queryFn: () => api.alerts({ status: "pending", audible: true, limit: 200 }),
    refetchInterval: 15_000, // belt and braces: the WebSocket invalidates this on every event
  });
}

export interface AlarmGroup {
  key: string;
  primary: Alert; // the event's oldest pending alert: its card, its Acknowledge
  alerts: Alert[]; // oldest first, primary included
}

/**
 * [Phase 5] One card per EVENT: a person's pending alerts on ONE camera (the server's alerts/events.py) share a
 * card, and one Acknowledge acknowledges exactly those. Cameras are independent: two cameras never share a card.
 * FAULTs and alerts without an event stay one card each. Oldest first.
 */
export function groupByEvent(alerts: readonly Alert[]): AlarmGroup[] {
  const groups = new Map<string, Alert[]>();
  for (const a of [...alerts].sort((x, y) => x.id - y.id)) {
    // typeof, not !== null: an older API omits the field (undefined) and must never merge every alarm into one card
    const key = typeof a.event_id === "number" && a.scenario !== "FAULT" ? `event-${String(a.event_id)}` : `alert-${String(a.id)}`;
    const list = groups.get(key);
    if (list) list.push(a);
    else groups.set(key, [a]);
  }
  return [...groups.entries()].map(([key, list]) => ({ key, primary: list[0] as Alert, alerts: list }));
}

/** [Stage 4] Every drowning alarm is one "Drowning Alert"; the server adds one short reason (`alert_reason`:
 * "Submersion / sudden blackout", "Active drowning" or "Drowning"). There are no per-type tiers any more. */
export const DROWNING_ALERT = "Drowning Alert";

export function alarmReason(a: Pick<Alert, "alert_reason">): string {
  return a.alert_reason ?? "Drowning";
}

export function alarmTitle(a: Alert): string {
  if (a.scenario === "FAULT") return `FAULT ${a.fault_code ?? ""}`.trim();
  return `${DROWNING_ALERT} · ${alarmReason(a)}`;
}

/** [Stage 4] The alarm group to show first: the highest risk, then the oldest (risk-based ordering). */
export function mostUrgent(groups: readonly AlarmGroup[]): AlarmGroup | undefined {
  const risk = (g: AlarmGroup) => Math.max(...g.alerts.map((a) => a.risk_score));
  return [...groups].sort((x, y) => risk(y) - risk(x) || x.primary.id - y.primary.id)[0];
}

/**
 * Which tone sounds. Priority: an unacknowledged drowning alarm (the alarm siren - one tone for every drowning
 * alarm) > NOT MONITORING (full-screen overlay, not collapsed) or an unacknowledged FAULT (fault tone).
 */
export function chooseTone(o: { armed: boolean; alarms: number; faults: number; notMonitoringOverlay: boolean }): Tone | null {
  if (!o.armed) return null;
  if (o.alarms > 0) return "alarm";
  if (o.notMonitoringOverlay || o.faults > 0) return "fault";
  return null;
}

/** "[ONGOING 8 s]" at the end of an alert's reason (backend alerts/writer.record_ongoing) -> 8, else null. */
export function ongoingSeconds(reason: string | null | undefined): number | null {
  const m = /\[ONGOING (\d+) s\]\s*$/.exec(reason ?? "");
  return m ? Number(m[1]) : null;
}

/** Camera ids with an unacknowledged drowning alarm - their tiles get the red pulsing ring. */
export function useAlarmCameras(): Set<string> {
  const q = useActiveAlarms();
  return new Set((q.data ?? []).flatMap((a) => (a.scenario !== "FAULT" && a.camera_id ? [a.camera_id] : [])));
}
