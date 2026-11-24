/**
 * System health: one verdict from every status source the operator UI can read (Phase 9).
 *
 * Levels, worst first: critical > unknown > degraded > ok. "unknown" ranks ABOVE degraded on purpose:
 * a status we cannot read must never look healthy (same rule as the NOT MONITORING overlay).
 *
 *   critical  the pool is, or may be, unprotected now (feed down, camera not analysed, scheduler
 *             stalled, supervisor down, open SYSTEM FAULT, no working off-site alert channel)
 *   unknown   a status source did not answer
 *   degraded  protected, but with reduced margin (camera lagging, one of several channels down,
 *             alerts queued, no camera running, inference not on the GPU)
 *
 * Inputs a role cannot read (delivery: supervisor+) are passed as `undefined` and simply skipped.
 */
import type { DeliveryStatus, LiveStatus } from "../api/types";

export type HealthLevel = "ok" | "degraded" | "unknown" | "critical";

export interface HealthCheck {
  key: string;
  label: string;
  level: HealthLevel;
  detail: string;
}

export interface HealthReport {
  level: HealthLevel;
  checks: HealthCheck[];
  /** the non-ok checks, worst first */
  problems: HealthCheck[];
}

/** A query result: data, or the fact that it failed. `undefined` = not applicable to this role. */
export interface Source<T> {
  data?: T | undefined;
  error: boolean;
}

export interface HealthInput {
  /** the real-time feed has been down for longer than the grace period */
  wsDown: boolean;
  live: Source<LiveStatus>;
  delivery?: Source<DeliveryStatus> | undefined;
  /** open (uncleared) SYSTEM FAULT alarms */
  openFaults?: Source<number> | undefined;
  /** camera id -> name (GET /cameras), so a camera that never started is named, not shown as a UUID */
  cameraNames?: Record<string, string> | undefined;
}

const RANK: Record<HealthLevel, number> = { ok: 0, degraded: 1, unknown: 2, critical: 3 };

/** Alerts waiting longer than this are worth a look (delivery retries within seconds normally). */
export const PENDING_DEGRADED_S = 60;

export function worst(levels: HealthLevel[]): HealthLevel {
  return levels.reduce<HealthLevel>((w, l) => (RANK[l] > RANK[w] ? l : w), "ok");
}

function check(key: string, label: string, level: HealthLevel, detail: string): HealthCheck {
  return { key, label, level, detail };
}

function liveChecks(live: Source<LiveStatus>, names: Record<string, string>): HealthCheck[] {
  const st = live.data;
  if (!st) {
    const why = live.error ? "status request failed" : "waiting for the first status";
    return [check("live", "Live monitoring", "unknown", why)];
  }
  if (!st.enabled) return [check("live", "Live monitoring", "critical", "switched OFF on the server - no camera is analysed")];
  const out: HealthCheck[] = [];
  if (st.error) out.push(check("live", "Live monitoring", "critical", `runtime failed to start: ${st.error}`));
  else if (st.running === false) out.push(check("live", "Live monitoring", "critical", "runtime is not running"));
  else out.push(check("live", "Live monitoring", "ok", "running"));

  if (st.scheduler) {
    const s = st.scheduler;
    out.push(
      s.stalled
        ? check("scheduler", "GPU scheduler", "critical", "STALLED - frames are not being analysed")
        : check("scheduler", "GPU scheduler", "ok", `running, last round ${String(s.last_round_ms)} ms`),
    );
  } else if (st.running) {
    out.push(check("scheduler", "GPU scheduler", "unknown", "no scheduler statistics"));
  }

  // [v4] the provider actually RUNNING: the requested one is often "auto", which never names the GPU
  const provider = st.active_provider ?? st.provider;
  if (provider) {
    const gpu = /cuda|tensorrt/i.test(provider);
    out.push(
      check("provider", "Inference", gpu ? "ok" : "degraded", gpu ? provider : `${provider} - not on the GPU, capacity is reduced`),
    );
  }

  out.push(
    st.supervisor_down
      ? check("supervisor", "Supervisor service", "critical", "DOWN - out-of-band SYSTEM FAULT alarms are not guaranteed")
      : check("supervisor", "Supervisor service", "ok", "heartbeat fresh"),
  );

  const cams = st.cameras ?? [];
  const notStarted = Object.entries(st.not_started ?? {});
  for (const c of cams) {
    if (c.stalled) out.push(check(`cam:${c.camera_id}`, c.name, "critical", "no video (stream stalled)"));
    else if (c.monitoring === "NOT_MONITORED") out.push(check(`cam:${c.camera_id}`, c.name, "critical", "not analysed"));
    else if (c.lagging) out.push(check(`cam:${c.camera_id}`, c.name, "degraded", `lagging ${c.lag_s.toFixed(1)} s behind live`));
  }
  for (const [id, reason] of notStarted) {
    const why = reason === "capacity" ? "not started: GPU capacity reached" : reason === "invalid_source" ? "not started: invalid video source" : `not started: ${reason}`;
    out.push(check(`cam:${id}`, names[id] ?? `Camera ${id}`, "critical", why));
  }
  const healthy = cams.filter((c) => !c.stalled && c.monitoring === "MONITORED").length;
  if (cams.length === 0 && notStarted.length === 0) out.push(check("cameras", "Cameras", "degraded", "no camera is running"));
  else out.push(check("cameras", "Cameras", healthy === cams.length && notStarted.length === 0 ? "ok" : "critical", `${String(healthy)} of ${String(cams.length + notStarted.length)} monitored`));
  return out;
}

function deliveryChecks(d: Source<DeliveryStatus>): HealthCheck[] {
  const st = d.data;
  if (!st) return [check("delivery", "Alert delivery", "unknown", d.error ? "status request failed" : "waiting for the first status")];
  const out: HealthCheck[] = [];
  const names = Object.keys(st.channels);
  const down = names.filter((n) => st.channels[n]?.down);
  if (names.length === 0) out.push(check("delivery", "Alert delivery", "critical", "no off-site channel configured - alarms reach this screen only"));
  else if (down.length === names.length) out.push(check("delivery", "Alert delivery", "critical", `every channel is down (${down.join(", ")})`));
  else if (down.length > 0) out.push(check("delivery", "Alert delivery", "degraded", `down: ${down.join(", ")}; working: ${names.filter((n) => !down.includes(n)).join(", ")}`));
  else out.push(check("delivery", "Alert delivery", "ok", `${String(names.length)} channel(s) working`));
  if (!st.dispatcher_alive || !st.escalator_alive) {
    const which = [!st.dispatcher_alive && "dispatcher", !st.escalator_alive && "escalator"].filter(Boolean).join(" and ");
    out.push(check("delivery-workers", "Delivery workers", "critical", `${which} not running - alerts are not sent or escalated`));
  }
  if (st.oldest_pending_s !== null && st.oldest_pending_s > PENDING_DEGRADED_S) {
    out.push(check("delivery-queue", "Delivery queue", "degraded", `oldest alert waiting ${String(Math.round(st.oldest_pending_s))} s`));
  }
  return out;
}

export function systemHealth(i: HealthInput): HealthReport {
  const checks: HealthCheck[] = [
    i.wsDown
      ? check("feed", "Server connection", "critical", "real-time feed lost - this screen is not live")
      : check("feed", "Server connection", "ok", "live"),
    ...liveChecks(i.live, i.cameraNames ?? {}),
  ];
  if (i.delivery) checks.push(...deliveryChecks(i.delivery));
  if (i.openFaults) {
    const n = i.openFaults.data;
    if (n === undefined) checks.push(check("faults", "System faults", "unknown", i.openFaults.error ? "status request failed" : "waiting for the first status"));
    else checks.push(check("faults", "System faults", n > 0 ? "critical" : "ok", n > 0 ? `${String(n)} open SYSTEM FAULT alarm(s)` : "none open"));
  }
  const problems = checks.filter((c) => c.level !== "ok").sort((a, b) => RANK[b.level] - RANK[a.level]);
  return { level: worst(checks.map((c) => c.level)), checks, problems };
}

export const LEVEL_TEXT: Record<HealthLevel, string> = {
  ok: "All systems OK",
  degraded: "DEGRADED",
  unknown: "STATUS UNKNOWN",
  critical: "ACTION REQUIRED",
};
