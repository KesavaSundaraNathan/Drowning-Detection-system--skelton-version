import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { AlarmClock, X } from "lucide-react";
import { useEffect, useState } from "react";

import { alarmTitle, groupByEvent, useActiveAlarms } from "../alarmModel";
import { api, media } from "../api/endpoints";
import type { Alert } from "../api/types";
import { ago, errorText, useNow } from "../hooks";
import { useLive } from "../state/liveStore";
import { MediaViewer } from "./MediaViewer";

export function EscalationBadge({ alert, now }: { alert: Alert; now: number }) {
  if (alert.escalation_level <= 0) return null;
  return (
    <span className="badge badge-escalated" data-testid={`escalated-${String(alert.id)}`}>
      ESCALATED x{alert.escalation_level} · last {ago(alert.escalated_at, now)}
    </span>
  );
}

export function AlarmCard({ alert, now, linked = [] }: { alert: Alert; now: number; linked?: Alert[] }) {
  const qc = useQueryClient();
  const [viewing, setViewing] = useState(false);
  const ack = useMutation({
    mutationFn: () => api.acknowledge(alert.id),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ["alerts"] });
    },
  });
  const fault = alert.scenario === "FAULT";
  const cameras = useLive((s) => s.cameras);
  // [Phase 11] a camera that is NOT running (stopped, or down - exactly when it matters) is not in the live
  // status: fall back to the registered cameras for its name
  const registered = useQuery({ queryKey: ["cameras"], queryFn: api.cameras });
  const nameOf = (a: Alert) =>
    a.camera_id
      ? (cameras?.find((c) => c.camera_id === a.camera_id)?.name ??
        registered.data?.find((c) => c.id === a.camera_id)?.name ??
        `Camera ${a.camera_id}`)
      : (a.fault_source ?? "system");
  const where = nameOf(alert);
  const count = 1 + linked.length;
  return (
    <article
      className={`alarm ${fault ? "alarm-fault" : "alarm-critical"}`}
      aria-label={`Alarm ${String(alert.id)}`}
      data-testid={`alarm-${String(alert.id)}`}
    >
      {alert.snapshot_available && (
        <button type="button" className="thumb" onClick={() => { setViewing(true); }} aria-label="View snapshot and recording">
          <img src={media.snapshot(alert.id)} alt={`Alert ${String(alert.id)} snapshot`} loading="lazy" />
        </button>
      )}
      <div className="alarm-body">
        <h3>{alarmTitle(alert)}</h3>
        <p className="alarm-meta">
          {where}
          {alert.zone_name ? ` · ${alert.zone_name}` : ""} · {ago(alert.created_at, now)}
          {alert.cleared_at ? " · condition cleared" : ""}
        </p>
        <p className="alarm-reason">{alert.reason}</p>
        {count > 1 && (
          <p className="alarm-person" data-testid={`event-count-${String(alert.id)}`}>
            {String(count)} alerts in this event (same person, this camera)
          </p>
        )}
        {linked.length > 0 && (
          <ul className="alarm-linked" aria-label="Other alerts of this event">
            {linked.map((l) => (
              <li key={l.id} data-testid={`linked-${String(l.id)}`}>
                #{l.id} {alarmTitle(l)} · {nameOf(l)} · {ago(l.created_at, now)}
              </li>
            ))}
          </ul>
        )}
        <EscalationBadge alert={alert} now={now} />
      </div>
      <div className="alarm-actions">
        <button type="button" className="btn-ack" disabled={ack.isPending} onClick={() => { ack.mutate(); }}>
          {ack.isPending ? "Acknowledging..." : count > 1 ? `Acknowledge event (${String(count)} alerts)` : "Acknowledge"}
        </button>
        {(alert.snapshot_available || alert.recording_available) && (
          <button type="button" onClick={() => { setViewing(true); }}>
            Evidence
          </button>
        )}
        {ack.isError && (
          <span role="alert" className="error">
            {errorText(ack.error)}
          </span>
        )}
      </div>
      {viewing && <MediaViewer alert={alert} onClose={() => { setViewing(false); }} />}
    </article>
  );
}

/** The alarm-clock button at the top right of the Dashboard: a red badge counts the alarms waiting for an acknowledge. */
export function AlarmBell({ open, onToggle }: { open: boolean; onToggle: () => void }) {
  const q = useActiveAlarms();
  const count = q.data ? groupByEvent(q.data).length : 0;
  const label = q.isError ? "Active alarms: could not load" : `Active alarms: ${String(count)}`;
  return (
    <button
      type="button"
      className={`alarm-bell${count > 0 ? " alarm-bell-ringing" : ""}`}
      aria-label={label}
      aria-expanded={open}
      aria-controls="alarm-panel"
      title={label}
      onClick={onToggle}
    >
      <AlarmClock className="h-6 w-6" aria-hidden />
      {(count > 0 || q.isError) && (
        <span className="alarm-bell-badge" data-testid="alarm-bell-badge">
          {q.isError ? "!" : count > 99 ? "99+" : String(count)}
        </span>
      )}
    </button>
  );
}

/** The list under the alarm clock: every active alarm, with its Acknowledge. Escape or Close shuts it. */
export function AlarmPanel({ onClose }: { onClose: () => void }) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && !document.querySelector(".media-viewer")) onClose(); // not while the evidence viewer is open
    };
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("keydown", onKey);
    };
  }, [onClose]);
  return (
    <div id="alarm-panel" className="alarm-panel">
      <button type="button" className="alarm-panel-close" aria-label="Close alarm list" onClick={onClose}>
        <X className="h-4 w-4" aria-hidden />
      </button>
      <ActiveAlarms />
    </div>
  );
}

/**
 * The Dashboard's alarms without the clutter: the oldest alarm on top of a stack (its Acknowledge works right
 * there), the others as cards behind it, and a button that opens the full list under the alarm clock.
 * Nothing at all when no alarm is waiting.
 */
export function AlarmStack({ onOpenList }: { onOpenList: () => void }) {
  const now = useNow(1000);
  const q = useActiveAlarms();
  if (!q.data) return null;
  const alarms = groupByEvent(q.data);
  const top = alarms[0];
  if (!top) return null;
  const behind = Math.min(alarms.length - 1, 2);
  return (
    <section aria-label="Alarm stack" className="alarm-stack-wrap">
      <div className={`alarm-stack alarm-stack-depth-${String(behind)}`}>
        {Array.from({ length: behind }, (_, i) => (
          <div key={i} className={`alarm-stack-layer alarm-stack-layer-${String(i + 1)}`} aria-hidden="true" />
        ))}
        <AlarmCard key={top.key} alert={top.primary} linked={top.alerts.slice(1)} now={now} />
      </div>
      <div className="alarm-stack-foot">
        <span>
          {alarms.length > 1 ? `+${String(alarms.length - 1)} more ${alarms.length === 2 ? "alarm" : "alarms"} waiting` : ""}
        </span>
        <button type="button" className="btn-primary" onClick={onOpenList}>
          Open alert list ({alarms.length})
        </button>
      </div>
    </section>
  );
}

export function ActiveAlarms() {
  const now = useNow(1000);
  const q = useActiveAlarms();
  if (q.isPending) return <p>Loading alarms...</p>;
  if (q.isError) {
    return (
      <p role="alert" className="error">
        Could not load alarms: {errorText(q.error)}
      </p>
    );
  }
  const alarms = groupByEvent(q.data); // oldest unacknowledged first, one card per event [Phase 5]
  return (
    <section aria-label="Active alarms" className="alarms">
      <h2>
        Active alarms <span className="count">{alarms.length}</span>
      </h2>
      {alarms.length === 0 ? (
        <p className="ok" data-testid="no-alarms">
          No unacknowledged alarms.
        </p>
      ) : (
        alarms.map((g) => <AlarmCard key={g.key} alert={g.primary} linked={g.alerts.slice(1)} now={now} />)
      )}
    </section>
  );
}
