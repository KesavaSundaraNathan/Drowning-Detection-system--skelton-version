import * as Dialog from "@radix-ui/react-dialog";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import { alarmTitle } from "../alarmModel";
import { api } from "../api/endpoints";
import type { Alert, AlertStatus } from "../api/types";
import { atLeast, useAuth } from "../authModel";
import { EscalationBadge } from "../components/alarms";
import { EventCenter } from "../components/EventCenter";
import { MediaViewer } from "../components/MediaViewer";
import { ago, errorText, useNow } from "../hooks";

const FILTERS: (AlertStatus | "all")[] = ["pending", "acknowledged", "resolved", "false_alarm", "all"];

function CloseDialog({ alert, kind, onDone }: { alert: Alert; kind: "resolve" | "false_alarm"; onDone: () => void }) {
  const qc = useQueryClient();
  const [note, setNote] = useState("");
  const m = useMutation({
    mutationFn: () => (kind === "resolve" ? api.resolve(alert.id, note) : api.falseAlarm(alert.id, note)),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ["alerts"] });
      onDone();
    },
  });
  return (
    <Dialog.Root
      open
      onOpenChange={(o) => {
        if (!o) onDone();
      }}
    >
      <Dialog.Portal>
        <Dialog.Overlay className="dialog-overlay" />
        <Dialog.Content className="dialog" aria-describedby={undefined}>
          <Dialog.Title>
            {kind === "resolve" ? "Resolve" : "Mark as false alarm"}: alert {alert.id}
          </Dialog.Title>
          <label>
            Note (required, audited)
            <textarea value={note} onChange={(e) => { setNote(e.target.value); }} rows={3} />
          </label>
          <button type="button" className="btn-primary" disabled={!note.trim() || m.isPending} onClick={() => { m.mutate(); }}>
            Save
          </button>
          {m.isError && (
            <p role="alert" className="error">
              {errorText(m.error)}
            </p>
          )}
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}

function ClearAllDialog({ count, onDone }: { count: number; onDone: () => void }) {
  const qc = useQueryClient();
  const [note, setNote] = useState("");
  const m = useMutation({
    mutationFn: () => api.clearAllAlerts(note),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ["alerts"] });
      onDone();
    },
  });
  return (
    <Dialog.Root
      open
      onOpenChange={(o) => {
        if (!o) onDone();
      }}
    >
      <Dialog.Portal>
        <Dialog.Overlay className="dialog-overlay" />
        <Dialog.Content className="dialog" aria-describedby={undefined}>
          <Dialog.Title>Clear all alerts</Dialog.Title>
          <p>
            This resolves every open alert (at least {count} in this list), including ones nobody has
            looked at. Alerts are kept in the history and the action is audited. Only continue if no one is in danger.
          </p>
          <label>
            Note (required, audited)
            <textarea value={note} onChange={(e) => { setNote(e.target.value); }} rows={3} />
          </label>
          <button type="button" className="btn-primary" disabled={!note.trim() || m.isPending} onClick={() => { m.mutate(); }}>
            Clear all alerts
          </button>
          {m.isError && (
            <p role="alert" className="error">
              {errorText(m.error)}
            </p>
          )}
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}

export default function AlertsPage() {
  const { user } = useAuth();
  const qc = useQueryClient();
  const now = useNow(5000);
  const [filter, setFilter] = useState<AlertStatus | "all">("pending");
  const [viewing, setViewing] = useState<Alert | null>(null);
  const [clearing, setClearing] = useState(false);
  const [closing, setClosing] = useState<{ alert: Alert; kind: "resolve" | "false_alarm" } | null>(null);
  const alerts = useQuery({
    queryKey: ["alerts", "list", filter],
    queryFn: () => api.alerts(filter === "all" ? { limit: 200 } : { status: filter, limit: 200 }),
  });
  const ack = useMutation({
    mutationFn: api.acknowledge,
    onSuccess: () => void qc.invalidateQueries({ queryKey: ["alerts"] }),
  });
  const supervisor = atLeast(user, "supervisor");
  const openCount = (alerts.data ?? []).filter((a) => a.status === "pending" || a.status === "acknowledged").length;
  return (
    <div>
      <h1>Alerts</h1>
      <EventCenter />
      {supervisor && openCount > 0 && (
        <button type="button" className="btn-danger" onClick={() => { setClearing(true); }}>
          Clear all alerts
        </button>
      )}
      <div className="filters" role="group" aria-label="Status filter">
        {FILTERS.map((f) => (
          <button key={f} type="button" aria-pressed={filter === f} onClick={() => { setFilter(f); }}>
            {f.replace("_", " ")}
          </button>
        ))}
      </div>
      {alerts.isError && (
        <p role="alert" className="error">
          {errorText(alerts.error)}
        </p>
      )}
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>When</th>
            <th>Alert</th>
            <th>Where</th>
            <th>Status</th>
            <th>Escalation</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {alerts.data?.map((a) => (
            <tr key={a.id} className={a.status === "pending" && a.audible ? "row-pending" : ""}>
              <td>{a.id}</td>
              <td title={a.created_at}>{ago(a.created_at, now)}</td>
              <td>
                {alarmTitle(a)}
                {!a.audible && <span className="chip chip-muted">{"MUTED"}</span>}
                <div className="small">{a.reason}</div>
              </td>
              <td>
                {a.camera_id ?? a.fault_source ?? "system"}
                {typeof a.event_id === "number" && (
                  <div className="small" data-testid={`alert-event-${String(a.id)}`}>
                    event #{a.event_id}
                    {a.track_id !== null ? ` · person #${String(a.track_id)}` : ""}
                  </div>
                )}
              </td>
              <td>
                {a.status}
                {a.cleared_at && <div className="small">cleared {ago(a.cleared_at, now)}</div>}
              </td>
              <td>
                <EscalationBadge alert={a} now={now} />
              </td>
              <td className="actions">
                {a.status === "pending" && (
                  <button type="button" className="btn-ack" onClick={() => { ack.mutate(a.id); }}>
                    Acknowledge
                  </button>
                )}
                {(a.snapshot_available || a.recording_available) && (
                  <button type="button" onClick={() => { setViewing(a); }}>
                    Evidence
                  </button>
                )}
                {supervisor && (a.status === "pending" || a.status === "acknowledged") && (
                  <>
                    <button type="button" onClick={() => { setClosing({ alert: a, kind: "resolve" }); }}>
                      Resolve
                    </button>
                    <button type="button" onClick={() => { setClosing({ alert: a, kind: "false_alarm" }); }}>
                      False alarm
                    </button>
                  </>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {viewing && <MediaViewer alert={viewing} onClose={() => { setViewing(null); }} />}
      {clearing && <ClearAllDialog count={openCount} onDone={() => { setClearing(false); }} />}
      {closing && <CloseDialog alert={closing.alert} kind={closing.kind} onDone={() => { setClosing(null); }} />}
    </div>
  );
}
