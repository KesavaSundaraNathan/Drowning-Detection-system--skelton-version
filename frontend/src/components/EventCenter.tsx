import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import { DROWNING_ALERT } from "../alarmModel";
import { api } from "../api/endpoints";
import type { AlertEvent } from "../api/types";
import { atLeast, useAuth } from "../authModel";
import { useOpenEvents } from "../eventsModel";
import { ago, errorText, useNow } from "../hooks";
import { useLive } from "../state/liveStore";


function EventRow({ ev, now, supervisor }: { ev: AlertEvent; now: number; supervisor: boolean }) {
  const qc = useQueryClient();
  const [note, setNote] = useState("");
  const cameras = useLive((s) => s.cameras);
  // a camera that is not running is not in the live status: fall back to the registered cameras for its name
  const registered = useQuery({ queryKey: ["cameras"], queryFn: api.cameras });
  const where = ev.camera_id
    ? (cameras?.find((c) => c.camera_id === ev.camera_id)?.name ??
      registered.data?.find((c) => c.id === ev.camera_id)?.name ??
      `Camera ${ev.camera_id}`)
    : "system";
  const done = () => {
    void qc.invalidateQueries({ queryKey: ["events"] });
    void qc.invalidateQueries({ queryKey: ["alerts"] });
  };
  const ack = useMutation({ mutationFn: () => api.acknowledgeEvent(ev.id), onSuccess: done });
  const resolve = useMutation({ mutationFn: () => api.resolveEvent(ev.id, note.trim()), onSuccess: done });
  return (
    <li className="event-row" data-testid={`event-${String(ev.id)}`}>
      <span className="event-title">
        {ev.title ?? DROWNING_ALERT} · {ev.alert_reason ?? "Drowning"}
      </span>
      <span className="small">
        {where} · person #{ev.track_id ?? "?"} · {String(ev.alert_count)} {ev.alert_count === 1 ? "alert" : "alerts"} · since{" "}
        {ago(ev.opened_at, now)} · {ev.status}
      </span>
      <span className="actions">
        {ev.status === "open" && (
          <button type="button" className="btn-ack" disabled={ack.isPending} onClick={() => { ack.mutate(); }}>
            Acknowledge
          </button>
        )}
        {supervisor && (
          <>
            <input
              aria-label={`Resolution note for event ${String(ev.id)}`}
              placeholder="Resolution note"
              value={note}
              onChange={(e) => { setNote(e.target.value); }}
            />
            <button
              type="button"
              disabled={resolve.isPending || note.trim() === ""}
              onClick={() => { resolve.mutate(); }}
            >
              Resolve
            </button>
          </>
        )}
      </span>
      {(ack.isError || resolve.isError) && (
        <span role="alert" className="error">
          {errorText(ack.error ?? resolve.error)}
        </span>
      )}
    </li>
  );
}

/**
 * [Phase 5] The AlertCenter's event list: one row per person and incident on ONE camera (cameras are independent).
 * Acknowledge acknowledges every pending alert of the event; a supervisor resolves the event with a note.
 * The database is the source of truth, so the list survives a page reload.
 */
export function EventCenter() {
  const { user } = useAuth();
  const now = useNow(5000);
  const q = useOpenEvents();
  const supervisor = atLeast(user, "supervisor");
  return (
    <section aria-label="Open events" className="event-center">
      <h2>
        Open events <span className="count">{q.data?.length ?? 0}</span>
      </h2>
      {q.isError && (
        <p role="alert" className="error">
          Could not load events: {errorText(q.error)}
        </p>
      )}
      {q.data && q.data.length === 0 && (
        <p className="ok" data-testid="no-events">
          No open events.
        </p>
      )}
      <ul className="event-list">
        {q.data?.map((ev) => <EventRow key={ev.id} ev={ev} now={now} supervisor={supervisor} />)}
      </ul>
    </section>
  );
}
