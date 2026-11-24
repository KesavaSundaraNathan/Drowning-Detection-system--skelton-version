import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Siren } from "lucide-react";
import { useEffect } from "react";
import { Link } from "react-router-dom";

import { alarmTitle, groupByEvent, mostUrgent, ongoingSeconds, useActiveAlarms } from "../alarmModel";
import { api } from "../api/endpoints";
import type { Alert } from "../api/types";
import { ago, useNow } from "../hooks";
import { useLive } from "../state/liveStore";

/**
 * [2.19.0] The alarm nobody can miss on a workstation WITHOUT speakers (owner manual test 2026-11-12: "alerts are
 * passive and silent"). While a drowning alarm is unacknowledged, on EVERY page:
 *   - a red frame pulses around the whole window (pointer-events: none - it never covers or blocks the live feed);
 *   - a red bar at the top names the alarm, the camera, how long ago it rose and how long it has been ONGOING, with
 *     Acknowledge and a link to the camera;
 *   - the browser tab title flashes, so a tab in the background is noticed.
 * It stays until the alarm is acknowledged. One flash per second (below the 3 per second photosensitivity limit);
 * with the OS "reduce motion" setting the frame and bar are solid instead of pulsing.
 */
export function VisualAlarm() {
  const now = useNow(1000);
  const qc = useQueryClient();
  const q = useActiveAlarms();
  const cameras = useLive((s) => s.cameras);
  const registered = useQuery({ queryKey: ["cameras"], queryFn: api.cameras });
  const groups = groupByEvent((q.data ?? []).filter((a) => a.scenario !== "FAULT"));
  const top = mostUrgent(groups);
  const ack = useMutation({
    mutationFn: (a: Alert) => api.acknowledge(a.id),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ["alerts"] });
    },
  });

  const active = top !== undefined;
  useEffect(() => {
    if (!active) return;
    const original = document.title;
    let on = false;
    const t = setInterval(() => {
      on = !on;
      document.title = on ? "⚠ DROWNING ALARM" : original;
    }, 1000);
    return () => {
      clearInterval(t);
      document.title = original;
    };
  }, [active]);

  if (!top) return null;
  const a = top.primary;
  const latest = top.alerts[top.alerts.length - 1] ?? a;
  const where = a.camera_id
    ? (cameras?.find((c) => c.camera_id === a.camera_id)?.name ?? registered.data?.find((c) => c.id === a.camera_id)?.name ?? "camera")
    : "removed camera"; // camera deleted after the alert: its alerts keep no camera id
  const ongoing = ongoingSeconds(latest.reason);
  return (
    <>
      <div className="visual-alarm-frame" aria-hidden="true" data-testid="visual-alarm-frame" />
      <div className="visual-alarm-bar" role="alert" aria-live="assertive" data-testid="visual-alarm-bar">
        <Siren className="visual-alarm-icon" aria-hidden />
        <div className="visual-alarm-text">
          <strong>DROWNING ALARM · {alarmTitle(a)}</strong>
          <span>
            {where}
            {a.zone_name ? ` · ${a.zone_name}` : ""} · raised {ago(a.created_at, now)}
            {ongoing !== null ? ` · ONGOING ${String(ongoing)} s` : ""}
            {groups.length > 1 ? ` · +${String(groups.length - 1)} more` : ""}
          </span>
        </div>
        {a.camera_id && (
          <Link className="visual-alarm-link" to={`/cameras/${a.camera_id}`}>
            View camera
          </Link>
        )}
        <button
          type="button"
          className="visual-alarm-ack"
          disabled={ack.isPending}
          onClick={() => {
            ack.mutate(a);
          }}
        >
          {ack.isPending ? "Acknowledging..." : "Acknowledge"}
        </button>
      </div>
    </>
  );
}
