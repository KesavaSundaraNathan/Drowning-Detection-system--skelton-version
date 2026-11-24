import { useMutation, useQuery } from "@tanstack/react-query";

import { alarmTitle } from "../alarmModel";
import { api } from "../api/endpoints";
import { atLeast, useAuth } from "../authModel";
import { ago, errorText, useNow } from "../hooks";
import { LEVEL_TEXT } from "../state/health";
import { useSystemHealth } from "../state/useSystemHealth";

/**
 * System health: the overall verdict (state/health.ts) first, then its sources - live runtime and cameras,
 * open faults, out-of-band delivery + self-test, release gate. Every status query shows its own error:
 * a source that does not answer is "STATUS UNKNOWN", never a reassuring dash.
 */
export default function SystemPage() {
  const { user } = useAuth();
  const now = useNow(5000);
  const health = useSystemHealth();
  const live = useQuery({ queryKey: ["live", "status"], queryFn: api.liveStatus, refetchInterval: 5000 });
  const delivery = useQuery({ queryKey: ["delivery"], queryFn: api.delivery, refetchInterval: 5000 });
  const release = useQuery({ queryKey: ["release"], queryFn: api.release });
  const faults = useQuery({
    queryKey: ["alerts", "faults"],
    queryFn: async () => (await api.alerts({ limit: 200 })).filter((a) => a.scenario === "FAULT" && a.cleared_at === null),
  });
  const selfTest = useMutation({ mutationFn: api.selfTest });
  const cap = live.data?.capacity;
  const cams = live.data?.cameras ?? [];
  const cameraList = useQuery({ queryKey: ["cameras"], queryFn: api.cameras, staleTime: 60_000 });
  const nameOf = (id: string) => cameraList.data?.find((c) => c.id === id)?.name ?? id;
  return (
    <div className="system">
      <h1>System health</h1>
      <section className={`card health-card health-${health.level}`} aria-label="Overall status">
        <h2>
          Overall: <span className="health-word">{LEVEL_TEXT[health.level]}</span>
        </h2>
        <table aria-label="Health checks">
          <thead>
            <tr>
              <th>Check</th>
              <th>Status</th>
              <th>Detail</th>
            </tr>
          </thead>
          <tbody>
            {[...health.problems, ...health.checks.filter((c) => c.level === "ok")].map((c) => (
              <tr key={c.key} className={`lvl-${c.level}`}>
                <td>{c.label}</td>
                <td>{c.level.toUpperCase()}</td>
                <td>{c.detail}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
      <section className="card">
        <h2>Release gate</h2>
        <p>
          <strong>{release.data?.mode ?? "ASSIST_PILOT"}</strong> - {release.data?.reason || release.data?.message || "status unavailable"}
        </p>
        {release.data?.criteria && release.data.criteria.length > 0 && (
          <table className="gate-criteria" aria-label="Release gate criteria">
            <caption>
              Signed report{release.data.site ? ` for ${release.data.site}` : ""}
              {release.data.created_at ? `, ${release.data.created_at}` : ""}
            </caption>
            <thead>
              <tr>
                <th>Criterion</th>
                <th>Target</th>
                <th>Measured</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {release.data.criteria.map((c) => (
                <tr key={c.key} className={c.status === "MET" ? "ok" : "error"}>
                  <td>{c.key}</td>
                  <td>{c.target}</td>
                  <td>{JSON.stringify(c.value)}</td>
                  <td>
                    {c.status}
                    {c.note ? ` (${c.note})` : ""}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>
      <section className="card">
        <h2>Open faults ({faults.data?.length ?? "-"})</h2>
        {faults.isError && <p className="error">Status unavailable: {errorText(faults.error)}</p>}
        {faults.data?.length === 0 && <p className="ok">No open fault.</p>}
        <ul>
          {faults.data?.map((f) => (
            <li key={f.id}>
              <strong>{alarmTitle(f)}</strong> - {f.reason} ({ago(f.created_at, now)}, {f.status})
            </li>
          ))}
        </ul>
      </section>
      <section className="card">
        <h2>Live monitoring</h2>
        {live.isError && <p className="error">Status unavailable: {errorText(live.error)}</p>}
        {live.data?.enabled === false ? (
          <p className="error">Live monitoring is OFF.</p>
        ) : (
          live.data && (
            <dl className="facts">
              <dt>Runtime</dt>
              <dd className={live.data.error || live.data.running === false ? "error" : "ok"}>
                {live.data.error ? `failed: ${live.data.error}` : live.data.running === false ? "NOT running" : "running"}
              </dd>
              <dt>Inference</dt>
              <dd>
                {live.data.provider ?? "-"}
                {live.data.model ? ` (${live.data.model})` : ""}
              </dd>
              <dt>Capacity</dt>
              <dd>{cap ? `${String(cap.limit)} camera(s) at ${String(cap.target_fps)} fps` : "-"}</dd>
              <dt>GPU scheduler</dt>
              <dd className={live.data.scheduler?.stalled ? "error" : undefined}>
                {live.data.scheduler
                  ? `${live.data.scheduler.stalled ? "STALLED" : "running"}, mean batch ${String(live.data.scheduler.mean_batch)}, last round ${String(live.data.scheduler.last_round_ms)} ms`
                  : "-"}
              </dd>
              <dt>Supervisor service</dt>
              <dd className={live.data.supervisor_down ? "error" : "ok"}>{live.data.supervisor_down ? "DOWN" : "ok"}</dd>
            </dl>
          )
        )}
        {(cams.length > 0 || Object.keys(live.data?.not_started ?? {}).length > 0) && (
          <table aria-label="Cameras">
            <thead>
              <tr>
                <th>Camera</th>
                <th>Monitoring</th>
                <th>Capture</th>
                <th>Last frame</th>
                <th>Lag</th>
                <th>Reconnects</th>
              </tr>
            </thead>
            <tbody>
              {cams.map((c) => {
                const bad = c.stalled || c.monitoring === "NOT_MONITORED";
                return (
                  <tr key={c.camera_id} className={bad ? "lvl-critical" : c.lagging ? "lvl-degraded" : "lvl-ok"}>
                    <td>{c.name}</td>
                    <td>{c.stalled ? "STALLED" : c.monitoring === "MONITORED" ? "monitored" : "NOT MONITORED"}</td>
                    <td>{c.capture.fps.toFixed(1)} fps</td>
                    <td>{c.capture.last_frame_age_s.toFixed(1)} s ago</td>
                    <td>{c.lagging ? `${c.lag_s.toFixed(1)} s (lagging)` : `${c.lag_s.toFixed(1)} s`}</td>
                    <td>{c.capture.reconnects}</td>
                  </tr>
                );
              })}
              {Object.entries(live.data?.not_started ?? {}).map(([id, reason]) => (
                <tr key={id} className="lvl-critical">
                  <td>{nameOf(id)}</td>
                  <td>NOT STARTED</td>
                  <td colSpan={4}>{reason === "capacity" ? "GPU capacity reached" : reason === "invalid_source" ? "invalid video source" : reason}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>
      <section className="card">
        <h2>Alert delivery</h2>
        {delivery.isError && <p className="error">{errorText(delivery.error)}</p>}
        <table>
          <thead>
            <tr>
              <th>Channel</th>
              <th>State</th>
              <th>Pending</th>
              <th>Last success</th>
              <th>Last error</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(delivery.data?.channels ?? {}).map(([name, c]) => (
              <tr key={name}>
                <td>{name}</td>
                <td className={c.down ? "error" : "ok"}>{c.down ? "DOWN" : "ok"}</td>
                <td>
                  {c.pending}
                  {c.oldest_pending_s !== null ? ` (oldest ${String(c.oldest_pending_s)} s)` : ""}
                </td>
                <td>{ago(c.last_success_at, now)}</td>
                <td className="small">{c.last_error ?? ""}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {delivery.data && Object.keys(delivery.data.channels).length === 0 && (
          <p className="error">No out-of-band channel is configured: alarms are NOT delivered off-site.</p>
        )}
        <p>
          Escalation tiers: {delivery.data?.escalation_tiers ?? "-"} · Daily self-test at {delivery.data?.self_test.time ?? "-"} (
          {delivery.data?.self_test.enabled ? "on" : "off"})
        </p>
        {delivery.data?.self_test.last && (
          <p>
            Last self-test {delivery.data.self_test.last.event_id}:{" "}
            {Object.entries(delivery.data.self_test.last.results)
              .map(([ch, r]) => `${ch} ${r === null ? "OK" : `FAILED (${r})`}`)
              .join(", ")}
          </p>
        )}
        {atLeast(user, "admin") && (
          <button type="button" onClick={() => { selfTest.mutate(); }} disabled={selfTest.isPending}>
            Run channel self-test now
          </button>
        )}
        {selfTest.isSuccess && <p className="ok">Self-test queued: {selfTest.data.event_id}</p>}
        {selfTest.isError && <p className="error">{errorText(selfTest.error)}</p>}
      </section>
    </div>
  );
}
