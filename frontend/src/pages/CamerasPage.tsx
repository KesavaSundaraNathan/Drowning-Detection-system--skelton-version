import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import { ApiError } from "../api/client";
import { api } from "../api/endpoints";
import type { CapacityError, LiveCamera, SceneType } from "../api/types";
import { atLeast, useAuth } from "../authModel";
import { CameraWall } from "../components/CameraWall";
import { errorText, useNow } from "../hooks";
import { useLive } from "../state/liveStore";
import { verdict, WS_GRACE_MS } from "../state/monitoring";

function startError(e: unknown): string {
  if (e instanceof ApiError && e.status === 409) {
    const d = e.detail as CapacityError;
    if (d.behind)
      return `Refused: camera "${d.behind}" is already falling behind (${String(d.analysed_fps)} of ${String(d.target_fps)} fps analysed). Stop a camera, or set cameras that see no pool to Standard CCTV, before adding another.`;
    return `Refused: this server can analyse at most ${String(d.limit)} camera(s) at ${String(d.target_fps)} fps (camera ${String(d.requested)} would get ${String(d.analysed_fps)} fps).`;
  }
  return errorText(e);
}

const SCENE_LABEL: Record<SceneType, string> = { pool: "Pool area", standard_cctv: "Standard CCTV" };

/**
 * [v4] Pool area (drowning detection) vs standard CCTV (people only). The server only SUGGESTS a type from the
 * picture; switching drowning detection off is always an admin's click, confirmed.
 */
function SceneCell({
  cameraId,
  name,
  sceneType,
  setBy,
  live,
  isAdmin,
}: {
  cameraId: string;
  name: string;
  sceneType: SceneType;
  setBy: "default" | "admin";
  live: LiveCamera | undefined;
  isAdmin: boolean;
}) {
  const qc = useQueryClient();
  const patch = useMutation({
    mutationFn: (t: SceneType) => api.patchCamera(cameraId, { scene_type: t }),
    onSettled: () => {
      void qc.invalidateQueries({ queryKey: ["cameras"] });
      void qc.invalidateQueries({ queryKey: ["live"] });
    },
  });
  const choose = (t: SceneType) => {
    if (t === sceneType) return;
    if (
      t === "standard_cctv" &&
      !window.confirm(`Set ${name} to standard CCTV? Drowning detection will be OFF for this camera: people are still detected, but no drowning alert can fire.`)
    )
      return;
    patch.mutate(t);
  };
  const s = live?.scene?.suggestion;
  return (
    <div>
      {isAdmin ? (
        <select
          aria-label={`Area type of ${name}`}
          value={sceneType}
          disabled={patch.isPending}
          onChange={(e) => { choose(e.target.value as SceneType); }}
        >
          <option value="pool">{SCENE_LABEL.pool}</option>
          <option value="standard_cctv">{SCENE_LABEL.standard_cctv}</option>
        </select>
      ) : (
        SCENE_LABEL[sceneType]
      )}
      {setBy === "default" && <small> (default)</small>}
      {s && (
        <div>
          <small>
            Auto check: {s.suggested ? SCENE_LABEL[s.suggested] : "undecided"} - {s.reason}
          </small>
          {isAdmin && s.suggested && s.suggested !== sceneType && (
            <button type="button" onClick={() => { if (s.suggested) choose(s.suggested); }} disabled={patch.isPending}>
              Use {SCENE_LABEL[s.suggested]}
            </button>
          )}
        </div>
      )}
      {patch.isError && (
        <p role="alert" className="error">
          {errorText(patch.error)}
        </p>
      )}
    </div>
  );
}

/** Admin: register a camera. The RTSP URL (with its credentials) is stored encrypted and never shown again. */
function AddCamera() {
  const qc = useQueryClient();
  const [name, setName] = useState("");
  const [source, setSource] = useState("");
  const add = useMutation({
    mutationFn: () =>
      api.createCamera({ name, source_url: source, default_mode: "Medium" }),
    onSuccess: () => {
      setName("");
      setSource("");
      void qc.invalidateQueries({ queryKey: ["cameras"] });
    },
  });
  return (
    <form
      className="card"
      onSubmit={(e) => {
        e.preventDefault();
        add.mutate();
      }}
    >
      <h2>Add camera</h2>
      <label>
        Name <input value={name} onChange={(e) => { setName(e.target.value); }} required />
      </label>
      <label>
        RTSP URL (rtsp://user:password@host/stream)
        <input value={source} onChange={(e) => { setSource(e.target.value); }} required autoComplete="off" />
      </label>
      <button type="submit" className="btn-primary" disabled={add.isPending}>
        Add
      </button>
      {add.isError && (
        <p role="alert" className="error">
          {errorText(add.error)}
        </p>
      )}
    </form>
  );
}

export default function CamerasPage() {
  const { user } = useAuth();
  const qc = useQueryClient();
  const now = useNow(1000);
  const state = useLive((s) => s);
  const v = verdict(state, now);
  const feedDown = state.ws === "down" && state.wsDownSince !== null && now - state.wsDownSince > WS_GRACE_MS;
  const cameras = useQuery({ queryKey: ["cameras"], queryFn: api.cameras });
  // [v4] per-camera scene suggestion + people count (not in the WebSocket summary: it changes every few seconds)
  const liveStatus = useQuery({ queryKey: ["live", "status"], queryFn: api.liveStatus, refetchInterval: 5000 });
  const liveById = new Map((liveStatus.data?.cameras ?? []).map((c) => [c.camera_id, c]));
  const running = new Set(state.cameras?.map((c) => c.camera_id));
  const refresh = () => void qc.invalidateQueries({ queryKey: ["live"] });
  const start = useMutation({ mutationFn: api.startCamera, onSettled: refresh });
  const stop = useMutation({ mutationFn: api.stopCamera, onSettled: refresh });
  const remove = useMutation({
    mutationFn: api.deleteCamera,
    onSettled: () => {
      void qc.invalidateQueries({ queryKey: ["cameras"] });
      refresh();
    },
  });
  const canControl = atLeast(user, "supervisor");

  return (
    <div>
      <h1>Cameras</h1>
      {start.isError && (
        <p role="alert" className="error">
          {startError(start.error)}
        </p>
      )}
      <CameraWall
        cameras={state.cameras ?? []}
        notMonitored={(id) => v.cameras.has(id)}
        feedDown={feedDown}
        canControl={canControl}
        onStop={(c) => {
          if (window.confirm(`Stop monitoring ${c.name}? The pool area will NOT be watched.`)) stop.mutate(c.camera_id);
        }}
      />
      <h2>Registered cameras</h2>
      <table>
        <thead>
          <tr>
            <th>Name</th>
            <th>Source</th>
            <th>Area</th>
            <th>Enabled</th>
            <th>Live</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {cameras.data?.map((c) => (
            <tr key={c.id}>
              <td>{c.name}</td>
              <td>{c.source_type}</td>
              <td>
                <SceneCell
                  cameraId={c.id}
                  name={c.name}
                  sceneType={c.scene_type}
                  setBy={c.scene_type_set_by}
                  live={liveById.get(c.id)}
                  isAdmin={atLeast(user, "admin")}
                />
              </td>
              <td>{c.is_enabled ? "yes" : "no"}</td>
              <td>
                {running.has(c.id) ? "running" : state.notStarted.includes(c.id) ? "NOT STARTED" : "stopped"}
                {liveById.get(c.id)?.scene && <small> - {liveById.get(c.id)?.scene?.people} people</small>}
              </td>
              <td>
                {canControl && c.is_enabled && !running.has(c.id) && (
                  <button type="button" onClick={() => { start.mutate(c.id); }} disabled={start.isPending}>
                    Start monitoring
                  </button>
                )}
                {atLeast(user, "admin") && (
                  <button
                    type="button"
                    onClick={() => {
                      if (window.confirm(`Remove camera ${c.name}? It will stop being monitored and its zones are deleted.`))
                        remove.mutate(c.id);
                    }}
                    disabled={remove.isPending}
                  >
                    Remove camera
                  </button>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {atLeast(user, "admin") && <AddCamera />}
    </div>
  );
}
