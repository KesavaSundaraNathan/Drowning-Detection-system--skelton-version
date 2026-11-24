import type { ReactNode } from "react";

import type { LiveSummaryCamera } from "../api/types";
import { MainStream, SnapshotImage } from "./liveFrames";
import { useAlarmCameras } from "../alarmModel";
import { useSkeleton } from "./skeletonOverlay";
import { useViewMode, VIEW_MODES } from "./viewMode";

export type TileVariant = "tile" | "main" | "thumb";

/**
 * One live camera: the server's MJPEG Model View (same-origin <img>, cookie-authenticated) with a
 * red NOT MONITORING overlay ON the video whenever this camera is not analysed, its stream stalled,
 * or the real-time feed is down - a frozen last frame must never look like a calm pool.
 *
 * [v4] variants for the Cameras page video wall (CameraWall): "main" = the large view (high-quality stream,
 * labels drawn over the video like a CCTV monitor), "thumb" = a side preview with a legible label band.
 * Every variant keeps the same overlay, test ids and alt text.
 *
 * [view modes] Full / Clean / Alerts only per camera (components/viewMode.ts), display only: the red alarm ring here,
 * the visual alarm bar and the alarm sound never depend on the mode.
 */
export function ViewModeToggle({ cameraId, className = "view-modes", buttonClass = "wall-btn" }: { cameraId: string; className?: string; buttonClass?: string }) {
  const [mode, setMode] = useViewMode(cameraId);
  const [skeleton, setSkeleton] = useSkeleton(cameraId);
  return (
    <span className={className} role="group" aria-label="View mode">
      {VIEW_MODES.map((m) => (
        <button key={m.mode} type="button" className={buttonClass} aria-pressed={mode === m.mode} title={m.title} onClick={() => { setMode(m.mode); }}>
          {m.label}
        </button>
      ))}
      {/* [rtmpose clone] skeleton on / off, independent of the mode (Clean still hides it, Alerts only shows only the alerting person's) */}
      <button type="button" className={buttonClass} aria-pressed={skeleton} title="Show each person's RTMPose skeleton" onClick={() => { setSkeleton(!skeleton); }}>
        Skeleton
      </button>
    </span>
  );
}

export function CameraTile({
  camera,
  notMonitored,
  feedDown,
  large = false,
  variant = "tile",
  index,
  actions,
}: {
  camera: LiveSummaryCamera;
  notMonitored: boolean;
  feedDown: boolean;
  large?: boolean;
  variant?: TileVariant;
  index?: number; // 1-based camera number shown as "CAM n" (wall variants)
  actions?: ReactNode; // main variant: buttons drawn in the bottom bar over the video
}) {
  const overlay = notMonitored || feedDown;
  const alarm = useAlarmCameras().has(camera.camera_id); // [2.19.0] red ring while its alarm is unacknowledged
  const reason = feedDown ? "no connection to the server" : camera.stalled ? "stream stalled" : "not analysed";
  const chips = (
    <>
      <span className={`chip ${overlay ? "chip-bad" : "chip-ok"}`}>{overlay ? "NOT MONITORING" : "MONITORED"}</span>
      {camera.scene_type === "standard_cctv" && (
        <span className="chip chip-muted" title="Standard CCTV: people are detected, no drowning alerts">
          CCTV - NO DROWNING ALERTS
        </span>
      )}
      {camera.lagging && <span className="chip chip-warn">LAGGING</span>}
      {camera.muted_zones.map((z) => (
        <span key={z} className="chip chip-muted">
          MUTED: {z}
        </span>
      ))}
    </>
  );
  // [v4] wall variants: the main view holds the one live stream, previews poll snapshots (components/liveFrames.tsx)
  const img =
    variant === "main" ? (
      <MainStream cameraId={camera.camera_id} name={camera.name} />
    ) : variant === "thumb" ? (
      <SnapshotImage cameraId={camera.camera_id} name={camera.name} />
    ) : (
      <MainStream cameraId={camera.camera_id} name={camera.name} quality="normal" />
    );
  const nm = overlay && (
    <div className="cam-overlay" role="alert" data-testid={`camera-overlay-${camera.camera_id}`}>
      NOT MONITORING
      <small>{reason}</small>
    </div>
  );

  if (variant === "tile") {
    return (
      <figure className={`cam ${large ? "cam-large" : ""}${alarm ? " cam-alarm" : ""}`} data-testid={`camera-${camera.camera_id}`}>
        <div className="cam-video">
          {img}
          {nm}
        </div>
        <figcaption>
          <strong>{camera.name}</strong> {chips}
          <ViewModeToggle cameraId={camera.camera_id} className="cam-modes" buttonClass="" />
        </figcaption>
      </figure>
    );
  }
  const label = index !== undefined ? `CAM ${String(index)}` : null;
  return (
    <figure className={`wall-tile wall-${variant}${overlay ? " wall-down" : ""}${alarm ? " cam-alarm" : ""}`} data-testid={`camera-${camera.camera_id}`}>
      <div className="wall-video">
        {img}
        <figcaption className="wall-label">
          {label && <span className="wall-cam-no">{label}</span>}
          <strong className="wall-name">{camera.name}</strong>
          <span className="wall-chips">{chips}</span>
        </figcaption>
        {nm}
        {variant === "main" && (
          <div className="wall-actions">
            <ViewModeToggle cameraId={camera.camera_id} />
            {actions}
          </div>
        )}
      </div>
    </figure>
  );
}
