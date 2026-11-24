import { useRef, useState } from "react";
import { Link } from "react-router-dom";

import type { LiveSummaryCamera } from "../api/types";
import { CameraTile } from "./CameraTile";

const SELECTED_KEY = "lifeguard.cameraWall.main";

function readSelected(): string | null {
  try {
    return localStorage.getItem(SELECTED_KEY);
  } catch {
    return null;
  }
}

/**
 * [v4] Cameras page video wall: ONE large main view (high-quality stream) and the other cameras as side previews.
 * Clicking a preview swaps it into the main view; the camera that was there moves to the side. Every view keeps the
 * NOT MONITORING overlay (CameraTile), and the choice is remembered in this browser. Numbers (CAM 1..n) follow the
 * running cameras' order, so a camera keeps its number wherever it is shown.
 */
export function CameraWall({
  cameras,
  notMonitored,
  feedDown,
  canControl,
  onStop,
}: {
  cameras: LiveSummaryCamera[];
  notMonitored: (cameraId: string) => boolean;
  feedDown: boolean;
  canControl: boolean;
  onStop: (camera: LiveSummaryCamera) => void;
}) {
  const [selected, setSelected] = useState<string | null>(readSelected);
  const mainRef = useRef<HTMLDivElement>(null);
  const main = cameras.find((c) => c.camera_id === selected) ?? cameras[0];
  if (main === undefined) return null;
  const side = cameras.filter((c) => c.camera_id !== main.camera_id);
  const number = (c: LiveSummaryCamera) => cameras.indexOf(c) + 1;

  const choose = (id: string) => {
    setSelected(id);
    try {
      localStorage.setItem(SELECTED_KEY, id);
    } catch {
      /* not remembered: fine */
    }
  };
  const fullscreen = () => {
    const el = mainRef.current;
    if (!el) return;
    if (document.fullscreenElement) void document.exitFullscreen();
    else void el.requestFullscreen();
  };

  return (
    <section className={`wall ${side.length === 0 ? "wall-single" : ""}`} aria-label="Running cameras">
      <div className="wall-main-slot" ref={mainRef} key={main.camera_id}>
        <CameraTile
          camera={main}
          notMonitored={notMonitored(main.camera_id)}
          feedDown={feedDown}
          variant="main"
          index={number(main)}
          actions={
            <>
              <button type="button" className="wall-btn" onClick={fullscreen} title="Full screen (Esc to leave)">
                Full screen
              </button>
              <Link className="wall-btn" to={`/cameras/${encodeURIComponent(main.camera_id)}`}>
                Details
              </Link>
              {canControl && (
                <button type="button" className="wall-btn wall-btn-danger" onClick={() => { onStop(main); }}>
                  Stop monitoring
                </button>
              )}
            </>
          }
        />
      </div>
      {side.length > 0 && (
        <ul className="wall-side" aria-label="Other cameras - select one to show it large">
          {side.map((c) => (
            <li key={c.camera_id} className="wall-side-item">
              <CameraTile camera={c} notMonitored={notMonitored(c.camera_id)} feedDown={feedDown} variant="thumb" index={number(c)} />
              {/* over the preview, not around it: a button's content is hidden from assistive technology */}
              <button
                type="button"
                className="wall-thumb-btn"
                onClick={() => { choose(c.camera_id); }}
                aria-label={`Show ${c.name} in the main view`}
              />
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
