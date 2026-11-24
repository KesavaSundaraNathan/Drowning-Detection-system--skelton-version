import { useEffect, useRef, useState } from "react";

import { media } from "../api/endpoints";
import { latestSnapshot, useSnapshot } from "./liveFramesStore";
import { readMjpeg } from "./mjpegReader";
import { joints, skeletonPath, useSkeleton } from "./skeletonOverlay";
import { boxesFor, parseBoxes, useViewMode, type FrameBoxes, type ViewMode } from "./viewMode";

/**
 * [v4] Live pictures for the Cameras page video wall, built so that WATCHING never costs the analysis CPU:
 *
 *  - side previews poll short snapshots (frame.jpg?quality=thumb), one request at a time per camera - no
 *    long-lived connection, nothing piles up when the page is busy;
 *  - the main view holds the ONE live stream (mjpeg?quality=high) and cancels it explicitly when the camera
 *    changes or the view closes (measured 2026-10-15: streams left running dropped every camera's analysis to
 *    6-13 fps);
 *  - every preview keeps its latest snapshot here, so a camera switched into the main view shows a picture at
 *    once while its live stream starts.
 *
 * [view modes] Frames come without boxes (overlay=client) together with that frame's boxes, and the browser draws
 * the boxes the camera's view mode asks for (Full / Clean / Alerts only, components/viewMode.ts). If the stream
 * cannot be read that way, the view falls back to the server-drawn stream (Full) or the plain one.
 */

const RED = "#e62828";
const GREEN = "#3cc83c";
const GREY = "#a0a0a0";
const COAST = "#c8c8c8";
const LIMB = "#00c8ff"; // [rtmpose clone] skeleton: same colours as the server-drawn view (live/skeleton.py)
const JOINT = "#ffdc00";

/** The boxes of one frame over the picture (same box as the <img>: object-fit contain = xMidYMid meet). */
export function BoxOverlay({ boxes, mode, skeleton = true }: { boxes: FrameBoxes | null; mode: ViewMode; skeleton?: boolean }) {
  const b = boxesFor(mode, boxes);
  const k = skeleton ? (b?.k ?? []) : [];
  if (!b || (b.t.length === 0 && b.c.length === 0 && k.length === 0)) return null;
  const sw = Math.max(1, b.w / 480);
  const font = 11 * sw;
  return (
    <svg
      className="box-overlay"
      viewBox={`0 0 ${String(b.w)} ${String(b.h)}`}
      preserveAspectRatio="xMidYMid meet"
      aria-hidden="true"
      data-testid="box-overlay"
    >
      {k.map((row, i) => (
        <g key={`k${String(i)}`} data-skeleton={row[0]}>
          <path d={skeletonPath(row)} fill="none" stroke={LIMB} strokeWidth={sw} strokeLinecap="round" />
          {joints(row).map(([x, y], j) => (
            <circle key={j} cx={x} cy={y} r={sw * 2} fill={JOINT} />
          ))}
        </g>
      ))}
      {b.t.map(([id, x1, y1, x2, y2, tent, hot], i) => {
        const colour = hot ? RED : tent ? GREY : GREEN;
        return (
          <g key={`t${String(i)}`} data-box={hot ? "alert" : tent ? "tentative" : "track"}>
            <rect x={x1} y={y1} width={x2 - x1} height={y2 - y1} fill="none" stroke={colour} strokeWidth={hot ? sw * 2 : sw} />
            {id >= 0 && (
              <text x={x1} y={Math.max(font, y1 - 4)} fill={colour} fontSize={font}>
                {hot ? `#${String(id)} ALERT` : `#${String(id)}`}
              </text>
            )}
          </g>
        );
      })}
      {b.c.map(([id, x1, y1, x2, y2, followed], i) => (
        <g key={`c${String(i)}`} data-box="coasting">
          <rect x={x1} y={y1} width={x2 - x1} height={y2 - y1} fill="none" stroke={COAST} strokeWidth={sw} strokeDasharray={`${String(6 * sw)} ${String(6 * sw)}`} />
          <text x={x1} y={Math.max(font, y1 - 4)} fill={COAST} fontSize={font}>
            {followed ? `#${String(id)} under` : `#${String(id)} ?`}
          </text>
        </g>
      ))}
    </svg>
  );
}

/** A preview: the camera's newest snapshot and its boxes. [rtmpose clone] `quality`/`everyMs`: the Dashboard tiles
 * use normal-size snapshots about 3x a second (short requests, no long-lived connection). */
export function SnapshotImage({
  cameraId,
  name,
  quality = "thumb",
  everyMs = 500,
}: {
  cameraId: string;
  name: string;
  quality?: "thumb" | "normal";
  everyMs?: number;
}) {
  const [mode] = useViewMode(cameraId);
  const [skeleton] = useSkeleton(cameraId);
  const snap = useSnapshot(cameraId, everyMs, quality);
  if (!snap) return <div className="wall-waiting">Connecting…</div>;
  return (
    <>
      <img src={snap.url} alt={`Live view of ${name}`} />
      <BoxOverlay boxes={snap.boxes} mode={mode} skeleton={skeleton} />
    </>
  );
}

const canStream = typeof URL.createObjectURL === "function" && typeof ReadableStream === "function";

/** A live stream (the wall's main view, or a dashboard / detail tile): the latest snapshot at once, then live. */
export function MainStream({ cameraId, name, quality = "high" }: { cameraId: string; name: string; quality?: "normal" | "high" }) {
  const [mode] = useViewMode(cameraId);
  const [skeleton] = useSkeleton(cameraId);
  const [placeholder] = useState(() => latestSnapshot(cameraId));
  const url = media.mjpeg(cameraId, quality, "client");
  // keyed by the stream URL, so a switched camera never shows the previous camera's frame or fallback
  const [latest, setFrame] = useState<{ stream: string; url: string; boxes: FrameBoxes | null } | null>(null);
  const [failed, setFailed] = useState<string | null>(null);
  const frame = latest?.stream === url ? latest : null;
  const fallback = !canStream || failed === url;
  const ref = useRef<HTMLImageElement>(null);

  useEffect(() => {
    if (!canStream) return;
    const ctl = new AbortController();
    let current: string | null = null;
    readMjpeg(url, ctl.signal, (p) => {
      const next = URL.createObjectURL(new Blob([p.jpeg as BlobPart], { type: "image/jpeg" }));
      const old = current;
      current = next;
      setFrame({ stream: url, url: next, boxes: parseBoxes(p.headers["x-lifeguard-boxes"] ?? null) });
      if (old) setTimeout(() => { URL.revokeObjectURL(old); }, 1000);
    }).catch(() => {
      if (!ctl.signal.aborted) setFailed(url);
    });
    return () => {
      ctl.abort(); // ends the request: the server stops encoding for this viewer
      if (current) {
        const last = current;
        setTimeout(() => { URL.revokeObjectURL(last); }, 1000);
      }
    };
  }, [url]);

  // fallback (no fetch streaming): an <img> on the server-drawn stream (Full) or the plain one (Clean / Alerts only)
  useEffect(() => {
    const img = ref.current;
    if (!img || !fallback) return;
    img.src = mode === "full" ? media.mjpeg(cameraId, quality, "burned", skeleton) : url;
    return () => {
      img.removeAttribute("src"); // aborts the multipart request
    };
  }, [fallback, mode, skeleton, cameraId, quality, url]);

  return (
    <>
      {placeholder && !frame && <img className="wall-placeholder" src={placeholder.url} alt="" aria-hidden="true" />}
      {fallback ? (
        <img ref={ref} className="wall-live" alt={`Live view of ${name}`} data-stream={url} />
      ) : (
        <img className="wall-live" src={frame?.url} alt={`Live view of ${name}`} data-stream={url} />
      )}
      {!fallback && <BoxOverlay boxes={frame?.boxes ?? null} mode={mode} skeleton={skeleton} />}
    </>
  );
}
