import { useEffect, useState } from "react";

import { media } from "../api/endpoints";
import { parseBoxes, type FrameBoxes } from "./viewMode";

/** [v4] Snapshot polling for the video wall previews + the newest snapshot of every camera (see liveFrames.tsx).
 * [view modes] Snapshots come without boxes (overlay=client) with their boxes alongside; the view draws them. */

export interface Snapshot {
  url: string; // object URL of the JPEG
  boxes: FrameBoxes | null;
}

const latest = new Map<string, Snapshot>(); // camera id -> its newest snapshot

export function latestSnapshot(cameraId: string): Snapshot | null {
  return latest.get(cameraId) ?? null;
}

const canBlob = typeof URL.createObjectURL === "function";

/** [rtmpose clone] `quality` "normal" (mjpeg_width) for the Dashboard tiles, which poll snapshots instead of each
 * holding an MJPEG stream: over HTTP/1.1 (the dev server) a browser allows 6 connections per site, and 4+ tile
 * streams starved every other request (status, alerts, new streams) - frozen UI and black tiles. */
export function useSnapshot(cameraId: string, everyMs = 500, quality: "thumb" | "normal" = "thumb"): Snapshot | null {
  const [snap, setSnap] = useState<Snapshot | null>(() => latest.get(cameraId) ?? null);
  useEffect(() => {
    const ctl = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;
    const tick = async () => {
      try {
        const r = await fetch(media.frame(cameraId, quality, "client"), { signal: ctl.signal, credentials: "same-origin", cache: "no-store" });
        if (r.ok && canBlob) {
          const next = { url: URL.createObjectURL(await r.blob()), boxes: parseBoxes(r.headers.get("x-lifeguard-boxes")) };
          const old = latest.get(cameraId);
          latest.set(cameraId, next);
          setSnap(next);
          if (old) URL.revokeObjectURL(old.url);
        }
      } catch {
        if (ctl.signal.aborted) return;
      }
      if (!ctl.signal.aborted) timer = setTimeout(() => void tick(), everyMs);
    };
    void tick();
    return () => {
      ctl.abort();
      clearTimeout(timer);
    };
  }, [cameraId, everyMs, quality]);
  return snap;
}
