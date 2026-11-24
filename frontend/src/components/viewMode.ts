import { useCallback, useSyncExternalStore } from "react";

/**
 * Camera view modes (display only, per viewer, remembered per camera in this browser):
 *  - "full":   every box, as before (green tracked, grey tentative, dashed coasting, red + "ALERT" when alerting);
 *  - "clean":  no boxes on the video;
 *  - "alerts": no boxes except the red box of a person whose alert is raised right now.
 * No mode changes detection, thresholds or alerts. The alarm itself never depends on the mode: the tile's red ring,
 * the visual alarm bar and the alarm sound come from the alerts query (alarmModel), never from the boxes.
 */
export type ViewMode = "full" | "clean" | "alerts";

export const VIEW_MODES: readonly { mode: ViewMode; label: string; title: string }[] = [
  { mode: "full", label: "Full", title: "Show every detection box" },
  { mode: "clean", label: "Clean", title: "Video only, no boxes (alarms still show and sound)" },
  { mode: "alerts", label: "Alerts only", title: "No boxes except people in an active alert" },
];

const KEY = (cameraId: string) => `lifeguard.viewMode.${cameraId}`;
const EVENT = "lifeguard-viewmode";

export function readViewMode(cameraId: string): ViewMode {
  try {
    const v = localStorage.getItem(KEY(cameraId));
    return v === "clean" || v === "alerts" || v === "full" ? v : "full";
  } catch {
    return "full";
  }
}

function subscribe(cb: () => void): () => void {
  window.addEventListener(EVENT, cb);
  window.addEventListener("storage", cb); // another tab of this browser changed it
  return () => {
    window.removeEventListener(EVENT, cb);
    window.removeEventListener("storage", cb);
  };
}

/** The camera's view mode in this browser; every view of the camera (main view, preview, tile) follows a change. */
export function useViewMode(cameraId: string): [ViewMode, (m: ViewMode) => void] {
  const mode = useSyncExternalStore(subscribe, () => readViewMode(cameraId), (): ViewMode => "full");
  const set = useCallback(
    (m: ViewMode) => {
      try {
        localStorage.setItem(KEY(cameraId), m);
      } catch {
        /* not remembered */
      }
      window.dispatchEvent(new CustomEvent(EVENT, { detail: cameraId }));
    },
    [cameraId],
  );
  return [mode, set];
}

/** One frame's boxes (backend live/view.boxes_json): t = [id, x1, y1, x2, y2, tentative, alerting], c = coasting.
 * [rtmpose clone] k = skeletons, [id, x0, y0, c0, ..., x16, y16, c16] (c 0..100), components/skeletonOverlay.ts. */
export interface FrameBoxes {
  w: number;
  h: number;
  t: [number, number, number, number, number, number, number][];
  c: [number, number, number, number, number, number][];
  k?: number[][];
}

export function parseBoxes(raw: string | null): FrameBoxes | null {
  if (!raw) return null;
  try {
    const b = JSON.parse(raw) as FrameBoxes;
    return typeof b.w === "number" && typeof b.h === "number" && Array.isArray(b.t) ? { ...b, c: Array.isArray(b.c) ? b.c : [], k: Array.isArray(b.k) ? b.k : [] } : null;
  } catch {
    return null;
  }
}

/** The boxes a mode draws. */
export function boxesFor(mode: ViewMode, boxes: FrameBoxes | null): FrameBoxes | null {
  if (!boxes || mode === "full") return boxes;
  if (mode === "clean") return { ...boxes, t: [], c: [], k: [] };
  const hot = boxes.t.filter((t) => t[6] === 1);
  const ids = new Set(hot.map((t) => t[0]));
  return { ...boxes, t: hot, c: [], k: (boxes.k ?? []).filter((k) => ids.has(k[0] ?? -1)) };
}
