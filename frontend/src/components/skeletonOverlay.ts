import { useCallback, useSyncExternalStore } from "react";

/**
 * [rtmpose clone] Skeleton toggle (display only, per viewer, remembered per camera in this browser). The keypoints
 * come with each frame's boxes (backend live/skeleton.to_json, "k" in X-Lifeguard-Boxes): no extra request, no
 * inference. Hiding the skeleton changes nothing in detection, thresholds or alerts.
 */

/** COCO-17 limbs (same order as backend live/skeleton.LIMBS). */
export const LIMBS: readonly (readonly [number, number])[] = [
  [0, 1], [0, 2], [1, 3], [2, 4],
  [5, 7], [7, 9], [6, 8], [8, 10],
  [5, 6], [5, 11], [6, 12], [11, 12],
  [11, 13], [13, 15], [12, 14], [14, 16],
];
export const MIN_CONF = 30; // 0..100, same display cut-off as the backend

const KEY = (cameraId: string) => `lifeguard.skeleton.${cameraId}`;
const EVENT = "lifeguard-skeleton";

export function readSkeleton(cameraId: string): boolean {
  try {
    return localStorage.getItem(KEY(cameraId)) !== "off";
  } catch {
    return true;
  }
}

function subscribe(cb: () => void): () => void {
  window.addEventListener(EVENT, cb);
  window.addEventListener("storage", cb);
  return () => {
    window.removeEventListener(EVENT, cb);
    window.removeEventListener("storage", cb);
  };
}

export function useSkeleton(cameraId: string): [boolean, (on: boolean) => void] {
  const on = useSyncExternalStore(subscribe, () => readSkeleton(cameraId), () => true);
  const set = useCallback(
    (v: boolean) => {
      try {
        localStorage.setItem(KEY(cameraId), v ? "on" : "off");
      } catch {
        /* not remembered */
      }
      window.dispatchEvent(new CustomEvent(EVENT, { detail: cameraId }));
    },
    [cameraId],
  );
  return [on, set];
}

/** One SVG path string for every visible limb of one person: a single <path> per person keeps the DOM small. */
export function skeletonPath(k: readonly number[]): string {
  const parts: string[] = [];
  for (const [a, b] of LIMBS) {
    const ca = k[1 + a * 3 + 2] ?? 0;
    const cb = k[1 + b * 3 + 2] ?? 0;
    if (ca >= MIN_CONF && cb >= MIN_CONF) {
      parts.push(`M${String(k[1 + a * 3])} ${String(k[2 + a * 3])}L${String(k[1 + b * 3])} ${String(k[2 + b * 3])}`);
    }
  }
  return parts.join("");
}

/** [x, y] of each visible joint. */
export function joints(k: readonly number[]): [number, number][] {
  const out: [number, number][] = [];
  for (let j = 0; j < 17; j++) {
    if ((k[3 + j * 3] ?? 0) >= MIN_CONF) out.push([k[1 + j * 3] ?? 0, k[2 + j * 3] ?? 0]);
  }
  return out;
}
