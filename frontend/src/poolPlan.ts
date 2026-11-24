/** [Phase 11 WP2] Pool-plan geometry shared by the pool-zone editor and its tests. */
export type PlanPoint = [number, number];

/** Shoelace area of a polygon in square metres. */
export function polygonAreaM2(pts: readonly PlanPoint[]): number {
  let a = 0;
  pts.forEach(([x0, y0], i) => {
    const [x1, y1] = pts[(i + 1) % pts.length] ?? [x0, y0];
    a += x0 * y1 - x1 * y0;
  });
  return Math.abs(a) / 2;
}
