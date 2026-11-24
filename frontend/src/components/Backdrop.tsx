import { PoolBackgroundDoodles } from "./PoolBackgroundDoodles";

/** [Phase 9b] v2's ambient layer: soft colour blobs + pool-safety doodles (doodles fade out in the dark theme). */
export function Backdrop() {
  return (
    <>
      <div
        className="pointer-events-none absolute -left-40 -top-40 h-[36rem] w-[36rem] rounded-full blur-[160px]"
        style={{ background: "color-mix(in srgb, var(--primary) 12%, transparent)" }}
        aria-hidden
      />
      <div
        className="pointer-events-none absolute -bottom-32 right-0 h-[30rem] w-[30rem] rounded-full blur-[160px]"
        style={{ background: "color-mix(in srgb, var(--chart-3) 10%, transparent)" }}
        aria-hidden
      />
      <div
        className="pointer-events-none absolute bottom-20 left-[38%] h-[22rem] w-[22rem] rounded-full blur-[140px]"
        style={{ background: "color-mix(in srgb, var(--chart-5) 7%, transparent)" }}
        aria-hidden
      />
      <PoolBackgroundDoodles />
    </>
  );
}
