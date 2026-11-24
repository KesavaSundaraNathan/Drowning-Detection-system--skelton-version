import { useLive } from "../state/liveStore";
import { mutedZones } from "../state/monitoring";

/** Persistent while any zone's schedule window is closed. Submersion / sudden blackout alerts and faults still sound. */
export function MutedBanner() {
  const muted = useLive(mutedZones);
  if (muted.length === 0) return null;
  return (
    <div className="banner banner-muted" role="status" data-testid="muted-banner">
      <strong>MUTED</strong>{" "}
      {muted.map((m) => `${m.zone} (${m.camera})`).join(", ")} - schedule window closed: ordinary alerts are
      silent here. Submersion / sudden blackout alerts and faults always sound.
    </div>
  );
}
