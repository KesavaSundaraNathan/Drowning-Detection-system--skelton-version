import { useNow } from "../hooks";
import { liveStore, useLive } from "../state/liveStore";
import { verdict } from "../state/monitoring";

export const COLLAPSE_MS = 120_000;

/**
 * Full-screen red NOT MONITORING overlay (plan Phase 6). The operator may collapse it to a
 * persistent red bar for two minutes to reach the controls (acknowledge alarms, restart a camera);
 * if the condition still holds it comes back full screen. It can never be switched off.
 */
export function NotMonitoringOverlay() {
  const now = useNow(1000);
  const state = useLive((s) => s);
  const v = verdict(state, now);
  if (!v.notMonitoring) return null;
  const collapsed = now < state.overlayCollapsedUntil;
  if (collapsed) {
    return (
      <div className="nm-bar" role="alert" data-testid="not-monitoring-bar">
        NOT MONITORING - {v.reasons.join(" · ")}
      </div>
    );
  }
  return (
    <div className="nm-overlay" role="alertdialog" aria-modal="true" aria-label="NOT MONITORING" data-testid="not-monitoring">
      <div className="nm-box">
        <h1>NOT MONITORING</h1>
        <p>The pool is NOT being watched by Life Guard. Supervise the pool directly now.</p>
        <ul>
          {v.reasons.map((r) => (
            <li key={r}>{r}</li>
          ))}
        </ul>
        <button
          type="button"
          onClick={() => {
            liveStore.set({ overlayCollapsedUntil: Date.now() + COLLAPSE_MS });
          }}
        >
          I am supervising the pool - show the controls (2 min)
        </button>
      </div>
    </div>
  );
}
