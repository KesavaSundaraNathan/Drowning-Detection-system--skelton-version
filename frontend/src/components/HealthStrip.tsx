import { useState } from "react";
import { Link } from "react-router-dom";
import { atLeast, useAuth } from "../authModel";
import { LEVEL_TEXT } from "../state/health";
import { useSystemHealth } from "../state/useSystemHealth";

const DISMISS_KEY = "lifeguard.healthStrip.dismissed";

function readDismissed(): string | null {
  try {
    return sessionStorage.getItem(DISMISS_KEY);
  } catch {
    return null;
  }
}

/**
 * Dashboard summary of system health (every role); the detail is on the System health page.
 * [v4] Closable, but never invisible: closed, it shrinks to a one-line chip that still names the level and the
 * number of problems, and it opens fully again by itself as soon as the SET of problems changes (a new problem
 * appears, one clears, or one gets worse). Remembered for this browser tab only.
 */
export function HealthStrip() {
  const { user } = useAuth();
  const h = useSystemHealth();
  const shown = h.problems.slice(0, 4);
  // identity of what is wrong (not the changing numbers in the text): a new or worse problem re-opens the strip
  const signature = h.problems.map((p) => `${p.key}:${p.level}`).join("|");
  const [dismissed, setDismissed] = useState<string | null>(readDismissed);
  const collapsed = h.problems.length > 0 && dismissed === signature;

  const close = () => {
    setDismissed(signature);
    try {
      sessionStorage.setItem(DISMISS_KEY, signature);
    } catch {
      /* private mode: closed for this page view only */
    }
  };
  const open = () => {
    setDismissed(null);
    try {
      sessionStorage.removeItem(DISMISS_KEY);
    } catch {
      /* ignore */
    }
  };

  if (collapsed) {
    return (
      <section className={`health-strip health-strip-collapsed health-${h.level}`} role="status" aria-label="System health">
        <strong>{LEVEL_TEXT[h.level]}</strong>
        <span>
          {String(h.problems.length)} problem{h.problems.length === 1 ? "" : "s"} hidden
        </span>
        <button type="button" className="health-strip-toggle" onClick={open}>
          Show
        </button>
      </section>
    );
  }
  return (
    <section className={`health-strip health-${h.level}`} role="status" aria-label="System health">
      <strong>{LEVEL_TEXT[h.level]}</strong>
      {shown.length > 0 && (
        <ul>
          {shown.map((p) => (
            <li key={p.key}>
              {p.label}: {p.detail}
            </li>
          ))}
          {h.problems.length > shown.length && <li>and {String(h.problems.length - shown.length)} more</li>}
        </ul>
      )}
      {atLeast(user, "supervisor") && <Link to="/settings/system">Details</Link>}
      {h.problems.length > 0 && (
        <button type="button" className="health-strip-close" aria-label="Close system health message" onClick={close}>
          ×
        </button>
      )}
    </section>
  );
}
