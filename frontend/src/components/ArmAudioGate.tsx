import { useState, type ReactNode } from "react";

import { tones, type ToneEngine } from "../audio/tones";
import { liveStore, useLive } from "../state/liveStore";
import { CenteredScreen } from "./CenteredScreen";

/**
 * Browsers block sound until a user gesture. Monitoring is therefore shown only after the operator
 * clicks "Arm audio" - an alarm can never be silent because the page was never clicked.
 */
export function ArmAudioGate({ children, engine = tones }: { children: ReactNode; engine?: ToneEngine }) {
  const armed = useLive((s) => s.audioArmed);
  const [error, setError] = useState<string | null>(null);
  if (armed) return <>{children}</>;
  return (
    <CenteredScreen subtitle="Start of shift">
    <div className="gate" data-testid="arm-audio-gate">
      <h2>Arm alarm audio</h2>
      <p>Life Guard plays an alarm tone for drowning alerts and a fault tone when monitoring stops.</p>
      <button
        type="button"
        className="btn-primary btn-big"
        onClick={() => {
          void engine.arm().then((ok) => {
            if (ok) liveStore.set({ audioArmed: true });
            else setError("This browser did not allow audio. Check the speaker / site sound settings and try again.");
          });
        }}
      >
        Arm audio and start monitoring
      </button>
      {error && (
        <p role="alert" className="error">
          {error}
        </p>
      )}
    </div>
    </CenteredScreen>
  );
}
