import { useEffect } from "react";

import { chooseTone, useActiveAlarms } from "../alarmModel";
import { tones, type ToneEngine } from "../audio/tones";
import { useNow } from "../hooks";
import { useLive } from "../state/liveStore";
import { verdict } from "../state/monitoring";

/** Plays the tone chooseTone() picks (renders nothing). */
export function SafetyAudio({ engine = tones }: { engine?: ToneEngine }) {
  const now = useNow(1000);
  const state = useLive((s) => s);
  const active = useActiveAlarms();
  const list = active.data ?? [];
  const v = verdict(state, now);
  const tone = chooseTone({
    armed: state.audioArmed,
    alarms: list.filter((a) => a.scenario !== "FAULT").length,
    faults: list.filter((a) => a.scenario === "FAULT").length,
    notMonitoringOverlay: v.notMonitoring && now >= state.overlayCollapsedUntil,
  });
  useEffect(() => {
    if (tone === null) engine.stop();
    else engine.play(tone);
  }, [tone, engine]);
  useEffect(() => () => { engine.stop(); }, [engine]);
  return null;
}
