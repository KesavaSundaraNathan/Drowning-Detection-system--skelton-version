import { useEffect, useState } from "react";

/** Epoch ms, re-rendered every `periodMs` (timers on screen, the 5 s WebSocket grace). */
export function useNow(periodMs = 1000): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const t = setInterval(() => {
      setNow(Date.now());
    }, periodMs);
    return () => {
      clearInterval(t);
    };
  }, [periodMs]);
  return now;
}

export function ago(iso: string | null, now: number): string {
  if (!iso) return "-";
  const s = Math.max(0, Math.round((now - Date.parse(iso)) / 1000));
  if (s < 60) return `${String(s)} s ago`;
  if (s < 3600) return `${String(Math.floor(s / 60))} min ${String(s % 60)} s ago`;
  return `${String(Math.floor(s / 3600))} h ${String(Math.floor((s % 3600) / 60))} min ago`;
}

export function errorText(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}
