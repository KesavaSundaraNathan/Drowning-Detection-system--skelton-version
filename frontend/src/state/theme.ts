/**
 * [Phase 9b] v2's light / dark theme switch. Light (v2 "Apple glass") is the default; dark is v2's indigo
 * "command center". The choice is remembered per browser; storage that is blocked or broken never stops the UI.
 * The class goes on <html> so Radix portals (dialogs) inherit it, exactly as v2 did.
 */
import { useSyncExternalStore } from "react";

export type Theme = "light" | "dark";
const KEY = "lifeguard.theme";
const listeners = new Set<() => void>();

function read(): Theme {
  try {
    return localStorage.getItem(KEY) === "dark" ? "dark" : "light";
  } catch {
    return "light";
  }
}

let current: Theme = typeof window === "undefined" ? "light" : read();

export function applyTheme(t: Theme = current): void {
  document.documentElement.classList.toggle("dark", t === "dark");
}

export function setTheme(t: Theme): void {
  current = t;
  try {
    localStorage.setItem(KEY, t);
  } catch {
    // private window / blocked storage: the switch still works for this page
  }
  applyTheme(t);
  for (const l of listeners) l();
}

export function useTheme(): Theme {
  return useSyncExternalStore(
    (cb) => {
      listeners.add(cb);
      return () => listeners.delete(cb);
    },
    () => current,
    () => "light",
  );
}
