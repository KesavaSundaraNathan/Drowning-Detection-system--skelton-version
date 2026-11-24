import { useQuery } from "@tanstack/react-query";

import { api } from "./api/endpoints";

export const OPEN_EVENTS_KEY = ["events", "open"] as const;

/** [Phase 5] Open events: not closed and not resolved, newest first (backend alerts/events.py). */
export function useOpenEvents() {
  return useQuery({
    queryKey: OPEN_EVENTS_KEY,
    queryFn: () => api.events({ status: "open", limit: 100 }),
    refetchInterval: 15_000, // belt and braces: the WebSocket invalidates this on every event message
  });
}
