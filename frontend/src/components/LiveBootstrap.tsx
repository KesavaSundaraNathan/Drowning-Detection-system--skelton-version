import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect } from "react";

import { api } from "../api/endpoints";
import { liveStore } from "../state/liveStore";
import { summaryFromStatus } from "../state/monitoring";
import { LiveFeed } from "../ws/feed";

/**
 * Starts the real-time feed for the logged-in session and keeps the store in sync with
 * GET /live/status as a fallback (also covers a disabled live runtime, which sends no summaries).
 */
export function LiveBootstrap() {
  const qc = useQueryClient();
  const status = useQuery({ queryKey: ["live", "status"], queryFn: api.liveStatus, refetchInterval: 10_000 });

  useEffect(() => {
    const feed = new LiveFeed(qc, undefined, undefined, () => {
      qc.setQueryData(["me"], null); // session ended: back to the login screen
    });
    feed.start();
    return () => {
      feed.stop();
    };
  }, [qc]);

  useEffect(() => {
    const st = status.data;
    if (!st) return;
    liveStore.set({
      liveEnabled: st.enabled,
      cameras: summaryFromStatus(st),
      notStarted: Object.keys(st.not_started ?? {}),
    });
  }, [status.data]);

  return null;
}
