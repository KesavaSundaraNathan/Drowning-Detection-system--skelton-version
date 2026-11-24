import { useQuery } from "@tanstack/react-query";
import { api } from "../api/endpoints";
import { atLeast, useAuth } from "../authModel";
import { useNow } from "../hooks";
import { systemHealth, type HealthReport } from "./health";
import { useLive } from "./liveStore";
import { WS_GRACE_MS } from "./monitoring";

/** Polls every status source this user's role may read and maps it with `systemHealth` (5 s cadence). */
export function useSystemHealth(): HealthReport {
  const { user } = useAuth();
  const supervisor = atLeast(user, "supervisor");
  const now = useNow(1000);
  const ws = useLive((s) => s.ws);
  const wsDownSince = useLive((s) => s.wsDownSince);
  const live = useQuery({ queryKey: ["live", "status"], queryFn: api.liveStatus, refetchInterval: 5000 });
  const delivery = useQuery({ queryKey: ["delivery"], queryFn: api.delivery, refetchInterval: 5000, enabled: supervisor });
  const faults = useQuery({
    queryKey: ["alerts", "faults"],
    queryFn: async () => (await api.alerts({ limit: 200 })).filter((a) => a.scenario === "FAULT" && a.cleared_at === null),
    refetchInterval: 10_000,
  });
  const cameras = useQuery({ queryKey: ["cameras"], queryFn: api.cameras, staleTime: 60_000 });
  return systemHealth({
    wsDown: ws === "down" && wsDownSince !== null && now - wsDownSince > WS_GRACE_MS,
    live: { data: live.data, error: live.isError },
    delivery: supervisor ? { data: delivery.data, error: delivery.isError } : undefined,
    openFaults: { data: faults.data?.length, error: faults.isError },
    cameraNames: Object.fromEntries((cameras.data ?? []).map((c) => [c.id, c.name])),
  });
}
