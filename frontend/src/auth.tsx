import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useCallback, useMemo, type ReactNode } from "react";

import { ApiError } from "./api/client";
import { api } from "./api/endpoints";
import { AuthContext, type AuthValue } from "./authModel";
import { initialState, liveStore } from "./state/liveStore";

export function AuthProvider({ children }: { children: ReactNode }) {
  const qc = useQueryClient();
  const me = useQuery({
    queryKey: ["me"],
    queryFn: async () => {
      try {
        return await api.me();
      } catch (e) {
        if (e instanceof ApiError && e.status === 401) return null;
        throw e;
      }
    },
    retry: false,
    staleTime: 60_000,
  });
  const login = useCallback(
    async (username: string, password: string) => {
      const res = await api.login(username, password);
      qc.setQueryData(["me"], res.user);
    },
    [qc],
  );
  const logout = useCallback(async () => {
    try {
      await api.logout();
    } finally {
      // [Phase 9] Null ["me"] on its EXISTING query, then drop everything else. The old order (qc.clear()
      // first) orphaned this provider's observer: the screen stayed signed in on a shared poolside PC.
      qc.setQueryData(["me"], null);
      qc.removeQueries({ predicate: (q) => q.queryKey[0] !== "me" });
      // the next operator starts from a clean safety state and must arm audio again (OPERATOR_MANUAL.md §2)
      liveStore.set({ ...initialState });
    }
  }, [qc]);
  const value = useMemo<AuthValue>(
    () => ({ user: me.data ?? null, loading: me.isPending, login, logout }),
    [me.data, me.isPending, login, logout],
  );
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}
