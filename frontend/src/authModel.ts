import { createContext, useContext } from "react";

import type { Role, User } from "./api/types";

const RANK: Record<Role, number> = { lifeguard: 1, supervisor: 2, admin: 3 };

export function atLeast(user: User | null | undefined, role: Role): boolean {
  return !!user && RANK[user.role] >= RANK[role];
}

export interface AuthValue {
  user: User | null;
  loading: boolean;
  login: (username: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
}

export const AuthContext = createContext<AuthValue | null>(null);

export function useAuth(): AuthValue {
  const v = useContext(AuthContext);
  if (v === null) throw new Error("useAuth outside AuthProvider");
  return v;
}
