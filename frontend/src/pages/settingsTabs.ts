import type { Role } from "../api/types";

/** The sections of the Settings page (the old side-nav pages), each shown only to the roles that may use it. */
export const SETTINGS_TABS: { to: string; label: string; role: Role }[] = [
  { to: "zones", label: "Zones", role: "supervisor" },
  { to: "calibration", label: "Pool calibration", role: "supervisor" },
  { to: "schedules", label: "Schedules", role: "supervisor" },
  { to: "system", label: "System health", role: "supervisor" },
  { to: "users", label: "Users", role: "admin" },
  { to: "audit", label: "Audit log", role: "admin" },
];

