import { Navigate, NavLink, Outlet } from "react-router-dom";

import { atLeast, useAuth } from "../authModel";
import { SectionHeader } from "../components/primitives/SectionHeader";
import { cn } from "../components/ui/utils";
import { SETTINGS_TABS } from "./settingsTabs";

/** /settings -> the first section the signed-in role may open. */
export function SettingsIndex() {
  const { user, loading } = useAuth();
  if (loading) return null;
  const first = SETTINGS_TABS.find((t) => atLeast(user, t.role));
  return <Navigate to={first ? first.to : "/"} replace />;
}

export default function SettingsPage() {
  const { user } = useAuth();
  const tabs = SETTINGS_TABS.filter((t) => atLeast(user, t.role));
  return (
    <div className="space-y-6">
      <SectionHeader eyebrow="Administration" title="Settings" subtitle="Zones, pool calibration, schedules, system health, users and the audit log." />
      <div className="grid gap-6 lg:grid-cols-[220px_1fr]">
        <div role="group" aria-label="Settings sections" className="glass flex h-fit flex-row flex-wrap gap-1 rounded-2xl p-2 lg:flex-col">
          {tabs.map((t) => (
            <NavLink
              key={t.to}
              to={t.to}
              className={({ isActive }) =>
                cn(
                  "rounded-xl px-3 py-2 text-sm no-underline transition-colors",
                  isActive ? "bg-primary/10 text-primary ring-1 ring-primary/20" : "text-muted-foreground hover:bg-black/[0.04] hover:text-foreground dark:hover:bg-white/[0.04]",
                )
              }
            >
              {t.label}
            </NavLink>
          ))}
        </div>
        <div className="min-w-0">
          <Outlet />
        </div>
      </div>
    </div>
  );
}
