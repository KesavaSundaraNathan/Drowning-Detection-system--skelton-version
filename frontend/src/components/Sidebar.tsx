import {
  Bell,
  ChevronLeft,
  ChevronRight,
  LayoutDashboard,
  LogOut,
  Settings,
  UserRound,
  Video,
  type LucideIcon,
} from "lucide-react";
import { motion } from "motion/react";
import { useState } from "react";
import { NavLink } from "react-router-dom";
import { useActiveAlarms } from "../alarmModel";
import type { Role } from "../api/types";
import { atLeast, useAuth } from "../authModel";
import { useLive } from "../state/liveStore";
import { setTheme, useTheme } from "../state/theme";
import { BrandLogo } from "./BrandLogo";
import { StatusPill } from "./primitives/StatusPill";
import { Switch } from "./ui/switch";
import { cn } from "./ui/utils";

/** [Phase 9b] v2's collapsible glass sidebar, carrying v3's routes, roles and live-connection state. */
const NAV: { to: string; label: string; role: Role; icon: LucideIcon; color: string }[] = [
  { to: "/", label: "Dashboard", role: "lifeguard", icon: LayoutDashboard, color: "#007AFF" },
  { to: "/cameras", label: "Cameras", role: "lifeguard", icon: Video, color: "#5AC8FA" },
  { to: "/alerts", label: "Alerts", role: "lifeguard", icon: Bell, color: "#FF3B30" },
  { to: "/settings", label: "Settings", role: "supervisor", icon: Settings, color: "#8E8E93" },
];

const WS_TEXT = { open: "live", connecting: "connecting", down: "down" } as const;

/** Animated label; when the sidebar is collapsed the text stays as screen-reader-only (it names the link). */
function Fade({ show, children, className }: { show: boolean; children: React.ReactNode; className?: string }) {
  if (!show) return <span className="sr-only">{children}</span>;
  return (
    <motion.span
      initial={{ opacity: 0, x: -6 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.16 }}
      className={className}
    >
      {children}
    </motion.span>
  );
}

export function Sidebar() {
  const { user, logout } = useAuth();
  const ws = useLive((s) => s.ws);
  const theme = useTheme();
  const alarms = useActiveAlarms().data?.length ?? 0;
  const [collapsed, setCollapsed] = useState(false);
  const open = !collapsed;
  const who = user ? `${user.username} (${user.role})` : "";
  return (
    <motion.aside
      animate={{ width: collapsed ? 76 : 264 }}
      transition={{ type: "spring", stiffness: 340, damping: 34 }}
      className="glass-strong relative z-10 flex shrink-0 flex-col rounded-3xl p-3"
      style={{ minWidth: collapsed ? 76 : 264 }}
    >
      <div className={cn("flex items-center gap-3 px-1.5 py-2", collapsed && "justify-center")}>
        <BrandLogo size={40} />
        <Fade show={open} className="min-w-0 leading-tight">
          <span className="block text-sm font-semibold text-foreground">Life Guard</span>
          <span className="block font-mono text-[10px] uppercase tracking-[0.16em] text-muted-foreground">
            Pool Safety System
          </span>
        </Fade>
      </div>

      <div className={cn("my-2 px-1.5", collapsed && "flex justify-center")} title="real-time connection">
        {open ? (
          <StatusPill tone={ws === "open" ? "online" : ws === "connecting" ? "idle" : "offline"} className="border-border bg-card">
            <span className={`ws ws-${ws}`}>{WS_TEXT[ws]}</span>
          </StatusPill>
        ) : (
          <span className={`ws ws-${ws}`}>{WS_TEXT[ws]}</span>
        )}
      </div>

      <nav className="mt-1 flex flex-1 flex-col gap-0.5 overflow-y-auto scroll-area-soft">
        {NAV.filter((n) => atLeast(user, n.role)).map((n) => {
          const Icon = n.icon;
          return (
            <NavLink
              key={n.to}
              to={n.to}
              end={n.to === "/"}
              title={collapsed ? n.label : undefined}
              className={({ isActive }) =>
                cn(
                  "relative flex items-center gap-3 rounded-xl px-2.5 py-2.5 text-sm no-underline transition-colors",
                  collapsed && "justify-center",
                  isActive ? "text-foreground" : "text-muted-foreground hover:bg-black/[0.04] hover:text-foreground dark:hover:bg-white/[0.04]",
                )
              }
            >
              {({ isActive }) => (
                <>
                  {isActive && (
                    <motion.span
                      layoutId="nav-active"
                      className="absolute inset-0 rounded-xl bg-primary/10 ring-1 ring-primary/20"
                      transition={{ type: "spring", stiffness: 500, damping: 38 }}
                    />
                  )}
                  <span className="relative z-10 flex h-5 w-5 shrink-0 items-center justify-center" style={{ color: isActive ? n.color : undefined }}>
                    <Icon className="h-[18px] w-[18px]" aria-hidden />
                  </span>
                  <Fade show={open} className="relative z-10 flex-1 whitespace-nowrap text-left">
                    {n.label}
                  </Fade>
                  {n.to === "/alerts" && alarms > 0 && (
                    <span
                      aria-hidden
                      className={cn(
                        "z-10 flex h-5 min-w-5 items-center justify-center rounded-full bg-destructive px-1.5 font-mono text-[11px] text-destructive-foreground",
                        collapsed ? "absolute right-1 top-1" : "relative",
                      )}
                    >
                      {alarms}
                    </span>
                  )}
                </>
              )}
            </NavLink>
          );
        })}
      </nav>

      <div className="mt-2 flex flex-col gap-1 border-t border-border pt-2">
        <NavLink
          to="/account"
          title="My account: change password"
          className={({ isActive }) =>
            cn(
              "flex items-center gap-3 rounded-xl px-2.5 py-2 text-sm no-underline transition-colors",
              collapsed && "justify-center",
              isActive ? "bg-primary/10 text-foreground" : "text-muted-foreground hover:bg-black/[0.04] hover:text-foreground dark:hover:bg-white/[0.04]",
            )
          }
        >
          <UserRound className="h-[18px] w-[18px] shrink-0" aria-hidden />
          <Fade show={open} className="min-w-0 truncate">
            {who}
          </Fade>
        </NavLink>
        <label className={cn("m-0 flex cursor-pointer items-center gap-3 px-2.5 py-1.5 text-sm font-normal text-muted-foreground", collapsed && "justify-center px-0")}>
          <Switch checked={theme === "dark"} onCheckedChange={(on) => { setTheme(on ? "dark" : "light"); }} aria-label="Dark theme" />
          <Fade show={open}>Dark theme</Fade>
        </label>
        <button
          type="button"
          onClick={() => void logout()}
          className={cn("flex items-center gap-3 rounded-xl border-0 bg-transparent px-2.5 py-2 text-sm text-muted-foreground shadow-none hover:bg-destructive/10 hover:text-destructive", collapsed && "justify-center")}
        >
          <LogOut className="h-[18px] w-[18px] shrink-0" aria-hidden />
          <Fade show={open}>Log out</Fade>
        </button>
        <button
          type="button"
          onClick={() => { setCollapsed((c) => !c); }}
          aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
          className={cn("flex items-center gap-3 rounded-xl px-2.5 py-2 text-sm text-primary", collapsed && "justify-center")}
        >
          {collapsed ? <ChevronRight className="h-[18px] w-[18px]" aria-hidden /> : <ChevronLeft className="h-[18px] w-[18px]" aria-hidden />}
          <Fade show={open}>Collapse</Fade>
        </button>
      </div>
    </motion.aside>
  );
}
