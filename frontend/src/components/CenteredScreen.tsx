import type { ReactNode } from "react";
import { Backdrop } from "./Backdrop";
import { BrandLogo } from "./BrandLogo";

/** [Phase 9b] v2's full-screen glass stage for the screens outside the shell (log in, arm audio). */
export function CenteredScreen({ children, subtitle }: { children: ReactNode; subtitle: string }) {
  return (
    <div className="relative flex min-h-screen flex-col items-center justify-center gap-5 overflow-hidden bg-background p-6 text-foreground ambient-grid">
      <Backdrop />
      <div className="relative z-10 flex items-center gap-3">
        <BrandLogo size={72} />
        <span className="leading-tight">
          <span className="block text-lg font-semibold">Life Guard</span>
          <span className="block font-mono text-[11px] uppercase tracking-[0.16em] text-muted-foreground">{subtitle}</span>
        </span>
      </div>
      {children}
    </div>
  );
}
