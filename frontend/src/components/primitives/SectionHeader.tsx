import type { ReactNode } from "react";
import { cn } from "../ui/utils";

interface SectionHeaderProps {
  title: string;
  subtitle?: string;
  eyebrow?: string;
  actions?: ReactNode;
  className?: string;
}

export function SectionHeader({
  title,
  subtitle,
  eyebrow,
  actions,
  className,
}: SectionHeaderProps) {
  return (
    <div className={cn("flex items-end justify-between gap-4", className)}>
      <div>
        {eyebrow && (
          <p className="font-mono text-xs uppercase tracking-[0.18em] text-primary/80">
            {eyebrow}
          </p>
        )}
        <h1 className="mb-0 mt-1 text-2xl">{title}</h1>
        {subtitle && (
          <p className="mt-1 text-sm text-muted-foreground">{subtitle}</p>
        )}
      </div>
      {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
    </div>
  );
}
