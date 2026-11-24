import { cn } from "../ui/utils";

type Tone = "online" | "idle" | "offline" | "active" | "info";

const toneMap: Record<Tone, { color: string; label?: string }> = {
  online: { color: "var(--status-online)" },
  active: { color: "var(--status-online)" },
  idle: { color: "var(--status-idle)" },
  offline: { color: "var(--status-offline)" },
  info: { color: "var(--primary)" },
};

interface StatusPillProps {
  tone: Tone;
  children: React.ReactNode;
  pulse?: boolean;
  className?: string;
}

export function StatusPill({
  tone,
  children,
  pulse = true,
  className,
}: StatusPillProps) {
  const color = toneMap[tone].color;
  return (
    <span
      className={cn(
        "inline-flex items-center gap-2 rounded-full border px-3 py-1 text-sm",
        "border-white/10 bg-white/[0.04]",
        className,
      )}
    >
      <span className="relative inline-flex h-2 w-2">
        <span
          className={cn("relative inline-flex h-2 w-2 rounded-full", pulse && "status-ping")}
          style={{ color, backgroundColor: color }}
        />
      </span>
      <span className="text-foreground/90">{children}</span>
    </span>
  );
}
