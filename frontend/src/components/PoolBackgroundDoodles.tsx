/**
 * Subtle pool-safety SVG doodles that float behind the main UI.
 * Opacity is kept at 4–8% so they never compete with content.
 * Only visible in light mode — hidden in dark via opacity-0.
 */
export function PoolBackgroundDoodles() {
  return (
    <div
      className="pointer-events-none absolute inset-0 overflow-hidden transition-opacity duration-500 dark:opacity-0"
      aria-hidden="true"
    >
      {/* ── Top-right: Lifebuoy ring ────────────────────────────── */}
      <svg
        className="absolute -right-8 -top-8 opacity-[0.07] animate-float-slow"
        width="260"
        height="260"
        viewBox="0 0 260 260"
        fill="none"
      >
        <circle cx="130" cy="130" r="110" stroke="#007AFF" strokeWidth="18" />
        <circle cx="130" cy="130" r="68" stroke="#007AFF" strokeWidth="18" />
        {/* 4 coloured segments */}
        <path
          d="M130 20 A110 110 0 0 1 240 130 L196 130 A66 66 0 0 0 130 64 Z"
          fill="#007AFF"
          opacity="0.6"
        />
        <path
          d="M240 130 A110 110 0 0 1 130 240 L130 196 A66 66 0 0 0 196 130 Z"
          fill="#007AFF"
          opacity="0.6"
        />
        {/* Cross lines */}
        <line x1="130" y1="20" x2="130" y2="64" stroke="#007AFF" strokeWidth="18" strokeLinecap="round" />
        <line x1="240" y1="130" x2="196" y2="130" stroke="#007AFF" strokeWidth="18" strokeLinecap="round" />
        <line x1="130" y1="196" x2="130" y2="240" stroke="#007AFF" strokeWidth="18" strokeLinecap="round" />
        <line x1="20" y1="130" x2="64" y2="130" stroke="#007AFF" strokeWidth="18" strokeLinecap="round" />
      </svg>

      {/* ── Top-left: CCTV camera outline ───────────────────────── */}
      <svg
        className="absolute left-8 top-10 opacity-[0.065] animate-float-medium"
        width="120"
        height="90"
        viewBox="0 0 120 90"
        fill="none"
        style={{ animationDelay: "3s" }}
      >
        {/* Camera body */}
        <rect x="8" y="22" width="66" height="46" rx="8" stroke="#007AFF" strokeWidth="4" />
        {/* Lens */}
        <circle cx="41" cy="45" r="14" stroke="#007AFF" strokeWidth="3.5" />
        <circle cx="41" cy="45" r="6" stroke="#007AFF" strokeWidth="2.5" />
        {/* Viewfinder triangle */}
        <path d="M74 30 L108 16 L108 74 L74 60 Z" stroke="#007AFF" strokeWidth="4" strokeLinejoin="round" />
        {/* Mount base */}
        <path d="M41 68 L41 86 M28 86 L54 86" stroke="#007AFF" strokeWidth="4" strokeLinecap="round" />
      </svg>

      {/* ── Center-left large: Safety shield watermark ───────────── */}
      <svg
        className="absolute left-1/4 top-1/2 -translate-y-1/2 opacity-[0.04]"
        width="320"
        height="360"
        viewBox="0 0 320 360"
        fill="none"
      >
        <path
          d="M160 10 L300 60 L300 180 C300 270 160 350 160 350 C160 350 20 270 20 180 L20 60 Z"
          stroke="#007AFF"
          strokeWidth="8"
          fill="#007AFF"
          fillOpacity="0.06"
        />
        {/* Inner shield */}
        <path
          d="M160 50 L260 90 L260 180 C260 240 160 300 160 300 C160 300 60 240 60 180 L60 90 Z"
          stroke="#007AFF"
          strokeWidth="5"
          fill="none"
          strokeDasharray="12 6"
        />
        {/* Check mark */}
        <path
          d="M110 175 L145 215 L210 140"
          stroke="#007AFF"
          strokeWidth="12"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>

      {/* ── Bottom: Water waves ───────────────────────────────────── */}
      <svg
        className="absolute bottom-0 left-0 w-full opacity-[0.07]"
        height="160"
        viewBox="0 0 1440 160"
        preserveAspectRatio="none"
        fill="none"
      >
        {/* Wave 1 */}
        <path
          d="M0 80 C180 40 360 120 540 80 C720 40 900 120 1080 80 C1260 40 1380 100 1440 80 L1440 160 L0 160 Z"
          fill="#007AFF"
          fillOpacity="0.06"
        />
        {/* Wave 2 */}
        <path
          d="M0 100 C200 60 400 130 600 100 C800 70 1000 130 1200 100 C1320 80 1400 110 1440 100 L1440 160 L0 160 Z"
          fill="#5AC8FA"
          fillOpacity="0.05"
        />
        {/* Wave lines */}
        <path
          d="M0 60 C180 30 360 90 540 60 C720 30 900 90 1080 60 C1260 30 1380 70 1440 60"
          stroke="#007AFF"
          strokeWidth="1.5"
          strokeOpacity="0.25"
        />
        <path
          d="M0 75 C200 45 400 100 600 75 C800 50 1000 100 1200 75 C1320 60 1400 85 1440 75"
          stroke="#5AC8FA"
          strokeWidth="1.2"
          strokeOpacity="0.20"
        />
      </svg>

      {/* ── Bottom-left: Swimmer silhouette ──────────────────────── */}
      <svg
        className="absolute bottom-24 left-12 opacity-[0.065] animate-float-slow"
        width="140"
        height="70"
        viewBox="0 0 140 70"
        fill="none"
        style={{ animationDelay: "6s" }}
      >
        {/* Head */}
        <circle cx="118" cy="18" r="10" stroke="#007AFF" strokeWidth="3.5" />
        {/* Body arm stroke */}
        <path
          d="M108 22 C90 28 70 22 50 30 C30 38 16 36 4 34"
          stroke="#007AFF"
          strokeWidth="4"
          strokeLinecap="round"
        />
        {/* Leg kick */}
        <path
          d="M90 32 C80 44 68 50 56 44"
          stroke="#007AFF"
          strokeWidth="4"
          strokeLinecap="round"
        />
        <path
          d="M104 28 C94 40 82 48 70 44"
          stroke="#007AFF"
          strokeWidth="3.5"
          strokeLinecap="round"
        />
      </svg>

      {/* ── Right center: Pool lane lines ────────────────────────── */}
      <svg
        className="absolute right-0 top-1/4 opacity-[0.055]"
        width="80"
        height="400"
        viewBox="0 0 80 400"
        fill="none"
      >
        {[0, 16, 32, 48, 64].map((x) => (
          <g key={x}>
            {/* Lane rope — dashes */}
            {Array.from({ length: 12 }, (_, i) => (
              <circle
                key={i}
                cx={x + 8}
                cy={18 + i * 33}
                r="5"
                fill={i % 3 === 0 ? "#FF3B30" : "#007AFF"}
                opacity="0.5"
              />
            ))}
          </g>
        ))}
      </svg>

      {/* ── Mid-top: AI node network ─────────────────────────────── */}
      <svg
        className="absolute right-64 top-6 opacity-[0.06]"
        width="200"
        height="140"
        viewBox="0 0 200 140"
        fill="none"
      >
        {/* Edges */}
        <line x1="30" y1="70" x2="90" y2="30" stroke="#007AFF" strokeWidth="1.5" />
        <line x1="30" y1="70" x2="90" y2="110" stroke="#007AFF" strokeWidth="1.5" />
        <line x1="90" y1="30" x2="160" y2="50" stroke="#007AFF" strokeWidth="1.5" />
        <line x1="90" y1="110" x2="160" y2="90" stroke="#007AFF" strokeWidth="1.5" />
        <line x1="160" y1="50" x2="160" y2="90" stroke="#007AFF" strokeWidth="1.5" />
        <line x1="90" y1="30" x2="90" y2="110" stroke="#007AFF" strokeWidth="1" strokeDasharray="4 4" />
        <line x1="160" y1="70" x2="190" y2="70" stroke="#007AFF" strokeWidth="1.5" />
        {/* Nodes */}
        {[
          [30, 70], [90, 30], [90, 110], [160, 50], [160, 90], [190, 70],
        ].map(([cx, cy], i) => (
          <circle key={i} cx={cx} cy={cy} r="6" fill="#007AFF" fillOpacity="0.35" stroke="#007AFF" strokeWidth="2" />
        ))}
      </svg>

      {/* ── Scattered bubbles ────────────────────────────────────── */}
      {[
        { cx: 200, cy: 180, r: 9 },
        { cx: 280, cy: 100, r: 5 },
        { cx: 320, cy: 220, r: 7 },
        { cx: 180, cy: 320, r: 11 },
        { cx: 400, cy: 280, r: 6 },
        { cx: 500, cy: 150, r: 4 },
        { cx: 600, cy: 80, r: 8 },
        { cx: 700, cy: 310, r: 5 },
        { cx: 820, cy: 90, r: 10 },
        { cx: 900, cy: 260, r: 6 },
        { cx: 1000, cy: 180, r: 7 },
        { cx: 1100, cy: 310, r: 4 },
        { cx: 1200, cy: 120, r: 9 },
      ].map(({ cx, cy, r }, i) => (
        <svg
          key={i}
          className="absolute opacity-[0.09]"
          style={{ left: cx, top: cy, animationDelay: `${String(i * 1.3)}s` }}
          width={r * 2 + 8}
          height={r * 2 + 8}
          viewBox={`0 0 ${String(r * 2 + 8)} ${String(r * 2 + 8)}`}
          fill="none"
        >
          <circle
            cx={r + 4}
            cy={r + 4}
            r={r}
            stroke="#5AC8FA"
            strokeWidth="2"
            fill="#5AC8FA"
            fillOpacity="0.15"
          />
          {/* Highlight */}
          <circle cx={r + 1} cy={r + 1} r={r * 0.3} fill="white" fillOpacity="0.5" />
        </svg>
      ))}
    </div>
  );
}
