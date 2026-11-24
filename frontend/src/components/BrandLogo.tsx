/** The Life Guard badge (branding/lifeguard-logo-*.png, served from frontend/public). Never drawn over video. */
export function BrandLogo({ size }: { size: number }) {
  return (
    <img
      src="/lifeguard-logo-192.png"
      width={size}
      height={size}
      alt="Life Guard"
      className="shrink-0 rounded-full"
      draggable={false}
    />
  );
}
