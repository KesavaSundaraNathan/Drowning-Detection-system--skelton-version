import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState, type MouseEvent } from "react";

import { api, media, type PoolZoneBody } from "../api/endpoints";
import type { Zone, ZoneType } from "../api/types";
import { errorText } from "../hooks";
import { polygonAreaM2 } from "../poolPlan";
import { ZONE_COLOURS } from "../zoneColours";

type Pt = [number, number];

const DECK_M = 1.5; // deck drawn around the pool (zones may reach onto it)
const PX = 40; // SVG units per metre
const MIN_AREA_M2 = 0.25; // the API refuses smaller (schemas.PoolZoneIn)
const MAX_POINTS = 64;

const round1 = (v: number) => Math.round(v * 10) / 10;
const svgPts = (pts: readonly Pt[], k = PX) => pts.map(([x, y]) => `${String(x * k)},${String(y * k)}`).join(" ");
const centroid = (pts: readonly Pt[]): Pt => [
  pts.reduce((s, p) => s + p[0], 0) / Math.max(1, pts.length),
  pts.reduce((s, p) => s + p[1], 0) / Math.max(1, pts.length),
];

/**
 * [Phase 11 WP2] Pool-plan zone editor (supervisor+). A zone is drawn ONCE on the pool plan in metres - click on the
 * plan or type the corners - and every calibrated camera of the pool projects it into its own view, so the same spot
 * of the pool is in the same zone on every camera. While drawing, each calibrated camera's current frame shows where
 * the zone will land (server-side projection through the camera's calibration). Corner letters match the
 * calibration: A = (0, 0), B = (length, 0), C = (length, width), D = (0, width).
 */
export function PoolZonesEditor({
  poolSize,
  poolId,
  cameras,
}: {
  poolSize: Pt;
  poolId: string;
  cameras: { id: string; name: string }[];
}) {
  const qc = useQueryClient();
  const [length, width] = poolSize;
  const zones = useQuery({ queryKey: ["pool-zones"], queryFn: api.poolZones });
  const schedules = useQuery({ queryKey: ["schedules"], queryFn: api.schedules });
  const [points, setPoints] = useState<Pt[]>([]);
  const [name, setName] = useState("");
  const [type, setType] = useState<ZoneType>("SHALLOW");
  const [sensitivity, setSensitivity] = useState(1);
  const [scheduleId, setScheduleId] = useState<number | null>(null);
  const [editing, setEditing] = useState<Zone | null>(null);

  const area = points.length >= 3 ? polygonAreaM2(points) : 0;
  const drawable = points.length >= 3 && area >= MIN_AREA_M2;
  const preview = useQuery({
    queryKey: ["pool-zone-preview", poolId, points],
    queryFn: () => api.previewPoolZone({ pool_id: poolId, pool_polygon_m: points }),
    enabled: drawable,
    placeholderData: (prev) => prev,
  });

  const reset = () => {
    setPoints([]);
    setName("");
    setEditing(null);
  };
  const body = (): PoolZoneBody => ({
    name,
    zone_type: type,
    pool_id: poolId,
    pool_polygon_m: points,
    sensitivity,
    schedule_id: scheduleId,
  });
  const save = useMutation({
    mutationFn: () => (editing ? api.updatePoolZone(editing.id, body()) : api.createPoolZone(body())),
    onSuccess: () => {
      reset();
      void qc.invalidateQueries({ queryKey: ["pool-zones"] });
    },
  });
  const remove = useMutation({
    mutationFn: api.deleteZone,
    onSuccess: () => void qc.invalidateQueries({ queryKey: ["pool-zones"] }),
  });

  const edit = (z: Zone) => {
    setEditing(z);
    setPoints((z.pool_polygon_m ?? []).map(([x, y]) => [x, y]));
    setName(z.name);
    setType(z.zone_type);
    setSensitivity(z.sensitivity);
    setScheduleId(z.schedule_id);
  };
  const click = (e: MouseEvent<SVGSVGElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const fx = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width));
    const fy = Math.min(1, Math.max(0, (e.clientY - r.top) / r.height));
    const p: Pt = [round1(-DECK_M + fx * (length + 2 * DECK_M)), round1(-DECK_M + fy * (width + 2 * DECK_M))];
    setPoints((pts) => (pts.length >= MAX_POINTS ? pts : [...pts, p]));
  };
  const setCoord = (i: number, axis: 0 | 1, v: number) => {
    setPoints((pts) => pts.map((p, j) => (j === i ? (axis === 0 ? [v, p[1]] : [p[0], v]) : p)));
  };

  const vb = `${String(-DECK_M * PX)} ${String(-DECK_M * PX)} ${String((length + 2 * DECK_M) * PX)} ${String((width + 2 * DECK_M) * PX)}`;
  const gridX = Array.from({ length: Math.floor(length) + 1 }, (_, i) => i);
  const gridY = Array.from({ length: Math.floor(width) + 1 }, (_, i) => i);
  const others = (zones.data ?? []).filter((z) => z.pool_id === poolId && z.id !== editing?.id);

  return (
    <section className="card" aria-label="Pool zones">
      <h2>Pool zones</h2>
      <p>
        Draw a zone once on the pool plan ({length} x {width} m). Every calibrated camera of this pool projects it into its
        own view, so a person at the same spot is in the same zone on every camera. Click on the plan to add corners
        (0.1 m steps; the shaded band is the deck), or type them below.
      </p>
      <div className="zone-editor">
        <div className="plan-column">
        <div className="pool-plan" style={{ aspectRatio: `${String(length + 2 * DECK_M)} / ${String(width + 2 * DECK_M)}` }}>
          <svg viewBox={vb} preserveAspectRatio="none" onClick={click} role="img" aria-label="Pool plan">
            <rect x={-DECK_M * PX} y={-DECK_M * PX} width={(length + 2 * DECK_M) * PX} height={(width + 2 * DECK_M) * PX} className="plan-deck" />
            <rect x={0} y={0} width={length * PX} height={width * PX} className="plan-water" />
            {gridX.map((x) => (
              <line key={`x${String(x)}`} x1={x * PX} y1={0} x2={x * PX} y2={width * PX} className={x % 5 === 0 ? "plan-grid5" : "plan-grid"} />
            ))}
            {gridY.map((y) => (
              <line key={`y${String(y)}`} x1={0} y1={y * PX} x2={length * PX} y2={y * PX} className={y % 5 === 0 ? "plan-grid5" : "plan-grid"} />
            ))}
            {(
              [
                ["A", 0, 0],
                ["B", length, 0],
                ["C", length, width],
                ["D", 0, width],
              ] as const
            ).map(([l, x, y]) => (
              <text key={l} x={x * PX + (x > 0 ? 6 : -30)} y={y * PX + (y > 0 ? 34 : -8)} className="plan-corner">
                {l}
              </text>
            ))}
            {others.map((z) => {
              const pts = z.pool_polygon_m ?? [];
              const [cx, cy] = centroid(pts);
              return (
                <g key={z.id} data-testid={`plan-zone-${String(z.id)}`}>
                  <polygon points={svgPts(pts)} fill={ZONE_COLOURS[z.zone_type]} fillOpacity={0.3} stroke={ZONE_COLOURS[z.zone_type]} strokeWidth={4} />
                  <text x={cx * PX} y={cy * PX} textAnchor="middle" className="plan-label">
                    {z.name}
                  </text>
                </g>
              );
            })}
            {points.length > 1 && (
              <polygon points={svgPts(points)} data-testid="plan-draft" fill="#ffffff" fillOpacity={0.25} stroke="#ffffff" strokeWidth={4} strokeDasharray="14 9" />
            )}
            {points.map(([x, y], i) => (
              <g key={i}>
                <circle cx={x * PX} cy={y * PX} r={9} fill="#ffffff" stroke="#1b2a4a" strokeWidth={2} />
                <text x={x * PX + 12} y={y * PX - 10} className="plan-vertex">
                  {i + 1}
                </text>
              </g>
            ))}
          </svg>
        </div>
          <h3>Preview on every calibrated camera</h3>
          {!drawable ? (
            <p className="muted">Place at least 3 corners to see where the zone lands in each camera.</p>
          ) : preview.isError ? (
            <p role="alert" className="error">
              Preview failed: {errorText(preview.error)}
            </p>
          ) : (
            <div className="pool-previews">
              {(preview.data ?? []).map((p) => (
                <figure key={p.camera_id} className="pool-preview" data-testid={`preview-${p.camera_id}`}>
                  <div className="zone-canvas">
                    <CameraFrame cameraId={p.camera_id} name={p.camera_name} />
                    <svg viewBox="0 0 1000 1000" preserveAspectRatio="none" aria-hidden="true">
                      {p.polygon && (
                        <polygon points={svgPts(p.polygon, 1000)} fill={ZONE_COLOURS[type]} fillOpacity={0.3} stroke={ZONE_COLOURS[type]} strokeWidth={4} />
                      )}
                    </svg>
                  </div>
                  <figcaption>
                    {p.camera_name}: {p.polygon ? "zone in view" : "not in this camera's view"}
                  </figcaption>
                </figure>
              ))}
              {preview.data?.length === 0 && <p className="muted">No calibrated camera on this pool.</p>}
            </div>
          )}
          {cameras.length > 0 && preview.data && preview.data.length < cameras.length && (
            <p className="muted">Cameras that are not calibrated on this pool plan do not use pool zones.</p>
          )}
        </div>
        <form
          className="card"
          onSubmit={(e) => {
            e.preventDefault();
            save.mutate();
          }}
        >
          <h3>{editing ? `Edit ${editing.name}` : "New pool zone"}</h3>
          {points.length > 0 && (
            <table className="calib-points pool-points">
              <thead>
                <tr>
                  <th aria-label="Corner">#</th>
                  <th>x (m)</th>
                  <th>y (m)</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {points.map(([x, y], i) => (
                  <tr key={i}>
                    <td>{i + 1}</td>
                    <td>
                      <input aria-label={`Corner ${String(i + 1)} x (m)`} type="number" step={0.1} min={-10} max={200} value={x} onChange={(e) => { setCoord(i, 0, Number(e.target.value)); }} />
                    </td>
                    <td>
                      <input aria-label={`Corner ${String(i + 1)} y (m)`} type="number" step={0.1} min={-10} max={200} value={y} onChange={(e) => { setCoord(i, 1, Number(e.target.value)); }} />
                    </td>
                    <td>
                      <button type="button" aria-label={`Remove corner ${String(i + 1)}`} onClick={() => { setPoints((p) => p.filter((_, j) => j !== i)); }}>
                        x
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          <p data-testid="plan-status">
            {points.length} corner(s){points.length >= 3 ? `, ${area.toFixed(1)} m²` : " - at least 3"}
            {points.length >= 3 && area < MIN_AREA_M2 ? " - too small (at least 0.25 m²)" : ""}
          </p>
          <button type="button" onClick={() => { setPoints((p) => [...p, [0, 0]]); }} disabled={points.length >= MAX_POINTS}>
            Add corner by numbers
          </button>
          <label>
            Name
            <input value={name} onChange={(e) => { setName(e.target.value); }} required maxLength={100} />
          </label>
          <label>
            Type
            <select value={type} onChange={(e) => { setType(e.target.value as ZoneType); }}>
              <option value="WATER">WATER (monitored)</option>
              <option value="SHALLOW">SHALLOW (monitored)</option>
              <option value="DECK">DECK (not risk-scored)</option>
            </select>
          </label>
          <label>
            Sensitivity {sensitivity.toFixed(2)}
            <input type="range" min={0.25} max={3} step={0.05} value={sensitivity} onChange={(e) => { setSensitivity(Number(e.target.value)); }} />
          </label>
          <label>
            Schedule (outside its window ordinary alerts are MUTED; Submersion / sudden blackout alerts and faults never are)
            <select value={scheduleId ?? ""} onChange={(e) => { setScheduleId(e.target.value ? Number(e.target.value) : null); }}>
              <option value="">none - always monitored</option>
              {schedules.data?.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.name}
                </option>
              ))}
            </select>
          </label>
          <button type="submit" className="btn-primary" disabled={!drawable || !name.trim() || save.isPending}>
            {editing ? "Save changes" : "Save pool zone"}
          </button>
          <button type="button" onClick={() => { setPoints((p) => p.slice(0, -1)); }} disabled={points.length === 0}>
            Undo last corner
          </button>
          <button type="button" onClick={reset} disabled={points.length === 0 && editing === null}>
            {editing ? "Cancel edit" : "Clear"}
          </button>
          {(save.isError || remove.isError) && (
            <p role="alert" className="error">
              {errorText(save.error ?? remove.error)}
            </p>
          )}
        </form>
      </div>

      <h3>Zones on this pool</h3>
      {zones.data && zones.data.filter((z) => z.pool_id === poolId).length === 0 && <p className="muted">No pool zones yet.</p>}
      <ul className="zone-list">
        {zones.data
          ?.filter((z) => z.pool_id === poolId)
          .map((z) => (
            <li key={z.id}>
              <span className="swatch" style={{ background: ZONE_COLOURS[z.zone_type] }} /> {z.name} ({z.zone_type},{" "}
              {(z.pool_polygon_m ?? []).length} corners, {polygonAreaM2(z.pool_polygon_m ?? []).toFixed(1)} m², sensitivity{" "}
              {z.sensitivity}, {z.schedule_id === null ? "always monitored" : `schedule ${String(z.schedule_id)}`}){" "}
              <button type="button" onClick={() => { edit(z); }}>
                Edit
              </button>
              <button
                type="button"
                onClick={() => {
                  if (window.confirm(`Delete pool zone ${z.name}? Every camera stops using it.`)) remove.mutate(z.id);
                }}
              >
                Delete
              </button>
            </li>
          ))}
      </ul>
    </section>
  );
}

function CameraFrame({ cameraId, name }: { cameraId: string; name: string }) {
  const [ok, setOk] = useState(true);
  return ok ? (
    <img src={media.frame(cameraId)} alt={`Current frame of ${name}`} onError={() => { setOk(false); }} />
  ) : (
    <div className="no-frame">No live frame ({name} not running) - the outline still shows where the zone lands.</div>
  );
}
