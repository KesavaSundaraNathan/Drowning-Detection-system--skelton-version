import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState, type MouseEvent } from "react";
import { Link, useSearchParams } from "react-router-dom";

import { api, media } from "../api/endpoints";
import type { ZoneType } from "../api/types";
import { errorText } from "../hooks";
import { ZONE_COLOURS as COLOURS } from "../zoneColours";

/**
 * Zone editor (supervisor+): click on the camera's current frame to place the polygon's points (any number,
 * 3 or more, any shape; Undo removes the last), name it, pick its type, sensitivity and
 * (optionally) a schedule, save. A WATER zone is where the camera's water is: no corner letters, no metres. Coordinates are stored
 * normalised to [0, 1], so they survive a camera resolution change.
 * [v3] Replaces v2's ZoneMapping / PoolCalibrationPanel (MUI) with a small SVG editor.
 */
export default function ZonesPage() {
  const qc = useQueryClient();
  const [params, setParams] = useSearchParams();
  const cameras = useQuery({ queryKey: ["cameras"], queryFn: api.cameras });
  const cameraId = params.get("camera") ?? cameras.data?.[0]?.id ?? "";
  const zones = useQuery({ queryKey: ["zones", cameraId], queryFn: () => api.zones(cameraId), enabled: cameraId !== "" });
  const schedules = useQuery({ queryKey: ["schedules"], queryFn: api.schedules });
  const [points, setPoints] = useState<[number, number][]>([]);
  const [name, setName] = useState("");
  const [type, setType] = useState<ZoneType>("WATER");
  const [sensitivity, setSensitivity] = useState(1);
  const [scheduleId, setScheduleId] = useState<number | null>(null);
  const [frameOk, setFrameOk] = useState(true);
  const invalidate = () => void qc.invalidateQueries({ queryKey: ["zones", cameraId] });
  const create = useMutation({
    mutationFn: () =>
      api.createZone(cameraId, { name, zone_type: type, polygon: points, sensitivity, schedule_id: scheduleId }),
    onSuccess: () => {
      setPoints([]);
      setName("");
      invalidate();
    },
  });
  const remove = useMutation({ mutationFn: api.deleteZone, onSuccess: invalidate });

  const click = (e: MouseEvent<SVGSVGElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const x = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width));
    const y = Math.min(1, Math.max(0, (e.clientY - r.top) / r.height));
    setPoints((p) => [...p, [Number(x.toFixed(4)), Number(y.toFixed(4))]]);
  };
  const toSvg = (pts: [number, number][]) => pts.map(([x, y]) => `${String(x * 1000)},${String(y * 1000)}`).join(" ");

  return (
    <div>
      <h1>Zones</h1>
      <label>
        Camera{" "}
        <select value={cameraId} onChange={(e) => { setParams({ camera: e.target.value }); }}>
          {cameras.data?.map((c) => (
            <option key={c.id} value={c.id}>
              {c.name}
            </option>
          ))}
        </select>
      </label>
      {cameraId && (
        <div className="zone-editor">
          <div className="zone-canvas">
            {frameOk ? (
              <img src={media.frame(cameraId)} alt="Current camera frame" onError={() => { setFrameOk(false); }} />
            ) : (
              <div className="no-frame">No live frame (camera not running) - zones can still be drawn.</div>
            )}
            <svg viewBox="0 0 1000 1000" preserveAspectRatio="none" onClick={click} aria-label="Zone drawing area" role="img">
              {zones.data?.map((z) => (
                <polygon key={z.id} points={toSvg(z.polygon ?? [])} fill={COLOURS[z.zone_type]} fillOpacity={0.25} stroke={COLOURS[z.zone_type]} strokeWidth={3} />
              ))}
              {points.length > 0 && (
                <polygon points={toSvg(points)} fill="#ffffff" fillOpacity={0.2} stroke="#ffffff" strokeWidth={3} strokeDasharray="12 8" />
              )}
              {points.map(([x, y], i) => (
                <circle key={i} cx={x * 1000} cy={y * 1000} r={8} fill="#ffffff" />
              ))}
            </svg>
          </div>
          <form
            className="card"
            onSubmit={(e) => {
              e.preventDefault();
              create.mutate();
            }}
          >
            <p>{points.length} point(s) placed - click on the image to add more (3 or more, any shape); Undo removes the last point.</p>
            <p>Draw the WATER zone around the pool water. People outside it are treated as on land. No metres needed. <Link to={`/calibration?camera=${cameraId}`}>Metric calibration (advanced)</Link></p>
            <label>
              Name
              <input value={name} onChange={(e) => { setName(e.target.value); }} required />
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
            <button type="submit" className="btn-primary" disabled={points.length < 3 || !name || create.isPending}>
              Save zone
            </button>
            <button type="button" onClick={() => { setPoints((p) => p.slice(0, -1)); }} disabled={points.length === 0}>
              Undo last point
            </button>
            <button type="button" onClick={() => { setPoints([]); }}>
              Clear points
            </button>
            {create.isError && (
              <p role="alert" className="error">
                {errorText(create.error)}
              </p>
            )}
          </form>
        </div>
      )}
      <ul className="zone-list">
        {zones.data?.map((z) => (
          <li key={z.id}>
            <span className="swatch" style={{ background: COLOURS[z.zone_type] }} /> {z.name} ({z.zone_type}, sensitivity{" "}
            {z.sensitivity}, {z.schedule_id === null ? "always monitored" : `schedule ${String(z.schedule_id)}`}){" "}
            <button
              type="button"
              onClick={() => {
                if (window.confirm(`Delete zone ${z.name}?`)) remove.mutate(z.id);
              }}
            >
              Delete
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}
