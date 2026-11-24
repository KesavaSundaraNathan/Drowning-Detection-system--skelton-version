import { useMutation, useQueries, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState, type MouseEvent } from "react";
import { Link, useSearchParams } from "react-router-dom";

import { api, media } from "../api/endpoints";
import type { Calibration } from "../api/types";
import { PoolZonesEditor } from "../components/PoolZonesEditor";
import { errorText } from "../hooks";

type Pt = [number, number];

const CORNERS = ["A", "B", "C", "D"] as const;
const DEFAULT_EDGE_BAND_M = 0.6; // thresholds.json ZONE_EDGE_BAND_M
const FALLBACK_FRAME: Pt = [1920, 1080];

/** Corner order A, B, C, D = (0,0), (L,0), (L,W), (0,W) - the same plan for every camera of the pool. */
function cornerMetres(i: number, length: number, width: number): Pt {
  const corners: Pt[] = [[0, 0], [length, 0], [length, width], [0, width]];
  return corners[i] ?? [0, 0];
}

const toSvg = (pts: Pt[]) => pts.map(([x, y]) => `${String(x * 1000)},${String(y * 1000)}`).join(" ");
const round4 = (v: number) => Number(v.toFixed(4));

/**
 * Pool calibration (supervisor+), Phase 7. For each camera, click the pool's corners A, B, C, D on
 * its current frame (extra points take their position in metres), give the pool size and the
 * edge band, save. Every camera of one pool is calibrated against the same pool plan in metres, so
 * positions compare across cameras. A calibrated camera with no WATER zone drawn monitors the
 * projected pool outline; the edge band splits in-water / edge / deck.
 * [v3] Replaces v2's PoolCalibrationPanel (MUI).
 */
export default function CalibrationPage() {
  const [params, setParams] = useSearchParams();
  const cameras = useQuery({ queryKey: ["cameras"], queryFn: api.cameras });
  const cameraId = params.get("camera") ?? cameras.data?.[0]?.id ?? "";
  const all = useQueries({
    queries: (cameras.data ?? []).map((c) => ({
      queryKey: ["calibration", c.id],
      queryFn: () => api.calibration(c.id),
    })),
  });
  const loaded = cameras.isSuccess && all.every((q) => !q.isPending);
  const current = all.find((q) => q.data?.camera_id === cameraId)?.data;
  // multi-camera: the pool plan comes from a camera already calibrated on this pool
  const shared = all.map((q) => q.data).find((d): d is Calibration => d?.pool_size_m != null);
  const sizes = new Set(all.flatMap((q) => (q.data?.pool_size_m ? [q.data.pool_size_m.join(" x ")] : [])));
  // [Phase 11 WP2] pool zones are drawn on the plan the calibrated cameras share
  const calibratedCams = (cameras.data ?? []).filter((_, i) => {
    const cal = all[i]?.data?.calibration;
    return cal ? !cal.error : false;
  });

  return (
    <div>
      <h1>Pool calibration</h1>
      <p>
        Advanced: metric calibration (corners A-D and metres) is only for pool-plan zones and the coverage check. To say
        where the water is, draw a water zone with any number of points on the <Link to={`/zones${cameraId ? `?camera=${cameraId}` : ""}`}>Zones page</Link>;
        a drawn water zone takes precedence over the calibrated outline.
      </p>
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
      {cameraId && loaded && <CalibrationEditor key={cameraId} cameraId={cameraId} current={current} shared={shared} />}
      {sizes.size > 1 && (
        <p role="alert" className="error">
          Cameras are calibrated against different pool sizes ({[...sizes].join(" m, ")} m). Every camera of one pool must
          use the same pool plan, or positions will not match across cameras.
        </p>
      )}
      <section className="card" aria-label="Cameras on this pool">
        <h2>All cameras</h2>
        <ul>
          {cameras.data?.map((c, i) => {
            const cal = all[i]?.data;
            const state = cal?.calibration?.error
              ? "calibration unusable"
              : cal?.calibration
                ? `calibrated (${String(cal.pool_size_m?.[0])} x ${String(cal.pool_size_m?.[1])} m)`
                : "not calibrated";
            return (
              <li key={c.id}>
                {c.name}: {state}
              </li>
            );
          })}
        </ul>
      </section>
      {loaded &&
        (shared?.pool_size_m ? (
          <PoolZonesEditor
            poolSize={shared.pool_size_m}
            poolId={shared.calibration?.pool_id ?? "pool"}
            cameras={calibratedCams.map((c) => ({ id: c.id, name: c.name }))}
          />
        ) : (
          <section className="card" aria-label="Pool zones">
            <h2>Pool zones</h2>
            <p>Calibrate at least one camera first: pool zones are drawn on the pool plan the calibration defines.</p>
          </section>
        ))}
    </div>
  );
}

function CalibrationEditor({
  cameraId,
  current,
  shared,
}: {
  cameraId: string;
  current: Calibration | undefined;
  shared: Calibration | undefined;
}) {
  const qc = useQueryClient();
  const [image, setImage] = useState<Pt[]>([]);
  const [extraMetres, setExtraMetres] = useState<Record<number, Pt>>({});
  const [length, setLength] = useState(shared?.pool_size_m?.[0] ?? 25);
  const [width, setWidth] = useState(shared?.pool_size_m?.[1] ?? 12.5);
  const [edgeBand, setEdgeBand] = useState(current?.calibration?.edge_band_m ?? DEFAULT_EDGE_BAND_M);
  const [frame, setFrame] = useState<Pt>(FALLBACK_FRAME);
  const [frameOk, setFrameOk] = useState(true);
  const sizeFromPool = shared !== undefined;

  const metres: Pt[] = image.map((_, i) => (i < 4 ? cornerMetres(i, length, width) : (extraMetres[i] ?? [0, 0])));
  const done = (c: Calibration) => {
    qc.setQueryData(["calibration", c.camera_id], c);
    setImage([]);
    setExtraMetres({});
  };
  const save = useMutation({
    mutationFn: () =>
      api.saveCalibration(cameraId, {
        image_points: image,
        pool_points: metres,
        edge_band_m: edgeBand,
        frame_width: frame[0],
        frame_height: frame[1],
      }),
    onSuccess: done,
  });
  const remove = useMutation({ mutationFn: () => api.deleteCalibration(cameraId), onSuccess: done });

  const click = (e: MouseEvent<SVGSVGElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const x = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width));
    const y = Math.min(1, Math.max(0, (e.clientY - r.top) / r.height));
    setImage((p) => (p.length >= 16 ? p : [...p, [round4(x), round4(y)]]));
  };
  const stored = current?.calibration;
  const label = (i: number) => CORNERS[i] ?? `P${String(i + 1)}`;

  return (
    <>
        <div className="zone-editor">
          <div className="zone-canvas">
            {frameOk ? (
              <img
                src={media.frame(cameraId)}
                alt="Current camera frame"
                onLoad={(e) => { setFrame([e.currentTarget.naturalWidth || FALLBACK_FRAME[0], e.currentTarget.naturalHeight || FALLBACK_FRAME[1]]); }}
                onError={() => { setFrameOk(false); }}
              />
            ) : (
              <div className="no-frame">No live frame (camera not running) - start the camera to see the pool corners.</div>
            )}
            <svg viewBox="0 0 1000 1000" preserveAspectRatio="none" onClick={click} aria-label="Calibration area" role="img">
              {current?.water_outline_norm && (
                <polygon
                  data-testid="calibrated-outline"
                  points={toSvg(current.water_outline_norm)}
                  fill="#2b8cff"
                  fillOpacity={0.18}
                  stroke="#2b8cff"
                  strokeWidth={3}
                />
              )}
              {stored?.image_points.map(([x, y], i) => (
                <circle key={`s${String(i)}`} cx={x * 1000} cy={y * 1000} r={7} fill="#2b8cff" />
              ))}
              {image.length > 1 && (
                <polygon points={toSvg(image)} fill="none" stroke="#ffffff" strokeWidth={3} strokeDasharray="12 8" />
              )}
              {image.map(([x, y], i) => (
                <g key={i}>
                  <circle cx={x * 1000} cy={y * 1000} r={9} fill="#ffffff" />
                  <text x={x * 1000 + 14} y={y * 1000 - 12} fill="#ffffff" fontSize={36} fontWeight={700}>
                    {label(i)}
                  </text>
                </g>
              ))}
            </svg>
          </div>
          <form
            className="card"
            onSubmit={(e) => {
              e.preventDefault();
              save.mutate();
            }}
          >
            <p>
              Click the pool corners in order: <strong>A</strong> (0, 0), <strong>B</strong> (length, 0), <strong>C</strong>{" "}
              (length, width), <strong>D</strong> (0, width). Use the same corner letters on every camera of this pool.
              Extra points (up to 16) need their position in metres.
            </p>
            <label>
              Pool length (m)
              <input type="number" min={1} max={200} step={0.1} value={length} onChange={(e) => { setLength(Number(e.target.value)); }} required />
            </label>
            <label>
              Pool width (m)
              <input type="number" min={1} max={200} step={0.1} value={width} onChange={(e) => { setWidth(Number(e.target.value)); }} required />
            </label>
            {sizeFromPool && <p className="muted">Pool size taken from a camera already calibrated on this pool.</p>}
            <label>
              Edge band (m): people within this distance of the edge count as "edge", not in the water
              <input type="number" min={0} max={5} step={0.1} value={edgeBand} onChange={(e) => { setEdgeBand(Number(e.target.value)); }} required />
            </label>
            {image.length > 0 && (
              <table className="calib-points">
                <thead>
                  <tr>
                    <th>Point</th>
                    <th>Image (x, y)</th>
                    <th>Pool (m)</th>
                  </tr>
                </thead>
                <tbody>
                  {image.map(([x, y], i) => (
                    <tr key={i}>
                      <td>{label(i)}</td>
                      <td>
                        {x.toFixed(3)}, {y.toFixed(3)}
                      </td>
                      <td>
                        {i < 4 ? (
                          `${String(metres[i]?.[0])}, ${String(metres[i]?.[1])}`
                        ) : (
                          <>
                            <input
                              aria-label={`${label(i)} x metres`}
                              type="number"
                              step={0.1}
                              value={extraMetres[i]?.[0] ?? 0}
                              onChange={(e) => { setExtraMetres((m) => ({ ...m, [i]: [Number(e.target.value), m[i]?.[1] ?? 0] })); }}
                            />
                            <input
                              aria-label={`${label(i)} y metres`}
                              type="number"
                              step={0.1}
                              value={extraMetres[i]?.[1] ?? 0}
                              onChange={(e) => { setExtraMetres((m) => ({ ...m, [i]: [m[i]?.[0] ?? 0, Number(e.target.value)] })); }}
                            />
                          </>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
            <p>{image.length} point(s) placed - at least 4.</p>
            <button type="submit" className="btn-primary" disabled={image.length < 4 || save.isPending}>
              Save calibration
            </button>
            <button type="button" onClick={() => { setImage((p) => p.slice(0, -1)); }} disabled={image.length === 0}>
              Undo last point
            </button>
            <button type="button" onClick={() => { setImage([]); }} disabled={image.length === 0}>
              Clear points
            </button>
            {(save.isError || remove.isError) && (
              <p role="alert" className="error">
                {errorText(save.error ?? remove.error)}
              </p>
            )}
          </form>
        </div>
        <section className="card" aria-label="Current calibration">
          <h2>This camera</h2>
          {stored?.error ? (
            <p role="alert" className="error">
              Stored calibration is unusable ({stored.error}) - zones drawn on this camera still apply. Re-calibrate.
            </p>
          ) : current?.calibration ? (
            <>
              <p>
                Calibrated with {current.points} points; pool {current.pool_size_m?.[0]} x {current.pool_size_m?.[1]} m;
                reprojection error {current.reprojection_error_m} m; edge band {current.calibration.edge_band_m ?? DEFAULT_EDGE_BAND_M} m.
                {current.water_outline_norm ? " The blue outline is the pool as this camera sees it." : " The pool outline is not visible in this view."}
              </p>
              <button
                type="button"
                onClick={() => {
                  if (window.confirm("Remove this camera's pool calibration?")) remove.mutate();
                }}
              >
                Remove calibration
              </button>
            </>
          ) : (
            <p>Not calibrated: only zones drawn on this camera decide where the water is.</p>
          )}
        </section>
    </>
  );
}
