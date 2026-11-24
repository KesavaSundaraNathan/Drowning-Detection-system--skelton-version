import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { Calibration, Zone } from "../api/types";
import CalibrationPage from "../pages/CalibrationPage";
import { polygonAreaM2 } from "../poolPlan";
import { mockFetch, renderWith } from "../test/utils";

const cams = [
  { id: "c1", name: "Deep end NW", source_type: "rtsp", source_display: "rtsp://***", default_mode: "Medium", is_enabled: true },
  { id: "c2", name: "Shallow end SE", source_type: "rtsp", source_display: "rtsp://***", default_mode: "Medium", is_enabled: true },
  { id: "c3", name: "Lobby", source_type: "rtsp", source_display: "rtsp://***", default_mode: "Medium", is_enabled: true },
];
const none = (id: string): Calibration => ({
  camera_id: id,
  calibration: null,
  points: null,
  reprojection_error_m: null,
  pool_size_m: null,
  water_outline_norm: null,
});
const calibrated = (id: string): Calibration => ({
  camera_id: id,
  calibration: {
    image_points: [[0.2, 0.3], [0.8, 0.3], [0.95, 0.9], [0.05, 0.9]],
    pool_points: [[0, 0], [25, 0], [25, 12.5], [0, 12.5]],
    edge_band_m: 0.6,
  },
  points: 4,
  reprojection_error_m: 0,
  pool_size_m: [25, 12.5],
  water_outline_norm: [[0.2, 0.3], [0.8, 0.3], [0.95, 0.9], [0.05, 0.9]],
});
const TEACHING: [number, number][] = [[18, 9.5], [24, 9.5], [24, 12], [18, 12]];
const saved: Zone = {
  id: 7,
  camera_id: null,
  name: "Teaching platform",
  zone_type: "SHALLOW",
  polygon: null,
  pool_id: "pool",
  pool_polygon_m: TEACHING,
  sensitivity: 1.5,
  schedule_id: null,
  is_active: true,
};

// jsdom has no layout: the plan is a 1000 x 1000 box; the plan spans the pool (25 x 12.5 m) plus 1.5 m of deck
const clickMetres = (plan: Element, x: number, y: number) => {
  fireEvent.click(plan, { clientX: ((x + 1.5) / 28) * 1000, clientY: ((y + 1.5) / 15.5) * 1000 });
};

function routes(over: Record<string, unknown> = {}) {
  let zones: Zone[] = [];
  return mockFetch({
    "GET /api/v1/cameras": () => ({ body: cams }),
    "GET /api/v1/cameras/c1/calibration": () => ({ body: calibrated("c1") }),
    "GET /api/v1/cameras/c2/calibration": () => ({ body: calibrated("c2") }),
    "GET /api/v1/cameras/c3/calibration": () => ({ body: none("c3") }),
    "GET /api/v1/schedules": () => ({ body: [] }),
    "GET /api/v1/pool-zones": () => ({ body: zones }),
    "POST /api/v1/pool-zones/preview": () => ({
      body: [
        { camera_id: "c1", camera_name: "Deep end NW", polygon: null },
        { camera_id: "c2", camera_name: "Shallow end SE", polygon: [[0.1, 0.5], [0.4, 0.5], [0.4, 0.8], [0.1, 0.8]] },
      ],
    }),
    "POST /api/v1/pool-zones": () => {
      zones = [saved];
      return { status: 201, body: saved };
    },
    "PUT /api/v1/pool-zones/7": () => ({ body: saved }),
    "DELETE /api/v1/zones/7": () => {
      zones = [];
      return { status: 204, body: undefined };
    },
    ...over,
  });
}

describe("pool-plan zone editor (Phase 11 WP2)", () => {
  beforeEach(() => {
    vi.spyOn(Element.prototype, "getBoundingClientRect").mockReturnValue({
      left: 0, top: 0, width: 1000, height: 1000, right: 1000, bottom: 1000, x: 0, y: 0, toJSON: () => ({}),
    });
  });

  it("asks for a calibration first when no camera is calibrated", async () => {
    mockFetch({
      "GET /api/v1/cameras": () => ({ body: cams.slice(2) }),
      "GET /api/v1/cameras/c3/calibration": () => ({ body: none("c3") }),
    });
    renderWith(<CalibrationPage />, undefined, "/calibration");
    expect(await screen.findByText(/Calibrate at least one camera first/)).toBeInTheDocument();
    expect(screen.queryByRole("img", { name: "Pool plan" })).toBeNull();
  });

  it("draws a zone once in metres, previews it on every calibrated camera and saves it", async () => {
    const calls = routes();
    renderWith(<CalibrationPage />, undefined, "/calibration?camera=c1");
    const plan = await screen.findByRole("img", { name: "Pool plan" });
    const section = screen.getByRole("region", { name: "Pool zones" });
    expect(section).toHaveTextContent("25 x 12.5 m");
    for (const [x, y] of TEACHING) clickMetres(plan, x, y);
    expect(screen.getByLabelText("Corner 1 x (m)")).toHaveValue(18);
    expect(screen.getByLabelText("Corner 3 y (m)")).toHaveValue(12);
    expect(screen.getByTestId("plan-status")).toHaveTextContent("4 corner(s), 15.0 m²");
    // live preview: one projection per calibrated camera of the pool; the uncalibrated "Lobby" is not asked
    expect(await screen.findByTestId("preview-c2")).toHaveTextContent("Shallow end SE: zone in view");
    expect(screen.getByTestId("preview-c1")).toHaveTextContent("Deep end NW: not in this camera's view");
    expect(screen.queryByTestId("preview-c3")).toBeNull();
    // the preview follows the drawing: first with 3 corners, then with all 4
    const previews = calls.filter((c) => c.method === "POST" && c.url === "/api/v1/pool-zones/preview");
    expect(previews.map((c) => (c.body as { pool_polygon_m: unknown[] }).pool_polygon_m.length)).toEqual([3, 4]);
    expect(previews.at(-1)?.body).toEqual({ pool_id: "pool", pool_polygon_m: TEACHING });

    fireEvent.change(within(section).getByLabelText("Name"), { target: { value: "Teaching platform" } });
    fireEvent.change(within(section).getByLabelText("Type"), { target: { value: "SHALLOW" } });
    fireEvent.click(within(section).getByRole("button", { name: "Save pool zone" }));
    await waitFor(() => {
      expect(screen.getByTestId("plan-zone-7")).toBeInTheDocument();
    });
    const post = calls.find((c) => c.method === "POST" && c.url === "/api/v1/pool-zones");
    expect(post?.body).toEqual({
      name: "Teaching platform",
      zone_type: "SHALLOW",
      pool_id: "pool",
      pool_polygon_m: TEACHING,
      sensitivity: 1,
      schedule_id: null,
    });
    expect(within(section).getByText(/Teaching platform \(SHALLOW, 4 corners, 15.0 m²/)).toBeInTheDocument();
    expect(screen.getByTestId("plan-status")).toHaveTextContent("0 corner(s)"); // the form resets
  });

  it("edits an existing zone by numbers and deletes it", async () => {
    const calls = routes({ "GET /api/v1/pool-zones": () => ({ body: [saved] }) });
    vi.spyOn(window, "confirm").mockReturnValue(true);
    renderWith(<CalibrationPage />, undefined, "/calibration?camera=c1");
    const section = await screen.findByRole("region", { name: "Pool zones" });
    fireEvent.click(await within(section).findByRole("button", { name: "Edit" }));
    expect(within(section).getByRole("heading", { name: "Edit Teaching platform" })).toBeInTheDocument();
    expect(screen.queryByTestId("plan-zone-7")).toBeNull(); // drawn as the editable draft instead
    fireEvent.change(screen.getByLabelText("Corner 1 x (m)"), { target: { value: "17.5" } });
    fireEvent.click(within(section).getByRole("button", { name: "Save changes" }));
    await waitFor(() => {
      expect(calls.some((c) => c.method === "PUT" && c.url === "/api/v1/pool-zones/7")).toBe(true);
    });
    const put = calls.find((c) => c.method === "PUT");
    expect((put?.body as { pool_polygon_m: number[][] }).pool_polygon_m[0]).toEqual([17.5, 9.5]);
    expect((put?.body as { sensitivity: number }).sensitivity).toBe(1.5); // kept from the zone
    fireEvent.click(within(section).getByRole("button", { name: "Delete" }));
    await waitFor(() => {
      expect(calls.some((c) => c.method === "DELETE" && c.url === "/api/v1/zones/7")).toBe(true);
    });
  });

  it("refuses a zone that is too small and does not preview it", async () => {
    const calls = routes();
    renderWith(<CalibrationPage />, undefined, "/calibration?camera=c1");
    const plan = await screen.findByRole("img", { name: "Pool plan" });
    const section = screen.getByRole("region", { name: "Pool zones" });
    clickMetres(plan, 5, 5);
    clickMetres(plan, 5.3, 5);
    clickMetres(plan, 5.3, 5.3);
    expect(screen.getByTestId("plan-status")).toHaveTextContent("too small");
    fireEvent.change(within(section).getByLabelText("Name"), { target: { value: "Speck" } });
    expect(within(section).getByRole("button", { name: "Save pool zone" })).toBeDisabled();
    expect(calls.some((c) => c.url === "/api/v1/pool-zones/preview")).toBe(false);
  });

  it("computes polygon areas in square metres", () => {
    expect(polygonAreaM2(TEACHING)).toBeCloseTo(15);
    expect(polygonAreaM2([[0, 0], [10, 0], [20, 0]])).toBe(0);
  });
});
