import { fireEvent, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { Calibration } from "../api/types";
import { mockFetch, renderWith } from "../test/utils";
import CalibrationPage from "./CalibrationPage";

const cams = [
  { id: "c1", name: "Deep end", source_type: "rtsp", source_display: "rtsp://***", default_mode: "Medium", is_enabled: true },
  { id: "c2", name: "Shallow end", source_type: "rtsp", source_display: "rtsp://***", default_mode: "Medium", is_enabled: true },
];
const none = (id: string): Calibration => ({
  camera_id: id,
  calibration: null,
  points: null,
  reprojection_error_m: null,
  pool_size_m: null,
  water_outline_norm: null,
});
const calibrated = (id: string, size: [number, number]): Calibration => ({
  camera_id: id,
  calibration: {
    image_points: [[0.2, 0.3], [0.8, 0.3], [0.95, 0.9], [0.05, 0.9]],
    pool_points: [[0, 0], [size[0], 0], [size[0], size[1]], [0, size[1]]],
    edge_band_m: 0.8,
  },
  points: 4,
  reprojection_error_m: 0.0,
  pool_size_m: size,
  water_outline_norm: [[0.2, 0.3], [0.8, 0.3], [0.95, 0.9], [0.05, 0.9]],
});

// jsdom has no layout: give the drawing area a 1000 x 1000 box so clicks map to x/1000, y/1000
function clickAt(area: Element, x: number, y: number) {
  fireEvent.click(area, { clientX: x * 1000, clientY: y * 1000 });
}

describe("CalibrationPage", () => {
  beforeEach(() => {
    vi.spyOn(Element.prototype, "getBoundingClientRect").mockReturnValue({
      left: 0, top: 0, width: 1000, height: 1000, right: 1000, bottom: 1000, x: 0, y: 0, toJSON: () => ({}),
    });
  });

  it("takes the pool plan from a calibrated camera and saves corners A-D with their metres", async () => {
    const calls = mockFetch({
      "GET /api/v1/cameras": () => ({ body: cams }),
      "GET /api/v1/cameras/c1/calibration": () => ({ body: none("c1") }),
      "GET /api/v1/cameras/c2/calibration": () => ({ body: calibrated("c2", [25, 12.5]) }),
      "PUT /api/v1/cameras/c1/calibration": () => ({ body: calibrated("c1", [25, 12.5]) }),
    });
    renderWith(<CalibrationPage />, undefined, "/calibration?camera=c1");
    const area = await screen.findByRole("img", { name: "Calibration area" });
    expect(screen.getByLabelText("Pool length (m)")).toHaveValue(25);
    expect(screen.getByLabelText("Pool width (m)")).toHaveValue(12.5);
    expect(screen.getByText(/taken from a camera already calibrated/)).toBeInTheDocument();
    expect(screen.getByText(/Not calibrated/)).toBeInTheDocument();

    const save = screen.getByRole("button", { name: "Save calibration" });
    clickAt(area, 0.2, 0.3);
    clickAt(area, 0.8, 0.3);
    clickAt(area, 0.95, 0.9);
    expect(save).toBeDisabled(); // < 4 points
    clickAt(area, 0.05, 0.9);
    fireEvent.change(screen.getByLabelText(/Edge band/), { target: { value: "1.2" } });
    fireEvent.click(save);

    await waitFor(() => { expect(calls.some((c) => c.method === "PUT")).toBe(true); });
    const put = calls.find((c) => c.method === "PUT");
    expect(put?.body).toEqual({
      image_points: [[0.2, 0.3], [0.8, 0.3], [0.95, 0.9], [0.05, 0.9]],
      pool_points: [[0, 0], [25, 0], [25, 12.5], [0, 12.5]],
      edge_band_m: 1.2,
      frame_width: 1920,
      frame_height: 1080,
    });
    expect(await screen.findByText(/Calibrated with 4 points/)).toBeInTheDocument();
    expect(screen.getByTestId("calibrated-outline")).toBeInTheDocument();
  });

  it("extra points need their position in metres; undo removes the last point", async () => {
    const calls = mockFetch({
      "GET /api/v1/cameras": () => ({ body: cams.slice(0, 1) }),
      "GET /api/v1/cameras/c1/calibration": () => ({ body: none("c1") }),
      "PUT /api/v1/cameras/c1/calibration": () => ({ body: calibrated("c1", [25, 12.5]) }),
    });
    renderWith(<CalibrationPage />, undefined, "/calibration");
    const area = await screen.findByRole("img", { name: "Calibration area" });
    for (const [x, y] of [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9], [0.5, 0.5], [0.6, 0.6]] as const) clickAt(area, x, y);
    fireEvent.click(screen.getByRole("button", { name: "Undo last point" }));
    fireEvent.change(screen.getByLabelText("P5 x metres"), { target: { value: "12.5" } });
    fireEvent.change(screen.getByLabelText("P5 y metres"), { target: { value: "6.25" } });
    fireEvent.click(screen.getByRole("button", { name: "Save calibration" }));
    await waitFor(() => { expect(calls.some((c) => c.method === "PUT")).toBe(true); });
    const body = calls.find((c) => c.method === "PUT")?.body as { pool_points: number[][]; image_points: number[][] };
    expect(body.image_points).toHaveLength(5);
    expect(body.pool_points[4]).toEqual([12.5, 6.25]);
  });

  it("shows the server's reason when a calibration is rejected", async () => {
    mockFetch({
      "GET /api/v1/cameras": () => ({ body: cams.slice(0, 1) }),
      "GET /api/v1/cameras/c1/calibration": () => ({ body: none("c1") }),
      "PUT /api/v1/cameras/c1/calibration": () => ({
        status: 422,
        body: { detail: "invalid_calibration: the points do not define a valid mapping (are 3 of them on one line?)" },
      }),
    });
    renderWith(<CalibrationPage />, undefined, "/calibration");
    const area = await screen.findByRole("img", { name: "Calibration area" });
    for (const x of [0.1, 0.3, 0.6, 0.9]) clickAt(area, x, 0.5);
    fireEvent.click(screen.getByRole("button", { name: "Save calibration" }));
    expect(await screen.findByRole("alert")).toHaveTextContent(/one line/);
  });

  it("flags cameras calibrated against different pool sizes and unusable stored calibrations", async () => {
    mockFetch({
      "GET /api/v1/cameras": () => ({ body: [...cams, { ...cams[0], id: "c3", name: "Slide" }] }),
      "GET /api/v1/cameras/c1/calibration": () => ({ body: calibrated("c1", [25, 12.5]) }),
      "GET /api/v1/cameras/c2/calibration": () => ({ body: calibrated("c2", [50, 21]) }),
      "GET /api/v1/cameras/c3/calibration": () => ({
        body: { ...none("c3"), calibration: { image_points: [], pool_points: [], error: "need at least 4 image points" } },
      }),
    });
    renderWith(<CalibrationPage />, undefined, "/calibration?camera=c3");
    expect(await screen.findByText(/different pool sizes/)).toBeInTheDocument();
    expect(screen.getByText(/Stored calibration is unusable/)).toBeInTheDocument();
    expect(screen.getByText("Slide: calibration unusable")).toBeInTheDocument();
    expect(screen.getByText("Deep end: calibrated (25 x 12.5 m)")).toBeInTheDocument();
  });
});
