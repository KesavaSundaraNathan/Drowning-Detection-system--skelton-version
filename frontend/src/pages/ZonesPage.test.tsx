import { fireEvent, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { mockFetch, renderWith } from "../test/utils";
import ZonesPage from "./ZonesPage";

const cams = [
  { id: "c1", name: "Pool", source_type: "rtsp", source_display: "rtsp://***", default_mode: "Medium", is_enabled: true },
];

// jsdom has no layout: a 1000 x 1000 box maps a click at (x * 1000, y * 1000) to (x, y)
function clickAt(area: Element, x: number, y: number) {
  fireEvent.click(area, { clientX: x * 1000, clientY: y * 1000 });
}

/** n points around a circle inside the frame */
function ring(n: number): [number, number][] {
  return Array.from({ length: n }, (_, i) => {
    const a = (2 * Math.PI * i) / n;
    return [Number((0.5 + 0.4 * Math.cos(a)).toFixed(4)), Number((0.5 + 0.4 * Math.sin(a)).toFixed(4))];
  });
}

describe("ZonesPage free-form water zone", () => {
  beforeEach(() => {
    vi.spyOn(Element.prototype, "getBoundingClientRect").mockReturnValue({
      left: 0, top: 0, width: 1000, height: 1000, right: 1000, bottom: 1000, x: 0, y: 0, toJSON: () => ({}),
    });
  });

  it.each([3, 8, 20, 40])("saves a %i-point water zone: no corner letters, no metres, no point cap", async (n) => {
    const calls = mockFetch({
      "GET /api/v1/cameras": () => ({ body: cams }),
      "GET /api/v1/cameras/c1/zones": () => ({ body: [] }),
      "GET /api/v1/schedules": () => ({ body: [] }),
      "POST /api/v1/cameras/c1/zones": () => ({ body: { id: 1 } }),
    });
    renderWith(<ZonesPage />, undefined, "/zones?camera=c1");
    const area = await screen.findByRole("img", { name: "Zone drawing area" });
    const save = screen.getByRole("button", { name: "Save zone" });
    fireEvent.change(screen.getByLabelText("Name"), { target: { value: "Pool water" } });
    const pts = ring(n);
    for (const [x, y] of pts) clickAt(area, x, y);
    expect(screen.getByText(new RegExp(`${String(n)} point\\(s\\) placed`))).toBeInTheDocument();
    await waitFor(() => { expect(save).toBeEnabled(); });
    fireEvent.click(save);
    await waitFor(() => { expect(calls.some((c) => c.method === "POST")).toBe(true); });
    const post = calls.find((c) => c.method === "POST");
    expect(post?.body).toMatchObject({ name: "Pool water", zone_type: "WATER", polygon: pts });
  });

  it("needs at least 3 points and undo removes the last one", async () => {
    mockFetch({
      "GET /api/v1/cameras": () => ({ body: cams }),
      "GET /api/v1/cameras/c1/zones": () => ({ body: [] }),
      "GET /api/v1/schedules": () => ({ body: [] }),
    });
    renderWith(<ZonesPage />, undefined, "/zones?camera=c1");
    const area = await screen.findByRole("img", { name: "Zone drawing area" });
    fireEvent.change(screen.getByLabelText("Name"), { target: { value: "w" } });
    clickAt(area, 0.1, 0.1);
    clickAt(area, 0.9, 0.1);
    expect(screen.getByRole("button", { name: "Save zone" })).toBeDisabled();
    clickAt(area, 0.9, 0.9);
    expect(screen.getByRole("button", { name: "Save zone" })).toBeEnabled();
    fireEvent.click(screen.getByRole("button", { name: "Undo last point" }));
    expect(screen.getByText(/2 point\(s\) placed/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Save zone" })).toBeDisabled();
  });
});
