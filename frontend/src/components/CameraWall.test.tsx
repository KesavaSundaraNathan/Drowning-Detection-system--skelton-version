import { fireEvent, screen, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { LiveSummaryCamera } from "../api/types";
import { renderWith } from "../test/utils";
import { CameraWall } from "./CameraWall";

function cam(id: string, name: string, over: Partial<LiveSummaryCamera> = {}): LiveSummaryCamera {
  return { camera_id: id, name, monitoring: "MONITORED", stalled: false, lagging: false, muted_zones: [], ...over };
}

const cams = [cam("a", "Lap pool"), cam("b", "Deep end"), cam("c", "Shallow end", { stalled: true })];

describe("CameraWall", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it("shows the first camera large (high-quality stream) and the others as side previews", () => {
    renderWith(<CameraWall cameras={cams} notMonitored={(id) => id === "c"} feedDown={false} canControl onStop={vi.fn()} />);
    const main = screen.getByTestId("camera-a");
    expect(main).toHaveClass("wall-main");
    expect(within(main).getByRole("img", { name: "Live view of Lap pool" }).getAttribute("data-stream")).toContain("quality=high");
    expect(within(main).getByText("CAM 1")).toBeInTheDocument();
    expect(screen.getByTestId("camera-b")).toHaveClass("wall-thumb");
    // a preview that is not monitored still says so, on the video
    expect(screen.getByTestId("camera-overlay-c")).toHaveTextContent("stream stalled");
  });

  it("swaps a clicked preview into the main view and the old main view to the side", () => {
    renderWith(<CameraWall cameras={cams} notMonitored={() => false} feedDown={false} canControl onStop={vi.fn()} />);
    fireEvent.click(screen.getByRole("button", { name: "Show Deep end in the main view" }));
    expect(screen.getByTestId("camera-b")).toHaveClass("wall-main");
    expect(within(screen.getByTestId("camera-b")).getByText("CAM 2")).toBeInTheDocument(); // keeps its number
    expect(screen.getByTestId("camera-a")).toHaveClass("wall-thumb");
    expect(screen.getByRole("button", { name: "Show Lap pool in the main view" })).toBeInTheDocument();
    expect(localStorage.getItem("lifeguard.cameraWall.main")).toBe("b");
  });

  it("stops the camera in the main view through the confirm callback", () => {
    const onStop = vi.fn();
    renderWith(<CameraWall cameras={cams} notMonitored={() => false} feedDown={false} canControl onStop={onStop} />);
    fireEvent.click(screen.getByRole("button", { name: "Stop monitoring" }));
    expect(onStop).toHaveBeenCalledWith(cams[0]);
  });

  it("falls back to the first camera when the remembered one is not running", () => {
    localStorage.setItem("lifeguard.cameraWall.main", "gone");
    renderWith(<CameraWall cameras={cams} notMonitored={() => false} feedDown={false} canControl={false} onStop={vi.fn()} />);
    expect(screen.getByTestId("camera-a")).toHaveClass("wall-main");
    expect(screen.queryByRole("button", { name: "Stop monitoring" })).toBeNull(); // lifeguards cannot stop
  });
});

describe("live pictures (switching must never leave streams running)", () => {
  it("only the main view holds a live stream; previews use snapshots", () => {
    renderWith(<CameraWall cameras={cams} notMonitored={() => false} feedDown={false} canControl onStop={vi.fn()} />);
    const streams = Array.from(document.querySelectorAll("img[data-stream]"));
    expect(streams).toHaveLength(1);
    expect(streams[0]?.getAttribute("alt")).toBe("Live view of Lap pool");
  });

  it("cancels the old camera's stream when another camera is switched in", () => {
    const streams = new Map<string, AbortSignal>();
    vi.stubGlobal(
      "fetch",
      vi.fn((url: string, init: RequestInit = {}) => {
        if (url.includes("/mjpeg") && init.signal) streams.set(url, init.signal);
        return new Promise<Response>(() => undefined); // a live stream never ends by itself
      }),
    );
    renderWith(<CameraWall cameras={cams} notMonitored={() => false} feedDown={false} canControl onStop={vi.fn()} />);
    const first = screen.getByRole("img", { name: "Live view of Lap pool" });
    const a = "/api/v1/live/cameras/a/mjpeg?quality=high&overlay=client";
    expect(first.getAttribute("data-stream")).toBe(a);
    expect(streams.get(a)?.aborted).toBe(false);
    fireEvent.click(screen.getByRole("button", { name: "Show Deep end in the main view" }));
    expect(streams.get(a)?.aborted).toBe(true); // the request is aborted: the server stops encoding for it
    fireEvent.click(screen.getByRole("button", { name: "Show Shallow end in the main view" }));
    const open = [...streams.entries()].filter(([, s]) => !s.aborted).map(([u]) => u);
    expect(open).toEqual(["/api/v1/live/cameras/c/mjpeg?quality=high&overlay=client"]);
    vi.unstubAllGlobals();
  });
});
