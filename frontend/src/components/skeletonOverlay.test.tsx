import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { BoxOverlay } from "./liveFrames";
import { joints, skeletonPath } from "./skeletonOverlay";
import { parseBoxes, type FrameBoxes } from "./viewMode";

// [rtmpose clone] skeletons ride with the boxes ("k") and are drawn by the browser
const row = (id: number, conf: number) => [id, ...Array.from({ length: 17 }, (_, j) => [10 + j, 20 + 2 * j, conf]).flat()];
const BOXES: FrameBoxes = {
  w: 200,
  h: 100,
  t: [[3, 0, 0, 50, 90, 0, 1], [4, 60, 0, 110, 90, 0, 0]],
  c: [],
  k: [row(3, 90), row(4, 90)],
};

describe("skeleton overlay", () => {
  it("builds one path of the confident limbs and skips weak joints", () => {
    expect(skeletonPath(row(1, 90)).match(/M/g)).toHaveLength(16);
    expect(skeletonPath(row(1, 10))).toBe("");
    expect(joints(row(1, 90))).toHaveLength(17);
    expect(joints(row(1, 10))).toHaveLength(0);
  });

  it("draws per mode and toggle: Full both, Alerts only the alerting person, Clean or off none", () => {
    const n = (c: HTMLElement) => c.querySelectorAll("[data-skeleton]").length;
    const { container, rerender } = render(<BoxOverlay boxes={BOXES} mode="full" />);
    expect(n(container)).toBe(2);
    rerender(<BoxOverlay boxes={BOXES} mode="alerts" />);
    expect(n(container)).toBe(1);
    rerender(<BoxOverlay boxes={BOXES} mode="clean" />);
    expect(n(container)).toBe(0);
    rerender(<BoxOverlay boxes={BOXES} mode="full" skeleton={false} />);
    expect(n(container)).toBe(0);
  });

  it("parses frames from a server without skeletons", () => {
    expect(parseBoxes('{"w":2,"h":2,"t":[],"c":[]}')?.k).toEqual([]);
  });
});
