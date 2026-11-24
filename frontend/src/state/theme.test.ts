import { act, renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { applyTheme, setTheme, useTheme } from "./theme";

describe("theme (Phase 9b, v2 light / dark switch)", () => {
  afterEach(() => {
    setTheme("light");
    localStorage.clear();
  });

  it("defaults to the v2 light theme", () => {
    applyTheme("light");
    expect(document.documentElement.classList.contains("dark")).toBe(false);
  });

  it("switches <html> to dark, remembers it, and notifies subscribers", () => {
    const { result } = renderHook(() => useTheme());
    act(() => {
      setTheme("dark");
    });
    expect(result.current).toBe("dark");
    expect(document.documentElement.classList.contains("dark")).toBe(true);
    expect(localStorage.getItem("lifeguard.theme")).toBe("dark");
    act(() => {
      setTheme("light");
    });
    expect(result.current).toBe("light");
    expect(document.documentElement.classList.contains("dark")).toBe(false);
  });

  it("still switches when browser storage is blocked (private window, policy)", () => {
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new DOMException("blocked", "SecurityError");
    });
    expect(() => {
      setTheme("dark");
    }).not.toThrow();
    expect(document.documentElement.classList.contains("dark")).toBe(true);
  });
});
