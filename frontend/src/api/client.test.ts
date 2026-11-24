import { afterEach, describe, expect, it } from "vitest";

import { mockFetch } from "../test/utils";
import { ApiError, CSRF_COOKIE, readCookie, request } from "./client";

describe("API client", () => {
  afterEach(() => {
    document.cookie = `${CSRF_COOKIE}=; expires=Thu, 01 Jan 1970 00:00:00 GMT; path=/; Secure`;
  });

  it("reads a cookie by exact name", () => {
    expect(readCookie("b", "a=1; b=two%20words; bb=3")).toBe("two words");
    expect(readCookie("c", "a=1")).toBeNull();
  });

  it("sends the CSRF token on mutations only, same-origin, never cached", async () => {
    document.cookie = `${CSRF_COOKIE}=tok123; path=/; Secure`;
    const calls = mockFetch({ "GET /api/v1/x": () => ({ body: [] }), "POST /api/v1/y": () => ({ body: { ok: 1 } }) });
    await request("GET", "/x", { query: { a: 1, b: null } });
    await request("POST", "/y", { body: { z: 2 } });
    expect(calls[0]?.url).toBe("/api/v1/x?a=1");
    expect(calls[0]?.headers["X-CSRF-Token"]).toBeUndefined();
    expect(calls[1]?.headers["X-CSRF-Token"]).toBe("tok123");
    expect(calls[1]?.body).toEqual({ z: 2 });
  });

  it("turns an error body into ApiError with the server's detail", async () => {
    mockFetch({
      "POST /api/v1/live/cameras/c/start": () => ({
        status: 409,
        body: { detail: { code: "capacity_exceeded", limit: 1 }, message: "at most 1 camera" },
      }),
    });
    const err = await request("POST", "/live/cameras/c/start").catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect((err as ApiError).status).toBe(409);
    expect((err as ApiError).detail).toEqual({ code: "capacity_exceeded", limit: 1 });
    expect((err as ApiError).message).toBe("at most 1 camera");
  });
});
