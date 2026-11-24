/**
 * Same-origin JSON client. Auth is cookie-only (HttpOnly session cookie); every mutating request
 * carries the double-submit CSRF token read from the (JS-readable) CSRF cookie. Nothing secret is
 * ever stored in localStorage.
 */

export const CSRF_COOKIE = "__Host-lifeguard_csrf";
export const API = "/api/v1";

export class ApiError extends Error {
  readonly status: number;
  readonly detail: unknown;

  constructor(status: number, detail: unknown, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

export function readCookie(name: string, source: string = document.cookie): string | null {
  for (const part of source.split(";")) {
    const [k, ...rest] = part.trim().split("=");
    if (k === name) {
      return decodeURIComponent(rest.join("="));
    }
  }
  return null;
}

type Method = "GET" | "POST" | "PUT" | "PATCH" | "DELETE";

export interface RequestOptions {
  body?: unknown;
  raw?: Blob;
  query?: Record<string, string | number | boolean | null | undefined>;
}

function url(path: string, query?: RequestOptions["query"]): string {
  const qs = new URLSearchParams();
  for (const [k, v] of Object.entries(query ?? {})) {
    if (v !== null && v !== undefined) {
      qs.set(k, String(v));
    }
  }
  const q = qs.toString();
  return `${API}${path}${q ? `?${q}` : ""}`;
}

function messageOf(status: number, detail: unknown): string {
  if (typeof detail === "string") return detail;
  if (detail && typeof detail === "object" && "message" in detail && typeof detail.message === "string") {
    return detail.message;
  }
  return `HTTP ${String(status)}`;
}

export async function request<T>(method: Method, path: string, opts: RequestOptions = {}): Promise<T> {
  const headers: Record<string, string> = { Accept: "application/json" };
  if (method !== "GET") {
    const csrf = readCookie(CSRF_COOKIE);
    if (csrf) headers["X-CSRF-Token"] = csrf;
  }
  let body: BodyInit | undefined;
  if (opts.raw !== undefined) {
    body = opts.raw;
    headers["Content-Type"] = "application/octet-stream";
  } else if (opts.body !== undefined) {
    body = JSON.stringify(opts.body);
    headers["Content-Type"] = "application/json";
  }
  const init: RequestInit = { method, headers, credentials: "same-origin", cache: "no-store" };
  if (body !== undefined) init.body = body;
  const res = await fetch(url(path, opts.query), init);
  if (res.status === 204) {
    return undefined as T;
  }
  const text = await res.text();
  let data: unknown = null;
  if (text) {
    try {
      data = JSON.parse(text) as unknown;
    } catch {
      data = text;
    }
  }
  if (!res.ok) {
    const obj = data as { detail?: unknown; message?: unknown } | null;
    const detail = obj && typeof obj === "object" && "detail" in obj ? obj.detail : data;
    const msg = obj && typeof obj === "object" && typeof obj.message === "string" ? obj.message : messageOf(res.status, detail);
    throw new ApiError(res.status, detail, msg);
  }
  return data as T;
}
