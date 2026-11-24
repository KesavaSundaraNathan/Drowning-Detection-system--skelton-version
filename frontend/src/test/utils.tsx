import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, type RenderResult } from "@testing-library/react";
import type { ReactElement } from "react";
import { MemoryRouter } from "react-router-dom";
import { vi } from "vitest";

import { initialState, liveStore } from "../state/liveStore";

export interface Call {
  method: string;
  url: string;
  headers: Record<string, string>;
  body: unknown;
}

type Handler = (call: Call) => { status?: number; body?: unknown };

/** Stub global fetch with "METHOD /path" -> handler routes (query string ignored). Returns the calls. */
export function mockFetch(routes: Record<string, Handler>): Call[] {
  const calls: Call[] = [];
  vi.stubGlobal(
    "fetch",
    vi.fn((input: string, init: RequestInit = {}) => {
      const method = init.method ?? "GET";
      const path = input.split("?")[0] ?? input;
      const headers = (init.headers ?? {}) as Record<string, string>;
      const body = typeof init.body === "string" ? (JSON.parse(init.body) as unknown) : init.body;
      const call: Call = { method, url: input, headers, body };
      calls.push(call);
      const handler = routes[`${method} ${path}`];
      const res = handler ? handler(call) : { status: 404, body: { detail: "not mocked" } };
      const status = res.status ?? 200;
      const text = res.body === undefined ? "" : JSON.stringify(res.body);
      return Promise.resolve(new Response(status === 204 ? null : text, { status }));
    }),
  );
  return calls;
}

export function newClient(): QueryClient {
  return new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 }, mutations: { retry: false } } });
}

export function renderWith(ui: ReactElement, client: QueryClient = newClient(), route = "/"): RenderResult {
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={[route]}>{ui}</MemoryRouter>
    </QueryClientProvider>,
  );
}

export function resetStore(): void {
  liveStore.set({ ...initialState });
}
