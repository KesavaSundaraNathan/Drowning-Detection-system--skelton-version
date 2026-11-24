import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { App } from "./App";
import { makeAlert } from "./test/fixtures";
import { mockFetch, newClient, resetStore } from "./test/utils";

class QuietSocket {
  static last: QuietSocket | null = null;
  onopen: (() => void) | null = null;
  onmessage: ((ev: { data: unknown }) => void) | null = null;
  onclose: ((ev: { code: number }) => void) | null = null;
  constructor(readonly url: string) {
    QuietSocket.last = this;
  }
  close() {
    return undefined;
  }
}

const user = (role: string) => ({ id: 1, username: "pat", role, is_active: true, created_at: "2026-10-07T00:00:00Z" });

function backend(role = "lifeguard") {
  let loggedIn = false;
  return mockFetch({
    "GET /api/v1/auth/me": () => (loggedIn ? { body: user(role) } : { status: 401, body: { detail: "not_authenticated" } }),
    "POST /api/v1/auth/login": (c) => {
      const b = c.body as { password: string };
      if (b.password !== "Correct-Horse-9") return { status: 401, body: { detail: "invalid_credentials" } };
      loggedIn = true;
      return { body: { user: user(role), csrf_token: "t", expires_at: "2026-10-08T00:00:00Z" } };
    },
    "POST /api/v1/auth/logout": () => {
      loggedIn = false;
      return { status: 204 };
    },
    "GET /api/v1/system/release": () => ({ body: { mode: "ASSIST_PILOT", gate_passed: false, message: "", reason: "" } }),
    "GET /api/v1/live/status": () => ({ body: { enabled: true, cameras: [], not_started: {} } }),
    "GET /api/v1/alerts": () => ({ body: [makeAlert({ id: 3 })] }),
  });
}

describe("App", () => {
  beforeEach(() => {
    resetStore();
    vi.stubGlobal("WebSocket", QuietSocket);
    const node = () => ({ connect: (n: unknown) => n, start: () => undefined, stop: () => undefined, frequency: { value: 0 }, gain: { value: 0 }, type: "" });
    vi.stubGlobal(
      "AudioContext",
      class {
        state = "running";
        destination = {};
        resume() {
          return Promise.resolve();
        }
        createOscillator = node;
        createGain = node;
      },
    );
    window.history.pushState({}, "", "/");
  });

  it("logs in, requires arming audio, then shows the alarms (no ASSIST banner)", async () => {
    backend();
    render(<App client={newClient()} />);
    expect(await screen.findByRole("form", { name: "Log in" })).toBeInTheDocument();
    expect(screen.queryByText(/ASSIST \/ PILOT/)).toBeNull();
    fireEvent.change(screen.getByLabelText("Username"), { target: { value: "pat" } });
    fireEvent.change(screen.getByLabelText("Password"), { target: { value: "wrong" } });
    fireEvent.click(screen.getByRole("button", { name: "Log in" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("invalid_credentials");
    fireEvent.change(screen.getByLabelText("Password"), { target: { value: "Correct-Horse-9" } });
    fireEvent.click(screen.getByRole("button", { name: "Log in" }));
    const arm = await screen.findByRole("button", { name: /Arm audio/ });
    expect(screen.queryByText(/Active alarms/)).toBeNull(); // nothing before audio is armed
    fireEvent.click(arm);
    expect(await screen.findByTestId("alarm-3")).toBeInTheDocument();
    expect(screen.queryByText(/ASSIST \/ PILOT/)).toBeNull();
    expect(QuietSocket.last?.url).toBe("wss://lifeguard.test/api/v1/ws");
  });

  it("shows only the screens the role may use", async () => {
    backend("lifeguard");
    render(<App client={newClient()} />);
    fireEvent.change(await screen.findByLabelText("Username"), { target: { value: "pat" } });
    fireEvent.change(screen.getByLabelText("Password"), { target: { value: "Correct-Horse-9" } });
    fireEvent.click(screen.getByRole("button", { name: "Log in" }));
    fireEvent.click(await screen.findByRole("button", { name: /Arm audio/ }));
    await screen.findByRole("link", { name: "Alerts" });
    expect(screen.queryByRole("link", { name: "Users" })).toBeNull();
    expect(screen.queryByRole("link", { name: "Zones" })).toBeNull();
    expect(screen.queryByRole("link", { name: "Settings" })).toBeNull(); // supervisor and above
  });

  it("an old top-level path (a bookmark, an alert link) opens its page under Settings, keeping the query", async () => {
    backend("supervisor");
    window.history.pushState({}, "", "/zones?camera=c1");
    render(<App client={newClient()} />);
    fireEvent.change(await screen.findByLabelText("Username"), { target: { value: "pat" } });
    fireEvent.change(screen.getByLabelText("Password"), { target: { value: "Correct-Horse-9" } });
    fireEvent.click(screen.getByRole("button", { name: "Log in" }));
    fireEvent.click(await screen.findByRole("button", { name: /Arm audio/ }));
    await screen.findByRole("group", { name: "Settings sections" });
    expect(window.location.pathname).toBe("/settings/zones");
    expect(window.location.search).toBe("?camera=c1");
  });

  it("the side nav has exactly Dashboard, Cameras, Alerts and Settings", async () => {
    backend("admin");
    render(<App client={newClient()} />);
    fireEvent.change(await screen.findByLabelText("Username"), { target: { value: "pat" } });
    fireEvent.change(screen.getByLabelText("Password"), { target: { value: "Correct-Horse-9" } });
    fireEvent.click(screen.getByRole("button", { name: "Log in" }));
    fireEvent.click(await screen.findByRole("button", { name: /Arm audio/ }));
    const nav = await screen.findByRole("navigation");
    await waitFor(() => {
      expect(Array.from(nav.querySelectorAll("a")).map((a) => a.textContent.replace(/\d+$/, ""))).toEqual(["Dashboard", "Cameras", "Alerts", "Settings"]);
    });
  });

  it("Log out returns to the login screen at once; the next operator must arm audio again", async () => {
    // Phase 9 E2E found this: logout cleared the query cache BEFORE nulling ["me"], which orphaned the
    // AuthProvider's observer - the screen stayed signed in until something else re-rendered it.
    backend();
    render(<App client={newClient()} />);
    fireEvent.change(await screen.findByLabelText("Username"), { target: { value: "pat" } });
    fireEvent.change(screen.getByLabelText("Password"), { target: { value: "Correct-Horse-9" } });
    fireEvent.click(screen.getByRole("button", { name: "Log in" }));
    fireEvent.click(await screen.findByRole("button", { name: /Arm audio/ }));
    await screen.findByTestId("alarm-3");
    fireEvent.click(screen.getByRole("button", { name: "Log out" }));
    expect(await screen.findByRole("form", { name: "Log in" })).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("Username"), { target: { value: "pat" } });
    fireEvent.change(screen.getByLabelText("Password"), { target: { value: "Correct-Horse-9" } });
    fireEvent.click(screen.getByRole("button", { name: "Log in" }));
    expect(await screen.findByRole("button", { name: /Arm audio/ })).toBeInTheDocument();
  });

  it("goes back to the login screen when the feed reports the session ended", async () => {
    backend();
    render(<App client={newClient()} />);
    fireEvent.change(await screen.findByLabelText("Username"), { target: { value: "pat" } });
    fireEvent.change(screen.getByLabelText("Password"), { target: { value: "Correct-Horse-9" } });
    fireEvent.click(screen.getByRole("button", { name: "Log in" }));
    fireEvent.click(await screen.findByRole("button", { name: /Arm audio/ }));
    await screen.findByTestId("alarm-3");
    QuietSocket.last?.onclose?.({ code: 4401 });
    await waitFor(() => {
      expect(screen.getByRole("form", { name: "Log in" })).toBeInTheDocument();
    });
  });
});
