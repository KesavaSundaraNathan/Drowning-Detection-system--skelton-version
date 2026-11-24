/**
 * Every backend call the UI makes. ENDPOINTS lists them as (method, path template) and is checked
 * against the backend's reviewed route table by tests/api/test_frontend_contract.py, so a renamed
 * or removed route breaks CI instead of the operator's screen.
 */
import { API, request } from "./client";
import type {
  Alert,
  AlertEvent,
  AlertStatus,
  AuditEvent,
  Calibration,
  Camera,
  DeliveryStatus,
  LiveStatus,
  LoginResult,
  Mode,
  PoolZoneProjection,
  ReleaseStatus,
  Role,
  SceneType,
  Schedule,
  User,
  Weekday,
  Zone,
  ZoneType,
} from "./types";

export const ENDPOINTS = [
  ["POST", "/auth/login"],
  ["POST", "/auth/logout"],
  ["GET", "/auth/me"],
  ["POST", "/auth/password"],
  ["GET", "/alerts"],
  ["GET", "/alerts/{alert_id}"],
  ["POST", "/alerts/{alert_id}/acknowledge"],
  ["POST", "/alerts/{alert_id}/resolve"],
  ["POST", "/alerts/{alert_id}/false-alarm"],
  ["POST", "/alerts/clear-all"],
  ["GET", "/alerts/{alert_id}/snapshot"],
  ["GET", "/alerts/{alert_id}/recording"],
  ["GET", "/events"],
  ["POST", "/events/{event_id}/acknowledge"],
  ["POST", "/events/{event_id}/resolve"],
  ["GET", "/cameras"],
  ["POST", "/cameras"],
  ["PATCH", "/cameras/{camera_id}"],
  ["DELETE", "/cameras/{camera_id}"],
  ["GET", "/live/status"],
  ["POST", "/live/cameras/{camera_id}/start"],
  ["POST", "/live/cameras/{camera_id}/stop"],
  ["GET", "/live/cameras/{camera_id}/mjpeg"],
  ["GET", "/live/cameras/{camera_id}/frame.jpg"],
  ["GET", "/cameras/{camera_id}/calibration"],
  ["PUT", "/cameras/{camera_id}/calibration"],
  ["DELETE", "/cameras/{camera_id}/calibration"],
  ["GET", "/cameras/{camera_id}/zones"],
  ["POST", "/cameras/{camera_id}/zones"],
  ["DELETE", "/zones/{zone_id}"],
  ["GET", "/pool-zones"],
  ["POST", "/pool-zones"],
  ["PUT", "/pool-zones/{zone_id}"],
  ["POST", "/pool-zones/preview"],
  ["GET", "/schedules"],
  ["POST", "/schedules"],
  ["DELETE", "/schedules/{schedule_id}"],
  ["GET", "/system/delivery"],
  ["POST", "/system/self-test"],
  ["GET", "/system/release"],
  ["GET", "/users"],
  ["POST", "/users"],
  ["PATCH", "/users/{user_id}"],
  ["POST", "/users/{user_id}/password"],
  ["GET", "/audit"],
] as const;

function query(q: Record<string, string | null>): string {
  const parts = Object.entries(q).flatMap(([k, v]) => (v === null ? [] : [`${k}=${v}`]));
  return parts.length ? `?${parts.join("&")}` : "";
}

export const media = {
  snapshot: (id: number) => `${API}/alerts/${String(id)}/snapshot`,
  recording: (id: number) => `${API}/alerts/${String(id)}/recording`,
  // [view modes] overlay "client": frames without boxes + each frame's boxes in X-Lifeguard-Boxes (the browser draws)
  // [rtmpose clone] skeleton=false: the server-drawn stream without the RTMPose skeleton (fallback view only)
  mjpeg: (cameraId: string, quality: "normal" | "high" = "normal", overlay: "burned" | "client" = "burned", skeleton = true) =>
    `${API}/live/cameras/${encodeURIComponent(cameraId)}/mjpeg${query({ quality: quality === "high" ? "high" : null, overlay: overlay === "client" ? "client" : null, skeleton: skeleton ? null : "false" })}`,
  frame: (cameraId: string, quality: "normal" | "thumb" = "normal", overlay: "burned" | "client" = "burned") =>
    `${API}/live/cameras/${encodeURIComponent(cameraId)}/frame.jpg${query({ quality: quality === "thumb" ? "thumb" : null, overlay: overlay === "client" ? "client" : null })}`,
  ws: () => `${window.location.protocol === "https:" ? "wss" : "ws"}://${window.location.host}${API}/ws`,
};

const enc = encodeURIComponent;

export interface PoolZoneBody {
  name: string;
  zone_type: ZoneType;
  pool_id: string;
  pool_polygon_m: [number, number][];
  sensitivity: number;
  schedule_id: number | null;
}

export const api = {
  login: (username: string, password: string) => request<LoginResult>("POST", "/auth/login", { body: { username, password } }),
  logout: () => request<undefined>("POST", "/auth/logout"),
  me: () => request<User>("GET", "/auth/me"),
  changePassword: (current_password: string, new_password: string) =>
    request<undefined>("POST", "/auth/password", { body: { current_password, new_password } }),

  alerts: (q: { status?: AlertStatus; audible?: boolean; limit?: number } = {}) =>
    request<Alert[]>("GET", "/alerts", { query: q }),
  alert: (id: number) => request<Alert>("GET", `/alerts/${String(id)}`),
  acknowledge: (id: number) => request<Alert>("POST", `/alerts/${String(id)}/acknowledge`),
  resolve: (id: number, note: string) => request<Alert>("POST", `/alerts/${String(id)}/resolve`, { body: { note } }),
  clearAllAlerts: (note: string) => request<{ cleared: number }>("POST", "/alerts/clear-all", { body: { note } }),
  falseAlarm: (id: number, note: string) => request<Alert>("POST", `/alerts/${String(id)}/false-alarm`, { body: { note } }),
  // [Phase 5] events: one per person and incident on one camera
  events: (q: { status?: "open" | "all"; limit?: number } = {}) => request<AlertEvent[]>("GET", "/events", { query: q }),
  acknowledgeEvent: (id: number) => request<AlertEvent>("POST", `/events/${String(id)}/acknowledge`),
  resolveEvent: (id: number, note: string) =>
    request<AlertEvent>("POST", `/events/${String(id)}/resolve`, { body: { note } }),

  cameras: () => request<Camera[]>("GET", "/cameras"),
  createCamera: (c: { name: string; source_url: string; default_mode: Mode }) =>
    request<Camera>("POST", "/cameras", { body: c }),
  patchCamera: (id: string, p: { scene_type?: SceneType }) => request<Camera>("PATCH", `/cameras/${enc(id)}`, { body: p }),
  deleteCamera: (id: string) => request<undefined>("DELETE", `/cameras/${enc(id)}`),
  liveStatus: () => request<LiveStatus>("GET", "/live/status"),
  startCamera: (id: string) => request<unknown>("POST", `/live/cameras/${enc(id)}/start`),
  stopCamera: (id: string) => request<unknown>("POST", `/live/cameras/${enc(id)}/stop`),

  calibration: (cameraId: string) => request<Calibration>("GET", `/cameras/${enc(cameraId)}/calibration`),
  saveCalibration: (
    cameraId: string,
    c: {
      image_points: [number, number][];
      pool_points: [number, number][];
      edge_band_m: number;
      frame_width: number;
      frame_height: number;
    },
  ) => request<Calibration>("PUT", `/cameras/${enc(cameraId)}/calibration`, { body: c }),
  deleteCalibration: (cameraId: string) => request<Calibration>("DELETE", `/cameras/${enc(cameraId)}/calibration`),

  zones: (cameraId: string) => request<Zone[]>("GET", `/cameras/${enc(cameraId)}/zones`),
  createZone: (
    cameraId: string,
    z: { name: string; zone_type: ZoneType; polygon: [number, number][]; sensitivity: number; schedule_id: number | null },
  ) => request<Zone>("POST", `/cameras/${enc(cameraId)}/zones`, { body: z }),
  deleteZone: (id: number) => request<undefined>("DELETE", `/zones/${String(id)}`),
  // [Phase 11 WP2] pool zones: drawn once on the pool plan (metres), projected into every calibrated camera
  poolZones: () => request<Zone[]>("GET", "/pool-zones"),
  createPoolZone: (z: PoolZoneBody) => request<Zone>("POST", "/pool-zones", { body: z }),
  updatePoolZone: (id: number, z: PoolZoneBody) => request<Zone>("PUT", `/pool-zones/${String(id)}`, { body: z }),
  previewPoolZone: (b: { pool_id: string; pool_polygon_m: [number, number][] }) =>
    request<PoolZoneProjection[]>("POST", "/pool-zones/preview", { body: b }),

  schedules: () => request<Schedule[]>("GET", "/schedules"),
  createSchedule: (s: { name: string; days: Weekday[]; start_time: string; end_time: string; monitoring_mode: Mode }) =>
    request<Schedule>("POST", "/schedules", { body: s }),
  deleteSchedule: (id: number) => request<undefined>("DELETE", `/schedules/${String(id)}`),

  delivery: () => request<DeliveryStatus>("GET", "/system/delivery"),
  selfTest: () => request<{ event_id: string }>("POST", "/system/self-test"),
  release: () => request<ReleaseStatus>("GET", "/system/release"),

  users: () => request<User[]>("GET", "/users"),
  createUser: (u: { username: string; role: Role; password: string }) => request<User>("POST", "/users", { body: u }),
  resetPassword: (id: number, new_password: string) =>
    request<undefined>("POST", `/users/${String(id)}/password`, { body: { new_password } }),
  patchUser: (id: number, p: { role?: Role; is_active?: boolean }) => request<User>("PATCH", `/users/${String(id)}`, { body: p }),
  audit: () => request<AuditEvent[]>("GET", "/audit"),
};
