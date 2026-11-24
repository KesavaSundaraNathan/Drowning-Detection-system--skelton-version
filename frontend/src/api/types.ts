/** Wire types - mirror backend/lifeguard/app/api/v1/schemas.py (and the live / system payloads). */

export type Role = "lifeguard" | "supervisor" | "admin";
export type Mode = "Low" | "Medium" | "High";
export type AlertStatus = "pending" | "acknowledged" | "resolved" | "false_alarm";
export type ZoneType = "WATER" | "DECK" | "SHALLOW";
export type Weekday = "Mon" | "Tue" | "Wed" | "Thu" | "Fri" | "Sat" | "Sun";

export interface User {
  id: number;
  username: string;
  role: Role;
  is_active: boolean;
  created_at: string;
}

export interface LoginResult {
  user: User;
  csrf_token: string;
  expires_at: string;
}

export interface Alert {
  id: number;
  created_at: string;
  camera_id: string | null;
  track_id: number | null;
  scenario: string; // [Stage 4] "DROWNING" for every drowning alert, "FAULT" for a technical alarm
  title?: string; // [Stage 4] "Drowning Alert" (a FAULT: its fault code)
  alert_reason?: string | null; // [Stage 4] "Submersion / sudden blackout" | "Active drowning" | "Drowning"; null for a FAULT
  priority: string; // the risk band stored at first fire (sorting, overlays); not the alarm's urgency
  severity?: "critical" | "high" | "caution" | null; // [Phase 6] critical for every drowning alert (null for a FAULT)
  risk_score: number;
  confidence: number;
  zone_name: string | null;
  reason: string;
  status: AlertStatus;
  acknowledged_by: number | null;
  acknowledged_at: string | null;
  closed_by: number | null;
  closed_at: string | null;
  close_note: string | null;
  audible: boolean;
  fault_code: string | null;
  fault_source: string | null;
  cleared_at: string | null;
  escalation_level: number;
  escalated_at: string | null;
  global_id: number | null; // [Phase 11, removed in Phase 5] always null for new alerts
  incident_id: number | null; // [Phase 11, removed in Phase 5] always null for new alerts
  event_id?: number | null; // [Phase 5] the person's event on this camera (backend alerts/events.py)
  snapshot_available: boolean;
  recording_available: boolean;
}

/** [Phase 5] One person's incident on ONE camera: the alert fires of one track until 30 s quiet. */
export type EventSeverity = "critical" | "high" | "caution";
export interface AlertEvent {
  id: number;
  camera_id: string | null;
  track_id: number | null;
  opened_at: string;
  updated_at: string;
  closed_at: string | null;
  severity: EventSeverity;
  scenario: string; // [Stage 4] "DROWNING"
  title?: string; // [Stage 4] "Drowning Alert"
  alert_reason?: string | null; // [Stage 4] the reason of the event's alerts (see Alert.alert_reason)
  status: "open" | "acknowledged" | "resolved";
  alert_count: number;
  first_alert_id: number | null;
  last_alert_id: number | null;
  acknowledged_by: number | null;
  acknowledged_at: string | null;
  resolved_by: number | null;
  resolved_at: string | null;
}

/** [v4] "pool": drowning detection; "standard_cctv": person presence only, no drowning alerts. */
export type SceneType = "pool" | "standard_cctv";

export interface Camera {
  id: string;
  name: string;
  source_type: "rtsp" | "file";
  default_mode: Mode;
  is_enabled: boolean;
  scene_type: SceneType;
  scene_type_set_by: "default" | "admin";
  created_at: string;
  updated_at: string;
}

export interface Zone {
  id: number;
  camera_id: string | null; // null: a pool zone [Phase 11 WP2]
  name: string;
  zone_type: ZoneType;
  polygon: [number, number][] | null; // camera zone: normalised image coordinates
  pool_id?: string | null; // pool zone [Phase 11 WP2]
  pool_polygon_m?: [number, number][] | null; // pool zone: metres on the pool plan
  sensitivity: number;
  schedule_id: number | null;
  is_active: boolean;
}

/** Pool calibration of one camera (Phase 7): image corners <-> the shared pool plan in metres. */
export interface CalibrationData {
  image_points: [number, number][];
  pool_points: [number, number][];
  outline_m?: [number, number][];
  edge_band_m?: number;
  pool_id?: string; // [Phase 11] which pool (default "pool")
  error?: string; // a stored calibration that no longer validates
}

/** [Phase 11 WP2] Where a pool zone lands in one calibrated camera's view (normalised; null = not in view). */
export interface PoolZoneProjection {
  camera_id: string;
  camera_name: string;
  polygon: [number, number][] | null;
}

export interface Calibration {
  camera_id: string;
  calibration: CalibrationData | null;
  points: number | null;
  reprojection_error_m: number | null;
  pool_size_m: [number, number] | null;
  water_outline_norm: [number, number][] | null;
}

export interface Schedule {
  id: number;
  name: string;
  days: Weekday[];
  start_time: string;
  end_time: string;
  monitoring_mode: Mode;
  status: "Active" | "Disabled";
}

export interface AuditEvent {
  id: number;
  at: string;
  actor_username: string;
  ip: string;
  action: string;
  target_type: string;
  target_id: string;
  outcome: string;
  details: Record<string, unknown>;
}

export type Monitoring = "MONITORED" | "NOT_MONITORED";

export interface ZoneState {
  zone: string;
  zone_type: string;
  state: "MONITORED" | "MUTED";
  schedule_id: string | null;
}

export interface LiveCamera {
  camera_id: string;
  name: string;
  monitoring: Monitoring;
  stalled: boolean;
  lagging: boolean;
  lag_s: number;
  analysed_fraction: number | null;
  zones: ZoneState[];
  capture: { fps: number; hw_accel: number; frames_read: number; reconnects: number; last_frame_age_s: number };
  analysis: { analysed: number; proc_ms: number };
  scene?: {
    type: SceneType;
    /** automatic suggestion - never applied by the server (vision/scene/scene_type.py) */
    suggestion: { suggested: SceneType | null; reason: string; water_frac: number; observed_s: number } | null;
    people: number;
  };
}

export interface LiveStatus {
  enabled: boolean;
  running?: boolean;
  cameras?: LiveCamera[]; // always sent since Phase 7; optional so the UI never assumes it
  not_started?: Record<string, string>;
  supervisor_down?: boolean;
  capacity?: { limit: number; target_fps: number; batch_ms: Record<string, number>; overhead_ms: number } | null;
  scheduler?: { mean_batch: number; rounds: number; stalled: boolean; last_round_ms: number } | null;
  /** configured inference provider (settings.inference_provider), e.g. "cuda" */
  provider?: string;
  /** [v4] the provider actually running (provider is the requested one, e.g. "auto") */
  active_provider?: string | null;
  /** detector model file name */
  model?: string;
  /** runtime start error (LiveRuntime.start_error); null when it started */
  error?: string | null;
  any_lagging?: boolean;
}

/** live.summary event (WebSocket) - the compact form of LiveStatus. */
export interface LiveSummaryCamera {
  camera_id: string;
  name: string;
  monitoring: Monitoring;
  stalled: boolean;
  lagging: boolean;
  muted_zones: string[];
  scene_type?: SceneType; // [v4]
}

export interface ChannelStatus {
  pending: number;
  oldest_pending_s: number | null;
  consecutive_failures: number;
  down: boolean;
  sent: number;
  last_success_at: string | null;
  last_error: string | null;
}

export interface DeliveryStatus {
  channels: Record<string, ChannelStatus>;
  dispatcher_alive: boolean;
  escalator_alive: boolean;
  escalations: number;
  oldest_pending_s: number | null;
  channels_down: string[];
  escalation_tiers: number;
  self_test: { enabled: boolean; time: string; last: { event_id: string; started: string; results: Record<string, string | null> } | null };
}

export interface GateCriterion {
  key: string;
  target: string;
  value: unknown;
  status: "MET" | "NOT MET" | "NOT MEASURED";
  note?: string;
}

/** GET /system/release (Phase 8: the signed release gate, lifeguard.eval.release_gate). */
export interface ReleaseStatus {
  mode: "ASSIST_PILOT" | "GATE_PASSED";
  gate_passed: boolean;
  message: string;
  reason: string;
  site?: string | null;
  created_at?: string | null;
  criteria?: GateCriterion[];
}

export interface CapacityError {
  code: "capacity_exceeded";
  limit: number;
  requested: number;
  analysed_fps: number;
  target_fps: number;
  /** [v4] set when the refusal is MEASURED: this running camera is already falling behind */
  behind?: string | null;
}
