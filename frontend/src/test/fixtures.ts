import type { Alert } from "../api/types";

export function makeAlert(over: Partial<Alert> = {}): Alert {
  return {
    id: 1,
    created_at: new Date(Date.now() - 30_000).toISOString(),
    camera_id: "cam-1",
    track_id: 4,
    scenario: "DROWNING",
    title: "Drowning Alert",
    alert_reason: "Active drowning",
    priority: "Critical",
    severity: "critical", // [Stage 4] every drowning alert is critical
    risk_score: 92,
    confidence: 0.9,
    zone_name: "Deep end",
    reason: "vertical, no forward progress",
    status: "pending",
    acknowledged_by: null,
    acknowledged_at: null,
    closed_by: null,
    closed_at: null,
    close_note: null,
    audible: true,
    fault_code: null,
    fault_source: null,
    cleared_at: null,
    escalation_level: 0,
    escalated_at: null,
    global_id: null,
    incident_id: null,
    snapshot_available: false,
    recording_available: false,
    ...over,
  };
}
