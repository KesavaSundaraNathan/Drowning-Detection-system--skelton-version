import { describe, expect, it } from "vitest";

import { alarmTitle, chooseTone, mostUrgent } from "./alarmModel";
import { makeAlert } from "./test/fixtures";

describe("tone choice", () => {
  const base = { armed: true, alarms: 0, faults: 0, notMonitoringOverlay: false };
  it("is silent until audio is armed", () => {
    expect(chooseTone({ ...base, armed: false, alarms: 3 })).toBeNull();
  });
  it("puts a drowning alarm above everything else", () => {
    expect(chooseTone({ ...base, alarms: 1, faults: 2, notMonitoringOverlay: true })).toBe("alarm");
  });
  it("sounds the fault tone for NOT MONITORING or an unacknowledged FAULT", () => {
    expect(chooseTone({ ...base, notMonitoringOverlay: true })).toBe("fault");
    expect(chooseTone({ ...base, faults: 1 })).toBe("fault");
    expect(chooseTone(base)).toBeNull();
  });
});

describe("Drowning Alert (Stage 4)", () => {
  const base = { armed: true, alarms: 1, faults: 0, notMonitoringOverlay: false };
  it("sounds the one alarm tone for every drowning alarm", () => {
    expect(chooseTone(base)).toBe("alarm");
    expect(chooseTone({ ...base, alarms: 0, faults: 1 })).toBe("fault");
  });
  it("names faults by code and every drowning alarm 'Drowning Alert' with its reason", () => {
    expect(alarmTitle(makeAlert({ scenario: "FAULT", fault_code: "CAMERA_STALLED" }))).toBe("FAULT CAMERA_STALLED");
    expect(alarmTitle(makeAlert({ alert_reason: "Active drowning" }))).toBe("Drowning Alert · Active drowning");
    expect(alarmTitle(makeAlert({ alert_reason: "Submersion / sudden blackout" }))).toBe(
      "Drowning Alert · Submersion / sudden blackout",
    );
    const older = makeAlert({});
    delete older.alert_reason; // an older API without the field: the generic reason
    expect(alarmTitle(older)).toBe("Drowning Alert · Drowning");
  });
  it("shows the highest-risk alarm group first, then the oldest", () => {
    const g = (id: number, risk: number) => ({ key: `k${String(id)}`, primary: makeAlert({ id }), alerts: [makeAlert({ id, risk_score: risk })] });
    expect(mostUrgent([g(1, 60), g(2, 90), g(3, 90)])?.primary.id).toBe(2);
    expect(mostUrgent([])).toBeUndefined();
  });
});
