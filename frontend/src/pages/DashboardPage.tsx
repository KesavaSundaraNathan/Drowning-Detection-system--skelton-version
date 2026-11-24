import { useCallback, useState } from "react";

import { AlarmBell, AlarmPanel, AlarmStack } from "../components/alarms";
import { CameraTile } from "../components/CameraTile";
import { SectionHeader } from "../components/primitives/SectionHeader";
import { useNow } from "../hooks";
import { useLive } from "../state/liveStore";
import { verdict, WS_GRACE_MS } from "../state/monitoring";

export default function DashboardPage() {
  const now = useNow(1000);
  const state = useLive((s) => s);
  const v = verdict(state, now);
  const feedDown = state.ws === "down" && state.wsDownSince !== null && now - state.wsDownSince > WS_GRACE_MS;
  // the full alarm list lives under the alarm clock (top right); the page itself shows only the alarm stack
  const [listOpen, setListOpen] = useState(false);
  const closeList = useCallback(() => {
    setListOpen(false);
  }, []);
  return (
    <div className="dashboard">
      <SectionHeader
        eyebrow="Live monitoring"
        title="Dashboard"
        subtitle="Every camera, live."
        className="mb-2"
        actions={
          <div className="alarm-bell-anchor">
            <AlarmBell open={listOpen} onToggle={() => { setListOpen((o) => !o); }} />
            {listOpen && <AlarmPanel onClose={closeList} />}
          </div>
        }
      />
      {!listOpen && <AlarmStack onOpenList={() => { setListOpen(true); }} />}
      <section aria-label="Cameras" className="dash-cams">
        {state.cameras === null && <p>Connecting to the live feed...</p>}
        {state.cameras?.length === 0 && <p>No camera is running.</p>}
        {state.cameras?.map((c) => (
          <CameraTile key={c.camera_id} camera={c} notMonitored={v.cameras.has(c.camera_id)} feedDown={feedDown} />
        ))}
      </section>
    </div>
  );
}
