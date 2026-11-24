import { useQuery } from "@tanstack/react-query";
import { Link, useParams } from "react-router-dom";

import { api } from "../api/endpoints";
import { CameraTile } from "../components/CameraTile";
import { useNow } from "../hooks";
import { useLive } from "../state/liveStore";
import { verdict, WS_GRACE_MS } from "../state/monitoring";

/** Camera detail / Model View: the annotated live view large, plus its capture + analysis figures. */
export default function CameraDetailPage() {
  const { cameraId = "" } = useParams();
  const now = useNow(1000);
  const state = useLive((s) => s);
  const v = verdict(state, now);
  const feedDown = state.ws === "down" && state.wsDownSince !== null && now - state.wsDownSince > WS_GRACE_MS;
  const status = useQuery({ queryKey: ["live", "status"], queryFn: api.liveStatus, refetchInterval: 5000 });
  const cam = state.cameras?.find((c) => c.camera_id === cameraId);
  const detail = status.data?.cameras?.find((c) => c.camera_id === cameraId);
  if (!cam) {
    return (
      <p>
        Camera {cameraId} is not running. <Link to="/cameras">Back to cameras</Link>
      </p>
    );
  }
  return (
    <div>
      <h1>{cam.name}</h1>
      <CameraTile camera={cam} notMonitored={v.cameras.has(cam.camera_id)} feedDown={feedDown} large />
      {detail && (
        <dl className="facts">
          <dt>Source fps</dt>
          <dd>{detail.capture.fps}</dd>
          <dt>Hardware decoding</dt>
          <dd>{detail.capture.hw_accel ? "yes" : "no (software)"}</dd>
          <dt>Frames analysed</dt>
          <dd>
            {detail.analysis.analysed} ({detail.analysed_fraction === null ? "-" : `${(detail.analysed_fraction * 100).toFixed(1)} %`})
          </dd>
          <dt>Processing</dt>
          <dd>{detail.analysis.proc_ms} ms / frame</dd>
          <dt>Reconnects</dt>
          <dd>{detail.capture.reconnects}</dd>
          <dt>Zones</dt>
          <dd>{detail.zones.length === 0 ? "none (whole frame = water)" : detail.zones.map((z) => `${z.zone}: ${z.state}`).join(", ")}</dd>
        </dl>
      )}
      <Link to={`/settings/zones?camera=${encodeURIComponent(cam.camera_id)}`}>Edit zones</Link>
    </div>
  );
}
