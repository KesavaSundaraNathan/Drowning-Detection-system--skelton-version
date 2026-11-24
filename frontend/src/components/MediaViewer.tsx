import * as Dialog from "@radix-ui/react-dialog";

import { media } from "../api/endpoints";
import type { Alert } from "../api/types";
import { alarmTitle } from "../alarmModel";

/**
 * Snapshot + recording viewer. Recordings are WebM / VP8 (Phase 6 decision: royalty-free, plays in
 * every current browser's <video>, written by the FFmpeg bundled with OpenCV). Files are fetched
 * same-origin, so the session cookie authorises them; nothing is cached (Cache-Control: no-store).
 * A Phase 5 MPEG-4 Part 2 recording cannot play in a browser: the server serves it as a download.
 */
export function MediaViewer({ alert, onClose }: { alert: Alert; onClose: () => void }) {
  return (
    <Dialog.Root
      open
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
    >
      <Dialog.Portal>
        <Dialog.Overlay className="dialog-overlay" />
        <Dialog.Content className="dialog media-viewer" aria-describedby={undefined}>
          <Dialog.Title>
            Alert {alert.id}: {alarmTitle(alert)}
          </Dialog.Title>
          <div className="media-grid">
            {alert.snapshot_available ? (
              <figure>
                <img src={media.snapshot(alert.id)} alt={`Alert ${String(alert.id)} snapshot`} />
                <figcaption>Frame the alert fired on (box = the person)</figcaption>
              </figure>
            ) : (
              <p>No snapshot.</p>
            )}
            {alert.recording_available ? (
              <figure>
                <video
                  controls
                  preload="metadata"
                  src={media.recording(alert.id)}
                  data-testid="recording"
                  aria-label={`Alert ${String(alert.id)} recording`}
                >
                  <a href={media.recording(alert.id)}>Download the recording</a>
                </video>
                <figcaption>
                  About 5 s before to 5 s after the alarm.{" "}
                  <a href={media.recording(alert.id)} download>
                    Download
                  </a>
                </figcaption>
              </figure>
            ) : (
              <p>No recording (only live alarms are recorded).</p>
            )}
          </div>
          <Dialog.Close asChild>
            <button type="button">Close</button>
          </Dialog.Close>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
