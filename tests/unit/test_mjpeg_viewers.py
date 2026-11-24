"""[rtmpose clone] MJPEG viewer slots are counted by heartbeat, so an abandoned stream cannot leak a slot.

Before: the slot was freed in the generator's `finally`; a disconnect that left the generator suspended at `yield`
never ran it, the counter crept up to mjpeg_max_streams and every live tile got 503 too_many_viewers (black)."""

from __future__ import annotations

import time

from lifeguard.app.api.v1 import live as live_api


def test_abandoned_stream_frees_its_slot_after_the_stale_window(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(live_api, "_viewers", {})
    now = time.monotonic()
    live_api._viewers[1] = (now, False)  # an active viewer (sent a frame just now)
    live_api._viewers[2] = (now - live_api.VIEWER_STALE_S - 1, True)  # abandoned at `yield`: no frame since
    assert live_api._open_viewers() == 1
    assert live_api._open_viewers(hq_only=True) == 0
    assert 2 not in live_api._viewers  # dropped, not just ignored


def test_many_abandoned_streams_never_lock_out_new_viewers(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(live_api, "_viewers", {})
    stale = time.monotonic() - live_api.VIEWER_STALE_S - 1
    for token in range(100):
        live_api._viewers[token] = (stale, False)
    assert live_api._open_viewers() == 0
