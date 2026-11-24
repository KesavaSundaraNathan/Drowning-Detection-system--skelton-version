"""Server configuration (pydantic-settings), validated at load time.

Environment variables use the prefix ``LIFEGUARD_`` (e.g. ``LIFEGUARD_ENV=demo``). An env file
can be named with ``LIFEGUARD_ENV_FILE``; the Windows installer (Phase 8) writes one under the
data directory with an ACL restricted to the service account. An old ``AQUAGUARD_*`` variable
still works (``core.legacy``, imported below for its side effect of copying it to ``LIFEGUARD_*``
before this module's ``Settings`` class reads the environment).

Defaults are the production defaults: ``env=production``, ``debug=False``, Secure cookies, no CORS.
Insecure combinations are refused here (not warned about), and the server additionally refuses
to start when the session secret or the credential-encryption key is missing (core.security.keys).
"""

from __future__ import annotations

import os
from functools import cached_property
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from lifeguard.core import legacy  # also imported for its own side effect: AQUAGUARD_* -> LIFEGUARD_* in os.environ

Env = Literal["production", "demo", "development", "test"]
_DEV_ENVS = ("development", "test")
# Settings of removed features (RTSP-only rebuild: uploads, demo-profile file cameras). An installed env file may
# still name them: they are dropped before validation (extra="forbid" would otherwise refuse to start) and listed
# once at startup.
_REMOVED_SETTINGS = frozenset(
    {
        "upload_max_mb",
        "upload_quota_mb",
        "upload_validate_frames",
        "upload_probe_timeout_s",
        "upload_extensions",
        "upload_detector_model",
        "upload_worker_expected",
        "upload_job_max_attempts",
        "upload_worker_poll_s",
        "media_dir",
    }
)
IGNORED_SETTINGS: list[str] = []  # removed settings found in this process's configuration (logged at startup)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LIFEGUARD_",
        env_file=os.environ.get("LIFEGUARD_ENV_FILE") or None,
        env_file_encoding="utf-8",
        extra="forbid",
        frozen=True,
    )

    # ── runtime profile ───────────────────────────────────────────────────────
    env: Env = "production"
    debug: bool = False

    # ── storage ───────────────────────────────────────────────────────────────
    # the new default, unless only an existing install's old AquaGuard data directory is there (core.legacy)
    data_dir: Path = Field(default_factory=legacy.default_data_dir)
    database_url: str | None = None  # default: SQLite under data_dir/db (PostgreSQL: config switch)

    # ── secrets (production: DPAPI-protected files under data_dir/keys, see core.security.keys) ──
    session_secret: SecretStr | None = None  # HMAC key for session/CSRF token hashes (>= 32 chars)
    credential_key: SecretStr | None = None  # Fernet key; env-var form allowed ONLY in development/test

    # ── HTTP ──────────────────────────────────────────────────────────────────
    cookie_secure: bool = True
    cors_origins: list[str] = Field(default_factory=list)  # exact origins; empty = same-origin only

    # ── sessions / login ──────────────────────────────────────────────────────
    session_absolute_hours: float = Field(12.0, gt=0, le=24)
    session_idle_minutes: float = Field(30.0, gt=0, le=240)
    login_max_failures: int = Field(5, ge=1, le=20)
    login_lockout_minutes: float = Field(15.0, ge=1, le=24 * 60)
    password_min_length: int = Field(12, ge=12, le=64)
    bcrypt_rounds: int = Field(12, ge=4, le=15)  # < 10 only allowed in env=test

    # ── retention / backup ────────────────────────────────────────────────────
    retention_days: int = Field(30, ge=1, le=3650)
    backup_keep: int = Field(14, ge=1, le=365)

    # ── vision (Phase 3) ──────────────────────────────────────────────────────
    detector_model: Path | None = None  # default: backend/models/<COCO-only RTMDet-m> (lifeguard.vision.config)
    # top-down RTMPose model; None = backend/models/<RTMPose-m COCO>, "none" = no pose (shown in health + audit)
    pose_model: str | None = None
    # [v4] pool-vs-other scene model (keyframes only, a suggestion): None = backend/models/clip_vit_b32_scene.onnx if
    # exported (tools/export_scene_clip.py), "none" = colour-only suggestion
    scene_model: str | None = None
    # "auto" = TensorRT (FP16, if installed), then CUDA, then OpenVINO, then CPU; the provider in use is logged,
    # audited and shown in health, and live admission still refuses cameras the device cannot analyse at
    # target_analysis_fps (no silent drops). An explicit provider is strict: anything else refuses to start.
    inference_provider: Literal["auto", "tensorrt", "cuda", "openvino", "cpu"] = "auto"
    # [Phase 6] TensorRT engine + timing cache; None = <data_dir>/trt_cache (built at install by
    # tools/build_trt_engines.py, reused at every start). Tests point it at a scratch copy of prebuilt engines.
    trt_cache_dir: Path | None = None
    # Load the detector on `inference_provider` at startup; a failure records a FAULT alert and the
    # server refuses to start. Default: on in production/demo, off in development/test. Turning it off
    # in production is an explicit choice that is written to the audit log at every startup.
    vision_startup_check: bool | None = None

    # ── live runtime (Phase 4) ────────────────────────────────────────────────
    # Start the live runtime (capture + batched GPU scheduler) with the API. Default: on in
    # production/demo, off in development/test.
    live_enabled: bool | None = None
    stall_s: float = Field(5.0, gt=0, le=120)  # last-frame age that makes a camera STALLED (FAULT + reconnect)
    # [Phase 11 WP4] after each (re)open a camera has this long to deliver its first frame before the stall monitor
    # counts it (FFmpeg waits for a keyframe: a long GOP must not cause a reconnect loop). Also the open timeout and
    # the read timeout (OpenCV cannot shorten a read timeout after the open). docs/CAMERA_SETUP.md
    first_frame_timeout_s: float = Field(12.0, gt=0, le=120)
    # [Phase 11 WP5] the cameras' configured fps, used when a stream reports an implausible fps (outside 1..240,
    # e.g. 90000 = the RTP clock of a stream without frame-rate timing). Was a fixed 25. docs/CAMERA_SETUP.md
    camera_fps_fallback: float = Field(15.0, ge=1, le=240)
    # [Phase 11 WP7] open the RTSP cameras in parallel with the provider guard + capacity phase (live/preopen.py);
    # frames are discarded until admission. Default decided by the gates in eval/site/levers_phase11.json.
    camera_preopen: bool = True
    preopen_max_hold_s: float = Field(60.0, gt=0, le=600)
    # [Phase 11 WP5] capacity cache <data_dir>/capacity_cache.json (live/capacity_cache.py): a restart re-uses the
    # measured GPU batch latencies after a batch-1 spot check; off = always measure
    capacity_cache: bool = True
    capacity_cache_max_age_h: float = Field(168.0, gt=0, le=24 * 365)
    # admission: every camera must be analysed this fast (or at its real fps if higher). [Phase 11 WP4] 15 = the
    # site cameras' fps (docs/CAMERA_SETUP.md); was 25. [v7] 18 refused the 4th camera (skewed-load check 18.95 fps
    # < 18/0.9). Owner directive 2026-11-17: 4 cameras live, 15 fps is the lowest allowed per camera; also the cap a
    # faster stream is decimated to. The site NVR streams 15 fps (measured 2026-10-21).
    target_analysis_fps: float = Field(15.0, gt=0, le=120)
    # [dynamic fps, owner 2026-11-18] ceiling of the live analysis rate. target_analysis_fps stays the FLOOR (what
    # admission, capacity and CAMERA_LAGGING count); live/rate.py raises every RTSP camera towards this ceiling while
    # the GPU is lightly loaded and every camera keeps up, and drops back towards the floor at once when they do not.
    # <= target_analysis_fps = fixed rate (the old behaviour).
    max_analysis_fps: float = Field(20.0, gt=0, le=120)
    # at start-up only the first N enabled cameras (by name) are started; the rest show NOT STARTED and an operator
    # starts them with "Start monitoring" (admission still applies). 0 = start every enabled camera.
    startup_max_cameras: int = Field(4, ge=0, le=64)
    batch_max: int = Field(8, ge=1, le=64)  # frames per GPU round
    batch_gather_ms: float = Field(10.0, ge=0, le=200)  # wait for more cameras after the first frame of a round
    live_queue_s: float = Field(2.0, gt=0, le=30)  # per-camera frame queue (seconds of video); full -> drop + lag
    lag_threshold_s: float = Field(1.0, gt=0, le=30)  # oldest queued frame older than this = camera LAGGING
    scheduler_stall_s: float = Field(10.0, gt=0, le=120)  # a request pending this long = GPU scheduler STALLED
    capacity_overhead_ms: float | None = Field(None, ge=0, le=1000)  # per-frame CPU cost; None = measured
    # [Phase 11] the cameras' frame size for admission (DET_TILING tiles are counted at it): 4 MP bullets
    capacity_frame_size: tuple[int, int] = (2560, 1440)
    heartbeat_interval_s: float = Field(1.0, gt=0, le=30)

    # ── supervisor service + mutual watchdog (Phase 4) ────────────────────────
    # The API raises a FAULT when the supervisor's heartbeat is older than supervisor_timeout_s.
    # Default: expected in production/demo (it is installed as its own service), not in development/test.
    supervisor_expected: bool | None = None
    supervisor_timeout_s: float = Field(10.0, gt=0, le=600)  # + 1 s check period: inside STALL_S + 10 s
    supervisor_poll_s: float = Field(2.0, gt=0, le=60)
    # services start in any order: API_DOWN is not raised in the supervisor's first
    # seconds (camera, scheduler, disk and backup checks are never delayed)
    supervisor_startup_grace_s: float = Field(45.0, ge=0, le=600)
    # where the supervisor polls /api/v1/health/ready; [rtmpose clone] its own API port (the original uses 8000)
    supervisor_api_url: str = "http://127.0.0.1:8100"
    supervisor_renotify_s: float = Field(900.0, ge=1, le=86_400)  # repeat an unresolved SYSTEM FAULT
    disk_min_free_gb: float = Field(5.0, ge=0)
    backup_max_age_h: float = Field(26.0, gt=0, le=24 * 30)  # nightly backup + slack

    # ── out-of-band alert channels (Phase 4: supervisor SYSTEM FAULT; Phase 5: every alert) ──
    webhook_url: str | None = None
    webhook_secret: SecretStr | None = None  # HMAC-SHA256 key (>= 32 chars)
    ntfy_url: str | None = None  # e.g. http://127.0.0.1:2586 (self-hosted ntfy)
    ntfy_topic: str | None = None
    ntfy_token: SecretStr | None = None
    smtp_host: str | None = None
    smtp_port: int = Field(587, ge=1, le=65535)
    smtp_starttls: bool = True  # False only allowed in development/test
    smtp_ca_file: Path | None = None  # CA bundle for an internal relay's certificate (verification stays ON)
    smtp_username: str | None = None
    smtp_password: SecretStr | None = None
    smtp_from: str | None = None
    smtp_to: list[str] = Field(default_factory=list)
    channel_timeout_s: float = Field(10.0, gt=0, le=60)

    # ── delivery: outbox, escalation, self-test (Phase 5) ─────────────────────
    outbox_poll_s: float = Field(0.5, gt=0, le=10)
    outbox_retry_base_s: float = Field(2.0, gt=0, le=60)  # 2, 4, 8 ... s (+-20 % jitter)
    outbox_retry_max_s: float = Field(60.0, gt=0, le=3600)
    outbox_max_age_h: float = Field(24.0, gt=0, le=24 * 30)  # older undelivered rows -> "dead" + FAULT
    channel_down_after: int = Field(3, ge=1, le=100)  # consecutive failures -> CHANNEL_DOWN FAULT
    # An unacknowledged audible alarm (and FAULT, unless escalate_faults=false) goes to the next contact
    # tier on every channel after escalate_after_s; the last tier is repeated every escalation_repeat_s.
    escalate_after_s: float = Field(15.0, gt=0, le=3600)
    escalation_repeat_s: float = Field(120.0, gt=0, le=86_400)
    escalate_faults: bool = True
    # [Phase 11] cross-camera incident window. UNUSED since Phase 5 (cameras are independent); kept so existing env
    # files that set LIFEGUARD_INCIDENT_WINDOW_S still load
    incident_window_s: float = Field(120.0, gt=0, le=3600)
    escalation_ntfy_topics: list[str] = Field(default_factory=list)  # tier 1, 2, ... (tier 0 = ntfy_topic)
    escalation_smtp_to: list[list[str]] = Field(default_factory=list)  # tier 1, 2, ... (tier 0 = smtp_to)
    # Daily test message on every channel at self_test_time (site local HH:MM); a channel without a
    # delivery receipt within self_test_timeout_s raises CHANNEL_SELF_TEST_FAILED. Default: on in
    # production/demo, off in development/test.
    self_test_enabled: bool | None = None
    self_test_time: str = "07:00"
    self_test_timeout_s: float = Field(120.0, gt=0, le=3600)

    # ── alert media (Phase 5) ─────────────────────────────────────────────────
    recording_enabled: bool = True
    recording_pre_s: float = Field(5.0, ge=0, le=30)
    recording_post_s: float = Field(5.0, ge=0, le=60)
    recording_fps: float = Field(10.0, gt=0, le=30)
    recording_width: int = Field(640, ge=160, le=1920)

    # ── operator live view (Phase 6) ──────────────────────────────────────────
    mjpeg_fps: float = Field(5.0, gt=0, le=25)
    mjpeg_width: int = Field(960, ge=320, le=1920)
    # [v4] ?quality=high (the Cameras page's main view): sharper + smoother, for one stream at a time
    mjpeg_hq_width: int = Field(1920, ge=320, le=3840)
    mjpeg_hq_fps: float = Field(10.0, gt=0, le=25)
    # High-quality streams at once, server-wide; beyond it viewers get the normal stream. Measured 2026-10-15: rapid
    # switching left several 1920 px streams open and every camera's ANALYSIS fell behind (6-13 fps, CAMERA_LAGGING)
    mjpeg_hq_max_streams: int = Field(2, ge=0, le=16)
    # frame.jpg?quality=thumb: the Cameras page side previews (short requests, no long-lived connection)
    mjpeg_thumb_width: int = Field(640, ge=160, le=1920)
    mjpeg_max_streams: int = Field(16, ge=1, le=256)  # concurrent viewers per server (CPU guard)

    # ── WebSocket (Phase 5) ───────────────────────────────────────────────────
    ws_queue_size: int = Field(256, ge=8, le=10_000)  # per client; overflow -> that client is dropped
    ws_send_timeout_s: float = Field(2.0, gt=0, le=30)  # a client slower than this is dropped
    ws_recheck_s: float = Field(30.0, gt=0, le=600)  # re-validate the session (revocation, expiry)
    # [v3] Phase 7: an application-level heartbeat. A dead server behind a proxy (or a half-open TCP
    # connection) never closes the browser's socket; the UI declares the feed down after 3 missed beats.
    ws_heartbeat_s: float = Field(2.0, gt=0, le=10)

    # ── Prometheus metrics (Phase 12, app/metrics.py) ─────────────────────────
    # A separate listener, never a route of the API (Caddy proxies /api/* only). Default: on in production/demo,
    # off in development/test (tests start many apps). A non-loopback bind needs metrics_token (Bearer).
    metrics_enabled: bool | None = None
    metrics_bind: str = "127.0.0.1:9464"
    metrics_token: SecretStr | None = None

    # ── logging ───────────────────────────────────────────────────────────────
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_max_bytes: int = Field(10 * 1024 * 1024, ge=64 * 1024)
    log_backups: int = Field(10, ge=1, le=100)

    # ── validation ────────────────────────────────────────────────────────────
    @model_validator(mode="before")
    @classmethod
    def _drop_removed_settings(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        out = {}
        for k, v in data.items():
            name = k.lower().removeprefix("lifeguard_") if isinstance(k, str) else k
            if name in _REMOVED_SETTINGS:
                if name not in IGNORED_SETTINGS:
                    IGNORED_SETTINGS.append(name)
                continue
            out[k] = v
        return out

    @field_validator("cors_origins")
    @classmethod
    def _exact_origins(cls, v: list[str]) -> list[str]:
        for origin in v:
            parts = urlsplit(origin)
            if origin.strip() == "*" or "*" in origin:
                raise ValueError("wildcard CORS origins are not allowed")
            if parts.scheme not in ("https", "http") or not parts.hostname:
                raise ValueError(f"CORS origin must be scheme://host[:port], got {origin!r}")
            if parts.path not in ("", "/") or parts.query or parts.fragment or parts.username:
                raise ValueError(f"CORS origin must be an exact origin without path/query: {origin!r}")
        return [o.rstrip("/") for o in v]

    @field_validator("database_url")
    @classmethod
    def _db_scheme(cls, v: str | None) -> str | None:
        if v is not None and not v.startswith(("sqlite:", "postgresql")):
            raise ValueError("database_url must be a sqlite: or postgresql URL")
        return v

    @model_validator(mode="after")
    def _profile_rules(self) -> Settings:
        prod_like = self.env not in _DEV_ENVS
        if prod_like and self.debug:
            raise ValueError(f"debug=True is not allowed in env={self.env}")
        if prod_like and not self.cookie_secure:
            raise ValueError(f"cookie_secure=False is not allowed in env={self.env}")
        if prod_like and self.credential_key is not None:
            raise ValueError(
                "LIFEGUARD_CREDENTIAL_KEY may only be used in development/test; "
                "production keys are DPAPI-protected files (run `lifeguard keys init`)"
            )
        if prod_like and any(urlsplit(o).scheme != "https" for o in self.cors_origins):
            raise ValueError("CORS origins must be https in production/demo")
        if self.bcrypt_rounds < 10 and self.env != "test":
            raise ValueError("bcrypt_rounds < 10 is only allowed in env=test")
        if self.session_secret is not None and len(self.session_secret.get_secret_value()) < 32:
            raise ValueError("session_secret must be at least 32 characters")
        self._channel_rules(prod_like)
        self._metrics_rules()
        return self

    def _metrics_rules(self) -> None:
        host, _ = self.metrics_address  # raises on a malformed bind
        token = self.metrics_token.get_secret_value() if self.metrics_token else ""
        if self.metrics_token is not None and len(token) < 32:
            raise ValueError("metrics_token must be at least 32 characters")
        if not _is_loopback(host) and not token:
            raise ValueError(
                f"metrics_bind {self.metrics_bind!r} is not a loopback address: set metrics_token (Bearer) to expose "
                "the metrics beyond this machine"
            )

    def _channel_rules(self, prod_like: bool) -> None:
        for name in ("webhook_url", "ntfy_url", "supervisor_api_url"):
            url = getattr(self, name)
            if url is None:
                continue
            parts = urlsplit(url)
            if parts.scheme not in ("http", "https") or not parts.hostname or parts.username or parts.password:
                raise ValueError(f"{name} must be http(s)://host[:port][/path] without credentials")
        if prod_like and self.webhook_url and urlsplit(self.webhook_url).scheme != "https":
            raise ValueError("webhook_url must be https in production/demo")
        if self.webhook_url and (self.webhook_secret is None or len(self.webhook_secret.get_secret_value()) < 32):
            raise ValueError("webhook_url needs webhook_secret (>= 32 characters) - deliveries are always signed")
        if bool(self.ntfy_url) != bool(self.ntfy_topic):
            raise ValueError("ntfy_url and ntfy_topic must be set together")
        if self.smtp_host and (not self.smtp_from or not self.smtp_to):
            raise ValueError("smtp_host needs smtp_from and at least one smtp_to address")
        if prod_like and self.smtp_host and not self.smtp_starttls:
            raise ValueError("smtp_starttls=False is only allowed in development/test")
        if bool(self.smtp_username) != bool(self.smtp_password):
            raise ValueError("smtp_username and smtp_password must be set together")
        from lifeguard.core.schedule import parse_hhmm

        parse_hhmm(self.self_test_time)
        if self.escalation_ntfy_topics and not self.ntfy_url:
            raise ValueError("escalation_ntfy_topics need ntfy_url")
        if self.escalation_smtp_to and not self.smtp_host:
            raise ValueError("escalation_smtp_to needs smtp_host")
        if any(not tier for tier in self.escalation_smtp_to):
            raise ValueError("every escalation_smtp_to tier needs at least one address")

    # ── derived paths ─────────────────────────────────────────────────────────
    @cached_property
    def db_dir(self) -> Path:
        return self.data_dir / "db"

    @cached_property
    def keys_dir(self) -> Path:
        return self.data_dir / "keys"

    @cached_property
    def log_dir(self) -> Path:
        return self.data_dir / "logs"

    @cached_property
    def snapshots_dir(self) -> Path:
        return self.data_dir / "snapshots"

    @cached_property
    def recordings_dir(self) -> Path:
        return self.data_dir / "recordings"

    @cached_property
    def backups_dir(self) -> Path:
        return self.data_dir / "backups"

    @property
    def trt_engine_dir(self) -> Path:
        return self.trt_cache_dir if self.trt_cache_dir is not None else self.data_dir / "trt_cache"

    @cached_property
    def sqlite_path(self) -> Path:
        # "lifeguard.db", unless only an existing install's "aquaguard.db" is there (core.legacy)
        return legacy.sqlite_file(self.db_dir)

    @property
    def vision_check_enabled(self) -> bool:
        if self.vision_startup_check is not None:
            return self.vision_startup_check
        return self.env not in _DEV_ENVS

    @property
    def live_runtime_enabled(self) -> bool:
        return self.live_enabled if self.live_enabled is not None else self.env not in _DEV_ENVS

    @property
    def supervisor_is_expected(self) -> bool:
        return self.supervisor_expected if self.supervisor_expected is not None else self.env not in _DEV_ENVS

    @property
    def metrics_is_enabled(self) -> bool:
        return self.metrics_enabled if self.metrics_enabled is not None else self.env not in _DEV_ENVS

    @property
    def metrics_address(self) -> tuple[str, int]:
        host, sep, port = self.metrics_bind.rpartition(":")
        host = host.strip("[]")
        if not sep or not host or not port.isdigit() or not 0 <= int(port) <= 65535:
            raise ValueError(f"metrics_bind must be host:port, got {self.metrics_bind!r}")
        return host, int(port)

    @property
    def self_test_is_enabled(self) -> bool:
        return self.self_test_enabled if self.self_test_enabled is not None else self.env not in _DEV_ENVS

    @property
    def escalation_tiers(self) -> int:
        """Number of contact tiers (tier 0 = the primary contacts)."""
        return 1 + max(len(self.escalation_ntfy_topics), len(self.escalation_smtp_to))

    def ntfy_topic_for(self, tier: int) -> str | None:
        if tier <= 0 or not self.escalation_ntfy_topics:
            return self.ntfy_topic
        return self.escalation_ntfy_topics[min(tier, len(self.escalation_ntfy_topics)) - 1]

    def smtp_to_for(self, tier: int) -> list[str]:
        if tier <= 0 or not self.escalation_smtp_to:
            return list(self.smtp_to)
        return list(self.escalation_smtp_to[min(tier, len(self.escalation_smtp_to)) - 1])

    @property
    def channels_configured(self) -> list[str]:
        out = []
        if self.webhook_url:
            out.append("webhook")
        if self.ntfy_url:
            out.append("ntfy")
        if self.smtp_host:
            out.append("smtp")
        return out

    @cached_property
    def run_dir(self) -> Path:
        return self.data_dir / "run"

    @cached_property
    def heartbeat_dir(self) -> Path:
        return self.run_dir / "heartbeats"

    @property
    def effective_database_url(self) -> str:
        return self.database_url or f"sqlite:///{self.sqlite_path.as_posix()}"

    @property
    def is_sqlite(self) -> bool:
        return self.effective_database_url.startswith("sqlite:")

    def ensure_dirs(self) -> None:
        for d in (
            self.db_dir,
            self.keys_dir,
            self.log_dir,
            self.snapshots_dir,
            self.recordings_dir,
            self.backups_dir,
            self.heartbeat_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


def _is_loopback(host: str) -> bool:
    import ipaddress

    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False  # a host name other than localhost may resolve anywhere


def load_settings(**overrides: object) -> Settings:
    """Settings from the environment (+ env file), with explicit overrides (tests, CLI)."""
    return Settings(**overrides)  # type: ignore[arg-type]
