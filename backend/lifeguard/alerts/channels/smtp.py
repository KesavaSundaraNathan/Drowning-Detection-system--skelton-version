"""SMTP e-mail (stdlib smtplib). STARTTLS with certificate verification; plain SMTP only in development/test."""

from __future__ import annotations

import smtplib
import ssl
from email.message import EmailMessage
from email.utils import formatdate, make_msgid

from lifeguard.alerts.channels.base import ChannelError, Notification
from lifeguard.core.config import Settings


def build_message(settings: Settings, note: Notification) -> EmailMessage:
    if not settings.smtp_host or not settings.smtp_from or not settings.smtp_to_for(note.tier):
        raise ChannelError("smtp not configured")
    msg = EmailMessage()
    msg["Subject"] = note.subject().replace("\r", " ").replace("\n", " ")
    msg["From"] = settings.smtp_from
    msg["To"] = ", ".join(settings.smtp_to_for(note.tier))
    msg["Date"] = formatdate(localtime=False)
    msg["Message-ID"] = make_msgid(domain="lifeguard.local")
    msg["X-LifeGuard-Event"] = note.event_id
    msg["X-AquaGuard-Event"] = note.event_id  # rename shim, one release  # legacy-name
    msg["X-Priority"] = "1" if note.severity == "critical" else "3"
    lines = [note.message, "", f"code: {note.code}", f"source: {note.source}", f"at: {note.at}"]
    lines += [f"{k}: {v}" for k, v in sorted(note.detail.items())]
    msg.set_content("\n".join(lines))
    return msg


def send_smtp(settings: Settings, note: Notification) -> None:
    msg = build_message(settings, note)
    host = settings.smtp_host or ""
    try:
        with smtplib.SMTP(host, settings.smtp_port, timeout=settings.channel_timeout_s) as smtp:
            smtp.ehlo()
            if settings.smtp_starttls:
                # certificate + host name are always verified; smtp_ca_file only adds a trusted CA
                ctx = ssl.create_default_context(cafile=str(settings.smtp_ca_file) if settings.smtp_ca_file else None)
                smtp.starttls(context=ctx)
                smtp.ehlo()
            if settings.smtp_username and settings.smtp_password is not None:
                smtp.login(settings.smtp_username, settings.smtp_password.get_secret_value())
            refused = smtp.send_message(msg)
    except (smtplib.SMTPException, OSError) as exc:
        raise ChannelError(f"{type(exc).__name__}: {exc}") from exc
    if refused:
        raise ChannelError(f"recipients refused: {sorted(refused)}")
