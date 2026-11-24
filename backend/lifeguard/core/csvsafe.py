"""CSV export without formula injection.

A cell starting with ``= + - @`` (or tab / carriage return) is executed as a formula by
Excel/LibreOffice. Such cells are prefixed with a single quote, per OWASP's CSV-injection
guidance; everything is written through the csv module (correct quoting of , " and newlines).
"""

from __future__ import annotations

import csv
import io
from collections.abc import Iterable, Sequence

_DANGEROUS = ("=", "+", "-", "@", "\t", "\r")


def escape_cell(value: object) -> str:
    text = "" if value is None else str(value)
    return f"'{text}" if text.startswith(_DANGEROUS) else text


def to_csv(header: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\r\n")
    w.writerow([escape_cell(h) for h in header])
    for row in rows:
        w.writerow([escape_cell(v) for v in row])
    return buf.getvalue()
