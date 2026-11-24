"""``python -m lifeguard`` - administrative CLI (see lifeguard.cli)."""

from __future__ import annotations

import sys

if __name__ == "__main__":
    from lifeguard.cli import main

    sys.exit(main())
