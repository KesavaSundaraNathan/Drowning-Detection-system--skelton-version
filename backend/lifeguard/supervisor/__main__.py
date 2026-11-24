"""``python -m lifeguard.supervisor`` - the supervisor service (see lifeguard.supervisor.main)."""

from __future__ import annotations

import sys

if __name__ == "__main__":
    from lifeguard.supervisor.main import main

    sys.exit(main())
