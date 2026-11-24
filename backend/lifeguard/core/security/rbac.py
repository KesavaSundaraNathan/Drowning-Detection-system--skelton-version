"""Roles and the authenticated principal.

Hierarchy: admin > supervisor > lifeguard. A route declares the minimum role it needs with
``Depends(RequireRole(Role.X))``; tests/api/test_auth_matrix.py walks every route and fails if
one declares no role (or is reachable without a session) - a new route cannot ship unguarded.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Role(StrEnum):
    LIFEGUARD = "lifeguard"
    SUPERVISOR = "supervisor"
    ADMIN = "admin"

    @property
    def rank(self) -> int:
        return _RANK[self]

    def at_least(self, other: Role) -> bool:
        return self.rank >= other.rank


_RANK = {Role.LIFEGUARD: 1, Role.SUPERVISOR: 2, Role.ADMIN: 3}


@dataclass(frozen=True)
class Principal:
    user_id: int
    username: str
    role: Role
    session_id: int
