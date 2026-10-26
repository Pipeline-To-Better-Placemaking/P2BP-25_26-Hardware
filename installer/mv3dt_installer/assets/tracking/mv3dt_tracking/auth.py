"""Ticket verification for the `/tracking` routes (STEP-8 section 3.2).

The web app mints a short-lived HS256 JWT and the browser sends it as
`Authorization: Bearer <ticket>`. A ticket that is missing, malformed, badly
signed, expired, for another audience or (when TICKET_ISSUER is set) from
another issuer is refused with 401. A valid ticket with the wrong scope or
for another project is refused with 403.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jwt
from fastapi import HTTPException, Request

AUDIENCE = "p2bp-workstation"
SCOPE = "tracking"
ALGORITHM = "HS256"


@dataclass(frozen=True)
class Claims:
    sub: str
    pid: str
    name: str


class TicketError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def _has_scope(value: object) -> bool:
    if isinstance(value, str):
        return SCOPE in value.split()
    if isinstance(value, (list, tuple)):
        return SCOPE in value
    return False


def verify_ticket(
    token: str, secret: str, project_id: str, issuer: str | None = None, leeway: float = 0
) -> Claims:
    """Return the ticket's claims or raise TicketError(401 or 403)."""
    if not secret:
        # Never verify against an empty key: anyone could sign with it.
        raise TicketError(401, "ticket verification is not configured")
    try:
        claims = jwt.decode(
            token,
            secret,
            algorithms=[ALGORITHM],
            audience=AUDIENCE,
            issuer=issuer,
            leeway=leeway,
            options={"require": ["exp", "aud"] + (["iss"] if issuer else [])},
        )
    except jwt.ExpiredSignatureError:
        raise TicketError(401, "ticket expired") from None
    except jwt.InvalidTokenError as exc:
        raise TicketError(401, f"invalid ticket: {exc}") from None

    sub = claims.get("sub")
    if not isinstance(sub, str) or not sub:
        raise TicketError(401, "invalid ticket: missing sub")
    if not _has_scope(claims.get("scope")):
        raise TicketError(403, "ticket scope does not include tracking")
    pid = claims.get("pid")
    if pid is None or str(pid) != str(project_id):
        raise TicketError(403, "ticket is for another project")
    name = claims.get("name")
    return Claims(sub=sub, pid=str(pid), name=name if isinstance(name, str) and name else sub)


def bearer_token(header: str | None) -> str:
    if not header:
        raise TicketError(401, "missing Authorization header")
    scheme, _, token = header.strip().partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise TicketError(401, "Authorization must be a Bearer ticket")
    return token.strip()


def ticket_dependency(
    secret: str, project_id: str, issuer: str | None = None
) -> Callable[[Request], Claims]:
    """A FastAPI dependency returning the request's verified Claims."""

    def dependency(request: Request) -> Claims:
        try:
            token = bearer_token(request.headers.get("authorization"))
            return verify_ticket(token, secret, project_id, issuer)
        except TicketError as exc:
            headers = {"WWW-Authenticate": "Bearer"} if exc.status == 401 else None
            raise HTTPException(exc.status, exc.message, headers=headers) from None

    return dependency
