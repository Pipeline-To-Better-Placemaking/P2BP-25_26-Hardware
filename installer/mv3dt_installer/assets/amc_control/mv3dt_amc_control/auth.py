"""Ticket verification for the `/amc` and `/projects` routes.

The web app (POST /api/workstation/project/:projectId/ticket) mints a
short-lived HS256 JWT and the browser sends it as
`Authorization: Bearer <ticket>`. A ticket that is missing, malformed, badly
signed, expired, for another audience, without a project (`pid`) or (when
TICKET_ISSUER is set) from another issuer is refused with 401.

The service serves any web project: the web app checks the user's project
permission before it signs, so any valid ticket's `pid` is accepted. Each
route accepts exactly the scopes it lists: "amc" (AMC control and the read
routes), "amc-provision" (POST /projects) or, for GET /projects/current,
either. A valid ticket with any other scope is refused with 403. A tracking ticket (scope "tracking") signed with
the same secret therefore cannot control AMC.
"""

from __future__ import annotations

from typing import Callable, Iterable, Union

import jwt
from fastapi import HTTPException, Request

AUDIENCE = "p2bp-workstation"
SCOPE = "amc"
PROVISION_SCOPE = "amc-provision"
ALGORITHM = "HS256"
INVALID = "Invalid or expired ticket"
_PURPOSE = {SCOPE: "AMC control", PROVISION_SCOPE: "project provisioning"}

# One scope, or a set of scopes any of which a route accepts.
Scopes = Union[str, Iterable[str]]


def _allowed(scope: Scopes) -> frozenset[str]:
    return frozenset((scope,) if isinstance(scope, str) else scope)


class TicketError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def verify_ticket(token: str, secret: str, issuer: str | None = None, scope: Scopes = SCOPE) -> dict:
    """Return the ticket's claims, or raise TicketError with the HTTP status to answer.

    `scope` is one scope or a set of them; the ticket's scope must be one of
    those exactly. Its `pid` may be any web project.
    """
    allowed = _allowed(scope)
    options = {"require": ["exp", "sub", "pid"]}
    kwargs = {"issuer": issuer} if issuer else {}
    try:
        claims = jwt.decode(token, secret, algorithms=[ALGORITHM], audience=AUDIENCE, options=options, **kwargs)
    except jwt.PyJWTError:
        raise TicketError(401, INVALID) from None
    # `require` only checks presence; an empty pid names no project.
    if claims["pid"] in (None, ""):
        raise TicketError(401, INVALID)
    given = claims.get("scope")
    if not isinstance(given, str) or given not in allowed:
        purposes = " or ".join(sorted(_PURPOSE.get(s, s) for s in allowed))
        raise TicketError(403, f"This ticket is not for {purposes}")
    return claims


def ticket_dependency(secret: str, issuer: str | None, scope: Scopes = SCOPE) -> Callable[[Request], dict]:
    """A FastAPI dependency that returns the verified claims of a ticket with one of `scope`."""
    allowed = _allowed(scope)

    def dependency(request: Request) -> dict:
        if not secret:
            raise HTTPException(503, "WORKSTATION_TICKET_SECRET is not configured")
        header = request.headers.get("authorization", "")
        scheme, _, token = header.partition(" ")
        if scheme.lower() != "bearer" or not token.strip():
            raise HTTPException(401, "Missing bearer ticket")
        try:
            return verify_ticket(token.strip(), secret, issuer, allowed)
        except TicketError as exc:
            raise HTTPException(exc.status, exc.message) from None

    return dependency
