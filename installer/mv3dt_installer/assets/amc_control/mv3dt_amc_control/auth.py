"""Ticket verification for the `/amc` routes.

The web app (POST /api/workstation/project/:projectId/ticket) mints a
short-lived HS256 JWT with scope "amc" and the browser sends it as
`Authorization: Bearer <ticket>`. A ticket that is missing, malformed, badly
signed, expired, for another audience or (when TICKET_ISSUER is set) from
another issuer is refused with 401. A valid ticket with the wrong scope or
for another project is refused with 403. A tracking ticket (scope
"tracking") signed with the same secret therefore cannot control AMC.
"""

from __future__ import annotations

from typing import Callable

import jwt
from fastapi import HTTPException, Request

AUDIENCE = "p2bp-workstation"
SCOPE = "amc"
ALGORITHM = "HS256"
INVALID = "Invalid or expired ticket"


class TicketError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def verify_ticket(token: str, secret: str, project_id: str = "", issuer: str | None = None) -> dict:
    """Return the ticket's claims, or raise TicketError with the HTTP status to answer."""
    options = {"require": ["exp", "sub", "pid"]}
    kwargs = {"issuer": issuer} if issuer else {}
    try:
        claims = jwt.decode(token, secret, algorithms=[ALGORITHM], audience=AUDIENCE, options=options, **kwargs)
    except jwt.PyJWTError:
        raise TicketError(401, INVALID) from None
    if claims.get("scope") != SCOPE:
        raise TicketError(403, "This ticket is not for AMC control")
    if project_id and claims.get("pid") != project_id:
        raise TicketError(403, "This workstation serves a different project")
    return claims


def ticket_dependency(secret: str, project_id: str, issuer: str | None) -> Callable[[Request], dict]:
    """A FastAPI dependency that returns the verified claims."""

    def dependency(request: Request) -> dict:
        if not secret:
            raise HTTPException(503, "WORKSTATION_TICKET_SECRET is not configured")
        header = request.headers.get("authorization", "")
        scheme, _, token = header.partition(" ")
        if scheme.lower() != "bearer" or not token.strip():
            raise HTTPException(401, "Missing bearer ticket")
        try:
            return verify_ticket(token.strip(), secret, project_id, issuer)
        except TicketError as exc:
            raise HTTPException(exc.status, exc.message) from None

    return dependency
