"""The read-only tracking history API (STEP-8 sections 8 and 9).

`create_app` builds the FastAPI app around a TrackStore, the settings and an
optional running Recorder. Session endpoints answer with PascalCase keys,
data endpoints with lowercase keys, and every error is `{"Message": ...}`.
Data handlers are plain functions, so FastAPI runs them in its threadpool
and SQLite reads never block the event loop.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel
from starlette.background import BackgroundTask

from . import __version__, calib
from .auth import Claims, ticket_dependency
from .lock import Busy, LeaseManager, Locked, NoLease, NotHolder, RateLimited
from .settings import Settings
from .stitch import people_series, raw_series, stitch
from .store import Detection, TrackStore

HOUR_MS = 3600 * 1000
RANGE_MAX_MS = 6 * HOUR_MS
EXPORT_MAX_MS = 24 * HOUR_MS
EXPORT_CHUNK_MS = HOUR_MS
MAX_HZ = 25.0
STITCH_HZ = 5.0
SESSION_HEADER = "X-Tracking-Session"
COLUMNS = ("ts_ms", "sensor", "track_id", "x", "y", "z", "w", "l", "h", "conf")


class SessionRequest(BaseModel):
    kind: Literal["replay", "export"]


def _iso(seconds: float) -> str:
    """ISO 8601 UTC with milliseconds, e.g. 2026-10-25T12:00:45.000Z."""
    stamp = datetime.fromtimestamp(seconds, tz=timezone.utc)
    return stamp.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _error(status: int, message: str, headers: dict | None = None) -> JSONResponse:
    return JSONResponse({"Message": message}, status_code=status, headers=headers)


def _locked(state: Locked) -> JSONResponse:
    return JSONResponse({"HeldBy": state.held_by, "Until": _iso(state.until)}, status_code=423)


def _check_window(from_ms: int, to_ms: int, max_ms: int) -> None:
    if to_ms < from_ms:
        raise HTTPException(400, "to must not be before from")
    if to_ms - from_ms > max_ms:
        raise HTTPException(400, f"range is wider than {max_ms // HOUR_MS} h")


def _positioned(rows: list[Detection]) -> list[Detection]:
    return [r for r in rows if r.x is not None and r.y is not None]


def create_app(
    settings: Settings,
    store: TrackStore,
    recorder=None,
    leases: LeaseManager | None = None,
) -> FastAPI:
    app = FastAPI(title="mv3dt-tracking", version=__version__, docs_url=None, redoc_url=None, openapi_url=None)
    leases = leases or LeaseManager()
    app.state.leases = leases
    app.state.store = store

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.allowed_origins),
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Authorization", "Content-Type", SESSION_HEADER],
        expose_headers=["Content-Disposition", "Retry-After"],
        allow_credentials=False,
    )

    @app.exception_handler(HTTPException)
    async def _http_error(request: Request, exc: HTTPException):
        return _error(exc.status_code, str(exc.detail), getattr(exc, "headers", None))

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError):
        parts = []
        for err in exc.errors():
            loc = ".".join(str(p) for p in err.get("loc", ()) if p not in ("query", "body"))
            parts.append(f"{loc}: {err.get('msg')}" if loc else str(err.get("msg")))
        return _error(400, "; ".join(parts) or "invalid request")

    ticket = ticket_dependency(settings.ticket_secret, settings.project_id, settings.ticket_issuer)

    def guard(op: str, session: str | None, claims: Claims):
        try:
            return leases.begin(session, claims.sub, op)
        except NoLease as exc:
            raise HTTPException(423, str(exc)) from None
        except (RateLimited, Busy) as exc:
            raise HTTPException(429, str(exc), headers={"Retry-After": str(exc.retry_after)}) from None

    # -- open ---------------------------------------------------------------

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "version": __version__,
            "recorder": recorder.stats() if recorder is not None else None,
        }

    # -- metadata ------------------------------------------------------------

    @app.get("/tracking/meta")
    def meta(claims: Claims = Depends(ticket)):
        span = store.span()
        size = calib.png_size(Path(settings.calib_dir) / "map.png") if settings.calib_dir else None
        t_gt2px = calib.read_t_gt2px(settings.calib_dir)
        floorplan = None
        if size is not None and t_gt2px is not None:
            floorplan = {"width": size[0], "height": size[1], "T_gt2px": t_gt2px}
        return {
            "location_id": settings.location_id or None,
            "span": [span[0], span[1]] if span else None,
            "floorplan": floorplan,
            "cameras": calib.read_cameras(settings.camera_models_dir) or [],
        }

    @app.get("/tracking/floorplan.png")
    def floorplan(claims: Claims = Depends(ticket)):
        path = Path(settings.calib_dir) / "map.png" if settings.calib_dir else None
        if path is None or not path.is_file():
            raise HTTPException(404, "no floor plan image")
        return FileResponse(path, media_type="image/png")

    # -- lease ---------------------------------------------------------------

    @app.post("/tracking/session")
    def acquire(body: SessionRequest, claims: Claims = Depends(ticket)):
        state = leases.acquire(claims.sub, claims.name, body.kind)
        if isinstance(state, Locked):
            return _locked(state)
        return {"SessionId": state.id, "ExpiresAt": _iso(leases.expires_at(state))}

    @app.post("/tracking/session/{session_id}/heartbeat")
    def heartbeat(session_id: str, claims: Claims = Depends(ticket)):
        state = leases.heartbeat(session_id, claims.sub)
        if state is None:
            raise HTTPException(404, "no such session; acquire a new one")
        if isinstance(state, Locked):
            return _locked(state)
        return {"ExpiresAt": _iso(leases.expires_at(state))}

    @app.delete("/tracking/session/{session_id}", status_code=204)
    def release(session_id: str, claims: Claims = Depends(ticket)):
        try:
            released = leases.release(session_id, claims.sub)
        except NotHolder as exc:
            raise HTTPException(403, str(exc)) from None
        if not released:
            raise HTTPException(404, "no such session")
        return Response(status_code=204)

    # -- data ----------------------------------------------------------------

    @app.get("/tracking/range")
    def range_(
        from_ms: int = Query(alias="from"),
        to_ms: int = Query(alias="to"),
        hz: float = Query(5.0),
        mode: Literal["stitched", "raw"] = Query("stitched"),
        session: str | None = Header(None, alias=SESSION_HEADER),
        claims: Claims = Depends(ticket),
    ):
        _check_window(from_ms, to_ms, RANGE_MAX_MS)
        if not 0 < hz <= MAX_HZ:
            raise HTTPException(400, f"hz must be > 0 and <= {MAX_HZ:g}")
        admitted = guard("range", session, claims)
        try:
            rows = _positioned(store.query_range(from_ms, to_ms, hz))
            if mode == "stitched":
                people = people_series(rows, stitch(rows), hz)
            else:
                people = raw_series(rows, hz)
        finally:
            leases.end(admitted)
        return {"from": from_ms, "to": to_ms, "hz": hz, "mode": mode, "people": people}

    @app.get("/tracking/export")
    def export(
        from_ms: int = Query(alias="from"),
        to_ms: int = Query(alias="to"),
        mode: Literal["stitched", "raw"] = Query("raw"),
        format: Literal["csv", "jsonl"] = Query("csv"),
        session: str | None = Header(None, alias=SESSION_HEADER),
        claims: Claims = Depends(ticket),
    ):
        _check_window(from_ms, to_ms, EXPORT_MAX_MS)
        admitted = guard("export", session, claims)
        try:
            mapping = None
            if mode == "stitched":
                # Person ids come from the same 5 Hz stitching `range` uses by
                # default, so an export and a replay agree on who is who.
                mapping = stitch(_positioned(store.query_range(from_ms, to_ms, STITCH_HZ)))
        except BaseException:
            leases.end(admitted)
            raise

        columns = COLUMNS + (("person_id",) if mapping is not None else ())

        def records() -> Iterator[dict]:
            start = from_ms
            while start <= to_ms:
                stop = min(start + EXPORT_CHUNK_MS - 1, to_ms)
                for row in store.query_range(start, stop):
                    rec = asdict(row)
                    if mapping is not None:
                        rec["person_id"] = mapping.get(row.track_id)
                    yield rec
                start = stop + 1

        def body() -> Iterator[str]:
            try:
                if format == "csv":
                    buf = io.StringIO()
                    writer = csv.writer(buf, lineterminator="\n")
                    writer.writerow(columns)
                    for rec in records():
                        writer.writerow(["" if rec[c] is None else rec[c] for c in columns])
                        if buf.tell() > 64 * 1024:
                            yield buf.getvalue()
                            buf.seek(0)
                            buf.truncate()
                    yield buf.getvalue()
                else:
                    chunk: list[str] = []
                    for rec in records():
                        chunk.append(json.dumps({c: rec[c] for c in columns}, separators=(",", ":")))
                        if len(chunk) >= 1000:
                            yield "\n".join(chunk) + "\n"
                            chunk = []
                    if chunk:
                        yield "\n".join(chunk) + "\n"
            finally:
                leases.end(admitted)

        media = "text/csv" if format == "csv" else "application/x-ndjson"
        filename = f"tracking_{from_ms}_{to_ms}_{mode}.{format}"
        return StreamingResponse(
            body(),
            media_type=media,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
            # The generator's finally does not run if the client goes away
            # before streaming starts; ending the guard here as well covers that.
            background=BackgroundTask(leases.end, admitted),
        )

    return app
