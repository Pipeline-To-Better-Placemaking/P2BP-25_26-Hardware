"""The read-only tracking history API (STEP-8 sections 8 and 9).

`create_app` builds the FastAPI app around a TrackStore, the settings and an
optional running Recorder. `range` and `export` default to (and `export`
offers) fused people from `fuse`, cleaned with the workstation's
`cleaning.json` and `offsets.json`, both re-read when they change on disk. Session endpoints answer with PascalCase keys,
data endpoints with lowercase keys, and every error is `{"Message": ...}`.
Data handlers are plain functions, so FastAPI runs them in its threadpool
and SQLite reads never block the event loop.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator, Iterator, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool

from . import __version__, calib
from .auth import Claims, ticket_dependency
from .cleaning import CleaningConfig, CleaningFile, resolve_room
from .fuse import FusedSample, Fuser, fuse, offsets_summary, summarize
from .lock import Busy, LeaseManager, Locked, NoLease, NotHolder, RateLimited
from .offsets import load_offsets
from .settings import Settings
from .stitch import raw_series
from .store import Detection, TrackStore, bucket_ms

log = logging.getLogger(__name__)

HOUR_MS = 3600 * 1000
RANGE_MAX_MS = 6 * HOUR_MS
EXPORT_MAX_MS = 24 * HOUR_MS
MAX_HZ = 25.0
# Samples per range request (hz x window seconds): 6 h at 5 Hz, 72 min at 25 Hz.
RANGE_MAX_SAMPLES = 108_000
# A fused export feeds one Fuser epoch-aligned 10-minute windows of raw rows;
# a raw export reads one minute at a time, so memory is bounded by a chunk.
FUSE_CHUNK_MS = 10 * 60 * 1000
RAW_CHUNK_MS = 60 * 1000
SESSION_HEADER = "X-Tracking-Session"
COLUMNS = ("ts_ms", "sensor", "track_id", "x", "y", "z", "w", "l", "h", "conf")
FUSED_COLUMNS = ("ts_ms", "person_id", "x", "y", "cameras")
STITCHED_DEPRECATION = "mode=stitched is an alias of fused and will be removed; use mode=fused"


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


def _windows(from_ms: int, to_ms: int, width: int) -> Iterator[tuple[int, int]]:
    """Inclusive (start, stop) windows covering [from_ms, to_ms], with inner
    edges on epoch multiples of `width`."""
    start = from_ms
    while start <= to_ms:
        stop = min(start - start % width + width - 1, to_ms)
        yield start, stop
        start = stop + 1


class OffsetsFile:
    """offsets.json, re-read when its mtime changes. A missing file gives
    None; an invalid one gives None with the reason in `error`."""

    def __init__(self, path: Path | None) -> None:
        self.path = None if path is None else Path(path)
        self._stamp: object = object()
        self._offsets: dict | None = None
        self.error: str | None = None

    def get(self) -> dict | None:
        stamp = None
        if self.path is not None:
            try:
                st = os.stat(self.path)
                stamp = (st.st_mtime_ns, st.st_size, st.st_ino)
            except FileNotFoundError:
                pass
        if stamp != self._stamp:
            self._stamp = stamp
            try:
                self._offsets = load_offsets(self.path)
                self.error = None
            except (OSError, ValueError) as exc:  # JSONDecodeError is a ValueError
                self._offsets, self.error = None, str(exc)
                log.error("offsets: %s; not applying a correction", exc)
        return self._offsets


class Cleaning:
    """What one fused request runs with: the current cleaning config (with
    its load error and ignored keys), the learned offsets and the room."""

    def __init__(self, settings: Settings) -> None:
        self.config_file = CleaningFile(settings.cleaning_file)
        self.offsets_file = OffsetsFile(settings.offsets_file)
        self.calib_dir = settings.calib_dir

    def current(self) -> tuple[CleaningConfig, dict | None, object]:
        config = self.config_file.get()
        return config, self.offsets_file.get(), resolve_room(config, self.calib_dir)

    def summary_extras(self, config: CleaningConfig, offsets: dict | None) -> dict:
        """Fields every fused summary adds to the engine's own."""
        return {
            "config_error": self.config_file.error,
            "config_warnings": [f"ignored unknown key {k}" for k in config.ignored_keys],
            "offsets_extra": {
                "enabled": config.offsets_apply,
                "learned": offsets is not None and bool(offsets.get("sensors")),
                "error": self.offsets_file.error,
            },
        }


def finish_summary(summary: dict, extras: dict) -> dict:
    extras = dict(extras)
    summary["offsets"].update(extras.pop("offsets_extra"))
    summary.update(extras)
    return summary


def _renumber(people: list[dict], persons: list[dict]) -> None:
    """Person ids 1..N by first appearance, in place."""
    order = sorted(people, key=lambda p: (p["t"][0], p["id"]))
    new = {p["id"]: i for i, p in enumerate(order, 1)}
    for p in people:
        p["id"] = new[p["id"]]
    for p in persons:
        p["id"] = new[p["id"]]
    people.sort(key=lambda p: p["id"])
    persons.sort(key=lambda p: p["id"])


def _downsample(person: dict, width: int) -> dict:
    """A fused person's series re-bucketed to `width` ms: mean position and
    the largest camera count per bucket."""
    acc: dict[int, list] = {}
    for t, x, y, n in zip(person["t"], person["x"], person["y"], person["n_cams"]):
        cell = acc.setdefault(t - t % width, [0.0, 0.0, 0, 0])
        cell[0] += x
        cell[1] += y
        cell[2] += 1
        cell[3] = max(cell[3], n)
    ts = sorted(acc)
    return {
        **person,
        "t": ts,
        "x": [acc[b][0] / acc[b][2] for b in ts],
        "y": [acc[b][1] / acc[b][2] for b in ts],
        "n_cams": [acc[b][3] for b in ts],
    }


def fused_range(store: TrackStore, cleaning: Cleaning, from_ms: int, to_ms: int, hz: float) -> dict:
    """The fused `range` body (without from/to/mode): people renumbered by
    first appearance, downsampled when `hz` is coarser than the fusion
    bucket, and the summary. `hz` in the result is the output rate."""
    config, offsets, room = cleaning.current()
    rows = store.query_range(from_ms, to_ms)
    result = fuse(_positioned(rows), config, offsets, room)
    summary = finish_summary(result.summary, cleaning.summary_extras(config, offsets))
    people = result.people
    _renumber(people, summary["persons"])
    width = bucket_ms(hz)
    if width > config.bucket_ms:
        people = [_downsample(p, width) for p in people]
    else:
        width = config.bucket_ms
    return {"hz": 1000.0 / width, "people": people, "summary": summary}


def fused_samples(store: TrackStore, fuser: Fuser, from_ms: int, to_ms: int) -> Iterator[FusedSample]:
    """Stream [from_ms, to_ms] through one Fuser in epoch-aligned 10-minute
    windows, yielding samples as their people become final, then the rest.
    Gives the same people and ids as one `fuse` call over the window."""
    for start, stop in _windows(from_ms, to_ms, FUSE_CHUNK_MS):
        yield from fuser.feed(_positioned(store.query_range(start, stop)))
    yield from fuser.finish()


def stream_summary(samples: list[FusedSample], fuser: Fuser, offsets: dict | None) -> dict:
    """The engine summary for samples streamed through `fuser`, as `fuse`
    builds it for a one-shot call."""
    _, summary = summarize(samples)
    summary["dropped"] = {**fuser.dropped, "zones": dict(fuser.dropped["zones"])}
    summary["offsets"] = offsets_summary(offsets, fuser.offsets_applied)
    summary["preset"] = fuser.config.preset
    return summary


def export_records(
    store: TrackStore, from_ms: int, to_ms: int, fused: bool, cleaning: Cleaning | None = None
) -> Iterator[dict]:
    """Records of [from_ms, to_ms] as dicts.

    Raw: every stored row, in ts order, read a minute at a time. Fused: one
    record per person per fusion bucket (`ts_ms, person_id, x, y, cameras`,
    cameras joined with ";"), in blocks as people become final; ids match
    a fused `range` over the same window.
    """
    if fused:
        config, offsets, room = (cleaning.current() if cleaning is not None
                                 else (CleaningConfig(), None, None))
        for s in fused_samples(store, Fuser(config, offsets, room), from_ms, to_ms):
            yield {"ts_ms": s.ts_ms, "person_id": s.person_id, "x": s.x, "y": s.y,
                   "cameras": ";".join(s.cameras)}
        return
    for start, stop in _windows(from_ms, to_ms, RAW_CHUNK_MS):
        for row in store.query_range(start, stop):
            yield asdict(row)


def encode_export(records: Iterator[dict], format: str, columns: tuple[str, ...]) -> Iterator[str]:
    """CSV or JSON Lines text for the records, in chunks of about 64 KiB."""
    if format == "csv":
        buf = io.StringIO()
        writer = csv.writer(buf, lineterminator="\n")
        writer.writerow(columns)
        for rec in records:
            writer.writerow(["" if rec[c] is None else rec[c] for c in columns])
            if buf.tell() > 64 * 1024:
                yield buf.getvalue()
                buf.seek(0)
                buf.truncate()
        yield buf.getvalue()
    else:
        lines: list[str] = []
        size = 0
        for rec in records:
            line = json.dumps({c: rec[c] for c in columns}, separators=(",", ":"))
            lines.append(line)
            size += len(line) + 1
            if size > 64 * 1024:
                yield "\n".join(lines) + "\n"
                lines, size = [], 0
        if lines:
            yield "\n".join(lines) + "\n"


def create_app(
    settings: Settings,
    store: TrackStore,
    recorder=None,
    leases: LeaseManager | None = None,
) -> FastAPI:
    if any("*" in o for o in settings.allowed_origins):
        raise ValueError("ALLOWED_ORIGINS must list origins, not a wildcard")
    app = FastAPI(title="mv3dt-tracking", version=__version__, docs_url=None, redoc_url=None, openapi_url=None)
    leases = leases or LeaseManager()
    app.state.leases = leases
    app.state.store = store
    cleaning = Cleaning(settings)
    app.state.cleaning = cleaning

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
        mode: Literal["fused", "stitched", "raw"] = Query("fused"),
        session: str | None = Header(None, alias=SESSION_HEADER),
        claims: Claims = Depends(ticket),
    ):
        _check_window(from_ms, to_ms, RANGE_MAX_MS)
        if not 0 < hz <= MAX_HZ:
            raise HTTPException(400, f"hz must be > 0 and <= {MAX_HZ:g}")
        if hz * (to_ms - from_ms) / 1000 > RANGE_MAX_SAMPLES:
            raise HTTPException(
                400,
                f"hz x window seconds must not exceed {RANGE_MAX_SAMPLES}"
                " (6 h at 5 Hz, 72 min at 25 Hz); lower hz or narrow the window",
            )
        admitted = guard("range", session, claims)
        try:
            if mode == "raw":
                rows = _positioned(store.query_range(from_ms, to_ms, hz))
                body = {"from": from_ms, "to": to_ms, "hz": hz, "mode": "raw", "people": raw_series(rows, hz)}
            else:
                fused = fused_range(store, cleaning, from_ms, to_ms, hz)
                body = {"from": from_ms, "to": to_ms, "hz": fused["hz"], "mode": "fused",
                        "people": fused["people"], "summary": fused["summary"]}
                if mode == "stitched":
                    body["deprecated"] = STITCHED_DEPRECATION
            # Serialize while the guard is held, so the work counts as in flight.
            return JSONResponse(body)
        finally:
            leases.end(admitted)

    @app.get("/tracking/export")
    def export(
        request: Request,
        from_ms: int = Query(alias="from"),
        to_ms: int = Query(alias="to"),
        mode: Literal["fused", "stitched", "raw"] = Query("raw"),
        format: Literal["csv", "jsonl"] = Query("csv"),
        session: str | None = Header(None, alias=SESSION_HEADER),
        claims: Claims = Depends(ticket),
    ):
        _check_window(from_ms, to_ms, EXPORT_MAX_MS)
        admitted = guard("export", session, claims)
        fused = mode != "raw"
        mode = "fused" if fused else "raw"
        columns = FUSED_COLUMNS if fused else COLUMNS

        chunks = encode_export(export_records(store, from_ms, to_ms, fused, cleaning), format, columns)

        async def body() -> AsyncIterator[str]:
            # uvicorn advertises ASGI 2.4, so Starlette does not watch for a
            # disconnect and send() returns quietly once the client is gone.
            # Check between chunks, so an abandoned export stops generating
            # and frees the guard at once rather than at the end of the data.
            try:
                while not await request.is_disconnected():
                    chunk = await run_in_threadpool(next, chunks, None)
                    if chunk is None:
                        break
                    yield chunk
            finally:
                leases.end(admitted)
                try:
                    chunks.close()
                except ValueError:
                    # Cancelled while a worker thread is inside next(); that
                    # chunk finishes and the generator is never resumed.
                    pass

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
