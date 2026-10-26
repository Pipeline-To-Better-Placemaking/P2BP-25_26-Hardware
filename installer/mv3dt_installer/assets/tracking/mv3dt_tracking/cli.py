"""`python -m mv3dt_tracking serve | stats | wipe --yes [--before YYYY-MM-DD]`.

`serve` runs the recorder and the API in one process (the unit's
ExecStart). `stats` and `wipe` work on the data directory directly; stopping
the service around a wipe is the caller's job.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date

from .settings import Settings, is_loopback
from .store import Detection, TrackStore

log = logging.getLogger("mv3dt_tracking")

_FIELDS = ("ts_ms", "sensor", "track_id", "x", "y", "z", "w", "l", "h", "conf")


def rows_to_detections(rows: list[dict]) -> list[Detection]:
    """Recorder row dicts to Detections; rows without a timestamp or id are dropped."""
    out = []
    for r in rows:
        if r.get("ts_ms") is None or r.get("track_id") is None:
            continue
        out.append(Detection(**{k: r.get(k) for k in _FIELDS}))
    return out


def _serve(settings: Settings) -> int:
    missing = [
        name
        for name, value in (
            ("WORKSTATION_TICKET_SECRET", settings.ticket_secret),
            ("TRACKING_PROJECT_ID", settings.project_id),
            ("MQTT_TOPIC or LOCATION_ID", settings.mqtt_topic),
        )
        if not value
    ]
    if missing:
        print(f"refusing to serve: {', '.join(missing)} not set", file=sys.stderr)
        return 2
    if any("*" in o for o in settings.allowed_origins):
        print("refusing to serve: ALLOWED_ORIGINS must list origins, not *", file=sys.stderr)
        return 2
    try:
        host, port = settings.bind_address()
    except ValueError as exc:
        print(f"refusing to serve: {exc}", file=sys.stderr)
        return 2
    if not is_loopback(host) and not settings.allow_non_local_bind:
        print(
            f"refusing to bind {host}: only loopback is allowed"
            " (set ALLOW_NON_LOCAL_BIND=1 to override)",
            file=sys.stderr,
        )
        return 2

    import uvicorn

    from .api import create_app
    from .recorder import Recorder

    store = TrackStore(settings.data_dir)
    recorder = Recorder(
        lambda rows: store.insert_many(rows_to_detections(rows)),
        settings.mqtt_topic,
        settings.mqtt_host,
        settings.mqtt_port,
    )
    app = create_app(settings, store, recorder)
    recorder.start()
    try:
        uvicorn.run(app, host=host, port=port, log_level="info", proxy_headers=False)
    finally:
        recorder.stop()
    return 0


def _stats(settings: Settings) -> int:
    store = TrackStore(settings.data_dir)
    stats = store.stats()
    span = store.span()
    stats["span"] = {"from": span[0], "to": span[1]} if span else None
    print(json.dumps(stats, indent=2))
    return 0


def _wipe(settings: Settings, yes: bool, before: date | None) -> int:
    if not yes:
        print("refusing to wipe without --yes", file=sys.stderr)
        return 2
    removed = TrackStore(settings.data_dir).wipe(before)
    print(json.dumps({"removed": removed, "before": before.isoformat() if before else None}))
    return 0


def _day(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected YYYY-MM-DD, got {value!r}") from None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mv3dt_tracking", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("serve", help="record the tracker stream and serve the history API")
    sub.add_parser("stats", help="print row counts per day file as JSON")
    wipe = sub.add_parser("wipe", help="delete recorded data")
    wipe.add_argument("--yes", action="store_true", help="confirm the deletion")
    wipe.add_argument("--before", type=_day, help="only delete UTC days before YYYY-MM-DD")
    return parser


def main(argv: list[str] | None = None, env=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = Settings.from_env(env)
    if args.command == "serve":
        return _serve(settings)
    if args.command == "stats":
        return _stats(settings)
    return _wipe(settings, args.yes, args.before)
