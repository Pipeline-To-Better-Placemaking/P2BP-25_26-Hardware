"""`python -m mv3dt_tracking serve | stats | wipe | zones-suggest | learn-offsets | clean-report`.

`serve` runs the recorder and the API in one process (the unit's
ExecStart). `stats` and `wipe` work on the data directory directly; stopping
the service around a wipe is the caller's job. `zones-suggest`,
`learn-offsets` and `clean-report` read the day files, `cleaning.json`,
`offsets.json` and the calibration exactly as `serve` does, and need neither
the lease nor a ticket; only `learn-offsets --save` writes anything.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from datetime import date, datetime
from typing import Iterable, Iterator

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


# -- cleaning commands -----------------------------------------------------------

HOUR_MS = 3600 * 1000
CHUNK_MS = 10 * 60 * 1000

ZONES_WARNING = (
    "WARNING: these are static hotspots seen by one camera only. A seated person"
    " (for example someone on a couch that only one camera sees) looks the same."
    " Check each one in Replay or in the camera footage before pasting it into"
    " cleaning.json; nothing has been written."
)


def _time_ms(value: str) -> int:
    """Unix milliseconds, or ISO 8601 (no offset means workstation local time)."""
    v = value.strip()
    if v.lstrip("-").isdigit():
        return int(v)
    try:
        stamp = datetime.fromisoformat(v.replace("Z", "+00:00"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected unix ms or ISO 8601, got {value!r}") from None
    if stamp.tzinfo is None:
        stamp = stamp.astimezone()  # local time
    return int(stamp.timestamp() * 1000)


def _window(args, default_hours: float) -> tuple[int, int]:
    to_ms = args.to_ms if args.to_ms is not None else int(time.time() * 1000)
    if args.from_ms is not None:
        from_ms = args.from_ms
    else:
        hours = args.hours if args.hours is not None else default_hours
        from_ms = to_ms - int(hours * HOUR_MS)
    if from_ms > to_ms:
        raise SystemExit("--from must not be after --to")
    return from_ms, to_ms


def _chunks(store: TrackStore, from_ms: int, to_ms: int) -> Iterator[list[Detection]]:
    """Positioned raw rows in epoch-aligned 10-minute chunks, in time order."""
    start = from_ms
    while start <= to_ms:
        stop = min(start - start % CHUNK_MS + CHUNK_MS - 1, to_ms)
        yield [r for r in store.query_range(start, stop) if r.x is not None and r.y is not None]
        start = stop + 1


def survivors(chunks: Iterable[list[Detection]], config, room, drop_short: bool = False) -> Iterator[Detection]:
    """Detections surviving fusion stages 1 and 2 (room bounds, zones, the
    per-(track, camera) speed gate), and with `drop_short` also stage 3's
    short-tracklet rule. Uses the same rules and thresholds as `fuse`, with
    no offset correction. Without `drop_short` rows come out in time order;
    with it, each tracklet's rows come out when it closes."""
    m = config.room_margin_m
    gap_ms = config.tracklet_gap_s * 1000
    open_: dict[tuple[int, str], list] = {}  # key -> [last (ts, x, y), rows]

    def release(rows: list[Detection]) -> list[Detection]:
        if not drop_short:
            return []
        buckets = {r.ts_ms // config.bucket_ms for r in rows}
        span_s = (max(buckets) - min(buckets)) * config.bucket_ms / 1000.0
        if len(buckets) >= config.min_tracklet_samples and span_s >= config.min_tracklet_s:
            return rows
        return []

    for chunk in chunks:
        for r in chunk:
            if room is not None:
                (x0, x1), (y0, y1) = room
                if not (x0 - m <= r.x <= x1 + m and y0 - m <= r.y <= y1 + m):
                    continue
            if any(z.contains(r.sensor, r.x, r.y) for z in config.zones):
                continue
            key = (r.track_id, r.sensor)
            t = open_.get(key)
            if t is not None and r.ts_ms - t[0][0] > gap_ms:
                yield from release(open_.pop(key)[1])
                t = None
            if t is not None:
                lt, lx, ly = t[0]
                dt = max((r.ts_ms - lt) / 1000.0, config.min_dt_s)
                if math.hypot(r.x - lx, r.y - ly) > config.vmax_mps * dt + config.speed_slack_m:
                    continue
                t[0] = (r.ts_ms, r.x, r.y)
            else:
                t = open_[key] = [(r.ts_ms, r.x, r.y), []]
            if drop_short:
                t[1].append(r)
            else:
                yield r
        if drop_short:
            last = chunk[-1].ts_ms if chunk else None
            if last is not None:
                for key in [k for k, t in open_.items() if last - t[0][0] > gap_ms]:
                    yield from release(open_.pop(key)[1])
    for t in open_.values():
        yield from release(t[1])


def _context(settings: Settings):
    """Store, cleaning state and the current (config, offsets, room)."""
    from .api import Cleaning

    cleaning = Cleaning(settings)
    config, offsets, room = cleaning.current()
    if cleaning.config_file.error:
        print(f"warning: {cleaning.config_file.error}; using the last valid config", file=sys.stderr)
    for k in config.ignored_keys:
        print(f"warning: cleaning.json: ignored unknown key {k}", file=sys.stderr)
    return TrackStore(settings.data_dir), cleaning, config, offsets, room


def _zones_suggest(settings: Settings, args) -> int:
    from .cleaning import suggest_zones

    store, _, config, _, room = _context(settings)
    from_ms, to_ms = _window(args, 24)
    found = suggest_zones(survivors(_chunks(store, from_ms, to_ms), config, room))
    if args.json:
        print(json.dumps({"from": from_ms, "to": to_ms, "warning": ZONES_WARNING, "suggestions": found}, indent=2))
        return 0
    print(ZONES_WARNING)
    print()
    if not found:
        print("no static single-camera hotspots found")
        return 0
    for s in found:
        st = s["stats"]
        print(
            f"{s['zone']['name']}: camera {st['camera']} ({st['camera_share']:.0%} of samples),"
            f" occupied {st['occupied_s']} s ({st['occupancy_frac']:.0%} of occupied time),"
            f" {st['ids']} tracker ids, median speed {st['median_speed_mps']} m/s"
        )
        print("  " + json.dumps(s["zone"]) + ",")
    return 0


def _learn_offsets(settings: Settings, args) -> int:
    from .offsets import learn_offsets, save_offsets

    store, _, config, _, room = _context(settings)
    from_ms, to_ms = _window(args, 72)
    rows = survivors(_chunks(store, from_ms, to_ms), config, room, drop_short=True)
    result = learn_offsets(rows, bucket_ms=config.bucket_ms, gate_m=config.shared_id_gate_m,
                           model=args.model, window={"from_ms": from_ms, "to_ms": to_ms})
    report = {k: result.get(k) for k in (
        "model", "pairs", "min_pairs", "median_dist_before_m", "median_dist_after_m", "gain_m", "needed",
    )}
    report["sensors"] = result["sensors"]
    report["window"] = result["window"]
    report["saved"] = None
    code = 0
    if args.save:
        try:
            save_offsets(settings.offsets_file, result)
            report["saved"] = str(settings.offsets_file)
        except (ValueError, OSError) as exc:
            print(f"not saved: {exc}", file=sys.stderr)
            code = 1
    print(json.dumps(report, indent=2))
    return code


def _clean_report(settings: Settings, args) -> int:
    from .api import finish_summary, fused_samples, stream_summary
    from .fuse import Fuser

    store, cleaning, config, offsets, room = _context(settings)
    fuser = Fuser(config, offsets, room)
    samples = list(fused_samples(store, fuser, args.from_ms, args.to_ms))
    summary = finish_summary(stream_summary(samples, fuser, offsets), cleaning.summary_extras(config, offsets))
    print(json.dumps({"from": args.from_ms, "to": args.to_ms, "summary": summary}, indent=2))
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

    def window(p, hours: float | None, required: bool = False) -> None:
        p.add_argument("--from", dest="from_ms", type=_time_ms, required=required,
                       help="window start, unix ms or ISO 8601 (local time without an offset)")
        p.add_argument("--to", dest="to_ms", type=_time_ms, required=required,
                       help="window end, unix ms or ISO 8601 (default now)")
        if hours is not None:
            p.add_argument("--hours", type=float, help=f"window length ending at --to (default {hours:g})")

    zones = sub.add_parser("zones-suggest", help="list static single-camera hotspots as zone entries")
    window(zones, 24)
    zones.add_argument("--json", action="store_true", help="print JSON instead of text")
    learn = sub.add_parser("learn-offsets", help="learn the per-camera offset correction")
    window(learn, 72)
    learn.add_argument("--model", choices=("translation", "similarity"), default="translation")
    learn.add_argument("--save", action="store_true", help="write offsets.json (otherwise only print)")
    report = sub.add_parser("clean-report", help="print the fused summary and drop counters")
    window(report, None, required=True)
    return parser


def main(argv: list[str] | None = None, env=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = Settings.from_env(env)
    if args.command == "serve":
        return _serve(settings)
    if args.command == "stats":
        return _stats(settings)
    if args.command == "zones-suggest":
        return _zones_suggest(settings, args)
    if args.command == "learn-offsets":
        return _learn_offsets(settings, args)
    if args.command == "clean-report":
        return _clean_report(settings, args)
    return _wipe(settings, args.yes, args.before)
