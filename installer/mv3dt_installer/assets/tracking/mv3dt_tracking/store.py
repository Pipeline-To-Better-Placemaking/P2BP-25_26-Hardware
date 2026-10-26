"""Per-UTC-day SQLite storage for tracker detections.

One file per UTC day (`YYYY-MM-DD.sqlite`) in the data directory, in WAL
mode so a single writer thread and any number of reader threads can work at
once. Every call opens its own connection, so a `TrackStore` can be shared
between threads freely. Sensor names are mapped to small integer ids per
file (`sensors` table); reads map them back.
"""

from __future__ import annotations

import os
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

_SUFFIX = ".sqlite"
_DAY_NAME = re.compile(r"^\d{4}-\d{2}-\d{2}\.sqlite$")
_SIDECARS = ("-wal", "-shm")
_MEASURES = ("x", "y", "z", "w", "l", "h", "conf")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sensors (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE
);
CREATE TABLE IF NOT EXISTS detections (
    ts_ms INTEGER NOT NULL,
    sensor INTEGER NOT NULL,
    track_id INTEGER NOT NULL,
    x REAL, y REAL, z REAL, w REAL, l REAL, h REAL, conf REAL,
    PRIMARY KEY (ts_ms, sensor, track_id)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS detections_track_ts ON detections (track_id, ts_ms);
"""


@dataclass(frozen=True)
class Detection:
    ts_ms: int
    sensor: str
    track_id: int
    x: float
    y: float
    z: float | None = None
    w: float | None = None
    l: float | None = None  # noqa: E741 (length, named after the schema column)
    h: float | None = None
    conf: float | None = None


def utc_day(ts_ms: int) -> date:
    """The UTC calendar day a millisecond epoch timestamp falls on."""
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).date()


def bucket_ms(hz: float) -> int:
    """Bucket width in whole milliseconds for a sample rate (at least 1 ms).

    Shared with `stitch` so the store and the series agree on bucket edges:
    a bucket starts at `ts_ms - ts_ms % width` (floor, also for negatives).
    """
    if not hz > 0:
        raise ValueError(f"hz must be > 0, got {hz}")
    return max(1, round(1000.0 / hz))


def _is_missing(err: sqlite3.OperationalError, path: Path) -> bool:
    """True when a read failed only because the file or its tables are absent
    (wiped between listing and opening, or created but not yet initialised)."""
    msg = str(err)
    if msg.startswith("no such table"):
        return True
    return msg.startswith("unable to open database file") and not path.exists()


class _SensorNames(dict):
    """Sensor id -> name for one day file. Reads take rows and names from
    one snapshot, so an unknown id should not happen; if it ever does, look
    it up again on a fresh connection, and fall back to the id itself rather
    than failing the whole query."""

    def __init__(self, path: Path, names: Iterable[tuple[int, str]]) -> None:
        super().__init__(names)
        self._path = path

    def __missing__(self, sensor_id: int) -> str:
        fresh = TrackStore._read(self._path, "SELECT id, name FROM sensors", ())
        self.update(fresh or ())
        return self.setdefault(sensor_id, str(sensor_id))


class TrackStore:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)

    # -- files ---------------------------------------------------------------

    def path_for(self, day: date) -> Path:
        return self.data_dir / f"{day.isoformat()}{_SUFFIX}"

    def _day_files(self) -> list[tuple[date, Path]]:
        """Existing day files, oldest first. Unrelated files are ignored."""
        if not self.data_dir.is_dir():
            return []
        found = []
        for p in self.data_dir.iterdir():
            # Only names this store would have produced: date.fromisoformat
            # alone also accepts forms like 20261020 or 2026-W43-1.
            if not _DAY_NAME.match(p.name):
                continue
            try:
                day = date.fromisoformat(p.name[: -len(_SUFFIX)])
            except ValueError:
                continue
            if self.path_for(day).name == p.name and p.is_file():
                found.append((day, p))
        return sorted(found)

    @staticmethod
    def _read(
        path: Path, sql: str, params: tuple, names: bool = False
    ) -> list[tuple] | tuple[list[tuple], _SensorNames] | None:
        """Run a read on an existing day file, opened read-only so a reader
        never creates one. With `names`, also return the sensor id -> name
        map, read in the same transaction so both see one WAL snapshot. None
        if the file or its tables are missing; a lock outlasting
        busy_timeout and any other error raise."""
        try:
            conn = sqlite3.connect(
                f"{path.resolve().as_uri()}?mode=ro", uri=True, timeout=5.0, isolation_level=None
            )
        except sqlite3.OperationalError as e:
            if _is_missing(e, path):
                return None
            raise
        try:
            if not names:
                return conn.execute(sql, params).fetchall()
            # Without one transaction each SELECT gets its own snapshot, and a
            # writer committing a new sensor and its rows in between would
            # leave rows whose sensor id is not in the map.
            conn.execute("BEGIN")
            try:
                rows = conn.execute(sql, params).fetchall()
                sensor_names = _SensorNames(path, conn.execute("SELECT id, name FROM sensors"))
            finally:
                conn.execute("COMMIT")
            return rows, sensor_names
        except sqlite3.OperationalError as e:
            if _is_missing(e, path):
                return None
            raise
        finally:
            conn.close()

    def _open_for_write(self, day: date) -> sqlite3.Connection:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path_for(day), timeout=10.0)
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.executescript(_SCHEMA)
        return conn

    # -- writes --------------------------------------------------------------

    def insert_many(self, rows: Iterable[Detection]) -> int:
        """Insert rows, one transaction per UTC day file. Returns rows written.

        A row with the same (ts_ms, sensor, track_id) as a stored one
        replaces it.
        """
        by_day: dict[date, list[Detection]] = defaultdict(list)
        for r in rows:
            by_day[utc_day(r.ts_ms)].append(r)

        written = 0
        for day, day_rows in sorted(by_day.items()):
            conn = self._open_for_write(day)
            try:
                with conn:
                    ids = self._sensor_ids(conn, {r.sensor for r in day_rows})
                    conn.executemany(
                        "INSERT OR REPLACE INTO detections"
                        " (ts_ms, sensor, track_id, x, y, z, w, l, h, conf)"
                        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        [
                            (int(r.ts_ms), ids[r.sensor], int(r.track_id),
                             r.x, r.y, r.z, r.w, r.l, r.h, r.conf)
                            for r in day_rows
                        ],
                    )
                written += len(day_rows)
            finally:
                conn.close()
        return written

    @staticmethod
    def _sensor_ids(conn: sqlite3.Connection, names: set[str]) -> dict[str, int]:
        conn.executemany(
            "INSERT OR IGNORE INTO sensors (name) VALUES (?)",
            [(n,) for n in sorted(names)],
        )
        return {name: sid for sid, name in conn.execute("SELECT id, name FROM sensors")}

    # -- reads ---------------------------------------------------------------

    def query_range(self, from_ms: int, to_ms: int, hz: float | None = None) -> list[Detection]:
        """Detections with from_ms <= ts_ms <= to_ms, sorted by ts_ms.

        With `hz`, rows are downsampled per (sensor, track_id) into buckets
        of `bucket_ms(hz)` ms aligned to the epoch: measures are averaged
        (NULLs ignored) and ts_ms is the bucket start.
        """
        if from_ms > to_ms:
            raise ValueError(f"from_ms ({from_ms}) is after to_ms ({to_ms})")
        width = None if hz is None else bucket_ms(hz)

        paths = []
        day, last = utc_day(from_ms), utc_day(to_ms)
        while day <= last:
            p = self.path_for(day)
            if p.exists():
                paths.append(p)
            day += timedelta(days=1)

        if width is None:
            out = []
            for p in paths:
                out.extend(self._raw(p, from_ms, to_ms))
        else:
            out = self._downsampled(paths, from_ms, to_ms, width)
        out.sort(key=lambda d: (d.ts_ms, d.sensor, d.track_id))
        return out

    def _raw(self, path: Path, from_ms: int, to_ms: int) -> list[Detection]:
        got = self._read(
            path,
            "SELECT ts_ms, sensor, track_id, x, y, z, w, l, h, conf FROM detections"
            " WHERE ts_ms BETWEEN ? AND ?",
            (from_ms, to_ms),
            names=True,
        )
        if got is None:
            return []
        rows, names = got
        return [Detection(ts, names[s], tid, *rest) for ts, s, tid, *rest in rows]

    def _downsampled(
        self, paths: list[Path], from_ms: int, to_ms: int, width: int
    ) -> list[Detection]:
        # Bucket start in integer SQL arithmetic. SQLite's % truncates toward
        # zero, so ((ts % w) + w) % w is the floor remainder and negative
        # timestamps bucket the same way as Python's ts - ts % w. Grouping
        # with the bucket first and on the integer sensor id keeps the
        # temporary b-tree cheap; names are mapped afterwards.
        start = "(ts_ms - ((ts_ms % ?1) + ?1) % ?1)"
        where = " FROM detections WHERE ts_ms BETWEEN ?2 AND ?3 GROUP BY b, sensor, track_id"
        params = (width, from_ms, to_ms)

        if len(paths) == 1:
            path = paths[0]
            avgs = ", ".join(f"AVG({m})" for m in _MEASURES)
            got = self._read(path, f"SELECT {start} AS b, sensor, track_id, {avgs}{where}", params, True)
            if got is None:
                return []
            rows, names = got
            return [Detection(b, names[s], tid, *rest) for b, s, tid, *rest in rows]

        # Several day files: a bucket can straddle UTC midnight when the width
        # does not divide a day, so sum and count per measure and merge.
        sums = ", ".join(f"SUM({m}), COUNT({m})" for m in _MEASURES)
        acc: dict[tuple[int, str, int], list[float]] = {}
        for path in paths:
            got = self._read(path, f"SELECT {start} AS b, sensor, track_id, {sums}{where}", params, True)
            if got is None:
                continue
            rows, names = got
            for b, s, tid, *vals in rows:
                key = (b, names[s], tid)
                prev = acc.get(key)
                if prev is None:
                    acc[key] = [v or 0 for v in vals]
                else:
                    acc[key] = [a + (v or 0) for a, v in zip(prev, vals)]
        return [
            Detection(b, name, tid, *(
                vals[i] / vals[i + 1] if vals[i + 1] else None
                for i in range(0, 2 * len(_MEASURES), 2)
            ))
            for (b, name, tid), vals in acc.items()
        ]

    def _count_and_bounds(self, path: Path) -> tuple[int, int | None, int | None]:
        rows = self._read(path, "SELECT COUNT(*), MIN(ts_ms), MAX(ts_ms) FROM detections", ())
        return rows[0] if rows else (0, None, None)

    def span(self) -> tuple[int, int] | None:
        """(oldest_ms, newest_ms) over all day files, or None if empty."""
        files = self._day_files()
        oldest = newest = None
        for _, p in files:
            _, lo, _ = self._count_and_bounds(p)
            if lo is not None:
                oldest = lo
                break
        for _, p in reversed(files):
            _, _, hi = self._count_and_bounds(p)
            if hi is not None:
                newest = hi
                break
        if oldest is None or newest is None:
            return None
        return oldest, newest

    def stats(self) -> dict:
        """Per-day file name, row count and bytes (incl. WAL/SHM), plus totals."""
        days = []
        for _, p in self._day_files():
            rows, _, _ = self._count_and_bounds(p)
            size = 0
            for f in (p, *(Path(f"{p}{s}") for s in _SIDECARS)):
                try:
                    size += f.stat().st_size
                except FileNotFoundError:
                    pass
            days.append({"file": p.name, "rows": rows, "bytes": size})
        return {
            "days": days,
            "files": len(days),
            "rows": sum(d["rows"] for d in days),
            "bytes": sum(d["bytes"] for d in days),
        }

    # -- maintenance ---------------------------------------------------------

    def wipe(self, before: date | None = None) -> int:
        """Delete every day file, or only days strictly before `before`.

        Returns the number of day files removed. WAL/SHM sidecars go too.
        """
        removed = 0
        for day, p in self._day_files():
            if before is not None and not day < before:
                continue
            for f in (p, *(Path(f"{p}{s}") for s in _SIDECARS)):
                try:
                    os.remove(f)
                    if f == p:
                        removed += 1
                except FileNotFoundError:
                    pass
        return removed
