"""Per-UTC-day SQLite storage for tracker detections.

One file per UTC day (`YYYY-MM-DD.sqlite`) in the data directory, in WAL
mode so a single writer thread and any number of reader threads can work at
once. Every call opens its own connection, so a `TrackStore` can be shared
between threads freely. Sensor names are mapped to small integer ids per
file (`sensors` table); reads map them back.
"""

from __future__ import annotations

import os
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

_SUFFIX = ".sqlite"
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
    x REAL, y REAL, z REAL, w REAL, l REAL, h REAL, conf REAL
);
CREATE INDEX IF NOT EXISTS detections_ts ON detections (ts_ms);
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
        for p in self.data_dir.glob(f"*{_SUFFIX}"):
            try:
                day = date.fromisoformat(p.name[: -len(_SUFFIX)])
            except ValueError:
                continue
            found.append((day, p))
        return sorted(found)

    def _connect(self, path: Path) -> sqlite3.Connection:
        conn = sqlite3.connect(path, timeout=10.0)
        conn.execute("PRAGMA busy_timeout = 10000")
        return conn

    def _open_for_write(self, day: date) -> sqlite3.Connection:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        conn = self._connect(self.path_for(day))
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.executescript(_SCHEMA)
        return conn

    # -- writes --------------------------------------------------------------

    def insert_many(self, rows: Iterable[Detection]) -> int:
        """Insert rows, one transaction per UTC day file. Returns rows written."""
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
                        "INSERT INTO detections"
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

        With `hz`, rows are downsampled per (sensor, track_id) into buckets of
        1000/hz ms aligned to the epoch: measures are averaged and ts_ms is
        the bucket start.
        """
        if from_ms > to_ms:
            raise ValueError(f"from_ms ({from_ms}) is after to_ms ({to_ms})")
        if hz is not None and not hz > 0:
            raise ValueError(f"hz must be > 0, got {hz}")

        paths = []
        day, last = utc_day(from_ms), utc_day(to_ms)
        while day <= last:
            p = self.path_for(day)
            if p.exists():
                paths.append(p)
            day += timedelta(days=1)

        if hz is None:
            out = []
            for p in paths:
                out.extend(self._raw(p, from_ms, to_ms))
        else:
            out = self._downsampled(paths, from_ms, to_ms, 1000.0 / hz)
        out.sort(key=lambda d: (d.ts_ms, d.sensor, d.track_id))
        return out

    def _raw(self, path: Path, from_ms: int, to_ms: int) -> list[Detection]:
        conn = self._connect(path)
        try:
            cur = conn.execute(
                "SELECT d.ts_ms, s.name, d.track_id, d.x, d.y, d.z, d.w, d.l, d.h, d.conf"
                " FROM detections d JOIN sensors s ON s.id = d.sensor"
                " WHERE d.ts_ms BETWEEN ? AND ?",
                (from_ms, to_ms),
            )
            return [Detection(*row) for row in cur]
        except sqlite3.OperationalError:
            # A file created but not yet initialised by the writer.
            return []
        finally:
            conn.close()

    def _downsampled(
        self, paths: list[Path], from_ms: int, to_ms: int, bucket_ms: float
    ) -> list[Detection]:
        # SQL returns SUM and COUNT per measure rather than AVG so that a
        # bucket straddling UTC midnight (bucket_ms need not divide a day)
        # can be merged exactly across the two day files.
        sums = ", ".join(f"SUM(d.{m}), COUNT(d.{m})" for m in _MEASURES)
        sql = (
            f"SELECT s.name, d.track_id, CAST(d.ts_ms / ? AS INTEGER) AS b, {sums}"
            " FROM detections d JOIN sensors s ON s.id = d.sensor"
            " WHERE d.ts_ms BETWEEN ? AND ?"
            " GROUP BY d.sensor, d.track_id, b"
        )
        acc: dict[tuple[str, int, int], list[float]] = {}
        for p in paths:
            conn = self._connect(p)
            try:
                rows = conn.execute(sql, (bucket_ms, from_ms, to_ms)).fetchall()
            except sqlite3.OperationalError:
                rows = []
            finally:
                conn.close()
            for name, track_id, b, *vals in rows:
                key = (name, track_id, b)
                vals = [v or 0 for v in vals]
                if key in acc:
                    acc[key] = [a + v for a, v in zip(acc[key], vals)]
                else:
                    acc[key] = vals

        out = []
        for (name, track_id, b), vals in acc.items():
            means = [
                (vals[2 * i] / vals[2 * i + 1]) if vals[2 * i + 1] else None
                for i in range(len(_MEASURES))
            ]
            out.append(Detection(int(b * bucket_ms), name, track_id, *means))
        return out

    def _count_and_bounds(self, path: Path) -> tuple[int, int | None, int | None]:
        conn = self._connect(path)
        try:
            n, lo, hi = conn.execute(
                "SELECT COUNT(*), MIN(ts_ms), MAX(ts_ms) FROM detections"
            ).fetchone()
            return n, lo, hi
        except sqlite3.OperationalError:
            return 0, None, None
        finally:
            conn.close()

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
