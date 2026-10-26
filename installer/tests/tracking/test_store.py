"""Tests for `mv3dt_tracking.store` (per-UTC-day SQLite detection store).

Run from installer/: `python3 -m pytest tests/tracking/test_store.py -v`
"""

from __future__ import annotations

import sqlite3
import threading
import time
from datetime import date, datetime, timezone

import pytest

from mv3dt_tracking.store import Detection, TrackStore, bucket_ms


def _ms(y, mo, d, h=0, mi=0, s=0, ms=0) -> int:
    dt = datetime(y, mo, d, h, mi, s, tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000) + ms


NOON = _ms(2026, 10, 20, 12)
MIDNIGHT = _ms(2026, 10, 21)


def _det(ts, sensor="cam1", tid=1, x=1.0, y=2.0, **kw) -> Detection:
    return Detection(ts, sensor, tid, x, y, **kw)


@pytest.fixture
def store(tmp_path):
    return TrackStore(tmp_path / "data")


def test_round_trip_preserves_every_field(store):
    rows = [
        _det(NOON, "cam1", 7, 1.5, -2.25, z=0.0, w=0.5, l=0.4, h=1.8, conf=0.9),
        _det(NOON + 100, "cam2", 8, 3.0, 4.0),
    ]
    assert store.insert_many(rows) == 2
    assert store.query_range(NOON, NOON + 100) == rows
    assert (store.data_dir / "2026-10-20.sqlite").exists()


def test_file_is_wal_clustered_by_time_with_track_index(store):
    store.insert_many([_det(NOON)])
    conn = sqlite3.connect(store.data_dir / "2026-10-20.sqlite")
    try:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        cols = [r[1] for r in conn.execute("PRAGMA table_info(detections)")]
        assert cols == ["ts_ms", "sensor", "track_id", "x", "y", "z", "w", "l", "h", "conf"]
        pk = sorted((r[5], r[1]) for r in conn.execute("PRAGMA table_info(detections)") if r[5])
        assert [name for _, name in pk] == ["ts_ms", "sensor", "track_id"]
        ddl = conn.execute("SELECT sql FROM sqlite_master WHERE name = 'detections'").fetchone()[0]
        assert "WITHOUT ROWID" in ddl
        indexed = {
            tuple(c[2] for c in conn.execute(f"PRAGMA index_info({name!r})"))
            for _, name, _, origin, *_ in conn.execute("PRAGMA index_list(detections)")
            if origin == "c"
        }
        assert indexed == {("track_id", "ts_ms")}
    finally:
        conn.close()


def test_sensor_names_are_mapped_to_ids_per_file(store):
    store.insert_many([_det(NOON, "camA"), _det(NOON + 1, "camB"), _det(NOON + 2, "camA")])
    store.insert_many([_det(NOON + 3, "camB")])
    conn = sqlite3.connect(store.data_dir / "2026-10-20.sqlite")
    try:
        sensors = dict(conn.execute("SELECT name, id FROM sensors"))
        assert set(sensors) == {"camA", "camB"}
        stored = [r[0] for r in conn.execute("SELECT sensor FROM detections ORDER BY ts_ms")]
    finally:
        conn.close()
    assert stored == [sensors["camA"], sensors["camB"], sensors["camA"], sensors["camB"]]
    assert [d.sensor for d in store.query_range(NOON, NOON + 3)] == ["camA", "camB", "camA", "camB"]


def test_range_spans_utc_midnight_across_two_files(store):
    before = _det(MIDNIGHT - 500, tid=1)
    after = _det(MIDNIGHT + 500, tid=2)
    outside = _det(MIDNIGHT + 5000, tid=3)
    # Insert out of order: sorting is the query's job.
    store.insert_many([after, outside, before])
    names = sorted(p.name for p in store.data_dir.iterdir() if p.suffix == ".sqlite")
    assert names == ["2026-10-20.sqlite", "2026-10-21.sqlite"]
    assert store.query_range(MIDNIGHT - 1000, MIDNIGHT + 1000) == [before, after]


def test_range_bounds_are_inclusive_and_missing_days_skipped(store):
    store.insert_many([_det(NOON), _det(NOON + 1000)])
    assert len(store.query_range(NOON, NOON + 1000)) == 2
    assert store.query_range(NOON + 1, NOON + 999) == []
    # A week-long range where only one day file exists.
    assert len(store.query_range(NOON - 3 * 86_400_000, NOON + 3 * 86_400_000)) == 2
    assert TrackStore(store.data_dir / "nope").query_range(NOON, NOON + 1) == []


def test_downsampling_averages_per_sensor_track_and_bucket(store):
    # hz=5 -> 200 ms buckets aligned to the epoch. NOON is a multiple of 200.
    store.insert_many([
        _det(NOON + 10, "cam1", 1, 1.0, 10.0, conf=0.5),
        _det(NOON + 190, "cam1", 1, 3.0, 20.0, conf=None),
        _det(NOON + 200, "cam1", 1, 5.0, 30.0),       # next bucket
        _det(NOON + 50, "cam2", 1, 100.0, 100.0),     # other sensor, own row
        _det(NOON + 60, "cam1", 2, 7.0, 7.0),         # other track, own row
    ])
    out = store.query_range(NOON, NOON + 1000, hz=5)
    got = {(d.sensor, d.track_id, d.ts_ms): d for d in out}
    assert set(got) == {
        ("cam1", 1, NOON), ("cam1", 1, NOON + 200), ("cam2", 1, NOON), ("cam1", 2, NOON),
    }
    first = got[("cam1", 1, NOON)]
    assert (first.x, first.y) == (2.0, 15.0)
    assert first.conf == 0.5  # NULLs are ignored, as AVG would
    assert first.z is None
    assert got[("cam1", 1, NOON + 200)].x == 5.0
    assert [d.ts_ms for d in out] == sorted(d.ts_ms for d in out)


def test_downsampling_merges_a_bucket_straddling_midnight(store):
    # hz=1/7 -> 7 s buckets, which do not divide a day, so one bucket holds
    # samples from both day files.
    hz = 1 / 7
    width = bucket_ms(hz)
    assert width == 7000
    start = MIDNIGHT - MIDNIGHT % width
    assert start < MIDNIGHT - 1 and MIDNIGHT + 1 < start + width
    store.insert_many([_det(MIDNIGHT - 1, x=1.0), _det(MIDNIGHT + 1, x=3.0)])
    (row,) = store.query_range(MIDNIGHT - 1, MIDNIGHT + 1, hz=hz)
    assert row.x == 2.0
    assert row.ts_ms == start


def test_span_and_stats(store):
    assert store.span() is None
    assert store.stats() == {"days": [], "files": 0, "rows": 0, "bytes": 0}
    store.insert_many([_det(MIDNIGHT - 10), _det(MIDNIGHT - 5), _det(MIDNIGHT + 20)])
    assert store.span() == (MIDNIGHT - 10, MIDNIGHT + 20)
    st = store.stats()
    assert [(d["file"], d["rows"]) for d in st["days"]] == [
        ("2026-10-20.sqlite", 2), ("2026-10-21.sqlite", 1),
    ]
    assert all(d["bytes"] > 0 for d in st["days"])
    assert st["files"] == 2 and st["rows"] == 3
    assert st["bytes"] == sum(d["bytes"] for d in st["days"])


def test_wipe_all_removes_day_files_and_sidecars(store):
    store.insert_many([_det(NOON), _det(MIDNIGHT + 1)])
    (store.data_dir / "2026-10-20.sqlite-wal").touch()
    (store.data_dir / "notes.txt").write_text("keep me")
    assert store.wipe() == 2
    assert sorted(p.name for p in store.data_dir.iterdir()) == ["notes.txt"]
    assert store.wipe() == 0
    assert TrackStore(store.data_dir / "missing").wipe() == 0


def test_wipe_before_date_is_strict(store):
    days = [_ms(2026, 10, d, 12) for d in (18, 19, 20)]
    store.insert_many([_det(t) for t in days])
    assert store.wipe(before=date(2026, 10, 19)) == 1
    assert [d["file"] for d in store.stats()["days"]] == ["2026-10-19.sqlite", "2026-10-20.sqlite"]
    # Recording resumes after a wipe.
    store.wipe()
    store.insert_many([_det(NOON)])
    assert store.stats()["rows"] == 1


@pytest.mark.parametrize("args", [(NOON + 1, NOON), (NOON, NOON + 1, 0), (NOON, NOON + 1, -2.0)])
def test_invalid_query_raises_value_error(store, args):
    with pytest.raises(ValueError):
        store.query_range(*args)


def test_concurrent_writer_and_reader(store):
    errors: list[BaseException] = []
    done = threading.Event()
    seen = []

    def writer():
        try:
            for i in range(50):
                store.insert_many([_det(NOON + i * 10 + k, tid=k) for k in range(20)])
        except BaseException as e:  # noqa: BLE001
            errors.append(e)
        finally:
            done.set()

    def reader():
        try:
            while not done.is_set():
                seen.append(len(store.query_range(NOON, NOON + 10_000, hz=10)))
                store.stats()
                store.span()
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert not errors
    assert store.stats()["rows"] == 1000
    assert seen == sorted(seen)  # readers only ever see committed, growing data


def test_duplicate_key_replaces_the_stored_row(store):
    store.insert_many([_det(NOON, x=1.0)])
    store.insert_many([_det(NOON, x=9.0), _det(NOON, "cam2", x=5.0)])
    assert [(d.sensor, d.x) for d in store.query_range(NOON, NOON)] == [("cam1", 9.0), ("cam2", 5.0)]


def test_negative_timestamps_bucket_by_floor(store):
    # -150 lies in [-200, 0) and 50 in [0, 200): truncation would merge them.
    store.insert_many([_det(-150, x=1.0), _det(50, x=3.0)])
    out = store.query_range(-1000, 1000, hz=5)
    assert [(d.ts_ms, d.x) for d in out] == [(-200, 1.0), (0, 3.0)]


@pytest.mark.parametrize("hz, width", [(5, 200), (3, 333), (0.5, 2000), (5000, 1)])
def test_bucket_width_is_whole_milliseconds(hz, width):
    assert bucket_ms(hz) == width


def test_decoy_files_are_never_touched(store):
    store.insert_many([_det(NOON)])
    d = store.data_dir
    decoys = ["20261020.sqlite", "2026-W43-1.sqlite", "notes.sqlite", "other.txt", "2026-10-20.sqlite.bak"]
    for name in decoys:
        conn = sqlite3.connect(d / name) if name.endswith(".sqlite") else None
        if conn:
            conn.execute("CREATE TABLE detections (ts_ms INTEGER)")
            conn.execute("INSERT INTO detections VALUES (1)")
            conn.commit()
            conn.close()
        else:
            (d / name).write_text("x")
    assert store.span() == (NOON, NOON)
    st = store.stats()
    assert [x["file"] for x in st["days"]] == ["2026-10-20.sqlite"] and st["rows"] == 1
    assert store.wipe() == 1
    assert sorted(p.name for p in d.iterdir()) == sorted(decoys)


def test_readers_never_create_files(store):
    store.data_dir.mkdir()
    assert store.query_range(NOON, NOON + 1) == []
    assert store.query_range(NOON, NOON + 1, hz=5) == []
    assert store.span() is None
    assert list(store.data_dir.iterdir()) == []


def test_reader_treats_a_vanished_file_as_empty(store, monkeypatch):
    store.insert_many([_det(NOON)])
    path = store.path_for(date(2026, 10, 20))
    # The file is listed, then wiped before the reader opens it.
    monkeypatch.setattr(TrackStore, "_day_files", lambda self: [(date(2026, 10, 20), path)])
    for f in path.parent.iterdir():
        f.unlink()
    assert store.span() is None
    assert store.stats()["rows"] == 0
    assert not path.exists()


def test_reader_raises_on_errors_other_than_missing(store):
    store.data_dir.mkdir()
    store.path_for(date(2026, 10, 20)).write_bytes(b"not a database" * 100)
    with pytest.raises(sqlite3.DatabaseError):
        store.query_range(NOON, NOON + 1)
    with pytest.raises(sqlite3.DatabaseError):
        store.stats()


def test_twenty_minute_query_at_5hz_is_fast(store, capsys):
    # 2 sensors x 3 tracks x 22 Hz x 20 min = 158,400 rows in one day file.
    start = NOON - 600_000
    rows = [
        Detection(start + i * 1000 // 22, s, t, float(i % 50), float(t), 0.0, 0.5, 0.5, 1.7, 0.9)
        for i in range(22 * 60 * 20)
        for s in ("cam1", "cam2")
        for t in (1, 2, 3)
    ]
    assert store.insert_many(rows) == 158_400
    best = float("inf")
    for _ in range(3):
        t0 = time.perf_counter()
        out = store.query_range(start, start + 1_200_000 - 1, hz=5)
        best = min(best, time.perf_counter() - t0)
    assert len(out) == 2 * 3 * 5 * 60 * 20
    with capsys.disabled():
        print(f"\n20 min @ 5 Hz over 158,400 rows: {best * 1000:.0f} ms")
    # Generous bound so a loaded CI runner does not flake; target is < 300 ms.
    assert best < 1.5
