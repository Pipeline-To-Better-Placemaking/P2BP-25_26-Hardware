"""Tests for mv3dt_tracking.decode (STEP-8, plan Part A).

Run from installer/: `python3 -m pytest tests/tracking/test_decode.py -v`
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "mv3dt_installer" / "assets" / "tracking"))

from mv3dt_tracking.decode import DecodeError, decode_frame  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "frame_camera1.hex"


def _fixture() -> bytes:
    return bytes.fromhex(FIXTURE.read_text().strip())


# -- minimal protobuf encoders for synthetic frames --------------------------


def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _key(field: int, wire: int) -> bytes:
    return _varint(field << 3 | wire)


def _len(field: int, data: bytes) -> bytes:
    return _key(field, 2) + _varint(len(data)) + data


def _str(field: int, text: str) -> bytes:
    return _len(field, text.encode())


def _object(obj_id: str, *, foot: str | None = None, coords=None, conf: float = 0.5) -> bytes:
    body = _str(1, obj_id) + _key(4, 5) + struct.pack("<f", conf)
    if foot is not None:
        body += _len(5, _str(1, "footLocation3D") + _str(2, foot))
    if coords is not None:
        body += _len(14, _len(1, struct.pack(f"<{len(coords)}d", *coords)))
    return _len(5, body)


def _frame(*objects: bytes, seconds: int | None = 100, nanos: int = 250_000_000) -> bytes:
    body = _str(1, "4.0") + _str(2, "1")
    if seconds is not None:
        body += _len(3, _key(1, 0) + _varint(seconds) + _key(2, 0) + _varint(nanos))
    body += _str(4, "camera2")
    return body + b"".join(objects)


# -- tests ------------------------------------------------------------------


def test_fixture_decodes_to_expected_row():
    rows = decode_frame(_fixture(), recv_ms=1)
    assert len(rows) == 1
    row = rows[0]
    assert set(row) == {"ts_ms", "sensor", "track_id", "x", "y", "z", "w", "l", "h", "conf"}
    assert row["ts_ms"] == 1792982719133
    assert row["sensor"] == "camera1"
    assert row["track_id"] == 128
    assert row["x"] == pytest.approx(1.860936, abs=1e-9)
    assert row["y"] == pytest.approx(-4.363389, abs=1e-9)
    assert row["z"] == pytest.approx(0.595, abs=1e-6)
    assert row["w"] == pytest.approx(0.6, abs=1e-6)
    assert row["l"] == pytest.approx(0.6, abs=1e-6)
    assert row["h"] == pytest.approx(1.19, abs=1e-6)
    assert row["conf"] == pytest.approx(0.393257946, abs=1e-6)


def test_synthetic_frame_timestamp_and_sensor():
    rows = decode_frame(_frame(_object("7", foot="1.5,2.5")))
    assert rows[0]["ts_ms"] == 100_250
    assert rows[0]["sensor"] == "camera2"
    assert (rows[0]["x"], rows[0]["y"]) == (1.5, 2.5)
    assert rows[0]["z"] is None and rows[0]["h"] is None


def test_missing_timestamp_falls_back_to_recv_ms():
    rows = decode_frame(_frame(_object("7", foot="0,0"), seconds=None), recv_ms=123456)
    assert rows[0]["ts_ms"] == 123456


def test_non_numeric_object_id_is_skipped():
    rows = decode_frame(_frame(_object("abc", foot="1,1"), _object("9", foot="2,2")))
    assert [r["track_id"] for r in rows] == [9]


def test_missing_foot_location_falls_back_to_bbox3d():
    rows = decode_frame(_frame(_object("3", coords=[4.0, -5.0, 0.9, 0.5, 0.6, 1.8])))
    row = rows[0]
    assert (row["x"], row["y"], row["z"], row["w"], row["l"], row["h"]) == (4.0, -5.0, 0.9, 0.5, 0.6, 1.8)


def test_malformed_foot_location_falls_back_to_bbox3d():
    rows = decode_frame(_frame(_object("3", foot="", coords=[4.0, -5.0])))
    assert (rows[0]["x"], rows[0]["y"], rows[0]["z"]) == (4.0, -5.0, None)


def test_frame_with_zero_objects_returns_empty_list():
    assert decode_frame(_frame()) == []


def test_unknown_fields_are_skipped():
    extra = (
        _key(99, 0) + _varint(300)
        + _key(100, 1) + b"\x00" * 8
        + _str(101, "unknown")
        + _key(102, 5) + b"\x00" * 4
    )
    assert decode_frame(_fixture() + extra) == decode_frame(_fixture())


@pytest.mark.parametrize(
    "payload",
    [
        _fixture()[:-5],  # truncated inside the last object
        b"\x0a\xff",  # length runs past the end
        b"\xff\xff\xff",  # unterminated varint
        b"\x0b\x00",  # wire type 3 (group) unsupported
        b"\x00\x00",  # field number 0
    ],
)
def test_truncated_or_garbage_payload_raises(payload):
    with pytest.raises(DecodeError):
        decode_frame(payload)


def test_non_bytes_payload_raises():
    with pytest.raises(DecodeError):
        decode_frame("not bytes")  # type: ignore[arg-type]
