"""Decode one DeepStream ``nv.Frame`` protobuf payload into detection rows.

The pipeline publishes one ``nv.Frame`` per camera frame to
``mv3dt/<LOCATION_ID>/sv3d``. Rather than depend on the protobuf package and a
generated ``schema_pb2.py``, this module reads the protobuf wire format
directly for the handful of fields the recorder needs, and skips everything
else. Field numbers follow DeepStream's ``nvmsgconv_mega`` ``schema.proto``:

- ``Frame``: id=2 (string), timestamp=3 (``google.protobuf.Timestamp``:
  seconds=1, nanos=2), sensorId=4 (string), objects=5 (repeated ``Object``)
- ``Object``: id=1 (string tracker id), confidence=4 (float),
  info=5 (``map<string,string>``: key=1, value=2),
  bbox3d=14 (``Bbox3d``: coordinates=1, packed repeated double, ordered
  x, y, z, width, length, height, ...)

Row mapping: ``x``/``y`` come from ``info["footLocation3D"]`` ("x,y" in
metres), falling back to bbox3d x/y; ``z``/``w``/``l``/``h`` come from bbox3d
coordinates 2..5 when present. Objects whose id is not numeric are skipped.
"""

from __future__ import annotations

import struct

__all__ = ["DecodeError", "decode_frame"]

# Wire types.
_VARINT = 0
_I64 = 1
_LEN = 2
_I32 = 5


class DecodeError(ValueError):
    """The payload is not a well-formed ``nv.Frame``."""


def _read_varint(buf: bytes, pos: int, end: int) -> tuple[int, int]:
    result = 0
    shift = 0
    while True:
        if pos >= end:
            raise DecodeError("truncated varint")
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, pos
        shift += 7
        if shift >= 70:
            raise DecodeError("varint too long")


def _fields(buf: bytes, start: int = 0, end: int | None = None):
    """Yield ``(field_number, wire_type, value)`` for one message.

    ``value`` is an int for varints, raw bytes for 64-bit/32-bit fields, and
    a ``(start, end)`` slice tuple for length-delimited fields.
    """
    if end is None:
        end = len(buf)
    pos = start
    while pos < end:
        key, pos = _read_varint(buf, pos, end)
        field, wire = key >> 3, key & 0x7
        if field == 0:
            raise DecodeError("field number 0")
        if wire == _VARINT:
            value, pos = _read_varint(buf, pos, end)
        elif wire == _I64:
            if pos + 8 > end:
                raise DecodeError("truncated 64-bit field")
            value = buf[pos:pos + 8]
            pos += 8
        elif wire == _LEN:
            length, pos = _read_varint(buf, pos, end)
            if pos + length > end:
                raise DecodeError("truncated length-delimited field")
            value = (pos, pos + length)
            pos += length
        elif wire == _I32:
            if pos + 4 > end:
                raise DecodeError("truncated 32-bit field")
            value = buf[pos:pos + 4]
            pos += 4
        else:
            raise DecodeError(f"unsupported wire type {wire}")
        yield field, wire, value


def _string(buf: bytes, span: tuple[int, int]) -> str:
    try:
        return buf[span[0]:span[1]].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DecodeError("invalid utf-8 string") from exc


def _signed64(v: int) -> int:
    v &= (1 << 64) - 1
    return v - (1 << 64) if v >> 63 else v


def _timestamp_ms(buf: bytes, span: tuple[int, int]) -> int:
    seconds = 0
    nanos = 0
    for field, wire, value in _fields(buf, *span):
        if field == 1 and wire == _VARINT:
            seconds = _signed64(value)
        elif field == 2 and wire == _VARINT:
            nanos = _signed64(value)
    return seconds * 1000 + nanos // 1_000_000


def _map_entry(buf: bytes, span: tuple[int, int]) -> tuple[str, str]:
    key = ""
    val = ""
    for field, wire, value in _fields(buf, *span):
        if field == 1 and wire == _LEN:
            key = _string(buf, value)
        elif field == 2 and wire == _LEN:
            val = _string(buf, value)
    return key, val


def _bbox3d_coords(buf: bytes, span: tuple[int, int]) -> list[float]:
    coords: list[float] = []
    for field, wire, value in _fields(buf, *span):
        if field != 1:
            continue
        if wire == _LEN:  # packed
            s, e = value
            if (e - s) % 8:
                raise DecodeError("packed doubles length not a multiple of 8")
            coords.extend(struct.unpack_from(f"<{(e - s) // 8}d", buf, s))
        elif wire == _I64:  # unpacked element
            coords.append(struct.unpack("<d", value)[0])
    return coords


def _foot_xy(text: str) -> tuple[float, float] | None:
    parts = text.split(",")
    if len(parts) < 2:
        return None
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        return None


def _object_row(buf: bytes, span: tuple[int, int]) -> dict | None:
    obj_id = ""
    conf = None
    info: dict[str, str] = {}
    coords: list[float] = []
    for field, wire, value in _fields(buf, *span):
        if field == 1 and wire == _LEN:
            obj_id = _string(buf, value)
        elif field == 4 and wire == _I32:
            conf = struct.unpack("<f", value)[0]
        elif field == 5 and wire == _LEN:
            k, v = _map_entry(buf, value)
            info[k] = v
        elif field == 14 and wire == _LEN:
            coords = _bbox3d_coords(buf, value)

    try:
        track_id = int(obj_id)
    except ValueError:
        return None

    xy = _foot_xy(info.get("footLocation3D", ""))
    if xy is None and len(coords) >= 2:
        xy = (coords[0], coords[1])
    x, y = xy if xy is not None else (None, None)

    def coord(i: int) -> float | None:
        return coords[i] if len(coords) > i else None

    return {
        "track_id": track_id,
        "x": x,
        "y": y,
        "z": coord(2),
        "w": coord(3),
        "l": coord(4),
        "h": coord(5),
        "conf": conf,
    }


def decode_frame(payload: bytes, recv_ms: int | None = None) -> list[dict]:
    """Decode one ``nv.Frame`` payload into a list of row dicts.

    Each row has keys ``ts_ms, sensor, track_id, x, y, z, w, l, h, conf``.
    ``ts_ms`` is the frame timestamp, or ``recv_ms`` when the frame carries
    none. Raises :class:`DecodeError` on malformed input.
    """
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise DecodeError("payload must be bytes")
    buf = bytes(payload)
    ts_ms = None
    sensor = ""
    objects: list[dict] = []
    for field, wire, value in _fields(buf):
        if field == 3 and wire == _LEN:
            ts_ms = _timestamp_ms(buf, value)
        elif field == 4 and wire == _LEN:
            sensor = _string(buf, value)
        elif field == 5 and wire == _LEN:
            row = _object_row(buf, value)
            if row is not None:
                objects.append(row)

    if ts_ms is None:
        ts_ms = recv_ms
    return [{"ts_ms": ts_ms, "sensor": sensor, **row} for row in objects]
