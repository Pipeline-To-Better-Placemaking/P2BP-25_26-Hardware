"""Tests for mv3dt_tracking.recorder (STEP-8, plan Part A).

No broker: a fake client stands in for paho, and messages are delivered by
calling the recorder's on_message handler directly.

Run from installer/: `python3 -m pytest tests/tracking/test_recorder.py -v`
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "mv3dt_installer" / "assets" / "tracking"))

from mv3dt_tracking.recorder import Recorder  # noqa: E402

PAYLOAD = bytes.fromhex((Path(__file__).parent / "fixtures" / "frame_camera1.hex").read_text().strip())


class FakeClient:
    def __init__(self):
        self.subscribed = []
        self.connected_to = None
        self.loop_running = False

    def reconnect_delay_set(self, min_delay, max_delay):
        pass

    def connect_async(self, host, port, keepalive=60):
        self.connected_to = (host, port)

    def loop_start(self):
        self.loop_running = True

    def loop_stop(self):
        self.loop_running = False

    def disconnect(self):
        pass

    def subscribe(self, topic, qos=0):
        self.subscribed.append(topic)

    def deliver(self, payload: bytes):
        self.on_message(self, None, SimpleNamespace(topic="mv3dt/test/sv3d", payload=payload))


class CollectingSink:
    def __init__(self, fail_times: int = 0):
        self.batches: list[list[dict]] = []
        self.fail_times = fail_times
        self.calls = 0
        self.threads: set[str] = set()
        self.event = threading.Event()

    def __call__(self, rows):
        self.calls += 1
        self.threads.add(threading.current_thread().name)
        if self.calls <= self.fail_times:
            raise RuntimeError("sink down")
        self.batches.append(list(rows))
        self.event.set()


def _wait(pred, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.01)
    return False


@pytest.fixture
def client():
    return FakeClient()


def _recorder(sink, client, **kw):
    return Recorder(sink, "mv3dt/test/sv3d", "127.0.0.1", 1883, client_factory=lambda: client, **kw)


def test_start_connects_and_subscribes_on_connect(client):
    rec = _recorder(CollectingSink(), client)
    rec.start()
    try:
        assert client.connected_to == ("127.0.0.1", 1883)
        assert client.loop_running
        client.on_connect(client, None, {}, SimpleNamespace(is_failure=False), None)
        assert client.subscribed == ["mv3dt/test/sv3d"]
        assert rec.stats()["connected"]
    finally:
        rec.stop()
    assert not client.loop_running


def test_flush_by_interval(client):
    sink = CollectingSink()
    rec = _recorder(sink, client, flush_interval_s=0.05, max_batch=10_000)
    rec.start()
    try:
        client.deliver(PAYLOAD)
        assert sink.event.wait(2.0)
        assert len(sink.batches) == 1 and sink.batches[0][0]["track_id"] == 128
        assert sink.threads == {"mv3dt-recorder-flush"}
    finally:
        rec.stop()
    s = rec.stats()
    assert s["messages"] == 1 and s["rows"] == 1 and s["rows_flushed"] == 1
    assert s["last_message_ms"] is not None


def test_flush_by_max_batch(client):
    sink = CollectingSink()
    rec = _recorder(sink, client, flush_interval_s=60.0, max_batch=3)
    rec.start()
    try:
        for _ in range(3):
            client.deliver(PAYLOAD)
        assert sink.event.wait(2.0), "max_batch should wake the flusher before the interval"
        assert sum(len(b) for b in sink.batches) == 3
    finally:
        rec.stop()


def test_stop_flushes_remaining_rows(client):
    sink = CollectingSink()
    rec = _recorder(sink, client, flush_interval_s=60.0, max_batch=10_000)
    rec.start()
    client.deliver(PAYLOAD)
    client.deliver(PAYLOAD)
    rec.stop()
    assert sum(len(b) for b in sink.batches) == 2
    assert rec.stats()["rows_buffered"] == 0


def test_decode_error_is_counted_and_not_fatal(client):
    sink = CollectingSink()
    rec = _recorder(sink, client, flush_interval_s=0.05)
    rec.start()
    try:
        client.deliver(b"\xff\xff\xff")
        client.deliver(PAYLOAD)
        assert sink.event.wait(2.0)
    finally:
        rec.stop()
    s = rec.stats()
    assert s["decode_errors"] == 1
    assert s["messages"] == 2
    assert s["rows_flushed"] == 1


def test_sink_exception_counted_and_recorder_keeps_going(client):
    sink = CollectingSink(fail_times=1)
    rec = _recorder(sink, client, flush_interval_s=0.05)
    rec.start()
    try:
        client.deliver(PAYLOAD)
        assert _wait(lambda: rec.stats()["sink_errors"] == 1)
        client.deliver(PAYLOAD)
        assert sink.event.wait(2.0)
    finally:
        rec.stop()
    s = rec.stats()
    assert s["sink_errors"] == 1
    assert s["rows_dropped"] == 1
    assert s["rows_flushed"] == 1


def test_stop_without_start_is_safe():
    rec = Recorder(CollectingSink(), "t")
    rec.stop()
    assert rec.stats()["messages"] == 0
