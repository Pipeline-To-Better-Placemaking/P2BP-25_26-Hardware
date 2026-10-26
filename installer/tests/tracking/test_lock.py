"""Tests for `mv3dt_tracking.lock` (the in-memory data-access lease).

Run from installer/: `python3 -m pytest tests/tracking/test_lock.py -v`
"""

from __future__ import annotations

import threading

import pytest

from mv3dt_tracking.lock import Busy, LeaseManager, Locked, NoLease, NotHolder, RateLimited


class Clock:
    def __init__(self, t: float = 1_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def mgr(clock):
    return LeaseManager(clock=clock)


def test_acquire_then_second_holder_is_locked(mgr, clock):
    lease = mgr.acquire("alice", "Alice", "replay")
    assert lease.kind == "replay"
    state = mgr.acquire("bob", "Bob", "export")
    assert state == Locked("Alice", clock.t + 45)


def test_same_holder_reacquire_is_idempotent(mgr, clock):
    first = mgr.acquire("alice", "Alice", "replay")
    clock.t += 30
    again = mgr.acquire("alice", "Alice", "export")
    assert again.id == first.id
    assert again.kind == "export"
    assert mgr.expires_at(again) == clock.t + 45


def test_unknown_kind_is_rejected(mgr):
    with pytest.raises(ValueError):
        mgr.acquire("alice", "Alice", "upload")


def test_heartbeat_extends_and_lease_expires_45s_after_last(mgr, clock):
    lease = mgr.acquire("alice", "Alice", "replay")
    clock.t += 40
    assert mgr.heartbeat(lease.id, "alice") is lease
    clock.t += 44.9
    assert mgr.current() is lease
    clock.t += 0.1
    assert mgr.current() is None
    assert mgr.heartbeat(lease.id, "alice") is None
    assert not isinstance(mgr.acquire("bob", "Bob", "replay"), Locked)


def test_hard_cap_is_15_minutes_despite_heartbeats(mgr, clock):
    lease = mgr.acquire("alice", "Alice", "replay")
    start = clock.t
    while clock.t + 15 < start + 900:
        clock.t += 15
        assert mgr.heartbeat(lease.id, "alice") is lease
    assert mgr.expires_at(lease) == start + 900
    clock.t = start + 900
    assert mgr.current() is None
    fresh = mgr.acquire("alice", "Alice", "replay")
    assert fresh.id != lease.id


def test_heartbeat_by_other_holder_or_stale_id(mgr):
    lease = mgr.acquire("alice", "Alice", "replay")
    with pytest.raises(NotHolder):
        mgr.heartbeat(lease.id, "bob")
    assert mgr.heartbeat("nope", "bob") is None
    assert mgr.heartbeat("nope", "alice") is None


def test_release_only_by_holder(mgr):
    lease = mgr.acquire("alice", "Alice", "replay")
    with pytest.raises(NotHolder):
        mgr.release(lease.id, "bob")
    assert mgr.current() is lease
    assert mgr.release("other-id", "alice") is False
    assert mgr.release(lease.id, "alice") is True
    assert mgr.current() is None
    assert mgr.release(lease.id, "alice") is False


def test_request_needs_matching_session_and_holder(mgr):
    lease = mgr.acquire("alice", "Alice", "replay")
    for sid, sub in ((None, "alice"), ("wrong", "alice"), (lease.id, "bob")):
        with pytest.raises(NoLease):
            mgr.begin(sid, sub, "range")


def test_one_in_flight_request_per_lease(mgr):
    lease = mgr.acquire("alice", "Alice", "replay")
    with mgr.request(lease.id, "alice", "range"):
        with pytest.raises(Busy):
            mgr.begin(lease.id, "alice", "range")
        with pytest.raises(Busy):
            mgr.begin(lease.id, "alice", "export")
    with mgr.request(lease.id, "alice", "export"):
        with pytest.raises(Busy):
            mgr.begin(lease.id, "alice", "range")
    with mgr.request(lease.id, "alice", "range"):
        pass


def test_end_is_idempotent(mgr):
    lease = mgr.acquire("alice", "Alice", "export")
    guard = mgr.begin(lease.id, "alice", "export")
    mgr.end(guard)
    second = mgr.begin(lease.id, "alice", "export")
    mgr.end(guard)  # a late repeat must not free the second request
    with pytest.raises(Busy):
        mgr.begin(lease.id, "alice", "range")
    mgr.end(second)


def test_31st_range_request_in_a_minute_is_rate_limited(mgr, clock):
    lease = mgr.acquire("alice", "Alice", "replay")
    for _ in range(30):
        mgr.end(mgr.begin(lease.id, "alice", "range"))
        clock.t += 1
        mgr.heartbeat(lease.id, "alice")
    with pytest.raises(RateLimited) as exc:
        mgr.begin(lease.id, "alice", "range")
    # The first request was at t0 and the clock is at t0 + 30.
    assert exc.value.retry_after == 30
    # Exports are not counted against the range limit.
    mgr.end(mgr.begin(lease.id, "alice", "export"))
    clock.t += 30
    mgr.heartbeat(lease.id, "alice")
    mgr.end(mgr.begin(lease.id, "alice", "range"))


def test_concurrent_acquire_gives_one_lease(mgr):
    results = []
    barrier = threading.Barrier(8)

    def worker(i):
        barrier.wait()
        results.append(mgr.acquire(f"user{i}", f"User {i}", "replay"))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(not isinstance(r, Locked) for r in results) == 1
