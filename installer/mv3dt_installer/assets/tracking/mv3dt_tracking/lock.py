"""The data-access lease (STEP-8 section 9).

One lease per project, held in memory; a service restart clears it. The
service accepts tickets for a single project, so the manager holds a single
lease slot. A lease expires 45 s after its last heartbeat and never outlives
15 minutes from acquisition. Each lease allows one in-flight data request,
30 `range` requests per rolling minute, and the whole service one export at a
time. All times are seconds from the injectable clock.
"""

from __future__ import annotations

import math
import secrets
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Iterator

KINDS = ("replay", "export")
TTL_S = 45.0
MAX_AGE_S = 15 * 60.0
RANGE_LIMIT = 30
RATE_WINDOW_S = 60.0


@dataclass
class Lease:
    id: str
    holder_sub: str
    holder_name: str
    kind: str
    acquired_at: float
    last_heartbeat: float
    in_flight: bool = False
    range_times: deque = field(default_factory=deque)

    def expires_at(self, ttl_s: float = TTL_S, max_age_s: float = MAX_AGE_S) -> float:
        return min(self.last_heartbeat + ttl_s, self.acquired_at + max_age_s)


@dataclass(frozen=True)
class Locked:
    """Another holder has the lease."""

    held_by: str
    until: float


@dataclass
class Guard:
    """One admitted data request; end() it exactly once (repeats are no-ops)."""

    lease: Lease
    op: str
    done: bool = False


class LeaseError(Exception):
    """Base for request-guard refusals."""


class NoLease(LeaseError):
    """No live lease matches the session id and caller (423)."""


class Busy(LeaseError):
    """A request is already in flight on this lease, or an export is running (429)."""

    retry_after = 1


class RateLimited(LeaseError):
    """The lease used up its range requests for the rolling minute (429)."""

    def __init__(self, retry_after: int) -> None:
        super().__init__(f"range rate limit reached; retry in {retry_after} s")
        self.retry_after = retry_after


class NotHolder(LeaseError):
    """The caller does not hold the lease it tried to change (403)."""


class LeaseManager:
    def __init__(
        self,
        clock: Callable[[], float] = time.time,
        ttl_s: float = TTL_S,
        max_age_s: float = MAX_AGE_S,
        range_limit: int = RANGE_LIMIT,
        rate_window_s: float = RATE_WINDOW_S,
    ) -> None:
        self._clock = clock
        self.ttl_s = ttl_s
        self.max_age_s = max_age_s
        self.range_limit = range_limit
        self.rate_window_s = rate_window_s
        self._mutex = threading.Lock()
        self._lease: Lease | None = None
        self._exporting = False

    def expires_at(self, lease: Lease) -> float:
        return lease.expires_at(self.ttl_s, self.max_age_s)

    def _live(self, now: float) -> Lease | None:
        """The current lease, dropping it first if it has expired."""
        lease = self._lease
        if lease is not None and now >= self.expires_at(lease):
            self._lease = lease = None
        return lease

    def current(self) -> Lease | None:
        with self._mutex:
            return self._live(self._clock())

    def acquire(self, holder_sub: str, holder_name: str, kind: str) -> Lease | Locked:
        """The caller's lease (new, or their existing one) or Locked."""
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        with self._mutex:
            now = self._clock()
            lease = self._live(now)
            if lease is not None:
                if lease.holder_sub != holder_sub:
                    return Locked(lease.holder_name, self.expires_at(lease))
                lease.last_heartbeat = now
                lease.kind = kind
                return lease
            lease = Lease(
                id=secrets.token_urlsafe(24),
                holder_sub=holder_sub,
                holder_name=holder_name,
                kind=kind,
                acquired_at=now,
                last_heartbeat=now,
            )
            self._lease = lease
            return lease

    def heartbeat(self, lease_id: str, holder_sub: str) -> Lease | Locked | None:
        """Extend the lease. None if no live lease has this id; Locked if the
        caller is not its holder."""
        with self._mutex:
            now = self._clock()
            lease = self._live(now)
            if lease is None or lease.id != lease_id:
                return None
            if lease.holder_sub != holder_sub:
                return Locked(lease.holder_name, self.expires_at(lease))
            lease.last_heartbeat = now
            return lease

    def release(self, lease_id: str, holder_sub: str) -> bool:
        """Drop the lease. False if no live lease has this id; NotHolder if
        the caller is not its holder."""
        with self._mutex:
            lease = self._live(self._clock())
            if lease is None or lease.id != lease_id:
                return False
            if lease.holder_sub != holder_sub:
                raise NotHolder("only the lease holder may release it")
            self._lease = None
            return True

    # -- data requests -------------------------------------------------------

    def begin(self, lease_id: str | None, holder_sub: str, op: str) -> Guard:
        """Admit a `range` or `export` request on the lease; pair with end()."""
        if op not in ("range", "export"):
            raise ValueError(f"op must be range or export, got {op!r}")
        with self._mutex:
            now = self._clock()
            lease = self._live(now)
            if lease is None or not lease_id or lease.id != lease_id or lease.holder_sub != holder_sub:
                raise NoLease("X-Tracking-Session does not match the live lease")
            if lease.in_flight:
                raise Busy("a request is already in flight on this lease")
            if op == "export" and self._exporting:
                raise Busy("an export is already running")
            if op == "range":
                times = lease.range_times
                while times and now - times[0] >= self.rate_window_s:
                    times.popleft()
                if len(times) >= self.range_limit:
                    wait = times[0] + self.rate_window_s - now
                    raise RateLimited(max(1, math.ceil(wait)))
                times.append(now)
            else:
                self._exporting = True
            lease.in_flight = True
            return Guard(lease, op)

    def end(self, guard: Guard) -> None:
        with self._mutex:
            if guard.done:
                return
            guard.done = True
            guard.lease.in_flight = False
            if guard.op == "export":
                self._exporting = False

    @contextmanager
    def request(self, lease_id: str | None, holder_sub: str, op: str) -> Iterator[Lease]:
        guard = self.begin(lease_id, holder_sub, op)
        try:
            yield guard.lease
        finally:
            self.end(guard)
