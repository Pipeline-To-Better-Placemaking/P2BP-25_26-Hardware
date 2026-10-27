"""Pipeline hang watchdog: `mv3dt-installer watchdog --project-slug <slug>`.

A Python port of the hand-deployed `/opt/mv3dt/bin/mv3dt-watchdog.sh`
(LIVE-BASELINE-AND-EDIT-PROTECTION.md section 6.3). DeepStream can stay
running while every RTSP source has stalled; the process is alive, so
systemd's `Restart=on-failure` never fires, and the only symptom is
`**PERF:` lines reporting 0 fps. Every `CHECK_INTERVAL_S`:

1. skip unless `mv3dt-pipeline@<slug>.service` is `active`, and has been
   for at least `GRACE_S` (sources take a while to connect after a start);
2. read the unit's journal for the last `GRACE_S` and keep the lines that
   start with `**PERF:`;
3. no such lines: not this failure mode, leave the unit alone;
4. any source with a current fps above zero: running, nothing to do;
5. otherwise the pipeline is hung: unless `MAX_RESTARTS` restarts already
   happened in the last `WINDOW_S`, do a clean restart (`systemctl stop`,
   `RESTART_PAUSE_S`, `systemctl start`).

A `**PERF:` line carries one ``cur (avg)`` pair per source, for example
``**PERF:  22.00 (23.23)    21.80 (23.26)``; the header line
``**PERF:  FPS 0 (Avg)  FPS 1 (Avg)`` has no pairs and counts as zero, as
it did in the shell script.

`mv3dt-watchdog@.service` runs this as root (it stops and starts another
system unit and reads that unit's journal) with
`SyslogIdentifier=mv3dt-watchdog`, so its log lines land in the journal
under that identifier, like the script's `logger -t mv3dt-watchdog`.
Every external effect (`runner`, `sleep`, both clocks) is injectable.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional

from mv3dt_installer.logs import log

if TYPE_CHECKING:  # pragma: no cover
    from mv3dt_installer.app import Context

__all__ = [
    "CHECK_INTERVAL_S",
    "GRACE_S",
    "MAX_RESTARTS",
    "WINDOW_S",
    "RESTART_PAUSE_S",
    "PERF_PREFIX",
    "parse_perf_line",
    "classify_perf",
    "Watchdog",
    "handle_watchdog_subcommand",
]

CHECK_INTERVAL_S = 10.0
GRACE_S = 30.0
MAX_RESTARTS = 5
WINDOW_S = 600.0
RESTART_PAUSE_S = 5.0

PERF_PREFIX = "**PERF:"
#: One ``cur (avg)`` pair; the first number is the current fps.
_FPS_PAIR_RE = re.compile(r"(\d+\.\d+) \((\d+\.\d+)\)")
#: The Step 5 slug charset, as the polkit rule anchors it.
_SLUG_RE = re.compile(r"^[a-z0-9-]{1,64}$")
_CALL_TIMEOUT_S = 30.0

Runner = Callable[..., "subprocess.CompletedProcess[Any]"]


def parse_perf_line(line: str) -> Optional[list[float]]:
    """Current fps per source of one `**PERF:` line, or None when `line` is
    not a PERF line. A PERF line without pairs (the header) gives []."""
    if not line.startswith(PERF_PREFIX):
        return None
    return [float(cur) for cur, _avg in _FPS_PAIR_RE.findall(line)]


def classify_perf(journal_text: str) -> str:
    """``"none"`` when there is no PERF line, ``"running"`` when any source
    on any line reports a current fps above zero, ``"hung"`` otherwise."""
    seen = False
    for line in journal_text.splitlines():
        values = parse_perf_line(line)
        if values is None:
            continue
        seen = True
        if any(value > 0 for value in values):
            return "running"
    return "hung" if seen else "none"


def _pipeline_unit(slug: str) -> str:
    return f"mv3dt-pipeline@{slug}.service"


@dataclass
class Watchdog:
    slug: str
    runner: Runner = subprocess.run
    sleep: Callable[[float], None] = time.sleep
    #: Wall clock, for `journalctl --since` and the restart window.
    clock: Callable[[], float] = time.time
    #: CLOCK_MONOTONIC, the clock of systemd's `*TimestampMonotonic`.
    monotonic: Callable[[], float] = time.monotonic
    check_interval_s: float = CHECK_INTERVAL_S
    grace_s: float = GRACE_S
    max_restarts: int = MAX_RESTARTS
    window_s: float = WINDOW_S
    restart_pause_s: float = RESTART_PAUSE_S
    restarts: list[float] = field(default_factory=list)

    @property
    def unit(self) -> str:
        return _pipeline_unit(self.slug)

    def _call(self, *argv: str) -> "subprocess.CompletedProcess[Any]":
        try:
            return self.runner(
                list(argv),
                check=False,
                capture_output=True,
                text=True,
                timeout=_CALL_TIMEOUT_S,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return subprocess.CompletedProcess(list(argv), 1, "", str(exc))

    def _log(self, emit: Callable[[str], None], message: str) -> None:
        emit(f"{self.slug}: {message}")

    def _active_for_s(self) -> Optional[float]:
        """Seconds the unit has been active, or None when it is not active
        or its start time cannot be read."""
        state = self._call("systemctl", "is-active", self.unit)
        if (state.stdout or "").strip() != "active":
            return None
        shown = self._call(
            "systemctl",
            "show",
            "-p",
            "ActiveEnterTimestampMonotonic",
            "--value",
            self.unit,
        )
        try:
            entered_us = int((shown.stdout or "").strip())
        except ValueError:
            return None
        if shown.returncode != 0 or entered_us <= 0:
            return None
        return self.monotonic() - entered_us / 1_000_000

    def _recent_journal(self, now: float) -> str:
        result = self._call(
            "journalctl",
            "-u",
            self.unit,
            "--since",
            f"@{int(now - self.grace_s)}",
            "--no-pager",
            "-o",
            "cat",
        )
        return result.stdout or ""

    def check_once(self) -> str:
        """One check. Returns what it decided: ``inactive``, ``grace``,
        ``no-perf``, ``running``, ``rate-limited`` or ``restarted``."""
        active_for = self._active_for_s()
        if active_for is None:
            return "inactive"
        if active_for < self.grace_s:
            return "grace"
        now = self.clock()
        verdict = classify_perf(self._recent_journal(now))
        if verdict == "none":
            return "no-perf"
        if verdict == "running":
            return "running"
        self.restarts = [t for t in self.restarts if now - t < self.window_s]
        if len(self.restarts) >= self.max_restarts:
            self._log(
                log.warn,
                f"pipeline at 0 fps but {len(self.restarts)} restarts in the last "
                f"{int(self.window_s)}s; not restarting again yet",
            )
            return "rate-limited"
        self._log(
            log.warn,
            f"pipeline at 0 fps for {int(self.grace_s)}s after start; restarting",
        )
        self._call("systemctl", "stop", self.unit)
        self.sleep(self.restart_pause_s)
        self._call("systemctl", "start", self.unit)
        self.restarts.append(self.clock())
        return "restarted"

    def run_forever(self, *, iterations: Optional[int] = None) -> None:
        """Sleep, check, repeat. `iterations` bounds the loop for tests."""
        count = 0
        while iterations is None or count < iterations:
            self.sleep(self.check_interval_s)
            self.check_once()
            count += 1


def _build_watchdog_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mv3dt-installer watchdog", add_help=True)
    parser.add_argument("--project-slug", required=True)
    parser.add_argument("--install-dir", default=None)
    parser.add_argument("--non-interactive", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--log-dir", default=None)
    return parser


def handle_watchdog_subcommand(
    argv: list,
    ctx: "Context",
    *,
    geteuid: Callable[[], int] = os.geteuid,
    watchdog_factory: Callable[[str], Watchdog] = Watchdog,
) -> int:
    """Run the watchdog loop for one project until the process is stopped.

    Registered with ``requires_root=False`` so the bootstrap only reads
    installer.conf (no config write, no transcript per start), but the loop
    itself needs root to stop and start the pipeline unit and read its
    journal, so it refuses to start otherwise."""
    args = _build_watchdog_parser().parse_args(argv)
    if not _SLUG_RE.match(args.project_slug):
        ctx.log.error(f"watchdog: invalid project slug {args.project_slug!r}")
        return 2
    if geteuid() != 0:
        ctx.log.error(
            "watchdog: must run as root (it restarts "
            f"{_pipeline_unit(args.project_slug)}); it is started by "
            f"mv3dt-watchdog@{args.project_slug}.service"
        )
        return 1
    dog = watchdog_factory(args.project_slug)
    ctx.log.info(
        f"watchdog: supervising {dog.unit} (check every {int(dog.check_interval_s)}s, "
        f"grace {int(dog.grace_s)}s, at most {dog.max_restarts} restarts per "
        f"{int(dog.window_s)}s)"
    )
    dog.run_forever()
    return 0  # pragma: no cover -- run_forever only returns in tests
