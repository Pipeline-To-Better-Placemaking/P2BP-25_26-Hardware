"""Tests for mv3dt_installer.watchdog, the Python port of mv3dt-watchdog.sh.

No real systemctl or journalctl: every call goes through a fake runner, and
the sleep and both clocks are injected.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from mv3dt_installer import app, logs  # noqa: E402
from mv3dt_installer import watchdog as wd  # noqa: E402

UNIT = "mv3dt-pipeline@living-room.service"
HUNG = "**PERF:  0.00 (0.00)\t0.00 (0.00)\n"
RUNNING = "**PERF:  22.00 (23.23)\t21.80 (23.26)\n"
HEADER = "**PERF:  FPS 0 (Avg)\tFPS 1 (Avg)\n"


# ---------------------------------------------------------------------------
# PERF parsing
# ---------------------------------------------------------------------------


def test_parse_perf_line_reads_current_fps_per_source():
    assert wd.parse_perf_line("**PERF:  0.00 (0.00)        0.00 (0.00)") == [0.0, 0.0]
    assert wd.parse_perf_line("**PERF:  22.00 (23.23)    21.80 (23.26)") == [22.0, 21.8]


def test_parse_perf_line_header_has_no_pairs_and_other_lines_are_ignored():
    assert wd.parse_perf_line(HEADER.strip()) == []
    assert wd.parse_perf_line("Active sources : 2") is None
    assert wd.parse_perf_line("  **PERF:  22.00 (23.23)") is None


@pytest.mark.parametrize(
    "text,expected",
    [
        ("", "none"),
        ("Active sources : 2\n", "none"),
        (HEADER + HUNG + HUNG, "hung"),
        (HUNG + RUNNING, "running"),
        ("**PERF:  0.00 (5.00)\t0.40 (5.00)\n", "running"),
        ("**PERF:  0.00 (23.23)\t0.00 (23.26)\n", "hung"),
        (HEADER, "hung"),
    ],
)
def test_classify_perf(text, expected):
    assert wd.classify_perf(text) == expected


# ---------------------------------------------------------------------------
# Watchdog loop
# ---------------------------------------------------------------------------


class FakeSystem:
    """systemctl/journalctl stand-in with a clock the test advances."""

    def __init__(self, *, active=True, active_since=1.0, journal=HUNG):
        self.active = active
        self.active_since = active_since
        self.journal = journal
        self.now = 1000.0
        self.calls: list[list[str]] = []

    def monotonic(self):
        return self.now

    def clock(self):
        return 1_700_000_000.0 + self.now

    def sleep(self, seconds):
        self.now += seconds

    def __call__(self, argv, **kwargs):
        self.calls.append(list(argv))
        if argv[:2] == ["systemctl", "is-active"]:
            state = "active" if self.active else "inactive"
            return subprocess.CompletedProcess(argv, 0 if self.active else 3, state + "\n", "")
        if argv[:2] == ["systemctl", "show"]:
            micros = int(self.active_since * 1_000_000)
            return subprocess.CompletedProcess(argv, 0, f"{micros}\n", "")
        if argv[0] == "journalctl":
            return subprocess.CompletedProcess(argv, 0, self.journal, "")
        if argv[:2] == ["systemctl", "start"]:
            self.active_since = self.now
        return subprocess.CompletedProcess(argv, 0, "", "")

    def verbs(self, verb):
        return [c for c in self.calls if c[:2] == ["systemctl", verb]]


def _dog(system, **kwargs):
    return wd.Watchdog(
        "living-room",
        runner=system,
        sleep=system.sleep,
        clock=system.clock,
        monotonic=system.monotonic,
        **kwargs,
    )


def test_defaults_match_the_deployed_script():
    dog = wd.Watchdog("living-room")
    assert dog.unit == UNIT
    assert (dog.check_interval_s, dog.grace_s, dog.max_restarts, dog.window_s) == (
        10.0,
        30.0,
        5,
        600.0,
    )
    assert dog.restart_pause_s == 5.0


def test_inactive_unit_is_left_alone():
    system = FakeSystem(active=False)
    assert _dog(system).check_once() == "inactive"
    assert not system.verbs("stop")


def test_grace_period_after_start():
    system = FakeSystem(active_since=1000.0 - 29.0)
    assert _dog(system).check_once() == "grace"
    assert not any(c[0] == "journalctl" for c in system.calls)
    system.now += 1.0
    assert _dog(system).check_once() == "restarted"


def test_no_perf_lines_is_not_a_hang():
    system = FakeSystem(journal="nvstreammux: something\n")
    assert _dog(system).check_once() == "no-perf"
    assert not system.verbs("stop")


def test_running_pipeline_is_left_alone():
    system = FakeSystem(journal=HUNG + RUNNING)
    assert _dog(system).check_once() == "running"
    assert not system.verbs("stop")


def test_hung_pipeline_gets_a_clean_restart():
    system = FakeSystem()
    dog = _dog(system)
    before = system.now
    assert dog.check_once() == "restarted"
    restart = [c for c in system.calls if c[:2] in (["systemctl", "stop"], ["systemctl", "start"])]
    assert restart == [["systemctl", "stop", UNIT], ["systemctl", "start", UNIT]]
    assert system.now - before == 5.0
    journal = next(c for c in system.calls if c[0] == "journalctl")
    assert journal == [
        "journalctl",
        "-u",
        UNIT,
        "--since",
        f"@{int(1_700_000_000.0 + before - 30)}",
        "--no-pager",
        "-o",
        "cat",
    ]


def test_restarts_are_rate_limited_per_window():
    system = FakeSystem()
    dog = _dog(system)
    outcomes = []
    for _ in range(7):
        system.now += 30.0  # past the grace after each restart
        outcomes.append(dog.check_once())
    assert outcomes[:5] == ["restarted"] * 5
    assert outcomes[5:] == ["rate-limited", "rate-limited"]
    assert len(system.verbs("stop")) == 5
    # Once the oldest restart leaves the 600 s window, one more is allowed.
    system.now = dog.restarts[0] - 1_700_000_000.0 + 600.0
    assert dog.check_once() == "restarted"


def test_run_forever_sleeps_before_each_check():
    system = FakeSystem(journal=RUNNING)
    dog = _dog(system)
    dog.run_forever(iterations=3)
    assert system.now == 1000.0 + 30.0
    assert len([c for c in system.calls if c[0] == "journalctl"]) == 3


def test_unreadable_start_time_skips_the_check():
    system = FakeSystem()

    def runner(argv, **kwargs):
        if argv[:2] == ["systemctl", "show"]:
            return subprocess.CompletedProcess(argv, 0, "\n", "")
        return system(argv, **kwargs)

    dog = wd.Watchdog("living-room", runner=runner, sleep=system.sleep, monotonic=system.monotonic)
    assert dog.check_once() == "inactive"


def test_runner_errors_do_not_kill_the_loop():
    def runner(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, 30)

    assert wd.Watchdog("living-room", runner=runner).check_once() == "inactive"


# ---------------------------------------------------------------------------
# Subcommand
# ---------------------------------------------------------------------------


class _Ctx:
    log = logs.log


def test_subcommand_is_registered_without_the_root_bootstrap():
    registration = app.SUBCOMMAND_REGISTRY["watchdog"]
    assert registration.handler is wd.handle_watchdog_subcommand
    assert registration.requires_root is False


def test_subcommand_refuses_to_run_unprivileged():
    started = []
    rc = wd.handle_watchdog_subcommand(
        ["--project-slug", "living-room"],
        _Ctx(),
        geteuid=lambda: 1000,
        watchdog_factory=lambda slug: started.append(slug),
    )
    assert rc == 1
    assert started == []


def test_subcommand_rejects_an_invalid_slug():
    rc = wd.handle_watchdog_subcommand(
        ["--project-slug", "../etc"], _Ctx(), geteuid=lambda: 0
    )
    assert rc == 2


def test_subcommand_runs_the_loop_for_the_slug():
    class Dog:
        unit = UNIT
        check_interval_s = 10
        grace_s = 30
        max_restarts = 5
        window_s = 600

        def __init__(self, slug):
            self.slug = slug
            self.ran = False

        def run_forever(self):
            self.ran = True

    dogs = []

    def factory(slug):
        dogs.append(Dog(slug))
        return dogs[-1]

    rc = wd.handle_watchdog_subcommand(
        ["--project-slug", "living-room", "--non-interactive"],
        _Ctx(),
        geteuid=lambda: 0,
        watchdog_factory=factory,
    )
    assert rc == 0
    assert dogs[0].slug == "living-room" and dogs[0].ran
