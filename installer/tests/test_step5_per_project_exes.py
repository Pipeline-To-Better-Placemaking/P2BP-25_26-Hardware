"""Tests for mv3dt_installer.steps.step5_per_project_exes
(STEP-5-PER-PROJECT-EXES.md).

Run from installer/: `python3 -m pytest tests/test_step5_per_project_exes.py -v`

No test here shells out for real, opens a browser, or touches
docker/git/systemctl -- every `ctx.run_root`/`ctx.run_as_user` call is
served by a `ScriptedRunner` fake (mirroring `test_step3_amc_launcher.py`'s
convention), `os.execv` is always injected, and the start-or-close /
reconciliation prompts are injected too.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from mv3dt_installer import app  # noqa: E402
from mv3dt_installer import prompting  # noqa: E402
from mv3dt_installer import config as config_mod  # noqa: E402
from mv3dt_installer import logs, report, shellout  # noqa: E402
from mv3dt_installer.steps import STEP_REGISTRY, StepStatus  # noqa: E402
from mv3dt_installer.steps import step3_amc_launcher as step3  # noqa: E402
from mv3dt_installer.steps import step5_per_project_exes as step5  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_transcript_state():
    logs._transcript_path = None
    yield
    logs._transcript_path = None


@pytest.fixture(autouse=True)
def _force_no_colour(monkeypatch):
    monkeypatch.setattr(sys.stderr, "isatty", lambda: False)


@pytest.fixture(autouse=True)
def _no_real_repo_root(monkeypatch):
    monkeypatch.setattr(step3, "repo_root", lambda: None)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class ScriptedRunner:
    """Stand-in for `ctx.run_root`/`ctx.run_as_user` (mirrors
    test_step3_amc_launcher.py's identical fake)."""

    def __init__(self, *, default_returncode: int = 0, default_stdout: str = "", default_stderr: str = ""):
        self.calls: list[tuple] = []
        self._rules: list[tuple] = []
        self.default_returncode = default_returncode
        self.default_stdout = default_stdout
        self.default_stderr = default_stderr

    def when(self, matcher, *, returncode=0, stdout="", stderr=""):
        self._rules.append((matcher, returncode, stdout, stderr))
        return self

    def __call__(self, *args, **kwargs):
        self.calls.append(args)
        for matcher, returncode, stdout, stderr in reversed(self._rules):
            if matcher(args):
                return subprocess.CompletedProcess(list(args), returncode, stdout, stderr)
        return subprocess.CompletedProcess(
            list(args), self.default_returncode, self.default_stdout, self.default_stderr
        )

    def called_with_prefix(self, *prefix) -> bool:
        return any(tuple(call[: len(prefix)]) == prefix for call in self.calls)


class FakeUser:
    name = "op"
    uid = 1000
    gid = 1000

    def __init__(self, home: pathlib.Path):
        self.home = home


class FakeContext:
    def __init__(
        self,
        tmp_path,
        *,
        conf=None,
        runner_root=None,
        runner_user=None,
        non_interactive=True,
    ):
        self.install_dir = tmp_path / "mv3dt"
        self.install_dir.mkdir(parents=True, exist_ok=True)
        self.conf = conf if conf is not None else {}
        self.user = FakeUser(home=tmp_path / "home" / "op")
        self.user.home.mkdir(parents=True, exist_ok=True)
        self.log = logs.log
        self.report_installed = report.report_installed
        self.report_already_installed = report.report_already_installed
        self.verify_pinned = report.verify_pinned
        self.runner_root = runner_root if runner_root is not None else ScriptedRunner()
        self.runner_user = runner_user if runner_user is not None else ScriptedRunner()
        self.non_interactive = non_interactive
        # Doc 08 section 3.1: `run()` announces its phases through
        # `ctx.progress`. Nothing here asserts on rendering, so the handle
        # only has to exist and swallow the calls.
        self.progress = SimpleNamespace(
            begin_step=lambda step, index: None,
            end_step=lambda: None,
            phase=lambda number: None,
            task=lambda name: None,
            bytes=lambda done, total: None,
            line=lambda text: None,
        )

    def run_root(self, *args, **kwargs):
        return self.runner_root(*args, **kwargs)

    def run_as_user(self, *args, **kwargs):
        return self.runner_user(*args, **kwargs)


def _make_entry(install_dir, *, project_name="North Lobby #2", slug="north-lobby-2", location_id="loc-1"):
    exe = install_dir / "bin" / f"pipeline-{slug}"
    exe.parent.mkdir(parents=True, exist_ok=True)
    exe.write_text(f'exec "x" pipeline --project "{project_name}" "$@"\n')
    exe.chmod(0o755)
    rendered = install_dir / "deepstream" / "deepstream_app_config.rendered.txt"
    rendered.parent.mkdir(parents=True, exist_ok=True)
    rendered.write_text("[application]\n")
    calib = install_dir / "deepstream" / "calibration" / location_id
    calib.mkdir(parents=True, exist_ok=True)
    return step5.upsert(
        install_dir,
        project_name=project_name,
        location_id=location_id,
        rendered_config=str(rendered),
        calibration_dir=str(calib),
        exe=str(exe),
        slug=slug,
    )


# ---------------------------------------------------------------------------
# Module identity + subcommand registration
# ---------------------------------------------------------------------------


def test_registers_itself_with_the_expected_identity():
    matches = [s for s in STEP_REGISTRY if s.id == "step5_per_project_exes"]
    assert len(matches) == 1
    step = matches[0]
    assert step.order == 5
    assert "Per-project executables" in step.title


def test_registers_the_pipeline_record_and_projects_subcommands():
    pipeline_reg = app.SUBCOMMAND_REGISTRY.get("pipeline")
    record_reg = app.SUBCOMMAND_REGISTRY.get("record")
    projects_reg = app.SUBCOMMAND_REGISTRY.get("projects")
    assert pipeline_reg is not None
    assert record_reg is not None
    assert projects_reg is not None
    assert pipeline_reg.handler is step5.handle_pipeline_subcommand
    assert record_reg.handler is step5.handle_record_subcommand
    assert projects_reg.handler is step5.handle_projects_subcommand
    # `record`/`projects` are unaffected by unit U6's fix -- unchanged
    # `requires_root=True` default. `pipeline` alone gets the mode-dependent
    # predicate (see the dedicated tests below).
    assert record_reg.requires_root is True
    assert projects_reg.requires_root is True


# ---------------------------------------------------------------------------
# section 3.1 -- slug sanitization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,expected",
    [
        ("North Lobby #2", "north-lobby-2"),
        ("  Weird   Spacing  ", "weird-spacing"),
        ("ALLCAPS", "allcaps"),
        ("--leading-and-trailing--", "leading-and-trailing"),
        ("a" * 100, "a" * 64),
    ],
)
def test_slugify_sanitizes(name, expected):
    assert step5.slugify(name) == expected


def test_slugify_falls_back_to_location_id_when_name_is_all_punctuation():
    assert step5.slugify("####", fallback="loc-9") == "loc-9"


def test_slugify_falls_back_to_project_literal_when_everything_is_empty():
    assert step5.slugify("###", fallback="***") == "project"
    assert step5.slugify("", fallback="") == "project"


# ---------------------------------------------------------------------------
# section 3.1 -- collision policy (resolve_slug)
# ---------------------------------------------------------------------------


def test_resolve_slug_no_collision(tmp_path):
    slug, err = step5.resolve_slug(tmp_path, "North Lobby #2", "loc-1")
    assert slug == "north-lobby-2"
    assert err is None


def test_resolve_slug_rerun_of_same_project_keeps_its_existing_slug(tmp_path):
    step5.upsert(
        tmp_path,
        project_name="North Lobby #2",
        location_id="loc-1",
        rendered_config="r",
        calibration_dir="c",
        exe="e",
        slug="custom-slug",
    )
    slug, err = step5.resolve_slug(tmp_path, "North Lobby #2", "loc-1")
    assert slug == "custom-slug"
    assert err is None


def test_resolve_slug_collision_appends_location_id(tmp_path):
    step5.upsert(
        tmp_path,
        project_name="Other Project",
        location_id="other-loc",
        rendered_config="r",
        calibration_dir="c",
        exe="e",
        slug="north-lobby-2",
    )
    slug, err = step5.resolve_slug(tmp_path, "North Lobby #2", "loc-1")
    assert slug == "north-lobby-2-loc-1"
    assert err is None


def test_resolve_slug_unresolvable_collision_is_an_error(tmp_path):
    step5.upsert(
        tmp_path,
        project_name="Other Project",
        location_id="other-loc",
        rendered_config="r",
        calibration_dir="c",
        exe="e",
        slug="north-lobby-2",
    )
    step5.upsert(
        tmp_path,
        project_name="Yet Another",
        location_id="yet-loc",
        rendered_config="r",
        calibration_dir="c",
        exe="e",
        slug="north-lobby-2-loc-1",
    )
    slug, err = step5.resolve_slug(tmp_path, "North Lobby #2", "loc-1")
    assert err is not None
    assert "distinct" in err


# ---------------------------------------------------------------------------
# section 4 -- the project registry
# ---------------------------------------------------------------------------


def test_load_registry_missing_file_is_empty(tmp_path):
    registry = step5.load_registry(tmp_path)
    assert registry.projects == {}
    assert registry.schema_version == step5.REGISTRY_SCHEMA_VERSION


def test_load_registry_corrupt_file_is_forgiving(tmp_path):
    path = step5.registry_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not valid json {{{")
    registry = step5.load_registry(tmp_path)
    assert registry.projects == {}


def test_load_registry_non_dict_json_is_forgiving(tmp_path):
    path = step5.registry_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("[1, 2, 3]")
    registry = step5.load_registry(tmp_path)
    assert registry.projects == {}


def test_save_registry_writes_atomically_and_round_trips(tmp_path):
    registry = step5.Registry()
    entry = step5.ProjectEntry(
        project_name="P1",
        slug="p1",
        location_id="loc",
        exe="e",
        rendered_config="r",
        calibration_dir="c",
        created_utc="2026-01-01T00:00:00Z",
        updated_utc="2026-01-01T00:00:00Z",
    )
    registry.projects["P1"] = entry
    step5.save_registry(tmp_path, registry)

    path = step5.registry_path(tmp_path)
    assert path.is_file()
    assert not path.with_suffix(path.suffix + ".tmp").exists()
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk["projects"]["P1"]["slug"] == "p1"

    reloaded = step5.load_registry(tmp_path)
    assert reloaded.projects["P1"].slug == "p1"


def test_upsert_creates_then_updates_bumping_calib_runs(tmp_path):
    first = step5.upsert(
        tmp_path,
        project_name="P1",
        location_id="loc",
        rendered_config="r1",
        calibration_dir="c1",
        exe="e1",
    )
    assert first.calib_runs == 1
    created = first.created_utc

    second = step5.upsert(
        tmp_path,
        project_name="P1",
        location_id="loc",
        rendered_config="r2",
        calibration_dir="c2",
        exe="e1",
    )
    assert second.calib_runs == 2
    assert second.created_utc == created
    assert second.rendered_config == "r2"
    assert second.slug == first.slug  # slug is stable across re-runs


def test_get_and_list_projects(tmp_path):
    assert step5.get(tmp_path, "nope") is None
    assert step5.list_projects(tmp_path) == []

    step5.upsert(
        tmp_path, project_name="P1", location_id="loc", rendered_config="r", calibration_dir="c", exe="e"
    )
    assert step5.get(tmp_path, "P1") is not None
    assert len(step5.list_projects(tmp_path)) == 1


def test_remove_registry_entry(tmp_path):
    step5.upsert(
        tmp_path, project_name="P1", location_id="loc", rendered_config="r", calibration_dir="c", exe="e"
    )
    assert step5.remove_registry_entry(tmp_path, "P1") is True
    assert step5.get(tmp_path, "P1") is None
    assert step5.remove_registry_entry(tmp_path, "P1") is False


# ---------------------------------------------------------------------------
# section 2 -- start-or-close prompt
# ---------------------------------------------------------------------------


def test_confirm_start_now_defaults_to_close_under_non_interactive(tmp_path, monkeypatch):
    ctx = FakeContext(tmp_path, non_interactive=True)

    def _boom(_prompt):  # pragma: no cover -- must never be called
        raise AssertionError("input() must not be called under non_interactive")

    monkeypatch.setattr(step5, "_INPUT", _boom)
    assert step5._confirm_start_now(ctx) is False


def test_confirm_start_now_interactive_start(tmp_path, monkeypatch):
    ctx = FakeContext(tmp_path, non_interactive=False)
    monkeypatch.setattr(step5, "_INPUT", lambda _prompt: "S")
    assert step5._confirm_start_now(ctx) is True


def test_confirm_start_now_never_blocks_on_a_hidden_prompt(tmp_path, monkeypatch, capsys):
    """Live regression: under `sudo mv3dt-installer 2>&1 | tail -40` the
    prompt went into the pipe and the run hung on it. With no visible
    prompt the documented default (Close) is taken and said so."""
    ctx = FakeContext(tmp_path, non_interactive=False)
    monkeypatch.setattr(
        prompting, "prompt_unavailable_reason", lambda: "stdout is not a terminal (is the output piped?)"
    )

    def _boom(_prompt):  # pragma: no cover -- must never be called
        raise AssertionError("input() must not be called without a visible prompt")

    monkeypatch.setattr(step5, "_INPUT", _boom)
    assert step5._confirm_start_now(ctx) is False
    err = capsys.readouterr().err
    assert "not asking 'What next?'; closing (the default)" in err
    assert "without a pipe" in err


def test_ctrl_c_at_the_prompt_leaves_step5_incomplete(tmp_path, monkeypatch):
    """Ctrl-C at the What next? prompt propagates; the dispatch loop never
    reaches mark_complete, so the step re-runs next time."""
    ctx = FakeContext(tmp_path, non_interactive=False)

    def _ctrl_c(_prompt):
        raise KeyboardInterrupt

    monkeypatch.setattr(step5, "_INPUT", _ctrl_c)
    with pytest.raises(KeyboardInterrupt):
        step5._confirm_start_now(ctx)


@pytest.mark.parametrize("answer", ["c", "C", "", "no", "n"])
def test_confirm_start_now_interactive_close(tmp_path, monkeypatch, answer):
    ctx = FakeContext(tmp_path, non_interactive=False)
    monkeypatch.setattr(step5, "_INPUT", lambda _prompt: answer)
    assert step5._confirm_start_now(ctx) is False


# ---------------------------------------------------------------------------
# section 3.3 -- pipeline subcommand flag handling
# ---------------------------------------------------------------------------


def test_pipeline_subcommand_unknown_project_fails(tmp_path):
    ctx = FakeContext(tmp_path)
    rc = step5.handle_pipeline_subcommand(["--project", "nope"], ctx)
    assert rc == 1


def test_pipeline_subcommand_dry_run_does_not_exec(tmp_path):
    ctx = FakeContext(tmp_path)
    entry = _make_entry(ctx.install_dir)

    calls = []
    monkeypatch_execv = lambda *a: calls.append(a)  # noqa: E731
    rc = step5._start_pipeline_foreground(ctx, entry, dry_run=True, execv=monkeypatch_execv)
    assert rc == 0
    assert calls == []


def test_pipeline_execs_deepstream_test5_app(tmp_path, monkeypatch):
    """deepstream-app connects the msgbroker sink but never sends anything;
    deepstream-test5-app attaches the event metadata it publishes."""
    ctx = FakeContext(tmp_path)
    entry = _make_entry(ctx.install_dir)
    monkeypatch.setattr(step5, "ensure_mosquitto", lambda ctx_: None)
    monkeypatch.setattr(step5, "ping_sweep_cameras", lambda ctx_: None)
    monkeypatch.setattr(step5, "source_deepstream_env", lambda ctx_: {})
    monkeypatch.setattr(step5.os, "chdir", lambda path: None)
    calls = []
    rc = step5._start_pipeline_foreground(ctx, entry, execv=lambda binary, argv: calls.append(argv))
    assert rc == 0
    assert calls and calls[0][0] == "deepstream-test5-app"


def _service_exec_argv(tmp_path, monkeypatch, conf):
    monkeypatch.delenv(config_mod.GATE_VIEWER_STREAM, raising=False)
    ctx = FakeContext(tmp_path, conf=conf)
    entry = _make_entry(ctx.install_dir)
    monkeypatch.setattr(step5, "ensure_mosquitto", lambda ctx_: None)
    monkeypatch.setattr(step5, "source_deepstream_env", lambda ctx_: {})
    monkeypatch.setattr(step5.os, "chdir", lambda path: None)
    calls = []
    rc = step5._start_pipeline_service_exec(
        ctx, entry, execv=lambda binary, argv: calls.append(argv)
    )
    assert rc == 0
    return calls[0], entry


def test_service_exec_passes_t_when_viewer_stream_on(tmp_path, monkeypatch):
    argv, entry = _service_exec_argv(
        tmp_path, monkeypatch, {config_mod.GATE_VIEWER_STREAM: "on"}
    )
    assert argv == ["deepstream-test5-app", "-t", "-c", entry.rendered_config]


@pytest.mark.parametrize("conf", [{}, {config_mod.GATE_VIEWER_STREAM: "off"}])
def test_service_exec_omits_t_when_viewer_stream_off(tmp_path, monkeypatch, conf):
    argv, entry = _service_exec_argv(tmp_path, monkeypatch, conf)
    assert argv == ["deepstream-test5-app", "-c", entry.rendered_config]


def test_service_exec_reads_viewer_stream_from_environment(tmp_path, monkeypatch):
    ctx = FakeContext(tmp_path, conf={})
    monkeypatch.setenv(config_mod.GATE_VIEWER_STREAM, "on")
    assert step5._viewer_stream_on(ctx) is True
    ctx.conf[config_mod.GATE_VIEWER_STREAM] = "off"
    assert step5._viewer_stream_on(ctx) is False


def test_pipeline_process_name_is_truncated_for_pgrep():
    assert step5.DEEPSTREAM_APP == "deepstream-test5-app"
    assert step5.DEEPSTREAM_APP_PROCESS == "deepstream-test"


def test_pipeline_subcommand_dispatches_to_start_foreground(tmp_path, monkeypatch):
    ctx = FakeContext(tmp_path)
    _make_entry(ctx.install_dir)

    seen = {}

    def _fake_start(ctx_, entry_, **kwargs):
        seen["entry"] = entry_
        seen["kwargs"] = kwargs
        return 0

    monkeypatch.setattr(step5, "_start_pipeline_foreground", _fake_start)
    rc = step5.handle_pipeline_subcommand(
        ["--project", "North Lobby #2", "--preview", "--skip-ping", "--dry-run", "--config", "/tmp/x.txt"],
        ctx,
    )
    assert rc == 0
    assert seen["kwargs"]["preview"] is True
    assert seen["kwargs"]["skip_ping"] is True
    assert seen["kwargs"]["dry_run"] is True
    assert seen["kwargs"]["config_override"] == "/tmp/x.txt"


def test_pipeline_subcommand_missing_config_fails(tmp_path):
    ctx = FakeContext(tmp_path)
    entry = _make_entry(ctx.install_dir)
    entry = step5.upsert(
        ctx.install_dir,
        project_name=entry.project_name,
        location_id=entry.location_id,
        rendered_config=str(ctx.install_dir / "does-not-exist.txt"),
        calibration_dir=entry.calibration_dir,
        exe=entry.exe,
        slug=entry.slug,
    )
    rc = step5._start_pipeline_foreground(ctx, entry, dry_run=True)
    assert rc == 1


# ---------------------------------------------------------------------------
# section 3.4 -- --stop / --stop-all
# ---------------------------------------------------------------------------


def test_stop_pipeline_default_only_stops_deepstream(tmp_path):
    # [initial check: alive, poll 1: gone] -- SIGTERM sent, dies immediately.
    runner = _SequencedPgrepRunner([0, 1])
    ctx = FakeContext(tmp_path, runner_root=runner)
    entry = _make_entry(ctx.install_dir)

    def _boom(_s):  # pragma: no cover -- must never be called; process is already gone
        raise AssertionError("sleep() must not be called once the process has exited")

    rc = step5._stop_pipeline(ctx, entry, sleep=_boom)
    assert rc == 0
    assert runner.called_with_prefix("pkill", "-TERM", "-x", step5.DEEPSTREAM_PROCESS_PATTERN)
    assert not runner.called_with_prefix("pkill", "-KILL")
    assert not runner.called_with_prefix("systemctl", "stop", "mosquitto")


def test_stop_pipeline_deepstream_not_running_does_not_kill(tmp_path):
    runner = ScriptedRunner(default_returncode=1)  # pgrep finds nothing
    ctx = FakeContext(tmp_path, runner_root=runner)
    entry = _make_entry(ctx.install_dir)

    step5._stop_pipeline(ctx, entry)
    assert not runner.called_with_prefix("pkill")


class _SequencedPgrepRunner:
    """Serves a fixed sequence of `pgrep -x <DEEPSTREAM_APP_PROCESS>` returncodes (one
    per call -- the *initial* aliveness check consumes the first entry, then
    one entry per grace-period poll), then `1` (not running) once exhausted.
    Every other command returns 0. Used to exercise the SIGTERM->SIGKILL
    grace-period timing precisely."""

    def __init__(self, pgrep_returncodes):
        self.calls: list[tuple] = []
        self._pgrep_returncodes = list(pgrep_returncodes)

    def __call__(self, *args, **kwargs):
        self.calls.append(args)
        if args[:2] == ("pgrep", "-x"):
            rc = self._pgrep_returncodes.pop(0) if self._pgrep_returncodes else 1
            return subprocess.CompletedProcess(list(args), rc, "", "")
        return subprocess.CompletedProcess(list(args), 0, "", "")

    def called_with_prefix(self, *prefix) -> bool:
        return any(tuple(call[: len(prefix)]) == prefix for call in self.calls)

    def count_calls(self, *prefix) -> int:
        return sum(1 for call in self.calls if tuple(call[: len(prefix)]) == prefix)


def test_stop_pipeline_grace_period_lets_process_exit_without_sigkill(tmp_path):
    """Process dies partway through the grace period: SIGKILL must never be
    sent, and the poll loop must stop as soon as the process is gone rather
    than always running the full `grace_polls` iterations."""
    # [initial check: alive, poll 1: alive, poll 2: gone]
    runner = _SequencedPgrepRunner([0, 0, 1])
    ctx = FakeContext(tmp_path, runner_root=runner)
    entry = _make_entry(ctx.install_dir)

    sleeps: list[float] = []
    rc = step5._stop_pipeline(
        ctx, entry, sleep=sleeps.append, grace_polls=5, grace_interval_s=1.0
    )
    assert rc == 0
    assert runner.called_with_prefix("pkill", "-TERM", "-x", step5.DEEPSTREAM_PROCESS_PATTERN)
    assert not runner.called_with_prefix("pkill", "-KILL")
    # Only one grace-period poll happened before the process was confirmed
    # gone, so only one sleep -- not all 5.
    assert sleeps == [1.0]


def test_stop_pipeline_waits_full_grace_period_before_sigkill(tmp_path):
    """Regression test for the grace-period bug: SIGTERM must be followed
    by up to `grace_polls` waits (matching 99_stop_all.sh's `for _ in 1 2 3
    4 5; do pgrep || break; sleep 1; done`) before SIGKILL is ever sent --
    not an immediate SIGKILL synchronously after SIGTERM."""
    events: list[str] = []

    class _EventRunner:
        def __init__(self):
            self.calls: list[tuple] = []

        def __call__(self, *args, **kwargs):
            self.calls.append(args)
            if args[:2] == ("pgrep", "-x"):
                events.append("pgrep-alive")
                return subprocess.CompletedProcess(list(args), 0, "", "")  # always still running
            if args[:3] == ("pkill", "-TERM", "-x"):
                events.append("sigterm")
            elif args[:3] == ("pkill", "-KILL", "-x"):
                events.append("sigkill")
            return subprocess.CompletedProcess(list(args), 0, "", "")

        def called_with_prefix(self, *prefix) -> bool:
            return any(tuple(c[: len(prefix)]) == prefix for c in self.calls)

    runner = _EventRunner()
    ctx = FakeContext(tmp_path, runner_root=runner)
    entry = _make_entry(ctx.install_dir)

    def _fake_sleep(_seconds):
        events.append("sleep")

    rc = step5._stop_pipeline(ctx, entry, sleep=_fake_sleep, grace_polls=3, grace_interval_s=0.01)
    assert rc == 0
    assert runner.called_with_prefix("pkill", "-KILL", "-x", step5.DEEPSTREAM_PROCESS_PATTERN)

    sigterm_idx = events.index("sigterm")
    sigkill_idx = events.index("sigkill")
    sleeps_between = events[sigterm_idx:sigkill_idx].count("sleep")
    # All 3 grace-period sleeps must have happened between SIGTERM and
    # SIGKILL -- the bug this guards against sent SIGKILL immediately, with
    # zero sleeps in between.
    assert sleeps_between == 3


def test_stop_all_also_stops_mosquitto_and_amc(tmp_path, monkeypatch):
    runner = ScriptedRunner()
    runner.when(lambda a: a[:2] == ("pgrep", "-x"), returncode=1)
    runner.when(lambda a: a[:3] == ("systemctl", "is-active", "--quiet"), returncode=0)
    ctx = FakeContext(tmp_path, runner_root=runner)
    entry = _make_entry(ctx.install_dir)

    teardown_calls = []
    monkeypatch.setattr(step3, "teardown_amc", lambda ctx_: teardown_calls.append(ctx_))

    rc = step5._stop_pipeline(ctx, entry, stop_all=True)
    assert rc == 0
    assert runner.called_with_prefix("systemctl", "stop", "mosquitto")
    assert len(teardown_calls) == 1


def test_stop_all_skip_flags_are_honored(tmp_path, monkeypatch):
    runner = ScriptedRunner()
    runner.when(lambda a: a[:2] == ("pgrep", "-x"), returncode=1)
    runner.when(lambda a: a[:3] == ("systemctl", "is-active", "--quiet"), returncode=0)
    ctx = FakeContext(tmp_path, runner_root=runner)
    entry = _make_entry(ctx.install_dir)

    teardown_calls = []
    monkeypatch.setattr(step3, "teardown_amc", lambda ctx_: teardown_calls.append(ctx_))

    step5._stop_pipeline(ctx, entry, stop_all=True, skip_mosquitto=True, skip_amc=True)
    assert not runner.called_with_prefix("systemctl", "stop", "mosquitto")
    assert teardown_calls == []


def test_pipeline_subcommand_stop_flag_dispatches_to_stop_pipeline(tmp_path, monkeypatch):
    ctx = FakeContext(tmp_path)
    _make_entry(ctx.install_dir)

    seen = {}

    def _fake_stop(ctx_, entry_, **kwargs):
        seen["kwargs"] = kwargs
        return 0

    monkeypatch.setattr(step5, "_stop_pipeline", _fake_stop)
    rc = step5.handle_pipeline_subcommand(["--project", "North Lobby #2", "--stop-all", "--no-mosquitto"], ctx)
    assert rc == 0
    assert seen["kwargs"]["stop_all"] is True
    assert seen["kwargs"]["skip_mosquitto"] is True


# ---------------------------------------------------------------------------
# section 5.4 -- reconciliation
# ---------------------------------------------------------------------------


def test_reconcile_registry_present_amc_dir_is_a_no_op(tmp_path, monkeypatch):
    ctx = FakeContext(tmp_path)
    entry = _make_entry(ctx.install_dir)

    amc_dir = tmp_path / "amc-projects" / entry.project_name
    amc_dir.mkdir(parents=True)
    monkeypatch.setattr(step5, "amc_project_dir", lambda ctx_, name: amc_dir)

    result = step5.reconcile_registry(ctx, apply=True, yes=True)
    assert result == []
    assert step5.get(ctx.install_dir, entry.project_name) is not None
    assert pathlib.Path(entry.exe).exists()


def test_reconcile_registry_absent_amc_dir_read_only_never_deletes(tmp_path, monkeypatch):
    ctx = FakeContext(tmp_path)
    entry = _make_entry(ctx.install_dir)

    missing_dir = tmp_path / "amc-projects" / "does-not-exist"
    monkeypatch.setattr(step5, "amc_project_dir", lambda ctx_, name: missing_dir)

    result = step5.reconcile_registry(ctx, apply=False)
    assert result == [entry.project_name]
    assert step5.get(ctx.install_dir, entry.project_name) is not None
    assert pathlib.Path(entry.exe).exists()


def test_reconcile_registry_applies_deletion_when_non_interactive(tmp_path, monkeypatch):
    ctx = FakeContext(tmp_path, non_interactive=True)
    entry = _make_entry(ctx.install_dir)

    missing_dir = tmp_path / "amc-projects" / "does-not-exist"
    monkeypatch.setattr(step5, "amc_project_dir", lambda ctx_, name: missing_dir)

    result = step5.reconcile_registry(ctx, apply=True)
    assert result == [entry.project_name]
    assert step5.get(ctx.install_dir, entry.project_name) is None
    assert not pathlib.Path(entry.exe).exists()


def test_reconcile_registry_interactive_requires_confirm(tmp_path, monkeypatch):
    ctx = FakeContext(tmp_path, non_interactive=False)
    entry = _make_entry(ctx.install_dir)

    missing_dir = tmp_path / "amc-projects" / "does-not-exist"
    monkeypatch.setattr(step5, "amc_project_dir", lambda ctx_, name: missing_dir)

    # Declined -> nothing removed.
    result = step5.reconcile_registry(ctx, apply=True, confirm=lambda _msg: False)
    assert result == []
    assert step5.get(ctx.install_dir, entry.project_name) is not None

    # Confirmed -> removed.
    result = step5.reconcile_registry(ctx, apply=True, confirm=lambda _msg: True)
    assert result == [entry.project_name]
    assert step5.get(ctx.install_dir, entry.project_name) is None


def test_reconcile_registry_yes_flag_skips_confirm(tmp_path, monkeypatch):
    ctx = FakeContext(tmp_path, non_interactive=False)
    entry = _make_entry(ctx.install_dir)

    missing_dir = tmp_path / "amc-projects" / "does-not-exist"
    monkeypatch.setattr(step5, "amc_project_dir", lambda ctx_, name: missing_dir)

    def _boom(_msg):  # pragma: no cover -- must never be called
        raise AssertionError("confirm() must not be called when yes=True")

    result = step5.reconcile_registry(ctx, apply=True, yes=True, confirm=_boom)
    assert result == [entry.project_name]


def _amc330_layout(tmp_path, *, project_dir=None):
    """An AMC 3.3.0 checkout whose compose/.env points PROJECT_DIR outside
    `<AMC_ROOT>/projects`, holding a project stored under its id."""
    amc_root = tmp_path / "home" / "op" / "amc-3.3.0" / "tools" / "auto-magic-calib"
    compose = amc_root / "compose"
    compose.mkdir(parents=True)
    projects = project_dir or (tmp_path / "amc-data" / "projects")
    projects.mkdir(parents=True, exist_ok=True)
    (compose / ".env").write_text(
        f"HOST_IP=127.0.0.1\nPROJECT_DIR={projects}\nMODEL_DIR={amc_root / 'models'}\n"
    )
    return amc_root, projects


def _amc330_ctx(tmp_path, amc_root, **kwargs):
    return FakeContext(
        tmp_path,
        conf={
            "AMC_ROOT": str(amc_root),
            "HOST_IP": "127.0.0.1",
            "PROJECT_NAME": "Living-Room",
            "AMC_PROJECT_ID": "20261023_091449_9720",
        },
        **kwargs,
    )


def _list_projects_call(args):
    return bool(args) and str(args[-1]).endswith("/v1/list_projects")


def test_amc_projects_root_reads_project_dir_from_compose_env(tmp_path):
    amc_root, projects = _amc330_layout(tmp_path)
    ctx = _amc330_ctx(tmp_path, amc_root)
    assert step5.amc_projects_root(ctx) == projects
    assert step5.amc_project_dir(ctx, "Living-Room") == projects / "Living-Room"


def test_amc_projects_root_falls_back_to_the_old_layout(tmp_path):
    amc_root = tmp_path / "auto-magic-calib"
    amc_root.mkdir()
    ctx = FakeContext(tmp_path, conf={"AMC_ROOT": str(amc_root), "HOST_IP": "127.0.0.1"})
    assert step5.amc_projects_root(ctx) == amc_root / "projects"


def test_amc_projects_root_ignores_an_unexpanded_reference(tmp_path):
    amc_root = tmp_path / "auto-magic-calib"
    (amc_root / "compose").mkdir(parents=True)
    (amc_root / "compose" / ".env").write_text('PROJECT_DIR="${AMC_ROOT}/projects"\n')
    ctx = FakeContext(tmp_path, conf={"AMC_ROOT": str(amc_root), "HOST_IP": "127.0.0.1"})
    assert step5.amc_projects_root(ctx) == amc_root / "projects"


def test_reconcile_finds_the_configured_project_by_id_on_disk(tmp_path):
    """Live regression: after the move to AMC 3.3.0 Step 5 logged "drift:
    registry entry 'Living-Room' has no matching AMC project dir" although
    the project existed under its AMC id in the compose PROJECT_DIR."""
    amc_root, projects = _amc330_layout(tmp_path)
    (projects / "20261023_091449_9720").mkdir()
    runner = ScriptedRunner().when(_list_projects_call, returncode=7)
    ctx = _amc330_ctx(tmp_path, amc_root, runner_root=runner)
    _make_entry(ctx.install_dir, project_name="Living-Room", slug="living-room")

    assert step5.reconcile_registry(ctx, apply=False) == []


def test_reconcile_prefers_the_amc_api_listing(tmp_path):
    amc_root, _projects = _amc330_layout(tmp_path)  # nothing on disk at all
    listing = {
        "code": 0,
        "projects": [
            {"project_id": "20261023_091449_9720", "project_name": "Living-Room"},
            {"project_id": "20261024_000000_0001", "project_name": "Lobby"},
        ],
    }
    runner = ScriptedRunner().when(_list_projects_call, stdout=json.dumps(listing))
    ctx = _amc330_ctx(tmp_path, amc_root, runner_root=runner)
    _make_entry(ctx.install_dir, project_name="Living-Room", slug="living-room")
    _make_entry(ctx.install_dir, project_name="Lobby", slug="lobby")
    _make_entry(ctx.install_dir, project_name="Gone", slug="gone")

    assert step5.reconcile_registry(ctx, apply=False) == ["Gone"]


def test_reconcile_api_listing_of_ids_only_falls_back_to_disk_for_names(tmp_path):
    """An id-only listing cannot rule out a project by name, so the disk
    check decides rather than reporting (and deleting) a live project."""
    amc_root, projects = _amc330_layout(tmp_path)
    (projects / "Lobby").mkdir()
    runner = ScriptedRunner().when(
        _list_projects_call, stdout=json.dumps(["20261023_091449_9720"])
    )
    ctx = _amc330_ctx(tmp_path, amc_root, runner_root=runner)
    _make_entry(ctx.install_dir, project_name="Living-Room", slug="living-room")
    _make_entry(ctx.install_dir, project_name="Lobby", slug="lobby")
    _make_entry(ctx.install_dir, project_name="Gone", slug="gone")

    assert step5.reconcile_registry(ctx, apply=False) == ["Gone"]


def test_reconcile_reports_a_project_the_api_no_longer_lists(tmp_path):
    amc_root, projects = _amc330_layout(tmp_path)
    (projects / "20261023_091449_9720").mkdir()
    listing = {"projects": [{"project_id": "other", "project_name": "Other"}]}
    runner = ScriptedRunner().when(_list_projects_call, stdout=json.dumps(listing))
    ctx = _amc330_ctx(tmp_path, amc_root, runner_root=runner)
    _make_entry(ctx.install_dir, project_name="Living-Room", slug="living-room")

    assert step5.reconcile_registry(ctx, apply=False) == ["Living-Room"]


# The live AMC 3.3.0 reply, verified on the workstation.
_LIVE_LIST_PROJECTS = {
    "code": 0,
    "message": "Projects listed successfully",
    "projects": [
        {
            "project_id": "20261023_091449_9720",
            "project_name": "Living-Room",
            "project_dir": "projects/project_20261023_091449_9720",
            "project_state": "COMPLETED",
        }
    ],
}


def test_live_list_projects_reply_resolves_living_room(tmp_path):
    amc_root, _projects = _amc330_layout(tmp_path)
    runner = ScriptedRunner().when(
        _list_projects_call, stdout=json.dumps(_LIVE_LIST_PROJECTS)
    )
    ctx = _amc330_ctx(tmp_path, amc_root, runner_root=runner)
    _make_entry(ctx.install_dir, project_name="Living-Room", slug="living-room")
    _make_entry(ctx.install_dir, project_name="Gone", slug="gone")

    assert step5.reconcile_registry(ctx, apply=False) == ["Gone"]


def test_disk_fallback_matches_the_amc330_project_folder(tmp_path):
    amc_root, projects = _amc330_layout(tmp_path)
    (projects / "project_20261023_091449_9720").mkdir()
    runner = ScriptedRunner().when(_list_projects_call, returncode=7)
    ctx = _amc330_ctx(tmp_path, amc_root, runner_root=runner)
    _make_entry(ctx.install_dir, project_name="Living-Room", slug="living-room")

    assert step5.reconcile_registry(ctx, apply=False) == []


def test_projects_subcommand_list_reports_drift_without_deleting(tmp_path, monkeypatch, capsys):
    ctx = FakeContext(tmp_path)
    entry = _make_entry(ctx.install_dir)
    missing_dir = tmp_path / "amc-projects" / "does-not-exist"
    monkeypatch.setattr(step5, "amc_project_dir", lambda ctx_, name: missing_dir)

    rc = step5.handle_projects_subcommand([], ctx)
    assert rc == 0
    assert step5.get(ctx.install_dir, entry.project_name) is not None


def test_projects_subcommand_reconcile_applies_under_yes(tmp_path, monkeypatch):
    ctx = FakeContext(tmp_path)
    entry = _make_entry(ctx.install_dir)
    missing_dir = tmp_path / "amc-projects" / "does-not-exist"
    monkeypatch.setattr(step5, "amc_project_dir", lambda ctx_, name: missing_dir)

    rc = step5.handle_projects_subcommand(["--reconcile", "--yes"], ctx)
    assert rc == 0
    assert step5.get(ctx.install_dir, entry.project_name) is None


def test_projects_subcommand_remove_removes_named_project(tmp_path):
    ctx = FakeContext(tmp_path)
    entry = _make_entry(ctx.install_dir)

    rc = step5.handle_projects_subcommand(["--remove", entry.project_name], ctx)
    assert rc == 0
    assert step5.get(ctx.install_dir, entry.project_name) is None
    assert not pathlib.Path(entry.exe).exists()


def test_projects_subcommand_remove_unknown_project_fails(tmp_path):
    ctx = FakeContext(tmp_path)
    rc = step5.handle_projects_subcommand(["--remove", "nope"], ctx)
    assert rc == 1


# ---------------------------------------------------------------------------
# section 3.2 -- generated wrapper content + idempotency
# ---------------------------------------------------------------------------


def test_write_pipeline_wrapper_is_content_idempotent(tmp_path):
    ctx = FakeContext(tmp_path)
    installer_bin = ctx.install_dir / "bin" / "mv3dt-installer"

    path1, changed1 = step5.write_pipeline_wrapper(ctx, installer_bin, "P1", "loc", "p1")
    assert changed1 is True
    assert "--project \"P1\"" in path1.read_text(encoding="utf-8")

    path2, changed2 = step5.write_pipeline_wrapper(ctx, installer_bin, "P1", "loc", "p1")
    assert changed2 is False
    assert path1 == path2


def test_write_record_wrapper_uses_record_subcommand(tmp_path):
    ctx = FakeContext(tmp_path)
    installer_bin = ctx.install_dir / "bin" / "mv3dt-installer"
    path, _changed = step5.write_record_wrapper(ctx, installer_bin, "P1", "loc", "p1")
    content = path.read_text(encoding="utf-8")
    assert '" record --project "P1"' in content


# ---------------------------------------------------------------------------
# generate_preview_config -- basic structural port check
# ---------------------------------------------------------------------------


def test_generate_preview_config_enables_display_and_keeps_mqtt():
    source = "\n".join(
        [
            "[source0]",
            "enable=1",
            "[source1]",
            "enable=1",
            "[sink1]",
            "enable=1",
            "msg-conv-config=msgconv_config.txt",
        ]
    )
    rendered = step5.generate_preview_config(source)
    assert "[tiled-display]" in rendered
    assert "rows=2" in rendered or "rows=1" in rendered
    assert "[sink0]" in rendered
    assert "enable=1" in rendered
    # sink1 (MQTT) config line is preserved.
    assert "msg-conv-config=msgconv_config.txt" in rendered


# ---------------------------------------------------------------------------
# STEP-6-REMOTE-SUPERVISION.md section A.2 -- supervised vs unsupervised
# pipeline dispatch. No test here imports step6_remote_supervision -- these
# exercise step5's own conditional purely through injected runners, exactly
# as an unsupervised workstation (Step 6 never run) must keep behaving.
# ---------------------------------------------------------------------------


def test_supervision_active_false_when_gate_off(tmp_path):
    ctx = FakeContext(tmp_path, conf={config_mod.GATE_REMOTE_SUPERVISION: "off"})
    assert step5._supervision_active(ctx, "north-lobby-2") is False
    # "off" must never even probe systemd.
    assert ctx.runner_root.calls == []


def test_supervision_active_false_when_gate_absent(tmp_path):
    ctx = FakeContext(tmp_path, conf={})
    assert step5._supervision_active(ctx, "north-lobby-2") is False
    assert ctx.runner_root.calls == []


def test_supervision_active_false_when_gate_on_but_unit_not_enabled(tmp_path):
    runner = ScriptedRunner(default_returncode=1)  # is-enabled fails: unit absent
    ctx = FakeContext(tmp_path, conf={config_mod.GATE_REMOTE_SUPERVISION: "local"}, runner_root=runner)
    assert step5._supervision_active(ctx, "north-lobby-2") is False


def test_supervision_active_true_when_gate_on_and_unit_enabled(tmp_path):
    runner = ScriptedRunner(default_returncode=0, default_stdout="enabled\n")  # is-enabled succeeds
    ctx = FakeContext(tmp_path, conf={config_mod.GATE_REMOTE_SUPERVISION: "remote"}, runner_root=runner)
    assert step5._supervision_active(ctx, "north-lobby-2") is True


def test_pipeline_start_dispatches_to_systemctl_when_supervised(tmp_path, monkeypatch):
    runner = ScriptedRunner(default_returncode=0, default_stdout="enabled\n")  # is-enabled AND systemctl start both succeed
    ctx = FakeContext(tmp_path, conf={config_mod.GATE_REMOTE_SUPERVISION: "local"}, runner_root=runner)
    _make_entry(ctx.install_dir)

    called = {}
    monkeypatch.setattr(
        step5, "_start_pipeline_foreground", lambda *a, **kw: (called.__setitem__("foreground", True), 0)[1]
    )

    rc = step5.handle_pipeline_subcommand(["--project", "North Lobby #2"], ctx)
    assert rc == 0
    assert "foreground" not in called
    assert runner.called_with_prefix("systemctl", "start", "mv3dt-pipeline@north-lobby-2.service")


def test_pipeline_start_stays_unsupervised_when_gate_off(tmp_path, monkeypatch):
    runner = ScriptedRunner()
    ctx = FakeContext(tmp_path, conf={}, runner_root=runner)
    _make_entry(ctx.install_dir)

    called = {}
    monkeypatch.setattr(
        step5, "_start_pipeline_foreground", lambda *a, **kw: (called.__setitem__("foreground", True), 0)[1]
    )

    rc = step5.handle_pipeline_subcommand(["--project", "North Lobby #2"], ctx)
    assert rc == 0
    assert called.get("foreground") is True
    assert not runner.called_with_prefix("systemctl", "start")


def test_pipeline_start_stays_unsupervised_when_gate_on_but_step6_never_ran(tmp_path, monkeypatch):
    """Gate flipped on but Step 6 has not (re)installed the unit for this
    project yet -- must behave exactly like an unsupervised workstation,
    never fail trying to systemctl-start a unit that doesn't exist."""
    runner = ScriptedRunner(default_returncode=1)  # is-enabled fails: no such unit
    ctx = FakeContext(tmp_path, conf={config_mod.GATE_REMOTE_SUPERVISION: "local"}, runner_root=runner)
    _make_entry(ctx.install_dir)

    called = {}
    monkeypatch.setattr(
        step5, "_start_pipeline_foreground", lambda *a, **kw: (called.__setitem__("foreground", True), 0)[1]
    )

    rc = step5.handle_pipeline_subcommand(["--project", "North Lobby #2"], ctx)
    assert rc == 0
    assert called.get("foreground") is True


def test_pipeline_start_foreground_flag_forces_unsupervised_even_when_active(tmp_path, monkeypatch):
    runner = ScriptedRunner(default_returncode=0)  # would say "supervised" if asked
    ctx = FakeContext(tmp_path, conf={config_mod.GATE_REMOTE_SUPERVISION: "remote"}, runner_root=runner)
    _make_entry(ctx.install_dir)

    called = {}
    monkeypatch.setattr(
        step5, "_start_pipeline_foreground", lambda *a, **kw: (called.__setitem__("foreground", True), 0)[1]
    )

    rc = step5.handle_pipeline_subcommand(["--project", "North Lobby #2", "--foreground"], ctx)
    assert rc == 0
    assert called.get("foreground") is True
    assert not runner.called_with_prefix("systemctl", "start")


def test_pipeline_stop_dispatches_to_systemctl_when_supervised(tmp_path):
    runner = ScriptedRunner(default_returncode=0, default_stdout="enabled\n")
    ctx = FakeContext(tmp_path, conf={config_mod.GATE_REMOTE_SUPERVISION: "local"}, runner_root=runner)
    _make_entry(ctx.install_dir)

    rc = step5.handle_pipeline_subcommand(["--project", "North Lobby #2", "--stop"], ctx)
    assert rc == 0
    assert runner.called_with_prefix("systemctl", "stop", "mv3dt-pipeline@north-lobby-2.service")
    assert not runner.called_with_prefix("pkill")


def test_pipeline_stop_all_supervised_still_tears_down_amc_and_mosquitto(tmp_path, monkeypatch):
    runner = ScriptedRunner(default_returncode=0, default_stdout="enabled\n")
    ctx = FakeContext(tmp_path, conf={config_mod.GATE_REMOTE_SUPERVISION: "local"}, runner_root=runner)
    _make_entry(ctx.install_dir)

    teardown_calls = []
    monkeypatch.setattr(step3, "teardown_amc", lambda ctx_: teardown_calls.append(ctx_))

    rc = step5.handle_pipeline_subcommand(["--project", "North Lobby #2", "--stop-all"], ctx)
    assert rc == 0
    assert runner.called_with_prefix("systemctl", "stop", "mv3dt-pipeline@north-lobby-2.service")
    assert len(teardown_calls) == 1
    assert runner.called_with_prefix("systemctl", "stop", "mosquitto")


def test_pipeline_service_exec_mode_skips_ping_sweep(tmp_path, monkeypatch):
    ctx = FakeContext(tmp_path, conf={})
    _make_entry(ctx.install_dir)

    ping_called = {}
    monkeypatch.setattr(step5, "ping_sweep_cameras", lambda ctx_: ping_called.setdefault("called", True))

    rc = step5.handle_pipeline_subcommand(
        ["--project", "North Lobby #2", "--service-exec", "--dry-run"], ctx
    )
    assert rc == 0
    assert "called" not in ping_called


def test_pipeline_registration_requires_root_predicate_false_for_service_exec():
    """unit U6's fix: `pipeline` is registered with a `requires_root`
    predicate (not a plain bool), and `--service-exec` -- the mode
    `mv3dt-pipeline@.service.in`'s own non-root `ExecStart=` invokes -- must
    resolve to `False` so `app._bootstrap_subcommand_context()` skips
    `privilege.require_root()` for it."""
    registration = app.SUBCOMMAND_REGISTRY["pipeline"]
    assert callable(registration.requires_root)
    assert registration.requires_root(["--project-slug", "north-lobby", "--service-exec"]) is False


def test_pipeline_registration_requires_root_predicate_true_for_other_modes():
    """Every other mode -- default start, `--stop`, `--stop-all`,
    `--foreground`, `--dry-run` -- must keep requiring root, unchanged from
    Step 5's original behavior (STEP-6's `--stop` still shells out to
    `systemctl stop` via `ctx.run_root`)."""
    registration = app.SUBCOMMAND_REGISTRY["pipeline"]
    assert registration.requires_root(["--project", "north-lobby"]) is True
    assert registration.requires_root(["--project", "north-lobby", "--stop"]) is True
    assert registration.requires_root(["--project", "north-lobby", "--stop-all"]) is True
    assert registration.requires_root(["--project", "north-lobby", "--foreground"]) is True
    assert registration.requires_root(["--project", "north-lobby", "--dry-run"]) is True


def test_pipeline_project_slug_resolves_via_registry(tmp_path):
    ctx = FakeContext(tmp_path, conf={})
    entry = _make_entry(ctx.install_dir)

    rc = step5.handle_pipeline_subcommand(["--project-slug", entry.slug, "--dry-run"], ctx)
    assert rc == 0


def test_pipeline_neither_project_nor_slug_is_an_error(tmp_path):
    ctx = FakeContext(tmp_path, conf={})
    rc = step5.handle_pipeline_subcommand([], ctx)
    assert rc == 1


def test_get_by_slug_finds_the_matching_entry(tmp_path):
    entry = _make_entry(tmp_path)
    assert step5.get_by_slug(tmp_path, entry.slug) is entry or step5.get_by_slug(
        tmp_path, entry.slug
    ).slug == entry.slug


def test_get_by_slug_returns_none_for_unknown_slug(tmp_path):
    _make_entry(tmp_path)
    assert step5.get_by_slug(tmp_path, "does-not-exist") is None


# ---------------------------------------------------------------------------
# STEP-6-REMOTE-SUPERVISION.md section A.3 -- the removal hook plumbing
# ---------------------------------------------------------------------------


def test_removal_hook_defaults_to_none_and_is_a_noop():
    previous = step5._REMOVAL_HOOK
    try:
        step5.register_removal_hook(None)
        assert step5._REMOVAL_HOOK is None
    finally:
        step5.register_removal_hook(previous)


def test_remove_project_artifacts_invokes_the_registered_hook(tmp_path):
    ctx = FakeContext(tmp_path, conf={})
    entry = _make_entry(ctx.install_dir)

    calls = []
    previous = step5._REMOVAL_HOOK
    try:
        step5.register_removal_hook(lambda ctx_, entry_: calls.append(entry_.slug))
        step5.remove_project_artifacts(ctx, entry)
        assert calls == [entry.slug]
    finally:
        step5.register_removal_hook(previous)


# ---------------------------------------------------------------------------
# DETECTION-EXCLUSION-ZONES sections 4.4, 5 and 6 -- the filter app
# ---------------------------------------------------------------------------

_DS_VERSION = "9.1.0"

_TEST5_MAIN = """\
#include <glib.h>
#include "deepstream_test5_app.h"
#include "deepstream_utc.h"

int main (int argc, char *argv[])
{
  for (i = 0; i < num_instances; i++) {
    if (!create_pipeline (appCtx[i], bbox_generated_probe_after_analytics,
            NULL, perf_cb, overlay_graphics)) {
      NVGSTDS_ERR_MSG_V ("Failed to create pipeline");
      return_value = -1;
      goto done;
    }
    /* Override default probe function */
  }
done:
  return return_value;
}
"""

_TEST5_MAKEFILE = """\
APP:= deepstream-test5-app
SRCS:= deepstream_test5_app_main.c deepstream_utc.c
SRCS+= $(wildcard /opt/nvidia/deepstream/deepstream/sources/libs/nvds_utils/*.c)
CFLAGS+= -I/opt/nvidia/deepstream/deepstream/sources/libs/kafka_client
"""


@pytest.fixture(autouse=True)
def _no_real_deepstream(tmp_path, monkeypatch):
    """No test reads /opt/nvidia or /usr/local/cuda; tests that need a
    DeepStream tree build one with `_fake_deepstream`."""
    monkeypatch.setattr(step5, "DEEPSTREAM_ROOT", tmp_path / "no-deepstream")
    monkeypatch.setattr(step5, "CUDA_LINK", tmp_path / "no-cuda")


def _fake_deepstream(tmp_path, monkeypatch, *, main=_TEST5_MAIN, makefile=_TEST5_MAKEFILE, version=_DS_VERSION):
    root = tmp_path / "deepstream"
    app_dir = root / "sources" / "apps" / "sample_apps" / "deepstream-test5"
    app_dir.mkdir(parents=True)
    (app_dir / "deepstream_test5_app_main.c").write_text(main)
    (app_dir / "Makefile").write_text(makefile)
    (root / "sources" / "includes").mkdir()
    (root / "sources" / "includes" / "nvdsmeta.h").write_text("/* header */\n")
    (root / "sources" / "libs" / "nvds_utils").mkdir(parents=True)
    (root / "sources" / "libs" / "nvds_utils" / "u.c").write_text("/* lib */\n")
    if version is not None:
        (root / "version").write_text(f"Version: {version}\nGCID: 1\n")
    cuda_dir = tmp_path / "cuda-13.2"
    cuda_dir.mkdir()
    link = tmp_path / "cuda"
    link.symlink_to(cuda_dir)
    monkeypatch.setattr(step5, "DEEPSTREAM_ROOT", root)
    monkeypatch.setattr(step5, "CUDA_LINK", link)
    return root


class _BuildRunner(ScriptedRunner):
    """dpkg-query reports the build dependencies installed unless told
    otherwise, and `make -C <dir>` drops a binary into `<dir>` when it
    succeeds."""

    def __init__(self, *, make_rc=0, make_stdout="", make_stderr="", installed=True, apt_rc=0):
        super().__init__()
        self.make_rc = make_rc
        self.make_stdout = make_stdout
        self.make_stderr = make_stderr
        self.installed = installed
        self.apt_rc = apt_rc

    def __call__(self, *args, **kwargs):
        self.calls.append(args)
        if args[0] == "dpkg-query":
            out = "install ok installed" if self.installed else ""
            return subprocess.CompletedProcess(list(args), 0 if self.installed else 1, out, "")
        if args[0] == "apt-get":
            return subprocess.CompletedProcess(list(args), self.apt_rc, "", "E: Unable to locate package")
        if args[0] == "make":
            if self.make_rc == 0:
                app_dir = pathlib.Path(args[2])
                (app_dir / "deepstream-test5-app").write_bytes(b"\x7fELF built")
            return subprocess.CompletedProcess(list(args), self.make_rc, self.make_stdout, self.make_stderr)
        return subprocess.CompletedProcess(list(args), 0, "", "")


def _expected_key(version=_DS_VERSION):
    digest = hashlib.sha256()
    names = sorted(
        ["mv3dt_exclusion.c", "mv3dt_exclusion.h", "mv3dt_exclusion_rules.c", "mv3dt_exclusion_rules.h"]
    )
    for name in names:
        digest.update(shellout.asset_path("deepstream", "exclusion", name).read_bytes())
    return f"{digest.hexdigest()} {step5.EXCLUSION_PATCH_VERSION} {version}"


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


# -- section 4.4: the hook ---------------------------------------------------


def test_patch_test5_main_adds_include_and_attach():
    patched = step5.patch_test5_main(_TEST5_MAIN, _DS_VERSION)
    assert '#include "deepstream_test5_app.h"\n#include "mv3dt_exclusion.h"\n' in patched
    assert (
        "      goto done;\n"
        "    }\n"
        "    mv3dt_exclusion_attach (appCtx[i]->pipeline.common_elements.\n"
        "        primary_gie_bin.bin);\n"
        "    /* Override default probe function */\n"
    ) in patched
    assert patched.count("mv3dt_exclusion_attach") == 1


@pytest.mark.parametrize(
    "broken, anchor",
    [
        (_TEST5_MAIN.replace('#include "deepstream_test5_app.h"\n', ""), "deepstream_test5_app.h"),
        (
            _TEST5_MAIN.replace(
                '#include "deepstream_test5_app.h"\n',
                '#include "deepstream_test5_app.h"\n#include "deepstream_test5_app.h"\n',
            ),
            "deepstream_test5_app.h",
        ),
        (_TEST5_MAIN.replace("      goto done;\n", "      exit (1);\n"), "Failed to create pipeline"),
    ],
)
def test_patch_test5_main_missing_or_repeated_anchor_names_anchor_and_version(broken, anchor):
    with pytest.raises(step5.ExclusionBuildError) as info:
        step5.patch_test5_main(broken, _DS_VERSION)
    assert info.value.stage == "patch"
    assert anchor in info.value.detail
    assert "DeepStream 9.1.0" in info.value.detail


def test_patch_test5_makefile_adds_sources_and_relativizes_libs():
    patched = step5.patch_test5_makefile(_TEST5_MAKEFILE, _DS_VERSION)
    assert (
        "SRCS:= deepstream_test5_app_main.c deepstream_utc.c mv3dt_exclusion.c mv3dt_exclusion_rules.c\n"
    ) in patched
    assert "/opt/nvidia" not in patched
    assert patched.count("../../../libs") == 2


@pytest.mark.parametrize(
    "broken, anchor",
    [
        (
            _TEST5_MAKEFILE.replace("SRCS:= deepstream_test5_app_main.c deepstream_utc.c", "SRCS:= main.c"),
            "SRCS:=",
        ),
        (
            _TEST5_MAKEFILE.replace("/opt/nvidia/deepstream/deepstream/sources/libs", "/elsewhere"),
            "/opt/nvidia",
        ),
    ],
)
def test_patch_test5_makefile_missing_anchor_fails(broken, anchor):
    with pytest.raises(step5.ExclusionBuildError) as info:
        step5.patch_test5_makefile(broken, _DS_VERSION)
    assert anchor in info.value.detail
    assert "DeepStream 9.1.0" in info.value.detail


# -- versions and the build key ----------------------------------------------


def test_cuda_version_follows_the_link_chain(tmp_path):
    target = tmp_path / "cuda-13.2"
    target.mkdir()
    alt = tmp_path / "alternatives-cuda"
    alt.symlink_to(target)
    link = tmp_path / "cuda"
    link.symlink_to(alt)
    assert step5.cuda_version(link) == "13.2"


@pytest.mark.parametrize("name", ["cuda", "cuda-13", "cuda-13.2.1", "cuda13.2"])
def test_cuda_version_rejects_other_names(tmp_path, name):
    target = tmp_path / name
    target.mkdir()
    assert step5.cuda_version(target) is None


def test_deepstream_version_reads_the_version_line(tmp_path):
    (tmp_path / "version").write_text("GCID: 123\nVersion: 9.1.0\n")
    assert step5.deepstream_version(tmp_path) == "9.1.0"


def test_deepstream_version_none_when_missing(tmp_path):
    assert step5.deepstream_version(tmp_path) is None
    (tmp_path / "version").write_text("nothing useful\n")
    assert step5.deepstream_version(tmp_path) is None


def test_build_key_hashes_the_sorted_bundled_files(tmp_path):
    ctx = FakeContext(tmp_path)
    key = step5.exclusion_build_key(ctx, ds_version=_DS_VERSION)
    assert key == _expected_key()
    assert key.split(" ")[1:] == [str(step5.EXCLUSION_PATCH_VERSION), _DS_VERSION]
    assert step5.exclusion_build_key(ctx, ds_version="9.2.0") != key


def test_build_key_changes_with_the_patch_version(tmp_path, monkeypatch):
    ctx = FakeContext(tmp_path)
    before = step5.exclusion_build_key(ctx, ds_version=_DS_VERSION)
    monkeypatch.setattr(step5, "EXCLUSION_PATCH_VERSION", step5.EXCLUSION_PATCH_VERSION + 1)
    assert step5.exclusion_build_key(ctx, ds_version=_DS_VERSION) != before


def test_build_key_none_without_deepstream(tmp_path):
    assert step5.exclusion_build_key(FakeContext(tmp_path)) is None


# -- section 5.2: pending_update ---------------------------------------------


def test_pending_update_none_without_deepstream(tmp_path):
    assert step5.Step5PerProjectExes().pending_update(FakeContext(tmp_path)) is None


def test_pending_update_without_a_stamp(tmp_path, monkeypatch):
    _fake_deepstream(tmp_path, monkeypatch)
    ctx = FakeContext(tmp_path)
    assert step5.Step5PerProjectExes().pending_update(ctx) == "bundled detection filter changed"
    assert ctx.runner_root.calls == []


def test_pending_update_none_when_stamp_matches(tmp_path, monkeypatch):
    _fake_deepstream(tmp_path, monkeypatch)
    ctx = FakeContext(tmp_path)
    _write(step5.exclusion_stamp_path(ctx.install_dir), _expected_key() + "\n")
    assert step5.Step5PerProjectExes().pending_update(ctx) is None


def test_pending_update_none_when_this_key_already_failed(tmp_path, monkeypatch):
    _fake_deepstream(tmp_path, monkeypatch)
    ctx = FakeContext(tmp_path)
    _write(step5.exclusion_stamp_path(ctx.install_dir), "old 1 9.0.0\n")
    _write(step5.exclusion_failed_path(ctx.install_dir), _expected_key() + "\n")
    assert step5.Step5PerProjectExes().pending_update(ctx) is None


def test_pending_update_when_an_older_key_failed(tmp_path, monkeypatch):
    _fake_deepstream(tmp_path, monkeypatch)
    ctx = FakeContext(tmp_path)
    _write(step5.exclusion_stamp_path(ctx.install_dir), "old 1 9.0.0\n")
    _write(step5.exclusion_failed_path(ctx.install_dir), "older 1 9.0.0\n")
    assert step5.Step5PerProjectExes().pending_update(ctx) == step5.EXCLUSION_UPDATE_REASON


def test_pending_update_works_on_the_status_view(tmp_path, monkeypatch):
    """`--status` passes a bare namespace: no runner, no log."""
    _fake_deepstream(tmp_path, monkeypatch)
    view = SimpleNamespace(install_dir=tmp_path / "mv3dt", conf={}, asset_path=shellout.asset_path, user=None)
    assert step5.Step5PerProjectExes().pending_update(view) == step5.EXCLUSION_UPDATE_REASON


# -- sections 5.1 and 5.3: the build -----------------------------------------


def test_build_installs_the_app_and_writes_the_stamp(tmp_path, monkeypatch):
    _fake_deepstream(tmp_path, monkeypatch)
    runner = _BuildRunner()
    ctx = FakeContext(tmp_path, runner_root=runner)
    stale = ctx.install_dir / "build" / "test5-exclusion" / "stale.o"
    _write(stale, "old")
    _write(step5.exclusion_failed_path(ctx.install_dir), "old 1 9.0.0\n")

    step5.ensure_exclusion_app(ctx)

    sources = ctx.install_dir / "build" / "test5-exclusion" / "sources"
    app_dir = sources / "apps" / "sample_apps" / "deepstream-test5"
    assert not stale.exists()
    assert (sources / "includes" / "nvdsmeta.h").is_file()
    assert (sources / "libs" / "nvds_utils" / "u.c").is_file()
    for name in step5.EXCLUSION_SOURCE_FILES:
        bundled = shellout.asset_path("deepstream", "exclusion", name).read_bytes()
        assert (app_dir / name).read_bytes() == bundled
    assert '#include "mv3dt_exclusion.h"' in (app_dir / "deepstream_test5_app_main.c").read_text()
    assert "../../../libs" in (app_dir / "Makefile").read_text()

    make = [c for c in runner.calls if c[0] == "make"]
    assert make == [("make", "-C", str(app_dir), f"-j{os.cpu_count() or 1}", "CUDA_VER=13.2")]
    assert not runner.called_with_prefix("apt-get")

    built = step5.exclusion_app_path(ctx.install_dir)
    assert built.read_bytes() == b"\x7fELF built"
    assert built.stat().st_mode & 0o777 == 0o755
    assert step5.exclusion_stamp_path(ctx.install_dir).read_text().strip() == _expected_key()
    assert not step5.exclusion_failed_path(ctx.install_dir).exists()
    assert step5.Step5PerProjectExes().pending_update(ctx) is None


def test_build_skipped_when_the_stamp_is_current(tmp_path, monkeypatch):
    _fake_deepstream(tmp_path, monkeypatch)
    runner = _BuildRunner()
    ctx = FakeContext(tmp_path, runner_root=runner)
    _write(step5.exclusion_app_path(ctx.install_dir), "bin")
    _write(step5.exclusion_stamp_path(ctx.install_dir), _expected_key())
    step5.ensure_exclusion_app(ctx)
    assert runner.calls == []


def test_build_runs_when_the_stamp_matches_but_the_binary_is_gone(tmp_path, monkeypatch):
    _fake_deepstream(tmp_path, monkeypatch)
    runner = _BuildRunner()
    ctx = FakeContext(tmp_path, runner_root=runner)
    _write(step5.exclusion_stamp_path(ctx.install_dir), _expected_key())
    step5.ensure_exclusion_app(ctx)
    assert runner.called_with_prefix("make")
    assert step5.exclusion_app_path(ctx.install_dir).is_file()


def test_build_installs_missing_dependencies(tmp_path, monkeypatch):
    _fake_deepstream(tmp_path, monkeypatch)
    runner = _BuildRunner(installed=False)
    ctx = FakeContext(tmp_path, runner_root=runner)
    step5.ensure_exclusion_app(ctx)
    assert runner.called_with_prefix(
        "apt-get", "install", "-y", "--no-install-recommends", "libjson-glib-dev", "libgstrtspserver-1.0-dev"
    )
    assert step5.exclusion_app_path(ctx.install_dir).is_file()


def _previous_install(ctx):
    _write(step5.exclusion_app_path(ctx.install_dir), "previous")
    step5.exclusion_app_path(ctx.install_dir).chmod(0o755)
    _write(step5.exclusion_stamp_path(ctx.install_dir), "old 1 9.0.0\n")


def _assert_failed_untouched(ctx, capsys, stage):
    err = capsys.readouterr().err
    assert f"build failed at {stage}" in err
    assert step5.exclusion_app_path(ctx.install_dir).read_text() == "previous"
    assert step5.exclusion_stamp_path(ctx.install_dir).read_text() == "old 1 9.0.0\n"
    assert step5.exclusion_failed_path(ctx.install_dir).read_text().strip() == _expected_key()
    return err


def test_compile_failure_warns_and_keeps_the_previous_build(tmp_path, monkeypatch, capsys):
    _fake_deepstream(tmp_path, monkeypatch)
    runner = _BuildRunner(
        make_rc=2,
        make_stdout="cc -c deepstream_test5_app_main.c\n",
        make_stderr=(
            "mv3dt_exclusion.c:12:3: error: unknown type name 'Foo'\n"
            "make: *** [Makefile:80: mv3dt_exclusion.o] Error 1\n"
        ),
    )
    ctx = FakeContext(tmp_path, runner_root=runner)
    _previous_install(ctx)
    step5.ensure_exclusion_app(ctx)
    err = _assert_failed_untouched(ctx, capsys, "compile")
    assert "mv3dt_exclusion.c:12:3: error: unknown type name 'Foo'" in err
    # The failed key stops the update check from retrying on every run.
    assert step5.Step5PerProjectExes().pending_update(ctx) is None


def test_anchor_failure_warns_without_compiling(tmp_path, monkeypatch, capsys):
    _fake_deepstream(tmp_path, monkeypatch, main=_TEST5_MAIN.replace("goto done;", "goto out;"))
    runner = _BuildRunner()
    ctx = FakeContext(tmp_path, runner_root=runner)
    _previous_install(ctx)
    step5.ensure_exclusion_app(ctx)
    err = _assert_failed_untouched(ctx, capsys, "patch")
    assert "DeepStream 9.1.0" in err
    assert not runner.called_with_prefix("make")


def test_dependency_failure_retries_after_update_then_warns(tmp_path, monkeypatch, capsys):
    _fake_deepstream(tmp_path, monkeypatch)
    runner = _BuildRunner(installed=False, apt_rc=100)
    ctx = FakeContext(tmp_path, runner_root=runner)
    _previous_install(ctx)
    step5.ensure_exclusion_app(ctx)
    _assert_failed_untouched(ctx, capsys, "dependencies")
    assert runner.called_with_prefix("apt-get", "update")
    assert sum(1 for c in runner.calls if c[:2] == ("apt-get", "install")) == 2
    assert not runner.called_with_prefix("make")


def test_stamp_write_failure_after_install_is_a_warning(tmp_path, monkeypatch, capsys):
    _fake_deepstream(tmp_path, monkeypatch)
    ctx = FakeContext(tmp_path, runner_root=_BuildRunner())

    def _fail(path, key):
        raise OSError("read-only file system")

    monkeypatch.setattr(step5, "_write_key", _fail)
    assert step5.build_exclusion_app(ctx, _expected_key(), _DS_VERSION) is True
    assert "could not record its build stamp" in capsys.readouterr().err
    assert step5.exclusion_app_path(ctx.install_dir).is_file()


def test_failed_file_removal_error_is_a_warning(tmp_path, monkeypatch, capsys):
    _fake_deepstream(tmp_path, monkeypatch)
    ctx = FakeContext(tmp_path, runner_root=_BuildRunner())
    failed = step5.exclusion_failed_path(ctx.install_dir)
    _write(failed, "old 1 9.0.0\n")
    real_unlink = pathlib.Path.unlink

    def _unlink(self, *args, **kwargs):
        if self == failed:
            raise PermissionError("denied")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "unlink", _unlink)
    assert step5.build_exclusion_app(ctx, _expected_key(), _DS_VERSION) is True
    assert "could not record its build stamp" in capsys.readouterr().err
    assert step5.exclusion_stamp_path(ctx.install_dir).read_text().strip() == _expected_key()


def test_run_stays_complete_when_the_stamp_cannot_be_written(tmp_path, monkeypatch):
    _fake_deepstream(tmp_path, monkeypatch)
    ctx = FakeContext(tmp_path, runner_root=_BuildRunner())
    ctx.update_pending = step5.EXCLUSION_UPDATE_REASON
    monkeypatch.setattr(step5, "_write_key", lambda path, key: (_ for _ in ()).throw(OSError("full")))
    result = step5.Step5PerProjectExes().run(ctx)
    assert result.status is StepStatus.COMPLETE


def test_missing_cuda_link_is_a_compile_failure(tmp_path, monkeypatch, capsys):
    _fake_deepstream(tmp_path, monkeypatch)
    monkeypatch.setattr(step5, "CUDA_LINK", tmp_path / "cuda-missing")
    ctx = FakeContext(tmp_path, runner_root=_BuildRunner())
    step5.ensure_exclusion_app(ctx)
    assert "build failed at compile" in capsys.readouterr().err
    assert not step5.exclusion_app_path(ctx.install_dir).exists()


def test_no_deepstream_warns_and_runs_nothing(tmp_path, capsys):
    ctx = FakeContext(tmp_path)
    step5.ensure_exclusion_app(ctx)
    assert "not built" in capsys.readouterr().err
    assert ctx.runner_root.calls == []


def _inventory(tmp_path, *, enabled=True, boxes='"530,175,245,225"'):
    path = tmp_path / "cameras.yml"
    path.write_text(
        "cameras:\n"
        "  - id: c2\n"
        "    ip: 169.254.124.198\n"
        "    rtsp_path: /Streaming/Channels/101\n"
        f"    enabled: {'true' if enabled else 'false'}\n"
        + (f"    exclude_boxes: {boxes}\n" if boxes else "")
    )
    return {config_mod.CAMERAS_FILE_KEY: str(path)}


def test_inactive_zones_warning_when_no_app_exists(tmp_path, capsys):
    ctx = FakeContext(tmp_path, conf=_inventory(tmp_path))
    step5.ensure_exclusion_app(ctx)
    assert "exclusion zones are inactive" in capsys.readouterr().err


@pytest.mark.parametrize("kwargs", [{"enabled": False}, {"boxes": ""}])
def test_no_inactive_zones_warning_without_enabled_boxes(tmp_path, capsys, kwargs):
    ctx = FakeContext(tmp_path, conf=_inventory(tmp_path, **kwargs))
    step5.ensure_exclusion_app(ctx)
    assert "exclusion zones are inactive" not in capsys.readouterr().err


def test_no_inactive_zones_warning_when_an_older_app_exists(tmp_path, capsys):
    ctx = FakeContext(tmp_path, conf=_inventory(tmp_path))
    _previous_install(ctx)
    step5.ensure_exclusion_app(ctx)
    assert "exclusion zones are inactive" not in capsys.readouterr().err


# -- Step 5 run(): where the build happens -----------------------------------


def _run_ctx(tmp_path, monkeypatch):
    conf = {"PROJECT_NAME": "Living Room", "LOCATION_ID": "loc-1"}
    ctx = FakeContext(tmp_path, conf=conf)
    _write(ctx.install_dir / "deepstream" / "deepstream_app_config.rendered.txt", "[application]\n")
    installer_bin = ctx.install_dir / "bin" / "mv3dt-installer"
    monkeypatch.setattr(step3, "ensure_installer_binary", lambda ctx_: installer_bin)
    events = []
    monkeypatch.setattr(step5, "ensure_exclusion_app", lambda ctx_: events.append("build"))
    monkeypatch.setattr(step5, "_confirm_start_now", lambda ctx_: events.append("prompt") or False)
    return ctx, events


def test_run_builds_the_filter_before_the_start_or_close_question(tmp_path, monkeypatch):
    ctx, events = _run_ctx(tmp_path, monkeypatch)
    result = step5.Step5PerProjectExes().run(ctx)
    assert result.status is StepStatus.COMPLETE
    assert events == ["build", "prompt"]
    assert step5.get(ctx.install_dir, "Living Room") is not None


def test_update_run_only_builds_the_filter(tmp_path, monkeypatch):
    ctx, events = _run_ctx(tmp_path, monkeypatch)
    ctx.update_pending = step5.EXCLUSION_UPDATE_REASON
    result = step5.Step5PerProjectExes().run(ctx)
    assert result.status is StepStatus.COMPLETE
    assert events == ["build"]
    assert step5.get(ctx.install_dir, "Living Room") is None


def test_filter_phase_is_declared():
    assert step5.Step5PerProjectExes.phases[3] == "detection exclusion filter"


# -- section 6: launch -------------------------------------------------------


def test_pipeline_app_prefers_the_built_filter_app(tmp_path):
    ctx = FakeContext(tmp_path)
    assert step5.pipeline_app(ctx) == "deepstream-test5-app"
    built = step5.exclusion_app_path(ctx.install_dir)
    _write(built, "bin")
    built.chmod(0o644)
    assert step5.pipeline_app(ctx) == "deepstream-test5-app"
    built.chmod(0o755)
    assert step5.pipeline_app(ctx) == str(built)


def _launch(tmp_path, monkeypatch, capsys, *, built=False, conf_file=False, service=False):
    monkeypatch.delenv(step5.EXCLUSION_CONFIG_ENV, raising=False)
    monkeypatch.delenv(config_mod.GATE_VIEWER_STREAM, raising=False)
    ctx = FakeContext(tmp_path)
    entry = _make_entry(ctx.install_dir)
    if built:
        _write(step5.exclusion_app_path(ctx.install_dir), "bin")
        step5.exclusion_app_path(ctx.install_dir).chmod(0o755)
    if conf_file:
        _write(ctx.install_dir / "deepstream" / "exclusion.conf", "[source1]\nrects=1,1,2,2\n")
    monkeypatch.setattr(step5, "ensure_mosquitto", lambda ctx_: None)
    monkeypatch.setattr(step5, "ping_sweep_cameras", lambda ctx_: None)
    monkeypatch.setattr(step5, "source_deepstream_env", lambda ctx_: {})
    monkeypatch.setattr(step5.os, "chdir", lambda path: None)
    seen = []

    def _execv(binary, argv):
        seen.append((binary, argv, os.environ.get(step5.EXCLUSION_CONFIG_ENV)))

    start = step5._start_pipeline_service_exec if service else step5._start_pipeline_foreground
    assert start(ctx, entry, execv=_execv) == 0
    return ctx, entry, seen[0], capsys.readouterr().err


@pytest.mark.parametrize("service", [False, True])
def test_launch_uses_the_built_app_and_exclusion_config(tmp_path, monkeypatch, capsys, service):
    ctx, entry, (binary, argv, env), err = _launch(
        tmp_path, monkeypatch, capsys, built=True, conf_file=True, service=service
    )
    built = str(step5.exclusion_app_path(ctx.install_dir))
    assert binary == built
    assert argv == [built, "-c", entry.rendered_config]
    assert env == str(ctx.install_dir / "deepstream" / "exclusion.conf")
    assert f"Launching: {built} -c" in err


@pytest.mark.parametrize("service", [False, True])
def test_launch_falls_back_to_the_stock_app_without_env(tmp_path, monkeypatch, capsys, service):
    _, entry, (_binary, argv, env), err = _launch(tmp_path, monkeypatch, capsys, service=service)
    assert argv == ["deepstream-test5-app", "-c", entry.rendered_config]
    assert env is None
    assert "Launching: deepstream-test5-app -c" in err


def test_process_pattern_matches_both_apps_exactly():
    assert len(step5.EXCLUSION_APP_NAME) == 15
    pattern = step5.DEEPSTREAM_PROCESS_PATTERN
    assert pattern == "deepstream-test|mv3dt-test5-app"
    # procps anchors the whole pattern under -x.
    anchored = re.compile(f"^({pattern})$")
    assert anchored.match("deepstream-test")
    assert anchored.match("mv3dt-test5-app")
    assert not anchored.match("deepstream-tes")
    assert not anchored.match("mv3dt-test5-ap")


def test_pgrep_matches_both_process_names(tmp_path):
    runner = ScriptedRunner(default_returncode=1)
    ctx = FakeContext(tmp_path, runner_root=runner)
    step5._pgrep_deepstream_running(ctx)
    assert runner.calls == [("pgrep", "-x", "deepstream-test|mv3dt-test5-app")]
