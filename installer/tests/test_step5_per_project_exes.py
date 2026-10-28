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

import json
import pathlib
import subprocess
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from mv3dt_installer import app  # noqa: E402
from mv3dt_installer import config as config_mod  # noqa: E402
from mv3dt_installer import logs, report  # noqa: E402
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
    assert runner.called_with_prefix("pkill", "-TERM", "-x", step5.DEEPSTREAM_APP_PROCESS)
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
    assert runner.called_with_prefix("pkill", "-TERM", "-x", step5.DEEPSTREAM_APP_PROCESS)
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
    assert runner.called_with_prefix("pkill", "-KILL", "-x", step5.DEEPSTREAM_APP_PROCESS)

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
