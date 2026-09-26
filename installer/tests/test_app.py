"""Tests for `mv3dt_installer.app` (doc 00 §3, §12.3).

Run from installer/: `python3 -m pytest tests/test_app.py -v`

Never touches the real `/var/lib/mv3dt-installer/state.json`,
`/var/lib/mv3dt-installer/logs/`, or `/opt/mv3dt` -- every test passes
`state_path=` (a test-only override on `app.main()`, see its docstring) and
`--install-dir`/`--log-dir` pointing under `tmp_path`. `--status` is the one
flow that deliberately runs before `privilege.require_root()` (see
`app.main()`'s docstring for why); every other flow in these tests
monkeypatches `privilege.require_root` to a no-op, since these tests exist
to exercise app.py's own logic, not re-prove doc 00 §9.1's "must run as
root" requirement (a dedicated test below does confirm that gate exists).
"""

from __future__ import annotations

import io
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mv3dt_installer import __version__  # noqa: E402
from mv3dt_installer import app
from mv3dt_installer import logs as logs_mod  # noqa: E402
from mv3dt_installer import cameras as cameras_mod  # noqa: E402
from mv3dt_installer import config as config_mod  # noqa: E402
from mv3dt_installer.privilege import InvokingUser  # noqa: E402
from mv3dt_installer.state import STEP_IDS, StateMachine  # noqa: E402
from mv3dt_installer.steps import (  # noqa: E402
    StepResult,
    StepStatus,
    UserAction,
)


@pytest.fixture(autouse=True)
def _force_no_colour(monkeypatch):
    """Match test_logs.py / test_privilege.py's convention, so the stderr
    assertions below see plain, un-escaped lines."""
    monkeypatch.setattr(sys.stderr, "isatty", lambda: False)


@pytest.fixture(autouse=True)
def _reset_transcript_state():
    from mv3dt_installer import logs

    logs._transcript_path = None
    yield
    logs._transcript_path = None


def _bypass_onboarding(monkeypatch, tmp_path: Path) -> None:
    """These tests exist to exercise app.py's own dispatch/config/reboot
    logic end-to-end through `app.main()`, not onboarding's platform
    preflight or NGC key capture (each has its own dedicated tests in
    test_preflight.py / test_ngc.py / test_onboarding.py). Real platform
    preflight would make these tests depend on the runner's actual OS/arch
    and invoking user; a real NGC key prompt would hang or die under
    `--non-interactive` with no key available. Bypass both the same way
    `require_root` is already bypassed above.
    """
    monkeypatch.setattr(
        app.onboarding,
        "run_platform_preflight",
        lambda: InvokingUser(
            name="tester", home=tmp_path, uid=os.getuid(), gid=os.getgid()
        ),
    )
    monkeypatch.setenv("NGC_API_KEY", "nvapi-test-key-do-not-use")
    monkeypatch.setenv("CAM_USER", "admin")
    monkeypatch.setenv("CAM_PASSWORD", "camera-test-password")


# ---------------------------------------------------------------------------
# Test double satisfying the `Step` protocol (doc 00 §12.1)
# ---------------------------------------------------------------------------


class _DummyStep:
    def __init__(
        self,
        step_id: str,
        title: str,
        order: int,
        *,
        preflight: StepResult | None = None,
        run: StepResult | None = None,
        verify: StepResult | None = None,
    ) -> None:
        self.id = step_id
        self.title = title
        self.order = order
        self._preflight_result = preflight or StepResult(status=StepStatus.COMPLETE)
        self._run_result = run or StepResult(status=StepStatus.COMPLETE)
        self._verify_result = verify or StepResult(status=StepStatus.COMPLETE)
        self.report_calls = 0

    def preflight(self, ctx):
        return self._preflight_result

    def run(self, ctx):
        return self._run_result

    def verify(self, ctx):
        return self._verify_result

    def report(self, ctx):
        self.report_calls += 1


def _minimal_ctx(
    tmp_path: Path,
    *,
    webapp_integration: str = "off",
    remote_supervision: str = "off",
    non_interactive: bool = False,
    verbose: bool = False,
):
    """A `Context` + `Config` pair for direct `_dispatch()` unit tests,
    never touching a real install dir or a real invoking user."""
    install_dir = tmp_path / "install"
    install_dir.mkdir(exist_ok=True)
    cfg = config_mod.Config(
        install_dir=install_dir,
        remote_supervision=remote_supervision,
        webapp_integration=webapp_integration,
        values={},
    )
    user = InvokingUser(
        name="tester", home=tmp_path, uid=os.getuid(), gid=os.getgid()
    )
    ctx = app.build_context(cfg, user, non_interactive, verbose)
    return ctx, cfg


# ---------------------------------------------------------------------------
# doc 00 §3.3 -- CLI flag parsing
# ---------------------------------------------------------------------------


def test_parse_args_defaults():
    args = app.parse_args([])
    assert args.install_dir is None
    assert args.resume is False
    assert args.status is False
    assert args.reset_state is False
    assert args.reset_step is None
    assert args.non_interactive is False
    assert args.no_pause is False
    assert args.verbose is False
    assert args.log_dir is None
    # Both gate flags default to None, not to "off": config.load() has to
    # tell "not passed" from "explicitly passed off" (doc 00 §3.4).
    assert args.remote_supervision is None
    assert args.webapp_integration is None


def test_parse_args_install_dir():
    args = app.parse_args(["--install-dir", "/opt/custom"])
    assert args.install_dir == "/opt/custom"


def test_parse_args_resume():
    assert app.parse_args(["--resume"]).resume is True


def test_parse_args_status():
    assert app.parse_args(["--status"]).status is True


def test_parse_args_reset_state():
    assert app.parse_args(["--reset-state"]).reset_state is True


def test_parse_args_reset_step():
    assert app.parse_args(["--reset-step", "3"]).reset_step == 3


def test_parse_args_non_interactive():
    assert app.parse_args(["--non-interactive"]).non_interactive is True


def test_parse_args_no_pause():
    assert app.parse_args(["--no-pause"]).no_pause is True


def test_parse_args_log_dir():
    assert app.parse_args(["--log-dir", "/tmp/mv3dt-logs"]).log_dir == "/tmp/mv3dt-logs"


def test_camera_scan_cidr_accepts_a_valid_range():
    args = app.parse_args(["--camera-scan-cidr", "192.168.1.0/24"])
    assert args.camera_scan_cidr == "192.168.1.0/24"


@pytest.mark.parametrize("bad", ["169.254.0.0/33", "not-a-cidr", "300.1.1.1/8"])
def test_camera_scan_cidr_rejects_an_invalid_range_before_it_is_persisted(bad):
    with pytest.raises(SystemExit):
        app.parse_args(["--camera-scan-cidr", bad])


def test_parse_args_scan_cameras_defaults():
    args = app.parse_args([])
    assert args.scan_cameras is False
    assert args.camera_scan_cidr is None
    assert args.camera_scan_iface is None


def test_parse_args_scan_cameras_flags():
    args = app.parse_args(
        ["--scan-cameras", "--camera-scan-cidr", "10.0.0.0/24", "--camera-scan-iface", "eth1"]
    )
    assert args.scan_cameras is True
    assert args.camera_scan_cidr == "10.0.0.0/24"
    assert args.camera_scan_iface == "eth1"


@pytest.mark.parametrize("value", ["off", "local", "remote"])
def test_parse_args_remote_supervision_accepts_every_gate_value(value):
    args = app.parse_args(["--remote-supervision", value])
    assert args.remote_supervision == value


@pytest.mark.parametrize("value", ["off", "on"])
def test_parse_args_webapp_integration_accepts_every_gate_value(value):
    args = app.parse_args(["--webapp-integration", value])
    assert args.webapp_integration == value


def test_parse_args_gate_flags_reject_a_value_outside_their_choices(capsys):
    """argparse's `choices=` is fed straight from `config.GATE_CHOICES`, so
    a value the dispatch loop could not interpret never reaches
    `installer.conf`. `on` is deliberately not a remote-supervision value:
    that gate is three-valued (off/local/remote)."""
    with pytest.raises(SystemExit) as exc_info:
        app.parse_args(["--remote-supervision", "on"])
    assert exc_info.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_gate_overrides_from_args_omits_flags_that_were_not_passed():
    overrides = app._gate_overrides_from_args(app.parse_args([]))
    assert overrides == {}


def test_gate_overrides_from_args_maps_flags_to_installer_conf_keys():
    args = app.parse_args(
        ["--remote-supervision", "local", "--webapp-integration", "on"]
    )
    assert app._gate_overrides_from_args(args) == {
        config_mod.GATE_REMOTE_SUPERVISION: "local",
        config_mod.GATE_WEBAPP_INTEGRATION: "on",
    }


def test_gate_overrides_from_args_keeps_an_explicit_off():
    """An explicit `off` is an instruction (overwrite whatever is
    persisted), not the absence of one, so it must survive the mapping."""
    args = app.parse_args(["--webapp-integration", "off"])
    assert app._gate_overrides_from_args(args) == {
        config_mod.GATE_WEBAPP_INTEGRATION: "off"
    }


def test_parse_args_version_prints_and_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc_info:
        app.parse_args(["--version"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    # Literal program name, not argparse's %(prog)s: under pytest `prog`
    # would be the test runner's name, and an operator pasting the banner
    # into a bug report must always see which tool it came from. Only the
    # prefix is asserted here because a release build appends its CI build
    # stamp; tests/test_version.py owns the full banner, both ways
    # (doc 00 section 4.1).
    assert captured.out.startswith(f"mv3dt-installer {__version__}")


def test_parse_args_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc_info:
        app.parse_args(["--help"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "mv3dt-installer" in captured.out


# ---------------------------------------------------------------------------
# --status (doc 00 §3.3) -- smoke test case, never requires root
# ---------------------------------------------------------------------------


def test_status_fresh_state_prints_and_exits_zero(tmp_path, capsys):
    state_path = tmp_path / "var" / "state.json"

    rc = app.main(["--status"], state_path=state_path)

    assert rc == 0
    captured = capsys.readouterr()
    assert "install_dir:" in captured.out
    assert "all complete: False" in captured.out
    for step_id in STEP_IDS:
        assert f"{step_id}: PENDING" in captured.out
    # --status performs no writes -- a fresh/missing state.json is never
    # created just by inspecting it.
    assert not state_path.exists()


def test_status_does_not_require_root(tmp_path):
    """Confirms the deliberate design choice: --status runs before
    privilege.require_root(), so it works for a non-root test process
    without any monkeypatching."""
    state_path = tmp_path / "var" / "state.json"
    assert app.main(["--status"], state_path=state_path) == 0


def test_main_requires_root_for_non_status_flow(tmp_path):
    """Every flow other than --status runs after privilege.require_root()
    (doc 00 §9.1). This process is not root, so main() must die() (exit 1)
    rather than silently proceeding."""
    state_path = tmp_path / "state.json"
    with pytest.raises(SystemExit) as exc_info:
        app.main(["--non-interactive"], state_path=state_path)
    assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# --reset-state (doc 00 §3.3)
# ---------------------------------------------------------------------------


def test_reset_state_wipes_and_reinitializes(tmp_path, monkeypatch):
    monkeypatch.setattr(app.privilege, "require_root", lambda: None)
    _bypass_onboarding(monkeypatch, tmp_path)
    monkeypatch.setattr(app, "STEP_REGISTRY", [])

    state_path = tmp_path / "var" / "state.json"
    sm = StateMachine(path=state_path)
    sm.mark_complete("step1_prerequisites")
    assert sm.load().steps["step1_prerequisites"].status == StepStatus.COMPLETE

    install_dir = tmp_path / "install"
    log_dir = tmp_path / "logs"
    rc = app.main(
        [
            "--reset-state",
            "--non-interactive",
            "--install-dir",
            str(install_dir),
            "--log-dir",
            str(log_dir),
        ],
        state_path=state_path,
    )

    assert rc == 0
    after = sm.load()
    assert after.steps["step1_prerequisites"].status == StepStatus.PENDING


def test_reset_state_preserves_previously_chosen_install_dir(tmp_path, monkeypatch):
    """Regression test (code review finding): `--reset-state` must not
    silently discard a previously-chosen custom `--install-dir`.

    `install_dir` is persisted inside `state.json` itself (doc 00 §6.2).
    Before this fix, `_reset_state()` rewrote state.json via
    `default_state()`, which hardcodes `install_dir` back to `/opt/mv3dt`.
    Because state.json then *exists* again, `config.py`'s precedence chain
    (`--install-dir override > state.json > ...`, §11.2) picked up that
    hardcoded value ahead of the operator's real install location on the
    very next run that doesn't re-pass `--install-dir` -- effectively
    relocating a live install out from under the operator. `--reset-state`
    is meant to clear step-completion status, not the install location.
    """
    monkeypatch.setattr(app.privilege, "require_root", lambda: None)
    _bypass_onboarding(monkeypatch, tmp_path)
    monkeypatch.setattr(app, "STEP_REGISTRY", [])

    state_path = tmp_path / "var" / "state.json"
    custom_install_dir = tmp_path / "custom-install-location"
    log_dir = tmp_path / "logs"

    # First run: operator picks a custom install dir; it's persisted.
    rc1 = app.main(
        [
            "--install-dir",
            str(custom_install_dir),
            "--non-interactive",
            "--log-dir",
            str(log_dir),
        ],
        state_path=state_path,
    )
    assert rc1 == 0
    sm = StateMachine(path=state_path)
    assert sm.load().install_dir == str(custom_install_dir)

    # Second run: --reset-state WITHOUT re-passing --install-dir. The
    # previously-chosen custom dir must survive the reset.
    rc2 = app.main(
        [
            "--reset-state",
            "--non-interactive",
            "--log-dir",
            str(log_dir),
        ],
        state_path=state_path,
    )
    assert rc2 == 0
    assert sm.load().install_dir == str(custom_install_dir)


# ---------------------------------------------------------------------------
# --reset-step (doc 00 §3.3)
# ---------------------------------------------------------------------------


def test_reset_step_no_match_prints_message_and_exits_cleanly(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(app.privilege, "require_root", lambda: None)
    _bypass_onboarding(monkeypatch, tmp_path)
    monkeypatch.setattr(app, "STEP_REGISTRY", [])

    state_path = tmp_path / "var" / "state.json"
    rc = app.main(["--reset-step", "3"], state_path=state_path)

    assert rc == 0
    captured = capsys.readouterr()
    assert "no step registered with order 3" in captured.err


def test_reset_step_match_clears_status(tmp_path, monkeypatch):
    monkeypatch.setattr(app.privilege, "require_root", lambda: None)
    _bypass_onboarding(monkeypatch, tmp_path)

    state_path = tmp_path / "var" / "state.json"
    sm = StateMachine(path=state_path)
    sm.mark_complete("step2_deepstream_sdk")

    step = _DummyStep("step2_deepstream_sdk", "DeepStream SDK", 2)
    monkeypatch.setattr(app, "STEP_REGISTRY", [step])

    rc = app.main(["--reset-step", "2"], state_path=state_path)

    assert rc == 0
    assert sm.load().steps["step2_deepstream_sdk"].status == StepStatus.PENDING


# ---------------------------------------------------------------------------
# Ordering fix: state loaded (and StateMachine passed to config.load())
# before config resolution, end-to-end through app.main().
# ---------------------------------------------------------------------------


def test_main_config_precedence_reads_back_from_state_json(tmp_path, monkeypatch):
    monkeypatch.setattr(app.privilege, "require_root", lambda: None)
    _bypass_onboarding(monkeypatch, tmp_path)
    monkeypatch.setattr(app, "STEP_REGISTRY", [])

    state_path = tmp_path / "var" / "state.json"
    log_dir = tmp_path / "logs"
    chosen_install_dir = tmp_path / "chosen-install-dir"

    rc1 = app.main(
        [
            "--install-dir",
            str(chosen_install_dir),
            "--non-interactive",
            "--log-dir",
            str(log_dir),
        ],
        state_path=state_path,
    )
    assert rc1 == 0

    sm = StateMachine(path=state_path)
    assert sm.load().install_dir == str(chosen_install_dir)

    # Second run: no --install-dir override at all. If config.load() had
    # NOT received the real, already-populated StateMachine (the ordering
    # bug the doc's five-line §3.2 summary would produce if taken
    # literally -- config loaded before state), this would fall through to
    # the hardcoded /opt/mv3dt default instead of state.json's recorded
    # value.
    rc2 = app.main(
        ["--non-interactive", "--log-dir", str(log_dir)],
        state_path=state_path,
    )
    assert rc2 == 0
    assert sm.load().install_dir == str(chosen_install_dir)


# ---------------------------------------------------------------------------
# doc 00 §12.3 -- Context construction
# ---------------------------------------------------------------------------


def test_build_context_bindings_work(tmp_path, capsys):
    ctx, cfg = _minimal_ctx(tmp_path)

    assert ctx.install_dir == cfg.install_dir
    assert ctx.conf is cfg.values
    assert ctx.user.name == "tester"

    ctx.report_installed("foo", "1.2.3")
    ctx.report_already_installed("bar", "4.5.6")
    captured = capsys.readouterr()
    assert "installed foo version 1.2.3" in captured.err
    assert "already installed bar version 4.5.6" in captured.err

    assert ctx.verify_pinned("label", "1.0", "1.0") is True
    assert ctx.verify_pinned("label", "1.0", "2.0") is False

    asset = ctx.asset_path("scripts", "example.sh")
    assert asset.parts[-2:] == ("scripts", "example.sh")

    # ngc handle bound to this install_dir; no key ever stored here.
    assert ctx.ngc.load_key() is None

    # webapp handle bound to gate "off" -> never enabled regardless of
    # whatever credentials might exist.
    assert ctx.webapp.enabled() is False

    assert ctx.reboot.request() is StepStatus.REBOOT_REQUIRED

    result = ctx.run_root(sys.executable, "-c", "pass")
    assert result.returncode == 0


def test_run_root_missing_executable_returns_127_instead_of_raising(tmp_path):
    """Regression for the crash seen on a fresh, driver-less workstation:
    `_driver_loaded()` (step1_prerequisites.py) calls `ctx.run_root("nvidia-
    smi", ..., check=False, capture_output=True, text=True)` expecting
    "not installed yet" to look like a nonzero returncode. Before this fix,
    a genuinely-missing executable raised `FileNotFoundError` straight out
    of `subprocess.run` -- before `check` is ever consulted -- crashing the
    installer instead of driving Step 1 into its "install the driver" path.
    """
    ctx, _cfg = _minimal_ctx(tmp_path)

    result = ctx.run_root(
        "this-binary-does-not-exist-on-any-path",
        "--query-gpu=driver_version",
        "--format=csv,noheader",
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 127
    assert result.stdout == ""
    assert result.stderr == ""


def test_run_root_missing_executable_without_capture_leaves_streams_none(tmp_path):
    ctx, _cfg = _minimal_ctx(tmp_path)

    result = ctx.run_root(
        "this-binary-does-not-exist-on-any-path", check=False
    )

    assert result.returncode == 127
    assert result.stdout is None
    assert result.stderr is None


def test_run_root_missing_executable_binary_mode_when_not_text(tmp_path):
    ctx, _cfg = _minimal_ctx(tmp_path)

    result = ctx.run_root(
        "this-binary-does-not-exist-on-any-path", check=False, capture_output=True
    )

    assert result.returncode == 127
    assert result.stdout == b""
    assert result.stderr == b""


def test_context_webapp_enabled_true_with_gate_on_and_stored_credentials(tmp_path):
    from mv3dt_installer import webapp as webapp_mod

    ctx, _cfg = _minimal_ctx(tmp_path, webapp_integration="on")
    webapp_mod.store_credentials(
        webapp_mod.Credentials(api_key="k", endpoint="https://example.test"),
        ctx.install_dir,
    )

    assert ctx.webapp.enabled() is True
    assert ctx.webapp.load_credentials().api_key == "k"


def test_context_run_as_user_delegates_to_privilege(tmp_path, monkeypatch):
    calls = []

    def fake_run_as_user(*args, **kwargs):
        calls.append((args, kwargs))
        return "sentinel"

    monkeypatch.setattr(app.privilege, "run_as_user", fake_run_as_user)
    ctx, _cfg = _minimal_ctx(tmp_path)

    result = ctx.run_as_user(
        "echo", "hi", check=True, capture_output=True, text=True
    )

    assert result == "sentinel"
    assert calls == [
        (
            ("echo", "hi"),
            {"check": True, "capture_output": True, "text": True},
        )
    ]


def test_context_run_as_user_streams_captured_text_through_sudo(tmp_path, monkeypatch):
    ctx, _cfg = _minimal_ctx(tmp_path)
    calls = []
    completed = subprocess.CompletedProcess(["sudo"], 0, "out\n", "err\n")

    def fake_run_streamed(command, **kwargs):
        calls.append((command, kwargs))
        return completed

    monkeypatch.setattr(app.shellout, "run_streamed", fake_run_streamed)

    result = ctx.run_as_user(
        "curl",
        "https://example.invalid",
        capture_output=True,
        text=True,
        stream=True,
    )

    assert result is completed
    assert calls[0][0] == (
        "sudo",
        "-u",
        ctx.user.name,
        "-H",
        "curl",
        "https://example.invalid",
    )
    assert calls[0][1]["renderer"] is ctx.progress
    assert calls[0][1]["redact_capture"] is False
    assert calls[0][1]["capture_output"] is True
    assert calls[0][1]["text"] is True


def test_run_observed_serialises_command_and_adapter_renderer_access():
    entered = threading.Event()

    class Renderer:
        live = False

        def __init__(self):
            self.active = False
            self.calls = []

        def line(self, text):
            assert self.active is False
            self.active = True
            entered.set()
            time.sleep(0.02)
            self.calls.append(("line", text))
            self.active = False

        def bytes(self, done, total):
            assert self.active is False
            self.calls.append(("bytes", done, total))

    renderer = Renderer()
    ctx = SimpleNamespace(progress=app.ProgressHandle(renderer))

    def run():
        ctx.progress.line("child output")
        return subprocess.CompletedProcess(["command"], 0, "", "")

    def observe(is_running):
        assert entered.wait(timeout=1)
        ctx.progress.bytes(1, 2)

    result = app.Context.run_observed(ctx, run, observe)

    assert result.returncode == 0
    assert renderer.calls == [("line", "child output"), ("bytes", 1, 2)]


# ---------------------------------------------------------------------------
# doc 00 §3.2 -- dispatch loop (direct _dispatch() unit tests)
# ---------------------------------------------------------------------------


def test_dispatch_empty_registry_completes_trivially(tmp_path, monkeypatch):
    sm = StateMachine(path=tmp_path / "state.json")
    ctx, cfg = _minimal_ctx(tmp_path)
    monkeypatch.setattr(app, "STEP_REGISTRY", [])

    assert app._dispatch(sm, ctx, cfg) == 0


def test_dispatch_completes_step_and_marks_state(tmp_path, monkeypatch):
    sm = StateMachine(path=tmp_path / "state.json")
    ctx, cfg = _minimal_ctx(tmp_path)
    step = _DummyStep("step1_prerequisites", "Prerequisites", 1)
    monkeypatch.setattr(app, "STEP_REGISTRY", [step])

    rc = app._dispatch(sm, ctx, cfg)

    assert rc == 0
    assert step.report_calls == 1
    assert sm.status("step1_prerequisites") is StepStatus.COMPLETE


def test_dispatch_skips_already_complete_step(tmp_path, monkeypatch):
    sm = StateMachine(path=tmp_path / "state.json")
    sm.mark_complete("step1_prerequisites")
    ctx, cfg = _minimal_ctx(tmp_path)
    step = _DummyStep("step1_prerequisites", "Prerequisites", 1)
    monkeypatch.setattr(app, "STEP_REGISTRY", [step])

    rc = app._dispatch(sm, ctx, cfg)

    assert rc == 0
    assert step.report_calls == 0  # skipped, never re-run


def test_dispatch_halts_on_reboot_required(tmp_path, monkeypatch):
    sm = StateMachine(path=tmp_path / "state.json")
    ctx, cfg = _minimal_ctx(tmp_path)
    step = _DummyStep(
        "step1_prerequisites",
        "Prerequisites",
        1,
        verify=StepResult(status=StepStatus.REBOOT_REQUIRED, message="reboot pls"),
    )
    monkeypatch.setattr(app, "STEP_REGISTRY", [step])
    monkeypatch.setattr(app.reboot_mod, "current_boot_id", lambda: "boot-abc")

    rc = app._dispatch(sm, ctx, cfg)

    assert rc == 0
    pending = sm.load().reboot_pending
    assert pending is not None
    assert pending.requested_by == "step1_prerequisites"
    assert pending.boot_id_at_request == "boot-abc"
    # A step that only got as far as verify() before requesting a reboot
    # is not (yet) marked COMPLETE.
    assert sm.status("step1_prerequisites") is not StepStatus.COMPLETE


def test_dispatch_halts_on_user_action_required(tmp_path, monkeypatch, capsys):
    sm = StateMachine(path=tmp_path / "state.json")
    ctx, cfg = _minimal_ctx(tmp_path)
    action = UserAction(text="do the thing", command="do-it --now")
    step = _DummyStep(
        "step2_deepstream_sdk",
        "DeepStream SDK",
        2,
        verify=StepResult(
            status=StepStatus.USER_ACTION_REQUIRED,
            message="needs manual input",
            user_actions=[action],
        ),
    )
    monkeypatch.setattr(app, "STEP_REGISTRY", [step])

    rc = app._dispatch(sm, ctx, cfg)

    assert rc == 0
    captured = capsys.readouterr()
    assert "ACTION REQUIRED" in captured.err
    assert "do the thing" in captured.err
    assert "do-it --now" in captured.err
    assert "Then run the installer again to continue." in captured.err
    assert sm.status("step2_deepstream_sdk") is not StepStatus.COMPLETE


def test_dispatch_halts_on_failed(tmp_path, monkeypatch, capsys):
    sm = StateMachine(path=tmp_path / "state.json")
    ctx, cfg = _minimal_ctx(tmp_path)
    step = _DummyStep(
        "step1_prerequisites",
        "Prerequisites",
        1,
        run=StepResult(status=StepStatus.FAILED, message="boom"),
    )
    monkeypatch.setattr(app, "STEP_REGISTRY", [step])

    rc = app._dispatch(sm, ctx, cfg)

    assert rc == 1
    captured = capsys.readouterr()
    assert "boom" in captured.err
    assert sm.status("step1_prerequisites") is not StepStatus.COMPLETE


# ---------------------------------------------------------------------------
# doc 00 §3.4 -- opt-in step gates
# ---------------------------------------------------------------------------


def test_dispatch_gate_off_autocompletes_and_never_runs_lifecycle(tmp_path, monkeypatch):
    sm = StateMachine(path=tmp_path / "state.json")
    ctx, cfg = _minimal_ctx(tmp_path)  # remote_supervision defaults to "off"
    step = _DummyStep("step6_remote_supervision", "Remote Supervision", 6)
    monkeypatch.setattr(app, "STEP_REGISTRY", [step])

    rc = app._dispatch(sm, ctx, cfg)

    assert rc == 0
    assert sm.status("step6_remote_supervision") is StepStatus.COMPLETE
    # Gate-off skip never invokes the lifecycle, so report() is never
    # called (mirrors the "already complete" skip discipline exactly).
    assert step.report_calls == 0


def test_dispatch_gate_on_runs_step7_normally(tmp_path, monkeypatch):
    sm = StateMachine(path=tmp_path / "state.json")
    ctx, cfg = _minimal_ctx(tmp_path, webapp_integration="on")
    step = _DummyStep("step7_webapp_integration", "Web-app Integration", 7)
    monkeypatch.setattr(app, "STEP_REGISTRY", [step])

    rc = app._dispatch(sm, ctx, cfg)

    assert rc == 0
    assert step.report_calls == 1
    assert sm.status("step7_webapp_integration") is StepStatus.COMPLETE


@pytest.mark.parametrize("gate_value", ["local", "remote"])
def test_dispatch_gate_three_valued_non_off_runs_step6_normally(
    tmp_path, monkeypatch, gate_value
):
    """Regression test: doc 00 §3.4's table gives Step 6's
    `MV3DT_REMOTE_SUPERVISION` gate three values (`off`/`local`/`remote`),
    unlike Step 7's binary `off`/`on`. An earlier version of `_dispatch`'s
    gate check tested `gate_value != "on"`, which incorrectly auto-skipped
    Step 6 for *both* `local` and `remote` (neither equals the literal
    string "on"). Only `"off"` may skip a gated step; every other value
    (however many a given gate defines) must run it normally."""
    sm = StateMachine(path=tmp_path / "state.json")
    ctx, cfg = _minimal_ctx(tmp_path, remote_supervision=gate_value)
    step = _DummyStep("step6_remote_supervision", "Remote Supervision", 6)
    monkeypatch.setattr(app, "STEP_REGISTRY", [step])

    rc = app._dispatch(sm, ctx, cfg)

    assert rc == 0
    assert step.report_calls == 1
    assert sm.status("step6_remote_supervision") is StepStatus.COMPLETE


def test_dispatch_gate_off_skips_step6_explicitly(tmp_path, monkeypatch):
    """Companion to the three-valued regression test above: "off" must
    still skip Step 6, exactly as it does for Step 7's binary gate."""
    sm = StateMachine(path=tmp_path / "state.json")
    ctx, cfg = _minimal_ctx(tmp_path, remote_supervision="off")
    step = _DummyStep("step6_remote_supervision", "Remote Supervision", 6)
    monkeypatch.setattr(app, "STEP_REGISTRY", [step])

    rc = app._dispatch(sm, ctx, cfg)

    assert rc == 0
    assert step.report_calls == 0
    assert sm.status("step6_remote_supervision") is StepStatus.COMPLETE


# ---------------------------------------------------------------------------
# doc 00 §3.4 -- gate flags wired through main() into installer.conf
# ---------------------------------------------------------------------------


def _read_conf(install_dir: Path) -> dict[str, str]:
    text = (install_dir / config_mod.CONF_FILENAME).read_text(encoding="utf-8")
    return dict(
        line.partition("=")[::2]
        for line in text.splitlines()
        if line and not line.startswith("#")
    )


def test_main_gate_flags_reach_installer_conf(tmp_path, monkeypatch):
    monkeypatch.setattr(app.privilege, "require_root", lambda: None)
    _bypass_onboarding(monkeypatch, tmp_path)
    monkeypatch.setattr(app, "STEP_REGISTRY", [])

    install_dir = tmp_path / "install"
    rc = app.main(
        [
            "--non-interactive",
            "--install-dir",
            str(install_dir),
            "--log-dir",
            str(tmp_path / "logs"),
            "--remote-supervision",
            "local",
            "--webapp-integration",
            "on",
        ],
        state_path=tmp_path / "var" / "state.json",
    )

    assert rc == 0
    values = _read_conf(install_dir)
    assert values[config_mod.GATE_REMOTE_SUPERVISION] == "local"
    assert values[config_mod.GATE_WEBAPP_INTEGRATION] == "on"


def test_main_without_gate_flags_leaves_unset_gates_off(tmp_path, monkeypatch):
    """`--non-interactive` with no flag and no env var: tier 4 (doc 00
    §3.4). The gate questions must never be asked, so `input` is patched to
    something that would fail loudly if they were."""
    monkeypatch.setattr(app.privilege, "require_root", lambda: None)
    _bypass_onboarding(monkeypatch, tmp_path)
    monkeypatch.setattr(app, "STEP_REGISTRY", [])
    monkeypatch.delenv("MV3DT_REMOTE_SUPERVISION", raising=False)
    monkeypatch.delenv("MV3DT_WEBAPP_INTEGRATION", raising=False)
    monkeypatch.setattr(
        "builtins.input",
        lambda *_a, **_kw: pytest.fail("main() prompted under --non-interactive"),
    )

    install_dir = tmp_path / "install"
    rc = app.main(
        [
            "--non-interactive",
            "--install-dir",
            str(install_dir),
            "--log-dir",
            str(tmp_path / "logs"),
        ],
        state_path=tmp_path / "var" / "state.json",
    )

    assert rc == 0
    values = _read_conf(install_dir)
    assert values[config_mod.GATE_REMOTE_SUPERVISION] == "off"
    assert values[config_mod.GATE_WEBAPP_INTEGRATION] == "off"


def test_main_gate_flag_flips_a_persisted_gate_on_a_later_run(
    tmp_path, monkeypatch, capsys
):
    """The end-to-end shape of the regression this unit fixes: a first run
    persists `off`, and a later `--webapp-integration on` turns it on
    without the operator hand-editing `installer.conf`."""
    monkeypatch.setattr(app.privilege, "require_root", lambda: None)
    _bypass_onboarding(monkeypatch, tmp_path)
    monkeypatch.setattr(app, "STEP_REGISTRY", [])

    install_dir = tmp_path / "install"
    state_path = tmp_path / "var" / "state.json"
    base_argv = [
        "--non-interactive",
        "--install-dir",
        str(install_dir),
        "--log-dir",
        str(tmp_path / "logs"),
    ]

    assert app.main(base_argv, state_path=state_path) == 0
    assert _read_conf(install_dir)[config_mod.GATE_WEBAPP_INTEGRATION] == "off"
    capsys.readouterr()

    assert (
        app.main(
            base_argv + ["--webapp-integration", "on"], state_path=state_path
        )
        == 0
    )

    assert _read_conf(install_dir)[config_mod.GATE_WEBAPP_INTEGRATION] == "on"
    assert "off -> on" in capsys.readouterr().err

    # And it stays on for the next flagless run.
    assert app.main(base_argv, state_path=state_path) == 0
    assert _read_conf(install_dir)[config_mod.GATE_WEBAPP_INTEGRATION] == "on"


def test_main_gate_env_var_still_seeds_a_first_run(tmp_path, monkeypatch):
    """Tier 2 survives the loss of `bootstrap.sh`'s `exec sudo -E`: anyone
    scripting `sudo -E ./mv3dt-installer` keeps the behaviour they had."""
    monkeypatch.setattr(app.privilege, "require_root", lambda: None)
    _bypass_onboarding(monkeypatch, tmp_path)
    monkeypatch.setattr(app, "STEP_REGISTRY", [])
    monkeypatch.setenv("MV3DT_WEBAPP_INTEGRATION", "on")

    install_dir = tmp_path / "install"
    rc = app.main(
        [
            "--non-interactive",
            "--install-dir",
            str(install_dir),
            "--log-dir",
            str(tmp_path / "logs"),
        ],
        state_path=tmp_path / "var" / "state.json",
    )

    assert rc == 0
    assert _read_conf(install_dir)[config_mod.GATE_WEBAPP_INTEGRATION] == "on"


# ---------------------------------------------------------------------------
# doc 00 §7 -- reboot reconciliation wired into main()
# ---------------------------------------------------------------------------


def test_main_reboot_still_pending_blocks_dispatch(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(app.privilege, "require_root", lambda: None)
    _bypass_onboarding(monkeypatch, tmp_path)
    monkeypatch.setattr(app, "STEP_REGISTRY", [])

    state_path = tmp_path / "state.json"
    sm = StateMachine(path=state_path)
    sm.set_reboot_pending("step1_prerequisites", boot_id="boot-xyz")
    monkeypatch.setattr(app.reboot_mod, "current_boot_id", lambda: "boot-xyz")

    install_dir = tmp_path / "install"
    log_dir = tmp_path / "logs"
    rc = app.main(
        [
            "--non-interactive",
            "--install-dir",
            str(install_dir),
            "--log-dir",
            str(log_dir),
        ],
        state_path=state_path,
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert "ACTION REQUIRED" in captured.err
    assert "reboot is required" in captured.err
    assert sm.load().reboot_pending is not None


def test_main_reboot_confirmed_advances_dispatch(tmp_path, monkeypatch):
    monkeypatch.setattr(app.privilege, "require_root", lambda: None)
    _bypass_onboarding(monkeypatch, tmp_path)
    monkeypatch.setattr(app, "STEP_REGISTRY", [])

    state_path = tmp_path / "state.json"
    sm = StateMachine(path=state_path)
    sm.set_reboot_pending("step1_prerequisites", boot_id="boot-old")
    monkeypatch.setattr(app.reboot_mod, "current_boot_id", lambda: "boot-new")

    install_dir = tmp_path / "install"
    log_dir = tmp_path / "logs"
    rc = app.main(
        [
            "--non-interactive",
            "--install-dir",
            str(install_dir),
            "--log-dir",
            str(log_dir),
        ],
        state_path=state_path,
    )

    assert rc == 0
    after = sm.load()
    assert after.reboot_pending is None
    assert after.steps["step1_prerequisites"].status is StepStatus.COMPLETE


# ---------------------------------------------------------------------------
# --scan-cameras (doc 00 §3.3, §15.5)
# ---------------------------------------------------------------------------


def test_scan_cameras_calls_refresh_and_never_dispatches(tmp_path, monkeypatch):
    monkeypatch.setattr(app.privilege, "require_root", lambda: None)
    _bypass_onboarding(monkeypatch, tmp_path)

    def _boom_dispatch(sm, ctx, cfg):
        raise AssertionError("--scan-cameras must exit before dispatch")

    monkeypatch.setattr(app, "_dispatch", _boom_dispatch)

    captured = {}

    def _fake_refresh(install_dir, **kwargs):
        captured["install_dir"] = install_dir
        captured["kwargs"] = kwargs
        return cameras_mod.ScanResult(cameras=[], unmatched=[], tool="arp-scan", interfaces=[])

    monkeypatch.setattr(app.cameras_mod, "refresh", _fake_refresh)

    install_dir = tmp_path / "install"
    state_path = tmp_path / "state.json"
    rc = app.main(
        [
            "--scan-cameras",
            "--non-interactive",
            "--install-dir",
            str(install_dir),
            "--log-dir",
            str(tmp_path / "logs"),
        ],
        state_path=state_path,
    )

    assert rc == 0
    assert captured["install_dir"] == install_dir
    assert captured["kwargs"]["non_interactive"] is True
    assert captured["kwargs"]["cam_user"] == "admin"
    assert captured["kwargs"]["cam_password"] == "camera-test-password"


def test_scan_cameras_persists_cidr_and_iface_flags(tmp_path, monkeypatch):
    monkeypatch.setattr(app.privilege, "require_root", lambda: None)
    _bypass_onboarding(monkeypatch, tmp_path)
    monkeypatch.setattr(
        app.cameras_mod,
        "refresh",
        lambda install_dir, **kw: cameras_mod.ScanResult(
            cameras=[], unmatched=[], tool="arp-scan", interfaces=[]
        ),
    )

    install_dir = tmp_path / "install"
    rc = app.main(
        [
            "--scan-cameras",
            "--non-interactive",
            "--install-dir",
            str(install_dir),
            "--log-dir",
            str(tmp_path / "logs"),
            "--camera-scan-cidr",
            "10.0.0.0/24",
            "--camera-scan-iface",
            "eth1",
        ],
        state_path=tmp_path / "state.json",
    )

    assert rc == 0
    conf_text = (install_dir / "installer.conf").read_text(encoding="utf-8")
    assert "CAMERA_SCAN_CIDR=10.0.0.0/24" in conf_text
    assert "CAMERA_SCAN_IFACE=eth1" in conf_text


def test_scan_cameras_persists_cameras_file_pointer(tmp_path, monkeypatch):
    monkeypatch.setattr(app.privilege, "require_root", lambda: None)
    _bypass_onboarding(monkeypatch, tmp_path)
    monkeypatch.setattr(
        app.cameras_mod,
        "refresh",
        lambda install_dir, **kw: cameras_mod.ScanResult(
            cameras=[], unmatched=[], tool="arp-scan", interfaces=[]
        ),
    )

    install_dir = tmp_path / "install"
    rc = app.main(
        [
            "--scan-cameras",
            "--non-interactive",
            "--install-dir",
            str(install_dir),
            "--log-dir",
            str(tmp_path / "logs"),
        ],
        state_path=tmp_path / "state.json",
    )

    assert rc == 0
    conf_text = (install_dir / "installer.conf").read_text(encoding="utf-8")
    assert f"CAMERAS_FILE={install_dir / 'cameras.yml'}" in conf_text


# ---------------------------------------------------------------------------
# STEP-3 §6.2 -- generic subcommand dispatch extension
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _empty_subcommand_registry():
    """Snapshot/restore `SUBCOMMAND_REGISTRY` so registry tests are
    order-independent and never leak a fake handler into another test
    module, exactly like `test_steps_protocol.py`'s `_empty_registry`
    fixture does for `STEP_REGISTRY`."""
    saved = dict(app.SUBCOMMAND_REGISTRY)
    app.SUBCOMMAND_REGISTRY.clear()
    yield
    app.SUBCOMMAND_REGISTRY.clear()
    app.SUBCOMMAND_REGISTRY.update(saved)


def test_register_subcommand_adds_to_registry():
    def _handler(argv, ctx):
        return 0

    app.register_subcommand("amc", _handler)
    assert app.SUBCOMMAND_REGISTRY["amc"].handler is _handler


def test_register_subcommand_overwrites_an_existing_name():
    app.register_subcommand("amc", lambda argv, ctx: 1)
    app.register_subcommand("amc", lambda argv, ctx: 2)
    assert app.SUBCOMMAND_REGISTRY["amc"].handler(None, None) == 2


def test_register_subcommand_defaults_to_requires_root_true():
    """unit U6: every registration made without an explicit `requires_root=`
    keeps requiring root exactly as before this parameter existed -- `amc`
    (Step 3), `ingest` (Step 4), `reporter`/`uploader` (Step 7), and
    `pipeline`'s pre-STEP-6 modes all rely on this default."""
    app.register_subcommand("amc", lambda argv, ctx: 0)
    assert app.SUBCOMMAND_REGISTRY["amc"].requires_root is True


def test_register_subcommand_accepts_requires_root_false():
    app.register_subcommand("agent", lambda argv, ctx: 0, requires_root=False)
    assert app.SUBCOMMAND_REGISTRY["agent"].requires_root is False


def test_register_subcommand_accepts_requires_root_predicate():
    predicate = lambda argv: "--service-exec" not in argv  # noqa: E731
    app.register_subcommand("pipeline", lambda argv, ctx: 0, requires_root=predicate)
    assert app.SUBCOMMAND_REGISTRY["pipeline"].requires_root is predicate


def test_main_dispatches_registered_subcommand_before_parse_args(
    tmp_path, monkeypatch
):
    """A registered subcommand name as the first token must bypass the
    framework's own `--status`/dispatch-loop flow entirely -- `_dispatch`
    (and therefore the step loop) must never run."""
    monkeypatch.setattr(app.privilege, "require_root", lambda: None)
    _bypass_onboarding(monkeypatch, tmp_path)

    def _boom_dispatch(sm, ctx, cfg):
        raise AssertionError("a subcommand must never re-enter _dispatch()")

    monkeypatch.setattr(app, "_dispatch", _boom_dispatch)

    captured = {}

    def _handler(argv, ctx):
        captured["argv"] = argv
        captured["ctx"] = ctx
        return 7

    app.register_subcommand("amc", _handler)

    install_dir = tmp_path / "install"
    rc = app.main(
        [
            "amc",
            "--project",
            "north-lobby",
            "--skip-pull",
            "--install-dir",
            str(install_dir),
            "--non-interactive",
            "--log-dir",
            str(tmp_path / "logs"),
        ],
        state_path=tmp_path / "state.json",
    )

    assert rc == 7
    assert captured["argv"] == [
        "--project",
        "north-lobby",
        "--skip-pull",
        "--install-dir",
        str(install_dir),
        "--non-interactive",
        "--log-dir",
        str(tmp_path / "logs"),
    ]
    assert isinstance(captured["ctx"], app.Context)
    assert captured["ctx"].install_dir == install_dir


def test_main_subcommand_bootstrap_resolves_install_dir_and_non_interactive(
    tmp_path, monkeypatch
):
    """The framework-owned `--install-dir`/`--non-interactive` flags a
    subcommand's own argv may carry (mirroring a systemd `ExecStart=` line
    for a later step's unit) must both reach the built `Context`, and the
    full, unmodified argv must still reach the handler."""
    monkeypatch.setattr(app.privilege, "require_root", lambda: None)
    _bypass_onboarding(monkeypatch, tmp_path)

    captured = {}

    def _handler(argv, ctx):
        captured["argv"] = argv
        captured["ctx"] = ctx
        return 0

    app.register_subcommand("ingest", _handler)

    install_dir = tmp_path / "custom-install"
    rc = app.main(
        [
            "ingest",
            "--project",
            "north-lobby",
            "--non-interactive",
            "--install-dir",
            str(install_dir),
            "--log-dir",
            str(tmp_path / "logs"),
        ],
        state_path=tmp_path / "state.json",
    )

    assert rc == 0
    ctx = captured["ctx"]
    assert ctx.install_dir == install_dir
    assert ctx.non_interactive is True
    # The handler still sees the framework-owned flags too -- argv is
    # forwarded unmodified, not stripped.
    assert captured["argv"] == [
        "--project",
        "north-lobby",
        "--non-interactive",
        "--install-dir",
        str(install_dir),
        "--log-dir",
        str(tmp_path / "logs"),
    ]


def test_main_subcommand_requires_root(tmp_path, monkeypatch):
    """A subcommand still needs root -- it touches docker, package state,
    and root-owned install-dir paths (STEP-3 §3-§4), same as the install
    flow."""
    called = {"count": 0}

    def _require_root():
        called["count"] += 1
        raise SystemExit(2)

    monkeypatch.setattr(app.privilege, "require_root", _require_root)
    app.register_subcommand("amc", lambda argv, ctx: 0)

    with pytest.raises(SystemExit):
        app.main(["amc"], state_path=tmp_path / "state.json")

    assert called["count"] == 1


def _boom_require_root():
    raise AssertionError(
        "a requires_root=False subcommand must never call privilege.require_root()"
    )


def test_main_subcommand_requires_root_false_skips_root_check(tmp_path, monkeypatch):
    """unit U6's fix: `agent` (registered with `requires_root=False`,
    mirroring STEP-6's real registration) must reach its handler without
    ever calling `privilege.require_root()` -- the exact bug PR #50's
    review surfaced (the agent runs as `User=@USER@`, non-root)."""
    monkeypatch.setattr(app.privilege, "require_root", _boom_require_root)
    _bypass_onboarding(monkeypatch, tmp_path)

    captured = {}

    def _handler(argv, ctx):
        captured["ctx"] = ctx
        return 0

    app.register_subcommand("agent", _handler, requires_root=False)

    install_dir = tmp_path / "install"
    rc = app.main(
        ["agent", "--install-dir", str(install_dir)],
        state_path=tmp_path / "state.json",
    )

    assert rc == 0
    assert isinstance(captured["ctx"], app.Context)
    assert captured["ctx"].install_dir == install_dir


def test_main_subcommand_requires_root_false_does_not_write_config_or_state(
    tmp_path, monkeypatch
):
    """The read-only bootstrap path a `requires_root=False` subcommand takes
    must never create `install_dir`, write `installer.conf`, or touch
    `state.json` -- `config.load()`'s side effects all assume the root
    ownership a real install run already established, which a non-root
    subcommand process cannot safely (or need to) reproduce."""
    monkeypatch.setattr(app.privilege, "require_root", _boom_require_root)
    _bypass_onboarding(monkeypatch, tmp_path)

    def _boom_load(*args, **kwargs):
        raise AssertionError("must not call config.load() for requires_root=False")

    monkeypatch.setattr(app.config_mod, "load", _boom_load)

    def _boom_open_transcript(*args, **kwargs):
        raise AssertionError("must not open a transcript for requires_root=False")

    monkeypatch.setattr(app, "open_transcript", _boom_open_transcript)

    app.register_subcommand("agent", lambda argv, ctx: 0, requires_root=False)

    install_dir = tmp_path / "install"  # deliberately does not exist yet
    state_path = tmp_path / "state.json"
    rc = app.main(["agent", "--install-dir", str(install_dir)], state_path=state_path)

    assert rc == 0
    assert not install_dir.exists()
    assert not state_path.exists()


def test_main_subcommand_requires_root_false_reads_existing_installer_conf(
    tmp_path, monkeypatch
):
    """A non-root subcommand still needs whatever install already persisted
    (e.g. the STEP-6 gate value) -- the read-only bootstrap must read
    `installer.conf` back into `ctx.conf`, it just must never write it."""
    monkeypatch.setattr(app.privilege, "require_root", _boom_require_root)
    _bypass_onboarding(monkeypatch, tmp_path)

    install_dir = tmp_path / "install"
    install_dir.mkdir()
    conf_path = install_dir / config_mod.CONF_FILENAME
    conf_path.write_text("MV3DT_REMOTE_SUPERVISION=local\n")

    captured = {}

    def _handler(argv, ctx):
        captured["ctx"] = ctx
        return 0

    app.register_subcommand("agent", _handler, requires_root=False)

    rc = app.main(
        ["agent", "--install-dir", str(install_dir)],
        state_path=tmp_path / "state.json",
    )

    assert rc == 0
    ctx = captured["ctx"]
    assert ctx.conf.get("MV3DT_REMOTE_SUPERVISION") == "local"
    # Read back correctly, but never rewritten.
    assert conf_path.read_text() == "MV3DT_REMOTE_SUPERVISION=local\n"


def test_main_subcommand_requires_root_predicate_true_for_default_mode(
    tmp_path, monkeypatch
):
    """unit U6's `pipeline`-shaped fix: a `requires_root` predicate lets a
    single subcommand registration require root for some modes and not
    others. The default (non-`--service-exec`) mode must still hit
    `privilege.require_root()`, exactly like before Step 6 existed."""
    called = {"count": 0}

    def _require_root():
        called["count"] += 1
        raise SystemExit(2)

    monkeypatch.setattr(app.privilege, "require_root", _require_root)

    predicate = lambda argv: "--service-exec" not in argv  # noqa: E731
    app.register_subcommand("pipeline", lambda argv, ctx: 0, requires_root=predicate)

    with pytest.raises(SystemExit):
        app.main(
            ["pipeline", "--project", "north-lobby"],
            state_path=tmp_path / "state.json",
        )

    assert called["count"] == 1


def test_main_subcommand_requires_root_predicate_true_for_stop_mode(
    tmp_path, monkeypatch
):
    called = {"count": 0}

    def _require_root():
        called["count"] += 1
        raise SystemExit(2)

    monkeypatch.setattr(app.privilege, "require_root", _require_root)

    predicate = lambda argv: "--service-exec" not in argv  # noqa: E731
    app.register_subcommand("pipeline", lambda argv, ctx: 0, requires_root=predicate)

    with pytest.raises(SystemExit):
        app.main(
            ["pipeline", "--project-slug", "north-lobby", "--stop"],
            state_path=tmp_path / "state.json",
        )

    assert called["count"] == 1


def test_main_subcommand_requires_root_predicate_false_for_service_exec_mode(
    tmp_path, monkeypatch
):
    """`--service-exec` -- the mode `mv3dt-pipeline@.service.in`'s own
    non-root `ExecStart=` invokes -- must NOT hit `privilege.require_root()`."""
    monkeypatch.setattr(app.privilege, "require_root", _boom_require_root)
    _bypass_onboarding(monkeypatch, tmp_path)

    captured = {}

    def _handler(argv, ctx):
        captured["ctx"] = ctx
        return 0

    predicate = lambda argv: "--service-exec" not in argv  # noqa: E731
    app.register_subcommand("pipeline", _handler, requires_root=predicate)

    install_dir = tmp_path / "install"
    rc = app.main(
        [
            "pipeline",
            "--project-slug",
            "north-lobby",
            "--service-exec",
            "--install-dir",
            str(install_dir),
        ],
        state_path=tmp_path / "state.json",
    )

    assert rc == 0
    assert isinstance(captured["ctx"], app.Context)


def test_main_unregistered_first_token_falls_through_to_normal_parsing(tmp_path):
    """A first token that is not a registered subcommand name must fall
    through to the ordinary `parse_args()`/dispatch flow -- and since
    nothing here is a real flag, argparse rejects it exactly as it always
    has, proving the peek never swallows a plain invocation error."""
    with pytest.raises(SystemExit):
        app.main(["not-a-subcommand"], state_path=tmp_path / "state.json")


def test_main_subcommand_dispatch_does_not_run_onboarding_or_reboot_reconcile(
    tmp_path, monkeypatch
):
    """Doc 00 §3.2's onboarding/reboot-reconcile steps are install-flow
    specific (this unit's own scope note); a standalone subcommand must
    skip both."""
    monkeypatch.setattr(app.privilege, "require_root", lambda: None)
    _bypass_onboarding(monkeypatch, tmp_path)

    def _boom_onboard(*args, **kwargs):
        raise AssertionError("a subcommand must never run onboarding.onboard()")

    def _boom_reconcile(*args, **kwargs):
        raise AssertionError("a subcommand must never run reboot.reconcile()")

    monkeypatch.setattr(app.onboarding, "onboard", _boom_onboard)
    monkeypatch.setattr(app.reboot_mod, "reconcile", _boom_reconcile)

    app.register_subcommand("amc", lambda argv, ctx: 0)
    rc = app.main(
        [
            "amc",
            "--install-dir",
            str(tmp_path / "install"),
            "--non-interactive",
            "--log-dir",
            str(tmp_path / "logs"),
        ],
        state_path=tmp_path / "state.json",
    )
    assert rc == 0


# ---------------------------------------------------------------------------
# doc 08 §9 -- the `ctx.progress` handle
# ---------------------------------------------------------------------------


class _RecordingRenderer:
    """Stands in for `progress.Progress` so these tests assert on the calls
    the handle makes, not on how the renderer draws them (that is
    test_progress.py's job)."""

    def __init__(self) -> None:
        self.steps: list = []
        self.phases: list = []
        self.tasks: list = []
        self.byte_calls: list = []
        self.lines: list = []
        self.ended = 0

    def begin_step(self, index, title, phases=()):
        self.steps.append((index, title, tuple(phases)))

    def end_step(self):
        self.ended += 1

    def phase(self, number):
        self.phases.append(number)

    def task(self, name):
        self.tasks.append(name)

    def bytes(self, done, total):
        self.byte_calls.append((done, total))

    def line(self, text):
        self.lines.append(text)


class _PhasedStep(_DummyStep):
    phases = ("first", "second")


def _recording_ctx(tmp_path: Path, **kwargs):
    """A real `Context` whose renderer records instead of drawing."""
    ctx, _cfg = _minimal_ctx(tmp_path, **kwargs)
    recorder = _RecordingRenderer()
    ctx.progress.renderer = recorder
    return ctx, recorder


def test_build_context_exposes_a_progress_handle(tmp_path):
    ctx, _cfg = _minimal_ctx(tmp_path)

    assert isinstance(ctx.progress, app.ProgressHandle)
    assert isinstance(ctx.progress.renderer, app.progress_mod.Progress)
    # Nothing is bound until the dispatch loop binds it.
    assert ctx.progress.step is None


def test_progress_step_banner_denominator_is_the_step_count(tmp_path, capsys):
    """Doc 08 §3: the "of 7" an operator has never been told."""
    ctx, _cfg = _minimal_ctx(tmp_path)
    step = _DummyStep("step1_prerequisites", "Prerequisites", 1)

    ctx.progress.begin_step(step, 1)

    assert f"[ 1/{len(STEP_IDS)} ] Prerequisites" in capsys.readouterr().err


def test_progress_phase_indexes_the_steps_own_declaration(tmp_path):
    ctx, recorder = _recording_ctx(tmp_path)
    step = _PhasedStep("step1_prerequisites", "Prerequisites", 1)

    ctx.progress.begin_step(step, 1)
    ctx.progress.phase(2)

    assert recorder.steps == [(1, "Prerequisites", ("first", "second"))]
    assert recorder.phases == [2]


def test_progress_phase_out_of_range_raises(tmp_path):
    """Doc 08 §9: a wrong denominator on screen is worse than a traceback,
    because the operator cannot tell it is wrong."""
    ctx, recorder = _recording_ctx(tmp_path)
    step = _PhasedStep("step1_prerequisites", "Prerequisites", 1)
    ctx.progress.begin_step(step, 1)

    with pytest.raises(IndexError):
        ctx.progress.phase(3)
    with pytest.raises(IndexError):
        ctx.progress.phase(0)

    assert recorder.phases == []


def test_progress_phase_out_of_range_raises_before_any_step_is_bound(tmp_path):
    ctx, recorder = _recording_ctx(tmp_path)

    # An unbound handle is one unnamed phase, so 1 is legal and 2 is not.
    ctx.progress.phase(1)
    with pytest.raises(IndexError):
        ctx.progress.phase(2)

    assert recorder.phases == []


def test_progress_phase_is_a_no_op_for_a_step_declaring_none(tmp_path):
    """Doc 08 §3.1's backward compatibility: a step that has not adopted
    phases still runs, in one unnamed phase."""
    ctx, recorder = _recording_ctx(tmp_path)
    step = _DummyStep("step4_calib_output_wiring", "Calibration wiring", 4)

    ctx.progress.begin_step(step, 4)
    ctx.progress.phase(1)

    assert recorder.steps == [(4, "Calibration wiring", ())]
    assert recorder.phases == []
    with pytest.raises(IndexError):
        ctx.progress.phase(2)


def test_progress_task_and_bytes_delegate(tmp_path):
    ctx, recorder = _recording_ctx(tmp_path)

    ctx.progress.task("downloading driver")
    ctx.progress.bytes(412, 606)
    ctx.progress.bytes(0, None)

    assert recorder.tasks == ["downloading driver"]
    assert recorder.byte_calls == [(412, 606), (0, None)]


def test_progress_end_step_unbinds_the_step(tmp_path):
    ctx, recorder = _recording_ctx(tmp_path)
    ctx.progress.begin_step(_PhasedStep("step1", "Prereqs", 1), 1)

    ctx.progress.end_step()

    assert recorder.ended == 1
    assert ctx.progress.step is None


def test_step_making_no_progress_calls_dispatches_unchanged(tmp_path, monkeypatch):
    """Doc 08 §9's load-bearing rule: omitting all three calls is legal and
    yields today's behaviour, which is what lets the seven steps adopt this
    one at a time."""
    ctx, cfg = _minimal_ctx(tmp_path)
    recorder = _RecordingRenderer()
    ctx.progress.renderer = recorder
    sm = StateMachine(tmp_path / "state.json")
    step = _DummyStep("step1_prerequisites", "Prerequisites", 1)
    monkeypatch.setattr(app, "STEP_REGISTRY", [step])

    rc = app._dispatch(sm, ctx, cfg)

    assert rc == 0
    assert step.report_calls == 1
    # The banner is the framework's and appears without the step asking for
    # it (doc 08 §3.2); the three step-author calls stay untouched.
    assert recorder.steps == [(1, "Prerequisites", ())]
    assert (recorder.phases, recorder.tasks, recorder.byte_calls, recorder.lines) == (
        [],
        [],
        [],
        [],
    )


def test_unnamed_phase_transition_reaches_the_transcript_not_the_screen(
    tmp_path, capsys
):
    """Doc 08 §3.3 and §7.1 together.

    A step that has declared no phases has exactly one, unnamed, and there
    is no label to draw beside it -- so nothing is rendered. §3.3 still
    wants every transition in the record and §7.1 forbids paying for a
    rendering decision with a hole in it, so the line goes to the
    transcript-only sink rather than being dropped.
    """
    from mv3dt_installer import logs

    logs.open_transcript(tmp_path / "logs")
    ctx, recorder = _recording_ctx(tmp_path)
    step = _DummyStep("step4_calib_output_wiring", "Calibration wiring", 4)
    ctx.progress.begin_step(step, 4)
    capsys.readouterr()  # discard the banner

    ctx.progress.phase(1)

    assert recorder.phases == []
    assert capsys.readouterr().err == ""
    written = logs._transcript_path.read_text(encoding="utf-8")
    assert "step4_calib_output_wiring phase 1/1: (unnamed)" in written


# ---------------------------------------------------------------------------
# doc 08 §3.2 -- the step banner in the dispatch loop
# ---------------------------------------------------------------------------


def _dispatch_ctx(tmp_path: Path, steps, monkeypatch, **kwargs):
    """A recording `Context`, a `StateMachine` and a registry of `steps`."""
    ctx, cfg = _minimal_ctx(tmp_path, **kwargs)
    recorder = _RecordingRenderer()
    ctx.progress.renderer = recorder
    monkeypatch.setattr(app, "STEP_REGISTRY", list(steps))
    return ctx, cfg, StateMachine(path=tmp_path / "state.json"), recorder


def test_dispatch_announces_every_step_it_runs_with_its_position(
    tmp_path, monkeypatch
):
    """Doc 08 §3.2's `[ 3/7 ] <title>`, and event 4 in §2 it closes.

    Position is the one thing a step cannot know about itself, which is why
    the banner is the dispatch loop's and not a step's.
    """
    steps = [
        _PhasedStep("step1_prerequisites", "Prerequisites", 1),
        _DummyStep("step2_deepstream_sdk", "DeepStream SDK", 2),
        _DummyStep("step7_webapp_integration", "Web-app Integration", 7),
    ]
    ctx, cfg, sm, recorder = _dispatch_ctx(
        tmp_path, steps, monkeypatch, webapp_integration="on"
    )

    assert app._dispatch(sm, ctx, cfg) == 0
    # The declared phases travel with the banner, so the renderer can pair
    # a later `phase(n)` with a label (doc 08 §3.1).
    assert recorder.steps == [
        (1, "Prerequisites", ("first", "second")),
        (2, "DeepStream SDK", ()),
        (3, "Web-app Integration", ()),
    ]
    assert recorder.ended == 3


def test_dispatch_never_announces_a_step_it_skips(tmp_path, monkeypatch):
    """A gate-off skip and an already-complete skip never begin a step, so
    neither opens a region -- but both still occupy their slot, so the one
    step that does run is announced with its true, absolute position. Were
    the counter advanced only for steps that actually ran, a resumed
    install would renumber the remainder and contradict `--status`."""
    steps = [
        _DummyStep("step1_prerequisites", "Prerequisites", 1),
        _DummyStep("step6_remote_supervision", "Remote Supervision", 6),
        _DummyStep("step7_webapp_integration", "Web-app Integration", 7),
    ]
    ctx, cfg, sm, recorder = _dispatch_ctx(
        tmp_path,
        steps,
        monkeypatch,
        remote_supervision="off",
        webapp_integration="on",
    )
    sm.mark_complete("step1_prerequisites")

    assert app._dispatch(sm, ctx, cfg) == 0
    assert recorder.steps == [(3, "Web-app Integration", ())]
    assert recorder.ended == 1


def test_dispatch_takes_the_region_down_before_report_writes(tmp_path, monkeypatch):
    """On a tty the live region is erased by cursor arithmetic over the rows
    it drew, so a `report()` line emitted while it was still up would be torn
    through by the erase."""
    events: list[str] = []

    class _OrderedRenderer(_RecordingRenderer):
        def end_step(self):
            events.append("end_step")
            super().end_step()

    class _ReportingStep(_DummyStep):
        def report(self, ctx):
            events.append("report")
            super().report(ctx)

    step = _ReportingStep("step1_prerequisites", "Prerequisites", 1)
    ctx, cfg = _minimal_ctx(tmp_path)
    ctx.progress.renderer = _OrderedRenderer()
    monkeypatch.setattr(app, "STEP_REGISTRY", [step])
    sm = StateMachine(path=tmp_path / "state.json")

    assert app._dispatch(sm, ctx, cfg) == 0
    assert events == ["end_step", "report"]


@pytest.mark.parametrize(
    "outcome, printer",
    [
        (
            StepResult(status=StepStatus.REBOOT_REQUIRED, message="reboot pls"),
            "show_reboot_required",
        ),
        (
            StepResult(status=StepStatus.USER_ACTION_REQUIRED, message="do a thing"),
            "show_user_action_block",
        ),
        (StepResult(status=StepStatus.FAILED, message="boom"), None),
    ],
    ids=["reboot", "user-action", "failed"],
)
def test_dispatch_ends_the_step_before_every_halting_block(
    tmp_path, monkeypatch, outcome, printer
):
    """Each halting branch prints its own block, so every one of them needs
    the region down first, not just the `COMPLETE` path.

    Asserted as an ordering and not as a call count. A count is satisfied
    just as happily when `end_step` runs *after* the block, which is the one
    thing this test exists to rule out: on a tty that ordering tears the
    block through with the erase. The technique is the one
    `test_dispatch_takes_the_region_down_before_report_writes` uses, applied
    to the three halting printers instead of to `report()`.
    """
    events: list[str] = []

    class _OrderedRenderer(_RecordingRenderer):
        def end_step(self):
            events.append("end_step")
            super().end_step()

    step = _DummyStep("step1_prerequisites", "Prerequisites", 1, verify=outcome)
    ctx, cfg = _minimal_ctx(tmp_path)
    recorder = _OrderedRenderer()
    ctx.progress.renderer = recorder
    monkeypatch.setattr(app, "STEP_REGISTRY", [step])
    monkeypatch.setattr(app.reboot_mod, "current_boot_id", lambda: "boot-abc")
    sm = StateMachine(path=tmp_path / "state.json")

    if printer is None:
        # The FAILED branch has no block printer of its own; its one
        # `log.error` line is the block.
        monkeypatch.setattr(
            app.log, "error", lambda *a, **k: events.append("block")
        )
    else:
        monkeypatch.setattr(
            app.privilege, printer, lambda *a, **k: events.append("block")
        )

    app._dispatch(sm, ctx, cfg)

    assert events == ["end_step", "block"]
    assert recorder.steps == [(1, "Prerequisites", ())]
    assert recorder.ended == 1


def test_dispatch_ends_the_step_when_a_step_raises(tmp_path, monkeypatch):
    """A step that raises must leave the terminal in the same clean state as
    one that merely fails -- otherwise the traceback lands inside a region
    that is still counting rows."""

    class _ExplodingStep(_DummyStep):
        def run(self, ctx):
            raise RuntimeError("boom")

    step = _ExplodingStep("step1_prerequisites", "Prerequisites", 1)
    ctx, cfg, sm, recorder = _dispatch_ctx(tmp_path, [step], monkeypatch)

    with pytest.raises(RuntimeError):
        app._dispatch(sm, ctx, cfg)

    assert recorder.steps == [(1, "Prerequisites", ())]
    assert recorder.ended == 1


class _TtyStream(io.StringIO):
    """A stream that claims to be a terminal, so a test can put the renderer
    in live mode deliberately instead of depending on the harness."""

    def isatty(self):
        return True


def test_dispatch_banner_degrades_to_a_plain_line_when_non_interactive(
    tmp_path, monkeypatch
):
    """Doc 08 §7: a pipe, a CI run and `--non-interactive` all get the phase
    line plain, and not one escape sequence.

    The stream reports a tty, so `live` is false because of the flag and not
    because pytest replaced stderr. Without that, the assertion passes on a
    renderer that is drawing at full tilt: `_colour_enabled()` is false under
    `capsys` whatever the renderer does, and a first draw emits no escape
    anyway, so both halves of the old assertion held with live rendering on.
    """
    stream = _TtyStream()
    monkeypatch.setattr(sys, "stderr", stream)
    # `_minimal_ctx` calls `build_context` directly, so it never reaches
    # `main()`'s `set_colour(not args.non_interactive)`. Settling it here is
    # what that line does in production; without it this test would be
    # asserting against `logs`' conservative pre-parse argv default, which
    # under pytest sees no flag at all.
    logs_mod.set_colour(False)

    # The control: the same stream with the flag off really does put the
    # renderer in live mode, so `live is False` below is the flag's doing.
    live_ctx, _ = _minimal_ctx(tmp_path, non_interactive=False)
    assert live_ctx.progress.renderer.live is True

    ctx, cfg = _minimal_ctx(tmp_path, non_interactive=True)
    assert ctx.progress.renderer.live is False
    step = _DummyStep("step1_prerequisites", "Prerequisites", 1)
    monkeypatch.setattr(app, "STEP_REGISTRY", [step])
    sm = StateMachine(path=tmp_path / "state.json")

    stream.truncate(0)
    stream.seek(0)
    assert app._dispatch(sm, ctx, cfg) == 0

    written = stream.getvalue()
    banner = f"[ 1/{len(STEP_IDS)} ] Prerequisites"
    assert banner in written
    # Exactly once: doc 08 §4.2's one-writer rule, and the shape defect 1 in
    # §12.2 describes on the live path.
    assert written.count(banner) == 1

    # No cursor arithmetic: nothing moved the cursor up and nothing cleared
    # below it, which is what a live region is made of and what §7 forbids
    # here.
    assert "\033[J" not in written
    assert not re.search(r"\033\[\d+A", written)

    # The full §7 rule now holds, not just the renderer's half of it: U5b
    # taught `logs` about the flag and U13 made the *parsed* value settle it,
    # so nothing colours a level label either.
    assert "\033" not in written


# ---------------------------------------------------------------------------
# doc 08 §4.1 -- streaming run_root
# ---------------------------------------------------------------------------


_ECHO = "import sys; print('out-line'); print('err-line', file=sys.stderr)"


def _run_echo(ctx, **kwargs):
    return ctx.run_root(
        sys.executable, "-c", _ECHO, capture_output=True, text=True, **kwargs
    )


def test_should_stream_auto_needs_captured_text_and_nothing_else():
    """Doc 00 §8.5's AUTO rule, stated as a table.

    AUTO asks one question: can the tee serve this call? A terminal is not
    part of the answer -- §7 requires per-line output off a tty too, and
    which shape those lines take is the renderer's decision.
    """
    captured_text = {"capture_output": True, "text": True}

    assert app._should_stream(None, captured_text) is True
    assert app._should_stream(None, {"capture_output": True}) is False
    assert app._should_stream(None, {"text": True}) is False
    older_text_spelling = {"capture_output": True, "universal_newlines": True}
    assert app._should_stream(None, older_text_spelling) is True
    assert app._should_stream(None, {}) is False

    # The escape hatch is absolute, and an explicit request is honoured.
    assert app._should_stream(False, captured_text) is False
    assert app._should_stream(True, captured_text) is True


def test_should_stream_rejects_forcing_a_shape_the_tee_cannot_return():
    with pytest.raises(ValueError):
        app._should_stream(True, {"capture_output": True})
    with pytest.raises(ValueError):
        app._should_stream(True, {})


def test_run_root_streams_to_the_progress_renderer(tmp_path):
    ctx, recorder = _recording_ctx(tmp_path)

    result = _run_echo(ctx)

    assert result.returncode == 0
    assert result.stdout == "out-line\n"
    assert result.stderr == "err-line\n"
    assert "out-line" in recorder.lines
    assert "err-line" in recorder.lines


def test_run_root_stream_false_suppresses_live_output(tmp_path):
    """The escape hatch in doc 08 §4.1: a probe whose output is noise."""
    ctx, recorder = _recording_ctx(tmp_path)

    result = _run_echo(ctx, stream=False)

    assert result.stdout == "out-line\n"
    assert result.stderr == "err-line\n"
    assert recorder.lines == []


def test_run_root_auto_still_streams_off_a_tty(tmp_path):
    """The resolution doc 08 §4.1 left open, now doc 00 §8.5.

    Nothing under pytest is a tty, so this is the `mv3dt-installer | tee
    install.log` case: §7 requires the operator to see per-line output
    there, and before this the runner sent them nothing at all.
    """
    ctx, recorder = _recording_ctx(tmp_path)

    result = _run_echo(ctx)

    assert result.stdout == "out-line\n"
    assert "out-line" in recorder.lines
    assert "err-line" in recorder.lines


def test_run_root_returns_the_same_completedprocess_either_way(tmp_path):
    """The compatibility contract behind doc 08 §4.1: 72 `capture_output`
    call sites and every test asserting on `.stdout` keep working because
    the two paths return the same thing."""
    ctx, _recorder = _recording_ctx(tmp_path)

    plain = _run_echo(ctx, stream=False)
    streamed = _run_echo(ctx)

    assert isinstance(streamed, type(plain))
    assert (streamed.returncode, streamed.stdout, streamed.stderr) == (
        plain.returncode,
        plain.stdout,
        plain.stderr,
    )


def test_run_root_streaming_reports_a_nonzero_exit_the_same_way(tmp_path):
    ctx, _recorder = _recording_ctx(tmp_path)

    result = ctx.run_root(
        sys.executable,
        "-c",
        "import sys; sys.exit(3)",
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 3
    assert result.stdout == ""


def test_run_root_streaming_still_honours_check(tmp_path):
    import subprocess

    ctx, _recorder = _recording_ctx(tmp_path)

    with pytest.raises(subprocess.CalledProcessError):
        ctx.run_root(
            sys.executable,
            "-c",
            "import sys; sys.exit(3)",
            check=True,
            capture_output=True,
            text=True,
        )


def test_run_root_streaming_passes_through_popen_kwargs(tmp_path):
    ctx, _recorder = _recording_ctx(tmp_path)

    result = ctx.run_root(
        sys.executable,
        "-c",
        "import os; print(os.getcwd())",
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == os.path.realpath(str(tmp_path))


def test_run_root_missing_executable_still_127_when_streaming(tmp_path):
    """The `_missing_executable_result` path survives the new route: the
    tee runner raises `FileNotFoundError` out of `Popen` exactly as
    `subprocess.run` did."""
    ctx, _recorder = _recording_ctx(tmp_path)

    result = ctx.run_root(
        "this-binary-does-not-exist-on-any-path",
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 127
    assert result.stdout == ""
    assert result.stderr == ""


# ---------------------------------------------------------------------------
# doc 00 §8.5 / doc 08 §8 -- verbosity
# ---------------------------------------------------------------------------


def test_parse_args_verbose():
    assert app.parse_args(["--verbose"]).verbose is True


def test_there_is_no_quiet_flag():
    """Doc 08 §11 decision 2: `--quiet` has no consumer and is not in this
    batch. The systemd units in STEP-6 run non-interactively and are
    already covered by the non-tty rule in §7."""
    with pytest.raises(SystemExit):
        app.parse_args(["--quiet"])


def _renderer_kwargs(monkeypatch, tmp_path, **ctx_kwargs) -> dict:
    """Capture what `build_context` hands `progress.Progress`."""
    seen: dict = {}
    real = app.progress_mod.Progress

    def _spy(**kwargs):
        seen.update(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(app.progress_mod, "Progress", _spy)
    _minimal_ctx(tmp_path, **ctx_kwargs)
    return seen


def test_build_context_defaults_to_the_rolling_window(monkeypatch, tmp_path):
    """Doc 08 §8: the default is the fixed 8-line window, not the full
    stream and not silence."""
    seen = _renderer_kwargs(monkeypatch, tmp_path)

    assert seen["verbose"] is False
    assert app.progress_mod.WINDOW_LINES == 8


def test_build_context_passes_verbose_to_the_renderer(monkeypatch, tmp_path):
    seen = _renderer_kwargs(monkeypatch, tmp_path, verbose=True)

    assert seen["verbose"] is True
    assert seen["non_interactive"] is False


def test_verbose_and_non_interactive_are_independent(monkeypatch, tmp_path):
    """Two different questions: `--verbose` says how much of the stream to
    show, `--non-interactive` says whether anything may be drawn."""
    seen = _renderer_kwargs(
        monkeypatch, tmp_path, verbose=True, non_interactive=True
    )

    assert (seen["verbose"], seen["non_interactive"]) == (True, True)


def test_verbose_reaches_the_renderer_on_the_context(tmp_path):
    ctx, _cfg = _minimal_ctx(tmp_path, verbose=True)
    default_ctx, _cfg2 = _minimal_ctx(tmp_path)

    assert ctx.progress.renderer._verbose is True
    assert default_ctx.progress.renderer._verbose is False


def test_verbose_does_not_change_whether_run_root_streams(tmp_path):
    """Doc 00 §8.5's matrix: verbosity is a rendering decision, so it never
    moves the AUTO answer. A `stream=False` probe stays silent under
    `--verbose` too -- its line is still in the transcript."""
    verbose_ctx, verbose_recorder = _recording_ctx(tmp_path, verbose=True)
    plain_ctx, plain_recorder = _recording_ctx(tmp_path)

    _run_echo(verbose_ctx)
    _run_echo(plain_ctx)
    assert "out-line" in verbose_recorder.lines
    assert verbose_recorder.lines == plain_recorder.lines

    _run_echo(verbose_ctx, stream=False)
    assert verbose_recorder.lines == plain_recorder.lines


def test_non_interactive_does_not_change_whether_run_root_streams(tmp_path):
    """Doc 08 §7's third row: a non-interactive run still gets per-line
    output, plain. Suppressing the drawing is the renderer's job."""
    ctx, recorder = _recording_ctx(tmp_path, non_interactive=True)

    _run_echo(ctx)

    assert "out-line" in recorder.lines


def test_subcommand_context_honours_verbose(monkeypatch, tmp_path):
    """A standalone `mv3dt-installer amc ...` re-launch gets the same flag
    the install run does; the peek parser plucks it without consuming it."""
    seen: dict = {}

    monkeypatch.setattr(
        app.onboarding,
        "run_platform_preflight",
        lambda: InvokingUser(
            name="tester", home=tmp_path, uid=os.getuid(), gid=os.getgid()
        ),
    )
    monkeypatch.setattr(
        app,
        "build_context",
        lambda cfg, user, non_interactive, verbose=False: seen.update(
            non_interactive=non_interactive, verbose=verbose
        ),
    )

    app._bootstrap_subcommand_context(
        ["amc", "--verbose", "--some-subcommand-flag"],
        requires_root=False,
        state_path=tmp_path / "state.json",
    )

    assert seen == {"non_interactive": False, "verbose": True}


def test_run_root_streaming_redacts_the_terminal_not_the_parsed_buffer(
    tmp_path, monkeypatch
):
    """Doc 08 §4.2: showing a command live must not be how a secret becomes
    visible. The buffer a step parses is still the child's own text, the
    way `subprocess.run` handed it over."""
    ctx, recorder = _recording_ctx(tmp_path)
    monkeypatch.setenv("NGC_API_KEY", "nvapi-secret-value")

    result = ctx.run_root(
        sys.executable,
        "-c",
        "print('key is nvapi-secret-value')",
        capture_output=True,
        text=True,
    )

    assert result.stdout == "key is nvapi-secret-value\n"
    assert recorder.lines == ["key is <redacted>"]


def test_main_settles_colour_from_the_parsed_flag(tmp_path, monkeypatch):
    """Doc 08 §7 and §12.2 defect 4, via U13.

    Until `main()` runs, `logs` answers the colour question by reading
    `sys.argv` itself, because the decision is needed before `argparse` has
    run. That default is conservative and not authoritative: it is wrong for
    any caller passing its own `argv`, which is exactly what `main(argv)`
    does. This pins the parsed flag settling it.
    """
    seen: list[bool | None] = []
    monkeypatch.setattr(app, "set_colour", seen.append)
    monkeypatch.setattr(app.privilege, "require_root", lambda: None)

    app.main(["--status", "--non-interactive"], state_path=tmp_path / "state.json")
    assert seen == [False]

    seen.clear()
    app.main(["--status"], state_path=tmp_path / "state.json")
    assert seen == [True]
