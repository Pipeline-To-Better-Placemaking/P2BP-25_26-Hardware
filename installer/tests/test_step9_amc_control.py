"""Tests for mv3dt_installer.steps.step9_amc_control (STEP-9-AMC-CONTROL.md).

Run from installer/: `python3 -m pytest tests/test_step9_amc_control.py -v`

Nothing here touches real systemd, sudo, tailscale, pip or a socket: every
command goes through a recording runner on a fake `Context`, the unit and
sudoers directories are redirected into `tmp_path`, and the health probe is
an injected function.
"""

from __future__ import annotations

import json
import pathlib
import stat
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from mv3dt_installer import config as config_mod, logs, report, shellout, systemd  # noqa: E402
from mv3dt_installer.steps import StepStatus  # noqa: E402
from mv3dt_installer.steps import step8_tracking_history as s8  # noqa: E402
from mv3dt_installer.steps import step9_amc_control as s9  # noqa: E402

GATE = config_mod.GATE_AMC_CONTROL
TRACKING_GATE = config_mod.GATE_TRACKING_HISTORY
SECRET = "s3cret-value-that-must-never-be-printed"
TS_IP = "100.64.0.7"


class Runner:
    """Records argv tuples and answers from a small rule table."""

    def __init__(self, *, python_version="3.12", tailscale_ip=TS_IP, tailscale_status=None, fail=()):
        self.calls = []
        self.python_version = python_version
        self.tailscale_ip = tailscale_ip
        self.tailscale_status = tailscale_status
        self.fail = set(fail)
        self.active = True
        self.visudo_saw = []  # (content, mode) of each file visudo checked

    def __call__(self, *argv, **kwargs):
        argv = tuple(str(a) for a in argv)
        self.calls.append(argv)
        out, code = "", 0
        if argv[:2] == ("python3", "-c"):
            if self.python_version is None:
                code = 1
            out = self.python_version or ""
        elif argv[:2] == ("python3", "-m") and "venv" in argv:
            venv = pathlib.Path(argv[-1])
            (venv / "bin").mkdir(parents=True, exist_ok=True)
            (venv / "bin" / "python3").write_text("", encoding="utf-8")
        elif argv[:3] == ("tailscale", "ip", "-4"):
            if self.tailscale_ip is None:
                code = 1
            out = f"{self.tailscale_ip}\n" if self.tailscale_ip else ""
        elif argv[:3] == ("tailscale", "status", "--json"):
            if self.tailscale_status is None:
                code = 1
            else:
                out = json.dumps(self.tailscale_status)
        elif argv[:2] == ("visudo", "-cf"):
            path = pathlib.Path(argv[2])
            self.visudo_saw.append((path.read_text(encoding="utf-8"), stat.S_IMODE(path.stat().st_mode)))
        elif argv[:2] == ("systemctl", "is-enabled"):
            out = "enabled\n"
        elif argv[:2] == ("systemctl", "is-active"):
            code = 0 if self.active else 3
        for needle in self.fail:
            if needle in argv:
                code = 1
        return SimpleNamespace(returncode=code, stdout=out, stderr="boom" if code else "")

    def named(self, *prefix):
        return [c for c in self.calls if c[: len(prefix)] == prefix]


class Context:
    def __init__(self, tmp_path, *, conf=None, runner=None, tracking_env=True, amc_bin=True):
        self.install_dir = tmp_path / "mv3dt"
        self.install_dir.mkdir(parents=True, exist_ok=True)
        self.user = SimpleNamespace(name="operator", uid=1000, gid=1000, home=tmp_path)
        self.conf = {GATE: "on", TRACKING_GATE: "on"} if conf is None else conf
        self.non_interactive = True
        self.runner = runner or Runner()
        self.log = logs.log
        self.reported = []
        self.progress = SimpleNamespace(phase=lambda n: None, task=lambda n: None)
        self.asset_path = shellout.asset_path
        if amc_bin:
            (self.install_dir / "bin").mkdir(exist_ok=True)
            (self.install_dir / "bin" / "amc").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
        if tracking_env:
            secrets_dir = self.install_dir / "secrets"
            secrets_dir.mkdir(exist_ok=True)
            (secrets_dir / "tracking.env").write_text(
                f"WORKSTATION_TICKET_SECRET={SECRET}\nTRACKING_PROJECT_ID=proj-1\n"
                "ALLOWED_ORIGINS=https://app.example.org\n",
                encoding="utf-8",
            )

    def report_installed(self, name, version):
        self.reported.append(("installed", name, version))
        report.report_installed(name, version)

    def report_already_installed(self, name, version):
        self.reported.append(("already", name, version))
        report.report_already_installed(name, version)

    def run_root(self, *args, **kwargs):
        return self.runner(*args, **kwargs)

    def run_as_user(self, *args, **kwargs):
        return self.runner(*args, **kwargs)


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    logs._transcript_path = None
    monkeypatch.setattr(systemd, "UNIT_DIR", tmp_path / "units")
    monkeypatch.setattr(s9, "SUDOERS_DIR", tmp_path / "sudoers.d")
    monkeypatch.setattr(s9, "_WHICH", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(s8, "_WHICH", lambda name: f"/usr/bin/{name}")  # tailscale_state is Step 8's
    monkeypatch.setattr(s9, "_SLEEP", lambda s: None)
    monkeypatch.setattr(s9, "_HTTP_GET", _healthy)
    yield
    logs._transcript_path = None


def _healthy(url):
    assert url == s9.HEALTH_URL
    return 200, json.dumps({"status": "ok", "version": "0.1.0", "ticket_configured": True}).encode()


def _sudoers(tmp_path):
    return tmp_path / "sudoers.d" / s9.SUDOERS_NAME


def _unit(tmp_path):
    return (tmp_path / "units" / s9.UNIT_NAME).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------


def test_gate_off_is_a_no_op(tmp_path):
    ctx = Context(tmp_path, conf={TRACKING_GATE: "on"})
    step = s9.Step9AmcControl()

    for phase in (step.preflight, step.run, step.verify):
        assert phase(ctx).status is StepStatus.COMPLETE
    step.report(ctx)
    assert step.pending_update(ctx) is None

    assert ctx.runner.calls == []
    assert not (ctx.install_dir / "amc_control").exists()
    assert not _sudoers(tmp_path).exists()


def test_step_is_registered_as_step_9():
    from mv3dt_installer.state import STEP_IDS
    from mv3dt_installer.steps import STEP_REGISTRY

    assert STEP_IDS[-1] == s9.STEP_ID
    assert any(step.id == s9.STEP_ID and step.order == 9 for step in STEP_REGISTRY)


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------


def test_preflight_passes_on_a_ready_box(tmp_path):
    assert s9.Step9AmcControl().preflight(Context(tmp_path)).status is StepStatus.COMPLETE


def test_preflight_needs_python_venv(tmp_path):
    result = s9.Step9AmcControl().preflight(Context(tmp_path, runner=Runner(python_version=None)))
    assert result.status is StepStatus.USER_ACTION_REQUIRED
    assert "python3-venv" in result.user_actions[0].command


def test_preflight_needs_the_amc_launcher(tmp_path):
    result = s9.Step9AmcControl().preflight(Context(tmp_path, amc_bin=False))
    assert result.status is StepStatus.USER_ACTION_REQUIRED
    assert "Step 3" in result.user_actions[0].text


def test_preflight_needs_step_8s_secret_file(tmp_path):
    result = s9.Step9AmcControl().preflight(Context(tmp_path, tracking_env=False))
    assert result.status is StepStatus.USER_ACTION_REQUIRED
    assert "WORKSTATION_TICKET_SECRET" in result.message
    assert "--tracking-history on" in result.user_actions[0].text


def test_preflight_needs_the_tracking_gate_on(tmp_path):
    result = s9.Step9AmcControl().preflight(Context(tmp_path, conf={GATE: "on"}))
    assert result.status is StepStatus.USER_ACTION_REQUIRED
    assert "turned off" in result.message


def test_preflight_needs_a_tailscale_address(tmp_path):
    result = s9.Step9AmcControl().preflight(Context(tmp_path, runner=Runner(tailscale_ip=None)))
    assert result.status is StepStatus.USER_ACTION_REQUIRED
    assert result.user_actions[0].command == "sudo tailscale up"


def test_preflight_needs_visudo(tmp_path, monkeypatch):
    monkeypatch.setattr(s9, "_WHICH", lambda name: None if name == "visudo" else f"/usr/bin/{name}")
    result = s9.Step9AmcControl().preflight(Context(tmp_path))
    assert result.status is StepStatus.USER_ACTION_REQUIRED and "visudo" in result.message


def test_preflight_refuses_an_install_dir_sudoers_cannot_express(tmp_path):
    ctx = Context(tmp_path)
    ctx.install_dir = tmp_path / "my install,dir"
    (ctx.install_dir / "bin").mkdir(parents=True)
    (ctx.install_dir / "bin" / "amc").write_text("", encoding="utf-8")
    (ctx.install_dir / "secrets").mkdir()
    (ctx.install_dir / "secrets" / "tracking.env").write_text(
        f"WORKSTATION_TICKET_SECRET={SECRET}\nTRACKING_PROJECT_ID=p\nALLOWED_ORIGINS=https://a.example\n",
        encoding="utf-8",
    )
    result = s9.Step9AmcControl().preflight(ctx)
    assert result.status is StepStatus.FAILED and "sudoers" in result.message


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def test_run_installs_everything(tmp_path):
    ctx = Context(tmp_path, runner=Runner(tailscale_status={"BackendState": "Running"}))
    result = s9.Step9AmcControl().run(ctx)
    assert result.status is StepStatus.COMPLETE, result.message

    control = ctx.install_dir / "amc_control"
    assert (control / "mv3dt_amc_control" / "api.py").is_file()
    assert (control / "mv3dt_amc_control" / "amc_automation.example.yaml").is_file()
    assert (control / "requirements.txt").read_text() == shellout.asset_path(
        "amc_control", "requirements.txt"
    ).read_text()
    assert not list(control.rglob("__pycache__"))
    assert stat.S_IMODE((control / "state").stat().st_mode) == 0o750

    # venv built and requirements installed as the invoking user
    assert ctx.runner.named("python3", "-m", "venv")
    pip = [c for c in ctx.runner.calls if "pip" in c]
    assert pip and pip[0][-1] == str(control / "requirements.txt")

    assert ctx.runner.named("systemctl", "restart", s9.UNIT_NAME)
    assert ctx.runner.named(*s9.TAILSCALE_SERVE_ARGV)
    assert ctx.reported == [("installed", s9.UNIT_NAME, s9.bundled_version(ctx))]


def test_sudo_rule_allows_exactly_the_two_amc_commands(tmp_path):
    ctx = Context(tmp_path)
    s9.Step9AmcControl().run(ctx)

    path = _sudoers(tmp_path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o440
    amc = ctx.install_dir / "bin" / "amc"
    rules = [line for line in path.read_text().splitlines() if line and not line.startswith("#")]
    assert rules == [
        f"operator ALL=(root) NOPASSWD: {amc} --keep-up --no-open --non-interactive "
        f"--host-ip {TS_IP}, {amc} --down"
    ]
    assert "ALL=(ALL)" not in path.read_text() and "NOPASSWD: ALL" not in path.read_text()


def test_sudo_rule_matches_what_the_service_runs(tmp_path):
    """The rule is only useful if it equals the controller's argv."""
    pytest.importorskip("fastapi")
    sys.path.insert(0, str(shellout.asset_path("amc_control")))
    import inspect

    from mv3dt_amc_control import controller

    source = inspect.getsource(controller.AmcController)
    assert json.dumps(list(s9.AMC_START_ARGS))[1:-1] in source
    assert '["--down"]' in source and s9.AMC_STOP_ARGS == ("--down",)


def test_sudo_rule_is_checked_by_visudo_before_it_is_installed(tmp_path):
    ctx = Context(tmp_path)
    s9.Step9AmcControl().run(ctx)

    (content, mode), = ctx.runner.visudo_saw
    assert content == _sudoers(tmp_path).read_text() and mode == 0o440
    checked = pathlib.Path(ctx.runner.named("visudo", "-cf")[0][2])
    assert "." in checked.name          # a name sudo ignores while unchecked
    assert not checked.exists()         # and it does not linger


def test_a_rule_visudo_rejects_is_never_installed(tmp_path):
    ctx = Context(tmp_path, runner=Runner(fail={"visudo"}))
    result = s9.Step9AmcControl().run(ctx)

    assert result.status is StepStatus.FAILED and "visudo rejected" in result.message
    assert list((tmp_path / "sudoers.d").iterdir()) == []
    assert not (tmp_path / "units" / s9.UNIT_NAME).exists()


def test_unit_runs_the_package_with_the_shared_secret_file(tmp_path):
    ctx = Context(tmp_path)
    s9.Step9AmcControl().run(ctx)
    unit = _unit(tmp_path)
    control = ctx.install_dir / "amc_control"

    assert "User=operator" in unit
    assert f"EnvironmentFile={ctx.install_dir}/installer.conf" in unit
    assert f"EnvironmentFile={ctx.install_dir}/secrets/tracking.env" in unit
    assert f'ExecStart="{control}/venv/bin/python3" -m mv3dt_amc_control serve' in unit
    for line in (
        f'Environment="PYTHONPATH={control}"',
        f'Environment="MV3DT_INSTALL_DIR={ctx.install_dir}"',
        f'Environment="AMC_STATE_DIR={control}/state"',
        f'Environment="AMC_AUTOMATION_CONFIG={control}/amc_automation.yaml"',
        f'Environment="AMC_PUBLIC_HOST={TS_IP}"',
        'Environment="BIND=127.0.0.1:8091"',
    ):
        assert line in unit
    assert "NoNewPrivileges" not in unit.split("[Unit]")[1]   # sudo must keep working
    assert "@" not in unit.split("[Unit]")[1].replace("multi-user", "")


def test_the_secret_never_leaves_its_file(tmp_path, capsys):
    ctx = Context(tmp_path, runner=Runner(tailscale_status={"BackendState": "Running"}))
    step = s9.Step9AmcControl()
    step.run(ctx)
    step.verify(ctx)
    step.report(ctx)

    captured = capsys.readouterr()
    assert SECRET not in captured.out + captured.err
    assert SECRET not in _unit(tmp_path) and SECRET not in _sudoers(tmp_path).read_text()
    assert all(SECRET not in " ".join(call) for call in ctx.runner.calls)


def test_config_is_seeded_once_with_a_safe_project_mode(tmp_path):
    ctx = Context(tmp_path)
    s9.Step9AmcControl().run(ctx)
    path = ctx.install_dir / "amc_control" / "amc_automation.yaml"
    assert "mode: create_new" in path.read_text()

    path.write_text("project:\n  mode: reuse_installer_project\n")
    s9.Step9AmcControl().run(ctx)
    assert path.read_text() == "project:\n  mode: reuse_installer_project\n"


def test_seeded_config_is_valid_for_the_package(tmp_path):
    pytest.importorskip("yaml")
    pytest.importorskip("pydantic")
    sys.path.insert(0, str(shellout.asset_path("amc_control")))
    from mv3dt_amc_control.automation_config import load_config

    ctx = Context(tmp_path)
    s9.Step9AmcControl().run(ctx)
    cfg = load_config(ctx.install_dir / "amc_control" / "amc_automation.yaml")
    assert cfg.project.mode == "create_new" and cfg.layout.allowed_hosts == []


def test_second_run_changes_nothing(tmp_path):
    ctx = Context(tmp_path)
    step = s9.Step9AmcControl()
    step.run(ctx)
    ctx.runner.calls.clear()
    ctx.reported.clear()

    assert step.run(ctx).status is StepStatus.COMPLETE
    assert ctx.reported == [("already", s9.UNIT_NAME, s9.bundled_version(ctx))]
    assert not ctx.runner.named("systemctl", "restart")
    assert not ctx.runner.named("visudo")
    assert not [c for c in ctx.runner.calls if "pip" in c]


def test_run_stops_without_a_tailscale_address(tmp_path):
    ctx = Context(tmp_path, runner=Runner(tailscale_ip=None))
    result = s9.Step9AmcControl().run(ctx)
    assert result.status is StepStatus.USER_ACTION_REQUIRED
    assert not (ctx.install_dir / "amc_control").exists()


def test_run_stops_when_tracking_env_is_incomplete(tmp_path):
    ctx = Context(tmp_path)
    (ctx.install_dir / "secrets" / "tracking.env").write_text(f"WORKSTATION_TICKET_SECRET={SECRET}\n")
    result = s9.Step9AmcControl().run(ctx)
    assert result.status is StepStatus.USER_ACTION_REQUIRED
    assert "TRACKING_PROJECT_ID" in result.message and SECRET not in result.message


def test_pip_failure_fails_the_step(tmp_path):
    result = s9.Step9AmcControl().run(Context(tmp_path, runner=Runner(fail={"pip"})))
    assert result.status is StepStatus.FAILED and "pip install" in result.message


def test_tailscale_serve_failure_is_only_a_warning(tmp_path, capsys):
    ctx = Context(tmp_path, runner=Runner(fail={"serve"}))
    assert s9.Step9AmcControl().run(ctx).status is StepStatus.COMPLETE
    assert s9.TAILSCALE_SERVE_COMMAND in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Upgrade
# ---------------------------------------------------------------------------


def test_no_update_pending_right_after_install(tmp_path):
    ctx = Context(tmp_path)
    step = s9.Step9AmcControl()
    step.run(ctx)
    assert step.pending_update(ctx) is None


def test_changed_package_is_an_update(tmp_path):
    ctx = Context(tmp_path)
    step = s9.Step9AmcControl()
    step.run(ctx)
    (ctx.install_dir / "amc_control" / "mv3dt_amc_control" / "api.py").write_text("# old\n")
    assert "package changed" in step.pending_update(ctx)


def test_changed_tailscale_address_rewrites_unit_and_rule(tmp_path):
    ctx = Context(tmp_path)
    step = s9.Step9AmcControl()
    step.run(ctx)

    ctx.runner.tailscale_ip = "100.64.0.99"
    assert s9.UNIT_NAME in step.pending_update(ctx)
    ctx.update_pending = "address changed"
    assert step.run(ctx).status is StepStatus.COMPLETE
    assert "--host-ip 100.64.0.99" in _sudoers(tmp_path).read_text()
    assert 'AMC_PUBLIC_HOST=100.64.0.99' in _unit(tmp_path)
    assert step.pending_update(ctx) is None


def test_status_without_a_user_checks_only_the_package(tmp_path):
    ctx = Context(tmp_path)
    s9.Step9AmcControl().run(ctx)
    ctx.user = None
    ctx.runner.calls.clear()
    assert s9.update_reason(ctx) is None
    assert ctx.runner.calls == []


# ---------------------------------------------------------------------------
# Verify and report
# ---------------------------------------------------------------------------


def test_verify_passes_after_run(tmp_path):
    ctx = Context(tmp_path)
    step = s9.Step9AmcControl()
    step.run(ctx)
    assert step.verify(ctx).status is StepStatus.COMPLETE


def test_verify_reports_an_inactive_unit(tmp_path):
    ctx = Context(tmp_path)
    step = s9.Step9AmcControl()
    step.run(ctx)
    ctx.runner.active = False
    result = step.verify(ctx)
    assert result.status is StepStatus.USER_ACTION_REQUIRED
    assert "journalctl -u mv3dt-amc-control" in result.user_actions[0].command


def test_verify_fails_on_a_missing_or_loose_sudo_rule(tmp_path):
    ctx = Context(tmp_path)
    step = s9.Step9AmcControl()
    step.run(ctx)
    _sudoers(tmp_path).chmod(0o644)
    assert step.verify(ctx).status is StepStatus.FAILED
    _sudoers(tmp_path).unlink()
    assert "missing" in step.verify(ctx).message


def test_verify_notices_a_rule_for_an_old_address(tmp_path):
    ctx = Context(tmp_path)
    step = s9.Step9AmcControl()
    step.run(ctx)
    ctx.runner.tailscale_ip = "100.64.0.99"
    result = step.verify(ctx)
    assert result.status is StepStatus.USER_ACTION_REQUIRED and "100.64.0.99" in result.message


def test_verify_fails_when_the_service_does_not_answer(tmp_path, monkeypatch):
    ctx = Context(tmp_path)
    step = s9.Step9AmcControl()
    step.run(ctx)
    monkeypatch.setattr(s9, "_HTTP_GET", lambda url: (None, b""))
    result = step.verify(ctx)
    assert result.status is StepStatus.FAILED and s9.HEALTH_URL in result.message


def test_report_tells_the_operator_what_to_set_in_the_web_app(tmp_path, capsys):
    status = {"BackendState": "Running", "Self": {"DNSName": "ws.tailnet.ts.net."}}
    ctx = Context(tmp_path, runner=Runner(tailscale_status=status))
    step = s9.Step9AmcControl()
    step.run(ctx)
    capsys.readouterr()
    step.report(ctx)
    err = capsys.readouterr().err
    assert "WORKSTATION_BASE_URL=https://ws.tailnet.ts.net:8444" in err
    assert "amc_automation.yaml" in err and "journalctl -u mv3dt-amc-control" in err
