"""Tests for mv3dt_installer.steps.step8_tracking_history
(STEP-8-TRACKING-HISTORY.md).

Run from installer/: `python3 -m pytest tests/test_step8_tracking_history.py -v`

Nothing here touches real systemd, tailscale, pip or a socket: every
command goes through a recording runner on a fake `Context`, the unit
directory and data directory are redirected into `tmp_path`, and the health
probe is an injected function.
"""

from __future__ import annotations

import json
import pathlib
import os
import stat
import subprocess
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from mv3dt_installer import app, config as config_mod, logs, report, shellout, systemd  # noqa: E402
from mv3dt_installer.state import StateMachine  # noqa: E402
from mv3dt_installer.steps import StepStatus  # noqa: E402
from mv3dt_installer.steps import step8_tracking_history as s8  # noqa: E402

GATE = config_mod.GATE_TRACKING_HISTORY


class Runner:
    """Records argv tuples and answers from a small rule table."""

    def __init__(self, *, python_version="3.12", tailscale_status=None, fail=()):
        self.calls = []
        self.python_version = python_version
        self.tailscale_status = tailscale_status
        self.fail = set(fail)
        self.active = None  # None: is-active succeeds except in wipe tests
        self.kwargs = []
        self.exit_code = 0  # what a package run (`-m mv3dt_tracking`) returns

    def __call__(self, *argv, **kwargs):
        argv = tuple(str(a) for a in argv)
        self.calls.append(argv)
        self.kwargs.append(kwargs)
        out, code = "", 0
        if argv[1:3] == ("-I", "-c") and argv[3] == s8._SEED_PROGRAM:
            # The real seed program, run against the copied package.
            done = subprocess.run(
                [sys.executable, *argv[1:]], capture_output=True, text=True, check=False
            )
            out, code = done.stdout, done.returncode
        elif "-m" in argv and s8.PACKAGE_NAME in argv:
            code = self.exit_code
        elif argv[:2] == ("python3", "-c"):
            if self.python_version is None:
                code = 1
            out = self.python_version or ""
        elif argv[:2] == ("python3", "-m") and "venv" in argv:
            venv = pathlib.Path(argv[-1])
            (venv / "bin").mkdir(parents=True, exist_ok=True)
            (venv / "bin" / "python3").write_text("", encoding="utf-8")
        elif argv[:3] == ("tailscale", "status", "--json"):
            if self.tailscale_status is None:
                code = 1
            else:
                out = json.dumps(self.tailscale_status)
        elif argv[:2] == ("systemctl", "is-enabled"):
            out = "enabled\n"
        elif argv[:2] == ("systemctl", "is-active") and self.active is not None:
            code = 0 if self.active else 3
        for needle in self.fail:
            if needle in argv:
                code = 1
        return SimpleNamespace(returncode=code, stdout=out, stderr="")

    def named(self, *prefix):
        return [c for c in self.calls if c[: len(prefix)] == prefix]


class Context:
    def __init__(self, tmp_path, *, conf=None, runner=None, non_interactive=True):
        self.install_dir = tmp_path / "mv3dt"
        self.install_dir.mkdir(parents=True, exist_ok=True)
        self.user = SimpleNamespace(name="operator", uid=1000, gid=1000, home=tmp_path)
        self.conf = {GATE: "on", "LOCATION_ID": "site-42"} if conf is None else conf
        self.non_interactive = non_interactive
        self.runner = runner or Runner()
        self.log = logs.log
        self.reported = []
        self.progress = SimpleNamespace(phase=lambda n: None, task=lambda n: None)
        self.asset_path = shellout.asset_path

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
    monkeypatch.setattr(s8, "DATA_DIR", tmp_path / "var-lib" / "mv3dt-tracking")
    monkeypatch.setattr(s8, "_WHICH", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(s8, "_SLEEP", lambda s: None)
    monkeypatch.setattr(s8, "_PROMPT", _no_prompt)
    for key in (s8.KEY_PROJECT, s8.KEY_ORIGINS, s8.KEY_ISSUER, s8.KEY_SECRET):
        monkeypatch.delenv(key, raising=False)
    yield
    logs._transcript_path = None


def _no_prompt(msg):
    raise AssertionError(f"unexpected prompt: {msg!r}")


@pytest.fixture
def env_values(monkeypatch):
    monkeypatch.setenv(s8.KEY_PROJECT, "proj-1")
    monkeypatch.setenv(s8.KEY_ORIGINS, "https://app.example.org/, http://localhost:5173")


def _health(seq):
    """A health probe that returns each recorder snapshot in turn."""
    docs = list(seq)

    def probe(url):
        assert url == s8.HEALTH_URL
        doc = docs.pop(0) if len(docs) > 1 else docs[0]
        if doc is None:
            return None, b""
        return 200, json.dumps({"status": "ok", "recorder": doc}).encode()

    return probe


def _logged_in():
    return {"BackendState": "Running", "Self": {"DNSName": "ws.tailnet.ts.net."}}


# ---------------------------------------------------------------------------
# Gate (section 4.1)
# ---------------------------------------------------------------------------


def test_gate_off_is_a_no_op(tmp_path):
    ctx = Context(tmp_path, conf={"LOCATION_ID": "site-42"})
    step = s8.Step8TrackingHistory()

    for phase in (step.preflight, step.run, step.verify):
        assert phase(ctx).status is StepStatus.COMPLETE
    step.report(ctx)

    assert ctx.runner.calls == []
    assert not (ctx.install_dir / "tracking").exists()
    assert not (ctx.install_dir / "secrets").exists()
    assert not systemd.UNIT_DIR.exists()


def test_step_is_registered_and_gated_by_order_8():
    import mv3dt_installer.__main__  # noqa: F401

    assert any(step.id == s8.STEP_ID and step.order == 8 for step in app.STEP_REGISTRY)
    assert app._GATE_ORDER_TO_CONF_ATTR[8] == "tracking_history"
    assert "tracking" in app.SUBCOMMAND_REGISTRY


def test_flag_plumbs_into_gate_overrides():
    args = app.parse_args(["--tracking-history", "on"])
    assert app._gate_overrides_from_args(args) == {GATE: "on"}
    assert app.parse_args([]).tracking_history is None
    with pytest.raises(SystemExit):
        app.parse_args(["--tracking-history", "maybe"])


def test_dispatch_skips_step8_when_gate_off(tmp_path):
    cfg = config_mod.Config(install_dir=tmp_path, values={})
    step = s8.Step8TrackingHistory()
    assert app._gate_is_off(app._gate_value_for_step(step, cfg))


# ---------------------------------------------------------------------------
# Preflight (section 11.1)
# ---------------------------------------------------------------------------


def test_preflight_warns_but_passes_without_calibration_or_tailscale(tmp_path, capsys):
    ctx = Context(tmp_path, runner=Runner(tailscale_status=None))

    result = s8.Step8TrackingHistory().preflight(ctx)

    assert result.status is StepStatus.COMPLETE
    err = capsys.readouterr().err
    assert "map.png is missing" in err
    assert "transforms.yml is missing" in err
    assert "no camera models" in err
    assert "tailscale is not logged in" in err


def test_preflight_uses_persisted_calibration_dir(tmp_path, capsys):
    calib = tmp_path / "calib"
    calib.mkdir()
    (calib / "map.png").write_bytes(b"png")
    (calib / "transforms.yml").write_text("x: 1\n", encoding="utf-8")
    ctx = Context(
        tmp_path,
        conf={GATE: "on", "LOCATION_ID": "site-42", "CALIBRATION_DIR": str(calib)},
        runner=Runner(tailscale_status=_logged_in()),
    )
    models = ctx.install_dir / "deepstream" / "camera_models"
    models.mkdir(parents=True)
    (models / "camInfo_01.yml").write_text("{}\n", encoding="utf-8")

    assert s8.Step8TrackingHistory().preflight(ctx).status is StepStatus.COMPLETE
    assert "step8:" not in capsys.readouterr().err


def test_preflight_tailscale_missing_is_a_warning(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(s8, "_WHICH", lambda name: None if name == "tailscale" else "/x")
    ctx = Context(tmp_path)
    assert s8.Step8TrackingHistory().preflight(ctx).status is StepStatus.COMPLETE
    assert "tailscale is not installed" in capsys.readouterr().err


@pytest.mark.parametrize("version", [None, "3.8"])
def test_preflight_requires_python_310_with_venv(tmp_path, version):
    ctx = Context(tmp_path, runner=Runner(python_version=version))
    result = s8.Step8TrackingHistory().preflight(ctx)
    assert result.status is StepStatus.USER_ACTION_REQUIRED
    assert "python3-venv" in result.user_actions[0].command


def test_preflight_requires_mosquitto(tmp_path, monkeypatch):
    monkeypatch.setattr(s8, "_WHICH", lambda name: None)
    monkeypatch.setattr(s8.pathlib.Path, "exists", lambda self: False)
    ctx = Context(tmp_path)
    result = s8.Step8TrackingHistory().preflight(ctx)
    assert result.status is StepStatus.USER_ACTION_REQUIRED
    assert "Mosquitto" in result.message


def test_preflight_requires_location_id(tmp_path):
    ctx = Context(tmp_path, conf={GATE: "on"})
    result = s8.Step8TrackingHistory().preflight(ctx)
    assert result.status is StepStatus.USER_ACTION_REQUIRED
    assert "LOCATION_ID" in result.message


# ---------------------------------------------------------------------------
# Package copy and venv (section 11.2 steps 1-2)
# ---------------------------------------------------------------------------


def test_package_copy_is_idempotent_and_replaces_on_change(tmp_path):
    ctx = Context(tmp_path)
    dest = ctx.install_dir / "tracking" / "mv3dt_tracking"

    assert s8.sync_package(ctx) is True
    assert (dest / "__init__.py").is_file()
    assert (dest / "cli.py").is_file()
    assert (ctx.install_dir / "tracking" / "requirements.txt").is_file()
    assert not list(dest.rglob("__pycache__"))

    assert s8.sync_package(ctx) is False

    (dest / "__init__.py").write_text('__version__ = "0.0.1"\n', encoding="utf-8")
    (dest / "stale.py").write_text("", encoding="utf-8")
    assert s8.sync_package(ctx) is True
    assert not (dest / "stale.py").exists()
    assert s8.bundled_version(ctx) in (dest / "__init__.py").read_text(encoding="utf-8")


def test_bundled_version_matches_package():
    init = shellout.asset_path("tracking", "mv3dt_tracking", "__init__.py")
    ctx = SimpleNamespace(asset_path=shellout.asset_path)
    assert f'"{s8.bundled_version(ctx)}"' in init.read_text(encoding="utf-8")


def test_venv_and_pip_commands(tmp_path):
    ctx = Context(tmp_path)
    s8.sync_package(ctx)
    venv = ctx.install_dir / "tracking" / "venv"
    requirements = ctx.install_dir / "tracking" / "requirements.txt"

    changed, error = s8.ensure_venv(ctx)

    assert (changed, error) == (True, None)
    assert ctx.runner.named("python3", "-m", "venv") == [
        ("python3", "-m", "venv", str(venv))
    ]
    pip = ctx.runner.named(str(venv / "bin" / "python3"), "-m", "pip")
    assert pip == [
        (
            str(venv / "bin" / "python3"), "-m", "pip", "install",
            "--disable-pip-version-check", "--no-input", "-r", str(requirements),
        )
    ]

    ctx.runner.calls.clear()
    assert s8.ensure_venv(ctx) == (False, None)
    assert ctx.runner.calls == []

    requirements.write_text(requirements.read_text() + "extra==1.0\n")
    assert s8.ensure_venv(ctx) == (True, None)
    assert len(ctx.runner.named(str(venv / "bin" / "python3"), "-m", "pip")) == 1


def test_requirements_are_pinned():
    text = shellout.asset_path("tracking", "requirements.txt").read_text()
    lines = [l for l in text.splitlines() if l.strip() and not l.startswith("#")]
    assert lines and all("==" in l for l in lines)


def test_pip_failure_fails_run(tmp_path, env_values):
    ctx = Context(tmp_path, runner=Runner(fail={"pip"}))
    result = s8.Step8TrackingHistory().run(ctx)
    assert result.status is StepStatus.FAILED
    assert "pip install" in result.message
    assert not (systemd.UNIT_DIR / s8.UNIT_NAME).exists()


# ---------------------------------------------------------------------------
# Secrets (section 4.3)
# ---------------------------------------------------------------------------


def test_secret_created_600_never_overwritten_never_logged(tmp_path, env_values, capsys):
    ctx = Context(tmp_path)
    path = ctx.install_dir / "secrets" / "tracking.env"

    changed, missing = s8.ensure_tracking_env(ctx)

    assert changed is True and missing == []
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    values = s8._read_env_file(path)
    secret = values[s8.KEY_SECRET]
    assert len(secret) >= 64
    assert values[s8.KEY_PROJECT] == "proj-1"
    assert values[s8.KEY_ORIGINS] == "https://app.example.org,http://localhost:5173"
    assert s8.KEY_ISSUER not in values  # never asked, so left unset

    assert s8.ensure_tracking_env(ctx) == (False, [])
    assert s8._read_env_file(path)[s8.KEY_SECRET] == secret

    captured = capsys.readouterr()
    assert secret not in captured.err + captured.out
    assert "WORKSTATION_TICKET_SECRET=<redacted>" in captured.err


def test_existing_values_are_kept_and_missing_keys_added(tmp_path, monkeypatch):
    ctx = Context(tmp_path)
    path = ctx.install_dir / "secrets" / "tracking.env"
    path.parent.mkdir(parents=True)
    path.write_text("WORKSTATION_TICKET_SECRET=keepme\nMQTT_HOST=127.0.0.1\n")
    monkeypatch.setenv(s8.KEY_PROJECT, "proj-2")
    monkeypatch.setenv(s8.KEY_ORIGINS, "https://a.example.org")
    monkeypatch.setenv(s8.KEY_ISSUER, "p2bp")

    changed, missing = s8.ensure_tracking_env(ctx)

    values = s8._read_env_file(path)
    assert changed is True and missing == []
    assert values[s8.KEY_SECRET] == "keepme"
    assert values["MQTT_HOST"] == "127.0.0.1"
    assert values[s8.KEY_ISSUER] == "p2bp"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_interactive_prompts_for_values(tmp_path, monkeypatch):
    answers = iter(["proj-3", "not-a-url", "https://b.example.org", ""])
    monkeypatch.setattr(s8, "_PROMPT", lambda msg: next(answers))
    ctx = Context(tmp_path, non_interactive=False)

    assert s8.ensure_tracking_env(ctx)[1] == []
    values = s8._read_env_file(ctx.install_dir / "secrets" / "tracking.env")
    assert values[s8.KEY_PROJECT] == "proj-3"
    assert values[s8.KEY_ORIGINS] == "https://b.example.org"
    assert values[s8.KEY_ISSUER] == ""  # answered blank: never asked again


def test_non_interactive_without_values_stops_before_the_unit(tmp_path):
    ctx = Context(tmp_path)
    result = s8.Step8TrackingHistory().run(ctx)

    assert result.status is StepStatus.USER_ACTION_REQUIRED
    assert s8.KEY_PROJECT in result.message and s8.KEY_ORIGINS in result.message
    path = ctx.install_dir / "secrets" / "tracking.env"
    assert s8._read_env_file(path).get(s8.KEY_SECRET)
    assert not (systemd.UNIT_DIR / s8.UNIT_NAME).exists()


# ---------------------------------------------------------------------------
# Unit (section 4.4)
# ---------------------------------------------------------------------------


def test_unit_renders_fully_with_correct_execstart(tmp_path):
    ctx = Context(tmp_path)
    text = s8.render_tracking_unit(ctx)
    tracking = ctx.install_dir / "tracking"

    assert not systemd._MARKER_RE.search(text)
    assert f'ExecStart="{tracking}/venv/bin/python3" -m mv3dt_tracking serve' in text
    assert "User=operator" in text
    assert f"EnvironmentFile={ctx.install_dir}/installer.conf" in text
    assert f"EnvironmentFile={ctx.install_dir}/secrets/tracking.env" in text
    assert f'Environment="DATA_DIR={s8.DATA_DIR}"' in text
    calib = ctx.install_dir / "deepstream" / "calibration" / "site-42"
    assert f'Environment="CALIB_DIR={calib}"' in text
    assert f'Environment="PYTHONPATH={tracking}"' in text
    assert 'Environment="BIND=127.0.0.1:8090"' in text
    assert f'Environment="CLEANING_FILE={tracking}/cleaning.json"' in text
    assert f'Environment="OFFSETS_FILE={tracking}/offsets.json"' in text
    assert 'Environment="PYTHONUNBUFFERED=1"' in text
    assert "After=network.target mosquitto.service" in text
    assert "Wants=mosquitto.service" in text
    assert "Restart=on-failure" in text
    assert "StandardOutput=journal" in text
    assert "0.0.0.0" not in text


def _unit_environment(text):
    """The unit's Environment= lines as a dict, in order."""
    out = {}
    for line in text.splitlines():
        if line.startswith("Environment="):
            key, _, value = line[len("Environment="):].strip('"').partition("=")
            out[key] = value
    return out


def test_unit_environment_is_the_service_environment(tmp_path):
    ctx = Context(tmp_path)
    env = s8.service_environment(ctx)
    assert _unit_environment(s8.render_tracking_unit(ctx)) == env
    # The same paths the package falls back to when the variables are unset.
    tracking = ctx.install_dir / "tracking"
    assert env["CLEANING_FILE"] == str(tracking / "cleaning.json")
    assert env["OFFSETS_FILE"] == str(tracking / "offsets.json")
    assert env["BIND"] == s8.LISTEN_URL.removeprefix("http://")
    assert not any(s8.KEY_SECRET in k or s8.KEY_SECRET in v for k, v in env.items())


def test_unit_escapes_percent_in_environment(tmp_path):
    ctx = Context(tmp_path, conf={GATE: "on", "LOCATION_ID": "site%42"})
    env = _unit_environment(s8.render_tracking_unit(ctx))
    assert env["CALIB_DIR"].endswith("/site%%42")


def test_unit_rejects_unquotable_paths(tmp_path):
    ctx = Context(tmp_path, conf={GATE: "on", "LOCATION_ID": 'bad"id'})
    with pytest.raises(ValueError):
        s8.render_tracking_unit(ctx)


# ---------------------------------------------------------------------------
# Full run and tailscale (section 11.2 steps 5-7)
# ---------------------------------------------------------------------------


def test_run_installs_and_exposes_when_tailscale_logged_in(tmp_path, env_values, capsys):
    ctx = Context(tmp_path, runner=Runner(tailscale_status=_logged_in()))

    result = s8.Step8TrackingHistory().run(ctx)

    assert result.status is StepStatus.COMPLETE
    assert (systemd.UNIT_DIR / s8.UNIT_NAME).is_file()
    assert ("systemctl", "daemon-reload") in ctx.runner.calls
    assert ("systemctl", "enable", "--now", s8.UNIT_NAME) in ctx.runner.calls
    assert ctx.runner.named("tailscale", "serve") == [
        ("tailscale", "serve", "--bg", "--https=8443", "http://127.0.0.1:8090")
    ]
    assert not any("funnel" in c for call in ctx.runner.calls for c in call)
    version = s8.bundled_version(ctx)
    assert ctx.reported == [("installed", s8.UNIT_NAME, version)]
    assert f"installed mv3dt-tracking.service version {version}" in capsys.readouterr().err
    assert s8.DATA_DIR.is_dir()
    assert stat.S_IMODE(s8.DATA_DIR.stat().st_mode) == s8.DATA_DIR_MODE


def test_rerun_reports_already_installed(tmp_path, env_values):
    ctx = Context(tmp_path, runner=Runner(tailscale_status=_logged_in()))
    step = s8.Step8TrackingHistory()
    step.run(ctx)
    ctx.reported.clear()
    ctx.runner.calls.clear()

    assert step.run(ctx).status is StepStatus.COMPLETE
    assert ctx.reported == [("already", s8.UNIT_NAME, s8.bundled_version(ctx))]
    assert ("systemctl", "restart", s8.UNIT_NAME) not in ctx.runner.calls


def test_run_without_tailscale_warns_with_manual_command(tmp_path, env_values, monkeypatch, capsys):
    monkeypatch.setattr(s8, "_WHICH", lambda name: None)
    ctx = Context(tmp_path)

    assert s8.Step8TrackingHistory().run(ctx).status is StepStatus.COMPLETE
    assert ctx.runner.named("tailscale") == []
    err = capsys.readouterr().err
    assert "tailscale is not installed" in err
    assert "sudo tailscale serve --bg --https=8443 http://127.0.0.1:8090" in err


def test_tailscale_serve_timeout_falls_back_to_manual_command(
    tmp_path, env_values, capsys
):
    runner = Runner(tailscale_status=_logged_in())

    def run(*argv, **kwargs):
        if argv[:2] == ("tailscale", "serve"):
            assert kwargs["timeout"] == s8.TAILSCALE_TIMEOUT_S
            raise s8.subprocess.TimeoutExpired(argv, kwargs["timeout"])
        return runner(*argv, **kwargs)

    ctx = Context(tmp_path, runner=run)

    assert s8.Step8TrackingHistory().run(ctx).status is StepStatus.COMPLETE
    err = capsys.readouterr().err
    assert "tailscale serve failed (timed out after 30s)" in err
    assert s8.TAILSCALE_SERVE_COMMAND in err


def test_run_never_logs_the_secret(tmp_path, env_values, capsys):
    ctx = Context(tmp_path, runner=Runner(tailscale_status=_logged_in()))
    step = s8.Step8TrackingHistory()
    step.run(ctx)
    secret = s8._read_env_file(ctx.install_dir / "secrets" / "tracking.env")[s8.KEY_SECRET]
    step.report(ctx)

    captured = capsys.readouterr()
    assert secret not in captured.err + captured.out
    assert not any(secret in arg for call in ctx.runner.calls for arg in call)
    assert "sudo cat" in captured.err
    assert "WORKSTATION_TRACKING_URL=https://ws.tailnet.ts.net:8443" in captured.err


# ---------------------------------------------------------------------------
# Verify (section 11.3)
# ---------------------------------------------------------------------------


def _installed(tmp_path, monkeypatch):
    monkeypatch.setenv(s8.KEY_PROJECT, "proj-1")
    monkeypatch.setenv(s8.KEY_ORIGINS, "https://app.example.org")
    ctx = Context(tmp_path, runner=Runner(tailscale_status=_logged_in()))
    s8.Step8TrackingHistory().run(ctx)
    return ctx


def test_verify_success_when_rows_grow(tmp_path, monkeypatch, capsys):
    ctx = _installed(tmp_path, monkeypatch)
    monkeypatch.setattr(s8, "_HTTP_GET", _health([
        {"connected": True, "messages": 10, "rows_flushed": 40},
        {"connected": True, "messages": 50, "rows_flushed": 200},
    ]))

    assert s8.Step8TrackingHistory().verify(ctx).status is StepStatus.COMPLETE
    assert "recorded 160 detections" in capsys.readouterr().err


def test_verify_warns_when_pipeline_not_running(tmp_path, monkeypatch, capsys):
    ctx = _installed(tmp_path, monkeypatch)
    idle = {"connected": True, "messages": 0, "rows_flushed": 0}
    monkeypatch.setattr(s8, "_HTTP_GET", _health([idle, idle]))

    assert s8.Step8TrackingHistory().verify(ctx).status is StepStatus.COMPLETE
    err = capsys.readouterr().err
    assert "pipeline is probably not running" in err
    assert "mv3dt/site-42/sv3d" in err


def test_verify_warns_when_broker_disconnected(tmp_path, monkeypatch, capsys):
    ctx = _installed(tmp_path, monkeypatch)
    down = {"connected": False, "messages": 0, "rows_flushed": 0}
    monkeypatch.setattr(s8, "_HTTP_GET", _health([down, down]))

    assert s8.Step8TrackingHistory().verify(ctx).status is StepStatus.COMPLETE
    assert "not connected to the local broker" in capsys.readouterr().err


def test_verify_fails_when_health_never_answers(tmp_path, monkeypatch):
    ctx = _installed(tmp_path, monkeypatch)
    monkeypatch.setattr(s8, "_HTTP_GET", _health([None]))

    result = s8.Step8TrackingHistory().verify(ctx)
    assert result.status is StepStatus.FAILED
    assert "/health" in result.message


def test_verify_requires_active_unit(tmp_path, monkeypatch):
    ctx = _installed(tmp_path, monkeypatch)
    ctx.runner.fail = {"is-active"}
    result = s8.Step8TrackingHistory().verify(ctx)
    assert result.status is StepStatus.USER_ACTION_REQUIRED
    assert "journalctl -u mv3dt-tracking" in result.user_actions[0].command


def test_verify_requires_secret_mode_600(tmp_path, monkeypatch):
    ctx = _installed(tmp_path, monkeypatch)
    (ctx.install_dir / "secrets" / "tracking.env").chmod(0o644)
    monkeypatch.setattr(s8, "_HTTP_GET", _health([{}]))
    result = s8.Step8TrackingHistory().verify(ctx)
    assert result.status is StepStatus.FAILED
    assert "600" in result.message


# ---------------------------------------------------------------------------
# `tracking` subcommand (section 10)
# ---------------------------------------------------------------------------


def _with_venv(tmp_path):
    ctx = Context(tmp_path)
    python = ctx.install_dir / "tracking" / "venv" / "bin" / "python3"
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")
    ctx.runner.active = False  # the unit stops cleanly unless a test says otherwise
    return ctx, python


def test_wipe_requires_yes_and_touches_nothing(tmp_path):
    ctx, _ = _with_venv(tmp_path)
    assert s8._handle_tracking(["wipe"], ctx) == 2
    assert ctx.runner.calls == []


def test_wipe_stops_runs_package_then_starts(tmp_path):
    ctx, python = _with_venv(tmp_path)

    rc = s8._handle_tracking(
        ["wipe", "--yes", "--before", "2026-10-01", "--non-interactive"], ctx
    )

    assert rc == 0
    stop, check, wipe, start = ctx.runner.calls
    assert stop == ("systemctl", "stop", s8.UNIT_NAME)
    assert check == ("systemctl", "is-active", "--quiet", s8.UNIT_NAME)
    assert start == ("systemctl", "start", s8.UNIT_NAME)
    assert wipe[0] == "env"
    assert f"DATA_DIR={s8.DATA_DIR}" in wipe
    assert wipe[wipe.index(str(python)):] == (
        str(python), "-m", "mv3dt_tracking", "wipe", "--yes", "--before", "2026-10-01",
    )
    assert not any(s8.KEY_SECRET in arg for arg in wipe)


def test_wipe_refuses_when_unit_stays_active(tmp_path, capsys):
    ctx, _ = _with_venv(tmp_path)
    ctx.runner.active = True

    assert s8._handle_tracking(["wipe", "--yes"], ctx) == 1
    assert ctx.runner.calls == [
        ("systemctl", "stop", s8.UNIT_NAME),
        ("systemctl", "is-active", "--quiet", s8.UNIT_NAME),
    ]
    assert "refusing to wipe" in capsys.readouterr().err


def test_wipe_runs_when_unit_does_not_exist(tmp_path):
    ctx, python = _with_venv(tmp_path)
    ctx.runner.fail = {"stop", "start"}

    s8._handle_tracking(["wipe", "--yes"], ctx)

    assert any(str(python) in call and "wipe" in call for call in ctx.runner.calls)


def test_wipe_restarts_unit_even_when_wipe_fails(tmp_path):
    ctx, _ = _with_venv(tmp_path)
    ctx.runner.fail = {"wipe"}
    assert s8._handle_tracking(["wipe", "--yes"], ctx) == 1
    assert ctx.runner.calls[-1] == ("systemctl", "start", s8.UNIT_NAME)


def test_wipe_rejects_bad_date(tmp_path):
    ctx, _ = _with_venv(tmp_path)
    with pytest.raises(SystemExit):
        s8._handle_tracking(["wipe", "--yes", "--before", "yesterday"], ctx)
    assert ctx.runner.calls == []


def test_stats_runs_package_as_user(tmp_path):
    ctx, python = _with_venv(tmp_path)
    assert s8._handle_tracking(["stats"], ctx) == 0
    (call,) = ctx.runner.calls
    assert call[-3:] == ("-m", "mv3dt_tracking", "stats")
    assert str(python) in call


def test_subcommand_refuses_when_not_installed(tmp_path):
    ctx = Context(tmp_path)
    assert s8._handle_tracking(["stats"], ctx) == 1
    assert ctx.runner.calls == []


# ---------------------------------------------------------------------------
# cleaning.json seed (TRACKING-FUSION-AND-CLEANING.md sections 4.1 and 10)
# ---------------------------------------------------------------------------

STRICT_SEED = {"version": 1, "preset": "strict", "zones": []}


def _synced(tmp_path):
    ctx = Context(tmp_path)
    s8.sync_package(ctx)
    return ctx, ctx.install_dir / "tracking" / "cleaning.json"


def test_seed_comes_from_the_package_defaults(tmp_path):
    ctx, path = _synced(tmp_path)
    seed, error = s8._strict_seed(ctx)
    assert error is None and seed == STRICT_SEED
    (call,) = ctx.runner.calls
    assert call[:3] == (str(s8._venv_python(ctx)), "-I", "-c")
    assert call[-1] == str(ctx.install_dir / "tracking")
    # The installer restates none of the values; they come from cleaning.py.
    assert "'version': STRICT['version']" in s8._SEED_PROGRAM
    assert "list(STRICT['zones'])" in s8._SEED_PROGRAM


def test_seed_writes_once_with_mode_644_owned_by_the_service_user(tmp_path, monkeypatch):
    ctx, path = _synced(tmp_path)
    owners = []
    monkeypatch.setattr(s8.os, "fchown", lambda fd, uid, gid: owners.append((uid, gid)))
    old_umask = os.umask(0o077)
    try:
        assert s8.seed_cleaning_file(ctx) == (True, None)
    finally:
        os.umask(old_umask)

    assert json.loads(path.read_text(encoding="utf-8")) == STRICT_SEED
    assert stat.S_IMODE(path.stat().st_mode) == 0o644
    assert owners == [(ctx.user.uid, ctx.user.gid)]
    assert [p.name for p in path.parent.iterdir() if p.name.endswith(".tmp")] == []

    ctx.runner.calls.clear()
    assert s8.seed_cleaning_file(ctx) == (False, None)
    assert ctx.runner.calls == []


def test_seed_never_overwrites_an_existing_file_or_symlink(tmp_path):
    ctx, path = _synced(tmp_path)
    path.write_text('{"preset": "relaxed"}\n', encoding="utf-8")
    path.chmod(0o600)
    ctx.runner.calls.clear()

    assert s8.seed_cleaning_file(ctx) == (False, None)
    assert path.read_text(encoding="utf-8") == '{"preset": "relaxed"}\n'
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert ctx.runner.calls == []

    path.unlink()
    path.symlink_to(tmp_path / "elsewhere.json")  # dangling
    assert s8.seed_cleaning_file(ctx) == (False, None)
    assert path.is_symlink() and not (tmp_path / "elsewhere.json").exists()


def test_seed_failure_is_a_warning(tmp_path, env_values, capsys):
    ctx = Context(tmp_path)
    ctx.runner.fail = {s8._SEED_PROGRAM}

    assert s8.Step8TrackingHistory().run(ctx).status is StepStatus.COMPLETE
    assert not (ctx.install_dir / "tracking" / "cleaning.json").exists()
    assert "cleaning.json was not seeded" in capsys.readouterr().err


def test_run_seeds_cleaning_but_never_offsets(tmp_path, env_values):
    ctx = Context(tmp_path, runner=Runner(tailscale_status=_logged_in()))
    assert s8.Step8TrackingHistory().run(ctx).status is StepStatus.COMPLETE
    tracking = ctx.install_dir / "tracking"
    assert json.loads((tracking / "cleaning.json").read_text(encoding="utf-8")) == STRICT_SEED
    assert not (tracking / "offsets.json").exists()


def test_upgrade_rerun_restarts_and_keeps_both_files(tmp_path, env_values):
    ctx = Context(tmp_path, runner=Runner(tailscale_status=_logged_in()))
    step = s8.Step8TrackingHistory()
    step.run(ctx)
    tracking = ctx.install_dir / "tracking"
    cleaning = tracking / "cleaning.json"
    offsets = tracking / "offsets.json"
    cleaning.write_text('{"preset": "relaxed", "zones": []}\n', encoding="utf-8")
    offsets.write_text('{"version": 1}\n', encoding="utf-8")
    # An older installed package, as after an installer upgrade.
    (tracking / "mv3dt_tracking" / "cleaning.py").write_text("# old\n", encoding="utf-8")
    unit = systemd.UNIT_DIR / s8.UNIT_NAME
    unit.write_text("[Unit]\nDescription=old\n", encoding="utf-8")
    ctx.runner.calls.clear()
    ctx.reported.clear()

    assert step.run(ctx).status is StepStatus.COMPLETE
    assert ("systemctl", "restart", s8.UNIT_NAME) in ctx.runner.calls
    assert ctx.reported == [("installed", s8.UNIT_NAME, s8.bundled_version(ctx))]
    assert unit.read_text(encoding="utf-8") == s8.render_tracking_unit(ctx)
    bundled = shellout.asset_path("tracking", "mv3dt_tracking", "cleaning.py")
    assert (tracking / "mv3dt_tracking" / "cleaning.py").read_bytes() == bundled.read_bytes()
    assert cleaning.read_text(encoding="utf-8") == '{"preset": "relaxed", "zones": []}\n'
    assert offsets.read_text(encoding="utf-8") == '{"version": 1}\n'

    # A second re-run with nothing new is a no-op.
    ctx.runner.calls.clear()
    ctx.reported.clear()
    assert step.run(ctx).status is StepStatus.COMPLETE
    assert ("systemctl", "restart", s8.UNIT_NAME) not in ctx.runner.calls
    assert ctx.reported == [("already", s8.UNIT_NAME, s8.bundled_version(ctx))]
    assert cleaning.read_text(encoding="utf-8") == '{"preset": "relaxed", "zones": []}\n'


# ---------------------------------------------------------------------------
# Cleaning command passthrough (TRACKING-FUSION-AND-CLEANING.md section 9)
# ---------------------------------------------------------------------------


def _env_pairs(call):
    """The KEY=VALUE arguments of an `env ... python -m pkg` call."""
    assert call[0] == "env"
    end = next(i for i, a in enumerate(call) if "=" not in a and i > 0)
    return dict(a.split("=", 1) for a in call[1:end]), call[end:]


@pytest.mark.parametrize(
    "argv",
    [
        ["zones-suggest", "--hours", "12", "--json"],
        ["learn-offsets", "--from", "2026-10-01T08:00", "--model", "similarity", "--save"],
        ["clean-report", "--from", "1700000000000", "--to", "1700003600000"],
        ["learn-offsets", "--help"],
    ],
)
def test_cleaning_commands_forward_args_with_the_service_environment(tmp_path, argv):
    ctx, python = _with_venv(tmp_path)
    as_user = []
    ctx.run_as_user = lambda *a, **kw: as_user.append(a) or ctx.runner(*a, **kw)

    assert s8._handle_tracking([*argv, "--non-interactive"], ctx) == 0

    (call,) = ctx.runner.calls
    assert as_user == [call]  # as the service user, never as root
    env, command = _env_pairs(call)
    assert command == (str(python), "-m", "mv3dt_tracking", *argv)
    assert env == s8.service_environment(ctx)
    assert env == _unit_environment(s8.render_tracking_unit(ctx))
    assert ctx.runner.kwargs[0]["cwd"] == str(ctx.install_dir / "tracking")
    assert not any(s8.KEY_SECRET in arg for arg in call)


def test_stats_and_wipe_use_the_service_environment_too(tmp_path):
    ctx, _ = _with_venv(tmp_path)
    s8._handle_tracking(["stats"], ctx)
    s8._handle_tracking(["wipe", "--yes"], ctx)
    package_calls = [c for c in ctx.runner.calls if c[0] == "env"]
    assert len(package_calls) == 2
    for call in package_calls:
        assert _env_pairs(call)[0] == s8.service_environment(ctx)


def test_cleaning_command_exit_code_is_forwarded(tmp_path):
    ctx, _ = _with_venv(tmp_path)
    ctx.runner.exit_code = 1  # e.g. learn-offsets --save with too few pairs
    assert s8._handle_tracking(["learn-offsets", "--save"], ctx) == 1


def test_cleaning_commands_refuse_when_gated_off(tmp_path, capsys):
    ctx, _ = _with_venv(tmp_path)
    ctx.conf = {GATE: "off", "LOCATION_ID": "site-42"}
    assert s8._handle_tracking(["zones-suggest"], ctx) == 1
    assert ctx.runner.calls == []
    assert "tracking history is off" in capsys.readouterr().err


def test_cleaning_commands_refuse_when_not_installed(tmp_path, capsys):
    ctx = Context(tmp_path)
    assert s8._handle_tracking(["clean-report", "--from", "0", "--to", "1"], ctx) == 1
    assert ctx.runner.calls == []
    assert "not installed" in capsys.readouterr().err


def test_tracking_help_lists_the_cleaning_commands(capsys):
    with pytest.raises(SystemExit):
        s8._handle_tracking(["--help"], SimpleNamespace())
    out = capsys.readouterr().out
    for name in s8.CLEANING_COMMANDS:
        assert name in out


# ---------------------------------------------------------------------------
# Upgrade without a step reset (section 11.5)
# ---------------------------------------------------------------------------


def _dispatch_ctx(tmp_path, monkeypatch, *, gate="on", non_interactive=True):
    """A completed Step 8 install driven through the real dispatch loop."""
    monkeypatch.setenv(s8.KEY_PROJECT, "proj-1")
    monkeypatch.setenv(s8.KEY_ORIGINS, "https://app.example.org")
    monkeypatch.setattr(s8, "_HTTP_GET", _health([
        {"connected": True, "messages": 1, "rows_flushed": 1},
    ]))
    ctx = Context(tmp_path, runner=Runner(tailscale_status=_logged_in()))
    ctx.progress = SimpleNamespace(
        phase=lambda n: None, task=lambda n: None,
        begin_step=lambda step, position: None, end_step=lambda: None,
    )
    step = s8.Step8TrackingHistory()
    assert step.run(ctx).status is StepStatus.COMPLETE
    ctx.conf[GATE] = gate
    cfg = config_mod.Config(
        install_dir=ctx.install_dir, tracking_history=gate, values=ctx.conf
    )
    sm = StateMachine(path=tmp_path / "state.json")
    sm.mark_complete(s8.STEP_ID)
    monkeypatch.setattr(app, "STEP_REGISTRY", [step])
    ctx.non_interactive = non_interactive
    ctx.runner.calls.clear()
    ctx.reported.clear()
    return ctx, cfg, sm


def _operator_files(ctx):
    tracking = ctx.install_dir / "tracking"
    files = {
        tracking / "cleaning.json": '{"preset": "relaxed", "zones": []}\n',
        tracking / "offsets.json": '{"version": 1}\n',
        s8.DATA_DIR / "2026-01-01.sqlite": "rows\n",
    }
    for path, text in files.items():
        path.write_text(text, encoding="utf-8")
    env = s8._tracking_env_path(ctx)
    return files, env, env.read_bytes()


def test_upgrade_with_a_changed_bundle_resyncs_and_restarts(tmp_path, monkeypatch, capsys):
    ctx, cfg, sm = _dispatch_ctx(tmp_path, monkeypatch)
    files, env, env_bytes = _operator_files(ctx)
    installed = ctx.install_dir / "tracking" / "mv3dt_tracking" / "cleaning.py"
    installed.write_text("# previous release\n", encoding="utf-8")
    capsys.readouterr()

    assert app._dispatch(sm, ctx, cfg) == 0

    err = capsys.readouterr().err
    assert "step8_tracking_history: bundled tracking package changed; updating" in err
    assert "already complete" not in err
    bundled = shellout.asset_path("tracking", "mv3dt_tracking", "cleaning.py")
    assert installed.read_bytes() == bundled.read_bytes()
    assert ("systemctl", "restart", s8.UNIT_NAME) in ctx.runner.calls
    assert sm.status(s8.STEP_ID) is StepStatus.COMPLETE
    for path, text in files.items():
        assert path.read_text(encoding="utf-8") == text
    assert env.read_bytes() == env_bytes
    assert not ctx.runner.named("python3", "-m")  # venv kept


def test_upgrade_with_changed_requirements_rebuilds_pip_and_restarts(
    tmp_path, monkeypatch, capsys
):
    ctx, cfg, sm = _dispatch_ctx(tmp_path, monkeypatch)
    # What the previous release installed: its requirements and the venv
    # marker pip left for them.
    (ctx.install_dir / "tracking" / "requirements.txt").write_text("old==1\n")
    marker = ctx.install_dir / "tracking" / "venv" / s8.REQUIREMENTS_MARKER
    marker.write_text("0" * 64 + "\n", encoding="utf-8")
    capsys.readouterr()

    assert app._dispatch(sm, ctx, cfg) == 0

    assert "bundled tracking requirements changed; updating" in capsys.readouterr().err
    assert any("pip" in call for call in ctx.runner.calls)
    assert ("systemctl", "restart", s8.UNIT_NAME) in ctx.runner.calls


def test_upgrade_with_a_stale_unit_rerenders_it(tmp_path, monkeypatch, capsys):
    ctx, cfg, sm = _dispatch_ctx(tmp_path, monkeypatch)
    unit = systemd.UNIT_DIR / s8.UNIT_NAME
    unit.write_text("[Unit]\nDescription=old\n", encoding="utf-8")
    capsys.readouterr()

    assert app._dispatch(sm, ctx, cfg) == 0

    assert f"{s8.UNIT_NAME} differs from this release; updating" in capsys.readouterr().err
    assert unit.read_text(encoding="utf-8") == s8.render_tracking_unit(ctx)
    assert ("systemctl", "restart", s8.UNIT_NAME) in ctx.runner.calls


def test_upgrade_with_an_identical_bundle_runs_nothing(tmp_path, monkeypatch, capsys):
    ctx, cfg, sm = _dispatch_ctx(tmp_path, monkeypatch)
    files, env, env_bytes = _operator_files(ctx)
    capsys.readouterr()

    assert app._dispatch(sm, ctx, cfg) == 0

    assert "step8_tracking_history: already complete" in capsys.readouterr().err
    assert ctx.runner.calls == []
    assert ctx.reported == []
    for path, text in files.items():
        assert path.read_text(encoding="utf-8") == text
    assert env.read_bytes() == env_bytes


def test_upgrade_with_the_gate_off_stays_skipped(tmp_path, monkeypatch, capsys):
    ctx, cfg, sm = _dispatch_ctx(tmp_path, monkeypatch, gate="off")
    installed = ctx.install_dir / "tracking" / "mv3dt_tracking" / "cleaning.py"
    installed.write_text("# previous release\n", encoding="utf-8")
    capsys.readouterr()

    assert app._dispatch(sm, ctx, cfg) == 0

    assert "step8_tracking_history: skipped (gate off)" in capsys.readouterr().err
    assert installed.read_text(encoding="utf-8") == "# previous release\n"
    assert ctx.runner.calls == []
    assert s8.Step8TrackingHistory().pending_update(ctx) is None


def test_update_reason_ignores_bytecode_order_and_mtimes(tmp_path, monkeypatch):
    ctx, _cfg, _sm = _dispatch_ctx(tmp_path, monkeypatch)
    assert s8.update_reason(ctx) is None
    dest = ctx.install_dir / "tracking" / "mv3dt_tracking"
    cache = dest / "__pycache__"
    cache.mkdir(exist_ok=True)
    (cache / "cli.cpython-312.pyc").write_bytes(b"\x00compiled")
    (dest / "stray.pyc").write_bytes(b"\x00compiled")
    for path in dest.rglob("*.py"):
        os.utime(path, (1_000_000, 1_000_000))
    assert s8.update_reason(ctx) is None

    first = s8._tree_digest(dest)
    # Recreate the tree in reverse order: same content, new inodes.
    copy = tmp_path / "copy"
    for rel in sorted(first, reverse=True):
        target = copy / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((dest / rel).read_bytes())
    assert s8._tree_digest(copy) == first

    (dest / "cli.py").write_text("# edited\n", encoding="utf-8")
    assert s8.update_reason(ctx) == "bundled tracking package changed"


def test_status_shows_a_pending_tracking_update(tmp_path, monkeypatch, capsys):
    ctx, _cfg, sm = _dispatch_ctx(tmp_path, monkeypatch)
    (ctx.install_dir / config_mod.CONF_FILENAME).write_text(
        f"{GATE}=on\nLOCATION_ID=site-42\n", encoding="utf-8"
    )
    state = sm.load()
    state.install_dir = str(ctx.install_dir)
    sm.save(state)

    assert app.main(["--status"], state_path=sm.path) == 0
    assert f"{s8.STEP_ID}: COMPLETE\n" in capsys.readouterr().out

    (ctx.install_dir / "tracking" / "mv3dt_tracking" / "cli.py").write_text("# old\n")
    assert app.main(["--status"], state_path=sm.path) == 0
    out = capsys.readouterr().out
    assert (
        f"{s8.STEP_ID}: COMPLETE, update pending (bundled tracking package "
        "changed): runs on next install run"
    ) in out


def _age_the_package(ctx):
    installed = ctx.install_dir / "tracking" / "mv3dt_tracking" / "cleaning.py"
    installed.write_text("# previous release\n", encoding="utf-8")
    return installed


def _restarts(ctx):
    return ctx.runner.calls.count(("systemctl", "restart", s8.UNIT_NAME))


def test_interactive_update_never_prompts_for_an_absent_optional_key(
    tmp_path, monkeypatch, capsys
):
    # Installed without TICKET_ISSUER; the autouse _PROMPT raises if asked.
    ctx, cfg, sm = _dispatch_ctx(tmp_path, monkeypatch, non_interactive=False)
    env = s8._tracking_env_path(ctx)
    assert s8.KEY_ISSUER not in s8._read_env_file(env)
    _age_the_package(ctx)

    assert app._dispatch(sm, ctx, cfg) == 0

    assert _restarts(ctx) == 1
    assert sm.status(s8.STEP_ID) is StepStatus.COMPLETE
    assert sm.update_pending(s8.STEP_ID) is None
    assert s8.KEY_ISSUER not in s8._read_env_file(env)


def test_interactive_install_records_a_blank_issuer_and_never_asks_again(
    tmp_path, monkeypatch
):
    answers = iter(["proj-3", "https://b.example.org", ""])
    monkeypatch.setattr(s8, "_PROMPT", lambda msg: next(answers))
    ctx = Context(tmp_path, non_interactive=False)
    s8.ensure_tracking_env(ctx)
    path = s8._tracking_env_path(ctx)
    assert f"{s8.KEY_ISSUER}=\n" in path.read_text(encoding="utf-8")

    monkeypatch.setattr(s8, "_PROMPT", _no_prompt)
    assert s8.ensure_tracking_env(ctx) == (False, [])


def test_update_missing_a_required_key_stops_without_prompting_then_restarts(
    tmp_path, monkeypatch
):
    ctx, cfg, sm = _dispatch_ctx(tmp_path, monkeypatch, non_interactive=False)
    env = s8._tracking_env_path(ctx)
    secret = s8._read_env_file(env)[s8.KEY_SECRET]
    # A release that needs a key this install never had.
    env.write_text(
        f"{s8.KEY_SECRET}={secret}\n{s8.KEY_ORIGINS}=https://app.example.org\n",
        encoding="utf-8",
    )
    env.chmod(0o600)
    monkeypatch.delenv(s8.KEY_PROJECT)
    installed = _age_the_package(ctx)

    assert app._dispatch(sm, ctx, cfg) == 0  # USER-ACTION, not a prompt

    bundled = shellout.asset_path("tracking", "mv3dt_tracking", "cleaning.py")
    assert installed.read_bytes() == bundled.read_bytes()  # synced already
    assert _restarts(ctx) == 0
    assert sm.status(s8.STEP_ID) is StepStatus.PENDING
    assert sm.update_pending(s8.STEP_ID) == "bundled tracking package changed"

    # The operator adds the key; the files already match, yet the service
    # still runs the previous release, so the retry restarts it.
    with env.open("a", encoding="utf-8") as f:
        f.write(f"{s8.KEY_PROJECT}=proj-1\n")
    ctx.runner.calls.clear()
    assert app._dispatch(sm, ctx, cfg) == 0

    assert _restarts(ctx) == 1
    assert sm.status(s8.STEP_ID) is StepStatus.COMPLETE
    assert sm.update_pending(s8.STEP_ID) is None

    # Completed: the next run is a no-op again.
    ctx.runner.calls.clear()
    assert app._dispatch(sm, ctx, cfg) == 0
    assert ctx.runner.calls == []


def test_update_interrupted_before_the_unit_restarts_on_the_next_run(
    tmp_path, monkeypatch, capsys
):
    ctx, cfg, sm = _dispatch_ctx(tmp_path, monkeypatch)
    _age_the_package(ctx)
    real = s8.ensure_tracking_env

    def interrupted(ctx, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(s8, "ensure_tracking_env", interrupted)
    with pytest.raises(KeyboardInterrupt):
        app._dispatch(sm, ctx, cfg)
    assert _restarts(ctx) == 0
    assert sm.update_pending(s8.STEP_ID) == "bundled tracking package changed"
    assert s8.update_reason(ctx) is None  # the files already match

    monkeypatch.setattr(s8, "ensure_tracking_env", real)
    ctx.runner.calls.clear()
    capsys.readouterr()
    assert app._dispatch(sm, ctx, cfg) == 0

    assert "resuming an update that did not finish" in capsys.readouterr().err
    assert _restarts(ctx) == 1
    assert ctx.reported == [("installed", s8.UNIT_NAME, s8.bundled_version(ctx))]
    assert sm.status(s8.STEP_ID) is StepStatus.COMPLETE
    assert sm.update_pending(s8.STEP_ID) is None


def test_status_shows_an_update_that_did_not_finish(tmp_path, capsys):
    sm = StateMachine(path=tmp_path / "state.json")
    sm.mark_complete(s8.STEP_ID)
    sm.mark_update_pending(s8.STEP_ID, "bundled tracking package changed")

    assert app.main(["--status"], state_path=sm.path) == 0
    assert (
        f"{s8.STEP_ID}: PENDING, update did not finish (bundled tracking "
        "package changed): runs on next install run"
    ) in capsys.readouterr().out
