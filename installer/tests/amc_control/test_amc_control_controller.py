"""The controller runs exactly the commands Step 9's sudoers rule allows.

Run from installer/: `python3 -m pytest tests/amc_control/test_amc_control_controller.py -v`

No process is started: `subprocess.Popen` is replaced by a recorder.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

_INSTALLER = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_INSTALLER))
sys.path.insert(0, str(_INSTALLER / "mv3dt_installer" / "assets" / "amc_control"))

from mv3dt_amc_control import controller as controller_mod  # noqa: E402
from mv3dt_installer.steps import step9_amc_control as s9  # noqa: E402

TS_IP = "100.64.0.7"
USER = "operator"


class _Proc:
    def poll(self):
        return None


class _Ctx:
    def __init__(self, install_dir):
        self.install_dir = install_dir
        from types import SimpleNamespace

        self.user = SimpleNamespace(name=USER)


@pytest.fixture
def launched(monkeypatch, tmp_path):
    install_dir = tmp_path / "mv3dt"
    monkeypatch.setenv("MV3DT_INSTALL_DIR", str(install_dir))
    monkeypatch.setenv("AMC_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("AMC_PUBLIC_HOST", TS_IP)
    monkeypatch.delenv("AMC_BIN", raising=False)
    monkeypatch.delenv("AMC_USE_SUDO", raising=False)
    calls = []

    def fake_popen(cmd, **kwargs):
        calls.append(list(cmd))
        return _Proc()

    monkeypatch.setattr(controller_mod.subprocess, "Popen", fake_popen)
    ctl = controller_mod.AmcController()
    monkeypatch.setattr(ctl, "ready", lambda: False)
    return ctl, calls, install_dir


def _allowed_commands(install_dir):
    """The command lines in the sudoers rule Step 9 renders."""
    rule = s9.render_sudoers(_Ctx(install_dir), TS_IP)
    line = next(l for l in rule.splitlines() if l and not l.startswith("#"))
    _, _, commands = line.partition("NOPASSWD: ")
    return [c.strip().split(" ") for c in commands.split(",")]


def test_start_runs_the_allowed_start_command_under_sudo(launched):
    ctl, calls, install_dir = launched
    ctl.start()
    start, _stop, _record = _allowed_commands(install_dir)
    assert calls == [["sudo", "-n", *start]]
    assert start[:2] == [str(install_dir / "bin" / "mv3dt-installer"), "amc"]


def test_stop_runs_the_allowed_stop_command_under_sudo(launched):
    ctl, calls, install_dir = launched
    ctl.stop()
    _start, stop, _record = _allowed_commands(install_dir)
    assert calls == [["sudo", "-n", *stop]]
    assert stop == [str(install_dir / "bin" / "mv3dt-installer"), "amc", "--down"]


def test_the_user_owned_wrapper_is_never_run(launched):
    ctl, calls, install_dir = launched
    ctl.start()
    ctl.stop()
    assert all(str(install_dir / "bin" / "amc") not in call for call in calls)


def test_amc_bin_override_runs_the_program_with_the_amc_arguments(launched, monkeypatch):
    ctl, calls, _ = launched
    monkeypatch.setenv("AMC_BIN", "/fake/amc")
    monkeypatch.setenv("AMC_USE_SUDO", "0")
    ctl.start()
    assert calls == [["/fake/amc", "--keep-up", "--no-open", "--non-interactive", "--host-ip", TS_IP]]
