"""Tests for mv3dt_installer.steps.step6_remote_supervision
(STEP-6-REMOTE-SUPERVISION.md).

Run from installer/: `python3 -m pytest tests/test_step6_remote_supervision.py -v`

No test here touches a real systemctl/mosquitto/polkit or opens a real MQTT
connection -- every systemctl-shaped call goes through an injected `runner`
callable (mirrors `test_step5_per_project_exes.py`'s `ScriptedRunner`
convention), and every timing-sensitive function (`_settle`, `RequestDedup`)
takes an injectable `clock`/`sleep`.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
from datetime import datetime, timezone

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from mv3dt_installer import app  # noqa: E402
from mv3dt_installer import config as config_mod  # noqa: E402
from mv3dt_installer.steps import STEP_REGISTRY  # noqa: E402
from mv3dt_installer.steps import step5_per_project_exes as step5  # noqa: E402
from mv3dt_installer.steps import step6_remote_supervision as step6  # noqa: E402


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class ScriptedRunner:
    """Stand-in for `ctx.run_root`/`ctx.run_as_user` and the module's own
    `runner`/`run_as_agent_user` callables (mirrors
    `test_step5_per_project_exes.py`'s identical fake). Called with a single
    argv **list** here (this module's own convention, matching
    `systemd.py`'s `Runner` shape), not the varargs `Context.run_root` uses.
    """

    def __init__(self, *, default_returncode: int = 0, default_stdout: str = "", default_stderr: str = ""):
        self.calls: list[list[str]] = []
        self._rules: list[tuple] = []
        self.default_returncode = default_returncode
        self.default_stdout = default_stdout
        self.default_stderr = default_stderr

    def when(self, matcher, *, returncode=0, stdout="", stderr=""):
        self._rules.append((matcher, returncode, stdout, stderr))
        return self

    def __call__(self, argv, **kwargs):
        self.calls.append(list(argv))
        for matcher, returncode, stdout, stderr in reversed(self._rules):
            if matcher(argv):
                return subprocess.CompletedProcess(list(argv), returncode, stdout, stderr)
        return subprocess.CompletedProcess(
            list(argv), self.default_returncode, self.default_stdout, self.default_stderr
        )


def _prefix(*prefix):
    return lambda argv: tuple(argv[: len(prefix)]) == prefix


def _make_entry(install_dir, *, project_name="North Lobby #2", slug="north-lobby-2"):
    exe = install_dir / "bin" / f"pipeline-{slug}"
    exe.parent.mkdir(parents=True, exist_ok=True)
    rendered = install_dir / "deepstream" / "deepstream_app_config.rendered.txt"
    rendered.parent.mkdir(parents=True, exist_ok=True)
    calib = install_dir / "deepstream" / "calibration" / "loc-1"
    calib.mkdir(parents=True, exist_ok=True)
    return step5.upsert(
        install_dir,
        project_name=project_name,
        location_id="loc-1",
        rendered_config=str(rendered),
        calibration_dir=str(calib),
        exe=str(exe),
        slug=slug,
    )


@pytest.fixture(autouse=True)
def _reset_removal_hook():
    # Importing step6_remote_supervision as a module-level side effect (any
    # earlier test file importing it) installs the real hook; make sure each
    # test starts from a known state and restores it afterward.
    previous = step5._REMOVAL_HOOK
    yield
    step5.register_removal_hook(previous)


# ---------------------------------------------------------------------------
# Module identity + subcommand registration
# ---------------------------------------------------------------------------


def test_registers_itself_with_the_expected_identity():
    matches = [s for s in STEP_REGISTRY if s.id == "step6_remote_supervision"]
    assert len(matches) == 1
    step = matches[0]
    assert step.order == 6
    assert "Remote supervision" in step.title


def test_registers_the_agent_subcommand():
    registration = app.SUBCOMMAND_REGISTRY.get("agent")
    assert registration is not None
    assert registration.handler is step6.handle_agent_subcommand


def test_agent_subcommand_does_not_require_root():
    """unit U6's fix for a PR #50 review defect: `mv3dt-agent.service`
    (this module's own docstring, section B.1) deliberately runs as
    `User=@USER@`, so `agent` must be registered with
    `requires_root=False` -- otherwise `app._bootstrap_subcommand_context()`
    would call `privilege.require_root()` and exit before this module's
    `handle_agent_subcommand` ever ran."""
    registration = app.SUBCOMMAND_REGISTRY.get("agent")
    assert registration is not None
    assert registration.requires_root is False


def test_registers_the_removal_hook_on_import():
    assert step5._REMOVAL_HOOK is step6._disable_pipeline_unit_before_remove


# ---------------------------------------------------------------------------
# section C.1 -- HOST_ID derivation priority
# ---------------------------------------------------------------------------


def test_host_id_prefers_conf_override(tmp_path):
    machine_id = tmp_path / "machine-id"
    machine_id.write_text("abc123\n")
    host_id = step6.resolve_host_id(
        {"MV3DT_HOST_ID": "desk-lab-01"},
        machine_id_path=machine_id,
        hostname=lambda: "should-not-be-used",
    )
    assert host_id == "desk-lab-01"


def test_host_id_falls_back_to_machine_id(tmp_path):
    machine_id = tmp_path / "machine-id"
    machine_id.write_text("abcdef0123456789\n")
    host_id = step6.resolve_host_id(
        {}, machine_id_path=machine_id, hostname=lambda: "should-not-be-used"
    )
    assert host_id == "abcdef0123456789"


def test_host_id_falls_back_to_sanitized_hostname_when_machine_id_missing(tmp_path):
    missing = tmp_path / "no-such-file"
    host_id = step6.resolve_host_id(
        {}, machine_id_path=missing, hostname=lambda: "Desk Lab #1"
    )
    assert host_id == "desk-lab-1"


def test_host_id_falls_back_to_hostname_when_machine_id_empty(tmp_path):
    empty = tmp_path / "machine-id"
    empty.write_text("   \n")
    host_id = step6.resolve_host_id({}, machine_id_path=empty, hostname=lambda: "lab01")
    assert host_id == "lab01"


# ---------------------------------------------------------------------------
# section C.2 -- topic layout
# ---------------------------------------------------------------------------


def test_topic_layout():
    assert step6.cmd_topic("desk-lab-01") == "mv3dt/desk-lab-01/cmd"
    assert step6.result_topic("desk-lab-01") == "mv3dt/desk-lab-01/cmd/result"
    assert step6.status_topic("desk-lab-01") == "mv3dt/desk-lab-01/status"


# ---------------------------------------------------------------------------
# section B.1.1 -- polkit rule content / anchoring
# ---------------------------------------------------------------------------


def test_polkit_rule_substitutes_the_invoking_user():
    rule = step6.render_polkit_rule("alice")
    assert 'subject.user !== "alice"' in rule
    assert "@USER@" not in rule


def test_polkit_rule_has_the_anchored_unit_regex():
    rule = step6.render_polkit_rule("alice")
    assert r"/^mv3dt-pipeline@[a-z0-9-]{1,64}\.service$/" in rule


def test_polkit_rule_verb_allowlist_excludes_enable_disable_mask():
    rule = step6.render_polkit_rule("alice")
    assert '"start"' in rule
    assert '"stop"' in rule
    assert '"restart"' in rule
    assert '"enable"' not in rule
    assert '"disable"' not in rule
    assert '"mask"' not in rule


def test_polkit_rule_uses_result_yes_not_auth_admin():
    rule = step6.render_polkit_rule("alice")
    assert "polkit.Result.YES" in rule
    # AUTH_ADMIN may appear in the file's explanatory comments, but the
    # executable rule body must never return it.
    body = "\n".join(
        line for line in rule.splitlines() if not line.strip().startswith("//")
    )
    assert "AUTH_ADMIN" not in body


def test_polkit_rule_defaults_to_not_handled():
    rule = step6.render_polkit_rule("alice")
    assert rule.count("polkit.Result.NOT_HANDLED") >= 2


def test_pipeline_unit_runs_from_the_directory_step4_renders_into():
    """systemd exits 200/CHDIR before ExecStart if WorkingDirectory is
    missing; Step 4 renders a flat config set, never a per-slug folder."""
    unit = step6.render_pipeline_unit("alice")
    working_dirs = [l for l in unit.splitlines() if l.startswith("WorkingDirectory=")]
    assert working_dirs == [
        "WorkingDirectory=/opt/mv3dt/deepstream"
    ]
    assert "%i" not in working_dirs[0]


def test_pipeline_unit_has_no_root_only_exec_stop():
    """`pipeline --stop` requires root and the unit runs as User=@USER@."""
    unit = step6.render_pipeline_unit("alice")
    assert "User=alice" in unit
    assert not any(l.startswith("ExecStop=") for l in unit.splitlines())


def test_polkit_rule_checks_the_manage_units_action_id():
    rule = step6.render_polkit_rule("alice")
    assert '"org.freedesktop.systemd1.manage-units"' in rule


# ---------------------------------------------------------------------------
# section E.1 -- polkit positive/negative authorization check
# ---------------------------------------------------------------------------


def test_check_polkit_authorization_all_pass_when_scoped_correctly():
    runner = ScriptedRunner()
    runner.when(
        _prefix("systemctl", "--no-ask-password", "start", "mv3dt-pipeline@north-lobby-2.service"),
        returncode=0,
    )
    runner.when(_prefix("systemctl", "--no-ask-password", "start", "mosquitto"), returncode=1)
    runner.when(
        _prefix("systemctl", "--no-ask-password", "enable", "mv3dt-pipeline@north-lobby-2.service"),
        returncode=1,
    )

    result = step6.check_polkit_authorization(run_as_agent_user=runner, slug="north-lobby-2")
    assert result == {
        "start_pipeline_permitted": True,
        "start_mosquitto_denied": True,
        "enable_pipeline_denied": True,
    }


def test_check_polkit_authorization_never_prompts_for_a_password():
    """An admin answering a polkit TTY prompt would make both negative
    probes pass; every probe must run non-interactively."""
    runner = ScriptedRunner()
    step6.check_polkit_authorization(run_as_agent_user=runner, slug="north-lobby-2")
    assert len(runner.calls) == 3
    assert all(list(argv[:2]) == ["systemctl", "--no-ask-password"] for argv in runner.calls)


def test_check_polkit_authorization_catches_an_overly_broad_rule():
    """A rule that (wrongly) grants everything must fail this check -- the
    exact bug the positive+negative design (section B.1.1) exists to catch."""
    runner = ScriptedRunner(default_returncode=0)  # everything "succeeds"
    result = step6.check_polkit_authorization(run_as_agent_user=runner, slug="north-lobby-2")
    assert result["start_pipeline_permitted"] is True
    assert result["start_mosquitto_denied"] is False
    assert result["enable_pipeline_denied"] is False
    assert not all(result.values())


# ---------------------------------------------------------------------------
# section C.6 -- settle-and-poll state machine
# ---------------------------------------------------------------------------


def _fake_clock_sleep():
    state = {"t": 0.0}

    def clock():
        return state["t"]

    def sleep(seconds):
        state["t"] += seconds

    return clock, sleep


def test_settle_returns_immediately_once_active():
    runner = ScriptedRunner(default_stdout="active\n")
    clock, sleep = _fake_clock_sleep()
    state = step6._settle("mv3dt-pipeline@x.service", runner=runner, clock=clock, sleep=sleep)
    assert state == "active"
    assert len(runner.calls) == 1


def test_settle_polls_through_activating_until_active():
    responses = iter(["activating", "activating", "active"])
    calls = []

    def runner(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, next(responses) + "\n", "")

    clock, sleep = _fake_clock_sleep()
    state = step6._settle(
        "mv3dt-pipeline@x.service", runner=runner, poll_s=1.0, clock=clock, sleep=sleep
    )
    assert state == "active"
    assert len(calls) == 3


def test_settle_gives_up_at_timeout_without_reporting_active():
    runner = ScriptedRunner(default_stdout="activating\n")
    clock, sleep = _fake_clock_sleep()
    state = step6._settle(
        "mv3dt-pipeline@x.service",
        runner=runner,
        timeout_s=3.0,
        poll_s=1.0,
        clock=clock,
        sleep=sleep,
    )
    assert state == "activating"  # never optimistically "active"


def test_settle_handles_stop_via_deactivating_to_inactive():
    responses = iter(["deactivating", "inactive"])

    def runner(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, next(responses) + "\n", "")

    clock, sleep = _fake_clock_sleep()
    state = step6._settle("mv3dt-pipeline@x.service", runner=runner, clock=clock, sleep=sleep)
    assert state == "inactive"


# ---------------------------------------------------------------------------
# section C.6 -- action -> systemctl mapping, idempotency, unknown projects
# ---------------------------------------------------------------------------


def test_handle_command_run_maps_to_systemctl_start_and_reports_active(tmp_path):
    install_dir = tmp_path / "mv3dt"
    _make_entry(install_dir)
    runner = ScriptedRunner()
    runner.when(_prefix("systemctl", "start", "mv3dt-pipeline@north-lobby-2.service"), returncode=0)
    runner.when(_prefix("systemctl", "is-active", "mv3dt-pipeline@north-lobby-2.service"), stdout="active\n")
    runner.when(_prefix("systemctl", "is-enabled", "mv3dt-pipeline@north-lobby-2.service"), stdout="enabled\n")

    result = step6.handle_command(
        {"action": "run", "project": "North Lobby #2", "request_id": "r1", "ts": "x"},
        install_dir=install_dir,
        runner=runner,
        dedup=step6.RequestDedup(),
    )
    assert result["ok"] is True
    assert result["project"] == "north-lobby-2"
    assert result["state"] == "active"
    assert result["enabled"] == "enabled"
    assert result["error"] is None
    assert any(argv[:2] == ["systemctl", "start"] for argv in runner.calls)


def test_handle_command_resolves_project_by_slug_too(tmp_path):
    install_dir = tmp_path / "mv3dt"
    _make_entry(install_dir)
    runner = ScriptedRunner(default_stdout="active\n")
    result = step6.handle_command(
        {"action": "status", "project": "north-lobby-2", "request_id": "r2", "ts": "x"},
        install_dir=install_dir,
        runner=runner,
        dedup=step6.RequestDedup(),
    )
    assert result["ok"] is True
    assert result["project"] == "north-lobby-2"


def test_handle_command_stop_maps_to_systemctl_stop(tmp_path):
    install_dir = tmp_path / "mv3dt"
    _make_entry(install_dir)
    runner = ScriptedRunner()
    runner.when(_prefix("systemctl", "stop", "mv3dt-pipeline@north-lobby-2.service"), returncode=0)
    runner.when(_prefix("systemctl", "is-active", "mv3dt-pipeline@north-lobby-2.service"), stdout="inactive\n")
    runner.when(_prefix("systemctl", "is-enabled", "mv3dt-pipeline@north-lobby-2.service"), stdout="enabled\n")

    result = step6.handle_command(
        {"action": "stop", "project": "North Lobby #2", "request_id": "r3", "ts": "x"},
        install_dir=install_dir,
        runner=runner,
        dedup=step6.RequestDedup(),
    )
    assert result["ok"] is True
    assert result["state"] == "inactive"


def test_handle_command_restart_requires_active_after_settle(tmp_path):
    install_dir = tmp_path / "mv3dt"
    _make_entry(install_dir)
    runner = ScriptedRunner()
    runner.when(_prefix("systemctl", "restart", "mv3dt-pipeline@north-lobby-2.service"), returncode=0)
    runner.when(_prefix("systemctl", "is-active", "mv3dt-pipeline@north-lobby-2.service"), stdout="failed\n")
    runner.when(_prefix("systemctl", "is-enabled", "mv3dt-pipeline@north-lobby-2.service"), stdout="enabled\n")

    result = step6.handle_command(
        {"action": "restart", "project": "North Lobby #2", "request_id": "r4", "ts": "x"},
        install_dir=install_dir,
        runner=runner,
        dedup=step6.RequestDedup(),
    )
    assert result["ok"] is False
    assert "not active" in result["error"]


def test_handle_command_unknown_project_never_calls_systemctl(tmp_path):
    install_dir = tmp_path / "mv3dt"
    install_dir.mkdir(parents=True)
    runner = ScriptedRunner()

    result = step6.handle_command(
        {"action": "run", "project": "does-not-exist", "request_id": "r5", "ts": "x"},
        install_dir=install_dir,
        runner=runner,
        dedup=step6.RequestDedup(),
    )
    assert result["ok"] is False
    assert result["error"] == "unknown project"
    assert result["state"] == "unknown"
    assert runner.calls == []


def test_handle_command_malformed_missing_action_never_calls_systemctl(tmp_path):
    install_dir = tmp_path / "mv3dt"
    install_dir.mkdir(parents=True)
    runner = ScriptedRunner()
    result = step6.handle_command(
        {"request_id": "r6", "ts": "x"}, install_dir=install_dir, runner=runner, dedup=step6.RequestDedup()
    )
    assert result["ok"] is False
    assert "invalid command" in result["error"]
    assert runner.calls == []


def test_handle_command_unknown_action_is_rejected():
    err = step6.validate_command({"action": "delete", "request_id": "r", "ts": "x"})
    assert err is not None and "unknown action" in err


def test_handle_command_not_a_dict_is_rejected():
    assert step6.validate_command(["not", "a", "dict"]) is not None
    assert step6.validate_command(None) is not None


def test_handle_command_systemctl_failure_is_reported_not_raised(tmp_path):
    install_dir = tmp_path / "mv3dt"
    _make_entry(install_dir)
    runner = ScriptedRunner()
    runner.when(_prefix("systemctl", "start", "mv3dt-pipeline@north-lobby-2.service"), returncode=1)
    runner.when(_prefix("systemctl", "is-active", "mv3dt-pipeline@north-lobby-2.service"), stdout="inactive\n")

    result = step6.handle_command(
        {"action": "run", "project": "North Lobby #2", "request_id": "r7", "ts": "x"},
        install_dir=install_dir,
        runner=runner,
        dedup=step6.RequestDedup(),
    )
    assert result["ok"] is False
    assert "systemctl start failed" in result["error"]


# ---------------------------------------------------------------------------
# section C.6 -- request_id de-dup window
# ---------------------------------------------------------------------------


def test_dedup_returns_cached_result_within_ttl():
    clock, _sleep = _fake_clock_sleep()
    dedup = step6.RequestDedup(ttl_s=300.0, clock=clock)
    dedup.put("r1", {"ok": True, "cached": True})
    assert dedup.get("r1") == {"ok": True, "cached": True}


def test_dedup_expires_after_ttl():
    clock, sleep = _fake_clock_sleep()
    dedup = step6.RequestDedup(ttl_s=10.0, clock=clock)
    dedup.put("r1", {"ok": True})
    sleep(11.0)
    assert dedup.get("r1") is None


def test_dedup_ignores_falsy_request_id():
    dedup = step6.RequestDedup()
    dedup.put("", {"ok": True})
    dedup.put(None, {"ok": True})
    assert dedup.get("") is None
    assert dedup.get(None) is None


def test_handle_command_redelivery_does_not_double_execute(tmp_path):
    install_dir = tmp_path / "mv3dt"
    _make_entry(install_dir)
    runner = ScriptedRunner()
    runner.when(_prefix("systemctl", "start", "mv3dt-pipeline@north-lobby-2.service"), returncode=0)
    runner.when(_prefix("systemctl", "is-active", "mv3dt-pipeline@north-lobby-2.service"), stdout="active\n")
    runner.when(_prefix("systemctl", "is-enabled", "mv3dt-pipeline@north-lobby-2.service"), stdout="enabled\n")
    dedup = step6.RequestDedup()

    payload = {"action": "run", "project": "North Lobby #2", "request_id": "dupe", "ts": "x"}
    first = step6.handle_command(payload, install_dir=install_dir, runner=runner, dedup=dedup)
    calls_after_first = len(runner.calls)
    second = step6.handle_command(payload, install_dir=install_dir, runner=runner, dedup=dedup)

    assert second == first
    assert len(runner.calls) == calls_after_first  # no new systemctl calls


# ---------------------------------------------------------------------------
# section B.2 -- broker topology decision
# ---------------------------------------------------------------------------


def test_broker_topology_off_gate_is_local_only():
    decision = step6.decide_broker_topology({}, gate_value="off")
    assert decision.topology is step6.BrokerTopology.LOCAL_ONLY
    assert decision.missing_cloud_endpoint is False


def test_broker_topology_local_gate_is_local_only_even_with_cloud_keys_set():
    conf = {
        "MV3DT_CLOUD_BROKER_HOST": "cloud.example",
        "MV3DT_CLOUD_BROKER_PORT": "8883",
        "MV3DT_CLOUD_BROKER_USER": "u",
        "MV3DT_CLOUD_BROKER_PASS": "p",
    }
    decision = step6.decide_broker_topology(conf, gate_value="local")
    assert decision.topology is step6.BrokerTopology.LOCAL_ONLY
    assert decision.missing_cloud_endpoint is False


def test_broker_topology_remote_gate_with_full_config_gets_bridge():
    conf = {
        "MV3DT_CLOUD_BROKER_HOST": "cloud.example",
        "MV3DT_CLOUD_BROKER_PORT": "8883",
        "MV3DT_CLOUD_BROKER_USER": "u",
        "MV3DT_CLOUD_BROKER_PASS": "p",
    }
    decision = step6.decide_broker_topology(conf, gate_value="remote")
    assert decision.topology is step6.BrokerTopology.LOCAL_WITH_BRIDGE
    assert decision.missing_cloud_endpoint is False


def test_broker_topology_remote_gate_missing_config_flags_user_action():
    decision = step6.decide_broker_topology({"MV3DT_CLOUD_BROKER_HOST": "cloud.example"}, gate_value="remote")
    assert decision.topology is step6.BrokerTopology.LOCAL_ONLY
    assert decision.missing_cloud_endpoint is True


def test_missing_cloud_broker_keys_lists_only_the_absent_ones():
    conf = {"MV3DT_CLOUD_BROKER_HOST": "cloud.example", "MV3DT_CLOUD_BROKER_USER": "u"}
    missing = step6.missing_cloud_broker_keys(conf)
    assert missing == ["MV3DT_CLOUD_BROKER_PORT", "MV3DT_CLOUD_BROKER_PASS"]


# ---------------------------------------------------------------------------
# section A.3 -- disable-before-remove hook
# ---------------------------------------------------------------------------


class _FakeUser:
    name = "op"


class _FakeCtx:
    def __init__(self, install_dir, conf, runner, *, non_interactive=True):
        self.install_dir = install_dir
        self.conf = conf
        self.user = _FakeUser()
        self._runner = runner
        self.non_interactive = non_interactive
        from mv3dt_installer import logs

        self.log = logs.log

    def run_root(self, *args, **kwargs):
        return self._runner(list(args), **kwargs)

    def run_as_user(self, *args, **kwargs):
        return self._runner(list(args), **kwargs)


def test_disable_hook_is_a_noop_when_gate_off(tmp_path):
    install_dir = tmp_path / "mv3dt"
    entry = _make_entry(install_dir)
    runner = ScriptedRunner()
    ctx = _FakeCtx(install_dir, {config_mod.GATE_REMOTE_SUPERVISION: "off"}, runner)

    step6._disable_pipeline_unit_before_remove(ctx, entry)
    assert runner.calls == []


def test_disable_hook_runs_systemctl_disable_now_when_gate_active(tmp_path):
    install_dir = tmp_path / "mv3dt"
    entry = _make_entry(install_dir)
    runner = ScriptedRunner()
    ctx = _FakeCtx(install_dir, {config_mod.GATE_REMOTE_SUPERVISION: "local"}, runner)

    step6._disable_pipeline_unit_before_remove(ctx, entry)
    assert runner.calls == [["systemctl", "disable", "--now", "mv3dt-pipeline@north-lobby-2.service"]]


def test_remove_project_artifacts_calls_the_registered_hook_before_removing(tmp_path):
    install_dir = tmp_path / "mv3dt"
    entry = _make_entry(install_dir)
    runner = ScriptedRunner()
    ctx = _FakeCtx(install_dir, {config_mod.GATE_REMOTE_SUPERVISION: "local"}, runner)

    step5.register_removal_hook(step6._disable_pipeline_unit_before_remove)
    step5.remove_project_artifacts(ctx, entry)

    assert runner.calls == [["systemctl", "disable", "--now", "mv3dt-pipeline@north-lobby-2.service"]]


# ---------------------------------------------------------------------------
# section C.5 -- build_status_payload() / uptime_s
# ---------------------------------------------------------------------------


def _show_stdout(**props):
    return "\n".join(f"{k}={v}" for k, v in props.items()) + "\n"


def test_uptime_s_computed_from_active_enter_timestamp(tmp_path):
    install_dir = tmp_path / "mv3dt"
    _make_entry(install_dir)
    runner = ScriptedRunner()
    runner.when(
        _prefix("systemctl", "show", "mv3dt-pipeline@north-lobby-2.service"),
        stdout=_show_stdout(
            ActiveState="active",
            SubState="running",
            ActiveEnterTimestamp="Wed 2024-01-17 10:15:00 UTC",
            ExecMainStatus="0",
            NRestarts="0",
        ),
    )
    runner.when(_prefix("systemctl", "is-enabled"), stdout="enabled\n")
    runner.when(_prefix("nvidia-smi"), returncode=1)

    fixed_now = lambda: datetime(2024, 1, 17, 10, 15, 30, tzinfo=timezone.utc)  # noqa: E731
    payload = step6.build_status_payload(
        host_id="desk-lab-01", install_dir=install_dir, runner=runner, now=fixed_now
    )
    assert payload["pipelines"]["north-lobby-2"]["uptime_s"] == 30


def test_uptime_s_is_zero_when_unit_is_not_active(tmp_path):
    install_dir = tmp_path / "mv3dt"
    _make_entry(install_dir)
    runner = ScriptedRunner()
    runner.when(
        _prefix("systemctl", "show", "mv3dt-pipeline@north-lobby-2.service"),
        stdout=_show_stdout(
            ActiveState="failed",
            SubState="failed",
            ActiveEnterTimestamp="Wed 2024-01-17 10:15:00 UTC",
            ExecMainStatus="1",
            NRestarts="5",
        ),
    )
    runner.when(_prefix("systemctl", "is-enabled"), stdout="enabled\n")
    runner.when(_prefix("nvidia-smi"), returncode=1)

    payload = step6.build_status_payload(host_id="desk-lab-01", install_dir=install_dir, runner=runner)
    entry = payload["pipelines"]["north-lobby-2"]
    assert entry["uptime_s"] == 0
    assert entry["active"] == "failed"
    assert entry["last_exit_code"] == 1
    assert entry["restarts"] == 5


def test_uptime_s_is_zero_when_timestamp_unparseable_or_unset(tmp_path):
    install_dir = tmp_path / "mv3dt"
    _make_entry(install_dir)
    runner = ScriptedRunner()
    runner.when(
        _prefix("systemctl", "show", "mv3dt-pipeline@north-lobby-2.service"),
        stdout=_show_stdout(ActiveState="active", SubState="running", ActiveEnterTimestamp="n/a"),
    )
    runner.when(_prefix("systemctl", "is-enabled"), stdout="enabled\n")
    runner.when(_prefix("nvidia-smi"), returncode=1)

    payload = step6.build_status_payload(host_id="desk-lab-01", install_dir=install_dir, runner=runner)
    assert payload["pipelines"]["north-lobby-2"]["uptime_s"] == 0


def test_parse_systemd_timestamp_accepts_the_documented_format():
    parsed = step6._parse_systemd_timestamp("Wed 2024-01-17 10:15:23 UTC")
    assert parsed == datetime(2024, 1, 17, 10, 15, 23, tzinfo=timezone.utc)


def test_parse_systemd_timestamp_returns_none_for_unset_sentinels():
    assert step6._parse_systemd_timestamp("n/a") is None
    assert step6._parse_systemd_timestamp("") is None
    assert step6._parse_systemd_timestamp("0") is None
    assert step6._parse_systemd_timestamp("garbage") is None


def test_build_status_payload_falls_back_to_unknown_when_show_fails(tmp_path):
    install_dir = tmp_path / "mv3dt"
    _make_entry(install_dir)
    runner = ScriptedRunner()
    runner.when(_prefix("systemctl", "show"), returncode=1)
    runner.when(_prefix("systemctl", "is-enabled"), stdout="enabled\n")
    runner.when(_prefix("nvidia-smi"), returncode=1)

    payload = step6.build_status_payload(host_id="desk-lab-01", install_dir=install_dir, runner=runner)
    entry = payload["pipelines"]["north-lobby-2"]
    assert entry["active"] == "unknown"
    assert entry["sub"] == "unknown"
    assert entry["uptime_s"] == 0


def test_build_status_payload_omits_gpu_when_nvidia_smi_fails(tmp_path):
    install_dir = tmp_path / "mv3dt"
    _make_entry(install_dir)
    runner = ScriptedRunner(default_returncode=1)
    payload = step6.build_status_payload(host_id="desk-lab-01", install_dir=install_dir, runner=runner)
    assert "gpu" not in payload


def test_build_status_payload_includes_gpu_snapshot_on_success(tmp_path):
    install_dir = tmp_path / "mv3dt"
    _make_entry(install_dir)
    runner = ScriptedRunner()
    runner.when(_prefix("systemctl", "show"), returncode=1)
    runner.when(_prefix("systemctl", "is-enabled"), stdout="enabled\n")
    runner.when(_prefix("nvidia-smi"), returncode=0, stdout="61, 4210, 57\n")

    payload = step6.build_status_payload(host_id="desk-lab-01", install_dir=install_dir, runner=runner)
    assert payload["gpu"] == {"utilization_pct": 61, "memory_used_mb": 4210, "temperature_c": 57}


# ---------------------------------------------------------------------------
# section C.3 -- whole-host status command (project omitted)
# ---------------------------------------------------------------------------


def test_handle_command_whole_host_status_returns_the_c5_payload(tmp_path):
    install_dir = tmp_path / "mv3dt"
    _make_entry(install_dir)
    runner = ScriptedRunner()
    runner.when(_prefix("systemctl", "show"), returncode=1)
    runner.when(_prefix("systemctl", "is-enabled"), stdout="enabled\n")
    runner.when(_prefix("nvidia-smi"), returncode=1)

    result = step6.handle_command(
        {"action": "status", "request_id": "r1", "ts": "x"},
        install_dir=install_dir,
        runner=runner,
        dedup=step6.RequestDedup(),
        host_id="desk-lab-01",
    )
    assert result["ok"] is True
    assert result["action"] == "status"
    assert result["request_id"] == "r1"
    assert result["error"] is None
    # The real section C.5 payload shape, not a per-unit placeholder.
    assert result["host_id"] == "desk-lab-01"
    assert "pipelines" in result
    assert "north-lobby-2" in result["pipelines"]
    assert "agent_version" in result


def test_handle_command_per_project_status_is_unaffected_by_whole_host_change(tmp_path):
    install_dir = tmp_path / "mv3dt"
    _make_entry(install_dir)
    runner = ScriptedRunner(default_stdout="active\n")
    result = step6.handle_command(
        {"action": "status", "project": "north-lobby-2", "request_id": "r2", "ts": "x"},
        install_dir=install_dir,
        runner=runner,
        dedup=step6.RequestDedup(),
        host_id="desk-lab-01",
    )
    assert result["project"] == "north-lobby-2"
    assert result["state"] == "active"
    assert "pipelines" not in result


# ---------------------------------------------------------------------------
# section E.1 -- REQUIRED round-trip status check
# ---------------------------------------------------------------------------


def test_status_round_trip_dry_run_invokes_handle_command_in_process(tmp_path):
    install_dir = tmp_path / "mv3dt"
    _make_entry(install_dir)
    runner = ScriptedRunner()
    runner.when(_prefix("systemctl", "show"), returncode=1)
    runner.when(_prefix("systemctl", "is-enabled"), stdout="enabled\n")
    runner.when(_prefix("nvidia-smi"), returncode=1)

    result = step6.status_round_trip_dry_run(install_dir=install_dir, runner=runner, host_id="desk-lab-01")
    assert result["ok"] is True
    assert result["host_id"] == "desk-lab-01"


def test_status_round_trip_check_uses_dry_run_when_non_interactive(tmp_path):
    install_dir = tmp_path / "mv3dt"
    _make_entry(install_dir)
    runner = ScriptedRunner()
    runner.when(_prefix("systemctl", "show"), returncode=1)
    runner.when(_prefix("systemctl", "is-enabled"), stdout="enabled\n")
    runner.when(_prefix("nvidia-smi"), returncode=1)
    ctx = _FakeCtx(install_dir, {}, runner, non_interactive=True)

    live_called = []
    ok, result = step6.status_round_trip_check(
        ctx=ctx,
        host_id="desk-lab-01",
        install_dir=install_dir,
        runner=runner,
        live_round_trip=lambda host_id: live_called.append(host_id) or {"ok": True},
    )
    assert ok is True
    assert live_called == []  # non-interactive skips the live attempt entirely


def test_status_round_trip_check_tries_live_first_when_interactive(tmp_path):
    install_dir = tmp_path / "mv3dt"
    _make_entry(install_dir)
    runner = ScriptedRunner()
    ctx = _FakeCtx(install_dir, {}, runner, non_interactive=False)

    ok, result = step6.status_round_trip_check(
        ctx=ctx,
        host_id="desk-lab-01",
        install_dir=install_dir,
        runner=runner,
        live_round_trip=lambda host_id: {"ok": True, "host_id": host_id, "via": "live"},
    )
    assert ok is True
    assert result["via"] == "live"


def test_status_round_trip_check_falls_back_to_dry_run_when_live_unreachable(tmp_path):
    install_dir = tmp_path / "mv3dt"
    _make_entry(install_dir)
    runner = ScriptedRunner()
    runner.when(_prefix("systemctl", "show"), returncode=1)
    runner.when(_prefix("systemctl", "is-enabled"), stdout="enabled\n")
    runner.when(_prefix("nvidia-smi"), returncode=1)
    ctx = _FakeCtx(install_dir, {}, runner, non_interactive=False)

    ok, result = step6.status_round_trip_check(
        ctx=ctx,
        host_id="desk-lab-01",
        install_dir=install_dir,
        runner=runner,
        live_round_trip=lambda host_id: None,  # no broker/agent reachable
    )
    assert ok is True
    assert result["host_id"] == "desk-lab-01"  # the dry-run payload, not a live one


def test_status_round_trip_check_reports_failure_from_live_result():
    ctx = _FakeCtx(pathlib.Path("/nonexistent"), {}, ScriptedRunner(), non_interactive=False)
    ok, result = step6.status_round_trip_check(
        ctx=ctx,
        host_id="desk-lab-01",
        install_dir=pathlib.Path("/nonexistent"),
        runner=ScriptedRunner(),
        live_round_trip=lambda host_id: {"ok": False, "error": "timed out"},
    )
    assert ok is False
    assert result["error"] == "timed out"
