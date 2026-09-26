"""Tests for the Step 4 AMC API result-ingest contract."""

from __future__ import annotations

import io
import pathlib
import shutil
import stat
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from mv3dt_installer import (
    app,
    config as config_mod,
    logs,
    report,
    systemd,
    waitui,
)  # noqa: E402
from mv3dt_installer.steps import STEP_REGISTRY, StepStatus  # noqa: E402
from mv3dt_installer.steps import step3_amc_launcher as step3  # noqa: E402
from mv3dt_installer.steps import step4_calib_output_wiring as step4  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
REAL_TEMPLATES_DIR = REPO_ROOT / "laptop" / "deepstream"


@pytest.fixture(autouse=True)
def _isolate_globals(monkeypatch, tmp_path):
    logs._transcript_path = None
    monkeypatch.setattr(systemd, "UNIT_DIR", tmp_path / "units")
    yield
    logs._transcript_path = None


@pytest.fixture
def templates(tmp_path):
    dest = tmp_path / "assets" / "deepstream"
    dest.mkdir(parents=True)
    for name in (
        step4.APP_CONFIG_TEMPLATE_NAME,
        step4.TRACKER_YAML_TEMPLATE_NAME,
        step4.INFER_PRIMARY_TEMPLATE_NAME,
        step4.MSGCONV_TEMPLATE_NAME,
    ):
        shutil.copy2(REAL_TEMPLATES_DIR / name, dest / name)
    return dest


def _zip_bytes(files=None, *, symlink=None) -> bytes:
    files = files or {"transforms.yml": "transforms: []\n", "metadata.json": "{}\n"}
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
        if symlink:
            info = zipfile.ZipInfo(symlink)
            info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(info, "transforms.yml")
    return stream.getvalue()


def _special_member_zip(name: str, mode: int) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("transforms.yml", "transforms: []\n")
        info = zipfile.ZipInfo(name)
        info.create_system = 3
        info.external_attr = mode << 16
        archive.writestr(info, "special")
    return stream.getvalue()


class Runner:
    def __init__(
        self,
        *,
        states=None,
        archive=None,
        status_rc=0,
        download_rc=0,
        log="solver failed",
        log_rc=0,
        status_payload=None,
        disable_rc=0,
        enabled_units=None,
    ):
        self.states = list(states or ["COMPLETED"])
        self.archive = archive if archive is not None else _zip_bytes()
        self.status_rc = status_rc
        self.download_rc = download_rc
        self.log = log
        self.log_rc = log_rc
        self.status_payload = status_payload
        self.disable_rc = disable_rc
        self.enabled_units = enabled_units
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if args and args[0] == "curl":
            url = args[-1]
            if "/get_project_info/" in url:
                if self.status_rc:
                    return subprocess.CompletedProcess(
                        args, self.status_rc, "", "status unavailable"
                    )
                if self.status_payload is not None:
                    return subprocess.CompletedProcess(args, 0, self.status_payload, "")
                state = self.states.pop(0) if len(self.states) > 1 else self.states[0]
                return subprocess.CompletedProcess(
                    args,
                    0,
                    f'{{"code":0,"project_info":{{"project_state":"{state}"}}}}',
                    "",
                )
            if "/amc/calibrate/" in url:
                return subprocess.CompletedProcess(
                    args,
                    self.log_rc,
                    self.log if not self.log_rc else "",
                    "log unavailable" if self.log_rc else "",
                )
            if "/mv3dt_result?result_type=amc" in url:
                if self.download_rc:
                    return subprocess.CompletedProcess(
                        args, self.download_rc, "", "download failed"
                    )
                pathlib.Path(args[args.index("-o") + 1]).write_bytes(self.archive)
                return subprocess.CompletedProcess(args, 0, "", "")
        if args[:3] == ("systemctl", "is-enabled", "--quiet"):
            enabled = (
                args[3].endswith(".timer")
                if self.enabled_units is None
                else args[3] in self.enabled_units
            )
            return subprocess.CompletedProcess(args, 0 if enabled else 1, "", "")
        if args[:3] == ("systemctl", "disable", "--now"):
            return subprocess.CompletedProcess(
                args,
                self.disable_rc,
                "",
                "disable failed" if self.disable_rc else "",
            )
        return subprocess.CompletedProcess(args, 0, "", "")


class User:
    def __init__(self, home):
        self.name = "operator"
        self.uid = 1000
        self.gid = 1000
        self.home = home


class Context:
    def __init__(
        self, tmp_path, templates, *, runner=None, conf=None, non_interactive=False
    ):
        self.install_dir = tmp_path / "mv3dt"
        self.install_dir.mkdir(parents=True, exist_ok=True)
        self.user = User(tmp_path / "home" / "operator")
        self.user.home.mkdir(parents=True)
        self.conf = conf or _conf(tmp_path)
        self.non_interactive = non_interactive
        self.runner = runner or Runner()
        self.log = logs.log
        self.report_installed = report.report_installed
        self.report_already_installed = report.report_already_installed
        self.verify_pinned = report.verify_pinned
        self.progress = SimpleNamespace(phase=lambda n: None, task=lambda n: None)
        self.templates = templates

    def asset_path(self, *parts):
        return self.templates.joinpath(*parts[1:])

    def run_root(self, *args, **kwargs):
        return self.runner(*args, **kwargs)

    def run_as_user(self, *args, **kwargs):
        return self.runner(*args, **kwargs)


def _conf(tmp_path):
    camera_file = tmp_path / "cameras.yml"
    camera_file.write_text(
        "cameras:\n"
        "  - id: cam1\n"
        "    ip: 10.0.0.1\n"
        "    rtsp_path: /Streaming/Channels/101\n"
        "    enabled: true\n",
        encoding="utf-8",
    )
    return {
        step4.CONF_LOCATION_ID_KEY: "site-42",
        step4.CONF_PROJECT_NAME_KEY: "site-42",
        step4.CONF_AMC_PROJECT_ID_KEY: "project-123",
        step4.CONF_MS_PORT_KEY: "8000",
        step3.CONF_UI_PORT_KEY: "5000",
        step4.CONF_CAM_USER_KEY: "admin",
        step4.CONF_CAM_PASSWORD_KEY: "secret",
        config_mod.CAMERAS_FILE_KEY: str(camera_file),
        step4.CONF_AMC_EXPORT_WAIT_S_KEY: "30",
    }


def _wrapper(ctx):
    path = ctx.install_dir / "bin" / step3.AMC_WRAPPER_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n", encoding="utf-8")
    path.chmod(0o755)
    installer = ctx.install_dir / "bin" / step3.INSTALLER_BIN_NAME
    installer.write_text("#!/bin/sh\n", encoding="utf-8")
    installer.chmod(0o755)


def _ctx(tmp_path, templates, **kwargs):
    ctx = Context(tmp_path, templates, **kwargs)
    _wrapper(ctx)
    return ctx


def _complete_wait(monkeypatch):
    monkeypatch.setattr(
        waitui,
        "wait_until",
        lambda predicate, **kwargs: (
            waitui.WaitOutcome.SATISFIED if predicate() else waitui.WaitOutcome.TIMEOUT
        ),
    )


def test_registration_and_subcommand():
    assert (
        len(
            [
                step
                for step in STEP_REGISTRY
                if step.id == step4.Step4CalibOutputWiring.id
            ]
        )
        == 1
    )
    assert "ingest" in app.SUBCOMMAND_REGISTRY


def test_resolve_inputs_consumes_step3_project_contract(tmp_path, templates):
    ctx = _ctx(tmp_path, templates)
    inputs, missing = step4.resolve_project_inputs(ctx)
    assert not missing
    assert inputs.project_id == "project-123"
    assert inputs.ms_port == "8000"


def test_missing_inputs_and_inventory_are_one_action_result(tmp_path, templates):
    conf = _conf(tmp_path)
    del conf[step4.CONF_AMC_PROJECT_ID_KEY]
    del conf[step4.CONF_CAM_PASSWORD_KEY]
    pathlib.Path(conf.pop(config_mod.CAMERAS_FILE_KEY)).unlink()
    ctx = _ctx(tmp_path, templates, conf=conf)
    result = step4.Step4CalibOutputWiring().preflight(ctx)
    assert result.status is StepStatus.USER_ACTION_REQUIRED
    assert "AMC_PROJECT_ID" in result.message and "CAM_PASSWORD" in result.message
    assert "camera inventory" in result.message
    assert len(result.user_actions) == 3
    assert "Step 3" in result.user_actions[0].text
    assert "AMC_PROJECT_ID" not in result.user_actions[1].text


def test_preflight_automatically_runs_first_camera_scan(
    tmp_path, templates, monkeypatch
):
    conf = _conf(tmp_path)
    pathlib.Path(conf.pop(config_mod.CAMERAS_FILE_KEY)).unlink()
    ctx = _ctx(tmp_path, templates, conf=conf)
    discovered = step4.cameras_mod.Camera(
        id="c1",
        mac="d0:3b:f4:00:00:01",
        ip="169.254.1.10",
        position="top-left",
        stream_ok=True,
    )

    def refresh(install_dir, **kwargs):
        assert kwargs["cam_user"] == "admin"
        assert kwargs["cam_password"] == "secret"
        (pathlib.Path(install_dir) / "cameras.yml").write_text(
            step4.cameras_mod.render_inventory([discovered], header=""),
            encoding="utf-8",
        )
        return step4.cameras_mod.ScanResult(
            cameras=[discovered], unmatched=[], tool="arp-scan", interfaces=["eth0"]
        )

    monkeypatch.setattr(step4.cameras_mod, "refresh", refresh)
    monkeypatch.setattr(step4.config_mod, "persist_value", lambda *args: None)
    _complete_wait(monkeypatch)

    result = step4.Step4CalibOutputWiring().preflight(ctx)

    assert result.status is StepStatus.COMPLETE
    assert pathlib.Path(ctx.conf[config_mod.CAMERAS_FILE_KEY]).is_file()


def test_preflight_zero_camera_scan_requests_connection_and_rerun(
    tmp_path, templates, monkeypatch
):
    conf = _conf(tmp_path)
    pathlib.Path(conf.pop(config_mod.CAMERAS_FILE_KEY)).unlink()
    ctx = _ctx(tmp_path, templates, conf=conf)
    monkeypatch.setattr(
        step4.cameras_mod,
        "refresh",
        lambda *args, **kwargs: step4.cameras_mod.ScanResult(
            cameras=[], unmatched=[], tool="arp-scan", interfaces=["eth0"]
        ),
    )

    result = step4.Step4CalibOutputWiring().preflight(ctx)

    assert result.status is StepStatus.USER_ACTION_REQUIRED
    assert "camera inventory" in result.message
    assert "Connect and activate" in result.user_actions[0].text


def test_preflight_polls_running_then_completed(tmp_path, templates, monkeypatch):
    runner = Runner(states=["RUNNING", "COMPLETED"])
    ctx = _ctx(tmp_path, templates, runner=runner)

    def wait(predicate, **kwargs):
        assert predicate() is False
        assert predicate() is True
        return waitui.WaitOutcome.SATISFIED

    monkeypatch.setattr(waitui, "wait_until", wait)
    result = step4.Step4CalibOutputWiring().preflight(ctx)
    assert result.status is StepStatus.COMPLETE
    urls = [call[0][-1] for call in runner.calls if call[0] and call[0][0] == "curl"]
    assert urls == [
        "http://localhost:8000/v1/get_project_info/project-123",
        "http://localhost:8000/v1/get_project_info/project-123",
    ]


def test_preflight_error_fetches_calibration_log(tmp_path, templates, monkeypatch):
    runner = Runner(states=["ERROR"], log="bundle adjustment exploded")
    ctx = _ctx(tmp_path, templates, runner=runner)
    _complete_wait(monkeypatch)
    result = step4.Step4CalibOutputWiring().preflight(ctx)
    assert result.status is StepStatus.FAILED
    assert "bundle adjustment exploded" in result.message
    assert any("/amc/calibrate/project-123/log" in c[0][-1] for c in runner.calls)


def test_status_and_log_polls_do_not_stream(tmp_path, templates, monkeypatch):
    runner = Runner(states=["ERROR"], log="bundle adjustment exploded")
    ctx = _ctx(tmp_path, templates, runner=runner)
    _complete_wait(monkeypatch)
    result = step4.Step4CalibOutputWiring().preflight(ctx)
    assert result.status is StepStatus.FAILED
    polls = [
        call
        for call in runner.calls
        if call[0]
        and call[0][0] == "curl"
        and ("/get_project_info/" in call[0][-1] or "/amc/calibrate/" in call[0][-1])
    ]
    assert len(polls) == 2
    assert all(kwargs.get("stream") is False for _, kwargs in polls)


def test_preflight_sets_waiting_task_before_wait(tmp_path, templates, monkeypatch):
    ctx = _ctx(tmp_path, templates)
    events = []
    ctx.progress = SimpleNamespace(
        phase=lambda n: None, task=lambda text: events.append(("task", text))
    )

    def wait(predicate, **kwargs):
        events.append(("wait", None))
        return waitui.WaitOutcome.SATISFIED if predicate() else waitui.WaitOutcome.TIMEOUT

    monkeypatch.setattr(waitui, "wait_until", wait)
    result = step4.Step4CalibOutputWiring().preflight(ctx)
    assert result.status is StepStatus.COMPLETE
    wait_index = events.index(("wait", None))
    assert events[wait_index - 1] == (
        "task",
        "waiting for AMC project site-42 to complete",
    )


def test_preflight_status_http_error_is_fatal(tmp_path, templates, monkeypatch):
    ctx = _ctx(tmp_path, templates, runner=Runner(status_rc=22))
    _complete_wait(monkeypatch)
    result = step4.Step4CalibOutputWiring().preflight(ctx)
    assert result.status is StepStatus.FAILED
    assert "status unavailable" in result.message


@pytest.mark.parametrize(
    "payload",
    [
        '{"project_state":"COMPLETED"}',
        '{"project_info":{"state":"COMPLETED"}}',
        '{"project_info":[]}',
    ],
)
def test_preflight_rejects_undocumented_status_shapes(
    tmp_path, templates, monkeypatch, payload
):
    ctx = _ctx(tmp_path, templates, runner=Runner(status_payload=payload))
    _complete_wait(monkeypatch)
    result = step4.Step4CalibOutputWiring().preflight(ctx)
    assert result.status is StepStatus.FAILED
    assert "project_info" in result.message or "project_state" in result.message


def test_preflight_error_reports_failed_log_fetch(tmp_path, templates, monkeypatch):
    ctx = _ctx(tmp_path, templates, runner=Runner(states=["ERROR"], log_rc=22))
    _complete_wait(monkeypatch)
    result = step4.Step4CalibOutputWiring().preflight(ctx)
    assert result.status is StepStatus.FAILED
    assert "log fetch failed" in result.message
    assert "log unavailable" in result.message


@pytest.mark.parametrize(
    "outcome",
    [
        waitui.WaitOutcome.TIMEOUT,
        waitui.WaitOutcome.CANCELLED,
        waitui.WaitOutcome.SKIPPED,
    ],
)
def test_preflight_bounded_wait_outcomes_need_action(
    tmp_path, templates, monkeypatch, outcome
):
    ctx = _ctx(
        tmp_path, templates, non_interactive=outcome is waitui.WaitOutcome.SKIPPED
    )

    def wait(predicate, **kwargs):
        assert kwargs["timeout_s"] == 30
        assert kwargs["non_interactive"] is ctx.non_interactive
        return outcome

    monkeypatch.setattr(waitui, "wait_until", wait)
    result = step4.Step4CalibOutputWiring().preflight(ctx)
    assert result.status is StepStatus.USER_ACTION_REQUIRED
    assert result.user_actions


def test_download_uses_current_amc_result_endpoint(tmp_path, templates):
    runner = Runner()
    ctx = _ctx(tmp_path, templates, runner=runner)
    inputs, _ = step4.resolve_project_inputs(ctx)
    dest = step4.default_calibration_dir(ctx, inputs.location_id)
    result = step4.download_and_ingest(ctx, inputs=inputs, dest=dest)
    assert result.changed
    assert (dest / "transforms.yml").is_file()
    assert runner.calls[0][0][-1] == (
        "http://localhost:8000/v1/result/project-123/mv3dt_result?result_type=amc"
    )


def test_download_http_error_preserves_existing_tree(tmp_path, templates):
    ctx = _ctx(tmp_path, templates, runner=Runner(download_rc=22))
    inputs, _ = step4.resolve_project_inputs(ctx)
    dest = step4.default_calibration_dir(ctx, inputs.location_id)
    dest.mkdir(parents=True)
    (dest / "transforms.yml").write_text("old\n", encoding="utf-8")
    with pytest.raises(step4.AmcApiError, match="download failed"):
        step4.download_and_ingest(ctx, inputs=inputs, dest=dest)
    assert (dest / "transforms.yml").read_text(encoding="utf-8") == "old\n"


@pytest.mark.parametrize(
    "archive, message",
    [
        (b"not zip", "invalid AMC result ZIP"),
        (_zip_bytes({"../escape": "bad", "transforms.yml": "ok"}), "unsafe path"),
        (_zip_bytes({"camera.json": "{}"}), "missing required transforms.yml"),
        (_zip_bytes({"/absolute": "bad", "transforms.yml": "ok"}), "unsafe path"),
        (_zip_bytes({"C:escape.txt": "bad", "transforms.yml": "ok"}), "unsafe path"),
        (_zip_bytes({"C:\\escape.txt": "bad", "transforms.yml": "ok"}), "unsafe path"),
        (_zip_bytes({"safe:stream": "bad", "transforms.yml": "ok"}), "unsafe path"),
    ],
)
def test_invalid_archives_are_rejected_without_partial_success(
    tmp_path, templates, archive, message
):
    ctx = _ctx(tmp_path, templates, runner=Runner(archive=archive))
    inputs, _ = step4.resolve_project_inputs(ctx)
    dest = step4.default_calibration_dir(ctx, inputs.location_id)
    with pytest.raises(step4.AmcApiError, match=message):
        step4.download_and_ingest(ctx, inputs=inputs, dest=dest)
    assert not dest.exists()
    assert not (tmp_path / "escape").exists()


def test_symlink_member_is_rejected(tmp_path, templates):
    ctx = _ctx(tmp_path, templates, runner=Runner(archive=_zip_bytes(symlink="link")))
    inputs, _ = step4.resolve_project_inputs(ctx)
    with pytest.raises(step4.AmcApiError, match="symlink"):
        step4.download_and_ingest(
            ctx,
            inputs=inputs,
            dest=step4.default_calibration_dir(ctx, inputs.location_id),
        )


def test_duplicate_member_is_rejected(tmp_path, templates):
    stream = io.BytesIO()
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(stream, "w") as archive:
            archive.writestr("transforms.yml", "first")
            archive.writestr("transforms.yml", "second")
    ctx = _ctx(tmp_path, templates, runner=Runner(archive=stream.getvalue()))
    inputs, _ = step4.resolve_project_inputs(ctx)
    with pytest.raises(step4.AmcApiError, match="duplicate path"):
        step4.download_and_ingest(
            ctx,
            inputs=inputs,
            dest=step4.default_calibration_dir(ctx, inputs.location_id),
        )


def test_unsupported_member_is_rejected(tmp_path, templates):
    archive = _special_member_zip("pipe", stat.S_IFIFO | 0o600)
    ctx = _ctx(tmp_path, templates, runner=Runner(archive=archive))
    inputs, _ = step4.resolve_project_inputs(ctx)
    with pytest.raises(step4.AmcApiError, match="unsupported member"):
        step4.download_and_ingest(
            ctx,
            inputs=inputs,
            dest=step4.default_calibration_dir(ctx, inputs.location_id),
        )


def test_expanded_size_limit_is_enforced(tmp_path, templates, monkeypatch):
    monkeypatch.setattr(step4, "_MAX_UNCOMPRESSED_BYTES", 1)
    ctx = _ctx(tmp_path, templates)
    inputs, _ = step4.resolve_project_inputs(ctx)
    with pytest.raises(step4.AmcApiError, match="1 GiB safety limit"):
        step4.download_and_ingest(
            ctx,
            inputs=inputs,
            dest=step4.default_calibration_dir(ctx, inputs.location_id),
        )


def test_encrypted_member_is_rejected(tmp_path, templates, monkeypatch):
    ctx = _ctx(tmp_path, templates)
    original = step4.zipfile.ZipFile

    class EncryptedZip:
        def __init__(self, *args, **kwargs):
            self.inner = original(*args, **kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.inner.close()

        def infolist(self):
            members = self.inner.infolist()
            members[0].flag_bits |= 0x1
            return members

        def open(self, member):
            return self.inner.open(member)

    monkeypatch.setattr(step4.zipfile, "ZipFile", EncryptedZip)
    inputs, _ = step4.resolve_project_inputs(ctx)
    with pytest.raises(step4.AmcApiError, match="encrypted member"):
        step4.download_and_ingest(
            ctx,
            inputs=inputs,
            dest=step4.default_calibration_dir(ctx, inputs.location_id),
        )


def test_empty_result_download_is_rejected(tmp_path, templates):
    ctx = _ctx(tmp_path, templates, runner=Runner(archive=b""))
    inputs, _ = step4.resolve_project_inputs(ctx)
    with pytest.raises(step4.AmcApiError, match="empty file"):
        step4.download_and_ingest(
            ctx,
            inputs=inputs,
            dest=step4.default_calibration_dir(ctx, inputs.location_id),
        )


def test_final_rename_failure_restores_previous_calibration(
    tmp_path, templates, monkeypatch
):
    ctx = _ctx(tmp_path, templates)
    inputs, _ = step4.resolve_project_inputs(ctx)
    dest = step4.default_calibration_dir(ctx, inputs.location_id)
    dest.mkdir(parents=True)
    (dest / "transforms.yml").write_text("previous\n", encoding="utf-8")
    real_replace = step4.os.replace
    calls = {"count": 0}

    def fail_final(source, target):
        calls["count"] += 1
        if calls["count"] == 2:
            raise OSError("final rename failed")
        return real_replace(source, target)

    monkeypatch.setattr(step4.os, "replace", fail_final)
    with pytest.raises(OSError, match="final rename failed"):
        step4.download_and_ingest(ctx, inputs=inputs, dest=dest)
    assert (dest / "transforms.yml").read_text(encoding="utf-8") == "previous\n"


def test_reingest_is_idempotent_for_same_archive(tmp_path, templates):
    runner = Runner()
    ctx = _ctx(tmp_path, templates, runner=runner)
    inputs, _ = step4.resolve_project_inputs(ctx)
    dest = step4.default_calibration_dir(ctx, inputs.location_id)
    first = step4.download_and_ingest(ctx, inputs=inputs, dest=dest)
    marker_mtime = (dest / step4._INGEST_LOG_NAME).stat().st_mtime_ns
    second = step4.download_and_ingest(ctx, inputs=inputs, dest=dest)
    assert first.changed is True and second.changed is False
    assert (dest / step4._INGEST_LOG_NAME).stat().st_mtime_ns == marker_mtime


def test_run_ingests_renders_and_installs_timer(tmp_path, templates):
    ctx = _ctx(tmp_path, templates, non_interactive=True)
    result = step4.Step4CalibOutputWiring().run(ctx)
    assert result.status is StepStatus.COMPLETE
    dest = step4.default_calibration_dir(ctx, "site-42")
    assert (dest / "transforms.yml").is_file()
    rendered = ctx.install_dir / "deepstream" / step4.RENDERED_APP_CONFIG_NAME
    assert "uri=rtsp://admin:secret@10.0.0.1" in rendered.read_text(encoding="utf-8")
    assert (systemd.UNIT_DIR / "mv3dt-ingest-site-42.timer").is_file()
    assert not (systemd.UNIT_DIR / "mv3dt-ingest-site-42.path").exists()
    timer_text = (systemd.UNIT_DIR / "mv3dt-ingest-site-42.timer").read_text(
        encoding="utf-8"
    )
    assert "OnUnitActiveSec=60s" in timer_text
    enabled_calls = [
        call[0][3]
        for call in ctx.runner.calls
        if call[0][:3] == ("systemctl", "enable", "--now")
    ]
    assert enabled_calls == ["mv3dt-ingest-site-42.timer"]


def test_run_removes_only_obsolete_legacy_path_unit(tmp_path, templates):
    ctx = _ctx(tmp_path, templates, non_interactive=True)
    systemd.UNIT_DIR.mkdir(parents=True)
    legacy = systemd.UNIT_DIR / "mv3dt-ingest-site-42.path"
    unrelated = systemd.UNIT_DIR / "mv3dt-ingest-other.path"
    legacy.write_text("old", encoding="utf-8")
    unrelated.write_text("keep", encoding="utf-8")
    assert step4.Step4CalibOutputWiring().run(ctx).status is StepStatus.COMPLETE
    assert not legacy.exists()
    assert unrelated.read_text(encoding="utf-8") == "keep"
    assert any(
        call[0][:4] == ("systemctl", "disable", "--now", "mv3dt-ingest-site-42.path")
        for call in ctx.runner.calls
    )


def test_run_preserves_legacy_path_when_disable_fails(tmp_path, templates):
    ctx = _ctx(
        tmp_path,
        templates,
        runner=Runner(disable_rc=1),
        non_interactive=True,
    )
    systemd.UNIT_DIR.mkdir(parents=True)
    legacy = systemd.UNIT_DIR / "mv3dt-ingest-site-42.path"
    legacy.write_text("old", encoding="utf-8")
    result = step4.Step4CalibOutputWiring().run(ctx)
    assert result.status is StepStatus.FAILED
    assert "disable failed" in result.message
    assert legacy.is_file()


def test_run_uses_interactive_alternate_destination(tmp_path, templates, monkeypatch):
    ctx = _ctx(tmp_path, templates, non_interactive=False)
    alternate = tmp_path / "custom calibration"
    monkeypatch.setattr(step4, "_PROMPT", lambda prompt: str(alternate))
    assert step4.Step4CalibOutputWiring().run(ctx).status is StepStatus.COMPLETE
    assert (alternate / "transforms.yml").is_file()
    assert ctx.conf[step4.CONF_CALIBRATION_DIR_KEY] == str(alternate)


def test_run_download_failure_does_not_persist_or_render(tmp_path, templates):
    ctx = _ctx(tmp_path, templates, runner=Runner(download_rc=22), non_interactive=True)
    result = step4.Step4CalibOutputWiring().run(ctx)
    assert result.status is StepStatus.FAILED
    assert step4.CONF_CALIBRATION_DIR_KEY not in ctx.conf
    assert not (
        ctx.install_dir / "deepstream" / step4.RENDERED_APP_CONFIG_NAME
    ).exists()


def test_verify_complete_after_run(tmp_path, templates):
    ctx = _ctx(tmp_path, templates, non_interactive=True)
    step = step4.Step4CalibOutputWiring()
    assert step.run(ctx).status is StepStatus.COMPLETE
    assert step.verify(ctx).status is StepStatus.COMPLETE


def test_verify_rejects_enabled_oneshot_service(tmp_path, templates):
    runner = Runner(
        enabled_units={
            "mv3dt-ingest-site-42.timer",
            "mv3dt-ingest-site-42.service",
        }
    )
    ctx = _ctx(tmp_path, templates, runner=runner, non_interactive=True)
    step = step4.Step4CalibOutputWiring()
    assert step.run(ctx).status is StepStatus.COMPLETE
    result = step.verify(ctx)
    assert result.status is StepStatus.FAILED
    assert "service must not be enabled" in result.message


def test_verify_requires_transforms_yml(tmp_path, templates):
    ctx = _ctx(tmp_path, templates, non_interactive=True)
    step = step4.Step4CalibOutputWiring()
    assert step.run(ctx).status is StepStatus.COMPLETE
    pathlib.Path(ctx.conf[step4.CONF_CALIBRATION_DIR_KEY], "transforms.yml").unlink()
    result = step.verify(ctx)
    assert result.status is StepStatus.FAILED
    assert "transforms.yml" in result.message


def test_ingest_subcommand_running_is_nonblocking(tmp_path, templates):
    runner = Runner(states=["RUNNING"])
    ctx = _ctx(tmp_path, templates, runner=runner, non_interactive=True)
    assert step4.handle_ingest_subcommand(["--project", "site-42"], ctx) == 0
    assert not any("mv3dt_result" in call[0][-1] for call in runner.calls)


def test_ingest_subcommand_completed_downloads_and_renders(tmp_path, templates):
    ctx = _ctx(tmp_path, templates, non_interactive=True)
    assert step4.handle_ingest_subcommand(["--project", "site-42"], ctx) == 0
    assert (step4.default_calibration_dir(ctx, "site-42") / "transforms.yml").is_file()


def test_ingest_subcommand_reuses_persisted_custom_destination(tmp_path, templates):
    conf = _conf(tmp_path)
    custom = tmp_path / "persisted calibration"
    conf[step4.CONF_CALIBRATION_DIR_KEY] = str(custom)
    ctx = _ctx(tmp_path, templates, conf=conf, non_interactive=True)
    assert step4.handle_ingest_subcommand(["--project", "site-42"], ctx) == 0
    assert (custom / "transforms.yml").is_file()
    assert not step4.default_calibration_dir(ctx, "site-42").exists()


def test_ingest_subcommand_error_log_returns_nonzero(tmp_path, templates):
    ctx = _ctx(
        tmp_path, templates, runner=Runner(states=["ERROR"], log="RMSE rejected")
    )
    assert step4.handle_ingest_subcommand([], ctx) == 1


def test_render_tracker_and_app_config(templates):
    tracker = step4.render_tracker_yaml(
        (templates / step4.TRACKER_YAML_TEMPLATE_NAME).read_text(encoding="utf-8"),
        location_id="site-42",
        calibration_directory="calibration/site-42",
    )
    assert "nodeID: site-42" in tracker
    assert "calibrationDirectory: calibration/site-42" in tracker
    camera = SimpleNamespace(ip="10.0.0.8", rtsp_path="/stream")
    app_text = step4.render_app_config(
        (templates / step4.APP_CONFIG_TEMPLATE_NAME).read_text(encoding="utf-8"),
        cam_user="operator",
        cam_password="pw",
        location_id="site-42",
        cameras=[camera],
        template_path="template",
        calibration_dir="calibration/site-42",
    )
    assert "uri=rtsp://operator:pw@10.0.0.8:554/stream" in app_text
    assert "topic=mv3dt/site-42/sv3d" in app_text
