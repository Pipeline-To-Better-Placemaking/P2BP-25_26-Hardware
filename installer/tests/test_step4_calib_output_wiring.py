"""Tests for the Step 4 AMC API result-ingest contract."""

from __future__ import annotations

import hashlib
import io
import json
import pathlib
import re
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
    footage as footage_mod,
    logs,
    managed_files,
    report,
    systemd,
    waitui,
)  # noqa: E402
from mv3dt_installer.steps import STEP_REGISTRY, StepStatus  # noqa: E402
from mv3dt_installer.steps import step3_amc_launcher as step3  # noqa: E402
from mv3dt_installer.steps import step4_calib_output_wiring as step4  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
REAL_TEMPLATES_DIR = REPO_ROOT / "laptop" / "deepstream"
FAKE_BODYPOSE = b"bodypose3dnet-onnx"


@pytest.fixture(autouse=True)
def _isolate_globals(monkeypatch, tmp_path):
    logs._transcript_path = None
    monkeypatch.setattr(systemd, "UNIT_DIR", tmp_path / "units")
    monkeypatch.setattr(
        managed_files, "DEFAULT_MANIFEST_PATH", tmp_path / "state" / "managed-files.json"
    )
    monkeypatch.setattr(step4, "BODYPOSE_MODEL_SIZE", len(FAKE_BODYPOSE))
    monkeypatch.setattr(
        step4, "BODYPOSE_MODEL_SHA256", hashlib.sha256(FAKE_BODYPOSE).hexdigest()
    )
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


# AMC 3.2.1's camInfo format: a 3x4 matrix plus a per-class modelInfo list
# (values from the workstation's 2026-10-20 Living-Room export).
AMC_CAMINFO = """projectionMatrix_3x4_w2p:
- -1587.628995154069
- 424.6347124685943
- 33.25890122231646
- 156440.66777167591
- -259.75150943048845
- 356.96099407611894
- -1370.071028032669
- 48320.507770494194
- -0.7542930219554992
- -0.5289291674665958
- -0.3889421201571442
- 58.03540606222064

modelInfo:
  - classID: 1 # person
    height: 1.7
    radius: 0.3
"""


def _zip_bytes(files=None, *, symlink=None) -> bytes:
    files = files or {
        "transforms.yml": "transforms: []\n",
        "metadata.json": "{}\n",
        "camInfo/camInfo_00.yml": AMC_CAMINFO,
    }
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
        reid_rc=0,
        bodypose_rc=0,
        bodypose_bytes=None,
        rectification=None,
    ):
        self.states = list(states or ["COMPLETED"])
        self.archive = archive if archive is not None else _zip_bytes()
        self.status_rc = status_rc
        self.download_rc = download_rc
        self.log = log
        self.log_rc = log_rc
        self.status_payload = status_payload
        self.disable_rc = disable_rc
        # Unit names `systemctl is-enabled` reports enabled; `enable --now`
        # adds to it and a successful `disable --now` removes from it.
        self.enabled_units = set(enabled_units or ())
        self.reid_rc = reid_rc
        self.bodypose_rc = bodypose_rc
        self.bodypose_bytes = bodypose_bytes
        # None: the endpoint answers 404, as an AMC without rectification.
        self.rectification = rectification
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if args and args[0] == "curl":
            url = args[-1]
            if url == step4.REID_MODEL_URL:
                if self.reid_rc:
                    return subprocess.CompletedProcess(args, self.reid_rc, "", "reid fetch failed")
                pathlib.Path(args[args.index("-o") + 1]).write_bytes(b"etlt-model")
                return subprocess.CompletedProcess(args, 0, "", "")
            if url == step4.BODYPOSE_MODEL_URL:
                if self.bodypose_rc:
                    return subprocess.CompletedProcess(
                        args, self.bodypose_rc, "", "bodypose fetch failed"
                    )
                pathlib.Path(args[args.index("-o") + 1]).write_bytes(
                    FAKE_BODYPOSE if self.bodypose_bytes is None else self.bodypose_bytes
                )
                return subprocess.CompletedProcess(args, 0, "", "")
            if "/rectification/" in url:
                if self.rectification is None:
                    return subprocess.CompletedProcess(
                        args, 22, "", "The requested URL returned error: 404"
                    )
                return subprocess.CompletedProcess(
                    args, 0, json.dumps(self.rectification), ""
                )
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
        if args[:3] == ("systemctl", "enable", "--now"):
            self.enabled_units.add(args[3])
            return subprocess.CompletedProcess(args, 0, "", "")
        if args[:3] == ("systemctl", "is-active", "--quiet"):
            return subprocess.CompletedProcess(
                args, 0 if args[3] in self.enabled_units else 3, "", ""
            )
        if args[:2] == ("systemctl", "is-enabled"):
            unit = args[-1]
            enabled = unit in self.enabled_units
            if enabled:
                return subprocess.CompletedProcess(args, 0, "enabled\n", "")
            if unit.endswith(".service"):
                # A oneshot service with no [Install] section: systemctl
                # prints `static` and still exits 0, as on a real host.
                return subprocess.CompletedProcess(args, 0, "static\n", "")
            return subprocess.CompletedProcess(args, 1, "disabled\n", "")
        if args[:3] == ("systemctl", "disable", "--now"):
            if not self.disable_rc:
                self.enabled_units.discard(args[3])
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
    # The first status request is the footage INIT check (footage spec 3).
    runner = Runner(states=["RUNNING", "RUNNING", "COMPLETED"])
    ctx = _ctx(tmp_path, templates, runner=runner)

    def wait(predicate, **kwargs):
        assert predicate() is False
        assert predicate() is True
        return waitui.WaitOutcome.SATISFIED

    monkeypatch.setattr(waitui, "wait_until", wait)
    result = step4.Step4CalibOutputWiring().preflight(ctx)
    assert result.status is StepStatus.COMPLETE
    urls = [call[0][-1] for call in runner.calls if call[0] and call[0][0] == "curl"]
    assert urls == ["http://localhost:8000/v1/get_project_info/project-123"] * 3


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
    service_text = (systemd.UNIT_DIR / "mv3dt-ingest-site-42.service").read_text(
        encoding="utf-8"
    )
    # A root service has no sudo; without SUDO_USER the installer's
    # preflight refuses to run, so the timer never ingested anything.
    assert "Environment=SUDO_USER=operator" in service_text
    # MV3DT_AUTO_INGEST defaults off: the units are installed, not enabled.
    assert not any(
        call[0][:3] == ("systemctl", "enable", "--now") for call in ctx.runner.calls
    )


def _systemctl_calls(ctx, *prefix):
    return [
        call[0] for call in ctx.runner.calls if call[0][: len(prefix)] == prefix
    ]


def test_run_enables_timer_when_auto_ingest_on(tmp_path, templates):
    conf = _conf(tmp_path)
    conf[config_mod.GATE_AUTO_INGEST] = "on"
    ctx = _ctx(tmp_path, templates, conf=conf, non_interactive=True)
    step = step4.Step4CalibOutputWiring()
    assert step.run(ctx).status is StepStatus.COMPLETE
    assert [c[3] for c in _systemctl_calls(ctx, "systemctl", "enable", "--now")] == [
        "mv3dt-ingest-site-42.timer"
    ]
    assert step.verify(ctx).status is StepStatus.COMPLETE


def test_run_disables_enabled_timer_when_auto_ingest_off(tmp_path, templates):
    runner = Runner(enabled_units={"mv3dt-ingest-site-42.timer"})
    ctx = _ctx(tmp_path, templates, runner=runner, non_interactive=True)
    step = step4.Step4CalibOutputWiring()
    assert step.run(ctx).status is StepStatus.COMPLETE
    assert ("systemctl", "disable", "--now", "mv3dt-ingest-site-42.timer") in [
        c[:4] for c in _systemctl_calls(ctx, "systemctl", "disable")
    ]
    assert "mv3dt-ingest-site-42.timer" not in runner.enabled_units
    assert step.verify(ctx).status is StepStatus.COMPLETE


def test_run_leaves_disabled_timer_alone_when_auto_ingest_off(tmp_path, templates):
    ctx = _ctx(tmp_path, templates, non_interactive=True)
    assert step4.Step4CalibOutputWiring().run(ctx).status is StepStatus.COMPLETE
    assert not _systemctl_calls(ctx, "systemctl", "disable")


def test_run_fails_when_timer_disable_fails(tmp_path, templates):
    runner = Runner(enabled_units={"mv3dt-ingest-site-42.timer"}, disable_rc=1)
    ctx = _ctx(tmp_path, templates, runner=runner, non_interactive=True)
    result = step4.Step4CalibOutputWiring().run(ctx)
    assert result.status is StepStatus.FAILED
    assert "mv3dt-ingest-site-42.timer" in result.message


def test_verify_rejects_enabled_timer_when_auto_ingest_off(tmp_path, templates):
    ctx = _ctx(tmp_path, templates, non_interactive=True)
    step = step4.Step4CalibOutputWiring()
    assert step.run(ctx).status is StepStatus.COMPLETE
    ctx.runner.enabled_units.add("mv3dt-ingest-site-42.timer")
    result = step.verify(ctx)
    assert result.status is StepStatus.FAILED
    assert "MV3DT_AUTO_INGEST is off" in result.message


def test_verify_requires_enabled_timer_when_auto_ingest_on(tmp_path, templates):
    conf = _conf(tmp_path)
    conf[config_mod.GATE_AUTO_INGEST] = "on"
    ctx = _ctx(tmp_path, templates, conf=conf, non_interactive=True)
    step = step4.Step4CalibOutputWiring()
    assert step.run(ctx).status is StepStatus.COMPLETE
    ctx.runner.enabled_units.discard("mv3dt-ingest-site-42.timer")
    result = step.verify(ctx)
    assert result.status is StepStatus.FAILED
    assert "is not enabled" in result.message


def test_run_keeps_hand_edited_ingest_unit_and_skips_reload(tmp_path, templates):
    ctx = _ctx(tmp_path, templates, non_interactive=True)
    step = step4.Step4CalibOutputWiring()
    assert step.run(ctx).status is StepStatus.COMPLETE
    assert _systemctl_calls(ctx, "systemctl", "daemon-reload")
    timer = systemd.UNIT_DIR / "mv3dt-ingest-site-42.timer"
    timer.write_text(timer.read_text(encoding="utf-8") + "# hand edit\n", encoding="utf-8")
    ctx.runner.calls.clear()
    assert step.run(ctx).status is StepStatus.COMPLETE
    assert timer.read_text(encoding="utf-8").endswith("# hand edit\n")
    # Unchanged installer content and an operator edit: nothing new to offer.
    assert not managed_files.pending_path(timer).exists()
    assert not _systemctl_calls(ctx, "systemctl", "daemon-reload")


def test_run_offers_new_ingest_unit_beside_pre_manifest_unit(tmp_path, templates):
    ctx = _ctx(tmp_path, templates, non_interactive=True)
    systemd.UNIT_DIR.mkdir(parents=True)
    timer = systemd.UNIT_DIR / "mv3dt-ingest-site-42.timer"
    timer.write_text("[Timer]\nOnUnitActiveSec=5min\n", encoding="utf-8")
    assert step4.Step4CalibOutputWiring().run(ctx).status is StepStatus.COMPLETE
    assert timer.read_text(encoding="utf-8") == "[Timer]\nOnUnitActiveSec=5min\n"
    pending = managed_files.pending_path(timer)
    assert "OnUnitActiveSec=60s" in pending.read_text(encoding="utf-8")


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


def test_verify_accepts_static_oneshot_service(tmp_path, templates):
    """The ingest `.service` has no [Install] section, so systemctl reports it
    `static` with exit 0. That is the correct, not-enabled state and must not
    fail verify (the v0.4.10 workstation failure)."""
    runner = Runner()
    ctx = _ctx(tmp_path, templates, runner=runner, non_interactive=True)
    step = step4.Step4CalibOutputWiring()
    assert step.run(ctx).status is StepStatus.COMPLETE
    assert step.verify(ctx).status is StepStatus.COMPLETE
    assert any(
        args[:2] == ("systemctl", "is-enabled") and args[-1].endswith(".service")
        for args, _ in runner.calls
    )


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
        camera_model_files=["/m/camInfo_00.yml", "/m/camInfo_01.yml"],
    )
    assert "${" not in tracker
    parsed = step4._parse_tracker_yaml(tracker)
    projection = parsed["ObjectModelProjection"]
    assert projection["cameraModelFilepath"] == ["/m/camInfo_00.yml", "/m/camInfo_01.yml"]
    assert projection["outputFootLocation"] == 1
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


def _source_enable(text, index):
    block = step4._source_block(text, index)
    return re.search(r"(?m)^enable=(\d)", block).group(1)


def test_render_app_config_disables_slots_without_a_camera(templates):
    """The template has eight [sourceN] slots. Slots past the inventory must
    be disabled, not left enabled on a placeholder URI that deepstream-app
    retries forever (2026-10-20 workstation run, two cameras)."""
    cameras = [
        SimpleNamespace(ip="10.0.0.8", rtsp_path="/stream"),
        SimpleNamespace(ip="10.0.0.9", rtsp_path="/stream"),
    ]
    app_text = step4.render_app_config(
        (templates / step4.APP_CONFIG_TEMPLATE_NAME).read_text(encoding="utf-8"),
        cam_user="operator",
        cam_password="pw",
        location_id="site-42",
        cameras=cameras,
        template_path="template",
        calibration_dir="calibration/site-42",
    )
    assert [_source_enable(app_text, i) for i in range(8)] == ["1", "1"] + ["0"] * 6


def test_convert_camera_model_flattens_amc_model_info():
    converted = step4.convert_camera_model(AMC_CAMINFO)
    data = step4.yaml.safe_load(converted)
    assert len(data["projectionMatrix_3x4_w2p"]) == 12
    assert data["projectionMatrix_3x4_w2p"][3] == 156440.66777167591
    assert data["modelInfo"] == {"height": 1.7, "radius": 0.3}


def test_convert_camera_model_rejects_a_short_matrix():
    with pytest.raises(ValueError):
        step4.convert_camera_model("projectionMatrix_3x4:\n- 1.0\nmodelInfo:\n  height: 1.7\n  radius: 0.3\n")


def test_run_writes_camera_models_and_lists_them(tmp_path, templates):
    ctx = _ctx(tmp_path, templates, non_interactive=True)
    step = step4.Step4CalibOutputWiring()
    assert step.run(ctx).status is StepStatus.COMPLETE
    model = ctx.install_dir / "deepstream" / step4.CAMERA_MODEL_DIR_NAME / "camInfo_00.yml"
    assert step4.yaml.safe_load(model.read_text())["modelInfo"] == {"height": 1.7, "radius": 0.3}
    tracker = step4._parse_tracker_yaml(
        (ctx.install_dir / "deepstream" / step4.TRACKER_YAML_TEMPLATE_NAME).read_text()
    )
    assert tracker["ObjectModelProjection"]["cameraModelFilepath"] == [str(model)]
    assert step.verify(ctx).status is StepStatus.COMPLETE


def test_run_fails_when_camera_models_do_not_match_cameras(tmp_path, templates):
    runner = Runner(archive=_zip_bytes({"transforms.yml": "transforms: []\n"}))
    ctx = _ctx(tmp_path, templates, runner=runner, non_interactive=True)
    result = step4.Step4CalibOutputWiring().run(ctx)
    assert result.status is StepStatus.FAILED
    assert "one per camera" in result.message


def test_rendered_header_names_the_template_without_its_unpack_path(templates):
    app_text = step4.render_app_config(
        (templates / step4.APP_CONFIG_TEMPLATE_NAME).read_text(encoding="utf-8"),
        cam_user="u",
        cam_password="p",
        location_id="site-42",
        cameras=[SimpleNamespace(ip="10.0.0.8", rtsp_path="/s")],
        template_path="/tmp/_MEIabc123/assets/deepstream/deepstream_app_config.txt",
        calibration_dir="calibration/site-42",
    )
    assert "# Source template: deepstream_app_config.txt\n" in app_text
    assert "_MEI" not in app_text


def test_run_downloads_reid_model_once(tmp_path, templates):
    runner = Runner()
    ctx = _ctx(tmp_path, templates, runner=runner, non_interactive=True)
    step = step4.Step4CalibOutputWiring()
    assert step.run(ctx).status is StepStatus.COMPLETE
    model = ctx.install_dir / step4.REID_RELATIVE_DIR / step4.REID_MODEL_NAME
    assert model.read_bytes() == b"etlt-model"
    fetches = [a for a, _ in runner.calls if a and a[0] == "curl" and a[-1] == step4.REID_MODEL_URL]
    assert len(fetches) == 1
    assert step.run(ctx).status is StepStatus.COMPLETE
    fetches = [a for a, _ in runner.calls if a and a[0] == "curl" and a[-1] == step4.REID_MODEL_URL]
    assert len(fetches) == 1


def test_run_fails_when_reid_download_fails(tmp_path, templates):
    runner = Runner(reid_rc=22)
    ctx = _ctx(tmp_path, templates, runner=runner, non_interactive=True)
    result = step4.Step4CalibOutputWiring().run(ctx)
    assert result.status is StepStatus.FAILED
    assert "ReID model download failed" in result.message
    reid_dir = ctx.install_dir / step4.REID_RELATIVE_DIR
    assert not any(reid_dir.glob("*")) if reid_dir.exists() else True


def test_verify_requires_reid_model(tmp_path, templates):
    ctx = _ctx(tmp_path, templates, non_interactive=True)
    step = step4.Step4CalibOutputWiring()
    assert step.run(ctx).status is StepStatus.COMPLETE
    (ctx.install_dir / step4.REID_RELATIVE_DIR / step4.REID_MODEL_NAME).unlink()
    result = step.verify(ctx)
    assert result.status is StepStatus.FAILED
    assert "ReID model" in result.message


# ---------------------------------------------------------------------------
# STEP-4-CALIBRATION-FOOTAGE sections 3 to 6: calibration footage wiring
# ---------------------------------------------------------------------------

FOOTAGE_PASSWORD = "hunter2-footage-secret"
FOOTAGE_CAMERAS = [
    step4.cameras_mod.Camera(
        id="c1",
        mac="d0:3b:f4:00:00:01",
        ip="169.254.1.10",
        position="top-left",
        stream_ok=True,
    ),
    step4.cameras_mod.Camera(
        id="c2",
        mac="d0:3b:f4:00:00:02",
        ip="169.254.1.11",
        position="top-right",
        stream_ok=False,
    ),
    step4.cameras_mod.Camera(
        id="c3",
        mac="d0:3b:f4:00:00:03",
        ip="169.254.1.12",
        position="",
        enabled=False,
    ),
]


def _footage_ctx(tmp_path, templates, *, states=("INIT", "COMPLETED"), **kwargs):
    conf = _conf(tmp_path)
    pathlib.Path(conf[config_mod.CAMERAS_FILE_KEY]).write_text(
        step4.cameras_mod.render_inventory(FOOTAGE_CAMERAS, header=""),
        encoding="utf-8",
    )
    conf[step4.CONF_CAM_PASSWORD_KEY] = FOOTAGE_PASSWORD
    enabled = [c for c in FOOTAGE_CAMERAS if c.enabled]
    archive = _zip_bytes(
        {
            "transforms.yml": "transforms: []\n",
            **{f"camInfo/camInfo_{i:02d}.yml": AMC_CAMINFO for i in range(len(enabled))},
        }
    )
    runner = kwargs.pop("runner", None) or Runner(states=list(states), archive=archive)
    ctx = _ctx(tmp_path, templates, conf=conf, runner=runner, **kwargs)
    ctx.events = []
    ctx.progress = SimpleNamespace(
        phase=lambda n: None,
        task=lambda text: ctx.events.append(("task", text)),
        percent=lambda value, note=None: ctx.events.append(("percent", value)),
    )
    return ctx


def _footage_project(ctx):
    return ctx.user.home / "Downloads" / footage_mod.FOOTAGE_DIRNAME / "site-42"


def _write_complete_set(ctx, *, seconds=300, persist=True):
    """A complete marked set; `persist` mirrors what recording leaves in
    installer.conf (the resolved footage root)."""
    project = _footage_project(ctx)
    project.mkdir(parents=True, exist_ok=True)
    if persist:
        ctx.conf[step4.CONF_CALIB_FOOTAGE_DIR_KEY] = str(project.parent)
    enabled = [c for c in FOOTAGE_CAMERAS if c.enabled]
    for camera in enabled:
        (project / footage_mod.clip_name(camera)).write_bytes(b"mp4")
    (project / footage_mod.MARKER_NAME).write_text(
        json.dumps(
            {
                "project_name": "site-42",
                "seconds": seconds,
                "recorded_utc": "2026-09-25T20:15:00Z",
                "cameras": [
                    {"id": c.id, "mac": c.mac, "file": footage_mod.clip_name(c)}
                    for c in enabled
                ],
            }
        ),
        encoding="utf-8",
    )
    return project


class FakeFootage:
    """Stands in for footage.record/has_room and the prompt seam."""

    def __init__(
        self, monkeypatch, *, answer="", room=True, failed=(), cancelled=False
    ):
        self.prompts = []
        self.record_calls = []
        self.room_calls = []

        def fake_input(prompt):
            self.prompts.append(prompt)
            return answer

        def fake_has_room(root, count):
            self.room_calls.append((root, count))
            needed = count * footage_mod.MIN_FREE_BYTES_PER_CAMERA
            return (True, needed * 10, needed) if room else (False, 1 << 29, needed)

        def fake_record(cameras, **kwargs):
            self.record_calls.append((list(cameras), kwargs))
            kwargs["on_progress"](0.5)
            project = kwargs["project_dir"]
            bad = list(cameras) if cancelled else list(failed)
            return footage_mod.RecordResult(
                project_dir=project,
                recorded=[
                    project / footage_mod.clip_name(c) for c in cameras if c not in bad
                ],
                failed=bad,
                cancelled=cancelled,
            )

        monkeypatch.setattr(step4, "_INPUT", fake_input)
        monkeypatch.setattr(footage_mod, "has_room", fake_has_room)
        monkeypatch.setattr(footage_mod, "record", fake_record)


def test_footage_init_interactive_prompts_and_records(
    tmp_path, templates, monkeypatch
):
    ctx = _footage_ctx(tmp_path, templates)
    fake = FakeFootage(monkeypatch)
    _complete_wait(monkeypatch)

    result = step4.Step4CalibOutputWiring().preflight(ctx)

    assert result.status is StepStatus.COMPLETE
    project = _footage_project(ctx)
    assert fake.prompts == [
        "Record 5 minutes of calibration footage from 2 cameras into\n"
        f"{project}?\n"
        "Have someone walk through the whole scene while it records.\n"
        "Press Enter to start, or type s to skip: "
    ]
    assert fake.room_calls == [(project.parent, 2)]
    [(cameras, kwargs)] = fake.record_calls
    assert [c.id for c in cameras] == ["c1", "c2"]  # stream_ok False included
    assert kwargs["user"] == "admin"
    assert kwargs["password"] == FOOTAGE_PASSWORD
    assert kwargs["project_dir"] == project
    assert kwargs["project_name"] == "site-42"
    assert kwargs["seconds"] == 300
    assert kwargs["user_prefix"] == ("sudo", "-u", "operator", "-H")
    assert ("task", "recording calibration footage from 2 cameras") in ctx.events
    assert ("percent", 50) in ctx.events
    assert ctx.events[-1] == ("task", "waiting for AMC project site-42 to complete")


def test_footage_settings_come_from_installer_conf(tmp_path, templates, monkeypatch):
    ctx = _footage_ctx(tmp_path, templates)
    custom = tmp_path / "custom footage"
    ctx.conf[step4.CONF_CALIB_FOOTAGE_SECONDS_KEY] = "90"
    ctx.conf[step4.CONF_CALIB_FOOTAGE_DIR_KEY] = str(custom)
    fake = FakeFootage(monkeypatch)
    _complete_wait(monkeypatch)

    assert step4.Step4CalibOutputWiring().preflight(ctx).status is StepStatus.COMPLETE

    assert fake.prompts[0].startswith("Record 1.5 minutes of calibration footage")
    [(_, kwargs)] = fake.record_calls
    assert kwargs["seconds"] == 90
    assert kwargs["project_dir"] == custom / "site-42"


@pytest.mark.parametrize("state_name", ["RUNNING", "COMPLETED", "READY"])
def test_footage_skipped_when_project_is_past_init(
    tmp_path, templates, monkeypatch, state_name
):
    ctx = _footage_ctx(tmp_path, templates, states=(state_name, "COMPLETED"))
    fake = FakeFootage(monkeypatch)
    _complete_wait(monkeypatch)
    assert step4.Step4CalibOutputWiring().preflight(ctx).status is StepStatus.COMPLETE
    assert fake.prompts == [] and fake.record_calls == []


def test_footage_skipped_non_interactive(tmp_path, templates, monkeypatch, capsys):
    ctx = _footage_ctx(tmp_path, templates, non_interactive=True)
    fake = FakeFootage(monkeypatch)
    monkeypatch.setattr(
        waitui, "wait_until", lambda predicate, **kwargs: waitui.WaitOutcome.SKIPPED
    )
    result = step4.Step4CalibOutputWiring().preflight(ctx)
    assert result.status is StepStatus.USER_ACTION_REQUIRED
    assert fake.prompts == [] and fake.record_calls == [] and fake.room_calls == []
    assert capsys.readouterr().err.count("--non-interactive") == 1


def test_footage_complete_set_skips_without_prompt(tmp_path, templates, monkeypatch):
    ctx = _footage_ctx(tmp_path, templates)
    _write_complete_set(ctx)
    fake = FakeFootage(monkeypatch)
    _complete_wait(monkeypatch)
    assert step4.Step4CalibOutputWiring().preflight(ctx).status is StepStatus.COMPLETE
    assert fake.prompts == [] and fake.record_calls == []


def test_footage_set_for_other_clip_length_is_rerecorded(
    tmp_path, templates, monkeypatch
):
    ctx = _footage_ctx(tmp_path, templates)
    _write_complete_set(ctx, seconds=120)
    fake = FakeFootage(monkeypatch)
    _complete_wait(monkeypatch)
    assert step4.Step4CalibOutputWiring().preflight(ctx).status is StepStatus.COMPLETE
    assert len(fake.record_calls) == 1


@pytest.mark.parametrize("answer", ["s", " S "])
def test_footage_s_answer_skips(tmp_path, templates, monkeypatch, answer):
    ctx = _footage_ctx(tmp_path, templates)
    fake = FakeFootage(monkeypatch, answer=answer)
    _complete_wait(monkeypatch)
    assert step4.Step4CalibOutputWiring().preflight(ctx).status is StepStatus.COMPLETE
    assert len(fake.prompts) == 1 and fake.record_calls == []


def test_footage_insufficient_space_needs_action_before_recording(
    tmp_path, templates, monkeypatch
):
    ctx = _footage_ctx(tmp_path, templates)
    fake = FakeFootage(monkeypatch, room=False)
    _complete_wait(monkeypatch)
    result = step4.Step4CalibOutputWiring().preflight(ctx)
    assert result.status is StepStatus.USER_ACTION_REQUIRED
    assert str(_footage_project(ctx).parent) in result.message
    assert "0.5 GiB" in result.message and "2.0 GiB" in result.message
    assert fake.prompts == [] and fake.record_calls == []


def test_footage_failed_camera_needs_action_naming_it(
    tmp_path, templates, monkeypatch
):
    ctx = _footage_ctx(tmp_path, templates)
    FakeFootage(monkeypatch, failed=[FOOTAGE_CAMERAS[1]])
    waits = []
    monkeypatch.setattr(waitui, "wait_until", lambda *a, **k: waits.append(1))
    result = step4.Step4CalibOutputWiring().preflight(ctx)
    assert result.status is StepStatus.USER_ACTION_REQUIRED
    assert "c2 (169.254.1.11)" in result.message
    assert "c1" not in result.message
    assert "activated" in result.user_actions[0].text
    assert "CAM_PASSWORD" in result.user_actions[0].text
    assert waits == []


def test_footage_cancel_needs_action_with_wait_hints(
    tmp_path, templates, monkeypatch
):
    ctx = _footage_ctx(tmp_path, templates)
    FakeFootage(monkeypatch, cancelled=True)
    waits = []
    monkeypatch.setattr(waitui, "wait_until", lambda *a, **k: waits.append(1))
    result = step4.Step4CalibOutputWiring().preflight(ctx)
    assert result.status is StepStatus.USER_ACTION_REQUIRED
    assert "cancelled" in result.message
    assert result.user_actions == step4._wait_hints(
        ctx, step4.resolve_project_inputs(ctx)[0]
    )
    assert waits == []


def test_footage_success_logs_upload_hint(tmp_path, templates, monkeypatch, capsys):
    ctx = _footage_ctx(tmp_path, templates)
    FakeFootage(monkeypatch)
    _complete_wait(monkeypatch)
    assert step4.Step4CalibOutputWiring().preflight(ctx).status is StepStatus.COMPLETE
    err = capsys.readouterr().err
    assert str(_footage_project(ctx)) in err
    assert "Upload these 2 files at the AMC Video Upload step." in err


def test_wait_hints_lead_with_upload_action_when_set_complete(tmp_path, templates):
    ctx = _footage_ctx(tmp_path, templates)
    inputs = step4.resolve_project_inputs(ctx)[0]
    without = step4._wait_hints(ctx, inputs)
    assert not any("Upload" in action.text for action in without)
    project = _write_complete_set(ctx)
    hints = step4._wait_hints(ctx, inputs)
    assert hints[0].text == (
        f"Upload the 2 clips in {project} at the AMC Video Upload step."
    )
    assert hints[1:] == without


def test_successful_ingest_deletes_owned_footage(tmp_path, templates):
    ctx = _footage_ctx(
        tmp_path, templates, states=("COMPLETED",), non_interactive=True
    )
    project = _write_complete_set(ctx)
    assert step4.Step4CalibOutputWiring().run(ctx).status is StepStatus.COMPLETE
    assert not project.exists()
    assert not project.parent.exists()  # the emptied footage root goes too


def test_successful_ingest_keeps_unmarked_directory(tmp_path, templates):
    ctx = _footage_ctx(
        tmp_path, templates, states=("COMPLETED",), non_interactive=True
    )
    project = _footage_project(ctx)
    project.mkdir(parents=True)
    ctx.conf[step4.CONF_CALIB_FOOTAGE_DIR_KEY] = str(project.parent)
    (project / "operator-notes.txt").write_text("keep", encoding="utf-8")
    assert step4.Step4CalibOutputWiring().run(ctx).status is StepStatus.COMPLETE
    assert (project / "operator-notes.txt").is_file()


def test_failed_ingest_keeps_footage(tmp_path, templates):
    ctx = _footage_ctx(
        tmp_path, templates, runner=Runner(download_rc=22), non_interactive=True
    )
    project = _write_complete_set(ctx)
    assert step4.Step4CalibOutputWiring().run(ctx).status is StepStatus.FAILED
    assert (project / footage_mod.MARKER_NAME).is_file()


@pytest.mark.parametrize(
    "outcome", [waitui.WaitOutcome.TIMEOUT, waitui.WaitOutcome.CANCELLED]
)
def test_unfinished_wait_keeps_footage_and_hints_upload(
    tmp_path, templates, monkeypatch, outcome
):
    ctx = _footage_ctx(tmp_path, templates, states=("RUNNING",))
    project = _write_complete_set(ctx)
    monkeypatch.setattr(waitui, "wait_until", lambda *a, **k: outcome)
    result = step4.Step4CalibOutputWiring().preflight(ctx)
    assert result.status is StepStatus.USER_ACTION_REQUIRED
    assert result.user_actions[0].text.startswith("Upload the 2 clips")
    assert (project / footage_mod.MARKER_NAME).is_file()


def test_error_state_keeps_footage(tmp_path, templates):
    ctx = _footage_ctx(tmp_path, templates, states=("ERROR",))
    project = _write_complete_set(ctx)
    assert step4.handle_ingest_subcommand([], ctx) == 1
    assert step4.Step4CalibOutputWiring().preflight(ctx).status is StepStatus.FAILED
    assert (project / footage_mod.MARKER_NAME).is_file()


def test_ingest_subcommand_deletes_owned_footage(tmp_path, templates):
    ctx = _footage_ctx(
        tmp_path, templates, states=("COMPLETED",), non_interactive=True
    )
    project = _write_complete_set(ctx)
    assert step4.handle_ingest_subcommand(["--project", "site-42"], ctx) == 0
    assert not project.exists()


def test_footage_deletion_failure_never_fails_ingest(
    tmp_path, templates, monkeypatch
):
    ctx = _footage_ctx(
        tmp_path, templates, states=("COMPLETED",), non_interactive=True
    )

    _write_complete_set(ctx)

    def boom(project_dir):
        raise PermissionError("denied")

    monkeypatch.setattr(footage_mod, "delete_if_owned", boom)
    assert step4.Step4CalibOutputWiring().run(ctx).status is StepStatus.COMPLETE


@pytest.mark.parametrize(
    "fake_kwargs",
    [{}, {"room": False}, {"failed": [FOOTAGE_CAMERAS[0]]}, {"cancelled": True}],
)
def test_footage_password_never_in_messages(
    tmp_path, templates, monkeypatch, capsys, fake_kwargs
):
    ctx = _footage_ctx(tmp_path, templates)
    fake = FakeFootage(monkeypatch, **fake_kwargs)
    _complete_wait(monkeypatch)
    result = step4.Step4CalibOutputWiring().preflight(ctx)
    captured = capsys.readouterr()
    texts = [result.message or "", captured.err, captured.out, *fake.prompts]
    texts += [a.text + (a.command or "") for a in result.user_actions or []]
    texts += [str(event) for event in ctx.events]
    assert not any(FOOTAGE_PASSWORD in text for text in texts)


def test_recording_persists_resolved_footage_root_when_unset(
    tmp_path, templates, monkeypatch
):
    ctx = _footage_ctx(tmp_path, templates)
    persisted = []
    monkeypatch.setattr(
        step4.config_mod, "persist_value", lambda *args: persisted.append(args)
    )
    FakeFootage(monkeypatch)
    _complete_wait(monkeypatch)
    assert step4.Step4CalibOutputWiring().preflight(ctx).status is StepStatus.COMPLETE
    root = str(_footage_project(ctx).parent)
    assert persisted == [(ctx.install_dir, "CALIB_FOOTAGE_DIR", root)]
    assert ctx.conf["CALIB_FOOTAGE_DIR"] == root


def test_recording_persists_to_installer_conf(tmp_path, templates, monkeypatch):
    ctx = _footage_ctx(tmp_path, templates)
    FakeFootage(monkeypatch)
    _complete_wait(monkeypatch)
    assert step4.Step4CalibOutputWiring().preflight(ctx).status is StepStatus.COMPLETE
    conf_text = (ctx.install_dir / config_mod.CONF_FILENAME).read_text(encoding="utf-8")
    assert f"CALIB_FOOTAGE_DIR={_footage_project(ctx).parent}" in conf_text


def test_recording_leaves_existing_footage_dir_alone(tmp_path, templates, monkeypatch):
    ctx = _footage_ctx(tmp_path, templates)
    custom = str(tmp_path / "custom footage")
    ctx.conf[step4.CONF_CALIB_FOOTAGE_DIR_KEY] = custom
    persisted = []
    monkeypatch.setattr(
        step4.config_mod, "persist_value", lambda *args: persisted.append(args)
    )
    fake = FakeFootage(monkeypatch)
    _complete_wait(monkeypatch)
    assert step4.Step4CalibOutputWiring().preflight(ctx).status is StepStatus.COMPLETE
    assert persisted == []
    assert ctx.conf[step4.CONF_CALIB_FOOTAGE_DIR_KEY] == custom
    assert fake.record_calls[0][1]["project_dir"] == pathlib.Path(custom) / "site-42"


def test_skipped_recording_does_not_persist_footage_root(
    tmp_path, templates, monkeypatch
):
    ctx = _footage_ctx(tmp_path, templates)
    persisted = []
    monkeypatch.setattr(
        step4.config_mod, "persist_value", lambda *args: persisted.append(args)
    )
    FakeFootage(monkeypatch, answer="s")
    _complete_wait(monkeypatch)
    assert step4.Step4CalibOutputWiring().preflight(ctx).status is StepStatus.COMPLETE
    assert persisted == []
    assert step4.CONF_CALIB_FOOTAGE_DIR_KEY not in ctx.conf


def test_root_ingest_deletes_footage_under_persisted_root(tmp_path, templates):
    ctx = _footage_ctx(
        tmp_path, templates, states=("COMPLETED",), non_interactive=True
    )
    project = _write_complete_set(ctx)
    # The systemd timer runs `ingest` as root: no SUDO_USER, home is /root.
    ctx.user.name = "root"
    ctx.user.home = pathlib.Path("/root")
    assert step4.handle_ingest_subcommand(["--project", "site-42"], ctx) == 0
    assert not project.exists()


def test_ingest_without_persisted_root_never_attempts_deletion(
    tmp_path, templates, monkeypatch
):
    ctx = _footage_ctx(
        tmp_path, templates, states=("COMPLETED",), non_interactive=True
    )
    project = _write_complete_set(ctx, persist=False)
    calls = []
    monkeypatch.setattr(footage_mod, "delete_if_owned", lambda p: calls.append(p))
    assert step4.handle_ingest_subcommand(["--project", "site-42"], ctx) == 0
    assert step4.Step4CalibOutputWiring().run(ctx).status is StepStatus.COMPLETE
    assert calls == []
    assert (project / footage_mod.MARKER_NAME).is_file()


# -- edit protection (managed_files) ----------------------------------------


def _deepstream_files(ctx):
    root = ctx.install_dir / "deepstream"
    return {
        path: path.stat().st_mtime_ns
        for path in root.rglob("*")
        if path.is_file()
        and step4.REID_MODEL_NAME not in path.name
        and step4.BODYPOSE_MODEL_NAME not in path.name
    }


def _tracker_path(ctx):
    return ctx.install_dir / "deepstream" / step4.TRACKER_YAML_TEMPLATE_NAME


def test_first_install_writes_and_records_every_rendered_file(tmp_path, templates):
    ctx = _ctx(tmp_path, templates, non_interactive=True)
    assert step4.Step4CalibOutputWiring().run(ctx).status is StepStatus.COMPLETE
    recorded = managed_files.Manifest().load()
    deepstream = ctx.install_dir / "deepstream"
    expected = {
        deepstream / step4.TRACKER_YAML_TEMPLATE_NAME,
        deepstream / step4.RENDERED_APP_CONFIG_NAME,
        deepstream / step4.INFER_PRIMARY_TEMPLATE_NAME,
        deepstream / step4.MSGCONV_TEMPLATE_NAME,
        deepstream / step4.CAMERA_MODEL_DIR_NAME / "camInfo_00.yml",
        deepstream / step4.MV3DT_DIR_NAME / step4.PUB_SUB_CONFIG_NAME,
        deepstream / step4.MV3DT_DIR_NAME / step4.MQTT_CONFIG_NAME,
        systemd.UNIT_DIR / "mv3dt-ingest-site-42.service",
        systemd.UNIT_DIR / "mv3dt-ingest-site-42.timer",
    }
    assert {pathlib.Path(p) for p in recorded} == expected
    assert all(value is not None for value in recorded.values())
    assert not list(deepstream.rglob("*" + managed_files.NEW_SUFFIX))


def test_operator_edit_is_kept_and_new_version_offered(tmp_path, templates, capsys):
    ctx = _ctx(tmp_path, templates, non_interactive=True)
    step = step4.Step4CalibOutputWiring()
    assert step.run(ctx).status is StepStatus.COMPLETE
    tracker = _tracker_path(ctx)
    edited = tracker.read_text() + "# operator tuning\n"
    tracker.write_text(edited)
    (templates / step4.TRACKER_YAML_TEMPLATE_NAME).write_text(
        (templates / step4.TRACKER_YAML_TEMPLATE_NAME).read_text() + "# new release\n"
    )
    capsys.readouterr()
    assert step.run(ctx).status is StepStatus.COMPLETE
    assert tracker.read_text() == edited
    pending = managed_files.pending_path(tracker)
    assert pending.read_text().endswith("# new release\n")
    err = capsys.readouterr().err
    assert "operator edited" in err and str(pending) in err


def test_untouched_rendered_file_is_updated(tmp_path, templates):
    ctx = _ctx(tmp_path, templates, non_interactive=True)
    step = step4.Step4CalibOutputWiring()
    assert step.run(ctx).status is StepStatus.COMPLETE
    infer = templates / step4.INFER_PRIMARY_TEMPLATE_NAME
    infer.write_text(infer.read_text() + "# new release\n")
    assert step.run(ctx).status is StepStatus.COMPLETE
    rendered = ctx.install_dir / "deepstream" / step4.INFER_PRIMARY_TEMPLATE_NAME
    assert rendered.read_text().endswith("# new release\n")
    assert not managed_files.pending_path(rendered).exists()


def test_pre_manifest_existing_differing_file_is_kept(tmp_path, templates):
    ctx = _ctx(tmp_path, templates, non_interactive=True)
    tracker = _tracker_path(ctx)
    tracker.parent.mkdir(parents=True)
    tracker.write_text("live tuned tracker\n")
    model = ctx.install_dir / "deepstream" / step4.CAMERA_MODEL_DIR_NAME / "camInfo_00.yml"
    model.parent.mkdir(parents=True)
    model.write_text("live camera model\n")
    assert step4.Step4CalibOutputWiring().run(ctx).status is StepStatus.COMPLETE
    assert tracker.read_text() == "live tuned tracker\n"
    assert model.read_text() == "live camera model\n"
    assert managed_files.pending_path(tracker).is_file()
    assert managed_files.pending_path(model).is_file()
    states = {e.path: e.state for e in managed_files.status(managed_files.Manifest())}
    assert states[tracker] is managed_files.FileState.PENDING_NEW
    assert states[model] is managed_files.FileState.PENDING_NEW


def test_ingest_with_unchanged_archive_writes_nothing(tmp_path, templates, capsys):
    ctx = _ctx(tmp_path, templates, non_interactive=True)
    assert step4.Step4CalibOutputWiring().run(ctx).status is StepStatus.COMPLETE
    infer = templates / step4.INFER_PRIMARY_TEMPLATE_NAME
    infer.write_text(infer.read_text() + "# new release\n")
    before = _deepstream_files(ctx)
    manifest_before = managed_files.Manifest().path.read_bytes()
    capsys.readouterr()
    assert step4.handle_ingest_subcommand(["--project", "site-42"], ctx) == 0
    assert _deepstream_files(ctx) == before
    assert managed_files.Manifest().path.read_bytes() == manifest_before
    assert "nothing to render" in capsys.readouterr().err


def test_ingest_with_changed_archive_renders(tmp_path, templates):
    runner = Runner()
    ctx = _ctx(tmp_path, templates, runner=runner, non_interactive=True)
    assert step4.Step4CalibOutputWiring().run(ctx).status is StepStatus.COMPLETE
    infer = templates / step4.INFER_PRIMARY_TEMPLATE_NAME
    infer.write_text(infer.read_text() + "# new release\n")
    runner.archive = _zip_bytes(
        {
            "transforms.yml": "transforms: [1]\n",
            "camInfo/camInfo_00.yml": AMC_CAMINFO,
        }
    )
    assert step4.handle_ingest_subcommand(["--project", "site-42"], ctx) == 0
    rendered = ctx.install_dir / "deepstream" / step4.INFER_PRIMARY_TEMPLATE_NAME
    assert rendered.read_text().endswith("# new release\n")


def test_ingest_unchanged_archive_renders_missing_targets(tmp_path, templates):
    ctx = _ctx(tmp_path, templates, non_interactive=True)
    assert step4.Step4CalibOutputWiring().run(ctx).status is StepStatus.COMPLETE
    _tracker_path(ctx).unlink()
    assert step4.handle_ingest_subcommand(["--project", "site-42"], ctx) == 0
    assert _tracker_path(ctx).is_file()


def test_operator_edit_without_new_render_gets_no_pending(tmp_path, templates):
    ctx = _ctx(tmp_path, templates, non_interactive=True)
    step = step4.Step4CalibOutputWiring()
    assert step.run(ctx).status is StepStatus.COMPLETE
    tracker = _tracker_path(ctx)
    edited = tracker.read_text() + "# operator tuning\n"
    tracker.write_text(edited)
    assert step.run(ctx).status is StepStatus.COMPLETE
    assert tracker.read_text() == edited
    assert not managed_files.pending_path(tracker).exists()


# -- generated per-install files: fusion, dewarpers, viewer, pose model -------

LIVE_FIXTURE_DIR = pathlib.Path(__file__).resolve().parent / "fixtures" / "live_baseline"


def _radial(camera_id, k1, k2, *, model="radial"):
    return {
        "camera_id": camera_id,
        "width": 1920,
        "height": 1080,
        "cx": 960.0,
        "cy": 540.0,
        "committed": {"model": model, "k1": k1, "k2": k2},
    }


def _two_camera_conf(tmp_path, *, viewer="off", location="site-42"):
    conf = _conf(tmp_path)
    pathlib.Path(conf[config_mod.CAMERAS_FILE_KEY]).write_text(
        "cameras:\n"
        "  - id: cam1\n"
        "    ip: 192.0.2.10\n"
        "    rtsp_path: /Streaming/Channels/101\n"
        "    enabled: true\n"
        "  - id: cam2\n"
        "    ip: 192.0.2.11\n"
        "    rtsp_path: /Streaming/Channels/101\n"
        "    enabled: true\n",
        encoding="utf-8",
    )
    conf[step4.CONF_LOCATION_ID_KEY] = location
    conf[step4.CONF_PROJECT_NAME_KEY] = location
    conf[step4.CONF_CAM_USER_KEY] = "user"
    conf[step4.CONF_CAM_PASSWORD_KEY] = "password"
    conf[config_mod.GATE_VIEWER_STREAM] = viewer
    return conf


def _two_camera_archive():
    return _zip_bytes(
        {
            "transforms.yml": "transforms: []\n",
            "camInfo/camInfo_00.yml": AMC_CAMINFO,
            "camInfo/camInfo_01.yml": AMC_CAMINFO,
        }
    )


def _two_camera_ctx(tmp_path, templates, *, viewer="off", rectification=None):
    runner = Runner(archive=_two_camera_archive(), rectification=rectification)
    return _ctx(
        tmp_path,
        templates,
        runner=runner,
        conf=_two_camera_conf(tmp_path, viewer=viewer),
        non_interactive=True,
    )


def _sections(text):
    """[(section, [(key, value), ...]), ...] of an INI-style DeepStream
    config, ignoring comments and blank lines."""
    sections = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            sections.append((line[1:-1], []))
            continue
        key, _, value = line.partition("=")
        sections[-1][1].append((key.strip(), value.strip()))
    return sections


@pytest.mark.parametrize(
    "k1, k2, expected",
    [
        # Live cam0. AMC's API shows k2 rounded to 1.0097e-13, which gives
        # 0.66172; the live file's 0.66173 came from the unrounded value.
        (-3.2126e-07, 1.0097e-13, "-0.82243;0.66172;0;0;0"),
        (-3.2126e-07, 1.00972e-13, "-0.82243;0.66173;0;0;0"),
        # Live cam1.
        (-2.9989e-07, 9.0769e-14, "-0.76772;0.59486;0;0;0"),
        (0.0, 0.0, "0;0;0;0;0"),
    ],
)
def test_dewarper_distortion_normalizes_by_focal_length(k1, k2, expected):
    assert step4.dewarper_distortion(k1, k2) == expected


def test_render_dewarper_config_matches_live_format():
    text = step4.render_dewarper_config(_radial("cam_01", -2.9989e-07, 9.0769e-14))
    assert text == (LIVE_FIXTURE_DIR / "dewarper" / "dewarper_cam1.txt").read_text()


def test_render_dewarper_config_keeps_fractional_principal_point():
    camera = _radial("cam_00", -3.2126e-07, 1.0097e-13)
    camera["cx"] = 961.5
    assert "src-x0=961.5\n" in step4.render_dewarper_config(camera)


@pytest.mark.parametrize(
    "camera",
    [
        {"camera_id": "cam_00", "width": 1920, "height": 1080, "cx": 960, "cy": 540},
        _radial("cam_00", 1e-7, 1e-13, model="fisheye"),
        {"camera_id": "cam_00", "committed": {"model": "radial", "k1": 1e-7}},
    ],
)
def test_render_dewarper_config_rejects_unusable_entries(camera):
    with pytest.raises(ValueError):
        step4.render_dewarper_config(camera)


def test_pub_sub_config_subscribes_every_camera_to_its_peers():
    data = step4.yaml.safe_load(step4.render_pub_sub_config(3))
    topic = "127.0.0.1:1883;/trck/cam{}".format
    assert data["pubBrokerTopicStr"] == [topic(1), topic(2), topic(3)]
    assert data["subPeerBrokerTopicStrs"] == [
        [topic(2), topic(3)],
        [topic(1), topic(3)],
        [topic(1), topic(2)],
    ]


def test_pub_sub_config_for_two_cameras_matches_live():
    rendered = step4.render_pub_sub_config(2)
    live = (LIVE_FIXTURE_DIR / "mv3dt" / step4.PUB_SUB_CONFIG_NAME).read_text()
    assert step4.yaml.safe_load(rendered) == step4.yaml.safe_load(live)


def test_mqtt_config_is_the_anonymous_sample_with_threading_off():
    lines = step4.MQTT_CONFIG_TEXT.splitlines()
    assert lines[0] == "[message-broker]"
    assert "set-threaded = 0" in lines
    assert "username = user" in lines
    assert "password = password" in lines


@pytest.mark.parametrize(
    "count, expected",
    [
        (1, (1, 1, 1920, 540)),
        (2, (1, 2, 1920, 540)),
        (3, (1, 3, 1920, 360)),
        (4, (1, 4, 1920, 270)),
        (6, (2, 4, 1920, 540)),
        (8, (2, 4, 1920, 540)),
    ],
)
def test_tiled_display_grid(count, expected):
    assert step4.tiled_display_grid(count) == expected


def _render_app(templates, **kwargs):
    return step4.render_app_config(
        (templates / step4.APP_CONFIG_TEMPLATE_NAME).read_text(encoding="utf-8"),
        cam_user="u",
        cam_password="p",
        location_id="site-42",
        cameras=[
            SimpleNamespace(ip="192.0.2.10", rtsp_path="/s"),
            SimpleNamespace(ip="192.0.2.11", rtsp_path="/s"),
        ],
        template_path="template",
        calibration_dir="calibration/site-42",
        **kwargs,
    )


def test_viewer_off_adds_no_viewer_sections(templates):
    sections = dict(_sections(_render_app(templates)))
    assert sections["tiled-display"] == [("enable", "0")]
    assert "osd" not in sections and "sink2" not in sections
    assert not any(name.startswith("dewarper") for name in sections)


def test_viewer_on_adds_tiled_display_osd_and_rtsp_sink(templates):
    sections = dict(_sections(_render_app(templates, viewer_stream=True)))
    assert sections["tiled-display"] == [
        ("enable", "1"),
        ("rows", "1"),
        ("columns", "2"),
        ("width", "1920"),
        ("height", "540"),
    ]
    assert dict(sections["sink2"]) == {
        "enable": "1",
        "type": "4",
        "codec": "1",
        "enc-type": "0",
        "sync": "0",
        "bitrate": "4000000",
        "rtsp-port": "8554",
        "udp-port": "5400",
    }
    assert dict(sections["osd"])["enable"] == "1"


def test_run_writes_fusion_files_and_no_dewarper_without_rectification(
    tmp_path, templates, capsys
):
    ctx = _two_camera_ctx(tmp_path, templates)
    assert step4.Step4CalibOutputWiring().run(ctx).status is StepStatus.COMPLETE
    deepstream = ctx.install_dir / "deepstream"
    pub_sub = step4.yaml.safe_load(
        (deepstream / step4.MV3DT_DIR_NAME / step4.PUB_SUB_CONFIG_NAME).read_text()
    )
    assert len(pub_sub["pubBrokerTopicStr"]) == 2
    assert (
        deepstream / step4.MV3DT_DIR_NAME / step4.MQTT_CONFIG_NAME
    ).read_text() == step4.MQTT_CONFIG_TEXT
    assert not (deepstream / step4.DEWARPER_DIR_NAME).exists()
    rendered = (deepstream / step4.RENDERED_APP_CONFIG_NAME).read_text()
    assert "[dewarper" not in rendered and "[sink2]" not in rendered
    assert "rendering no dewarpers" in capsys.readouterr().err


def test_run_skips_dewarper_for_a_camera_without_radial_model(
    tmp_path, templates, capsys
):
    rectification = {
        "cameras": [
            _radial("cam_00", -3.2126e-07, 1.0097e-13),
            {"camera_id": "cam_01", "width": 1920, "height": 1080, "committed": None},
        ]
    }
    ctx = _two_camera_ctx(tmp_path, templates, rectification=rectification)
    assert step4.Step4CalibOutputWiring().run(ctx).status is StepStatus.COMPLETE
    dewarper_dir = ctx.install_dir / "deepstream" / step4.DEWARPER_DIR_NAME
    assert sorted(p.name for p in dewarper_dir.iterdir()) == ["dewarper_cam0.txt"]
    sections = dict(
        _sections(
            (ctx.install_dir / "deepstream" / step4.RENDERED_APP_CONFIG_NAME).read_text()
        )
    )
    assert dict(sections["dewarper0"])["config-file"] == str(
        dewarper_dir / "dewarper_cam0.txt"
    )
    assert "dewarper1" not in sections
    assert "cam_01: no committed rectification" in capsys.readouterr().err


def test_rectification_request_uses_project_endpoint(tmp_path, templates):
    ctx = _two_camera_ctx(tmp_path, templates)
    assert step4.Step4CalibOutputWiring().run(ctx).status is StepStatus.COMPLETE
    urls = [args[-1] for args, _ in ctx.runner.calls if args and args[0] == "curl"]
    assert "http://localhost:8000/v1/rectification/project-123/cameras" in urls


def test_run_downloads_bodypose_model_once(tmp_path, templates):
    runner = Runner()
    ctx = _ctx(tmp_path, templates, runner=runner, non_interactive=True)
    step = step4.Step4CalibOutputWiring()
    assert step.run(ctx).status is StepStatus.COMPLETE
    model = ctx.install_dir / step4.BODYPOSE_RELATIVE_DIR / step4.BODYPOSE_MODEL_NAME
    assert model.read_bytes() == FAKE_BODYPOSE
    assert step.run(ctx).status is StepStatus.COMPLETE
    fetches = [
        a for a, _ in runner.calls if a and a[0] == "curl" and a[-1] == step4.BODYPOSE_MODEL_URL
    ]
    assert len(fetches) == 1


@pytest.mark.parametrize(
    "runner_kwargs, message",
    [
        ({"bodypose_rc": 22}, "BodyPose3DNet model download failed"),
        ({"bodypose_bytes": b"short"}, "expected"),
        ({"bodypose_bytes": b"x" * len(FAKE_BODYPOSE)}, "SHA-256"),
    ],
)
def test_bodypose_download_is_validated(tmp_path, templates, runner_kwargs, message):
    ctx = _ctx(tmp_path, templates, runner=Runner(**runner_kwargs), non_interactive=True)
    result = step4.Step4CalibOutputWiring().run(ctx)
    assert result.status is StepStatus.FAILED
    assert message in result.message
    model_dir = ctx.install_dir / step4.BODYPOSE_RELATIVE_DIR
    assert not any(model_dir.iterdir())


def test_verify_requires_bodypose_model(tmp_path, templates):
    ctx = _ctx(tmp_path, templates, non_interactive=True)
    step = step4.Step4CalibOutputWiring()
    assert step.run(ctx).status is StepStatus.COMPLETE
    (ctx.install_dir / step4.BODYPOSE_RELATIVE_DIR / step4.BODYPOSE_MODEL_NAME).unlink()
    result = step.verify(ctx)
    assert result.status is StepStatus.FAILED
    assert "BodyPose3DNet" in result.message


def test_bodypose_source_matches_the_live_model():
    """The live file is 70575749 bytes: NGC version
    deployable_accuracy_onnx_1.0 (deployable_accuracy_v1.0 is 20 bytes
    larger)."""
    assert "/versions/deployable_accuracy_onnx_1.0/" in step4.BODYPOSE_MODEL_URL
    assert step4.BODYPOSE_MODEL_URL.endswith("/" + step4.BODYPOSE_MODEL_NAME)


def test_golden_render_matches_live_workstation(tmp_path, templates, monkeypatch):
    """Render the full config set for the live two-camera layout with the
    viewer on and compare it to a sanitized copy of the live files, apart
    from comments, camera URIs and the install-dir prefix."""
    rectification = json.loads(
        (LIVE_FIXTURE_DIR / "rectification_cameras.json").read_text()
    )
    ctx = _two_camera_ctx(tmp_path, templates, viewer="on", rectification=rectification)
    assert step4.Step4CalibOutputWiring().run(ctx).status is StepStatus.COMPLETE
    deepstream = ctx.install_dir / "deepstream"
    live_prefix = "/opt/mv3dt"

    def rendered(name):
        return (deepstream / name).read_text(encoding="utf-8").replace(
            str(ctx.install_dir), live_prefix
        )

    def live(name):
        return (LIVE_FIXTURE_DIR / name).read_text(encoding="utf-8")

    def without_uris(sections):
        return [
            (name, [(k, "<uri>" if k == "uri" else v) for k, v in pairs])
            for name, pairs in sections
        ]

    app_rendered = _sections(rendered(step4.RENDERED_APP_CONFIG_NAME))
    app_live = _sections(live(step4.RENDERED_APP_CONFIG_NAME))
    assert without_uris(app_rendered) == without_uris(app_live)
    uris = [dict(pairs).get("uri") for name, pairs in app_rendered[:4] if name.startswith("source")]
    assert uris[:2] == [
        "rtsp://user:password@192.0.2.10:554/Streaming/Channels/101",
        "rtsp://user:password@192.0.2.11:554/Streaming/Channels/101",
    ]

    assert step4._parse_tracker_yaml(
        rendered(step4.TRACKER_YAML_TEMPLATE_NAME)
    ) == step4._parse_tracker_yaml(live(step4.TRACKER_YAML_TEMPLATE_NAME))
    assert _sections(rendered(step4.INFER_PRIMARY_TEMPLATE_NAME)) == _sections(
        live(step4.INFER_PRIMARY_TEMPLATE_NAME)
    )

    pub_sub = f"{step4.MV3DT_DIR_NAME}/{step4.PUB_SUB_CONFIG_NAME}"
    assert step4.yaml.safe_load(rendered(pub_sub)) == step4.yaml.safe_load(live(pub_sub))

    # The snapshot's password value is the collector's redaction marker.
    mqtt = f"{step4.MV3DT_DIR_NAME}/{step4.MQTT_CONFIG_NAME}"
    mqtt_rendered = rendered(mqtt).splitlines()
    mqtt_live = live(mqtt).splitlines()
    assert [l for l in mqtt_rendered if not l.startswith("password")] == [
        l for l in mqtt_live if not l.startswith("password")
    ]
    assert sum(l.startswith("password = ") for l in mqtt_rendered) == 1
    assert "username = user" in mqtt_rendered and "set-threaded = 0" in mqtt_rendered

    for index in (0, 1):
        name = f"{step4.DEWARPER_DIR_NAME}/dewarper_cam{index}.txt"
        assert rendered(name) == live(name)
