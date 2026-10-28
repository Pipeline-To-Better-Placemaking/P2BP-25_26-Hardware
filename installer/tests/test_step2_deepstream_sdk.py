"""Tests for mv3dt_installer.steps.step2_deepstream_sdk
(installer/plan/STEP-2-DEEPSTREAM-SDK.md).

Run from installer/: `python3 -m pytest tests/test_step2_deepstream_sdk.py -v`

No test spawns a real apt/dpkg/curl/docker/deepstream-app process or opens a
socket -- every `ctx.run_root`/`ctx.run_as_user` call is served by a
`ScriptedRunner` fake, and every fixed filesystem path the module reads
(`DS_SDK_DIR`, `DS_SDK_SYMLINK`, `PROFILE_D_PATH`) is monkeypatched at a
`tmp_path`, the same discipline `test_systemd.py` uses for `UNIT_DIR`.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
import zipfile
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from mv3dt_installer import config as config_mod  # noqa: E402
from mv3dt_installer import logs, report  # noqa: E402
from mv3dt_installer.steps import StepStatus  # noqa: E402
from mv3dt_installer.steps import step2_deepstream_sdk as step2  # noqa: E402

PINS_OK = {
    "nvidia-smi": ("595.58.03\n", 0),
}
@pytest.fixture(autouse=True)
def _reset_transcript_state():
    logs._transcript_path = None
    yield
    logs._transcript_path = None


@pytest.fixture(autouse=True)
def _force_no_colour(monkeypatch):
    monkeypatch.setattr(sys.stderr, "isatty", lambda: False)


@pytest.fixture(autouse=True)
def _supported_arch(monkeypatch):
    """Exercise the supported workstation architecture on every dev host."""
    monkeypatch.setattr(step2.platform, "machine", lambda: "x86_64")


@pytest.fixture(autouse=True)
def _sdk_paths(tmp_path, monkeypatch):
    """Point every fixed filesystem constant at a scratch tree so a bug in
    the code under test can never touch /opt/nvidia or /etc/profile.d."""
    sdk_dir = tmp_path / "opt" / "deepstream-9.1"
    symlink = tmp_path / "opt" / "deepstream"
    profile = tmp_path / "profile.d" / "deepstream.sh"
    monkeypatch.setattr(step2, "DS_SDK_DIR", sdk_dir)
    monkeypatch.setattr(step2, "DS_SDK_SYMLINK", symlink)
    monkeypatch.setattr(step2, "PROFILE_D_PATH", profile)
    return sdk_dir, symlink, profile


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class ScriptedRunner:
    """Stand-in for `ctx.run_root`/`ctx.run_as_user`. Records every call as
    the `args` tuple it was invoked with, and resolves a canned
    `CompletedProcess` by the first matching registered rule, else a
    configurable default."""

    def __init__(self, *, default_returncode: int = 1, default_stdout: str = "", default_stderr: str = ""):
        # Fails ("not found"/"not installed") by default: every detection
        # probe in this module (dpkg -s, docker image inspect, docker
        # info, ...) treats a nonzero exit as "absent", so an
        # unconfigured runner reads as "nothing is installed yet" rather
        # than accidentally satisfying an idempotent-reinstall branch a
        # test didn't ask for. Tests that need an action (curl, apt-get,
        # docker pull/login) to succeed register that explicitly.
        self.calls: list[tuple] = []
        self._rules: list[tuple] = []
        self.default_returncode = default_returncode
        self.default_stdout = default_stdout
        self.default_stderr = default_stderr

    def when(self, matcher, *, returncode=0, stdout="", stderr="", side_effect=None):
        self._rules.append((matcher, returncode, stdout, stderr, side_effect))
        return self

    def __call__(self, *args, **kwargs):
        self.calls.append(args)
        # Most-recently-added rule wins: lets a test start from a shared
        # "everything passes" runner (e.g. `_passing_pin_runner()`) and
        # override one probe's answer without needing to rebuild the whole
        # rule set.
        for matcher, returncode, stdout, stderr, side_effect in reversed(self._rules):
            if matcher(args):
                if side_effect is not None:
                    side_effect()
                return subprocess.CompletedProcess(list(args), returncode, stdout, stderr)
        return subprocess.CompletedProcess(
            list(args), self.default_returncode, self.default_stdout, self.default_stderr
        )

    def called_with_prefix(self, *prefix) -> bool:
        return any(tuple(call[: len(prefix)]) == prefix for call in self.calls)


class FakeUser:
    name = "op"
    home = pathlib.Path("/home/op")
    uid = 1000
    gid = 1000


class FakeNgc:
    def __init__(self, key="a-fake-ngc-key"):
        self._key = key
        self.configure_calls = 0

    def load_key(self):
        return self._key

    def configure_ngc_cli(self):
        self.configure_calls += 1
        return pathlib.Path("/home/op/.ngc/config")


class FakeContext:
    def __init__(
        self,
        tmp_path,
        *,
        conf=None,
        ngc=None,
        runner_root=None,
        runner_user=None,
        non_interactive=True,
    ):
        self.install_dir = tmp_path / "mv3dt"
        self.install_dir.mkdir(parents=True, exist_ok=True)
        self.conf = conf if conf is not None else {}
        self.user = FakeUser()
        self.log = logs.log
        self.report_installed = report.report_installed
        self.report_already_installed = report.report_already_installed
        self.verify_pinned = report.verify_pinned
        self.ngc = ngc if ngc is not None else FakeNgc()
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
            percent=lambda value, note=None: None,
            line=lambda text: None,
        )

    def run_root(self, *args, **kwargs):
        return self.runner_root(*args, **kwargs)

    def run_as_user(self, *args, **kwargs):
        return self.runner_user(*args, **kwargs)


def _passing_pin_runner() -> ScriptedRunner:
    """A `run_root` double that answers every doc section 2 prereq probe
    with the pinned value."""
    runner = ScriptedRunner()
    runner.when(lambda a: a[:1] == ("nvidia-smi",), stdout="595.58.03\n")
    runner.when(
        lambda a: a[:1] == (step2.CUDA_NVCC_PATH,),
        stdout="Cuda compilation tools, release 13.2, V13.2.1\n",
    )
    runner.when(
        lambda a: a[:2] == ("dpkg-query", "-W")
        and a[-1] == step2.CUDNN_QUERY_PACKAGE,
        stdout=f"{step2.CUDNN_APT_VERSION}\n",
    )
    runner.when(
        lambda a: a[:2] == ("dpkg-query", "-W") and a[-1] == "libnvinfer10",
        stdout="10.16.0.72-1+cuda13.2\n",
    )
    runner.when(lambda a: a[:1] == ("gst-inspect-1.0",), stdout="GStreamer 1.24.2\n")
    return runner


# ---------------------------------------------------------------------------
# detect_method -- doc section 4.2 precedence
# ---------------------------------------------------------------------------


def test_detect_method_explicit_override_wins(tmp_path):
    ctx = FakeContext(tmp_path, conf={"ds_install_method": "tar"})
    method, reason = step2.detect_method(ctx)
    assert method is step2.Method.TAR
    assert "override" in reason


def test_detect_method_unknown_override_is_ignored(tmp_path):
    ctx = FakeContext(tmp_path, conf={"ds_install_method": "bogus"})
    method, _ = step2.detect_method(ctx)
    # Falls through to the product default (host pipeline intent).
    assert method is step2.Method.DEB


def test_detect_method_already_installed_deb_wins_over_tar_marker(tmp_path):
    runner = ScriptedRunner()
    runner.when(lambda a: a[:2] == ("dpkg", "-s"), stdout="Version: 9.1.0-1\n")
    ctx = FakeContext(tmp_path, runner_root=runner)
    method, reason = step2.detect_method(ctx)
    assert method is step2.Method.DEB
    assert "already installed" in reason


def test_detect_method_already_installed_tar(tmp_path, _sdk_paths):
    sdk_dir, _symlink, _profile = _sdk_paths
    sdk_dir.mkdir(parents=True)
    (sdk_dir / "version").write_text("DeepStream 9.1.0\n")

    runner = ScriptedRunner(default_returncode=1)  # dpkg -s: not installed
    ctx = FakeContext(tmp_path, runner_root=runner)
    method, reason = step2.detect_method(ctx)
    assert method is step2.Method.TAR
    assert "already installed" in reason


def test_detect_method_already_installed_docker(tmp_path):
    runner_root = ScriptedRunner(default_returncode=1)
    runner_user = ScriptedRunner()
    runner_user.when(lambda a: a[:3] == ("docker", "image", "inspect"), returncode=0)
    ctx = FakeContext(tmp_path, runner_root=runner_root, runner_user=runner_user)
    method, reason = step2.detect_method(ctx)
    assert method is step2.Method.DOCKER
    assert "already installed" in reason


def test_detect_method_relocatable_requests_tar(tmp_path):
    runner = ScriptedRunner(default_returncode=1)
    ctx = FakeContext(tmp_path, conf={"ds_relocatable": "true"}, runner_root=runner)
    method, reason = step2.detect_method(ctx)
    assert method is step2.Method.TAR
    assert "ds_relocatable=true" in reason


def test_detect_method_docker_when_toolkit_usable_and_host_not_required(tmp_path):
    runner_root = ScriptedRunner(default_returncode=1)
    runner_root.when(lambda a: a[:2] == ("dpkg", "-s") and a[-1] == "nvidia-container-toolkit", returncode=0)
    runner_user = ScriptedRunner(default_returncode=1)
    runner_user.when(lambda a: a[:2] == ("docker", "info"), returncode=0, stdout="Runtimes: nvidia runc\n")
    runner_user.when(lambda a: a == ("which", "nvidia-ctk"), returncode=0)

    ctx = FakeContext(
        tmp_path,
        conf={"ds_host_pipeline_required": "false"},
        runner_root=runner_root,
        runner_user=runner_user,
    )
    method, reason = step2.detect_method(ctx)
    assert method is step2.Method.DOCKER
    assert "Docker" in reason


def test_detect_method_defaults_to_deb_for_host_pipeline_intent(tmp_path):
    runner_root = ScriptedRunner(default_returncode=1)
    runner_user = ScriptedRunner(default_returncode=1)
    ctx = FakeContext(tmp_path, runner_root=runner_root, runner_user=runner_user)
    method, reason = step2.detect_method(ctx)
    assert method is step2.Method.DEB
    assert "host pipeline intent" in reason


def test_detect_method_ambiguous_when_host_not_required_and_docker_unusable(tmp_path):
    runner_root = ScriptedRunner(default_returncode=1)
    runner_user = ScriptedRunner(default_returncode=1)  # docker info fails
    ctx = FakeContext(
        tmp_path,
        conf={"ds_host_pipeline_required": "false"},
        runner_root=runner_root,
        runner_user=runner_user,
    )
    method, reason = step2.detect_method(ctx)
    assert method is None
    assert reason == "ambiguous"


# ---------------------------------------------------------------------------
# run() -- ambiguous-method prompt / non-interactive fallback (doc section 12)
# ---------------------------------------------------------------------------


def _ambiguous_ctx(tmp_path, *, non_interactive: bool = True) -> FakeContext:
    runner_root = ScriptedRunner(default_returncode=1)
    runner_user = ScriptedRunner(default_returncode=1)
    return FakeContext(
        tmp_path,
        conf={"ds_host_pipeline_required": "false"},
        runner_root=runner_root,
        runner_user=runner_user,
        non_interactive=non_interactive,
    )


def test_resolve_method_non_interactive_ambiguous_defaults_to_deb(tmp_path):
    ctx = _ambiguous_ctx(tmp_path, non_interactive=True)
    method, reason = step2._resolve_method(ctx)
    assert method is step2.Method.DEB
    assert "non-interactive default" in reason


def test_resolve_method_interactive_ambiguous_prompts_and_honors_answer(tmp_path, monkeypatch):
    monkeypatch.setattr(step2, "_INPUT", lambda _prompt: "tar")
    ctx = _ambiguous_ctx(tmp_path, non_interactive=False)
    method, reason = step2._resolve_method(ctx)
    assert method is step2.Method.TAR
    assert "operator selection" in reason


def test_prompt_for_method_blank_answer_defaults_to_deb(monkeypatch):
    monkeypatch.setattr(step2, "_INPUT", lambda _prompt: "")
    assert step2._prompt_for_method() is step2.Method.DEB


def test_prompt_for_method_reprompts_on_garbage(monkeypatch):
    answers = iter(["nope", "docker"])
    monkeypatch.setattr(step2, "_INPUT", lambda _prompt: next(answers))
    assert step2._prompt_for_method() is step2.Method.DOCKER


# ---------------------------------------------------------------------------
# run() -- deb branch
# ---------------------------------------------------------------------------


def test_run_deb_already_installed_skips_download(tmp_path):
    runner_root = ScriptedRunner()
    runner_root.when(lambda a: a[:2] == ("dpkg", "-s"), stdout="Version: 9.1.0-1\n")
    runner_user = ScriptedRunner()
    ctx = FakeContext(tmp_path, runner_root=runner_root, runner_user=runner_user)
    _peoplenet_ready(ctx)

    step = step2.Step2DeepStreamSdk()
    result = step.run(ctx)

    assert result.status is StepStatus.COMPLETE
    assert not runner_user.calls  # curl never invoked
    assert ctx.conf["ds_install_method"] == "deb"


def test_run_deb_download_failure_is_user_action_required(tmp_path):
    runner_root = ScriptedRunner(default_returncode=1)  # dpkg -s: not installed
    runner_user = ScriptedRunner(default_returncode=1)  # curl fails
    ctx = FakeContext(tmp_path, runner_root=runner_root, runner_user=runner_user)

    step = step2.Step2DeepStreamSdk()
    result = step.run(ctx)

    assert result.status is StepStatus.USER_ACTION_REQUIRED
    assert result.user_actions
    action = result.user_actions[0]
    assert step2.DEB_ARTIFACT in action.text
    assert "curl" in (action.command or "")
    assert str(ctx.install_dir / "downloads" / "deepstream") == action.path


def test_run_deb_apt_install_failure_is_failed(tmp_path):
    runner_root = ScriptedRunner(default_returncode=1)
    runner_root.when(lambda a: a[:2] == ("apt-get", "install"), returncode=1, stderr="dependency problems")
    runner_user = ScriptedRunner()  # curl is never invoked: artifact is pre-placed below

    ctx = FakeContext(tmp_path, runner_root=runner_root, runner_user=runner_user)
    artifact_dir = ctx.install_dir / "downloads" / "deepstream"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / step2.DEB_ARTIFACT).write_bytes(b"stub")

    step = step2.Step2DeepStreamSdk()
    result = step.run(ctx)

    assert result.status is StepStatus.FAILED
    assert "apt-get install" in result.message
    assert "dependency problems" in result.message


def test_run_deb_fresh_install_reports_installed_and_completes(tmp_path):
    runner_root = ScriptedRunner()
    runner_root.when(lambda a: a[:2] == ("dpkg", "-s"), returncode=1)  # not installed yet
    runner_root.when(lambda a: a[:2] == ("apt-get", "install"), returncode=0)

    ctx = FakeContext(tmp_path, runner_root=runner_root)
    artifact_path = ctx.install_dir / "downloads" / "deepstream" / step2.DEB_ARTIFACT
    ctx.runner_user.when(
        lambda a: a[:1] == ("curl",),
        returncode=0,
        side_effect=lambda: artifact_path.write_bytes(b"stub"),
    )
    _peoplenet_ready(ctx)

    step = step2.Step2DeepStreamSdk()
    result = step.run(ctx)

    assert result.status is StepStatus.COMPLETE
    assert runner_root.called_with_prefix("apt-get", "install", "-y", f"./{step2.DEB_ARTIFACT}")
    assert ctx.conf["ds_install_method"] == "deb"
    # installer.conf was persisted so a resumed run reads back the choice.
    assert config_mod._read_conf(ctx.install_dir / config_mod.CONF_FILENAME)[
        step2.CONF_METHOD_KEY
    ] == "deb"
    # Post-install tail ran (update_rtpmanager.sh, ldconfig, profile.d).
    assert runner_root.called_with_prefix("ldconfig")


# ---------------------------------------------------------------------------
# run() -- docker branch (doc section 5.2 / 12: no fallback once key present)
# ---------------------------------------------------------------------------


def test_run_docker_already_local_skips_login_and_pull(tmp_path):
    runner_user = ScriptedRunner()
    runner_user.when(lambda a: a[:3] == ("docker", "image", "inspect"), returncode=0)
    ctx = FakeContext(tmp_path, conf={"ds_install_method": "docker"}, runner_user=runner_user)
    _peoplenet_ready(ctx)

    step = step2.Step2DeepStreamSdk()
    result = step.run(ctx)

    assert result.status is StepStatus.COMPLETE
    assert not runner_user.called_with_prefix("docker", "login")
    assert not runner_user.called_with_prefix("docker", "pull")


def test_run_docker_login_failure_is_failed_not_user_action(tmp_path):
    runner_user = ScriptedRunner(default_returncode=1)  # image inspect: absent
    runner_user.when(lambda a: len(a) >= 3 and a[2] == "bash", returncode=1, stderr="unauthorized")
    ctx = FakeContext(tmp_path, conf={"ds_install_method": "docker"}, runner_user=runner_user)

    step = step2.Step2DeepStreamSdk()
    result = step.run(ctx)

    assert result.status is StepStatus.FAILED
    assert "docker login" in result.message
    assert "unauthorized" in result.message
    assert result.user_actions == []


def test_run_docker_pull_failure_is_failed(tmp_path):
    runner_user = ScriptedRunner(default_returncode=0)  # image inspect + login + bash succeed
    # image inspect must report "not present" so the pull path is exercised.
    runner_user.when(lambda a: a[:3] == ("docker", "image", "inspect"), returncode=1)
    runner_user.when(lambda a: a[:2] == ("docker", "pull"), returncode=1, stderr="manifest unknown")
    ctx = FakeContext(tmp_path, conf={"ds_install_method": "docker"}, runner_user=runner_user)

    step = step2.Step2DeepStreamSdk()
    result = step.run(ctx)

    assert result.status is StepStatus.FAILED
    assert "docker pull" in result.message


def test_run_docker_success_reports_installed(tmp_path):
    runner_user = ScriptedRunner(default_returncode=0)
    runner_user.when(lambda a: a[:3] == ("docker", "image", "inspect"), returncode=1)
    runner_user.when(lambda a: a[:2] == ("docker", "pull"), returncode=0)
    ngc = FakeNgc()
    ctx = FakeContext(tmp_path, conf={"ds_install_method": "docker"}, runner_user=runner_user, ngc=ngc)
    _peoplenet_ready(ctx)

    step = step2.Step2DeepStreamSdk()
    result = step.run(ctx)

    assert result.status is StepStatus.COMPLETE
    assert ngc.configure_calls == 1
    assert runner_user.called_with_prefix("docker", "pull", step2.DOCKER_IMAGE)


# ---------------------------------------------------------------------------
# preflight() -- doc section 2 / 12
# ---------------------------------------------------------------------------


def test_prereq_probe_uses_absolute_cuda_path_and_concrete_cudnn_package(tmp_path):
    runner = _passing_pin_runner()
    ctx = FakeContext(tmp_path, runner_root=runner)

    result = step2.Step2DeepStreamSdk().preflight(ctx)

    assert result.status is StepStatus.COMPLETE
    assert (step2.CUDA_NVCC_PATH, "--version") in runner.calls
    assert (
        "dpkg-query",
        "-W",
        "-f=${Version}",
        step2.CUDNN_QUERY_PACKAGE,
    ) in runner.calls
    assert not runner.called_with_prefix("nvcc")


def test_probe_cudnn_normalizes_only_the_exact_apt_pin(tmp_path):
    runner = ScriptedRunner()
    runner.when(
        lambda a: a[-1] == step2.CUDNN_QUERY_PACKAGE,
        stdout=f"{step2.CUDNN_APT_VERSION}\n",
    )
    ctx = FakeContext(tmp_path, runner_root=runner)

    assert step2._probe_cudnn(ctx) == step2.CUDNN_VERSION


def test_preflight_rejects_different_cudnn_debian_revision(tmp_path):
    runner = _passing_pin_runner()
    wrong_revision = f"{step2.CUDNN_VERSION}-2"
    runner.when(
        lambda a: a[-1] == step2.CUDNN_QUERY_PACKAGE,
        stdout=f"{wrong_revision}\n",
    )
    ctx = FakeContext(tmp_path, runner_root=runner)

    result = step2.Step2DeepStreamSdk().preflight(ctx)

    assert result.status is StepStatus.FAILED
    assert "cuDNN" in result.message
    assert wrong_revision in result.message


def test_preflight_fails_on_prereq_pin_mismatch(tmp_path):
    runner = _passing_pin_runner()
    runner.when(lambda a: a[:1] == ("nvidia-smi",), stdout="000.00.00\n")  # wrong driver
    ctx = FakeContext(tmp_path, runner_root=runner)

    step = step2.Step2DeepStreamSdk()
    result = step.preflight(ctx)

    assert result.status is StepStatus.FAILED
    assert "NVIDIA driver" in result.message
    assert "re-run Step 1" in result.message


def test_preflight_fails_when_ngc_key_missing(tmp_path):
    runner = _passing_pin_runner()
    ctx = FakeContext(tmp_path, runner_root=runner, ngc=FakeNgc(key=None))

    step = step2.Step2DeepStreamSdk()
    result = step.preflight(ctx)

    assert result.status is StepStatus.FAILED
    assert "NGC API key" in result.message


def test_preflight_fails_on_wrong_arch(tmp_path, monkeypatch):
    monkeypatch.setattr(step2.platform, "machine", lambda: "aarch64")
    ctx = FakeContext(tmp_path, runner_root=_passing_pin_runner())

    step = step2.Step2DeepStreamSdk()
    result = step.preflight(ctx)

    assert result.status is StepStatus.FAILED
    assert "x86_64" in result.message


def test_preflight_passes_when_everything_pinned(tmp_path):
    ctx = FakeContext(tmp_path, runner_root=_passing_pin_runner())

    step = step2.Step2DeepStreamSdk()
    result = step.preflight(ctx)

    assert result.status is StepStatus.COMPLETE


# ---------------------------------------------------------------------------
# verify() -- doc section 7 / 12
# ---------------------------------------------------------------------------


def _host_ready_runner(*, version_ok=True, app_version_ok=True) -> ScriptedRunner:
    runner = _passing_pin_runner()
    runner.when(
        lambda a: a[:2] == ("dpkg", "-s") and a[-1] == "deepstream-9.1",
        stdout=f"Version: {step2.DS_VERSION_DEB if version_ok else '9.0.0-1'}\n",
    )
    version_stdout = "deepstream-app version 9.1.0\nDeepStreamSDK 9.1.0\n" if app_version_ok else "unknown\n"
    runner.when(lambda a: a[:1] == ("deepstream-app",), stdout=version_stdout)
    return runner


def test_verify_fails_without_recorded_method(tmp_path, _sdk_paths):
    ctx = FakeContext(tmp_path)
    step = step2.Step2DeepStreamSdk()
    result = step.verify(ctx)
    assert result.status is StepStatus.FAILED
    assert "no DeepStream install method" in result.message


def test_verify_host_fails_when_sdk_dir_missing(tmp_path, _sdk_paths):
    runner = _host_ready_runner()
    ctx = FakeContext(tmp_path, conf={"ds_install_method": "deb"}, runner_root=runner)

    step = step2.Step2DeepStreamSdk()
    result = step.verify(ctx)

    assert result.status is StepStatus.FAILED
    assert "not found after install" in result.message


def _peoplenet_ready(ctx) -> pathlib.Path:
    """Pre-place the PeopleNet ONNX (doc STEP-4 section 6.3) so a test
    about DS SDK install/verify -- not about PeopleNet acquisition itself
    -- doesn't also have to stub PeopleNet acquisition."""
    onnx_path = step2._peoplenet_dir(ctx) / step2.PEOPLENET_ONNX_NAME
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    onnx_path.write_bytes(b"stub-onnx")
    return onnx_path


def _make_sdk_tree(ctx, sdk_dir, symlink, profile, *, version_text="DeepStream 9.1.0\n"):
    """Simulate a completed `run()`: SDK tree, symlink, profile.d, the
    doc section 7.1 item 4 rtpmanager marker, and the PeopleNet model all
    present. Requires `ctx` (not just paths) since the marker and the
    PeopleNet dir both live under `ctx.install_dir`, not a fixed path."""
    sdk_dir.mkdir(parents=True)
    (sdk_dir / "version").write_text(version_text)
    symlink.symlink_to(sdk_dir)
    profile.parent.mkdir(parents=True, exist_ok=True)
    profile.write_text(step2._PROFILE_D_CONTENT)
    step2._write_rtpmanager_marker(ctx, "ok", 0)
    _peoplenet_ready(ctx)


def test_verify_host_fails_on_version_mismatch(tmp_path, _sdk_paths):
    sdk_dir, symlink, profile = _sdk_paths
    runner = _host_ready_runner(version_ok=False)
    ctx = FakeContext(tmp_path, conf={"ds_install_method": "deb"}, runner_root=runner)
    _make_sdk_tree(ctx, sdk_dir, symlink, profile)

    step = step2.Step2DeepStreamSdk()
    result = step.verify(ctx)

    assert result.status is StepStatus.FAILED
    assert "version pin mismatch" in result.message


def test_verify_host_fails_when_rtpmanager_marker_missing(tmp_path, _sdk_paths):
    """doc section 7.1 item 4 / 7.3: verify() must fail if the
    post-install tail never left a marker behind, even when everything
    else (SDK, symlink, version, profile.d) looks installed."""
    sdk_dir, symlink, profile = _sdk_paths
    sdk_dir.mkdir(parents=True)
    (sdk_dir / "version").write_text("DeepStream 9.1.0\n")
    symlink.symlink_to(sdk_dir)
    profile.parent.mkdir(parents=True, exist_ok=True)
    profile.write_text(step2._PROFILE_D_CONTENT)
    # Deliberately no _write_rtpmanager_marker(...) call.

    runner = _host_ready_runner()
    ctx = FakeContext(tmp_path, conf={"ds_install_method": "deb"}, runner_root=runner)

    step = step2.Step2DeepStreamSdk()
    result = step.verify(ctx)

    assert result.status is StepStatus.FAILED
    assert "update_rtpmanager.sh marker not found" in result.message


def test_verify_host_passes_with_failed_rtpmanager_marker(tmp_path, _sdk_paths):
    """A recorded rtpmanager *failure* (nonzero exit) is a warning (doc
    section 9 point 1), not a verify blocker -- only a missing marker is."""
    sdk_dir, symlink, profile = _sdk_paths
    runner = _host_ready_runner()
    ctx = FakeContext(tmp_path, conf={"ds_install_method": "deb"}, runner_root=runner)
    _make_sdk_tree(ctx, sdk_dir, symlink, profile)
    step2._write_rtpmanager_marker(ctx, "failed", 1, "boom")

    step = step2.Step2DeepStreamSdk()
    result = step.verify(ctx)

    assert result.status is StepStatus.COMPLETE


def test_verify_host_passes_end_to_end(tmp_path, _sdk_paths):
    sdk_dir, symlink, profile = _sdk_paths
    runner = _host_ready_runner()
    ctx = FakeContext(tmp_path, conf={"ds_install_method": "deb"}, runner_root=runner)
    _make_sdk_tree(ctx, sdk_dir, symlink, profile)

    step = step2.Step2DeepStreamSdk()
    result = step.verify(ctx)

    assert result.status is StepStatus.COMPLETE


def test_verify_tar_uses_short_version_pin(tmp_path, _sdk_paths):
    sdk_dir, symlink, profile = _sdk_paths
    runner = _host_ready_runner()
    ctx = FakeContext(tmp_path, conf={"ds_install_method": "tar"}, runner_root=runner)
    _make_sdk_tree(ctx, sdk_dir, symlink, profile)

    step = step2.Step2DeepStreamSdk()
    result = step.verify(ctx)

    assert result.status is StepStatus.COMPLETE


def test_verify_docker_fails_on_prereq_pin_mismatch(tmp_path):
    """doc section 7.4: "prereq pins still match" applies to the docker
    verify path too -- the container still depends on the host driver."""
    runner_root = _passing_pin_runner()
    runner_root.when(lambda a: a[:1] == ("nvidia-smi",), stdout="000.00.00\n")
    runner_user = ScriptedRunner(default_returncode=0)  # would otherwise pass
    ctx = FakeContext(
        tmp_path,
        conf={"ds_install_method": "docker"},
        runner_root=runner_root,
        runner_user=runner_user,
    )

    step = step2.Step2DeepStreamSdk()
    result = step.verify(ctx)

    assert result.status is StepStatus.FAILED
    assert "NVIDIA driver" in result.message


def test_verify_docker_fails_when_image_absent(tmp_path):
    runner_user = ScriptedRunner(default_returncode=1)
    ctx = FakeContext(
        tmp_path,
        conf={"ds_install_method": "docker"},
        runner_root=_passing_pin_runner(),
        runner_user=runner_user,
    )

    step = step2.Step2DeepStreamSdk()
    result = step.verify(ctx)

    assert result.status is StepStatus.FAILED
    assert "not present locally" in result.message


def test_verify_docker_passes_end_to_end(tmp_path):
    runner_user = ScriptedRunner()
    runner_user.when(lambda a: a[:3] == ("docker", "image", "inspect"), returncode=0)
    runner_user.when(
        lambda a: a[:2] == ("docker", "run") and "deepstream-app" in a,
        stdout="deepstream-app version 9.1.0\n",
    )
    runner_user.when(lambda a: a[:2] == ("docker", "info"), stdout="Runtimes: nvidia runc\n")
    ctx = FakeContext(
        tmp_path,
        conf={"ds_install_method": "docker"},
        runner_root=_passing_pin_runner(),
        runner_user=runner_user,
    )
    _peoplenet_ready(ctx)

    step = step2.Step2DeepStreamSdk()
    result = step.verify(ctx)

    assert result.status is StepStatus.COMPLETE


def test_verify_unrecognized_method_is_failed(tmp_path):
    ctx = FakeContext(tmp_path, conf={"ds_install_method": "bogus"})
    step = step2.Step2DeepStreamSdk()
    result = step.verify(ctx)
    assert result.status is StepStatus.FAILED
    assert "unrecognized" in result.message


# ---------------------------------------------------------------------------
# report() -- no side effects, human summary
# ---------------------------------------------------------------------------


def test_report_prints_summary_after_successful_run(tmp_path, capsys, _sdk_paths):
    """doc section 9 point 1: a non-zero update_rtpmanager.sh exit is
    "surfaced in report()" -- here the fake runner answers every call
    (including update_rtpmanager.sh) with its ScriptedRunner default
    (returncode 1), so report() must show it as failed, not silently drop
    it into an undifferentiated action-name list."""
    sdk_dir, symlink, profile = _sdk_paths
    runner_root = ScriptedRunner()
    runner_root.when(lambda a: a[:2] == ("dpkg", "-s"), returncode=1)
    runner_root.when(lambda a: a[:2] == ("apt-get", "install"), returncode=0)

    ctx = FakeContext(tmp_path, runner_root=runner_root)
    artifact_path = ctx.install_dir / "downloads" / "deepstream" / step2.DEB_ARTIFACT
    ctx.runner_user.when(
        lambda a: a[:1] == ("curl",),
        returncode=0,
        side_effect=lambda: artifact_path.write_bytes(b"stub"),
    )

    step = step2.Step2DeepStreamSdk()
    step.run(ctx)
    step.report(ctx)

    err = capsys.readouterr().err
    assert "DeepStream 9.1 SDK install summary" in err
    assert "method: deb" in err
    assert "post-install actions:" in err
    # update_rtpmanager.sh's default-runner exit (1) must be surfaced as a
    # failure, read back from the durable marker run() wrote -- not from
    # any in-memory state.
    assert "update_rtpmanager.sh: FAILED (exit 1)" in err


def test_report_shows_update_rtpmanager_success_from_marker(tmp_path, capsys):
    """report() reads the marker fresh from disk (doc section 7.1 item 4),
    so it reflects a successful update_rtpmanager.sh run too, not just
    failures."""
    ctx = FakeContext(tmp_path, conf={step2.CONF_METHOD_KEY: "deb"})
    step2._write_rtpmanager_marker(ctx, "ok", 0)

    step = step2.Step2DeepStreamSdk()
    step.report(ctx)

    err = capsys.readouterr().err
    assert "update_rtpmanager.sh: succeeded (exit 0)" in err


def test_report_docker_method_omits_rtpmanager_line(tmp_path, capsys):
    """Docker skips the host post-install tail entirely (doc section 9's
    header), so report() must not print a confusing "not run" line for a
    step that was never supposed to run under this method."""
    ctx = FakeContext(tmp_path, conf={step2.CONF_METHOD_KEY: "docker"})

    step = step2.Step2DeepStreamSdk()
    step.report(ctx)

    err = capsys.readouterr().err
    assert "update_rtpmanager.sh" not in err


# ---------------------------------------------------------------------------
# Full StepResult matrix -- doc section 12
# ---------------------------------------------------------------------------


def test_full_lifecycle_all_pass_is_complete(tmp_path, _sdk_paths):
    """preflight -> run -> verify, all COMPLETE, for a DeepStream 9.1 deb
    install already present (idempotent re-run) -- doc section 12's
    "deb/tar/docker install + post-install verification all pass" row."""
    sdk_dir, symlink, profile = _sdk_paths
    sdk_dir.mkdir(parents=True)
    (sdk_dir / "version").write_text("DeepStream 9.1.0\n")
    symlink.symlink_to(sdk_dir)
    profile.parent.mkdir(parents=True, exist_ok=True)
    profile.write_text(step2._PROFILE_D_CONTENT)

    runner_root = _passing_pin_runner()
    runner_root.when(
        lambda a: a[:2] == ("dpkg", "-s") and a[-1] == "deepstream-9.1",
        stdout=f"Version: {step2.DS_VERSION_DEB}\n",
    )
    runner_root.when(lambda a: a[:1] == ("deepstream-app",), stdout="deepstream-app version 9.1.0\n")

    ngc = FakeNgc()
    ctx = FakeContext(tmp_path, runner_root=runner_root, ngc=ngc)
    # This test exercises the "already installed" idempotent path (dpkg -s
    # already reports the pinned version), which short-circuits run()
    # before the post-install tail -- so the rtpmanager marker doc section
    # 7.1 item 4 requires must have been left by a prior real run. Seed it
    # the same way `_make_sdk_tree` does for the standalone verify() tests.
    step2._write_rtpmanager_marker(ctx, "ok", 0)
    _peoplenet_ready(ctx)

    step = step2.Step2DeepStreamSdk()

    pre = step.preflight(ctx)
    assert pre.status is StepStatus.COMPLETE

    run_result = step.run(ctx)
    assert run_result.status is StepStatus.COMPLETE

    verify_result = step.verify(ctx)
    assert verify_result.status is StepStatus.COMPLETE


# ---------------------------------------------------------------------------
# PeopleNet model acquisition -- doc STEP-4 section 6.3 (ported from
# laptop/scripts/00_bootstrap.sh Phase 10, previously never implemented --
# see STEP-2-DEEPSTREAM-SDK.md section 11's "Open gap, not yet resolved").
# ---------------------------------------------------------------------------


def _api_download_side_effect(runner, *, member=None, contents=b"onnx-bytes"):
    """Write a model ZIP where the API download command expects it."""

    def _write():
        args = runner.calls[-1]
        archive_arg = next(arg for arg in args if str(arg).startswith("NGC_MODEL_ARCHIVE="))
        archive = pathlib.Path(archive_arg.split("=", 1)[1])
        archive.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.writestr(member or f"peoplenet/files/{step2.PEOPLENET_ONNX_NAME}", contents)

    return _write


def _api_call(args):
    return args[:1] == ("env",) and any(
        str(arg).startswith("NGC_MODEL_URL=") for arg in args
    )


def _api_size_call(args):
    return args[:1] == ("env",) and any(
        str(arg).startswith("NGC_MODEL_SIZE_URL=") for arg in args
    )


def test_ensure_peoplenet_skips_when_already_present(tmp_path):
    runner_user = ScriptedRunner()  # unconfigured: any real call would fail
    ctx = FakeContext(tmp_path, runner_user=runner_user)
    _peoplenet_ready(ctx)

    result = step2._ensure_peoplenet_model(ctx)

    assert result is None
    assert not runner_user.calls


def test_ensure_peoplenet_writes_labels_when_onnx_present_but_labels_missing(tmp_path):
    ctx = FakeContext(tmp_path)
    onnx_path = _peoplenet_ready(ctx)

    result = step2._ensure_peoplenet_model(ctx)

    assert result is None
    labels_path = onnx_path.parent / step2.PEOPLENET_LABELS_NAME
    assert labels_path.read_text(encoding="utf-8") == "person\nbag\nface\n"


def test_peoplenet_api_url_with_team():
    assert step2._peoplenet_api_url(step2.PEOPLENET_NGC_TAG_DEFAULT) == (
        "https://api.ngc.nvidia.com/v2/org/nvidia/team/tao/models/peoplenet/"
        "versions/deployable_quantized_onnx_v2.6.3/zip"
    )


def test_peoplenet_api_url_without_team_and_escapes_version():
    assert step2._peoplenet_api_url("nvidia/peoplenet:v2+final") == (
        "https://api.ngc.nvidia.com/v2/org/nvidia/models/peoplenet/"
        "versions/v2%2Bfinal/zip"
    )


@pytest.mark.parametrize("tag", ["", "nvidia/tao/peoplenet", "too/many/path/parts:v1"])
def test_peoplenet_api_url_rejects_invalid_tag(tag):
    with pytest.raises(ValueError, match="expected org"):
        step2._peoplenet_api_url(tag)


def test_ensure_peoplenet_downloads_and_places_model(tmp_path):
    runner_user = ScriptedRunner(default_returncode=0)
    runner_user.when(
        _api_call,
        returncode=0,
        side_effect=_api_download_side_effect(runner_user),
    )
    ngc = FakeNgc()
    ctx = FakeContext(tmp_path, runner_user=runner_user, ngc=ngc)

    result = step2._ensure_peoplenet_model(ctx)

    assert result is None
    assert ngc.configure_calls == 0
    target_dir = step2._peoplenet_dir(ctx)
    assert (target_dir / step2.PEOPLENET_ONNX_NAME).read_bytes() == b"onnx-bytes"
    assert (target_dir / step2.PEOPLENET_LABELS_NAME).is_file()
    call = next(call for call in runner_user.calls if _api_call(call))
    assert any(arg.startswith("NGC_ENV_FILE=") for arg in call)
    assert "a-fake-ngc-key" not in " ".join(call)
    assert call[-1] == step2._NGC_API_DOWNLOAD_SCRIPT


def test_peoplenet_probes_authenticated_archive_size_for_progress(tmp_path):
    runner_user = ScriptedRunner(default_returncode=0)
    runner_user.when(_api_size_call, stdout="4096")
    runner_user.when(
        _api_call,
        side_effect=_api_download_side_effect(runner_user),
    )
    ctx = FakeContext(tmp_path, runner_user=runner_user)

    result = step2._ensure_peoplenet_model(ctx)

    assert result is None
    size_call = next(call for call in runner_user.calls if _api_size_call(call))
    assert any(arg.startswith("NGC_ENV_FILE=") for arg in size_call)
    assert "a-fake-ngc-key" not in " ".join(size_call)
    assert size_call[-1] == step2._NGC_API_CONTENT_LENGTH_SCRIPT


def test_ensure_peoplenet_uses_conf_override_tag(tmp_path):
    custom_tag = "nvidia/tao/peoplenet:custom_tag"
    runner_user = ScriptedRunner(default_returncode=0)
    runner_user.when(
        _api_call,
        returncode=0,
        side_effect=_api_download_side_effect(runner_user),
    )
    ctx = FakeContext(
        tmp_path, conf={step2.CONF_PEOPLENET_TAG_KEY: custom_tag}, runner_user=runner_user
    )

    result = step2._ensure_peoplenet_model(ctx)

    assert result is None
    call = next(call for call in runner_user.calls if _api_call(call))
    assert (
        "NGC_MODEL_URL=https://api.ngc.nvidia.com/v2/org/nvidia/team/tao/"
        "models/peoplenet/versions/custom_tag/zip"
    ) in call


def test_ensure_peoplenet_invalid_tag_is_failed_without_running_command(tmp_path):
    runner_user = ScriptedRunner(default_returncode=0)
    ctx = FakeContext(
        tmp_path,
        conf={step2.CONF_PEOPLENET_TAG_KEY: "invalid"},
        runner_user=runner_user,
    )

    result = step2._ensure_peoplenet_model(ctx)

    assert result is not None
    assert result.status is StepStatus.FAILED
    assert "invalid peoplenet_ngc_tag" in result.message
    assert not runner_user.calls


def test_ensure_peoplenet_api_download_failure_is_failed_without_manual_action(tmp_path):
    runner_user = ScriptedRunner(default_returncode=0)
    runner_user.when(
        _api_call,
        returncode=1,
        stderr="unauthorized",
    )
    ctx = FakeContext(tmp_path, runner_user=runner_user)

    result = step2._ensure_peoplenet_model(ctx)

    assert result is not None
    assert result.status is StepStatus.FAILED
    assert "NGC Catalog API download failed" in result.message
    assert "unauthorized" in result.message
    assert not result.user_actions


def test_ensure_peoplenet_archive_without_onnx_is_failed(tmp_path):
    runner_user = ScriptedRunner(default_returncode=0)
    runner_user.when(
        _api_call,
        returncode=0,
        side_effect=_api_download_side_effect(
            runner_user, member="peoplenet/README.txt", contents=b"metadata"
        ),
    )
    ctx = FakeContext(tmp_path, runner_user=runner_user)

    result = step2._ensure_peoplenet_model(ctx)

    assert result is not None
    assert result.status is StepStatus.FAILED
    assert "contained 0 copies" in result.message


@pytest.mark.parametrize(
    "archive_error",
    [
        RuntimeError("password required"),
        NotImplementedError("unsupported compression method"),
    ],
)
def test_ensure_peoplenet_archive_read_error_is_failed(
    tmp_path, monkeypatch, archive_error
):
    runner_user = ScriptedRunner(default_returncode=0)
    runner_user.when(
        _api_call,
        returncode=0,
        side_effect=_api_download_side_effect(runner_user),
    )
    ctx = FakeContext(tmp_path, runner_user=runner_user)
    monkeypatch.setattr(
        step2.shutil,
        "copyfileobj",
        lambda source, dest: (_ for _ in ()).throw(archive_error),
    )

    result = step2._ensure_peoplenet_model(ctx)

    assert result is not None
    assert result.status is StepStatus.FAILED
    assert "could not extract PeopleNet model archive" in result.message
    assert str(archive_error) in result.message


def test_run_deb_already_installed_still_fetches_peoplenet(tmp_path):
    """run()'s DS-install shortcut (already-installed deb) must not skip
    the PeopleNet fetch -- doc STEP-4 section 6.3 makes it Step 2's job
    independent of which install-method branch ran."""
    runner_root = ScriptedRunner()
    runner_root.when(lambda a: a[:2] == ("dpkg", "-s"), stdout="Version: 9.1.0-1\n")
    runner_user = ScriptedRunner(default_returncode=0)
    runner_user.when(
        _api_call,
        returncode=0,
        side_effect=_api_download_side_effect(runner_user),
    )
    ctx = FakeContext(tmp_path, runner_root=runner_root, runner_user=runner_user)

    step = step2.Step2DeepStreamSdk()
    result = step.run(ctx)

    assert result.status is StepStatus.COMPLETE
    assert (step2._peoplenet_dir(ctx) / step2.PEOPLENET_ONNX_NAME).is_file()


def test_run_deb_already_installed_peoplenet_failure_overrides_complete(tmp_path):
    runner_root = ScriptedRunner()
    runner_root.when(lambda a: a[:2] == ("dpkg", "-s"), stdout="Version: 9.1.0-1\n")
    runner_user = ScriptedRunner(default_returncode=1)
    ctx = FakeContext(tmp_path, runner_root=runner_root, runner_user=runner_user)

    step = step2.Step2DeepStreamSdk()
    result = step.run(ctx)

    assert result.status is StepStatus.FAILED
    assert "NGC Catalog API" in result.message


def test_verify_host_fails_when_peoplenet_model_missing(tmp_path, _sdk_paths):
    """verify() must fail even when the DS SDK itself verifies clean, if
    the PeopleNet model Step 5 needs at runtime was never fetched."""
    sdk_dir, symlink, profile = _sdk_paths
    runner = _host_ready_runner()
    ctx = FakeContext(tmp_path, conf={"ds_install_method": "deb"}, runner_root=runner)
    sdk_dir.mkdir(parents=True)
    (sdk_dir / "version").write_text("DeepStream 9.1.0\n")
    symlink.symlink_to(sdk_dir)
    profile.parent.mkdir(parents=True, exist_ok=True)
    profile.write_text(step2._PROFILE_D_CONTENT)
    step2._write_rtpmanager_marker(ctx, "ok", 0)
    # Deliberately no _peoplenet_ready(ctx) call.

    step = step2.Step2DeepStreamSdk()
    result = step.verify(ctx)

    assert result.status is StepStatus.FAILED
    assert "PeopleNet model not found" in result.message


def test_report_logs_peoplenet_present(tmp_path, capsys):
    ctx = FakeContext(tmp_path, conf={"ds_install_method": "deb"})
    onnx_path = _peoplenet_ready(ctx)

    step = step2.Step2DeepStreamSdk()
    step.report(ctx)

    captured = capsys.readouterr()
    assert str(onnx_path) in captured.err


def test_report_logs_peoplenet_missing(tmp_path, capsys):
    ctx = FakeContext(tmp_path, conf={"ds_install_method": "deb"})

    step = step2.Step2DeepStreamSdk()
    step.report(ctx)

    captured = capsys.readouterr()
    assert "PeopleNet model: MISSING" in captured.err


def _hide_prompts(monkeypatch):
    from mv3dt_installer import prompting

    monkeypatch.setattr(
        prompting, "prompt_unavailable_reason", lambda: "stdout is not a terminal (is the output piped?)"
    )


def _no_prompt(*_args, **_kwargs):  # pragma: no cover -- must never be called
    raise AssertionError("a hidden prompt was shown")


def test_resolve_method_hidden_prompt_takes_deb(tmp_path, monkeypatch, capsys):
    ctx = _ambiguous_ctx(tmp_path, non_interactive=False)
    _hide_prompts(monkeypatch)
    monkeypatch.setattr(step2, "_INPUT", _no_prompt)
    method, reason = step2._resolve_method(ctx)
    assert method is step2.Method.DEB
    assert "could not be shown" in reason
    assert "using deb (the default)" in capsys.readouterr().err
