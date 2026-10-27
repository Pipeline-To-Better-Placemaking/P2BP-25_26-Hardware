"""Step 4 -- ingest AMC 3.2.1 results and wire DeepStream configuration.

Step 3 persists the AMC project identity and local microservice port. This
step follows that project through the AMC API, downloads its documented
MV3DT result archive, validates it, and swaps it into the configured
calibration directory atomically. Later recalibrations are picked up by a
systemd timer that invokes the same one-shot ``ingest`` subcommand; the timer
is always installed but enabled only when ``MV3DT_AUTO_INGEST=on``, otherwise
re-ingest is the manual ``mv3dt-installer ingest``.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import pathlib
import re
import shutil
import stat
import tempfile
import zipfile
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

import yaml

from mv3dt_installer import app as app_mod
from mv3dt_installer import cameras as cameras_mod
from mv3dt_installer import config as config_mod
from mv3dt_installer import footage as footage_mod
from mv3dt_installer import managed_files
from mv3dt_installer import systemd, waitui
from mv3dt_installer.steps import StepResult, StepStatus, UserAction, register
from mv3dt_installer.steps import step3_amc_launcher as step3_mod

if TYPE_CHECKING:  # pragma: no cover
    from mv3dt_installer.app import Context

__all__ = [
    "ProjectInputs",
    "resolve_project_inputs",
    "default_calibration_dir",
    "IngestOutcome",
    "download_and_ingest",
    "render_tracker_yaml",
    "render_app_config",
    "render_dewarper_config",
    "render_pub_sub_config",
    "dewarper_distortion",
    "tiled_display_grid",
    "handle_ingest_subcommand",
    "Step4CalibOutputWiring",
]

CONF_LOCATION_ID_KEY = step3_mod.CONF_LOCATION_ID_KEY
CONF_PROJECT_NAME_KEY = step3_mod.CONF_PROJECT_NAME_KEY
CONF_AMC_PROJECT_ID_KEY = step3_mod.CONF_AMC_PROJECT_ID_KEY
CONF_MS_PORT_KEY = step3_mod.CONF_MS_PORT_KEY
CONF_CAM_USER_KEY = "CAM_USER"
CONF_CAM_PASSWORD_KEY = "CAM_PASSWORD"
CONF_AMC_EXPORT_WAIT_S_KEY = "AMC_EXPORT_WAIT_S"
CONF_CALIBRATION_DIR_KEY = "CALIBRATION_DIR"
CONF_CALIB_FOOTAGE_SECONDS_KEY = "CALIB_FOOTAGE_SECONDS"
CONF_CALIB_FOOTAGE_DIR_KEY = "CALIB_FOOTAGE_DIR"

DEFAULT_EXPORT_WAIT_S = 3600.0
RESULT_ARCHIVE_NAME = "mv3dt_result.zip"
TRANSFORMS_NAME = "transforms.yml"
_ARCHIVE_DIGEST_NAME = ".archive.sha256"
_INGEST_LOG_NAME = ".ingest.log"
_MAX_ARCHIVE_MEMBERS = 4096
_MAX_UNCOMPRESSED_BYTES = 1 << 30

APP_CONFIG_TEMPLATE_NAME = "deepstream_app_config.txt"
TRACKER_YAML_TEMPLATE_NAME = "config_tracker_NvMOT.yml"
INFER_PRIMARY_TEMPLATE_NAME = "config_infer_primary.txt"
MSGCONV_TEMPLATE_NAME = "msgconv_config.txt"
RENDERED_APP_CONFIG_NAME = "deepstream_app_config.rendered.txt"

# The tracker's ReID model (NVIDIA TAO ReidentificationNet), the option
# DeepStream 9.1's own sources/tracker_ReID/README recommends. DeepStream 9.1
# ships no samples/models/Tracker/ directory, so nothing provides it unless
# the installer does. The rendered tracker config names this file through
# `tltEncodedModel` relative to `<install_dir>/deepstream`, and DeepStream
# builds the TensorRT engine beside it on first start.
REID_MODEL_URL = (
    "https://api.ngc.nvidia.com/v2/models/nvidia/tao/reidentificationnet/"
    "versions/deployable_v1.0/files/resnet50_market1501.etlt"
)
REID_MODEL_NAME = "resnet50_market1501.etlt"
REID_RELATIVE_DIR = pathlib.Path("deepstream") / "models" / "reid"

# The tracker's PoseEstimator (poseEstimatorType 1) loads NVIDIA TAO
# BodyPose3DNet from this ONNX file; DeepStream builds the engine beside it on
# first start. NGC version `deployable_accuracy_onnx_1.0` holds exactly this
# one file (guest download, no API key); size and SHA-256 are the values the
# NGC file listing publishes for it.
BODYPOSE_MODEL_URL = (
    "https://api.ngc.nvidia.com/v2/models/nvidia/tao/bodypose3dnet/"
    "versions/deployable_accuracy_onnx_1.0/files/bodypose3dnet_accuracy.onnx"
)
BODYPOSE_MODEL_NAME = "bodypose3dnet_accuracy.onnx"
BODYPOSE_MODEL_SIZE = 70575749
BODYPOSE_MODEL_SHA256 = (
    "0452b785a70fcd6bc5bd4069249bdfd85eb139c9e9216bcf81f89df33945d028"
)
BODYPOSE_RELATIVE_DIR = pathlib.Path("deepstream") / "models" / "bodypose3dnet"

# MV3DT fusion: each camera's tracker publishes its tracklets on the local
# broker and subscribes to its peers (Communicator in the tracker config).
MV3DT_DIR_NAME = "mv3dt"
PUB_SUB_CONFIG_NAME = "pub_sub_info_config.yml"
MQTT_CONFIG_NAME = "config_mqtt.txt"
MV3DT_BROKER = "127.0.0.1:1883"

# nvdewarper configs generated from AMC's committed lens rectification.
DEWARPER_DIR_NAME = "dewarper"
DEWARPER_FOCAL_LENGTH = 1600

# The VLC viewer stream (tiled view, on-screen display, RTSP out on 8554).
CONF_VIEWER_STREAM_KEY = config_mod.GATE_VIEWER_STREAM

# The ingest timer is installed always but enabled only when this is on.
CONF_AUTO_INGEST_KEY = config_mod.GATE_AUTO_INGEST

# Single-view 3D tracking needs one camera model per camera
# (ObjectModelProjection.cameraModelFilepath). AMC's export carries them as
# camInfo/camInfo_NN.yml; Step 4 converts each into
# <install_dir>/deepstream/camera_models/ and lists them in source order.
AMC_CAMERA_MODEL_DIR = "camInfo"
CAMERA_MODEL_DIR_NAME = "camera_models"
_CAMERA_MODEL_MATRIX_KEYS = ("projectionMatrix_3x4_w2p", "projectionMatrix_3x4")


class AmcApiError(RuntimeError):
    """A fatal response or invalid result from the local AMC service."""


@dataclass(frozen=True)
class ProjectInputs:
    location_id: str
    project_name: str
    project_id: str
    ms_port: str
    cam_user: str
    cam_password: str
    export_wait_s: float


@dataclass(frozen=True)
class IngestOutcome:
    changed: bool
    stamp: str


def _to_float(value: Optional[str], default: float) -> float:
    try:
        return float(value) if value else default
    except ValueError:
        return default


def _missing_input_keys(ctx: "Context") -> list[str]:
    required = (
        CONF_LOCATION_ID_KEY,
        CONF_PROJECT_NAME_KEY,
        CONF_AMC_PROJECT_ID_KEY,
        CONF_MS_PORT_KEY,
        CONF_CAM_USER_KEY,
        CONF_CAM_PASSWORD_KEY,
    )
    return [key for key in required if not (ctx.conf.get(key) or "").strip()]


def resolve_project_inputs(ctx: "Context") -> tuple[Optional[ProjectInputs], list[str]]:
    missing = _missing_input_keys(ctx)
    if missing:
        return None, missing
    conf = ctx.conf
    return (
        ProjectInputs(
            location_id=conf[CONF_LOCATION_ID_KEY].strip(),
            project_name=conf[CONF_PROJECT_NAME_KEY].strip(),
            project_id=conf[CONF_AMC_PROJECT_ID_KEY].strip(),
            ms_port=conf[CONF_MS_PORT_KEY].strip(),
            cam_user=conf[CONF_CAM_USER_KEY],
            cam_password=conf[CONF_CAM_PASSWORD_KEY],
            export_wait_s=_to_float(
                conf.get(CONF_AMC_EXPORT_WAIT_S_KEY), DEFAULT_EXPORT_WAIT_S
            ),
        ),
        [],
    )


def default_calibration_dir(ctx: "Context", location_id: str) -> pathlib.Path:
    return ctx.install_dir / "deepstream" / "calibration" / location_id


def _configuration_action(
    ctx: "Context", missing: list[str], *, camera_missing: bool
) -> StepResult:
    parts: list[str] = []
    actions: list[UserAction] = []
    step3_keys = {
        CONF_LOCATION_ID_KEY,
        CONF_PROJECT_NAME_KEY,
        CONF_AMC_PROJECT_ID_KEY,
        CONF_MS_PORT_KEY,
    }
    missing_step3 = [key for key in missing if key in step3_keys]
    missing_credentials = [key for key in missing if key not in step3_keys]
    if missing:
        parts.append(f"missing installer configuration: {', '.join(missing)}")
    if missing_step3:
        actions.append(
            UserAction(
                text=(
                    "Re-run the automated Step 3 AMC launcher and project setup; "
                    f"it owns {', '.join(missing_step3)}."
                ),
                command=(
                    f"sudo {ctx.install_dir / 'bin' / step3_mod.INSTALLER_BIN_NAME} "
                    "--reset-step 3"
                ),
            )
        )
    if missing_credentials:
        actions.append(
            UserAction(
                text=(
                    "Re-run the installer interactively to capture the camera "
                    f"credentials ({', '.join(missing_credentials)})."
                ),
            )
        )
    if camera_missing:
        parts.append("camera inventory is unavailable")
        actions.append(
            UserAction(
                text=(
                    "Connect and activate the cameras, then re-run the installer. "
                    "Use the scan command later when explicitly refreshing the fleet."
                ),
                command=(
                    f"sudo {ctx.install_dir / 'bin' / step3_mod.INSTALLER_BIN_NAME} "
                    "--resume"
                ),
            )
        )
    return StepResult(
        status=StepStatus.USER_ACTION_REQUIRED,
        message="; ".join(parts),
        user_actions=actions,
    )


def _step3_complete(ctx: "Context") -> bool:
    wrapper = ctx.install_dir / "bin" / step3_mod.AMC_WRAPPER_NAME
    return wrapper.is_file() and os.access(wrapper, os.X_OK)


def _api_base(inputs: ProjectInputs) -> str:
    return f"http://localhost:{inputs.ms_port}/v1"


def _result_detail(result) -> str:
    detail = (
        getattr(result, "stderr", "") or getattr(result, "stdout", "") or "no details"
    )
    return " ".join(str(detail).split())[:1000]


def _json_object(result, label: str) -> dict:
    if getattr(result, "returncode", 1) != 0:
        raise AmcApiError(
            f"{label} failed (exit {getattr(result, 'returncode', '?')}): "
            f"{_result_detail(result)}"
        )
    try:
        payload = json.loads(result.stdout or "")
    except (TypeError, json.JSONDecodeError) as exc:
        raise AmcApiError(f"{label} returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise AmcApiError(f"{label} returned an unexpected JSON value")
    if "code" in payload and payload.get("code") != 0:
        raise AmcApiError(
            f"{label} returned an error: {payload.get('message') or payload}"
        )
    return payload


def _project_state(ctx: "Context", inputs: ProjectInputs) -> str:
    result = ctx.run_root(
        "curl",
        "-fsS",
        "--max-time",
        "10",
        f"{_api_base(inputs)}/get_project_info/{inputs.project_id}",
        check=False,
        capture_output=True,
        text=True,
        stream=False,
    )
    payload = _json_object(result, f"AMC project {inputs.project_id} status")
    info = payload.get("project_info")
    if not isinstance(info, dict):
        raise AmcApiError("AMC project status response has no project_info object")
    state_value = info.get("project_state")
    if not isinstance(state_value, str) or not state_value.strip():
        raise AmcApiError("AMC project status response has no project_state")
    state_name = state_value.strip().upper()
    if state_name == "ERROR":
        log_result = ctx.run_root(
            "curl",
            "-fsS",
            "--max-time",
            "10",
            f"{_api_base(inputs)}/amc/calibrate/{inputs.project_id}/log",
            check=False,
            capture_output=True,
            text=True,
            stream=False,
        )
        evidence = (
            (log_result.stdout or "").strip()
            if getattr(log_result, "returncode", 1) == 0
            else f"log fetch failed: {_result_detail(log_result)}"
        )
        raise AmcApiError(
            f"AMC calibration for project {inputs.project_id} entered ERROR: "
            f"{' '.join(evidence.split())[:2000] or 'no log evidence returned'}"
        )
    return state_name


def _fetch_rectification(ctx: "Context", inputs: ProjectInputs) -> Optional[list]:
    """AMC's per-camera lens rectification for the project, or `None` with a
    warning when the API is unreachable or answers with anything unusable.
    A missing rectification never fails Step 4; it only means no dewarper."""
    result = ctx.run_root(
        "curl",
        "-fsS",
        "--max-time",
        "10",
        f"{_api_base(inputs)}/rectification/{inputs.project_id}/cameras",
        check=False,
        capture_output=True,
        text=True,
        stream=False,
    )
    try:
        payload = _json_object(result, f"AMC rectification for {inputs.project_id}")
    except AmcApiError as exc:
        ctx.log.warn(f"{exc}; rendering no dewarpers")
        return None
    cameras = payload.get("cameras")
    if not isinstance(cameras, list):
        ctx.log.warn(
            "AMC rectification response has no cameras list; rendering no dewarpers"
        )
        return None
    return [camera for camera in cameras if isinstance(camera, dict)]


def _footage_seconds(ctx: "Context") -> int:
    """STEP-4-CALIBRATION-FOOTAGE §2 clip length: `CALIB_FOOTAGE_SECONDS`,
    or the pinned default when the key is absent or not a positive integer."""
    try:
        seconds = int(str(ctx.conf.get(CONF_CALIB_FOOTAGE_SECONDS_KEY) or "").strip())
    except ValueError:
        return footage_mod.DEFAULT_FOOTAGE_SECONDS
    return seconds if seconds > 0 else footage_mod.DEFAULT_FOOTAGE_SECONDS


def _footage_project_dir(ctx: "Context", inputs: ProjectInputs) -> pathlib.Path:
    """STEP-4-CALIBRATION-FOOTAGE §2 project directory under the invoking
    user's footage root (or the `CALIB_FOOTAGE_DIR` override)."""
    override = (ctx.conf.get(CONF_CALIB_FOOTAGE_DIR_KEY) or "").strip() or None
    root = footage_mod.footage_root(ctx.user.home, override)
    return footage_mod.project_dir(root, inputs.project_name)


def _complete_footage_count(ctx: "Context", inputs: ProjectInputs) -> Optional[int]:
    """The clip count of a complete footage set for this project, or `None`
    when no complete set exists or the inventory cannot be read."""
    try:
        cameras = _load_enabled_cameras(ctx)
    except (KeyError, OSError, ValueError):
        return None
    if not cameras:
        return None
    if not footage_mod.is_complete(
        _footage_project_dir(ctx, inputs),
        cameras,
        project_name=inputs.project_name,
        seconds=_footage_seconds(ctx),
    ):
        return None
    return len(cameras)


def _wait_hints(ctx: "Context", inputs: ProjectInputs) -> list[UserAction]:
    ui_port = ctx.conf.get(step3_mod.CONF_UI_PORT_KEY) or step3_mod.DEFAULT_UI_PORT
    footage_actions: list[UserAction] = []
    clip_count = _complete_footage_count(ctx, inputs)
    if clip_count is not None:
        # STEP-4-CALIBRATION-FOOTAGE §5: the upload hint comes first.
        footage_actions.append(
            UserAction(
                text=(
                    f"Upload the {clip_count} clips in "
                    f"{_footage_project_dir(ctx, inputs)} at the AMC Video Upload step."
                )
            )
        )
    return footage_actions + [
        UserAction(
            text=f"Continue calibration in the AMC UI at http://localhost:{ui_port}."
        ),
        UserAction(
            text=(
                f"Complete and execute AMC project {inputs.project_name} "
                f"({inputs.project_id}); the installer continues when its state is COMPLETED."
            )
        ),
    ]


def _wait_for_completed(ctx: "Context", inputs: ProjectInputs) -> waitui.WaitOutcome:
    last_state = {"value": ""}

    def ready() -> bool:
        state_name = _project_state(ctx, inputs)
        if state_name != last_state["value"]:
            ctx.log.info(f"AMC project {inputs.project_id} state: {state_name}")
            last_state["value"] = state_name
        return state_name == "COMPLETED"

    return waitui.wait_until(
        ready,
        description=f"Waiting for AMC project {inputs.project_name} to complete",
        hint_actions=_wait_hints(ctx, inputs),
        timeout_s=inputs.export_wait_s,
        non_interactive=ctx.non_interactive,
    )


def _utc_stamp() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _archive_digest(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_members(zf: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    members = zf.infolist()
    if not members or len(members) > _MAX_ARCHIVE_MEMBERS:
        raise AmcApiError("AMC result ZIP has an invalid member count")
    total = 0
    seen: set[str] = set()
    transforms = False
    for member in members:
        if "\\" in member.filename:
            raise AmcApiError(
                f"AMC result ZIP contains an unsafe path: {member.filename!r}"
            )
        name = member.filename
        path = pathlib.PurePosixPath(name)
        if (
            not name
            or name.startswith("/")
            or path.is_absolute()
            or ".." in path.parts
            or re.match(r"^[A-Za-z]:", name)
            or any(":" in part for part in path.parts)
        ):
            raise AmcApiError(
                f"AMC result ZIP contains unsafe path: {member.filename!r}"
            )
        normalized = str(path)
        if normalized in seen:
            raise AmcApiError(f"AMC result ZIP contains duplicate path: {normalized!r}")
        seen.add(normalized)
        mode_type = stat.S_IFMT(member.external_attr >> 16)
        if mode_type == stat.S_IFLNK:
            raise AmcApiError(f"AMC result ZIP contains a symlink: {member.filename!r}")
        if mode_type not in (0, stat.S_IFREG, stat.S_IFDIR):
            raise AmcApiError(
                f"AMC result ZIP contains an unsupported member: {member.filename!r}"
            )
        if member.flag_bits & 0x1:
            raise AmcApiError(
                f"AMC result ZIP contains an encrypted member: {member.filename!r}"
            )
        total += member.file_size
        if total > _MAX_UNCOMPRESSED_BYTES:
            raise AmcApiError("AMC result ZIP expands beyond the 1 GiB safety limit")
        if normalized == TRANSFORMS_NAME and not member.is_dir():
            transforms = True
    if not transforms:
        raise AmcApiError(f"AMC result ZIP is missing required {TRANSFORMS_NAME}")
    return members


def _extract_archive(archive: pathlib.Path, payload_dir: pathlib.Path) -> None:
    try:
        with zipfile.ZipFile(archive) as zf:
            members = _safe_members(zf)
            for member in members:
                relative = pathlib.PurePosixPath(member.filename.replace("\\", "/"))
                target = payload_dir.joinpath(*relative.parts)
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)
    except AmcApiError:
        raise
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        raise AmcApiError(f"invalid AMC result ZIP: {exc}") from exc


def _download_archive(
    ctx: "Context", inputs: ProjectInputs, archive: pathlib.Path
) -> None:
    result = ctx.run_root(
        "curl",
        "-fsS",
        "--max-time",
        "300",
        "-o",
        str(archive),
        f"{_api_base(inputs)}/result/{inputs.project_id}/mv3dt_result?result_type=amc",
        check=False,
        capture_output=True,
        text=True,
    )
    if getattr(result, "returncode", 1) != 0:
        raise AmcApiError(
            f"AMC MV3DT result download failed (exit {getattr(result, 'returncode', '?')}): "
            f"{_result_detail(result)}"
        )
    if not archive.is_file() or archive.stat().st_size == 0:
        raise AmcApiError("AMC MV3DT result download returned an empty file")


def download_and_ingest(
    ctx: "Context", *, inputs: ProjectInputs, dest: pathlib.Path
) -> IngestOutcome:
    """Download, validate, and atomically replace one calibration tree."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    stamp = _utc_stamp()
    with tempfile.TemporaryDirectory(
        prefix=f".{dest.name}.ingest-", dir=dest.parent
    ) as scratch:
        scratch_path = pathlib.Path(scratch)
        archive = scratch_path / RESULT_ARCHIVE_NAME
        payload = scratch_path / "payload"
        payload.mkdir()
        _download_archive(ctx, inputs, archive)
        digest = _archive_digest(archive)
        if dest.is_dir() and (dest / TRANSFORMS_NAME).is_file():
            try:
                if (dest / _ARCHIVE_DIGEST_NAME).read_text(
                    encoding="ascii"
                ).strip() == digest:
                    return IngestOutcome(changed=False, stamp=stamp)
            except OSError:
                pass
        _extract_archive(archive, payload)
        (payload / _ARCHIVE_DIGEST_NAME).write_text(digest + "\n", encoding="ascii")
        (payload / _INGEST_LOG_NAME).write_text(
            f"{stamp}  ingested AMC project {inputs.project_id}\n", encoding="utf-8"
        )
        backup = scratch_path / "previous"
        had_dest = dest.exists()
        if had_dest:
            os.replace(dest, backup)
        try:
            os.replace(payload, dest)
        except BaseException:
            if had_dest and backup.exists():
                os.replace(backup, dest)
            raise
    return IngestOutcome(changed=True, stamp=stamp)


_PROMPT: Callable[[str], str] = input


def _resolve_calibration_dir(
    ctx: "Context", inputs: ProjectInputs, *, allow_prompt: bool
) -> pathlib.Path:
    persisted = ctx.conf.get(CONF_CALIBRATION_DIR_KEY)
    if persisted:
        return pathlib.Path(persisted)
    default = default_calibration_dir(ctx, inputs.location_id)
    if not allow_prompt or ctx.non_interactive:
        return default
    answer = _PROMPT(
        f"Choose where the calibration output for this project should live [{default}]: "
    ).strip()
    return pathlib.Path(answer).expanduser() if answer else default


def _chown_tree(ctx: "Context", root: pathlib.Path) -> None:
    for entry in (root, *root.rglob("*")):
        try:
            os.chown(entry, ctx.user.uid, ctx.user.gid)
        except OSError:
            pass


_CAMERA_MODEL_PLACEHOLDER_RE = re.compile(
    r"(?m)^([ \t]*)- \$\{CAMERA_MODEL_FILEPATH\}[ \t]*$"
)


def render_tracker_yaml(template_text: str, *, camera_model_files: list) -> str:
    """Expand the `- ${CAMERA_MODEL_FILEPATH}` line into one list entry per
    camera model, in source order."""

    def expand(match: "re.Match[str]") -> str:
        indent = match.group(1)
        return "\n".join(f"{indent}- {path}" for path in camera_model_files)

    text, count = _CAMERA_MODEL_PLACEHOLDER_RE.subn(expand, template_text, count=1)
    if count != 1:
        raise ValueError("tracker template has no ${CAMERA_MODEL_FILEPATH} entry")
    return text


def convert_camera_model(text: str) -> str:
    """Convert one AMC camInfo file into the form DS 9.1's tracker loads.

    AMC 3.2.1 writes `modelInfo` as a list of per-class entries
    (`- classID: 1, height, radius`). NVIDIA's sample camInfo.yml, and the
    configuration proven on the workstation on 2026-10-20, use a single
    `modelInfo: {height, radius}` map. The projection matrix is kept under
    its original key.
    """
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError("camera model is not a YAML mapping")
    key = next((k for k in _CAMERA_MODEL_MATRIX_KEYS if k in data), None)
    if key is None:
        raise ValueError(
            "camera model has neither projectionMatrix_3x4_w2p nor projectionMatrix_3x4"
        )
    matrix = [float(value) for value in data[key]]
    if len(matrix) != 12:
        raise ValueError(f"{key} has {len(matrix)} values, expected 12")
    info = data.get("modelInfo")
    if isinstance(info, list):
        info = info[0] if info else None
    if not isinstance(info, dict) or "height" not in info or "radius" not in info:
        raise ValueError("camera model modelInfo needs height and radius")
    lines = [f"{key}:"]
    lines += [f"  - {value!r}" for value in matrix]
    lines += [
        "modelInfo:",
        f"  height: {float(info['height'])!r}",
        f"  radius: {float(info['radius'])!r}",
    ]
    return "\n".join(lines) + "\n"


def _write_camera_models(
    ctx: "Context", calibration_dir: pathlib.Path, camera_count: int
) -> list:
    """Convert AMC's camInfo_NN.yml files into the camera-model directory and
    return their paths in source order.

    **REQUIRED:** AMC camera N must be the Nth enabled camera in the
    inventory -- upload the clips at AMC's Video Upload step in inventory
    order. AMC names them cam_00, cam_01, ... in upload order.
    """
    sources = sorted((calibration_dir / AMC_CAMERA_MODEL_DIR).glob("camInfo_*.yml"))
    if len(sources) != camera_count:
        raise ValueError(
            f"AMC result has {len(sources)} camera model(s) under "
            f"{calibration_dir / AMC_CAMERA_MODEL_DIR} for {camera_count} enabled "
            "camera(s); 3D tracking needs exactly one per camera"
        )
    out_dir = ctx.install_dir / "deepstream" / CAMERA_MODEL_DIR_NAME
    paths = []
    for source in sources:
        try:
            converted = convert_camera_model(source.read_text(encoding="utf-8"))
        except (yaml.YAMLError, TypeError, ValueError) as exc:
            raise ValueError(f"{source}: {exc}") from exc
        target = out_dir / source.name
        _write_text_report(ctx, target, converted, label=f"camera model {source.name}")
        paths.append(target)
    _chown_tree(ctx, out_dir)
    return paths


def _parse_tracker_yaml(text: str) -> Optional[dict]:
    lines = text.splitlines()
    if lines and lines[0].strip().startswith("%YAML"):
        text = "\n".join(lines[1:])
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError:
        return None
    return data if isinstance(data, dict) else None


_SOURCE_BLOCK_RE = re.compile(r"\[source(\d+)\][^\[]*", re.DOTALL)
_URI_LINE_RE = re.compile(r"(?m)^uri=.*$")


_ENABLE_LINE_RE = re.compile(r"(?m)^enable=.*$")


def _rtsp_uri(cam, *, user: str, password: str) -> str:
    return f"rtsp://{user}:{password}@{cam.ip}:554{cam.rtsp_path}"


def render_app_config(
    template_text: str,
    *,
    cam_user: str,
    cam_password: str,
    location_id: str,
    cameras: list,
    template_path: str,
    calibration_dir: "pathlib.Path | str",
    viewer_stream: bool = False,
    dewarper_files: "Optional[dict[int, pathlib.Path | str]]" = None,
) -> str:
    """Render the deepstream-app config. `viewer_stream` turns on the tiled
    display and adds the on-screen display and the RTSP sink;
    `dewarper_files` maps a source index to its nvdewarper config."""
    text = template_text.replace("${CAM_USER}", cam_user)
    text = text.replace("${CAM_PASSWORD}", cam_password).replace(
        "${LOCATION_ID}", location_id
    )

    def replace_source(match: "re.Match[str]") -> str:
        index = int(match.group(1))
        if index >= len(cameras):
            # No camera for this slot: disable it rather than leave the
            # template's placeholder URI enabled, which deepstream-app
            # retries forever with "Failed to connect".
            return _ENABLE_LINE_RE.sub("enable=0", match.group(0), count=1)
        uri = _rtsp_uri(cameras[index], user=cam_user, password=cam_password)
        return _URI_LINE_RE.sub(f"uri={uri}", match.group(0), count=1)

    text = _SOURCE_BLOCK_RE.sub(replace_source, text)
    extra: list[str] = []
    if viewer_stream:
        text, count = _TILED_DISPLAY_BLOCK_RE.subn(
            _tiled_display_section(len(cameras)), text, count=1
        )
        if count != 1:
            raise ValueError("app config template has no [tiled-display] section")
        extra += [_OSD_SECTION, _RTSP_SINK_SECTION]
    for index in sorted(dewarper_files or {}):
        extra.append(_dewarper_section(index, (dewarper_files or {})[index]))
    if extra:
        text = text.rstrip("\n") + "\n\n" + "\n".join(extra)
    return (
        "# Rendered by mv3dt-installer step4_calib_output_wiring\n"
        f"# Source template: {pathlib.PurePath(template_path).name}\n"
        f"# Calibration dir: {calibration_dir}\n"
        f"# LOCATION_ID    : {location_id}\n"
        "# Edit the committed template, not this file; it is regenerated.\n\n" + text
    )


_TILED_DISPLAY_BLOCK_RE = re.compile(
    r"(?m)^\[tiled-display\][ \t]*\n(?:[^\[#\n][^\n]*\n|\n)*"
)

# Live workstation values (2 cameras, 2026-10-26 snapshot).
_OSD_SECTION = (
    "[osd]\n"
    "display-text=1\n"
    "enable=1\n"
    "gpu-id=0\n"
    "border-width=3\n"
    "text-size=22\n"
    "text-color=1;1;1;1\n"
    "text-bg-color=0.3;0.3;0.3;1\n"
    "font=Serif\n"
)
_RTSP_SINK_SECTION = (
    "[sink2]\n"
    "enable=1\n"
    "type=4\n"
    "codec=1\n"
    "enc-type=0\n"
    "sync=0\n"
    "bitrate=4000000\n"
    "rtsp-port=8554\n"
    "udp-port=5400\n"
)
_TILED_WIDTH = 1920
_TILED_MAX_COLUMNS = 4


def tiled_display_grid(camera_count: int) -> tuple[int, int, int, int]:
    """`(rows, columns, width, height)` of the viewer's tiled display.

    Up to two cameras: one row, 1920x540 (the live two-camera layout). More
    cameras: up to four tiles per row, 1920 wide, with the height chosen so
    each tile keeps the 16:9 stream aspect (3 cameras 1x3 at 1920x360, 4 at
    1920x270, 5 to 8 in 2x4 at 1920x540)."""
    count = max(camera_count, 1)
    if count <= 2:
        return 1, count, _TILED_WIDTH, 540
    columns = min(count, _TILED_MAX_COLUMNS)
    rows = -(-count // columns)
    height = _TILED_WIDTH * rows * 9 // (16 * columns)
    return rows, columns, _TILED_WIDTH, height


def _tiled_display_section(camera_count: int) -> str:
    rows, columns, width, height = tiled_display_grid(camera_count)
    return (
        "[tiled-display]\n"
        "enable=1\n"
        f"rows={rows}\n"
        f"columns={columns}\n"
        f"width={width}\n"
        f"height={height}\n\n"
    )


def _dewarper_section(index: int, config_file: "pathlib.Path | str") -> str:
    return (
        f"[dewarper{index}]\n"
        "enable=1\n"
        "gpu-id=0\n"
        f"source-id={index}\n"
        "num-surfaces-per-frame=1\n"
        "num-output-buffers=4\n"
        "nvbuf-memory-type=0\n"
        f"config-file={config_file}\n"
    )


def _format_number(value: float) -> str:
    """Integral values without a decimal point (960, not 960.0)."""
    return str(int(value)) if float(value).is_integer() else repr(float(value))


def _format_coefficient(value: float) -> str:
    """Five decimals, trailing zeros trimmed, as in the live dewarper files."""
    text = f"{value:.5f}".rstrip("0").rstrip(".")
    return "0" if text in ("", "-0") else text


def dewarper_distortion(k1: float, k2: float) -> str:
    """AMC's radial model is in pixels; nvdewarper projection type 3 wants
    the coefficients normalized by the focal length: k1*f^2 and k2*f^4."""
    f = DEWARPER_FOCAL_LENGTH
    return (
        f"{_format_coefficient(k1 * f**2)};{_format_coefficient(k2 * f**4)};0;0;0"
    )


def render_dewarper_config(camera: dict) -> str:
    """One nvdewarper config from an AMC rectification camera entry with a
    committed radial model. Raises ValueError for anything else."""
    committed = camera.get("committed")
    if not isinstance(committed, dict):
        raise ValueError("no committed rectification")
    model = committed.get("model")
    if model != "radial":
        raise ValueError(f"committed model is {model!r}, not radial")
    try:
        width = int(camera["width"])
        height = int(camera["height"])
        cx = float(camera["cx"])
        cy = float(camera["cy"])
        k1 = float(committed["k1"])
        k2 = float(committed["k2"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"incomplete rectification entry ({exc})") from exc
    return (
        "[property]\n"
        f"output-width={width}\n"
        f"output-height={height}\n"
        "num-batch-buffers=1\n"
        "\n"
        "[surface0]\n"
        "projection-type=3\n"
        f"width={width}\n"
        f"height={height}\n"
        f"focal-length={DEWARPER_FOCAL_LENGTH}\n"
        f"distortion={dewarper_distortion(k1, k2)}\n"
        f"src-x0={_format_number(cx)}\n"
        f"src-y0={_format_number(cy)}\n"
    )


def render_pub_sub_config(camera_count: int) -> str:
    """MV3DT vision-neighbor pub/sub for one DeepStream instance: camera k
    (1-based, source order) publishes /trck/cam<k> and subscribes to every
    other camera. Format and topic template of NVIDIA's
    deepstream-tracker-3d-multi-view generate_pub_sub_configs.py."""
    topics = [f"{MV3DT_BROKER};/trck/cam{k}" for k in range(1, camera_count + 1)]
    lines = [
        "# MV3DT vision-neighbor pub/sub, ordered by stream index.",
        "# Rendered by mv3dt-installer step4_calib_output_wiring: one DeepStream",
        f"# instance, broker {MV3DT_BROKER}, topic /trck/cam<N>. Each camera",
        "# publishes its tracklets and subscribes to every other camera.",
        "pubBrokerTopicStr:",
    ]
    lines += [f"- {topic}" for topic in topics]
    lines.append("subPeerBrokerTopicStrs:")
    for index in range(camera_count):
        peers = [topic for other, topic in enumerate(topics) if other != index]
        if not peers:
            lines.append("- []")
            continue
        lines.append(f"- - {peers[0]}")
        lines += [f"  - {peer}" for peer in peers[1:]]
    return "\n".join(lines) + "\n"


# NVIDIA's deepstream-tracker-3d-multi-view config_templates/config_mqtt.txt
# (DeepStream v9.2.0), verbatim. The username/password lines are the
# sample's fixed non-secret placeholders: the installer's local Mosquitto
# allows anonymous clients, so they are never a real credential.
# set-threaded=0 is required: the tracker communicator drives the client
# with mosquitto_loop_start.
MQTT_CONFIG_TEXT = (
    "[message-broker]\n"
    "username = user\n"
    "password = password\n"
    "#client-id = uniqueID\n"
    "#enable-tls = 1\n"
    "#tls-cafile =\n"
    "#tls-capath =\n"
    "#tls-certfile =\n"
    "#tls-keyfile =\n"
    "#share-connection = 1\n"
    "#loop-timeout = 2000\n"
    "#keep-alive = 60\n"
    "set-threaded = 0\n"
)


def _source_block(text: str, index: int) -> str:
    match = re.search(rf"\[source{index}\][^\[]*", text, re.DOTALL)
    return match.group(0) if match else ""




def _write_text_report(
    ctx: "Context", path: pathlib.Path, text: str, *, label: str
) -> managed_files.Outcome:
    """Write one rendered file through the edit protection in
    `managed_files`: an operator-edited file is kept and the new version
    lands beside it as `<path>.installer-new`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    outcome = managed_files.write_managed(
        path, text, manifest=managed_files.Manifest(), label=label
    )
    written = (
        managed_files.pending_path(path)
        if outcome is managed_files.Outcome.KEPT_MODIFIED
        else path
    )
    try:
        os.chown(written, ctx.user.uid, ctx.user.gid)
    except OSError:
        pass
    if outcome is managed_files.Outcome.WRITTEN:
        ctx.report_installed(label, str(path))
    elif outcome is managed_files.Outcome.UNCHANGED:
        ctx.report_already_installed(label, str(path))
    return outcome


def _load_enabled_cameras(ctx: "Context") -> list:
    camera_path = pathlib.Path(ctx.conf[config_mod.CAMERAS_FILE_KEY])
    return [
        camera
        for camera in cameras_mod.parse_inventory(
            camera_path.read_text(encoding="utf-8")
        )
        if camera.enabled
    ]


def _reid_model_path(ctx: "Context") -> pathlib.Path:
    return ctx.install_dir / REID_RELATIVE_DIR / REID_MODEL_NAME


def _ensure_reid_model(ctx: "Context") -> Optional[str]:
    """Download the tracker's ReID model once. Returns an error string, or
    `None` when the model is present (already, or freshly downloaded)."""
    target = _reid_model_path(ctx)
    if target.is_file() and target.stat().st_size > 0:
        ctx.report_already_installed("reid-model", REID_MODEL_NAME)
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    staged = target.with_name(f".{REID_MODEL_NAME}.tmp")
    try:
        result = ctx.run_root(
            "curl",
            "-fsSL",
            "--retry",
            "3",
            "-o",
            str(staged),
            REID_MODEL_URL,
            check=False,
            capture_output=True,
            text=True,
            stream=False,
        )
        if result.returncode != 0 or not staged.is_file():
            return (
                f"ReID model download failed (exit {result.returncode}): "
                f"{(result.stderr or '').strip()[-500:]}"
            )
        if staged.stat().st_size == 0:
            return "ReID model download produced an empty file"
        os.replace(staged, target)
    finally:
        staged.unlink(missing_ok=True)
    # The pipeline may run as the invoking user (STEP-6), and DeepStream
    # writes the engine next to the model on first start.
    _chown_tree(ctx, target.parent)
    ctx.report_installed("reid-model", REID_MODEL_NAME)
    return None


def _bodypose_model_path(ctx: "Context") -> pathlib.Path:
    return ctx.install_dir / BODYPOSE_RELATIVE_DIR / BODYPOSE_MODEL_NAME


def _file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_bodypose_model(ctx: "Context") -> Optional[str]:
    """Download the PoseEstimator's BodyPose3DNet ONNX once, checked against
    NGC's published size and SHA-256. Returns an error string, or `None`
    when the model is present (already, or freshly downloaded)."""
    target = _bodypose_model_path(ctx)
    if target.is_file() and target.stat().st_size > 0:
        ctx.report_already_installed("bodypose3dnet-model", BODYPOSE_MODEL_NAME)
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    staged = target.with_name(f".{BODYPOSE_MODEL_NAME}.tmp")
    try:
        result = ctx.run_root(
            "curl",
            "-fsSL",
            "--retry",
            "3",
            "-o",
            str(staged),
            BODYPOSE_MODEL_URL,
            check=False,
            capture_output=True,
            text=True,
            stream=False,
        )
        if result.returncode != 0 or not staged.is_file():
            return (
                f"BodyPose3DNet model download failed (exit {result.returncode}): "
                f"{(result.stderr or '').strip()[-500:]}"
            )
        size = staged.stat().st_size
        if size != BODYPOSE_MODEL_SIZE:
            return (
                f"BodyPose3DNet model download has {size} bytes, "
                f"expected {BODYPOSE_MODEL_SIZE}"
            )
        if _file_sha256(staged) != BODYPOSE_MODEL_SHA256:
            return "BodyPose3DNet model download failed its SHA-256 check"
        os.replace(staged, target)
    finally:
        staged.unlink(missing_ok=True)
    _chown_tree(ctx, target.parent)
    ctx.report_installed("bodypose3dnet-model", BODYPOSE_MODEL_NAME)
    return None


def _viewer_stream_on(ctx: "Context") -> bool:
    return (ctx.conf.get(CONF_VIEWER_STREAM_KEY) or "off").strip().lower() == "on"


def _write_dewarpers(
    ctx: "Context", inputs: ProjectInputs, camera_count: int
) -> dict[int, pathlib.Path]:
    """Write one nvdewarper config per enabled camera that has a committed
    radial rectification in AMC; return {source index: config path}.

    AMC camera cam_NN is source NN, the same upload-order mapping the
    camera models rely on."""
    rectification = _fetch_rectification(ctx, inputs)
    if rectification is None:
        return {}
    by_id = {str(camera.get("camera_id")): camera for camera in rectification}
    out_dir = ctx.install_dir / "deepstream" / DEWARPER_DIR_NAME
    written: dict[int, pathlib.Path] = {}
    for index in range(camera_count):
        camera_id = f"cam_{index:02d}"
        camera = by_id.get(camera_id)
        if camera is None:
            ctx.log.warn(f"AMC has no rectification for {camera_id}; no dewarper")
            continue
        try:
            text = render_dewarper_config(camera)
        except ValueError as exc:
            ctx.log.warn(f"{camera_id}: {exc}; no dewarper")
            continue
        target = out_dir / f"dewarper_cam{index}.txt"
        _write_text_report(ctx, target, text, label=f"dewarper {target.name}")
        written[index] = target
    return written


def _render_configs(
    ctx: "Context",
    *,
    inputs: ProjectInputs,
    calibration_dir: pathlib.Path,
    camera_model_files: list,
) -> None:
    deepstream_dir = ctx.install_dir / "deepstream"
    tracker_template_path = ctx.asset_path("deepstream", TRACKER_YAML_TEMPLATE_NAME)
    tracker_text = render_tracker_yaml(
        tracker_template_path.read_text(encoding="utf-8"),
        camera_model_files=[str(path) for path in camera_model_files],
    )
    _write_text_report(
        ctx,
        deepstream_dir / TRACKER_YAML_TEMPLATE_NAME,
        tracker_text,
        label=TRACKER_YAML_TEMPLATE_NAME,
    )
    cameras = _load_enabled_cameras(ctx)
    mv3dt_dir = deepstream_dir / MV3DT_DIR_NAME
    _write_text_report(
        ctx,
        mv3dt_dir / PUB_SUB_CONFIG_NAME,
        render_pub_sub_config(len(cameras)),
        label=PUB_SUB_CONFIG_NAME,
    )
    _write_text_report(
        ctx, mv3dt_dir / MQTT_CONFIG_NAME, MQTT_CONFIG_TEXT, label=MQTT_CONFIG_NAME
    )
    dewarper_files = _write_dewarpers(ctx, inputs, len(cameras))
    app_template_path = ctx.asset_path("deepstream", APP_CONFIG_TEMPLATE_NAME)
    app_text = render_app_config(
        app_template_path.read_text(encoding="utf-8"),
        cam_user=inputs.cam_user,
        cam_password=inputs.cam_password,
        location_id=inputs.location_id,
        cameras=cameras,
        template_path=str(app_template_path),
        calibration_dir=calibration_dir,
        viewer_stream=_viewer_stream_on(ctx),
        dewarper_files=dewarper_files,
    )
    _write_text_report(
        ctx,
        deepstream_dir / RENDERED_APP_CONFIG_NAME,
        app_text,
        label=RENDERED_APP_CONFIG_NAME,
    )
    for name in (INFER_PRIMARY_TEMPLATE_NAME, MSGCONV_TEMPLATE_NAME):
        source = ctx.asset_path("deepstream", name)
        _write_text_report(
            ctx, deepstream_dir / name, source.read_text(encoding="utf-8"), label=name
        )


_SLUG_INVALID_RE = re.compile(r"[^a-z0-9]+")


def _slugify(name: str, *, location_id: str) -> str:
    slug = _SLUG_INVALID_RE.sub("-", name.strip().lower()).strip("-")
    fallback = _SLUG_INVALID_RE.sub("-", location_id.strip().lower()).strip("-")
    return (slug or fallback or "project")[:64].strip("-") or "project"


def _systemd_value(value: object) -> str:
    text = str(value)
    if re.search(r'["\\\x00-\x1f\x7f]', text):
        raise ValueError(
            f"value cannot be represented safely in a systemd unit: {text!r}"
        )
    return text.replace("%", "%%")


def _render_reingest_units(ctx: "Context", *, inputs: ProjectInputs) -> dict[str, str]:
    slug = _slugify(inputs.project_name, location_id=inputs.location_id)
    installer = _systemd_value(ctx.install_dir / "bin" / step3_mod.INSTALLER_BIN_NAME)
    project = _systemd_value(inputs.project_name)
    install_dir = _systemd_value(ctx.install_dir)
    user = _systemd_value(ctx.user.name)
    service = f"""[Unit]
Description=Ingest AMC API result for project {project}
After=docker.service network-online.target

[Service]
Type=oneshot
# The installer refuses to run as root without a non-root invoking user
# (doc 00 section 9.2). A systemd service has no sudo, so name the user here.
Environment=SUDO_USER={user}
ExecStart=\"{installer}\" ingest --project \"{project}\" --non-interactive --install-dir \"{install_dir}\"
"""
    timer = f"""[Unit]
Description=Poll AMC API for recalibration of project {project}

[Timer]
OnBootSec=2min
OnUnitActiveSec=60s
Persistent=true
Unit=mv3dt-ingest-{slug}.service

[Install]
WantedBy=timers.target
"""
    return {
        f"mv3dt-ingest-{slug}.service": service,
        f"mv3dt-ingest-{slug}.timer": timer,
    }


def _systemd_runner(ctx: "Context"):
    return lambda argv, **kwargs: ctx.run_root(*argv, **kwargs)


def _auto_ingest_on(ctx: "Context") -> bool:
    return (ctx.conf.get(CONF_AUTO_INGEST_KEY) or "off").strip().lower() == "on"


def _install_reingest_units(ctx: "Context", *, inputs: ProjectInputs) -> None:
    runner = _systemd_runner(ctx)
    slug = _slugify(inputs.project_name, location_id=inputs.location_id)
    legacy_name = f"mv3dt-ingest-{slug}.path"
    legacy_path = systemd.UNIT_DIR / legacy_name
    changed_any = False
    if legacy_path.exists():
        result = ctx.run_root(
            "systemctl",
            "disable",
            "--now",
            legacy_name,
            check=False,
            capture_output=True,
            text=True,
        )
        if getattr(result, "returncode", 1) != 0:
            raise AmcApiError(
                f"could not disable obsolete {legacy_name}: {_result_detail(result)}"
            )
        legacy_path.unlink()
        changed_any = True
        ctx.log.info(f"removed obsolete re-ingest unit {legacy_name}")
    for name, content in _render_reingest_units(ctx, inputs=inputs).items():
        outcome = systemd.install_managed_unit(name, content)
        if outcome is managed_files.Outcome.WRITTEN:
            changed_any = True
            ctx.report_installed(name, "systemd unit")
        elif outcome is managed_files.Outcome.UNCHANGED:
            ctx.report_already_installed(name, "systemd unit")
    if changed_any:
        systemd.daemon_reload(runner=runner)
    timer = f"mv3dt-ingest-{slug}.timer"
    if _auto_ingest_on(ctx):
        systemd.enable_now(timer, runner=runner)
        return
    # MV3DT_AUTO_INGEST=off: re-ingest is the manual `mv3dt-installer ingest`.
    if systemd.is_enabled(timer, runner=runner) or systemd.is_active(
        timer, runner=runner
    ):
        result = ctx.run_root(
            "systemctl",
            "disable",
            "--now",
            timer,
            check=False,
            capture_output=True,
            text=True,
        )
        if getattr(result, "returncode", 1) != 0:
            raise AmcApiError(
                f"could not disable {timer} (MV3DT_AUTO_INGEST=off): "
                f"{_result_detail(result)}"
            )
        ctx.log.info(
            f"disabled {timer} (MV3DT_AUTO_INGEST=off); re-ingest with "
            "`mv3dt-installer ingest`"
        )


def _camera_inventory_missing(ctx: "Context") -> bool:
    value = ctx.conf.get(config_mod.CAMERAS_FILE_KEY)
    if not value or not pathlib.Path(value).is_file():
        return True
    try:
        return not _load_enabled_cameras(ctx)
    except OSError:
        return True


def _discover_camera_inventory(ctx: "Context", inputs: ProjectInputs) -> bool:
    """Run the first camera scan automatically; later refreshes use the CLI flag."""
    seed_path = ctx.asset_path("cameras", "cameras.yml")
    try:
        seed_text = seed_path.read_text(encoding="utf-8")
    except OSError:
        seed_text = ""
    seed_header = seed_text.split("\n\ncameras:", 1)[0]
    prime_ips = [cam.ip for cam in cameras_mod.parse_inventory(seed_text) if cam.ip]

    ctx.progress.task("discovering cameras and validating RTSP streams")
    result = cameras_mod.refresh(
        ctx.install_dir,
        seed_header=seed_header,
        cam_user=inputs.cam_user,
        cam_password=inputs.cam_password,
        cidr=ctx.conf.get(
            config_mod.CAMERA_SCAN_CIDR_KEY, cameras_mod.DEFAULT_SCAN_CIDR
        ),
        interfaces=(
            [ctx.conf[config_mod.CAMERA_SCAN_IFACE_KEY]]
            if ctx.conf.get(config_mod.CAMERA_SCAN_IFACE_KEY)
            else None
        ),
        prime_ips=prime_ips,
        non_interactive=ctx.non_interactive,
    )
    if not any(camera.enabled for camera in result.cameras):
        return False

    inventory_path = ctx.install_dir / "cameras.yml"
    config_mod.persist_value(
        ctx.install_dir, config_mod.CAMERAS_FILE_KEY, str(inventory_path)
    )
    ctx.conf[config_mod.CAMERAS_FILE_KEY] = str(inventory_path)
    return True


# ---------------------------------------------------------------------------
# STEP-4-CALIBRATION-FOOTAGE §3, §4, §6: calibration footage
# ---------------------------------------------------------------------------

# Injectable so tests never block on real input().
_INPUT: Callable[[str], str] = input


def _format_gib(count: int) -> str:
    return f"{count / (1 << 30):.1f} GiB"


def _format_minutes(seconds: int) -> str:
    minutes = round(seconds / 60, 1)
    return f"{minutes:g} minute" + ("" if minutes == 1 else "s")


def _footage_prompt(seconds: int, camera_count: int, project: pathlib.Path) -> str:
    cameras_word = "camera" if camera_count == 1 else "cameras"
    return (
        f"Record {_format_minutes(seconds)} of calibration footage from "
        f"{camera_count} {cameras_word} into\n"
        f"{project}?\n"
        "Have someone walk through the whole scene while it records.\n"
        "Press Enter to start, or type s to skip: "
    )


def _maybe_record_footage(
    ctx: "Context", inputs: ProjectInputs
) -> Optional[StepResult]:
    """STEP-4-CALIBRATION-FOOTAGE §3/§4: record one clip per enabled camera
    when the AMC project is still at `INIT`, no complete set exists, the run
    is interactive, and there is room. Returns `None` to continue to the
    completion wait, or the `StepResult` preflight must return instead.

    Raises `AmcApiError` if the one status request fails, exactly as the
    completion wait would on its first poll.
    """
    state_name = _project_state(ctx, inputs)
    if state_name != "INIT":
        return None
    cameras = _load_enabled_cameras(ctx)
    if not cameras:
        return None
    seconds = _footage_seconds(ctx)
    project = _footage_project_dir(ctx, inputs)
    if footage_mod.is_complete(
        project, cameras, project_name=inputs.project_name, seconds=seconds
    ):
        return None
    if ctx.non_interactive:
        ctx.log.info(
            "Skipping calibration footage recording under --non-interactive; "
            "recording needs someone walking through the scene."
        )
        return None
    ok, free, needed = footage_mod.has_room(project.parent, len(cameras))
    if not ok:
        return StepResult(
            status=StepStatus.USER_ACTION_REQUIRED,
            message=(
                f"not enough free space to record calibration footage in "
                f"{project.parent}: found {_format_gib(free)}, "
                f"need {_format_gib(needed)}"
            ),
            user_actions=[
                UserAction(
                    text=(
                        f"Free at least {_format_gib(needed)} on the filesystem "
                        f"holding {project.parent}, or set "
                        f"{CONF_CALIB_FOOTAGE_DIR_KEY} in installer.conf to a "
                        "directory with room, then re-run the installer."
                    ),
                    command=(
                        f"sudo {ctx.install_dir / 'bin' / step3_mod.INSTALLER_BIN_NAME} "
                        "--resume"
                    ),
                )
            ],
        )
    answer = _INPUT(_footage_prompt(seconds, len(cameras), project))
    if answer.strip().lower() == "s":
        ctx.log.info("Skipping calibration footage recording for this run.")
        return None

    camera_word = "camera" if len(cameras) == 1 else "cameras"
    ctx.progress.task(
        f"recording calibration footage from {len(cameras)} {camera_word}"
    )
    try:
        if not (ctx.conf.get(CONF_CALIB_FOOTAGE_DIR_KEY) or "").strip():
            # STEP-4-CALIBRATION-FOOTAGE §6: the timer-driven ingest runs as
            # root, so it can only find this footage through the persisted
            # root, never by re-resolving the invoking user's home.
            root = str(project.parent.absolute())
            config_mod.persist_value(ctx.install_dir, CONF_CALIB_FOOTAGE_DIR_KEY, root)
            ctx.conf[CONF_CALIB_FOOTAGE_DIR_KEY] = root
        result = footage_mod.record(
            cameras,
            user=inputs.cam_user,
            password=inputs.cam_password,
            project_dir=project,
            project_name=inputs.project_name,
            seconds=seconds,
            user_prefix=("sudo", "-u", ctx.user.name, "-H"),
            on_progress=lambda fraction: ctx.progress.percent(
                int(fraction * 100), f"{len(cameras)} {camera_word}"
            ),
        )
    except OSError as exc:
        return StepResult(
            status=StepStatus.FAILED,
            message=f"could not record calibration footage: {exc}",
        )
    if result.cancelled:
        return StepResult(
            status=StepStatus.USER_ACTION_REQUIRED,
            message="calibration footage recording was cancelled",
            user_actions=_wait_hints(ctx, inputs),
        )
    if result.failed:
        failed = ", ".join(f"{camera.id} ({camera.ip})" for camera in result.failed)
        return StepResult(
            status=StepStatus.USER_ACTION_REQUIRED,
            message=f"calibration footage failed for camera(s): {failed}",
            user_actions=[
                UserAction(
                    text=(
                        f"Check that camera(s) {failed} are activated and that "
                        f"{CONF_CAM_USER_KEY}/{CONF_CAM_PASSWORD_KEY} match the "
                        "camera credentials, then re-run the installer to "
                        "record the whole set again."
                    ),
                    command=(
                        f"sudo {ctx.install_dir / 'bin' / step3_mod.INSTALLER_BIN_NAME} "
                        "--resume"
                    ),
                )
            ],
        )
    ctx.log.info(
        f"Calibration footage recorded in {project}. "
        f"Upload these {len(result.recorded)} files at the AMC Video Upload step."
    )
    return None


def _delete_footage_after_ingest(ctx: "Context", inputs: ProjectInputs) -> None:
    """STEP-4-CALIBRATION-FOOTAGE §6: the footage has served its purpose once
    an export is installed. `delete_if_owned` only removes a directory
    carrying the installer's marker, and a failure is never a step failure.

    Only the persisted `CALIB_FOOTAGE_DIR` is consulted: the timer-driven
    `ingest` runs as root, where the invoking-user default would resolve
    under root's home. Absent the key, the installer never recorded footage,
    so there is nothing to delete."""
    persisted = (ctx.conf.get(CONF_CALIB_FOOTAGE_DIR_KEY) or "").strip()
    if not persisted:
        return
    try:
        footage_mod.delete_if_owned(
            footage_mod.project_dir(pathlib.Path(persisted), inputs.project_name)
        )
    except Exception as exc:  # noqa: BLE001 -- §6: never fail the step
        ctx.log.warn(f"Could not delete calibration footage: {exc}")


def _render_targets_exist(ctx: "Context", calibration_dir: pathlib.Path) -> bool:
    """Whether every file a render would write is already present: one
    camera model per enabled camera plus the DeepStream configs."""
    deepstream_dir = ctx.install_dir / "deepstream"
    sources = sorted((calibration_dir / AMC_CAMERA_MODEL_DIR).glob("camInfo_*.yml"))
    try:
        camera_count = len(_load_enabled_cameras(ctx))
    except OSError:
        return False
    if len(sources) != camera_count:
        return False
    targets = [deepstream_dir / CAMERA_MODEL_DIR_NAME / s.name for s in sources]
    targets += [
        deepstream_dir / name
        for name in (
            TRACKER_YAML_TEMPLATE_NAME,
            RENDERED_APP_CONFIG_NAME,
            INFER_PRIMARY_TEMPLATE_NAME,
            MSGCONV_TEMPLATE_NAME,
        )
    ]
    targets += [
        deepstream_dir / MV3DT_DIR_NAME / name
        for name in (PUB_SUB_CONFIG_NAME, MQTT_CONFIG_NAME)
    ]
    return all(target.is_file() for target in targets)


def _wire_download(
    ctx: "Context",
    inputs: ProjectInputs,
    *,
    allow_prompt: bool,
    skip_unchanged: bool = False,
) -> StepResult:
    """Ingest the AMC result and render from it.

    `skip_unchanged` is the timer-driven `ingest` path: when the archive is
    the one already installed and every render target exists, nothing is
    re-rendered. The step's own run always renders, so a new installer
    release still offers its new templates."""
    dest = _resolve_calibration_dir(ctx, inputs, allow_prompt=allow_prompt)
    existed = dest.is_dir() and (dest / TRANSFORMS_NAME).is_file()
    try:
        outcome = download_and_ingest(ctx, inputs=inputs, dest=dest)
    except (AmcApiError, OSError) as exc:
        return StepResult(status=StepStatus.FAILED, message=str(exc))
    _chown_tree(ctx, dest)
    label = f"{inputs.project_name}@{outcome.stamp}"
    reporter = (
        ctx.report_installed
        if outcome.changed and not existed
        else ctx.report_already_installed
    )
    reporter("calibration-export", label)
    config_mod.persist_value(ctx.install_dir, CONF_CALIBRATION_DIR_KEY, str(dest))
    ctx.conf[CONF_CALIBRATION_DIR_KEY] = str(dest)
    reid_failure = _ensure_reid_model(ctx)
    if reid_failure is not None:
        return StepResult(status=StepStatus.FAILED, message=reid_failure)
    bodypose_failure = _ensure_bodypose_model(ctx)
    if bodypose_failure is not None:
        return StepResult(status=StepStatus.FAILED, message=bodypose_failure)
    if skip_unchanged and not outcome.changed and _render_targets_exist(ctx, dest):
        ctx.log.info(
            f"AMC result for {inputs.project_name} is unchanged and the rendered "
            "configs exist; nothing to render"
        )
        _delete_footage_after_ingest(ctx, inputs)
        return StepResult(status=StepStatus.COMPLETE)
    try:
        camera_model_files = _write_camera_models(
            ctx, dest, len(_load_enabled_cameras(ctx))
        )
        _render_configs(
            ctx,
            inputs=inputs,
            calibration_dir=dest,
            camera_model_files=camera_model_files,
        )
    except ValueError as exc:
        return StepResult(status=StepStatus.FAILED, message=str(exc))
    _delete_footage_after_ingest(ctx, inputs)
    return StepResult(status=StepStatus.COMPLETE)


class Step4CalibOutputWiring:
    id = "step4_calib_output_wiring"
    title = "Calibration output wiring"
    phases = ("project inputs", "AMC result ingest", "ownership and verification")
    order = 4

    def preflight(self, ctx: "Context") -> StepResult:
        if not _step3_complete(ctx):
            return StepResult(
                status=StepStatus.FAILED,
                message="AutoMagicCalib launcher (Step 3) is not complete; run Step 3 first",
            )
        ctx.progress.phase(1)
        ctx.progress.task("resolving AMC project and camera inputs")
        inputs, missing = resolve_project_inputs(ctx)
        camera_missing = _camera_inventory_missing(ctx)
        if inputs is None:
            return _configuration_action(ctx, missing, camera_missing=camera_missing)
        if camera_missing:
            try:
                camera_missing = not _discover_camera_inventory(ctx, inputs)
            except (OSError, ValueError) as exc:
                return StepResult(
                    status=StepStatus.FAILED,
                    message=f"automatic camera discovery failed: {exc}",
                )
            if camera_missing:
                return _configuration_action(ctx, [], camera_missing=True)
        try:
            _load_enabled_cameras(ctx)
        except OSError as exc:
            return StepResult(
                status=StepStatus.FAILED,
                message=f"could not read camera inventory: {exc}",
            )
        for name in (
            APP_CONFIG_TEMPLATE_NAME,
            TRACKER_YAML_TEMPLATE_NAME,
            INFER_PRIMARY_TEMPLATE_NAME,
            MSGCONV_TEMPLATE_NAME,
        ):
            path = ctx.asset_path("deepstream", name)
            if not path.is_file():
                return StepResult(
                    status=StepStatus.FAILED,
                    message=f"missing bundled template: {path}",
                )
        try:
            footage_result = _maybe_record_footage(ctx, inputs)
        except AmcApiError as exc:
            return StepResult(status=StepStatus.FAILED, message=str(exc))
        except OSError as exc:
            return StepResult(
                status=StepStatus.FAILED,
                message=f"could not read camera inventory: {exc}",
            )
        if footage_result is not None:
            return footage_result
        ctx.progress.task(
            f"waiting for AMC project {inputs.project_name} to complete"
        )
        try:
            outcome = _wait_for_completed(ctx, inputs)
        except AmcApiError as exc:
            return StepResult(status=StepStatus.FAILED, message=str(exc))
        if outcome is not waitui.WaitOutcome.SATISFIED:
            return StepResult(
                status=StepStatus.USER_ACTION_REQUIRED,
                message="AMC calibration is not complete yet",
                user_actions=_wait_hints(ctx, inputs),
            )
        return StepResult(status=StepStatus.COMPLETE)

    def run(self, ctx: "Context") -> StepResult:
        inputs, missing = resolve_project_inputs(ctx)
        if inputs is None:
            return _configuration_action(
                ctx, missing, camera_missing=_camera_inventory_missing(ctx)
            )
        ctx.progress.phase(2)
        ctx.progress.task(f"downloading AMC result for {inputs.project_name}")
        result = _wire_download(ctx, inputs, allow_prompt=True)
        if result.status is not StepStatus.COMPLETE:
            return result
        ctx.progress.phase(3)
        ctx.progress.task("installing automatic AMC result polling")
        try:
            _install_reingest_units(ctx, inputs=inputs)
        except (AmcApiError, OSError, ValueError) as exc:
            return StepResult(status=StepStatus.FAILED, message=str(exc))
        return result

    def verify(self, ctx: "Context") -> StepResult:
        inputs, missing = resolve_project_inputs(ctx)
        if inputs is None:
            return StepResult(
                status=StepStatus.FAILED, message=f"missing {', '.join(missing)}"
            )
        calibration_dir = pathlib.Path(
            ctx.conf.get(CONF_CALIBRATION_DIR_KEY)
            or default_calibration_dir(ctx, inputs.location_id)
        )
        if not (calibration_dir / TRANSFORMS_NAME).is_file():
            return StepResult(
                status=StepStatus.FAILED,
                message=f"calibration dir {calibration_dir} has no {TRANSFORMS_NAME}",
            )
        deepstream_dir = ctx.install_dir / "deepstream"
        tracker_path = deepstream_dir / TRACKER_YAML_TEMPLATE_NAME
        if not tracker_path.is_file():
            return StepResult(
                status=StepStatus.FAILED, message=f"missing {tracker_path}"
            )
        tracker = _parse_tracker_yaml(tracker_path.read_text(encoding="utf-8"))
        if tracker is None:
            return StepResult(
                status=StepStatus.FAILED,
                message=f"could not parse {tracker_path} as YAML",
            )
        projection = tracker.get("ObjectModelProjection") or {}
        model_files = projection.get("cameraModelFilepath") or []
        camera_count = len(_load_enabled_cameras(ctx))
        if not isinstance(model_files, list) or len(model_files) != camera_count:
            return StepResult(
                status=StepStatus.FAILED,
                message=(
                    "ObjectModelProjection.cameraModelFilepath must list one camera "
                    f"model per enabled camera ({camera_count})"
                ),
            )
        for model_file in model_files:
            if not pathlib.Path(str(model_file)).is_file():
                return StepResult(
                    status=StepStatus.FAILED, message=f"missing camera model {model_file}"
                )
        checks = (
            (
                "ObjectModelProjection.outputFootLocation",
                str(projection.get("outputFootLocation", "")),
                "1",
            ),
        )
        for label, actual, expected_value in checks:
            if not ctx.verify_pinned(label, actual, expected_value):
                return StepResult(
                    status=StepStatus.FAILED, message=f"{label} changed unexpectedly"
                )
        rendered = deepstream_dir / RENDERED_APP_CONFIG_NAME
        if not rendered.is_file():
            return StepResult(status=StepStatus.FAILED, message=f"missing {rendered}")
        rendered_text = rendered.read_text(encoding="utf-8")
        if (
            "${" in rendered_text
            or "ll-config-file=config_tracker_NvMOT.yml" not in rendered_text
        ):
            return StepResult(
                status=StepStatus.FAILED, message=f"{rendered} is not fully wired"
            )
        for index in range(len(_load_enabled_cameras(ctx))):
            if "uri=rtsp://" not in _source_block(rendered_text, index):
                return StepResult(
                    status=StepStatus.FAILED, message=f"[source{index}] has no RTSP URI"
                )
        for name in (
            INFER_PRIMARY_TEMPLATE_NAME,
            MSGCONV_TEMPLATE_NAME,
            f"{MV3DT_DIR_NAME}/{PUB_SUB_CONFIG_NAME}",
            f"{MV3DT_DIR_NAME}/{MQTT_CONFIG_NAME}",
        ):
            if not (deepstream_dir / name).is_file():
                return StepResult(
                    status=StepStatus.FAILED, message=f"missing {deepstream_dir / name}"
                )
        reid_model = _reid_model_path(ctx)
        if not reid_model.is_file() or reid_model.stat().st_size == 0:
            return StepResult(
                status=StepStatus.FAILED, message=f"missing ReID model {reid_model}"
            )
        bodypose_model = _bodypose_model_path(ctx)
        if not bodypose_model.is_file() or bodypose_model.stat().st_size == 0:
            return StepResult(
                status=StepStatus.FAILED,
                message=f"missing BodyPose3DNet model {bodypose_model}",
            )
        slug = _slugify(inputs.project_name, location_id=inputs.location_id)
        timer = f"mv3dt-ingest-{slug}.timer"
        service = f"mv3dt-ingest-{slug}.service"
        if (
            not (systemd.UNIT_DIR / timer).is_file()
            or not (systemd.UNIT_DIR / service).is_file()
        ):
            return StepResult(
                status=StepStatus.FAILED, message="re-ingest systemd units are missing"
            )
        timer_enabled = systemd.is_enabled(timer, runner=_systemd_runner(ctx))
        if _auto_ingest_on(ctx) and not timer_enabled:
            return StepResult(
                status=StepStatus.FAILED, message=f"{timer} is not enabled"
            )
        if not _auto_ingest_on(ctx) and timer_enabled:
            return StepResult(
                status=StepStatus.FAILED,
                message=f"{timer} is enabled but MV3DT_AUTO_INGEST is off",
            )
        if systemd.is_enabled(service, runner=_systemd_runner(ctx)):
            return StepResult(
                status=StepStatus.FAILED, message=f"{service} must not be enabled"
            )
        legacy_path = f"mv3dt-ingest-{slug}.path"
        if (systemd.UNIT_DIR / legacy_path).exists() or systemd.is_enabled(
            legacy_path, runner=_systemd_runner(ctx)
        ):
            return StepResult(
                status=StepStatus.FAILED,
                message=f"obsolete {legacy_path} is still installed or enabled",
            )
        return StepResult(status=StepStatus.COMPLETE)

    def report(self, ctx: "Context") -> None:
        inputs, _ = resolve_project_inputs(ctx)
        if inputs is None:
            return
        calibration_dir = pathlib.Path(
            ctx.conf.get(CONF_CALIBRATION_DIR_KEY)
            or default_calibration_dir(ctx, inputs.location_id)
        )
        ctx.log.info(
            "Calibration output wired.\n"
            f"  AMC project:        {inputs.project_name} ({inputs.project_id})\n"
            f"  Calibration dir:    {calibration_dir}\n"
            f"  Required transform: {calibration_dir / TRANSFORMS_NAME}\n"
            f"  Rendered pipeline:  {ctx.install_dir / 'deepstream' / RENDERED_APP_CONFIG_NAME}\n"
            "\nNext: Step 5 builds the project-named DeepStream launcher."
        )


register(Step4CalibOutputWiring())


def _build_ingest_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(prog="mv3dt-installer ingest", add_help=True)
    parser.add_argument("--project", default=None)
    parser.add_argument("--install-dir", default=None)
    parser.add_argument("--non-interactive", action="store_true")
    parser.add_argument("--log-dir", default=None)
    return parser


def handle_ingest_subcommand(argv: list, ctx: "Context") -> int:
    args = _build_ingest_arg_parser().parse_args(argv)
    inputs, missing = resolve_project_inputs(ctx)
    if inputs is None or _camera_inventory_missing(ctx):
        ctx.log.error(
            f"ingest: configuration incomplete: {', '.join(missing) or 'CAMERAS_FILE'}"
        )
        return 1
    if args.project and args.project != inputs.project_name:
        ctx.log.warn(
            f"ingest: --project {args.project!r} does not match configured "
            f"PROJECT_NAME {inputs.project_name!r}; using the configured project"
        )
    try:
        state_name = _project_state(ctx, inputs)
    except AmcApiError as exc:
        ctx.log.error(f"ingest: {exc}")
        return 1
    if state_name != "COMPLETED":
        ctx.log.info(
            f"ingest: AMC project {inputs.project_id} is {state_name}; nothing to ingest"
        )
        return 0
    result = _wire_download(ctx, inputs, allow_prompt=False, skip_unchanged=True)
    if result.status is not StepStatus.COMPLETE:
        ctx.log.error(f"ingest: {result.message}")
        return 1
    return 0


app_mod.register_subcommand("ingest", handle_ingest_subcommand)
