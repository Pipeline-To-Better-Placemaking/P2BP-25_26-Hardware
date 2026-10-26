"""Guards on the DeepStream pipeline templates the binary bundles.

Each check pins a fix found on the live DeepStream 9.1 workstation on
2026-10-20, so the same breakage cannot return unnoticed.
"""

from __future__ import annotations

import pathlib
import re

TEMPLATES = pathlib.Path(__file__).resolve().parents[2] / "laptop" / "deepstream"
TEMPLATE_FILES = sorted(
    [*TEMPLATES.glob("*.txt"), *TEMPLATES.glob("*.yml")], key=lambda p: p.name
)
VERSIONED_DS_PATH = re.compile(r"/opt/nvidia/deepstream/deepstream-\d")


def _setting_lines(path: pathlib.Path) -> list[str]:
    return [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def test_templates_exist():
    names = {p.name for p in TEMPLATE_FILES}
    assert {
        "deepstream_app_config.txt",
        "config_tracker_NvMOT.yml",
        "config_infer_primary.txt",
        "msgconv_config.txt",
    } <= names


def test_no_version_specific_deepstream_paths():
    """The 9.0 paths broke the 9.1 pipeline. Use the version-neutral
    /opt/nvidia/deepstream/deepstream symlink the SDK installs."""
    offenders = [
        f"{path.name}: {line.strip()}"
        for path in TEMPLATE_FILES
        for line in _setting_lines(path)
        if VERSIONED_DS_PATH.search(line)
    ]
    assert offenders == []


def test_peoplenet_output_names_match_the_onnx_tensors():
    text = (TEMPLATES / "config_infer_primary.txt").read_text(encoding="utf-8")
    assert "output-blob-names=output_cov/Sigmoid:0;output_bbox/BiasAdd:0" in text
    assert "infer-dims=3;544;960" in text
    assert not re.search(r"(?m)^input-dims=", text)


def test_tracker_reid_matches_the_tao_model():
    text = (TEMPLATES / "config_tracker_NvMOT.yml").read_text(encoding="utf-8")
    reid = re.search(r"(?ms)^ReID:\n.*?(?=^\S)", text).group(0)
    assert re.search(r"(?m)^  inferDims: \[3, 256, 128\]", reid)
    assert re.search(r"(?m)^  inputOrder: 0", reid)
    assert 'tltEncodedModel: "models/reid/resnet50_market1501.etlt"' in reid
    assert 'tltModelKey: "nvidia_tao"' in reid


def _sections(path: pathlib.Path) -> dict:
    sections, current = {}, None
    for line in _setting_lines(path):
        if not line.startswith((" ", "\t")):
            current = line.split(":", 1)[0]
            sections[current] = []
        elif current:
            sections[current].append(line.strip())
    return sections


def test_tracker_has_no_parameters_deepstream_9_1_rejects():
    sections = _sections(TEMPLATES / "config_tracker_NvMOT.yml")
    base = sections.get("BaseConfig", [])
    for key in ("minTrackerConfidence", "maxTargetsPerStream"):
        assert not any(line.startswith(f"{key}:") for line in base), key
    visual = sections.get("VisualTracker", [])
    assert not any(line.startswith("SearchRegionPaddingScale:") for line in visual)


def test_tracker_configures_single_view_3d_the_way_deepstream_reads_it():
    """DeepStream silently ignores unknown sections: the old SV3DT:/MV3DT:
    sections were never read. Single-view 3D is ObjectModelProjection."""
    sections = _sections(TEMPLATES / "config_tracker_NvMOT.yml")
    assert "SV3DT" not in sections and "MV3DT" not in sections
    projection = sections["ObjectModelProjection"]
    assert "- ${CAMERA_MODEL_FILEPATH}" in projection
    assert any(line.startswith("outputFootLocation: 1") for line in projection)


def test_tracker_baseline_matches_the_live_tuned_workstation():
    """The template baseline is the tuned config running on the live
    workstation: BodyPose3DNet pose estimation and MQTT multi-view fusion."""
    sections = _sections(TEMPLATES / "config_tracker_NvMOT.yml")
    # useBatchNumForFrameId 1 deadlocked MV3DT startup on live RTSP starts.
    assert "useBatchNumForFrameId: 0" in sections["BaseConfig"]
    projection = sections["ObjectModelProjection"]
    assert "minPoseConfidence: 0.5" in projection
    pose = sections["PoseEstimator"]
    assert any(line.startswith("poseEstimatorType: 1") for line in pose)
    assert any(line.startswith("poseInferenceInterval: 5") for line in pose)
    bodypose = "/opt/mv3dt/deepstream/models/bodypose3dnet/bodypose3dnet_accuracy.onnx"
    assert any(line.startswith(f'onnxFile: "{bodypose}"') for line in pose)
    engine = f'modelEngineFile: "{bodypose}_b1_gpu0_fp16.engine"'
    assert any(line.startswith(engine) for line in pose)
    associator = sections["MultiViewAssociator"]
    for setting in (
        "multiViewAssociatorType: 1",
        "enableLatePeerReAssoc: 1",
        "enableIDCorrection: 1",
        "enableMsgSync: 0",
        "communicatorInitSleepTime: 0",
    ):
        assert setting in associator, setting
    communicator = sections["Communicator"]
    assert any(line.startswith("communicatorType: 2") for line in communicator)
    assert (
        "pubSubInfoConfigPath: /opt/mv3dt/deepstream/mv3dt/pub_sub_info_config.yml"
        in communicator
    )
    assert (
        "mqttProtoAdaptorConfigPath: /opt/mv3dt/deepstream/mv3dt/config_mqtt.txt"
        in communicator
    )


def test_detector_and_app_baseline_match_the_live_tuned_workstation():
    infer = _setting_lines(TEMPLATES / "config_infer_primary.txt")
    assert "pre-cluster-threshold=0.4" in infer
    app = (TEMPLATES / "deepstream_app_config.txt").read_text(encoding="utf-8")
    tracker = re.search(r"(?ms)^\[tracker\]\n.*?(?=^\[|\Z)", app).group(0)
    assert re.search(r"(?m)^display-tracking-id=1$", tracker)


def test_tracker_size_is_a_multiple_of_32():
    text = (TEMPLATES / "deepstream_app_config.txt").read_text(encoding="utf-8")
    width = int(re.search(r"(?m)^tracker-width=(\d+)", text).group(1))
    height = int(re.search(r"(?m)^tracker-height=(\d+)", text).group(1))
    assert width % 32 == 0 and height % 32 == 0, (width, height)


def test_message_sink_uses_the_3d_protobuf_converter():
    text = (TEMPLATES / "deepstream_app_config.txt").read_text(encoding="utf-8")
    assert "msg-conv-payload-type=2" in text
    assert "msg-conv-msg2p-lib=/opt/nvidia/deepstream/deepstream/lib/libnvds_msgconv_mega.so" in text
    assert "msg-conv-frame-interval=1" in text


def test_detector_reports_people_only():
    text = (TEMPLATES / "config_infer_primary.txt").read_text(encoding="utf-8")
    assert "filter-out-class-ids=1;2" in text


def test_app_tracker_group_has_no_removed_keys():
    lines = _setting_lines(TEMPLATES / "deepstream_app_config.txt")
    for key in ("enable-batch-process", "enable-past-frame"):
        assert not any(line.startswith(f"{key}=") for line in lines), key


def test_msgconv_defines_every_camera_index():
    text = (TEMPLATES / "msgconv_config.txt").read_text(encoding="utf-8")
    for group in ("sensor", "place", "analytics"):
        for index in range(0, 9):
            assert f"[{group}{index}]" in text, f"{group}{index}"
