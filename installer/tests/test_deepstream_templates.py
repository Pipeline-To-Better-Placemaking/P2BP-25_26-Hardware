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


def test_tracker_has_no_parameters_deepstream_9_1_rejects():
    lines = _setting_lines(TEMPLATES / "config_tracker_NvMOT.yml")
    for key in ("minTrackerConfidence", "maxTargetsPerStream", "SearchRegionPaddingScale"):
        assert not any(line.strip().startswith(f"{key}:") for line in lines), key


def test_app_tracker_group_has_no_removed_keys():
    lines = _setting_lines(TEMPLATES / "deepstream_app_config.txt")
    for key in ("enable-batch-process", "enable-past-frame"):
        assert not any(line.startswith(f"{key}=") for line in lines), key


def test_msgconv_defines_every_camera_index():
    text = (TEMPLATES / "msgconv_config.txt").read_text(encoding="utf-8")
    for group in ("sensor", "place", "analytics"):
        for index in range(0, 9):
            assert f"[{group}{index}]" in text, f"{group}{index}"
