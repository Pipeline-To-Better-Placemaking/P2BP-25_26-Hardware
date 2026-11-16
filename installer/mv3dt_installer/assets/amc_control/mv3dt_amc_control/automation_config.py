"""Settings for automating AMC, loaded from YAML.

The base file, amc_automation.example.yaml, ships inside this package. A
workstation changes settings with an override file that holds only the keys
it changes; it is merged on top of the base. The unit points
AMC_AUTOMATION_CONFIG at `<install_dir>/amc_control/amc_automation.yaml`,
which lives outside the package so an upgrade never replaces it. Unknown keys
are rejected so a typo fails loudly instead of being ignored.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

PACKAGE_DIR = Path(__file__).resolve().parent
BASE_CONFIG = PACKAGE_DIR / "amc_automation.example.yaml"
CONFIG_DIR = PACKAGE_DIR  # relative paths resolve here when there is no override file


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AmcSection(_Strict):
    start_if_down: bool = True
    ready_timeout_seconds: float = Field(1800, ge=0)
    api_base_url: str = ""
    request_timeout_seconds: float = Field(60, gt=0)
    upload_timeout_seconds: float = Field(600, gt=0)


class ProjectSection(_Strict):
    mode: Literal["reuse_installer_project", "create_new"] = "reuse_installer_project"
    name_template: str = "{title}-{date}"


class VideosSection(_Strict):
    enabled: bool = True
    source: str = "installer"
    pattern: str = "*.mp4"
    order: list[str] = []


class LayoutSection(_Strict):
    enabled: bool = True
    source: Literal["database", "file"] = "database"
    file: str = ""
    allowed_hosts: list[str] = []
    max_bytes: int = Field(25 * 1024 * 1024, gt=0)


class ScaleSection(_Strict):
    source: Literal["database", "fixed", "none"] = "database"
    fixed_px_per_m: float = Field(0, ge=0)
    settings_key: str = "layout_px_per_m"


class ParametersSection(_Strict):
    enabled: bool = True
    settings_file: str = ""
    settings_overrides: dict[str, Any] = {}
    scale: ScaleSection = ScaleSection()
    focal_lengths: list[float] = []

    @field_validator("focal_lengths")
    @classmethod
    def _positive(cls, v: list[float]) -> list[float]:
        if any(f <= 0 for f in v):
            raise ValueError("focal lengths must be positive")
        return v


class RequestSpec(_Strict):
    method: Literal["GET", "POST", "PUT"] = "POST"
    path: str = ""
    json_body: dict[str, Any] = Field(default_factory=dict, alias="json")

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class StatusSpec(_Strict):
    method: Literal["GET", "POST"] = "GET"
    path: str = ""
    state_field: str = ""
    done_states: list[str] = ["COMPLETED"]
    failed_states: list[str] = ["ERROR", "FAILED"]


class RectificationApi(_Strict):
    start: RequestSpec = RequestSpec()
    status: StatusSpec = StatusSpec()
    poll_seconds: float = Field(10, ge=0)
    timeout_seconds: float = Field(1800, gt=0)


class RectificationSection(_Strict):
    mode: Literal["ui", "api", "skip"] = "ui"
    api: RectificationApi = RectificationApi()


class AlignmentSection(_Strict):
    file: str = "auto"
    coord_space: Literal["rectified", "original"] = "rectified"


class CalibrationSection(_Strict):
    detector_type: Literal["resnet", "transformer"] = "resnet"
    poll_seconds: float = Field(10, ge=0)
    timeout_seconds: float = Field(5400, gt=0)


class ResultsSection(_Strict):
    download_dir: str = ""
    result_type: Literal["amc", "vggt"] = "amc"


class AutomationConfig(_Strict):
    amc: AmcSection = AmcSection()
    project: ProjectSection = ProjectSection()
    videos: VideosSection = VideosSection()
    layout: LayoutSection = LayoutSection()
    parameters: ParametersSection = ParametersSection()
    rectification: RectificationSection = RectificationSection()
    alignment: AlignmentSection = AlignmentSection()
    calibration: CalibrationSection = CalibrationSection()
    results: ResultsSection = ResultsSection()

    # Folder that relative paths in the config are resolved against.
    base_dir: Path = Field(default=CONFIG_DIR, exclude=True)

    def resolve(self, path: str) -> Path:
        p = Path(os.path.expanduser(path))
        return p if p.is_absolute() else (self.base_dir / p)


class ConfigError(ValueError):
    pass


def deep_merge(base: dict, override: dict) -> dict:
    """Return base with override merged in; nested dicts merge, everything else replaces."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _read_yaml(path: Path) -> dict:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as e:
        raise ConfigError(f"could not read {path}: {e}") from e
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a mapping at the top level")
    return data


def load_config(override_path: str | Path | None = None) -> AutomationConfig:
    """Base file merged with the override file (if one is named and exists), validated.

    A named override that does not exist counts as empty, so the service
    works before anyone has written overrides; relative paths then resolve
    against the folder the override would be in.
    """
    data = _read_yaml(BASE_CONFIG)
    chosen = override_path or os.environ.get("AMC_AUTOMATION_CONFIG") or ""
    base_dir = CONFIG_DIR
    if chosen:
        chosen = Path(chosen)
        base_dir = chosen.resolve().parent
        if chosen.exists():
            data = deep_merge(data, _read_yaml(chosen))
    try:
        cfg = AutomationConfig.model_validate(data)
    except Exception as e:  # pydantic.ValidationError, shown as a readable message
        raise ConfigError(f"invalid AMC automation config: {e}") from e
    cfg.base_dir = base_dir
    return cfg
