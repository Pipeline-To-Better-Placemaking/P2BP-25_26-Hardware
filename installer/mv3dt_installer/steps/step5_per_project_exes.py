"""Step 5 -- Start-or-close + per-project executables.

Implements `installer/plan/STEP-5-PER-PROJECT-EXES.md` against the
framework contract in `installer/plan/00-FRAMEWORK-AND-BOOTSTRAP.md`
(step-module interface section 12, `Context` section 12.3, atomic-write
helper section 6.3, logging/reporting section 8, privilege/USER-ACTION
section 9, install-location config section 11, camera discovery section
15), the subcommand dispatch extension `step3_amc_launcher.py` builds in
`app.py` (`SUBCOMMAND_REGISTRY`/`register_subcommand`), and the AMC exe
`step3_amc_launcher.py` already drops at `<install_dir>/bin/amc`.

Scope: after Step 4 has rendered a per-project DeepStream config and
established `PROJECT_NAME`/`LOCATION_ID`, Step 5 (a) generates a
project-named `pipeline-<slug>` exe plus a `record-<slug>` tracking-recorder
exe (section 3), (b) records the project in a persistent registry (section
4), (c) offers the operator the final start-or-close choice (section 2),
and (d) exposes the reusable reconciliation logic the AMC exe's `[X]
Remove` menu item and `verify()`'s drift check both need (section 5.4).

**Step 4 integration note.** Step 4 (`step4_calib_output_wiring.py`) is
being built in parallel by a different unit and is not merged into this
branch. Per this unit's own instructions, Step 5 does not import anything
from it; it consumes Step 4's outputs purely through the documented
contract -- `installer.conf` keys (`PROJECT_NAME`, `LOCATION_ID`,
`CALIBRATION_DIR`) and files under `<install_dir>/deepstream/`
(STEP-5 doc section 1.1, STEP-4 doc section 7). STEP-4 doc section 6.2
renders a single, fixed `<install_dir>/deepstream/
deepstream_app_config.rendered.txt` (not the per-slug subdirectory the
STEP-5 doc's own section 4.2 registry example shows) -- `resolve_rendered_config`
below honors an optional `RENDERED_CONFIG` override key first, then falls
back to that fixed path, so this module works against Step 4's actual
documented output regardless of which shape lands.

**AMC menu wiring note.** STEP-5 doc section 5 describes the standalone
`amc` exe growing an `[N]/[R]/[L]/[X]` menu that drives new-project,
re-run, list, and remove flows through this module's registry API.
`step3_amc_launcher.py` (which owns the `amc` exe/subcommand) is read-only
for this unit, so that menu is not wired up here. What this module
provides instead is every piece STEP-5 section 5 says that menu should
call -- `upsert`/`get`/`list_projects` (section 4.3), `write_pipeline_wrapper`
/`write_record_wrapper` (section 3), and `reconcile_registry` (section
5.4) -- plus its own standalone entry points (`mv3dt-installer projects
--list/--reconcile/--remove`) so the reconciliation and listing
requirements are met even before a later unit wires the `amc` menu itself.

**Step 6 handoff (STEP-6 doc section A.2 -- IMPLEMENTED, not out of scope).**
Systemd supervision of the per-project pipeline is no longer future work:
`step6_remote_supervision.py` supervises exactly what this module registers,
via the templated `mv3dt-pipeline@<slug>.service` instance unit. The default
`pipeline` runtime behavior (ensure mosquitto, ping-sweep, source the
DeepStream env, `exec deepstream-app -c <config>`) still lives entirely in
`_start_pipeline_foreground`, a function kept separate from the subcommand's
argument parsing and dispatch (`handle_pipeline_subcommand`) so it could be
swapped without touching argument parsing. `handle_pipeline_subcommand` now
branches at runtime (`_supervision_active`, below) between three bodies:
`_start_pipeline_foreground`/`_stop_pipeline` (unsupervised -- what every
workstation that never opted into Step 6 keeps getting, unchanged),
`systemctl start/stop mv3dt-pipeline@<slug>` (supervised -- the new default
once Step 6 is COMPLETE and active), and `_start_pipeline_service_exec`
(`--service-exec`, what the unit's own `ExecStart=` invokes). `--foreground`
is the retained debugging escape hatch that always takes the unsupervised
path regardless of Step 6's state.

Every subprocess call goes through `ctx.run_root`/`ctx.run_as_user` (doc 00
section 9.2), so no test here shells out to a real systemctl/ping/docker or
execs a real `deepstream-app`.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import pathlib
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional

from mv3dt_installer import app as app_mod
from mv3dt_installer import cameras as cameras_mod
from mv3dt_installer import config as config_mod
from mv3dt_installer import shellout
from mv3dt_installer import systemd as systemd_mod
from mv3dt_installer.state import write_json_atomic
from mv3dt_installer.steps import StepResult, StepStatus, UserAction, register
from mv3dt_installer.steps import step3_amc_launcher as step3_mod

if TYPE_CHECKING:  # pragma: no cover -- import-time only, never at runtime.
    from mv3dt_installer.app import Context

__all__ = [
    "ProjectEntry",
    "Registry",
    "registry_path",
    "load_registry",
    "save_registry",
    "slugify",
    "resolve_slug",
    "upsert",
    "get",
    "get_by_slug",
    "list_projects",
    "remove_registry_entry",
    "resolve_rendered_config",
    "render_pipeline_wrapper",
    "render_record_wrapper",
    "write_pipeline_wrapper",
    "write_record_wrapper",
    "ensure_mosquitto",
    "ping_sweep_cameras",
    "source_deepstream_env",
    "generate_preview_config",
    "render_validation_banner",
    "amc_project_dir",
    "amc_project_present",
    "amc_projects_root",
    "reconcile_registry",
    "remove_project_artifacts",
    "register_removal_hook",
    "handle_pipeline_subcommand",
    "handle_projects_subcommand",
    "handle_record_subcommand",
    "Step5PerProjectExes",
]

# ---------------------------------------------------------------------------
# section 3.2 / 3.1 -- naming
# ---------------------------------------------------------------------------

PIPELINE_PREFIX = "pipeline-"
RECORD_PREFIX = "record-"
INSTALLER_BIN_NAME = "mv3dt-installer"
AMC_WRAPPER_NAME = "amc"

# STEP-4 doc section 7 conf keys this module consumes (contract, not import).
CONF_PROJECT_NAME_KEY = "PROJECT_NAME"
CONF_LOCATION_ID_KEY = "LOCATION_ID"
CONF_CALIBRATION_DIR_KEY = "CALIBRATION_DIR"
# Optional override; STEP-4 doc section 6.2's fixed rendered-config path is
# the default when this is unset (see module docstring's Step 4 note).
CONF_RENDERED_CONFIG_KEY = "RENDERED_CONFIG"
CONF_CAMERAS_YML_KEY = "CAMERAS_YML"

DEFAULT_RENDERED_CONFIG_NAME = "deepstream_app_config.rendered.txt"

_SLUG_INVALID_RE = re.compile(r"[^a-z0-9]+")
_SLUG_MAX_LEN = 64


def slugify(name: str, *, fallback: str = "") -> str:
    """Section 3.1's sanitization: lower-case; collapse any run of
    non-`[a-z0-9]` chars to a single `-`; strip leading/trailing `-`;
    truncate to 64 chars. Falls back to `fallback` (typically
    `LOCATION_ID`), then to the literal `"project"`, when the result would
    otherwise be empty (e.g. the name was all punctuation)."""

    def _one(value: str) -> str:
        collapsed = _SLUG_INVALID_RE.sub("-", value.lower()).strip("-")
        return collapsed[:_SLUG_MAX_LEN].strip("-")

    slug = _one(name)
    if slug:
        return slug
    if fallback:
        slug = _one(fallback)
        if slug:
            return slug
    return "project"


# ---------------------------------------------------------------------------
# section 4 -- the project registry
# ---------------------------------------------------------------------------

REGISTRY_SCHEMA_VERSION = 1
_REGISTRY_RELATIVE_PATH = pathlib.Path("projects") / "registry.json"


def _now_utc_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class ProjectEntry:
    """One `registry.json["projects"][PROJECT_NAME]` record (section 4.2)."""

    project_name: str
    slug: str
    location_id: str
    exe: str
    rendered_config: str
    calibration_dir: str
    cameras_yml: Optional[str] = None
    created_utc: str = ""
    updated_utc: str = ""
    calib_runs: int = 1

    @property
    def record_exe(self) -> str:
        """`<install_dir>/bin/record-<slug>` (STEP-5 doc section 1.2),
        derived by convention rather than stored in the registry.

        STEP-5 doc section 4.2 pins the registry schema exactly, and does
        not list a `record_exe` field -- an earlier revision of this module
        stored one anyway (a silent deviation a reviewer flagged, since
        Step 7 also reads `registry.json` and should only ever see the
        documented fields). `write_record_wrapper` always writes to this
        exact path (`RECORD_PREFIX + slug` under `<install_dir>/bin/`), so
        recomputing it here is equivalent to storing it and keeps the
        on-disk schema exactly as documented.
        """
        return str(pathlib.Path(self.exe).parent / f"{RECORD_PREFIX}{self.slug}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "slug": self.slug,
            "location_id": self.location_id,
            "exe": self.exe,
            "rendered_config": self.rendered_config,
            "calibration_dir": self.calibration_dir,
            "cameras_yml": self.cameras_yml,
            "created_utc": self.created_utc,
            "updated_utc": self.updated_utc,
            "calib_runs": self.calib_runs,
        }

    @staticmethod
    def from_dict(project_name: str, d: dict[str, Any]) -> "ProjectEntry":
        return ProjectEntry(
            project_name=project_name,
            slug=d.get("slug", ""),
            location_id=d.get("location_id", ""),
            exe=d.get("exe", ""),
            rendered_config=d.get("rendered_config", ""),
            calibration_dir=d.get("calibration_dir", ""),
            cameras_yml=d.get("cameras_yml"),
            created_utc=d.get("created_utc", ""),
            updated_utc=d.get("updated_utc", ""),
            calib_runs=int(d.get("calib_runs", 1)),
        )


@dataclass
class Registry:
    schema_version: int = REGISTRY_SCHEMA_VERSION
    updated_utc: str = ""
    projects: dict[str, ProjectEntry] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "updated_utc": self.updated_utc,
            "projects": {name: e.to_dict() for name, e in self.projects.items()},
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Registry":
        projects_raw = d.get("projects", {})
        projects = {
            name: ProjectEntry.from_dict(name, entry)
            for name, entry in projects_raw.items()
            if isinstance(entry, dict)
        }
        return Registry(
            schema_version=int(d.get("schema_version", REGISTRY_SCHEMA_VERSION)),
            updated_utc=d.get("updated_utc", ""),
            projects=projects,
        )


def registry_path(install_dir: pathlib.Path) -> pathlib.Path:
    return pathlib.Path(install_dir) / _REGISTRY_RELATIVE_PATH


def load_registry(install_dir: pathlib.Path) -> Registry:
    """Forgiving reader (section 4.1): a missing or malformed file yields
    the empty registry, never an exception."""
    path = registry_path(install_dir)
    try:
        text = path.read_text(encoding="utf-8")
        data = json.loads(text)
    except (OSError, ValueError):
        return Registry()
    if not isinstance(data, dict):
        return Registry()
    return Registry.from_dict(data)


def save_registry(install_dir: pathlib.Path, registry: Registry) -> None:
    """Atomic write (section 4.1) via the shared framework helper."""
    registry.updated_utc = _now_utc_iso()
    path = registry_path(install_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, registry.to_dict())
    try:
        path.parent.chmod(0o755)
        path.chmod(0o644)
    except OSError:
        pass  # best-effort, e.g. under a non-root test process


def resolve_slug(
    install_dir: pathlib.Path,
    project_name: str,
    location_id: str,
    *,
    registry: Optional[Registry] = None,
) -> tuple[str, Optional[str]]:
    """Section 3.1's collision policy. Returns `(slug, error)` -- `error`
    is set only on an unresolvable collision (a distinct project already
    owns both the base slug and the `-<location_id>` fallback)."""
    registry = registry if registry is not None else load_registry(install_dir)
    base_slug = slugify(project_name, fallback=location_id)

    existing = registry.projects.get(project_name)
    if existing is not None:
        # Same registry key re-running (section 5.2) keeps its own slug.
        return existing.slug, None

    def _held_by_other(slug: str) -> bool:
        return any(
            entry.slug == slug for name, entry in registry.projects.items() if name != project_name
        )

    if not _held_by_other(base_slug):
        return base_slug, None

    alt_slug = f"{base_slug}-{slugify(location_id)}" if location_id else base_slug
    if _held_by_other(alt_slug):
        return alt_slug, (
            f"pipeline-{alt_slug} already exists for a different project; "
            "pick a distinct AMC project name"
        )
    return alt_slug, None


def upsert(
    install_dir: pathlib.Path,
    *,
    project_name: str,
    location_id: str,
    rendered_config: str,
    calibration_dir: str,
    exe: str,
    cameras_yml: Optional[str] = None,
    slug: Optional[str] = None,
) -> ProjectEntry:
    """Section 4.3: create or update the `project_name` entry, computing/
    validating the slug, bumping timestamps + `calib_runs`."""
    registry = load_registry(install_dir)
    now = _now_utc_iso()
    existing = registry.projects.get(project_name)

    resolved_slug = slug or (existing.slug if existing else slugify(project_name, fallback=location_id))

    if existing is not None:
        entry = ProjectEntry(
            project_name=project_name,
            slug=resolved_slug,
            location_id=location_id,
            exe=exe,
            rendered_config=rendered_config,
            calibration_dir=calibration_dir,
            cameras_yml=cameras_yml if cameras_yml is not None else existing.cameras_yml,
            created_utc=existing.created_utc or now,
            updated_utc=now,
            calib_runs=existing.calib_runs + 1,
        )
    else:
        entry = ProjectEntry(
            project_name=project_name,
            slug=resolved_slug,
            location_id=location_id,
            exe=exe,
            rendered_config=rendered_config,
            calibration_dir=calibration_dir,
            cameras_yml=cameras_yml,
            created_utc=now,
            updated_utc=now,
            calib_runs=1,
        )

    registry.projects[project_name] = entry
    save_registry(install_dir, registry)
    return entry


def get(install_dir: pathlib.Path, project_name: str) -> Optional[ProjectEntry]:
    return load_registry(install_dir).projects.get(project_name)


def get_by_slug(install_dir: pathlib.Path, slug: str) -> Optional[ProjectEntry]:
    """Slug-keyed lookup (STEP-6 doc section A.1): the registry itself is
    keyed by `PROJECT_NAME`, but `mv3dt-pipeline@<slug>.service`'s `%i` only
    ever carries the slug, so `--project-slug` (the flag the unit's own
    `ExecStart=`/`ExecStop=` pass) needs this instead of `get()`."""
    for entry in load_registry(install_dir).projects.values():
        if entry.slug == slug:
            return entry
    return None


def list_projects(install_dir: pathlib.Path) -> list[ProjectEntry]:
    return list(load_registry(install_dir).projects.values())


def remove_registry_entry(install_dir: pathlib.Path, project_name: str) -> bool:
    """Drop `project_name` from the registry (atomic write). Returns
    whether an entry was actually present to remove."""
    registry = load_registry(install_dir)
    if project_name not in registry.projects:
        return False
    del registry.projects[project_name]
    save_registry(install_dir, registry)
    return True


# ---------------------------------------------------------------------------
# STEP-4 doc section 6.2/7 -- rendered config resolution (contract, not import)
# ---------------------------------------------------------------------------


def resolve_rendered_config(ctx: "Context") -> pathlib.Path:
    """See this module's docstring's "Step 4 integration note": prefers an
    explicit `RENDERED_CONFIG` override, else STEP-4 doc section 6.2's
    fixed output path."""
    override = ctx.conf.get(CONF_RENDERED_CONFIG_KEY)
    if override:
        return pathlib.Path(override)
    return pathlib.Path(ctx.install_dir) / "deepstream" / DEFAULT_RENDERED_CONFIG_NAME


# ---------------------------------------------------------------------------
# section 3.2 -- generated wrapper exes
# ---------------------------------------------------------------------------


def render_pipeline_wrapper(installer_bin: pathlib.Path, project_name: str, location_id: str) -> str:
    """Section 3.2, verbatim."""
    return (
        "#!/usr/bin/env bash\n"
        "# Auto-generated by mv3dt-installer Step 5 (step5_per_project_exes).\n"
        f"# Project: {project_name}   LOCATION_ID: {location_id}\n"
        "# Do not edit; regenerate by re-running the installer or the AMC re-run flow.\n"
        f'exec "{installer_bin}" pipeline --project "{project_name}" "$@"\n'
    )


def render_record_wrapper(installer_bin: pathlib.Path, project_name: str, location_id: str) -> str:
    """Section 1.2: the tracking-recorder wrapper, generated in the same
    form as `render_pipeline_wrapper`."""
    return (
        "#!/usr/bin/env bash\n"
        "# Auto-generated by mv3dt-installer Step 5 (step5_per_project_exes).\n"
        f"# Project: {project_name}   LOCATION_ID: {location_id}\n"
        "# Do not edit; regenerate by re-running the installer or the AMC re-run flow.\n"
        f'exec "{installer_bin}" record --project "{project_name}" "$@"\n'
    )


def _write_generated_wrapper(
    ctx: "Context", dest: pathlib.Path, content: str
) -> tuple[pathlib.Path, bool]:
    """Shared write for `pipeline-<slug>`/`record-<slug>` (mirrors
    `step3_amc_launcher.write_amc_wrapper`'s content-idempotent pattern).
    Returns `(path, changed)`."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    changed = True
    if dest.is_file():
        try:
            changed = dest.read_text(encoding="utf-8") != content
        except OSError:
            changed = True

    if changed:
        dest.write_text(content, encoding="utf-8")

    dest.chmod(0o755)
    try:
        os.chown(dest, ctx.user.uid, ctx.user.gid)
    except OSError:
        pass  # best-effort, e.g. under a non-root test process

    return dest, changed


def write_pipeline_wrapper(
    ctx: "Context", installer_bin: pathlib.Path, project_name: str, location_id: str, slug: str
) -> tuple[pathlib.Path, bool]:
    dest = pathlib.Path(ctx.install_dir) / "bin" / f"{PIPELINE_PREFIX}{slug}"
    content = render_pipeline_wrapper(installer_bin, project_name, location_id)
    return _write_generated_wrapper(ctx, dest, content)


def write_record_wrapper(
    ctx: "Context", installer_bin: pathlib.Path, project_name: str, location_id: str, slug: str
) -> tuple[pathlib.Path, bool]:
    dest = pathlib.Path(ctx.install_dir) / "bin" / f"{RECORD_PREFIX}{slug}"
    content = render_record_wrapper(installer_bin, project_name, location_id)
    return _write_generated_wrapper(ctx, dest, content)


# ---------------------------------------------------------------------------
# section 3.3 step 2 -- ensure mosquitto
# ---------------------------------------------------------------------------


def ensure_mosquitto(ctx: "Context") -> bool:
    """Idempotent `systemctl start mosquitto` (port of
    `50_start_pipeline.sh`'s `ensure_mosquitto()`). Returns whether the
    broker is active afterward. Never raises -- a failure here only warns,
    matching the bash precedent's `die` being downgraded to a caller
    decision (the pipeline still tries to start; MQTT publish just won't
    reach a broker)."""
    active = ctx.run_root(
        "systemctl", "is-active", "--quiet", "mosquitto", check=False, capture_output=True, text=True
    )
    if active.returncode == 0:
        ctx.log.info("mosquitto is already active.")
        return True

    ctx.log.info("Starting mosquitto (systemctl start mosquitto)")
    ctx.run_root("systemctl", "start", "mosquitto", check=False, capture_output=True, text=True)
    recheck = ctx.run_root(
        "systemctl", "is-active", "--quiet", "mosquitto", check=False, capture_output=True, text=True
    )
    if recheck.returncode != 0:
        ctx.log.warn(
            "mosquitto failed to start. Run 'journalctl -u mosquitto' and re-check "
            "Step 1's mosquitto setup if needed."
        )
        return False
    return True


# ---------------------------------------------------------------------------
# section 3.3 step 3 -- ping sweep (CAMERAS_FILE, doc 00 section 15)
# ---------------------------------------------------------------------------


def ping_sweep_cameras(ctx: "Context") -> None:
    """Port of `50_start_pipeline.sh`'s ping-sweep, reading the discovered
    inventory named by `CAMERAS_FILE` (doc 00 section 11.2/15) rather than
    a compiled-in path. Warns and never blocks; a missing/unset
    `CAMERAS_FILE` is a warning, not a failure (the pipeline still starts --
    a stale inventory must not block it)."""
    cameras_file = ctx.conf.get("CAMERAS_FILE")
    if not cameras_file:
        ctx.log.warn(
            "CAMERAS_FILE not set; skipping ping-sweep. Run the camera discovery "
            "scan (--scan-cameras) to populate it."
        )
        return

    path = pathlib.Path(cameras_file)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        ctx.log.warn(f"CAMERAS_FILE ({path}) not found; skipping ping-sweep.")
        return

    cameras = cameras_mod.parse_inventory(text)
    miss = 0
    for cam in cameras:
        if not cam.enabled:
            ctx.log.info(f"  {cam.id:<4} {cam.ip:<16} SKIP (disabled)")
            continue
        result = ctx.run_root(
            "ping", "-c", "1", "-W", "2", cam.ip, check=False, capture_output=True, text=True
        )
        if result.returncode == 0:
            ctx.log.info(f"  {cam.id:<4} {cam.ip:<16} OK")
        else:
            ctx.log.info(f"  {cam.id:<4} {cam.ip:<16} MISS")
            miss += 1

    if miss:
        ctx.log.warn(f"{miss} enabled camera(s) did not respond to ping. Continuing.")


# ---------------------------------------------------------------------------
# section 3.3 step 4 -- source the DeepStream env
# ---------------------------------------------------------------------------

DEEPSTREAM_PROFILE = pathlib.Path("/etc/profile.d/deepstream.sh")

#: The DeepStream app the pipeline runs. NVIDIA's MV3DT reference is built on
#: deepstream-test5-app, which attaches the per-object event metadata the
#: msgconv/msgbroker sink publishes; plain deepstream-app connects the sink
#: but never gives it anything to send (workstation run, 2026-10-20). It
#: reads the same config file format.
DEEPSTREAM_APP = "deepstream-test5-app"

#: Linux truncates a process name to 15 characters, and `pgrep -x`/`pkill -x`
#: match that truncated name.
DEEPSTREAM_APP_PROCESS = DEEPSTREAM_APP[:15]


def source_deepstream_env(ctx: "Context") -> dict[str, str]:
    """Best-effort `. /etc/profile.d/deepstream.sh; env` diff, returned as
    an overlay dict for the eventual `deepstream-app` exec. An absent
    profile or a failing shell yields `{}` (a warning, never fatal --
    `_start_pipeline_foreground`'s caller decides what to do with a missing
    `deepstream-app`)."""
    result = ctx.run_root(
        "bash",
        "-c",
        f'set +u; [ -f "{DEEPSTREAM_PROFILE}" ] && . "{DEEPSTREAM_PROFILE}"; env',
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        ctx.log.warn(f"{DEEPSTREAM_PROFILE} not found or failed to source.")
        return {}

    overlay: dict[str, str] = {}
    for line in (result.stdout or "").splitlines():
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key and key not in os.environ or os.environ.get(key) != value:
            overlay[key] = value
    return overlay


# ---------------------------------------------------------------------------
# section 3.3 -- --preview (port of 50_start_pipeline.sh's
# generate_preview_config())
# ---------------------------------------------------------------------------

_SECTION_HEADER_RE = re.compile(r"^\[([^\]]+)\]\s*$")
_SECTION_KEY_RE = re.compile(r"^([A-Za-z0-9_-]+)\s*=.*$")
_SOURCE_SECTION_RE = re.compile(r"^source\d+$", re.IGNORECASE)
_ENABLE_KV_RE = re.compile(r"^enable\s*=\s*(.+)$", re.IGNORECASE)


def _section_ranges(lines: list[str]) -> list[tuple[str, int, int]]:
    starts = [
        (m.group(1), idx) for idx, line in enumerate(lines) if (m := _SECTION_HEADER_RE.match(line.strip()))
    ]
    ranges = []
    for i, (name, start) in enumerate(starts):
        end = starts[i + 1][1] if i + 1 < len(starts) else len(lines)
        ranges.append((name, start, end))
    return ranges


def _set_section_values(lines: list[str], section_name: str, updates: dict[str, str]) -> None:
    ranges = _section_ranges(lines)
    target = next(((s, e) for name, s, e in ranges if name.lower() == section_name.lower()), None)

    if target is None:
        if lines and lines[-1].strip():
            lines.append("")
        lines.append(f"[{section_name}]")
        for key, value in updates.items():
            lines.append(f"{key}={value}")
        return

    start, end = target
    key_positions: dict[str, int] = {}
    for idx in range(start + 1, end):
        m = _SECTION_KEY_RE.match(lines[idx].strip())
        if m and m.group(1) not in key_positions:
            key_positions[m.group(1)] = idx

    for key, value in updates.items():
        if key in key_positions:
            lines[key_positions[key]] = f"{key}={value}"

    missing = [k for k in updates if k not in key_positions]
    for offset, key in enumerate(missing):
        lines.insert(end + offset, f"{key}={updates[key]}")


def _count_enabled_sources(lines: list[str]) -> int:
    count = 0
    for name, start, end in _section_ranges(lines):
        if not _SOURCE_SECTION_RE.match(name):
            continue
        enabled = True
        for idx in range(start + 1, end):
            m = _ENABLE_KV_RE.match(lines[idx].strip())
            if m:
                enabled = m.group(1).strip().lower() not in ("0", "false", "no")
                break
        if enabled:
            count += 1
    return max(count, 1)


def generate_preview_config(source_text: str, *, source_label: str = "<config>") -> str:
    """Port of `50_start_pipeline.sh`'s `generate_preview_config()`
    (`--preview`, section 3.3): deterministically patches a display-enabled
    tiled-display/sink0/osd profile onto `source_text` while leaving
    `sink1` (MQTT publish) enabled, and returns the rendered text."""
    lines = source_text.splitlines()
    source_count = _count_enabled_sources(lines)
    rows = math.ceil(math.sqrt(source_count))
    cols = math.ceil(source_count / rows)

    _set_section_values(
        lines,
        "tiled-display",
        {"enable": "1", "rows": str(rows), "columns": str(cols), "width": "1920", "height": "1080"},
    )
    _set_section_values(lines, "sink0", {"enable": "1", "type": "2", "sync": "0", "gpu-id": "0"})
    _set_section_values(lines, "sink1", {"enable": "1"})
    _set_section_values(
        lines,
        "osd",
        {
            "enable": "1",
            "gpu-id": "0",
            "border-width": "2",
            "text-size": "16",
            "text-color": "1;1;1;1",
            "text-bg-color": "0;0;0;0.5",
            "font": "Serif",
            "show-clock": "0",
            "clock-x-offset": "40",
            "clock-y-offset": "20",
        },
    )

    header = (
        "# Auto-generated by mv3dt-installer pipeline --preview.\n"
        f"# Source config: {source_label}\n"
        f"# Generated UTC: {_now_utc_iso()}\n"
        f"# Active sources: {source_count}  tiled rows x cols: {rows} x {cols}\n\n"
    )
    return header + "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# section 6 -- validation banner
# ---------------------------------------------------------------------------


def render_validation_banner(entry: ProjectEntry, *, topic_base: str = "mv3dt") -> str:
    """Section 6, verbatim wording."""
    return (
        "\n[validation helpers -- SCRIPTED-WORKFLOW section 10.2]\n"
        f"  # Per-frame 3D track messages (binary protobuf, nv.Frame; topic base {topic_base}):\n"
        f"  mosquitto_sub -h 127.0.0.1 -t '{topic_base}/#' -F '%t %l bytes'\n\n"
        "  # GPU utilization / memory / temperature:\n"
        "  watch -n 1 'nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv'\n\n"
        "  # Broker health:\n"
        "  systemctl status mosquitto --no-pager\n\n"
        f"  # Stop this pipeline:\n  {entry.exe} --stop\n"
    )


# ---------------------------------------------------------------------------
# STEP-6 doc section A.2 -- is Step 6 actually supervising this project?
# ---------------------------------------------------------------------------


def _pipeline_unit_name(slug: str) -> str:
    return f"mv3dt-pipeline@{slug}.service"


def _supervision_active(ctx: "Context", slug: str) -> bool:
    """Whether `handle_pipeline_subcommand` should hand `start`/`stop` off
    to systemd (STEP-6 doc section A.2) instead of running the unsupervised
    foreground/pkill path it always used before Step 6 existed.

    Two real runtime checks, both required -- this is deliberately not a
    hardcoded "Step 6 exists so always supervise" assumption:

    1. The operator actually opted in (`MV3DT_REMOTE_SUPERVISION` != "off",
       config.py's already-resolved gate, doc 00 section 3.4). A workstation
       that never touched the gate keeps behaving exactly as it did before
       this unit -- the default is "off", so an absent/missing key also
       means unsupervised.
    2. Step 6 has actually installed and enabled the templated instance unit
       for *this* project. The gate can be flipped on before Step 6's own
       `run()` has ever executed (or before this particular project existed
       when it did), in which case `systemctl start mv3dt-pipeline@<slug>`
       would simply fail against a unit that was never enabled -- so this
       probes the live unit via `systemd.is_enabled` (through the same
       injected `ctx.run_root` every other systemctl call in this module
       uses) rather than trusting the gate alone.
    """
    gate = ctx.conf.get(config_mod.GATE_REMOTE_SUPERVISION, "off")
    if not gate or gate == "off":
        return False
    return systemd_mod.is_enabled(
        _pipeline_unit_name(slug),
        runner=lambda argv, **kwargs: ctx.run_root(*argv, **kwargs),
    )


# ---------------------------------------------------------------------------
# section 3.3 -- the internal "what to run" routine
# ---------------------------------------------------------------------------


def _start_pipeline_foreground(
    ctx: "Context",
    entry: ProjectEntry,
    *,
    preview: bool = False,
    config_override: Optional[str] = None,
    skip_ping: bool = False,
    dry_run: bool = False,
    execv: Callable[[str, list], None] = os.execv,
) -> int:
    """Section 3.3 steps 2-5: ensure mosquitto, ping-sweep, source the
    DeepStream env, print the validation banner, then hand off to
    `deepstream-app -c <config>` (a real `os.execv` unless `dry_run`)."""
    config_path = pathlib.Path(config_override) if config_override else pathlib.Path(entry.rendered_config)
    if not config_path.is_file():
        ctx.log.error(f"rendered config not found: {config_path}")
        return 1

    ensure_mosquitto(ctx)

    if not skip_ping:
        ping_sweep_cameras(ctx)
    else:
        ctx.log.info("--skip-ping set; skipping camera ping-sweep.")

    env_overlay = source_deepstream_env(ctx)

    if preview:
        try:
            source_text = config_path.read_text(encoding="utf-8")
        except OSError as exc:
            ctx.log.error(f"cannot read {config_path} for --preview: {exc}")
            return 1
        preview_path = config_path.with_name("deepstream_app_config.preview.txt")
        preview_path.write_text(
            generate_preview_config(source_text, source_label=str(config_path)), encoding="utf-8"
        )
        ctx.log.info(f"Preview mode ON: generated {preview_path}")
        config_path = preview_path

    ctx.log.info(render_validation_banner(entry))
    ctx.log.info(f"Launching: {DEEPSTREAM_APP} -c {config_path.name} (cwd: {config_path.parent})")

    argv = [DEEPSTREAM_APP, "-c", str(config_path)]
    if dry_run:
        ctx.log.info(f"[dry-run] cd {config_path.parent} && {' '.join(argv)}")
        return 0

    os.chdir(config_path.parent)
    child_env = {**os.environ, **env_overlay}
    binary = shutil.which(argv[0], path=child_env.get("PATH")) or argv[0]
    try:
        execv(binary, argv)
    except OSError as exc:
        ctx.log.error(f"failed to exec {DEEPSTREAM_APP}: {exc}")
        return 1
    return 0  # pragma: no cover -- execv never returns on success


def _viewer_stream_on(ctx: "Context") -> bool:
    """`MV3DT_VIEWER_STREAM`, read from installer.conf (`ctx.conf`) and, when
    absent there, from the environment the unit's `EnvironmentFile=` sets."""
    key = config_mod.GATE_VIEWER_STREAM
    value = ctx.conf.get(key) or os.environ.get(key) or "off"
    return value.strip().lower() == "on"


def _start_pipeline_service_exec(
    ctx: "Context",
    entry: ProjectEntry,
    *,
    config_override: Optional[str] = None,
    dry_run: bool = False,
    execv: Callable[[str, list], None] = os.execv,
) -> int:
    """STEP-6 doc section A.1's `ExecStart=` body (`--service-exec`): the
    same ensure-mosquitto -> source-env -> exec deepstream-app steps as
    `_start_pipeline_foreground`, minus the operator-facing ping-sweep and
    validation banner -- this runs non-interactively inside
    `mv3dt-pipeline@<slug>.service`, logging to journald rather than a TTY
    (doc section A.1's notes: "no ping-sweep gating, no validation-helper
    banner ... those are for the operator's foreground TTY"). Operators
    never invoke this mode directly; only the unit's own `ExecStart=` does.
    """
    config_path = pathlib.Path(config_override) if config_override else pathlib.Path(entry.rendered_config)
    if not config_path.is_file():
        ctx.log.error(f"rendered config not found: {config_path}")
        return 1

    ensure_mosquitto(ctx)
    env_overlay = source_deepstream_env(ctx)

    argv = [DEEPSTREAM_APP, "-c", str(config_path)]
    if _viewer_stream_on(ctx):
        # Tiled view labels for the VLC viewer stream Step 4 renders.
        argv.insert(1, "-t")
    if dry_run:
        ctx.log.info(f"[dry-run] cd {config_path.parent} && {' '.join(argv)}")
        return 0

    os.chdir(config_path.parent)
    child_env = {**os.environ, **env_overlay}
    binary = shutil.which(argv[0], path=child_env.get("PATH")) or argv[0]
    try:
        execv(binary, argv)
    except OSError as exc:
        ctx.log.error(f"failed to exec {DEEPSTREAM_APP}: {exc}")
        return 1
    return 0  # pragma: no cover -- execv never returns on success


def _start_pipeline_supervised(
    ctx: "Context", entry: ProjectEntry, *, dry_run: bool = False
) -> int:
    """STEP-6 doc section A.2's new `start` default once Step 6 is active:
    `systemctl start mv3dt-pipeline@<slug>` via `ctx.run_root`, instead of
    execing `deepstream-app` in the operator's own shell."""
    unit = _pipeline_unit_name(entry.slug)
    if dry_run:
        ctx.log.info(f"[dry-run] systemctl start {unit}")
        return 0

    result = ctx.run_root("systemctl", "start", unit, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        ctx.log.error(f"systemctl start {unit} failed (exit {result.returncode})")
        return result.returncode
    ctx.log.info(f"Started {unit} (supervised by systemd; runs 24/7 and survives reboot).")
    return 0


# ---------------------------------------------------------------------------
# section 3.4 -- stopping the pipeline
# ---------------------------------------------------------------------------


_STOP_GRACE_POLLS = 5
_STOP_GRACE_INTERVAL_S = 1.0


def _pgrep_deepstream_running(ctx: "Context") -> bool:
    result = ctx.run_root("pgrep", "-x", DEEPSTREAM_APP_PROCESS, check=False, capture_output=True, text=True)
    return result.returncode == 0


def _stop_pipeline(
    ctx: "Context",
    entry: ProjectEntry,
    *,
    stop_all: bool = False,
    skip_deepstream: bool = False,
    skip_amc: bool = False,
    skip_mosquitto: bool = False,
    sleep: Callable[[float], None] = time.sleep,
    grace_polls: int = _STOP_GRACE_POLLS,
    grace_interval_s: float = _STOP_GRACE_INTERVAL_S,
    use_systemctl: bool = False,
) -> int:
    """Port of `99_stop_all.sh` (section 3.4). `--stop` (the default) only
    stops `deepstream-app`; `--stop-all` additionally tears down the AMC
    stack and mosquitto, matching `99_stop_all.sh` run with no skip flags.

    The SIGTERM -> SIGKILL escalation mirrors `99_stop_all.sh` lines 62-78
    exactly: after sending SIGTERM, poll up to `grace_polls` times (default
    5, matching the bash `for _ in 1 2 3 4 5; do pgrep || break; sleep 1;
    done`), sleeping `grace_interval_s` between polls, before concluding the
    process is still alive and escalating to SIGKILL. `sleep`/`grace_polls`/
    `grace_interval_s` are injectable so a test can exercise the timing
    without a real wait.

    `use_systemctl` (STEP-6 doc section A.2, set by `handle_pipeline_subcommand`
    only when `_supervision_active` says so) swaps just the deepstream-stopping
    step for `systemctl stop mv3dt-pipeline@<slug>` -- the unit's own
    `KillSignal=SIGTERM` + `TimeoutStopSec=10` then reproduce the same
    graceful-then-forced shutdown systemd-side. `--stop-all`'s AMC/mosquitto
    teardown is unaffected either way (doc section A.2: "--stop-all still
    additionally stops the AMC stack / broker as in Step 5 section 3.4").
    """
    stop_amc = stop_all and not skip_amc
    stop_mosquitto = stop_all and not skip_mosquitto
    stop_deepstream = not skip_deepstream

    if stop_deepstream:
        if use_systemctl:
            unit = _pipeline_unit_name(entry.slug)
            ctx.log.info(f"Stopping {unit} (systemctl stop)")
            ctx.run_root("systemctl", "stop", unit, check=False, capture_output=True, text=True)
        elif _pgrep_deepstream_running(ctx):
            ctx.log.info(f"Stopping {DEEPSTREAM_APP} (SIGTERM)")
            ctx.run_root("pkill", "-TERM", "-x", DEEPSTREAM_APP_PROCESS, check=False, capture_output=True, text=True)

            still_running = True
            for _ in range(grace_polls):
                if not _pgrep_deepstream_running(ctx):
                    still_running = False
                    break
                sleep(grace_interval_s)

            if still_running:
                ctx.log.warn(f"{DEEPSTREAM_APP} still running after SIGTERM; sending SIGKILL")
                ctx.run_root(
                    "pkill", "-KILL", "-x", DEEPSTREAM_APP_PROCESS, check=False, capture_output=True, text=True
                )
        else:
            ctx.log.info(f"{DEEPSTREAM_APP} not running.")

    if stop_amc:
        step3_mod.teardown_amc(ctx)

    if stop_mosquitto:
        active = ctx.run_root(
            "systemctl", "is-active", "--quiet", "mosquitto", check=False, capture_output=True, text=True
        )
        if active.returncode == 0:
            ctx.log.info("Stopping mosquitto (systemctl stop mosquitto)")
            ctx.run_root("systemctl", "stop", "mosquitto", check=False, capture_output=True, text=True)
        else:
            ctx.log.info("mosquitto not active.")

    return 0


# ---------------------------------------------------------------------------
# section 3.3/3.4 -- the `pipeline` subcommand
# ---------------------------------------------------------------------------


def _build_pipeline_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mv3dt-installer pipeline", add_help=True)
    parser.add_argument("--project", default=None)
    # STEP-6 doc section A.1: mv3dt-pipeline@.service.in's ExecStart=/ExecStop=
    # only ever carry systemd's own `%i` (the slug), never PROJECT_NAME --
    # get_by_slug() is what this resolves against. Exactly one of --project /
    # --project-slug is required (checked below, not via argparse mutual
    # exclusivity, so the "neither given" error message can name both).
    parser.add_argument("--project-slug", default=None)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--config", default=None)
    parser.add_argument("--skip-ping", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--stop-all", action="store_true")
    parser.add_argument("--no-deepstream", action="store_true")
    parser.add_argument("--no-amc", action="store_true")
    parser.add_argument("--no-mosquitto", action="store_true")
    # STEP-6 doc section A.2: the unit's own ExecStart= mode (operators never
    # pass this) and the retained foreground debugging escape hatch, which
    # always takes the pre-Step-6 unsupervised path regardless of the gate.
    parser.add_argument("--service-exec", action="store_true")
    parser.add_argument("--foreground", action="store_true")
    # doc 00 section 3.3-shaped framework flags a subcommand's own argv may
    # carry (mirrors step3_amc_launcher's identical passthrough).
    parser.add_argument("--install-dir", default=None)
    parser.add_argument("--non-interactive", action="store_true")
    parser.add_argument("--log-dir", default=None)
    return parser


def _resolve_pipeline_entry(argv_project: Optional[str], argv_slug: Optional[str], ctx: "Context"):
    """Resolve the `--project`/`--project-slug` pair `handle_pipeline_subcommand`
    was called with into a `(entry, label)` pair, or `(None, label)` when
    unresolvable. Exactly one of the two is expected; `--project-slug` is
    what the systemd unit actually passes (STEP-6 doc section A.1)."""
    if argv_slug:
        return get_by_slug(ctx.install_dir, argv_slug), argv_slug
    if argv_project:
        return get(ctx.install_dir, argv_project), argv_project
    return None, None


def handle_pipeline_subcommand(argv: list, ctx: "Context") -> int:
    """`mv3dt-installer pipeline --project <NAME> [...]` (section 3.3/3.4),
    now also `--project-slug <SLUG>` (STEP-6 doc section A.1's `%i`) and the
    supervised/unsupervised branch STEP-6 doc section A.2 adds."""
    args = _build_pipeline_arg_parser().parse_args(argv)

    entry, label = _resolve_pipeline_entry(args.project, args.project_slug, ctx)
    if label is None:
        ctx.log.error("pipeline: one of --project or --project-slug is required")
        return 1
    if entry is None:
        ctx.log.error(
            f"unknown project '{label}'; run the amc exe to create/calibrate it first"
        )
        return 1

    if args.service_exec:
        return _start_pipeline_service_exec(
            ctx, entry, config_override=args.config, dry_run=args.dry_run
        )

    # STEP-6 doc section A.2's runtime conditional: --foreground always
    # forces the pre-Step-6 unsupervised path; otherwise it's whatever
    # _supervision_active actually finds on this workstation right now.
    supervised = not args.foreground and _supervision_active(ctx, entry.slug)

    if args.stop or args.stop_all:
        return _stop_pipeline(
            ctx,
            entry,
            stop_all=args.stop_all,
            skip_deepstream=args.no_deepstream,
            skip_amc=args.no_amc,
            skip_mosquitto=args.no_mosquitto,
            use_systemctl=supervised,
        )

    if supervised:
        return _start_pipeline_supervised(ctx, entry, dry_run=args.dry_run)

    return _start_pipeline_foreground(
        ctx,
        entry,
        preview=args.preview,
        config_override=args.config,
        skip_ping=args.skip_ping,
        dry_run=args.dry_run,
    )


# ---------------------------------------------------------------------------
# section 1.2 -- the `record` subcommand (owner of 60_record_tracking.sh)
# ---------------------------------------------------------------------------


def _build_record_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mv3dt-installer record", add_help=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--install-dir", default=None)
    parser.add_argument("--non-interactive", action="store_true")
    parser.add_argument("--log-dir", default=None)
    return parser


def handle_record_subcommand(
    argv: list,
    ctx: "Context",
    *,
    run_bundled_script: Callable[..., subprocess.CompletedProcess] = shellout.run_bundled_script,
) -> int:
    """`mv3dt-installer record --project <NAME>` (section 1.2): runs the
    bundled `60_record_tracking.sh` for the project's `LOCATION_ID`,
    writing under `<install_dir>/tracking_exports/`.

    **Pending asset note:** per this repo's CLAUDE.md, the bundled copy of
    `60_record_tracking.sh` lands under `installer/mv3dt_installer/assets/
    scripts/` with plan unit U12 (`feat/installer-bundled-scripts`), not yet
    merged -- that directory currently holds only `.gitkeep`. This handler
    is written against the same `run_bundled_script("scripts",
    "60_record_tracking.sh", ...)` contract `step1_prerequisites.py` already
    uses for `10_setup_mosquitto.sh`, so it starts working the moment U12
    lands with no change here.
    """
    known, rest = _build_record_arg_parser().parse_known_args(argv)

    entry = get(ctx.install_dir, known.project)
    if entry is None:
        ctx.log.error(f"unknown project '{known.project}'; run the amc exe to create/calibrate it first")
        return 1

    exports_dir = pathlib.Path(ctx.install_dir) / "tracking_exports" / entry.slug
    exports_dir.mkdir(parents=True, exist_ok=True)

    result = run_bundled_script(
        "scripts",
        "60_record_tracking.sh",
        args=list(rest),
        env={
            "LOCATION_ID": entry.location_id,
            "MV3DT_TRACKING_EXPORT_DIR": str(exports_dir),
        },
        tree=(),
    )
    if result.returncode != 0:
        ctx.log.error(f"tracking recorder exited {result.returncode}")
        return result.returncode
    return 0


# ---------------------------------------------------------------------------
# section 5.4 -- AMC-GUI-driven removal + reconciliation
# ---------------------------------------------------------------------------


def _env_value(path: pathlib.Path, key: str) -> Optional[str]:
    """`key`'s value in a dotenv file, or `None` when the file or key is
    missing. Surrounding quotes are dropped; a value that still holds a
    `$` reference is not expanded and is reported as missing."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        name, sep, value = line.partition("=")
        if not sep or name.strip() != key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        if not value or "$" in value:
            return None
        return value
    return None


def amc_projects_root(ctx: "Context") -> pathlib.Path:
    """Where AMC keeps its project directories on the host.

    AMC 3.3.0 mounts `PROJECT_DIR` from `<AMC_ROOT>/compose/.env`, which an
    operator may point anywhere; that value wins when it is set. Without it
    the older `<AMC_ROOT>/projects` layout is assumed. A relative value is
    taken relative to the compose directory, as `docker compose` does.
    """
    cfg = step3_mod.resolve_config(ctx)
    compose_dir = cfg.amc_root / "compose"
    value = _env_value(compose_dir / ".env", "PROJECT_DIR")
    if value:
        root = pathlib.Path(value).expanduser()
        return root if root.is_absolute() else compose_dir / root
    return cfg.amc_root / "projects"


def amc_project_dir(ctx: "Context", project_name: str) -> pathlib.Path:
    """`<projects root>/<PROJECT_NAME>/` (section 5.4), where the projects
    root is `amc_projects_root` (compose `PROJECT_DIR`, else
    `$AMC_ROOT/projects`)."""
    return amc_projects_root(ctx) / project_name


def _api_project_listing(
    ctx: "Context",
) -> Optional[tuple[Optional[set[str]], set[str]]]:
    """Ask the running AMC service which projects it has.

    Returns `(names, ids)`, or `None` when the service is not answering or
    its reply cannot be read. `names` is `None` when the reply carries ids
    only, so the caller does not mistake "no names listed" for "no project
    by that name".
    """
    cfg = step3_mod.resolve_config(ctx)
    result = ctx.run_root(
        "curl", "-fsS", "--max-time", "5",
        f"http://localhost:{cfg.ms_port}/v1/list_projects",
        check=False, capture_output=True, text=True, stream=False,
    )
    if result.returncode != 0:
        return None
    try:
        payload = json.loads(result.stdout or "")
    except json.JSONDecodeError:
        return None

    items: Any = payload
    if isinstance(payload, dict):
        if "code" in payload and payload.get("code") != 0:
            return None
        items = None
        for key in ("projects", "project_list", "project_info", "data", "result"):
            if isinstance(payload.get(key), list):
                items = payload[key]
                break
    if not isinstance(items, list):
        return None

    names: set[str] = set()
    ids: set[str] = set()
    for item in items:
        if isinstance(item, str):
            ids.add(item)
        elif isinstance(item, dict):
            ident = item.get("project_id") or item.get("id")
            name = item.get("project_name") or item.get("name")
            if ident:
                ids.add(str(ident))
            if name:
                names.add(str(name))
    return (names if names else None), ids


def amc_project_present(ctx: "Context", project_name: str) -> bool:
    """Whether AMC still has the project registered as `project_name`.

    AMC 3.x stores a project under its opaque `AMC_PROJECT_ID`, not under
    its name, so a name-shaped directory is no evidence either way for the
    configured project. The running service is asked first
    (`GET /v1/list_projects`): a matching name, or the configured project's
    `AMC_PROJECT_ID`, means present. When the service is down the projects
    root on disk is checked for a directory named after the project or
    after its `AMC_PROJECT_ID`.
    """
    configured_id = ""
    if project_name == ctx.conf.get(CONF_PROJECT_NAME_KEY):
        configured_id = (ctx.conf.get(step3_mod.CONF_AMC_PROJECT_ID_KEY) or "").strip()

    listing = _api_project_listing(ctx)
    if listing is not None:
        names, ids = listing
        if configured_id and configured_id in ids:
            return True
        if names is not None:
            return project_name in names

    candidates = [amc_project_dir(ctx, project_name)]
    if configured_id:
        candidates.append(amc_projects_root(ctx) / configured_id)
    return any(path.exists() for path in candidates)


def _project_artifact_paths(ctx: "Context", entry: ProjectEntry) -> list[pathlib.Path]:
    """Every install-side artifact section 5.4 says removal must sweep."""
    install_dir = pathlib.Path(ctx.install_dir)
    return [
        pathlib.Path(entry.exe),
        install_dir / "bin" / f"{RECORD_PREFIX}{entry.slug}",
        pathlib.Path(entry.rendered_config),
        pathlib.Path(entry.calibration_dir),
        install_dir / "projects" / entry.slug,
    ]


# STEP-6 doc section A.3: "before deleting install-side artifacts, run
# systemctl disable --now mv3dt-pipeline@<slug>". step5 has no business
# knowing about Step 6's gate/systemd calls (this module is Step 6's
# dependency, not the other way around), so Step 6 registers a gate-aware
# callback here at import time via `register_removal_hook`, instead of this
# module importing step6_remote_supervision. `None` (the default) means "no
# Step 6 loaded" -- `remove_project_artifacts` is then exactly what it was
# before this hook existed.
_REMOVAL_HOOK: Optional[Callable[["Context", ProjectEntry], None]] = None


def register_removal_hook(hook: Optional[Callable[["Context", ProjectEntry], None]]) -> None:
    """Register (or clear, with `None`) the pre-removal callback STEP-6
    doc section A.3 requires. Call once at module import time from
    `step6_remote_supervision.py`; the hook itself decides whether Step 6
    is actually active (gate on + unit installed) before touching systemd,
    so `remove_project_artifacts` never has to ask."""
    global _REMOVAL_HOOK
    _REMOVAL_HOOK = hook


def remove_project_artifacts(ctx: "Context", entry: ProjectEntry) -> list[str]:
    """Remove every install-side artifact for `entry`, then drop its
    registry key. Missing artifacts are logged and skipped (idempotent, per
    section 5.4). Returns the paths actually removed, as strings, for the
    caller's report."""
    if _REMOVAL_HOOK is not None:
        _REMOVAL_HOOK(ctx, entry)

    removed: list[str] = []
    for path in _project_artifact_paths(ctx, entry):
        try:
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
                removed.append(str(path))
            elif path.exists() or path.is_symlink():
                path.unlink()
                removed.append(str(path))
        except OSError as exc:
            ctx.log.warn(f"could not remove {path}: {exc}")

    remove_registry_entry(ctx.install_dir, entry.project_name)
    return removed


def reconcile_registry(
    ctx: "Context",
    *,
    apply: bool = True,
    yes: bool = False,
    confirm: Callable[[str], bool] = lambda _msg: True,
) -> list[str]:
    """Section 5.4's reconciliation check.

    For each registered project, tests whether AMC still has it
    (`amc_project_present`). A project AMC no longer has was removed in the
    AMC GUI.

    - `apply=False` (the read-only `verify()` drift check, section 7.3):
      never deletes; just returns the list of drifted `PROJECT_NAME`s.
    - `apply=True` (the default -- `[X] Remove` / `projects --reconcile`):
      actually removes the install-side artifacts (`remove_project_artifacts`)
      for each drifted project. In interactive mode this lists what will be
      deleted and calls `confirm(...)`; under `ctx.non_interactive` or
      `yes=True` it proceeds without prompting (section 5.4 point 1).

    Returns the list of `PROJECT_NAME`s that were (or, under `apply=False`,
    would be) reconciled away.
    """
    drifted = [
        entry
        for entry in list_projects(ctx.install_dir)
        if not amc_project_present(ctx, entry.project_name)
    ]
    if not drifted:
        return []

    names = [e.project_name for e in drifted]

    if not apply:
        for name in names:
            ctx.log.info(f"drift: registry entry '{name}' has no matching AMC project dir")
        return names

    proceed = yes or ctx.non_interactive
    if not proceed:
        listing = "\n".join(f"  - {name}" for name in names)
        proceed = confirm(
            f"The following project(s) were removed in the AMC GUI and will have "
            f"their local artifacts deleted:\n{listing}\nProceed?"
        )

    if not proceed:
        ctx.log.info("Reconciliation skipped by operator.")
        return []

    for entry in drifted:
        removed = remove_project_artifacts(ctx, entry)
        ctx.log.info(f"reconciled '{entry.project_name}': removed {len(removed)} artifact(s)")

    return names


# ---------------------------------------------------------------------------
# `projects` subcommand -- list / reconcile / remove
# ---------------------------------------------------------------------------


def _format_projects_table(entries: list[ProjectEntry]) -> str:
    if not entries:
        return "(no registered projects)"
    lines = ["PROJECT_NAME\tslug\texe\tlocation_id\tupdated_utc\tcalib_runs"]
    for e in entries:
        lines.append(f"{e.project_name}\t{e.slug}\t{e.exe}\t{e.location_id}\t{e.updated_utc}\t{e.calib_runs}")
    return "\n".join(lines)


def _build_projects_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mv3dt-installer projects", add_help=True)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--reconcile", action="store_true")
    parser.add_argument("--remove", metavar="PROJECT_NAME", default=None)
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--install-dir", default=None)
    parser.add_argument("--non-interactive", action="store_true")
    parser.add_argument("--log-dir", default=None)
    return parser


def handle_projects_subcommand(argv: list, ctx: "Context") -> int:
    """`mv3dt-installer projects --list/--reconcile/--remove <NAME>` (section
    5.3/5.4)."""
    args = _build_projects_arg_parser().parse_args(argv)

    if args.remove:
        entry = get(ctx.install_dir, args.remove)
        if entry is None:
            ctx.log.error(f"unknown project '{args.remove}'")
            return 1
        removed = remove_project_artifacts(ctx, entry)
        ctx.log.info(f"removed '{args.remove}': {len(removed)} artifact(s)")
        return 0

    if args.reconcile:
        reconcile_registry(ctx, apply=True, yes=args.yes or ctx.non_interactive)
        return 0

    # --list (default when nothing else was asked for -- section 5.3).
    reconcile_registry(ctx, apply=False)
    ctx.log.info(_format_projects_table(list_projects(ctx.install_dir)))
    return 0


# ---------------------------------------------------------------------------
# section 2 -- the start-or-close prompt
# ---------------------------------------------------------------------------

# Injectable so tests never block on real input().
_INPUT: Callable[[str], str] = input


def _confirm_start_now(ctx: "Context") -> bool:
    """Section 2: `[S] Start` / `[C] Close`, default Close.
    `--non-interactive`/`--no-pause` default to Close (section 2.1's closing
    note) -- `Context` carries `non_interactive` only (mirrors
    `step3_amc_launcher._confirm_launch_now`'s identical reasoning)."""
    if ctx.non_interactive:
        return False
    answer = _INPUT("What next? [S]tart the pipeline now / [C]lose (default: C): ").strip().lower()
    return answer in ("s", "start")


# ---------------------------------------------------------------------------
# The Step
# ---------------------------------------------------------------------------


class Step5PerProjectExes:
    """STEP-5-PER-PROJECT-EXES.md section 7: module identity."""

    id = "step5_per_project_exes"
    title = "Per-project executables"

    # Doc 08 §3.1. This step holds 45 direct `log` calls, more than any
    # other, which is what made it the worst case for the erase arithmetic
    # in §12.2 defect 3; the phases give that output somewhere to belong.
    phases = (
        "project resolution",
        "pipeline and record wrappers",
        "project registry",
    )
    order = 5

    # -- preflight (section 7.1) --------------------------------------------

    def preflight(self, ctx: "Context") -> StepResult:
        project_name = ctx.conf.get(CONF_PROJECT_NAME_KEY)
        location_id = ctx.conf.get(CONF_LOCATION_ID_KEY)
        rendered_config = resolve_rendered_config(ctx)

        if not project_name or not location_id or not rendered_config.is_file():
            return StepResult(
                status=StepStatus.USER_ACTION_REQUIRED,
                message=(
                    "Step 4 outputs are not ready yet (PROJECT_NAME/LOCATION_ID/"
                    f"rendered config at {rendered_config})"
                ),
                user_actions=[
                    UserAction(text="Complete Step 4 (calibration output wiring) first.")
                ],
            )

        amc_exe = pathlib.Path(ctx.install_dir) / "bin" / AMC_WRAPPER_NAME
        if not amc_exe.is_file():
            return StepResult(
                status=StepStatus.USER_ACTION_REQUIRED,
                message=f"{amc_exe} missing; run Step 3 (AMC launcher) first",
                user_actions=[UserAction(text="Complete Step 3 (AutoMagicCalib launcher) first.")],
            )

        probe = ctx.run_root(
            "bash",
            "-c",
            f'set +u; [ -f "{DEEPSTREAM_PROFILE}" ] && . "{DEEPSTREAM_PROFILE}"; command -v {DEEPSTREAM_APP}',
            check=False,
            capture_output=True,
            text=True,
        )
        if probe.returncode != 0:
            return StepResult(
                status=StepStatus.USER_ACTION_REQUIRED,
                message=f"{DEEPSTREAM_APP} is not resolvable after sourcing /etc/profile.d/deepstream.sh",
                user_actions=[UserAction(text="Complete Step 2 (DeepStream SDK install) first.")],
            )

        return StepResult(status=StepStatus.COMPLETE)

    # -- run (section 7.2) ---------------------------------------------------

    def run(self, ctx: "Context") -> StepResult:
        ctx.progress.phase(1)
        ctx.progress.task("resolving project and location")
        project_name = ctx.conf.get(CONF_PROJECT_NAME_KEY, "")
        location_id = ctx.conf.get(CONF_LOCATION_ID_KEY, "")
        rendered_config = resolve_rendered_config(ctx)
        calibration_dir = ctx.conf.get(CONF_CALIBRATION_DIR_KEY) or str(
            pathlib.Path(ctx.install_dir) / "deepstream" / "calibration" / location_id
        )
        cameras_yml = ctx.conf.get(CONF_CAMERAS_YML_KEY)

        slug, err = resolve_slug(ctx.install_dir, project_name, location_id)
        if err:
            return StepResult(status=StepStatus.FAILED, message=err)

        ctx.progress.phase(2)
        ctx.progress.task("writing per-project wrappers")
        installer_bin = step3_mod.ensure_installer_binary(ctx)

        pipeline_path, pipeline_changed = write_pipeline_wrapper(
            ctx, installer_bin, project_name, location_id, slug
        )
        record_path, record_changed = write_record_wrapper(
            ctx, installer_bin, project_name, location_id, slug
        )

        ctx.progress.phase(3)
        ctx.progress.task("updating the project registry")
        entry = upsert(
            ctx.install_dir,
            project_name=project_name,
            location_id=location_id,
            rendered_config=str(rendered_config),
            calibration_dir=calibration_dir,
            exe=str(pipeline_path),
            cameras_yml=cameras_yml,
            slug=slug,
        )

        label = f"{PIPELINE_PREFIX}{slug}"
        if pipeline_changed:
            ctx.report_installed(label, f"{entry.calib_runs}")
        else:
            ctx.report_already_installed(label, f"{entry.calib_runs}")

        record_label = f"{RECORD_PREFIX}{slug}"
        if record_changed:
            ctx.report_installed(record_label, f"{entry.calib_runs}")
        else:
            ctx.report_already_installed(record_label, f"{entry.calib_runs}")

        ctx.log.info(
            f"Installation complete for project: {project_name}\n"
            f"A pipeline executable was created: {pipeline_path}\n\n"
            "What next?\n"
            f"  [S] Start the DeepStream pipeline for {project_name} now\n"
            "  [C] Close (you can start it later with the command above)"
        )

        if _confirm_start_now(ctx):
            ctx.log.info(f"Starting {pipeline_path} ...")
            try:
                os.execv(str(pipeline_path), [str(pipeline_path)])
            except OSError as exc:
                return StepResult(status=StepStatus.FAILED, message=f"failed to exec {pipeline_path}: {exc}")

        return StepResult(status=StepStatus.COMPLETE)

    # -- verify (section 7.3) -------------------------------------------------

    def verify(self, ctx: "Context") -> StepResult:
        project_name = ctx.conf.get(CONF_PROJECT_NAME_KEY, "")
        entry = get(ctx.install_dir, project_name)
        if entry is None:
            return StepResult(
                status=StepStatus.FAILED, message=f"no registry entry for '{project_name}'"
            )

        exe = pathlib.Path(entry.exe)
        if not exe.is_file() or not os.access(exe, os.X_OK):
            return StepResult(status=StepStatus.FAILED, message=f"{exe} missing or not executable")
        try:
            content = exe.read_text(encoding="utf-8")
        except OSError:
            content = ""
        if f'--project "{project_name}"' not in content:
            return StepResult(
                status=StepStatus.FAILED, message=f"{exe} does not reference --project {project_name!r}"
            )

        amc_exe = pathlib.Path(ctx.install_dir) / "bin" / AMC_WRAPPER_NAME
        if not amc_exe.is_file() or not os.access(amc_exe, os.X_OK):
            return StepResult(status=StepStatus.FAILED, message=f"{amc_exe} missing or not executable")

        if not pathlib.Path(entry.rendered_config).exists():
            return StepResult(
                status=StepStatus.FAILED, message=f"rendered_config missing: {entry.rendered_config}"
            )
        if not pathlib.Path(entry.calibration_dir).exists():
            return StepResult(
                status=StepStatus.FAILED, message=f"calibration_dir missing: {entry.calibration_dir}"
            )

        # Section 5.4: read-only removal drift check -- report, do not delete.
        reconcile_registry(ctx, apply=False)

        return StepResult(status=StepStatus.COMPLETE)

    # -- report (section 7.4) --------------------------------------------------

    def report(self, ctx: "Context") -> None:
        project_name = ctx.conf.get(CONF_PROJECT_NAME_KEY, "")
        entry = get(ctx.install_dir, project_name)
        if entry is None:
            ctx.log.info("Step 5: no registry entry to report.")
            return

        ctx.log.info(
            "Per-project executables installed.\n"
            f"  Pipeline exe:    {entry.exe}\n"
            f"  Recorder exe:    {entry.record_exe}\n"
            f"  AMC exe:         {pathlib.Path(ctx.install_dir) / 'bin' / AMC_WRAPPER_NAME}\n"
            f"  Registry:        {registry_path(ctx.install_dir)}\n"
            f"  Rendered config: {entry.rendered_config}\n"
            f"  Calibration dir: {entry.calibration_dir}\n\n"
            f"Start later:  {entry.exe}\n"
            f"Stop:         {entry.exe} --stop\n"
        )


def _pipeline_requires_root(argv: list) -> bool:
    """`pipeline`'s root requirement is mode-dependent (unit U6's fix for a
    PR #50 review defect): `--service-exec` is the one mode
    `mv3dt-pipeline@.service.in`'s own `ExecStart=` invokes, as the unit's
    non-root `User=@USER@` process (STEP-6 doc section A.2) -- so it must
    NOT require root. Every other mode (the default start path, `--stop`,
    `--stop-all`, `--foreground`, `--dry-run`) still calls
    `ctx.run_root`/expects to run under `sudo`, exactly as before Step 6
    existed, so it still requires root.

    A plain substring check on the raw argv (mirroring
    `app._bootstrap_subcommand_context()`'s own lightweight
    `parse_known_args` peek, rather than a full
    `_build_pipeline_arg_parser().parse_args(argv)`) is deliberate: this
    predicate runs before a `Context` exists, so `handle_pipeline_subcommand`
    itself is untouched -- this is purely a root-check seam in the
    registration, not a change to the subcommand's own dispatch logic.
    """
    return "--service-exec" not in argv


register(Step5PerProjectExes())
app_mod.register_subcommand(
    "pipeline", handle_pipeline_subcommand, requires_root=_pipeline_requires_root
)
app_mod.register_subcommand("record", handle_record_subcommand)
app_mod.register_subcommand("projects", handle_projects_subcommand)
