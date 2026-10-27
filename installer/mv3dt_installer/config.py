"""Install-location config + opt-in step gate persistence (doc 00 §11, §3.4).

Owns `<install_dir>/installer.conf` — a small, plain `KEY=VALUE` file (same
shape as [`laptop/config/laptop.env.example`](../../laptop/config/laptop.env.example):
no quoting, no shell expansion) that mirrors `state.json`'s `install_dir`
(§6.2) for human inspection and for bundled bash fragments
(`set -a; . installer.conf`). This module is framework-only: it has no
knowledge of what any step or later module (`ngc.py`, `webapp.py`, ...) does
with the directory once resolved — it only resolves *where* `install_dir`
is, creates that directory, and reads/writes `installer.conf`.

Precedence resolved by `load()` (doc 00 §11.2, exact order):

    --install-dir  >  state.json's install_dir  >  installer.conf's own
    recorded install dir (edge case: state.json was reset but a previous
    installer.conf survived at the default location)  >  hardcoded default
    `/opt/mv3dt` (DEFAULT_INSTALL_DIR, doc 00 §11.1).

When resolution falls through to the hardcoded default *and* the caller is
interactive (`non_interactive=False`), `load()` prompts once with that
default prefilled (doc 00 §11.1's "TUI prompts ... with /opt/mv3dt
prefilled"); `--non-interactive` uses the default silently.

`installer.conf` also carries the three §3.4 opt-in step-gate keys
(`MV3DT_REMOTE_SUPERVISION`, `MV3DT_WEBAPP_INTEGRATION`,
`MV3DT_TRACKING_HISTORY`) and the `MV3DT_VIEWER_STREAM` feature gate (Step 4
renders the tiled view, on-screen display and RTSP viewer sink only when it
is `on`; it gates no step). The *first* time
`load()` ever writes a fresh `installer.conf` (i.e. the key is not already
present in the file), each gate is seeded through four tiers, in order:

    1. `gate_overrides[key]`  -- the matching CLI flag doc 00 §3.4's own
       table promises for every gate (`--remote-supervision`,
       `--webapp-integration`, `--tracking-history`, `--viewer-stream`,
       parsed by
       `app.build_parser()` with
       `default=None` so "not passed" stays distinguishable from an
       explicit `off`).
    2. `os.environ.get(key)`  -- the like-named environment variable, so
       `sudo -E ./mv3dt-installer MV3DT_WEBAPP_INTEGRATION=on ...` keeps
       working for anyone scripting an unattended run.
    3. An interactive prompt (`prompt`, injectable) listing the allowed
       values with `"off"` prefilled. Skipped entirely under
       `non_interactive`.
    4. `GATE_DEFAULTS[key]` -- `"off"`, which is also what
       `--non-interactive` yields when neither flag nor env var was given
       (doc 00 §3.4: an unattended run never blocks on a human and never
       opts itself into a gated step).

Once a gate is persisted, later runs never re-consult the environment or
the prompt for it -- same "capture once, then just read back" discipline as
the NGC/webapp credentials (§10, §14) -- so an operator's env var doesn't
silently flip an already-chosen value out from under them. The *flag* is
the one deliberate exception: passing `--webapp-integration on` against an
already-persisted `off` overwrites it and logs the change, because
otherwise there would be no way to turn a gate on after the first run
except by hand-editing `installer.conf`, which is exactly the "the
operator types a thing" failure mode the design rules out.

`load()` parses every gate into the returned `Config` so a later module
(`app.py`) can gate Steps 6/7/8 without re-implementing KEY=VALUE parsing.

Only `install_dir` itself is created by this module. The subdirectories
under it (`secrets/`, `bin/`, `deepstream/`, `projects/`, `agent/`,
`webapp/`, `run/`) are each created by their owning module/step (doc 00
§11.2's layout tree) — never here.

Public API:
    CONF_FILENAME       -- "installer.conf".
    GATE_REMOTE_SUPERVISION / GATE_WEBAPP_INTEGRATION / GATE_TRACKING_HISTORY
        -- the §3.4 key names. GATE_VIEWER_STREAM -- the viewer feature gate.
    GATE_KEYS            -- every key name, in table order.
    GATE_DEFAULTS         -- {key: "off"} for each.
    GATE_CHOICES          -- {key: allowed values}, in doc 00 §3.4's table
        order. `app.build_parser()` reads these straight into its argparse
        `choices=`, so the flag and the prompt can never disagree about
        what a gate accepts.
    GATE_FLAGS            -- {key: the CLI flag that overrides it}, used
        only to name the source in the "flag overrode a persisted value"
        log line.
    Config                -- dataclass: install_dir, remote_supervision,
        webapp_integration, tracking_history, viewer_stream, values (full parsed KEY=VALUE
        dict).
    load(install_dir_override, state, non_interactive, ...,
        gate_overrides=None) -> Config
"""

from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass, field
from typing import Callable

from mv3dt_installer.logs import log
from mv3dt_installer.state import DEFAULT_INSTALL_DIR, StateMachine

__all__ = [
    "CONF_FILENAME",
    "GATE_REMOTE_SUPERVISION",
    "GATE_WEBAPP_INTEGRATION",
    "GATE_TRACKING_HISTORY",
    "GATE_VIEWER_STREAM",
    "GATE_KEYS",
    "GATE_DEFAULTS",
    "GATE_CHOICES",
    "GATE_FLAGS",
    "CAMERAS_FILE_KEY",
    "CAMERA_SCAN_CIDR_KEY",
    "CAMERA_SCAN_IFACE_KEY",
    "Config",
    "load",
    "persist_value",
    "remove_values",
]

# Mirrored file under the install dir (doc 00 §11.2's layout tree).
CONF_FILENAME = "installer.conf"

# doc 00 §11.2/§15.5: camera-scan tuning and the CAMERAS_FILE pointer,
# persisted alongside the gates but not gate-shaped (no seeding precedence,
# no CLI-flag-overwrite-and-log rule) -- app.py writes these directly via
# persist_value() below, after a --scan-cameras run.
CAMERAS_FILE_KEY = "CAMERAS_FILE"
CAMERA_SCAN_CIDR_KEY = "CAMERA_SCAN_CIDR"
CAMERA_SCAN_IFACE_KEY = "CAMERA_SCAN_IFACE"

# §3.4 opt-in step gates. Key names match the installer.conf KEY=VALUE shape
# exactly (bash-fragment-safe, uppercase, no lowercase/mixed variants).
GATE_REMOTE_SUPERVISION = "MV3DT_REMOTE_SUPERVISION"
GATE_WEBAPP_INTEGRATION = "MV3DT_WEBAPP_INTEGRATION"
GATE_TRACKING_HISTORY = "MV3DT_TRACKING_HISTORY"
# Not a step gate: Step 4 reads it to add the VLC viewer stream (tiled view,
# OSD, RTSP sink on 8554) to the rendered DeepStream config. Seeded and
# overridden exactly like the step gates.
GATE_VIEWER_STREAM = "MV3DT_VIEWER_STREAM"
GATE_KEYS: tuple[str, ...] = (
    GATE_REMOTE_SUPERVISION,
    GATE_WEBAPP_INTEGRATION,
    GATE_TRACKING_HISTORY,
    GATE_VIEWER_STREAM,
)
GATE_DEFAULTS: dict[str, str] = {
    GATE_REMOTE_SUPERVISION: "off",
    GATE_WEBAPP_INTEGRATION: "off",
    GATE_TRACKING_HISTORY: "off",
    GATE_VIEWER_STREAM: "off",
}

# Allowed values per gate, in doc 00 §3.4's table order. Step 6's remote
# supervision is three-valued; Steps 7 and 8 are binary. Note
# that "off" is the uniform skip indicator across all three, which is why
# `app._gate_is_off()` tests for it by name rather than for "not on".
# `app.build_parser()` feeds these tuples straight into argparse's
# `choices=`, and `_prompt_for_gate()` below lists them in its prompt, so
# there is exactly one place to change if a gate ever grows a value.
GATE_CHOICES: dict[str, tuple[str, ...]] = {
    GATE_REMOTE_SUPERVISION: ("off", "local", "remote"),
    GATE_WEBAPP_INTEGRATION: ("off", "on"),
    GATE_TRACKING_HISTORY: ("off", "on"),
    GATE_VIEWER_STREAM: ("off", "on"),
}

# The CLI flag that overrides each gate (doc 00 §3.4: "a matching CLI
# flag"). Used only to name the source in the log line emitted when a flag
# overwrites an already-persisted value; `app.py` owns the argparse wiring.
GATE_FLAGS: dict[str, str] = {
    GATE_REMOTE_SUPERVISION: "--remote-supervision",
    GATE_WEBAPP_INTEGRATION: "--webapp-integration",
    GATE_TRACKING_HISTORY: "--tracking-history",
    GATE_VIEWER_STREAM: "--viewer-stream",
}

# installer.conf also records the resolved install_dir under this key
# (INSTALL_DIR=...), purely to support the "installer.conf's own record of a
# previously-chosen dir" precedence tier below -- the one case where
# state.json was reset (or never written) but a prior install's
# installer.conf survived at the default location.
_INSTALL_DIR_KEY = "INSTALL_DIR"


@dataclass
class Config:
    """Resolved install-location config (doc 00 §11) + parsed §3.4 gates."""

    install_dir: pathlib.Path
    remote_supervision: str = GATE_DEFAULTS[GATE_REMOTE_SUPERVISION]
    webapp_integration: str = GATE_DEFAULTS[GATE_WEBAPP_INTEGRATION]
    tracking_history: str = GATE_DEFAULTS[GATE_TRACKING_HISTORY]
    viewer_stream: str = GATE_DEFAULTS[GATE_VIEWER_STREAM]
    # Full parsed installer.conf KEY=VALUE map, including the two gate keys
    # and INSTALL_DIR, plus anything else already present in the file (other
    # modules, e.g. ngc.py/webapp.py, may add their own shared vars here
    # later -- config.py must not clobber keys it doesn't own).
    values: dict[str, str] = field(default_factory=dict)

    @property
    def conf_path(self) -> pathlib.Path:
        return self.install_dir / CONF_FILENAME


def _read_conf(path: pathlib.Path) -> dict[str, str]:
    """Parse a plain `KEY=VALUE` file. No quoting/expansion (doc 00 §11.2:
    same shape as laptop.env.example). Missing file -> empty dict; malformed
    lines (no `=`, blank, `#`-comment) are skipped, never fatal."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip()
    return values


def _write_conf(path: pathlib.Path, values: dict[str, str]) -> None:
    """Write `values` as a plain KEY=VALUE file, sorted for stable diffs."""
    lines = [
        "# installer.conf -- generated by mv3dt-installer (doc 00 SS11.2).",
        "# Plain KEY=VALUE, no quoting/expansion -- same shape as",
        "# laptop/config/laptop.env.example. Safe to `set -a; . installer.conf`.",
        "",
    ]
    lines.extend(f"{key}={values[key]}" for key in sorted(values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def persist_value(install_dir: "pathlib.Path | str", key: str, value: str) -> None:
    """Write a single non-gate `KEY=VALUE` into `installer.conf`, preserving
    every other key already there (doc 00 §11.2's `CAMERA_SCAN_CIDR` /
    `CAMERA_SCAN_IFACE`, doc 00 §15.5). Unlike the gate keys, there is no
    seeding precedence here -- this always overwrites, the same way a gate
    flag overwrites an already-persisted gate value."""
    conf_path = pathlib.Path(install_dir) / CONF_FILENAME
    values = _read_conf(conf_path)
    values[key] = value
    _write_conf(conf_path, values)


def remove_values(install_dir: "pathlib.Path | str", *keys: str) -> None:
    """Remove legacy or retired keys while preserving the remaining config."""
    conf_path = pathlib.Path(install_dir) / CONF_FILENAME
    values = _read_conf(conf_path)
    changed = False
    for key in keys:
        if values.pop(key, None) is not None:
            changed = True
    if changed:
        _write_conf(conf_path, values)


def _resolve_install_dir(
    override: str | os.PathLike[str] | None,
    state: StateMachine,
    default_install_dir: str | os.PathLike[str],
) -> tuple[pathlib.Path, str]:
    """Precedence chain (doc 00 §11.2), minus the interactive-prompt step
    (handled by the caller once it sees `source == "default"`)."""
    if override:
        return pathlib.Path(override).expanduser(), "override"

    # state.json is the single source of truth once it has actually been
    # written; a not-yet-existing file must not be confused with an
    # explicit choice of the (coincidentally identical) hardcoded default.
    if state.path.exists():
        loaded = state.load()
        if loaded.install_dir:
            return pathlib.Path(loaded.install_dir).expanduser(), "state"

    # Edge case (doc 00 §11.2, "installer.conf default"): state.json was
    # reset/missing but a previous run's installer.conf survived at the
    # default location. Handled simply: only the default location is ever
    # consulted here -- a prior custom --install-dir cannot be recovered
    # without state.json, by construction.
    default_dir = pathlib.Path(default_install_dir).expanduser()
    default_conf = default_dir / CONF_FILENAME
    if default_conf.exists():
        recorded = _read_conf(default_conf).get(_INSTALL_DIR_KEY)
        if recorded:
            return pathlib.Path(recorded).expanduser(), "conf"

    return default_dir, "default"


def _prompt_for_install_dir(
    default: pathlib.Path, prompt: Callable[[str], str]
) -> pathlib.Path:
    """doc 00 §11.1: TUI prompt, `/opt/mv3dt` (or whatever the resolved
    default is) prefilled; an empty answer accepts it."""
    answer = prompt(f"Install directory [{default}]: ").strip()
    return pathlib.Path(answer).expanduser() if answer else default


def _prompt_for_gate(key: str, prompt: Callable[[str], str]) -> str:
    """Tier 3 of the §3.4 gate seeding chain: ask the operator once, with
    the allowed values listed and `"off"` prefilled.

    An empty answer accepts the prefilled default, matching
    `_prompt_for_install_dir()`'s convention. An answer outside
    `GATE_CHOICES[key]` is rejected with a one-line explanation and the
    question is asked again rather than silently coerced, because a typo
    here decides whether a whole step ever runs. Answers are compared
    case-insensitively and stored lowercase so `installer.conf` only ever
    holds the canonical spelling the dispatch loop compares against.
    """
    choices = GATE_CHOICES[key]
    default = GATE_DEFAULTS[key]
    question = f"{key} ({'/'.join(choices)}) [{default}]: "
    while True:
        answer = prompt(question).strip().lower()
        if not answer:
            return default
        if answer in choices:
            return answer
        log.warn(
            f"{key}: {answer!r} is not one of {', '.join(choices)}; "
            "please answer again"
        )


def _seed_gate(
    key: str,
    *,
    gate_overrides: dict[str, str],
    non_interactive: bool,
    prompt: Callable[[str], str],
) -> str:
    """Resolve a gate's value for the *first* write of `installer.conf`.

    Four tiers, in the order the module docstring documents: CLI flag, then
    the like-named environment variable, then an interactive prompt, then
    `"off"`. `--non-interactive` skips only tier 3, so an unattended run
    with neither flag nor env var lands on `"off"` without ever blocking on
    a human (doc 00 §3.4).
    """
    override = gate_overrides.get(key)
    if override:
        return override

    env_value = os.environ.get(key)
    if env_value:
        return env_value

    if not non_interactive:
        return _prompt_for_gate(key, prompt)

    return GATE_DEFAULTS[key]


def _resolve_gates(
    values: dict[str, str],
    *,
    gate_overrides: dict[str, str],
    non_interactive: bool,
    prompt: Callable[[str], str],
) -> None:
    """Fill in / update the §3.4 gate keys of an already-parsed
    `installer.conf` map, in place.

    A key absent from the file is being written for the first time and goes
    through `_seed_gate()`'s four tiers. A key already present is read
    straight back -- the "capture once, then just read back" discipline --
    with exactly one exception: an explicit CLI flag overwrites it and logs
    the change, since otherwise the only way to turn a gate on after the
    first run would be to hand-edit `installer.conf`.
    """
    for key in GATE_KEYS:
        if key not in values:
            values[key] = _seed_gate(
                key,
                gate_overrides=gate_overrides,
                non_interactive=non_interactive,
                prompt=prompt,
            )
            continue

        override = gate_overrides.get(key)
        if override and override != values[key]:
            log.info(
                f"{key}: {values[key]} -> {override} "
                f"(overridden by {GATE_FLAGS[key]})"
            )
            values[key] = override


def _sync_state(state: StateMachine, resolved: pathlib.Path) -> None:
    """Write `resolved` into state.json's install_dir field (doc 00 §11.2)
    whenever it wasn't already recorded there -- i.e. state.json didn't
    exist yet (first run), or it recorded a different path."""
    file_existed = state.path.exists()
    current = state.load()
    if not file_existed or current.install_dir != str(resolved):
        current.install_dir = str(resolved)
        state.save(current)


def load(
    install_dir_override: str | os.PathLike[str] | None,
    state: StateMachine,
    non_interactive: bool = False,
    *,
    default_install_dir: str | os.PathLike[str] = DEFAULT_INSTALL_DIR,
    prompt: Callable[[str], str] = input,
    gate_overrides: dict[str, str] | None = None,
) -> Config:
    """Resolve the install directory, persist it, and return the parsed
    `installer.conf` config (doc 00 §11, §3.4).

    Args:
        install_dir_override: value of the `--install-dir` CLI flag, if any.
        state: the `StateMachine` for `state.json` (§6.3) -- the single
            source of truth for `install_dir` once chosen.
        non_interactive: mirrors `--non-interactive`; suppresses the §11.1
            prompt and silently accepts the resolved default instead.
        default_install_dir: override for the hardcoded `/opt/mv3dt`
            default (doc 00 §11.1). Production callers should never pass
            this; it exists so tests can exercise the "installer.conf
            default" precedence tier without touching the real
            `/opt/mv3dt`.
        prompt: injectable stand-in for `input()`, for testability. Used
            both for the §11.1 install-dir question and for the §3.4 gate
            questions.
        gate_overrides: `{gate key: value}` from the matching CLI flags
            (`--remote-supervision`, `--webapp-integration`). Only keys the
            operator actually passed appear here -- `app.py` builds the map
            from arguments whose argparse default is `None`, so an explicit
            `--webapp-integration off` is distinguishable from the flag
            being omitted, and only the former overwrites a persisted
            value.

    Side effects: creates `install_dir` if missing, reads-or-writes
    `installer.conf` under it, and -- when a new path was just resolved --
    writes it back into `state.json` via `state.save()`.
    """
    resolved, source = _resolve_install_dir(
        install_dir_override, state, default_install_dir
    )

    if source == "default" and not non_interactive:
        resolved = _prompt_for_install_dir(resolved, prompt)

    resolved.mkdir(parents=True, exist_ok=True)

    conf_path = resolved / CONF_FILENAME
    values = _read_conf(conf_path)
    _resolve_gates(
        values,
        gate_overrides=gate_overrides or {},
        non_interactive=non_interactive,
        prompt=prompt,
    )
    values[_INSTALL_DIR_KEY] = str(resolved)
    _write_conf(conf_path, values)

    _sync_state(state, resolved)

    return Config(
        install_dir=resolved,
        remote_supervision=values[GATE_REMOTE_SUPERVISION],
        webapp_integration=values[GATE_WEBAPP_INTEGRATION],
        tracking_history=values[GATE_TRACKING_HISTORY],
        viewer_stream=values[GATE_VIEWER_STREAM],
        values=values,
    )
