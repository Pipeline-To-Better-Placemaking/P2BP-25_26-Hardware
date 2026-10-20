"""systemd unit rendering, idempotent install, and lifecycle helpers.

Framework-only module. It owns the mechanical half of locked decision 3(b) of
the release-model plan -- after the first calibration, later recalibrations are
ingested automatically, with no installer run and no operator command -- plus
the shared unit-install helper
[`STEP-6-REMOTE-SUPERVISION.md`](../plan/STEP-6-REMOTE-SUPERVISION.md)
section A.4 needs (copy units, `daemon-reload`, `enable`).

Templates live under `assets/systemd/` and are resolved through
`shellout.asset_path`, so they work identically from a source checkout and from
the frozen PyInstaller binary. They use at-sign-delimited markers such as
`@EXPORT_DIR@` rather than a systemd template unit with `%i`: the watched
export directory is an arbitrary absolute path under the invoking user's home,
and instance-name path escaping is exactly the cleverness STEP-6 section B.1
already weighed and rejected once.

Two rules this module enforces rather than documents:

* `render_unit` raises if any marker survives substitution. A half-rendered
  unit file is worse than a missing one -- systemd would accept it and then
  behave in a way nobody wrote down.
* `enable_now` refuses when the unit's `ExecStart` binary is absent. The
  installed units point at `<install_dir>/bin/mv3dt-installer`; a unit enabled
  before that binary exists would fail on every single trigger.

**Only the trigger unit is ever enabled.** Step 4's ingest pair is a `.timer`
plus a `Type=oneshot` `.service` that Step 4 renders inline
(`STEP-4-CALIB-OUTPUT-WIRING.md` section 7); only the `.timer` is enabled, and
the `.service` is owned by its lifecycle. Enabling both makes the boot-time
state race the trigger (STEP-6 section A.4's lesson, and the carve-out spelled
out in section B.1).

`render_ingest_units` and the `mv3dt-ingest.path.in` / `mv3dt-ingest.service.in`
templates are the superseded `.path` design. Nothing in the installer calls
them; they remain only until they are removed.

Every `systemctl` invocation goes through an injected `runner`, defaulting to
`subprocess.run`, so tests never touch real systemd. The runner is called with
a single argv **list**, matching `subprocess.run`'s own signature. A production
caller holding an `app.Context` passes `ctx.run_root`, which takes its argv as
varargs, so it needs a one-line adapter::

    runner=lambda argv, **kwargs: ctx.run_root(*argv, **kwargs)

Nothing here writes to `/etc/systemd/system` unless a caller asks for the
default `unit_dir`, and no function in this module chowns anything: the
installer is already root by the time any step runs (`privilege.require_root`
gates `main()`), so files it creates are root-owned by construction.
"""

from __future__ import annotations

import pathlib
import re
import shlex
import subprocess
from typing import Any, Callable, Iterable, Mapping

from . import shellout
from .logs import log

#: Where systemd looks for locally-installed units.
UNIT_DIR = pathlib.Path("/etc/systemd/system")

#: Mode for an installed unit file: root-owned, world-readable, not writable.
UNIT_MODE = 0o644

#: Asset directory (under `assets/`) holding the `*.in` unit templates.
TEMPLATE_DIR = "systemd"

#: Matches an unsubstituted `@MARKER@`. Deliberately narrow -- upper-case,
#: digits and underscores only -- so a rendered value that happens to contain
#: an at-sign (an email address, say) is not mistaken for a live marker.
_MARKER_RE = re.compile(r"@[A-Z][A-Z0-9_]*@")

#: Characters that double-quoting an interpolated `ExecStart=` value cannot
#: rescue: a double quote or backslash escapes the quoting systemd applies, and
#: any control character (newline included) ends the directive or is rejected
#: outright by systemd's command-line parser.
_UNQUOTABLE_RE = re.compile(r'["\\\x00-\x1f\x7f]')

Runner = Callable[..., "subprocess.CompletedProcess[Any]"]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_unit(
    template_parts: Iterable[str],
    substitutions: Mapping[str, str],
) -> str:
    """Render a bundled unit template into finished unit-file text.

    ``template_parts`` are path components under the bundled ``assets/``
    directory, resolved with `shellout.asset_path` -- e.g.
    ``("systemd", "mv3dt-ingest.path.in")``.

    ``substitutions`` maps **bare** marker names to their values: the key
    ``"EXPORT_DIR"`` replaces every occurrence of ``@EXPORT_DIR@``. Values are
    substituted literally, with no shell or systemd escaping applied.

    Because substitution is literal, **quoting is the template's job and the
    templates do it**: any marker landing inside a word-split directive such as
    `ExecStart=` is written double-quoted in the `.in` file, so a value
    containing whitespace survives. Callers rendering their own templates owe
    the same. What literal substitution cannot express is a value containing a
    double quote, a backslash, or a newline -- those would break out of the
    quoting or out of the directive entirely, so `render_ingest_units` rejects
    them up front rather than emitting a unit that misbehaves.

    Raises `ValueError` if any marker survives substitution, naming the ones
    that were left behind. A partially rendered unit is worse than no unit at
    all: systemd will happily load it and then do something nobody specified.
    """
    parts = tuple(template_parts)
    source = shellout.asset_path(*parts)
    text = source.read_text(encoding="utf-8")

    for key, value in substitutions.items():
        text = text.replace(f"@{key}@", str(value))

    leftover = sorted(set(_MARKER_RE.findall(text)))
    if leftover:
        raise ValueError(
            f"unrendered placeholder(s) {', '.join(leftover)} in "
            f"{'/'.join(parts)} -- refusing to emit a half-rendered unit file"
        )
    return text


def render_ingest_units(
    *,
    project: str,
    slug: str,
    export_dir: str | pathlib.Path,
    install_dir: str | pathlib.Path,
    installer_bin: str | pathlib.Path,
) -> dict[str, str]:
    """Render the auto-ingest `.path`/`.service` pair for one project.

    **Unused.** Step 4 replaced this `.path` design with an API-polling
    `.timer` it renders inline (`STEP-4-CALIB-OUTPUT-WIRING.md` section 7);
    see the module docstring.

    Returns a mapping of unit file name to rendered content, ready to hand to
    `install_unit`::

        {
            "mv3dt-ingest-<slug>.path":    "...",
            "mv3dt-ingest-<slug>.service": "...",
        }

    The `.path` unit watches ``export_dir`` (AutoMagicCalib's export
    directory) and triggers the `.service`, whose `ExecStart` runs
    ``<installer_bin> ingest --project <project> --non-interactive
    --install-dir <install_dir>``. Only the `.path` is ever enabled -- see the
    module docstring.

    ``project`` is free-form operator text typed into the AMC GUI
    (`STEP-5-PER-PROJECT-EXES.md` section 3.1, worked example ``"North Lobby
    #2"``), so it routinely contains spaces. systemd word-splits `ExecStart=`,
    so the service template double-quotes every marker it interpolates there
    and a spaced project name arrives as one argument. A literal percent sign
    is escaped to `%%` for the same reason: every directive these markers land
    in expands systemd specifiers, so an unescaped `%` in operator text would
    be read as one. Raises `ValueError` for a value that neither quoting nor
    escaping can rescue -- one containing a double quote, backslash, or newline
    -- since such a value would escape the quoting or the directive and produce
    a unit that fails on every trigger.
    """
    for label, value in (
        ("project", project),
        ("export_dir", export_dir),
        ("install_dir", install_dir),
        ("installer_bin", installer_bin),
    ):
        _reject_unquotable(label, str(value))

    substitutions = {
        "PROJECT": _escape_specifiers(str(project)),
        # `slug` is a sanitized identifier by construction (STEP-5 section 3.1
        # narrows it to lower-case alphanumerics and hyphens) and it also forms
        # the returned file names, so it is interpolated as given.
        "SLUG": str(slug),
        "EXPORT_DIR": _escape_specifiers(str(export_dir)),
        "INSTALL_DIR": _escape_specifiers(str(install_dir)),
        "INSTALLER_BIN": _escape_specifiers(str(installer_bin)),
    }
    return {
        f"mv3dt-ingest-{slug}.path": render_unit(
            (TEMPLATE_DIR, "mv3dt-ingest.path.in"), substitutions
        ),
        f"mv3dt-ingest-{slug}.service": render_unit(
            (TEMPLATE_DIR, "mv3dt-ingest.service.in"), substitutions
        ),
    }


# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------


def install_unit(
    name: str,
    content: str,
    *,
    unit_dir: pathlib.Path = UNIT_DIR,
    runner: Runner = subprocess.run,
) -> bool:
    """Write ``content`` to ``<unit_dir>/<name>`` if it differs from what is
    already there. Returns True when the file changed, False when it did not.

    Content-idempotent by read-compare-write, so re-running the installer does
    not churn mtimes or force a needless `daemon-reload`. The return value is
    what lets the caller pick between `report.report_installed` and
    `report.report_already_installed` -- doc 00 section 8.3's exact strings.

    ``runner`` is accepted for signature symmetry with the rest of this module
    and is currently unused: writing a unit file is plain file I/O, and the
    process is already root.
    """
    del runner  # see docstring

    unit_dir.mkdir(parents=True, exist_ok=True)
    target = unit_dir / name

    if target.exists():
        try:
            existing = target.read_text(encoding="utf-8")
        except OSError:
            existing = None
        if existing == content:
            log.info(f"unit {name} already up to date at {target}")
            return False

    target.write_text(content, encoding="utf-8")
    target.chmod(UNIT_MODE)
    log.info(f"wrote unit {name} to {target}")
    return True


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def daemon_reload(runner: Runner = subprocess.run) -> None:
    """`systemctl daemon-reload`.

    Required after any unit file is written or removed, or systemd acts on a
    stale view of the unit tree (STEP-6 section A.4 step 2).
    """
    runner(["systemctl", "daemon-reload"], check=False)


def enable_now(name: str, runner: Runner = subprocess.run) -> None:
    """`systemctl enable --now <name>`, unless the unit's target binary is
    missing.

    ``name`` is normally the `.path` unit. A `.path` has no `ExecStart` of its
    own, so the check follows its ``Unit=`` directive to the triggered service
    and inspects that unit's `ExecStart` instead.

    If the resolved binary does not exist on disk, this logs an error and
    returns **without** calling systemctl. Enabling a unit whose `ExecStart` is
    absent produces a unit that fails on every trigger and fills the journal
    with noise; the ingest units point at ``<install_dir>/bin/mv3dt-installer``,
    which STEP-3 creates and STEP-3 is not implemented yet. Callers that need
    to surface this should treat a still-inactive unit as
    `USER_ACTION_REQUIRED` rather than assume success.
    """
    binary = _exec_start_binary(name)
    if binary is not None and not pathlib.Path(binary).exists():
        log.error(
            f"refusing to enable {name}: its ExecStart binary {binary} does "
            "not exist, so every trigger would fail. Enable it once that "
            "binary is installed."
        )
        return
    runner(["systemctl", "enable", "--now", name], check=False)


def is_active(name: str, runner: Runner = subprocess.run) -> bool:
    """True when `systemctl is-active <name>` exits 0."""
    result = runner(["systemctl", "is-active", "--quiet", name], check=False)
    return _returncode(result) == 0


def is_enabled(name: str, runner: Runner = subprocess.run) -> bool:
    """True when `systemctl is-enabled <name>` exits 0."""
    result = runner(["systemctl", "is-enabled", "--quiet", name], check=False)
    return _returncode(result) == 0


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _escape_specifiers(value: str) -> str:
    """Escape literal percent signs for a systemd unit value.

    `ExecStart=`, `Description=`, `PathChanged=` and friends all expand
    systemd specifiers, so a `%` in free-form operator text or in a path would
    be read as the start of one (`%i`, `%h`, and so on). Doubling it is
    systemd's own escape for a literal percent.
    """
    return value.replace("%", "%%")


def _reject_unquotable(label: str, value: str) -> None:
    """Raise `ValueError` if ``value`` cannot be safely double-quoted into a
    systemd `ExecStart=` command line.

    The templates quote every marker they interpolate into `ExecStart=`, which
    handles whitespace. It does not handle a value that closes the quote or
    ends the line, so those are refused here instead of silently producing a
    unit that fails on every trigger.
    """
    match = _UNQUOTABLE_RE.search(value)
    if match:
        raise ValueError(
            f"{label} contains {match.group()!r}, which cannot be quoted into "
            "a systemd ExecStart= command line -- refusing to emit a unit "
            "whose trigger would always fail"
        )


def _returncode(result: Any) -> int:
    """Best-effort exit status from whatever the injected runner returned.

    A runner that returns None (a stub that only records argv) is treated as
    a failure rather than an accidental success.
    """
    code = getattr(result, "returncode", None)
    return code if isinstance(code, int) else 1


def _directive(text: str, key: str) -> str | None:
    """First value of a `Key=value` directive in unit-file ``text``.

    Ignores comment lines, which is enough parsing for the two directives this
    module reads. Not a general INI parser and not trying to be one.
    """
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", ";")):
            continue
        head, sep, value = line.partition("=")
        if sep and head.strip() == key:
            return value.strip()
    return None


def _exec_start_binary(name: str, *, _depth: int = 0) -> str | None:
    """Resolve the executable an installed unit would run, or None.

    Reads ``UNIT_DIR / name`` (looked up at call time, so tests can point the
    module constant at a tmp dir). For a `.path` unit, follows a single
    ``Unit=`` hop to the triggered unit and reads that one's `ExecStart`.
    Returns None when the unit file is unreadable or declares no `ExecStart` --
    "cannot tell" is not "is broken", so the caller proceeds.
    """
    if _depth > 1:
        return None

    path = UNIT_DIR / name
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None

    exec_start = _directive(text, "ExecStart")
    if exec_start:
        try:
            # shlex agrees with systemd on the subset the templates emit:
            # whitespace separation plus double-quoted words. The quoting
            # matters -- the binary path is quoted so it survives spaces, and
            # an unstripped quote would make every existence check fail.
            tokens = shlex.split(exec_start)
        except ValueError:
            tokens = exec_start.split()
        if not tokens:
            return None
        # systemd allows prefix characters on ExecStart (`-`, `@`, `+`, `!`).
        # `%%` in a rendered unit always denotes a literal percent, so undo the
        # escaping before the path is compared against the filesystem.
        return tokens[0].lstrip("-@+!:").replace("%%", "%")

    triggered = _directive(text, "Unit")
    if triggered:
        return _exec_start_binary(triggered, _depth=_depth + 1)
    return None
