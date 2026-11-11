"""Root-privilege checks, invoking-user resolution, and the USER-ACTION
display contract for mv3dt-installer (doc 00 §9.1-9.4).

Mirrors the *behavior* of laptop/scripts/lib/common.sh's `require_root` and
the `$SUDO_USER` handling in `00_bootstrap.sh`, re-expressed for the unified
installer. This module is self-contained (doc 00 §3.1) and a sibling of
steps/ (doc 00 §9.3) — it does not import from steps/.

Public API:
    require_root() -- exit (via logs.die) unless running as uid 0.
    resolve() -> InvokingUser -- who invoked `sudo`, resolved from the
        passwd DB (never from $HOME).
    run_as_user(*args, **subprocess_kwargs) -> subprocess.CompletedProcess
        -- run a command as the invoking (non-root) user, equivalent to
        `sudo -u "$SUDO_USER" -H ...`.
    render_user_action_block(step_title, why, actions=()) -> str -- pure
        formatter for the §9.3 ACTION REQUIRED frame.
    show_user_action_block(step_title, why, actions=()) -- render + emit
        the block to stderr (and the transcript, once one is open, via
        logs.log.error).
    render_reboot_block(step_title) -> str / show_reboot_required(step_title)
        -- §9.4 wrapper around the above with the fixed reboot reason and no
        enumerated actions.

Calling convention for `actions` (read this before wiring app.py or any
step to this module):

    Each element of `actions` must expose, via plain attribute access
    (duck typing -- `getattr(action, "text")`, no isinstance/Protocol
    check is performed at runtime):

        .text     : str            -- required. The instruction text.
        .command  : str | None     -- optional (defaults to None via
                                       getattr). Shown verbatim on its own
                                       indented continuation line, prefixed
                                       "$ ", when present.
        .path     : str | None     -- optional (defaults to None via
                                       getattr). Shown on its own indented
                                       continuation line as "(edit: <path>)"
                                       when present.

    Any object with those attributes works: a `@dataclass`, a
    `types.SimpleNamespace(text=..., command=..., path=...)`, a
    `collections.namedtuple("UserAction", "text command path")` instance,
    etc. Plain 3-tuples do NOT work (no named attributes) -- use one of the
    above. This module intentionally does not import `steps.UserAction` (a
    separate, independent PR) -- it only requires structural compatibility
    with the `ActionLike` Protocol defined below, which exists purely for
    type-checking and is never used for runtime isinstance checks.
"""

from __future__ import annotations

import getpass
import os
import pathlib
import pwd
import subprocess
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Protocol, runtime_checkable

from .logs import die, log

__all__ = [
    "InvokingUser",
    "ActionLike",
    "require_root",
    "resolve",
    "run_as_user",
    "render_user_action_block",
    "show_user_action_block",
    "render_reboot_block",
    "show_reboot_required",
]


# ---------------------------------------------------------------------------
# 9.1 -- root check
# ---------------------------------------------------------------------------


def require_root() -> None:
    """Exit with a clear message unless running as root (doc 00 §9.1).

    Mirrors `require_root` in lib/common.sh: the installer performs
    package installs and writes to /etc/profile.d and /var/lib, so it must
    run as uid 0 (typically via `sudo -E mv3dt-installer`).
    """
    if os.geteuid() != 0:
        die(
            "mv3dt-installer must be run as root "
            "(try: sudo -E mv3dt-installer)."
        )


# ---------------------------------------------------------------------------
# 9.2 -- invoking-user resolution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InvokingUser:
    """The non-root user that invoked `sudo`, per doc 00 §9.2.

    `home`/`uid`/`gid` are resolved from the passwd database (equivalent to
    `getent passwd "$SUDO_USER" | cut -d: -f6`), NEVER from the `$HOME`
    environment variable -- under sudo, `$HOME` is root's home unless the
    operator passed `-H` themselves, which is exactly the bug this contract
    exists to avoid.
    """

    name: str
    home: pathlib.Path
    uid: int
    gid: int


def resolve() -> InvokingUser:
    """Resolve the invoking user (doc 00 §9.2).

    `name` is `$SUDO_USER` if set and not "root", else the current user
    (`getpass.getuser()`, i.e. no sudo in play -- e.g. local dev/testing).
    `home`/`uid`/`gid` come from `pwd.getpwnam(name)`, i.e. the passwd DB.
    """
    sudo_user = os.environ.get("SUDO_USER")
    name = sudo_user if sudo_user and sudo_user != "root" else getpass.getuser()

    entry = pwd.getpwnam(name)
    return InvokingUser(
        name=name,
        home=pathlib.Path(entry.pw_dir),
        uid=entry.pw_uid,
        gid=entry.pw_gid,
    )


def _missing_executable_result(
    cmd: list, kwargs: dict
) -> subprocess.CompletedProcess:
    """Synthesize the `CompletedProcess` `subprocess.run` never gets to
    return when `sudo` itself (or, in principle, the wrapped command) isn't
    on `PATH`.

    `FileNotFoundError` is raised before `check` is ever consulted, so a
    caller written as `run_as_user(..., check=False, capture_output=True,
    text=True)` -- expecting "not available" to look like a nonzero return
    code -- would otherwise crash instead. Mirrors
    `app._missing_executable_result`, duplicated rather than imported: this
    module is a deliberately self-contained sibling of app.py (module
    docstring). 127 is the shell convention for "command not found".
    """
    text_mode = bool(kwargs.get("text") or kwargs.get("universal_newlines"))
    empty: Any = "" if text_mode else b""
    captured = bool(kwargs.get("capture_output")) or kwargs.get("stdout") is not None
    return subprocess.CompletedProcess(
        cmd, 127, stdout=empty if captured else None, stderr=empty if captured else None
    )


def run_as_user(*args: str, **subprocess_kwargs: Any) -> subprocess.CompletedProcess:
    """Run a command as the invoking user (doc 00 §9.2).

    Equivalent to `sudo -u "$SUDO_USER" -H <args...>`. `args` are the
    command and its arguments (e.g. `run_as_user("ngc", "config", "set")`).
    Accepts the same keyword arguments as `subprocess.run` (e.g.
    `capture_output`, `text`, `env`, `check`) and returns its
    `CompletedProcess`.

    Anything that must run "as the user" -- the `ngc` CLI, `docker` without
    sudo, files under the user's home, the AMC clone under
    `$HOME/auto-magic-calib` -- MUST go through this function rather than
    running unwrapped as root.

    A missing executable (`sudo`, or the wrapped command itself) is treated
    the same as a `check=False` caller already treats a nonzero exit -- see
    `_missing_executable_result`.
    """
    user = resolve()
    cmd = ["sudo", "-u", user.name, "-H", *args]
    try:
        return subprocess.run(cmd, **subprocess_kwargs)
    except FileNotFoundError:
        return _missing_executable_result(cmd, subprocess_kwargs)


# ---------------------------------------------------------------------------
# 9.3 -- USER-ACTION display contract
# ---------------------------------------------------------------------------

# Total width of the box, corners included -- matches doc 00 §9.3's example
# frame exactly (measured, not guessed): 65 columns.
_BOX_WIDTH = 65
_BORDER = "+" + "-" * (_BOX_WIDTH - 2) + "+"
_HEADER_TEXT = "ACTION REQUIRED"
_CLOSING_LINE = "  Then run the installer again to continue."

# Continuation-line indent for an action's command/path, chosen to align
# under the text of a single-digit numbered item ("    1. " is 7 columns).
_ACTION_CONTINUATION_INDENT = "       "


@runtime_checkable
class ActionLike(Protocol):
    """Structural shape required of each element of `actions` below.

    Exists only for type-checking / documentation. Never used for an
    isinstance() check at runtime -- attributes are read via `getattr(...,
    default)` so any object exposing them (dataclass, SimpleNamespace,
    namedtuple, or `steps.UserAction` once that module exists) works
    without this module importing that type.
    """

    text: str
    command: Optional[str]
    path: Optional[str]


def _header_line() -> str:
    inner_width = _BOX_WIDTH - 2
    content = f"  {_HEADER_TEXT}"
    return "|" + content.ljust(inner_width) + "|"


def _render_action_lines(actions: Iterable[Any]) -> list[str]:
    lines: list[str] = []
    for index, action in enumerate(actions, start=1):
        text = getattr(action, "text")
        command = getattr(action, "command", None)
        path = getattr(action, "path", None)

        lines.append(f"    {index}. {text}")
        if command:
            lines.append(f"{_ACTION_CONTINUATION_INDENT}$ {command}")
        if path:
            lines.append(f"{_ACTION_CONTINUATION_INDENT}(edit: {path})")
    return lines


def render_user_action_block(
    step_title: str, why: str, actions: Iterable[Any] = ()
) -> str:
    """Render the §9.3 ACTION REQUIRED frame as a string.

    `actions` follows the calling convention documented at the top of this
    module (duck-typed `.text`/`.command`/`.path` attribute access). When
    `actions` is empty, the "Do the following, in order:" section (and its
    following blank line) is omitted entirely -- see `render_reboot_block`,
    which relies on this for the §9.4 reboot frame.

    The final line inside the frame is always exactly
    "Then run the installer again to continue." -- that phrase is the
    contract other code may grep the transcript for.
    """
    action_lines = _render_action_lines(actions)

    lines = [
        _BORDER,
        _header_line(),
        _BORDER,
        f"  Step: {step_title}",
        f"  Why:  {why}",
        "",
    ]

    if action_lines:
        lines.append("  Do the following, in order:")
        lines.extend(action_lines)
        lines.append("")

    lines.append(_CLOSING_LINE)
    lines.append(_BORDER)

    return "\n".join(lines)


def show_user_action_block(
    step_title: str, why: str, actions: Iterable[Any] = ()
) -> None:
    """Render and emit the §9.3 block to stderr (+ transcript once open).

    Routed through `logs.log.error` so it lands on stderr and, once
    `logs.open_transcript()` has been called, in the per-run transcript
    file too -- reusing logs.py's plumbing rather than duplicating it here.
    """
    log.error(render_user_action_block(step_title, why, actions))


# ---------------------------------------------------------------------------
# 9.4 -- reboot block
# ---------------------------------------------------------------------------

REBOOT_REASON = "a reboot is required before the next step can run"


def render_reboot_block(step_title: str) -> str:
    """Render the §9.4 reboot frame: same frame, fixed reason, no actions."""
    return render_user_action_block(step_title, REBOOT_REASON, actions=())


def show_reboot_required(step_title: str) -> None:
    """Render and emit the §9.4 reboot frame to stderr (+ transcript)."""
    log.error(render_reboot_block(step_title))
