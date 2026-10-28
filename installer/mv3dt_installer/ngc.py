"""NGC API key capture and local secure storage for mv3dt-installer (doc 00
§10).

The key is captured once in the onboarding stage (doc 00 §5.2) and consumed
later by Step 2 (PeopleNet model fetch, Docker install method) and any other
NGC-gated step, via `load_key()` / `configure_ngc_cli()`. The key is
REQUIRED: `capture_key()` never returns without one. This module is a
self-contained sibling of laptop/ (doc 00 §3.1) and of steps/ (doc 00 §9.3)
-- it does not import from either, and adds no step1-7 business logic.

Storage contract (doc 00 §10.1):
    Canonical secret file: `<install_dir>/secrets/ngc.env`, owned by the
    invoking user (privilege.resolve()), `chmod 600`, directory `chmod 700`.
    Single line: `NGC_API_KEY=<key>`. The raw key is NEVER written to the
    transcript log -- any log line that would contain it prints the literal
    `NGC_API_KEY=<redacted>` instead.

Public API (doc 00 §10.2):
    capture_key(non_interactive) -> str -- secret prompt (no echo,
        `getpass.getpass`); a blank answer re-prompts rather than being
        accepted. Under `--non-interactive` there is no human to re-prompt,
        so this dies immediately (doc 00 §4: "fail if a required value is
        missing") -- the caller (`onboarding.ensure_ngc_key()`) is expected
        to have already checked a pre-set `NGC_API_KEY` environment
        variable before ever calling this.
    store_key(key, install_dir) -> Path -- atomic write of `secrets/ngc.env`.
    load_key(install_dir=DEFAULT_INSTALL_DIR) -> str | None -- read back for
        a step; `None` only before onboarding has captured a key yet (or a
        hand-tampered/missing secrets file) -- every step that runs after
        onboarding can treat a present key as guaranteed.
    configure_ngc_cli(install_dir=DEFAULT_INSTALL_DIR) -> Path | None --
        writes `~/.ngc/config` for the invoking user, mirroring Phase 10 of
        00_bootstrap.sh's `NGCINI` heredoc.

`load_key`/`configure_ngc_cli` take an optional `install_dir` (defaulting to
the doc 00 §11.1 default install location, `/opt/mv3dt`) because `config.py`
(the module that will resolve the operator's actual chosen install_dir from
state.json) is a separate, not-yet-implemented PR (doc 00 §11) -- this
keeps the doc-quoted zero-arg call sites (`load_key()`, `configure_ngc_cli()`)
working today while still letting tests (and, later, config.py) point the
lookup elsewhere without touching a real `/opt/mv3dt`.

Design note on `configure_ngc_cli`: doc 00 §9.2 is explicit that anything
under the user's home "MUST be run via `privilege.run_as_user(...)` ...
exactly as Phases 6/10 of `00_bootstrap.sh` do for `ngc`", and
`privilege.py`'s own docstring repeats the same rule. `~/.ngc/config` is
exactly that case, so this module writes it entirely inside a
`privilege.run_as_user("bash", "-lc", ...)` call (equivalent to
`sudo -u "$SUDO_USER" -H bash -lc '...'`), mirroring 00_bootstrap.sh Phase
10's `NGCINI` heredoc, which itself runs inside a `sudo -u "$SUDO_USER" -H`
block (lines ~972-987) rather than being written by the root installer
process and chowned after the fact. The key is handed to that child process
via `NGC_ENV_FILE`, an env var pointing at the *already-invoking-user-owned*
`secrets/ngc.env` (written by `store_key`, §10.1) -- the child script
sources it (`set -a; . "$NGC_ENV_FILE"; set +a`) exactly as bootstrap's
Phase 10 sources `$BOOTSTRAP_ENV_FILE`. This means the raw key is never
passed as a CLI argument (so it never shows up in `ps`/`/proc/*/cmdline`)
and never appears as a literal in this module's Python source at all.
"""

from __future__ import annotations

import getpass
import os
import pathlib
import tempfile
from typing import Optional, Union

from . import privilege, prompting
from .logs import die, log

__all__ = [
    "DEFAULT_INSTALL_DIR",
    "capture_key",
    "store_key",
    "load_key",
    "configure_ngc_cli",
]

# doc 00 §11.1 default install directory. Only used as the default lookup
# location here -- config.py (doc 00 §11) is the eventual source of truth
# for the operator's actual chosen install_dir.
DEFAULT_INSTALL_DIR = pathlib.Path("/opt/mv3dt")

_SECRET_RELATIVE_PATH = pathlib.Path("secrets") / "ngc.env"
_KEY_PREFIX = "NGC_API_KEY="

# The ONLY string this module ever logs in place of a real key. Never
# interpolate the real key into any string passed to log.*().
_REDACTED_LOG_LINE = "NGC_API_KEY=<redacted>"

StrPath = Union[str, "os.PathLike[str]"]


# ---------------------------------------------------------------------------
# 10.2 -- capture
# ---------------------------------------------------------------------------


def capture_key(non_interactive: bool) -> str:
    """Prompt for the NGC API key (doc 00 §10.2). The key is REQUIRED --
    this never returns without one.

    Echo is suppressed via `getpass.getpass` -- the key is never displayed
    and never logged. A blank answer re-prompts (with a one-line reminder
    that the key is required) rather than being accepted, matching
    `config.py`'s `_prompt_for_gate()` retry-until-valid convention.

    Under `--non-interactive` (doc 00 §4: "never prompt; use
    defaults/config; fail if a required value is missing") there is no
    human to re-prompt, so this dies immediately instead of looping forever
    or silently accepting no key. The caller, `onboarding.ensure_ngc_key()`,
    is expected to have already checked a pre-set `NGC_API_KEY` environment
    variable (doc 00 §9.1's `sudo -E` path) before ever calling this, so a
    non-interactive run only reaches here when no key is available at all.
    """
    if non_interactive:
        die(
            "NGC API key is required but --non-interactive was passed with "
            "no NGC_API_KEY set in the environment; re-run with NGC_API_KEY "
            "exported (sudo -E) or without --non-interactive."
        )
    reason = prompting.prompt_unavailable_reason()
    if reason is not None:
        # The key has no safe default; fail rather than block on a prompt
        # the operator may not be able to answer.
        die(
            f"NGC API key is required but cannot be asked for ({reason}). "
            "Re-run without a pipe, or export NGC_API_KEY and run with sudo -E."
        )

    while True:
        raw = getpass.getpass("NGC API key (required): ").strip()
        if raw:
            log.info(f"Captured {_REDACTED_LOG_LINE}.")
            return raw
        log.warn("NGC API key cannot be blank; please try again.")


# ---------------------------------------------------------------------------
# Shared atomic-write helper (doc 00 §6.3 discipline, re-applied here: temp
# file in the same directory, flush + fsync, then os.replace -- see doc 00
# §14.3 for the sibling webapp.py credential using the identical pattern).
# The temp file is created at mode 0o600 from the very first os.open() call
# (tempfile.mkstemp's default), so the secret is never briefly
# world-readable between creation and the final chmod/chown.
# ---------------------------------------------------------------------------


def _atomic_write_secret(
    directory: pathlib.Path, target: pathlib.Path, contents: str
) -> None:
    """Write `contents` to `target` atomically. Caller must ensure
    `directory` already exists with its final owner/mode set -- this
    function only creates it as a fallback (`exist_ok=True`) and never
    loosens permissions the caller already applied."""
    directory.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=str(directory))
    tmp_path = pathlib.Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(contents)
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp_path, 0o600)
        os.replace(tmp_path, target)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------------
# 10.1 / 10.2 -- storage
# ---------------------------------------------------------------------------


def store_key(key: str, install_dir: StrPath) -> pathlib.Path:
    """Write `<install_dir>/secrets/ngc.env` (doc 00 §10.1/§10.2).

    Single line `NGC_API_KEY=<key>`. The secrets directory is `chmod 700`
    *before* anything is written into it (no window where a
    default-mode directory momentarily lists the file to other local
    users), the file itself `chmod 600`, both chowned to the invoking user
    (`privilege.resolve()`). Uses the atomic temp-file + `os.replace`
    pattern (see `_atomic_write_secret`) so a crash mid-write never leaves a
    truncated secret file behind.
    """
    secrets_dir = pathlib.Path(install_dir) / _SECRET_RELATIVE_PATH.parent
    target = secrets_dir / _SECRET_RELATIVE_PATH.name

    user = privilege.resolve()

    secrets_dir.mkdir(parents=True, exist_ok=True)
    os.chown(secrets_dir, user.uid, user.gid)
    os.chmod(secrets_dir, 0o700)

    _atomic_write_secret(secrets_dir, target, f"{_KEY_PREFIX}{key}\n")

    os.chown(target, user.uid, user.gid)
    os.chmod(target, 0o600)

    log.info(f"Stored {_REDACTED_LOG_LINE} at {target}.")
    return target


def load_key(install_dir: StrPath = DEFAULT_INSTALL_DIR) -> Optional[str]:
    """Read back the stored key (doc 00 §10.2).

    Returns `None` when `secrets/ngc.env` doesn't exist, or exists but has
    no usable `NGC_API_KEY=` value. Since the key is required, this should
    only happen before onboarding has run; every step that runs after
    onboarding can treat a present key as guaranteed.
    """
    path = pathlib.Path(install_dir) / _SECRET_RELATIVE_PATH
    if not path.is_file():
        return None

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith(_KEY_PREFIX):
            value = line[len(_KEY_PREFIX):]
            return value or None

    return None


# ---------------------------------------------------------------------------
# 10.2 -- ~/.ngc/config for steps that shell out to the `ngc` CLI
# ---------------------------------------------------------------------------


# Runs entirely as the invoking user (doc 00 §9.2), mirroring
# 00_bootstrap.sh Phase 10's NGCINI heredoc (lines ~972-987) verbatim,
# including the unquoted `<<NGCINI` delimiter so `${NGC_API_KEY}` expands.
# `$NGC_ENV_FILE` is supplied via the child process's environment (see
# `configure_ngc_cli`) -- never interpolated into this script string -- so
# the raw key is never a literal in Python source or a CLI argument.
_CONFIGURE_NGC_CLI_SCRIPT = (
    'set -e; '
    'set -a; . "$NGC_ENV_FILE"; set +a; '
    'mkdir -p "$HOME/.ngc"; '
    'chmod 700 "$HOME/.ngc"; '
    'cat > "$HOME/.ngc/config" <<NGCINI\n'
    '[CURRENT]\n'
    'apikey = ${NGC_API_KEY}\n'
    'format_type = ascii\n'
    'NGCINI\n'
    'chmod 600 "$HOME/.ngc/config"'
)


def configure_ngc_cli(
    install_dir: StrPath = DEFAULT_INSTALL_DIR,
) -> Optional[pathlib.Path]:
    """Write `~/.ngc/config` for the invoking user (doc 00 §10.2/§9.2).

    Mirrors Phase 10 of laptop/scripts/00_bootstrap.sh's `NGCINI` heredoc
    exactly:

        [CURRENT]
        apikey = <key>
        format_type = ascii

    Per doc 00 §9.2 ("files under the user's home MUST be run via
    `privilege.run_as_user(...)` ... exactly as Phases 6/10 of
    `00_bootstrap.sh` do for `ngc`"), the whole write happens inside a
    `privilege.run_as_user("bash", "-lc", ...)` call -- this process never
    touches `~/.ngc/config` directly or `chown`s it after the fact. See the
    module docstring's "Design note" for how the key reaches that child
    process without ever becoming a CLI argument or a Python-source literal.

    Reads the key via `load_key(install_dir)` only to decide whether there
    is anything to configure. If no key has been stored yet (only possible
    before onboarding has run), this is a no-op that returns `None`.
    """
    if load_key(install_dir) is None:
        log.warn(
            f"{_REDACTED_LOG_LINE} not stored yet; skipping ~/.ngc/config."
        )
        return None

    secrets_path = pathlib.Path(install_dir) / _SECRET_RELATIVE_PATH
    user = privilege.resolve()
    config_path = user.home / ".ngc" / "config"

    result = privilege.run_as_user(
        "env",
        f"NGC_ENV_FILE={secrets_path}",
        "bash",
        "-lc",
        _CONFIGURE_NGC_CLI_SCRIPT,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        log.warn(
            f"Failed to write ~/.ngc/config for {user.name} "
            f"(exit {result.returncode})."
        )
        return None

    log.info(f"Wrote ~/.ngc/config for {user.name} ({_REDACTED_LOG_LINE}).")
    return config_path
