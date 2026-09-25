"""mv3dt_installer — the single self-contained installer for the DeepStream
9.1 / AutoMagicCalib MV3DT workstation stack (installer/plan/00-FRAMEWORK-AND-BOOTSTRAP.md).

`__version__` below is the single source of truth for the installer's
version. `pyproject.toml` reads it back through
`[tool.setuptools.dynamic] version = {attr = "mv3dt_installer.__version__"}`
rather than carrying its own copy, so the two cannot drift, and
`.github/workflows/release.yml` refuses to publish a tag whose name is not
exactly `"v" + __version__`. Bumping the version is therefore an ordinary
source commit; pushing the matching tag is what publishes a release.
"""

import sys

__version__ = "0.4.8"

# Build stamp. `.github/workflows/release.yml` writes an untracked
# `_buildinfo.py` next to this file just before PyInstaller runs, so a
# released binary can name the exact tag, commit, and build time it came
# from. A source checkout has no such file and falls back to the values
# below; that fallback is why this import is allowed to fail instead of
# being a hard dependency of the package. The broad `except Exception`
# (not just `ImportError`) is deliberate: a malformed generated file should
# degrade to the same fallback, not crash every `import mv3dt_installer`.
try:
    from ._buildinfo import BUILT_UTC, COMMIT, TAG
except Exception:
    TAG, COMMIT, BUILT_UTC = "", "source", "unknown"
    _STAMPED = False
else:
    # `sys.frozen` is set by the PyInstaller bootloader on a running frozen
    # binary, and only there. Gating on it (not just on the import
    # succeeding) means an untracked `_buildinfo.py` left behind by an
    # earlier local `pyinstaller` build cannot be picked up the next time
    # the package runs unfrozen from source — the scenario the module's own
    # docstring below promises can never happen.
    _STAMPED = bool(getattr(sys, "frozen", False))
    if not _STAMPED:
        TAG, COMMIT, BUILT_UTC = "", "source", "unknown"


def build_info() -> tuple[str, str, str]:
    """Return `(tag, commit, built_utc)` for this build.

    A release binary returns the values CI stamped into `_buildinfo.py`,
    for example `("v0.2.0", "a1b2c3d", "2026-08-19T14:02:11Z")`. A source
    checkout — and a locally built binary, which has the same provenance
    story — returns `("", "source", "unknown")`: honest placeholders rather
    than values guessed from the working tree, which would claim a
    provenance the binary does not actually have.
    """
    return (TAG, COMMIT, BUILT_UTC)


def build_stamp() -> str:
    """Return the parenthesised provenance suffix for the `--version`
    banner, or an empty string when this is not a CI-stamped build.

    A release binary prints
    `mv3dt-installer 0.2.0 (v0.2.0, commit a1b2c3d, built 2026-08-19T14:02:11Z)`;
    a source checkout prints `mv3dt-installer 0.1.0` and nothing more.
    Keeping the suffix empty rather than filling it with the placeholders
    means an operator pasting `--version` into a bug report can never
    mistake a hand-built binary for a published release.

    The tag is dropped from the suffix when it is empty, which is the case
    for the CI builds that run on a pull request or a branch dispatch: those
    binaries have a real commit and build time but no release to name.
    """
    if not _STAMPED:
        return ""
    tag, commit, built = build_info()
    parts = [part for part in (tag, f"commit {commit}", f"built {built}") if part]
    return f" ({', '.join(parts)})"
