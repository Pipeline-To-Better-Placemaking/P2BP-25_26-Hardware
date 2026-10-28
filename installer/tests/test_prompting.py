"""Tests for mv3dt_installer.prompting."""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from mv3dt_installer import prompting  # noqa: E402

# Bound before conftest's autouse fixture replaces the module attribute.
from mv3dt_installer.prompting import prompt_unavailable_reason  # noqa: E402


class _Stream:
    def __init__(self, tty, fd=0):
        self._tty = tty
        self._fd = fd

    def isatty(self):
        return self._tty

    def fileno(self):
        return self._fd


def test_prompt_unavailable_reason_none_on_a_foreground_terminal(monkeypatch):
    monkeypatch.setattr(prompting.os, "tcgetpgrp", lambda fd: 4242)
    monkeypatch.setattr(prompting.os, "getpgrp", lambda: 4242)
    assert prompt_unavailable_reason(_Stream(True), _Stream(True)) is None


def test_prompt_unavailable_reason_when_stdout_is_piped():
    reason = prompt_unavailable_reason(_Stream(True), _Stream(False))
    assert reason is not None and "stdout" in reason


def test_prompt_unavailable_reason_when_stdin_is_not_a_terminal():
    reason = prompt_unavailable_reason(_Stream(False), _Stream(True))
    assert reason is not None and "stdin" in reason


def test_prompt_unavailable_reason_in_the_background_process_group(monkeypatch):
    """Reading the terminal from a background process group stops the
    process with SIGTTIN, after which only SIGKILL ends it."""
    monkeypatch.setattr(prompting.os, "tcgetpgrp", lambda fd: 1)
    monkeypatch.setattr(prompting.os, "getpgrp", lambda: 2)
    reason = prompt_unavailable_reason(_Stream(True), _Stream(True))
    assert reason is not None and "foreground" in reason
