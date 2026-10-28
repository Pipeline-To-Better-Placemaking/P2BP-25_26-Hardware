"""Whether an interactive prompt can be shown to the operator.

Every module that asks the operator something calls
`prompt_unavailable_reason()` first. It lives in its own module, importing
nothing from the installer, so `config`, `ngc`, `webapp`,
`camera_credentials` and the steps can all use it without an import cycle
through `app`.

Callers look it up as `prompting.prompt_unavailable_reason()` at call time
rather than importing the name, so a test can replace it in one place.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Optional

__all__ = ["prompt_unavailable_reason"]


def prompt_unavailable_reason(stdin: Any = None, stdout: Any = None) -> Optional[str]:
    """Why an interactive prompt cannot be shown here, or `None` if it can.

    `input()` writes its prompt to stdout and reads stdin. When stdout is a
    pipe (`mv3dt-installer 2>&1 | tail -40`) the prompt goes into the pipe
    and the operator never sees it. When stdin is not a terminal there is
    nobody to answer. When the process is not in the terminal's foreground
    process group (the shape `sudo` gives a command at the head of a
    pipeline), reading the terminal stops the process with SIGTTIN, and a
    stopped process acts on no signal but SIGKILL. A caller that gets a
    reason back uses its documented default, or fails with a message when
    there is none, instead of blocking.
    """
    stdin = sys.stdin if stdin is None else stdin
    stdout = sys.stdout if stdout is None else stdout

    def _isatty(stream: Any) -> bool:
        try:
            return bool(stream is not None and stream.isatty())
        except (AttributeError, OSError, ValueError):
            return False

    if not _isatty(stdin):
        return "stdin is not a terminal"
    if not _isatty(stdout):
        return "stdout is not a terminal (is the output piped?)"
    try:
        if os.tcgetpgrp(stdin.fileno()) != os.getpgrp():
            return "the installer is not in the terminal's foreground process group"
    except (AttributeError, OSError, ValueError):
        pass
    return None
