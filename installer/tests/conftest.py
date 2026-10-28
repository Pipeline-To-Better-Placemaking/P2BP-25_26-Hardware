"""Shared fixtures.

The one thing here is global-state hygiene. `logs` carries two pieces of
process-global state that a test can set and a later test would otherwise
inherit: the colour override (`logs.set_colour`, doc 08 §12.2 defect 4) and
the live-writer registration (`logs.set_live_writer`, defect 3). Both exist
for good reasons -- the colour question is answered before `argparse` runs,
and the renderer has to intercept writes it does not make -- but both are
sticky, and a leaked one makes a later test pass for a reason that has
nothing to do with what it asserts.

That is not hypothetical: without this fixture,
`test_dispatch_banner_degrades_to_a_plain_line_when_non_interactive` passes
in a full run and fails in isolation, because some earlier test had already
settled the colour.
"""

from __future__ import annotations

import pytest

from mv3dt_installer import logs, prompting


@pytest.fixture(autouse=True)
def _reset_logs_global_state():
    yield
    logs.set_colour(None)
    with logs._lock:
        logs._live_stream = None
        logs._live_writer = None


@pytest.fixture(autouse=True)
def _prompts_visible(monkeypatch):
    """pytest's stdin is never a terminal, so every prompt would take its
    hidden-prompt branch. Tests answer prompts through injected stand-ins,
    so treat prompts as visible unless a test patches this to return a
    reason. `tests/test_prompting.py` tests the real function."""
    monkeypatch.setattr(prompting, "prompt_unavailable_reason", lambda: None)
