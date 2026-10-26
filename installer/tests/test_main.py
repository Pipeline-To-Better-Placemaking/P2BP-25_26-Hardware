"""Tests for `mv3dt_installer.__main__` (doc 00 §3.1).

Guards against a regression where nothing on the real run path
(`__main__.py` -> `app.main()`) imports the seven `stepN_*.py` modules.
Each of those modules populates `steps.STEP_REGISTRY` (via
`steps.register(...)`) and, where applicable, `app.SUBCOMMAND_REGISTRY`
(via `app.register_subcommand(...)`) purely as an import-time side effect.
Without `__main__.py` importing them, both registries are empty at
runtime and the dispatch loop has no steps or subcommands to dispatch to.

Run from installer/: `python3 -m pytest tests/test_main.py -v`
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mv3dt_installer.__main__ as main_mod  # noqa: E402
from mv3dt_installer import app  # noqa: E402


def test_importing_main_populates_step_registry():
    """Importing `__main__` registers all eight steps (doc 00 §12.1)."""
    assert len(app.STEP_REGISTRY) == 8
    assert [step.order for step in app.STEP_REGISTRY] == sorted(
        step.order for step in app.STEP_REGISTRY
    )


def test_importing_main_populates_subcommand_registry():
    """Importing `__main__` registers every step-defined subcommand."""
    expected = {
        "amc",
        "ingest",
        "pipeline",
        "record",
        "projects",
        "agent",
        "reporter",
        "uploader",
        "tracking",
    }
    assert expected <= set(app.SUBCOMMAND_REGISTRY.keys())


def test_main_run_calls_app_main(monkeypatch):
    """`run()` forwards to `app.main()` and returns its exit code."""
    monkeypatch.setattr(app, "main", lambda: 7)
    assert main_mod.run() == 7
