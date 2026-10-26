"""`python -m mv3dt_installer` entrypoint (doc 00 §3.1's module layout).

Thin on purpose: all flag parsing, orchestration, and the dispatch loop
live in `app.py` (doc 00 §3.2 -- "`__main__.py` -> `app.main()`"). This
file's only job is to call it and forward the process exit code.
"""

from __future__ import annotations

import sys

from mv3dt_installer import app

# Import every step module for its registration side effect: each populates
# `steps.STEP_REGISTRY` via `steps.register(...)` and, where it defines a
# subcommand, `app.SUBCOMMAND_REGISTRY` via `app.register_subcommand(...)`.
# Nothing else on the run path imports these modules, so without this block
# both registries are empty and `app.main()` has no steps or subcommands to
# dispatch to.
from mv3dt_installer.steps import step1_prerequisites  # noqa: F401
from mv3dt_installer.steps import step2_deepstream_sdk  # noqa: F401
from mv3dt_installer.steps import step3_amc_launcher  # noqa: F401
from mv3dt_installer.steps import step4_calib_output_wiring  # noqa: F401
from mv3dt_installer.steps import step5_per_project_exes  # noqa: F401
from mv3dt_installer.steps import step6_remote_supervision  # noqa: F401
from mv3dt_installer.steps import step7_webapp_integration  # noqa: F401
from mv3dt_installer.steps import step8_tracking_history  # noqa: F401


def run() -> int:
    return app.main()


if __name__ == "__main__":
    sys.exit(run())
