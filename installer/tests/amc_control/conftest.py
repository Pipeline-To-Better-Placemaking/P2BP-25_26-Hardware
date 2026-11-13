"""Make the bundled `mv3dt_amc_control` package importable.

It ships as an installer asset (run later from its own venv on the
workstation), so it is not on the installer's import path; the tests put
`assets/amc_control/` there instead, plus this folder for the fake AMC.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ASSET_ROOT = Path(__file__).resolve().parents[2] / "mv3dt_installer" / "assets" / "amc_control"
for _path in (str(_ASSET_ROOT), str(Path(__file__).resolve().parent)):
    if _path not in sys.path:
        sys.path.insert(0, _path)
