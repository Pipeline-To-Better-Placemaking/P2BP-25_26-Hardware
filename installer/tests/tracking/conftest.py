"""Make the bundled `mv3dt_tracking` package importable.

It ships as an installer asset (run later from its own venv on the
workstation), so it is not on the installer's import path; the tests put
`assets/tracking/` there instead.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ASSET_ROOT = Path(__file__).resolve().parents[2] / "mv3dt_installer" / "assets" / "tracking"
if str(_ASSET_ROOT) not in sys.path:
    sys.path.insert(0, str(_ASSET_ROOT))
