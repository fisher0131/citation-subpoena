"""Runnable script entry point, for checkouts that are not installed.

The console script ``subpoena`` ships :func:`subpoena.cli.main`; this file keeps
``python scripts/audit_cli.py ...`` working from a source checkout.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from subpoena.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
