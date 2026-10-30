#!/usr/bin/env python3
"""Deprecated entrypoint — use scripts/trading/spxw_iron_fly_combo.py.

Keeps prior research docs working:
  python scripts\\research\\paper_spxw_iron_fly_combo.py --place
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
target = ROOT / "scripts" / "trading" / "spxw_iron_fly_combo.py"
sys.argv[0] = str(target)
runpy.run_path(str(target), run_name="__main__")
