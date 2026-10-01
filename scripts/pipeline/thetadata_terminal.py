#!/usr/bin/env python3
"""Launch Theta Terminal v3 using THETADATA_API_KEY from project .env."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

jar = ROOT / "tools" / "thetadata" / "ThetaTerminalv3.jar"
if not jar.exists():
    print("Missing", jar)
    print("Download: https://download-stable.thetadata.us/ThetaTerminalv3.jar")
    raise SystemExit(1)

key = os.getenv("THETADATA_API_KEY", "").strip()
if not key:
    print("THETADATA_API_KEY not set in .env")
    raise SystemExit(1)

cmd = ["java", "-jar", str(jar), "--api-key", key, "--dotenv-dir", str(ROOT)]
print("Starting Theta Terminal:", jar.name)
# Do not print the key
raise SystemExit(subprocess.call(cmd, cwd=str(jar.parent)))
