"""Poll Docker then bring up timescaledb and run parse-only backfill."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def docker_ready() -> bool:
    r = subprocess.run(["docker", "info"], capture_output=True, text=True)
    return r.returncode == 0


def main() -> int:
    print("Waiting for Docker engine (up to 15 min)...", flush=True)
    deadline = time.time() + 900
    while time.time() < deadline:
        if docker_ready():
            print("DOCKER_READY", flush=True)
            break
        print("waiting...", flush=True)
        time.sleep(20)
    else:
        print("DOCKER_TIMEOUT", flush=True)
        return 1

    # Ensure container up
    subprocess.run(
        ["docker", "compose", "up", "-d", "timescaledb"],
        cwd=str(ROOT),
        check=False,
    )
    # Wait for pg
    for i in range(60):
        r = subprocess.run(
            [
                "docker",
                "exec",
                "backtrader_timescaledb",
                "pg_isready",
                "-U",
                "backtrader_user",
                "-d",
                "backtrader",
            ],
            capture_output=True,
            text=True,
        )
        if r.returncode == 0:
            print("PG_READY", flush=True)
            break
        time.sleep(5)
    else:
        print("PG_TIMEOUT", flush=True)
        return 2

    schema = subprocess.run(
        [sys.executable, "scripts/db/apply_fundamentals_schema.py"],
        cwd=str(ROOT),
    )
    if schema.returncode != 0:
        return schema.returncode

    parse = subprocess.run(
        [
            sys.executable,
            "scripts/data/backfill_sec_fundamentals.py",
            "--universe",
            "daily",
            "--parse-only",
            "--workers",
            "4",
        ],
        cwd=str(ROOT),
    )
    return parse.returncode


if __name__ == "__main__":
    sys.exit(main())
