"""Heartbeat status for long-running PCS ThetaData scale job (read by /hot)."""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATUS_PATH = ROOT / "logs" / "research" / "pcs_thetadata_scale_status.json"
STALE_AFTER_SEC = 15 * 60
JOB_NAME = "pcs_thetadata_scale"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def status_path() -> Path:
    return Path(os.getenv("PCS_THETADATA_SCALE_STATUS", str(DEFAULT_STATUS_PATH)))


def write_status(payload: Dict[str, Any], *, path: Optional[Path] = None) -> Path:
    p = path or status_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    body = dict(payload)
    body.setdefault("job", JOB_NAME)
    body["updated_at"] = _utc_now_iso()
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(body, indent=2, default=str), encoding="utf-8")
    tmp.replace(p)
    return p


def read_status(*, path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    p = path or status_path()
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def effective_state(obj: Dict[str, Any], *, now: Optional[float] = None) -> str:
    """Map stored state; running with stale heartbeat becomes stale."""
    state = str(obj.get("state") or "unknown")
    if state != "running":
        return state
    updated = obj.get("updated_at")
    if not updated:
        return "stale"
    try:
        ts = datetime.fromisoformat(str(updated).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return "stale"
    age = (now if now is not None else time.time()) - ts
    if age > STALE_AFTER_SEC:
        return "stale"
    return "running"


def public_status(*, path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Shape for /hot / API consumers."""
    obj = read_status(path=path)
    if not obj:
        return None
    state = effective_state(obj)
    done = int(obj.get("done") or 0)
    total = int(obj.get("total") or 0)
    return {
        "job": obj.get("job") or JOB_NAME,
        "state": state,
        "stored_state": obj.get("state"),
        "started_at": obj.get("started_at"),
        "updated_at": obj.get("updated_at"),
        "done": done,
        "total": total,
        "ok": obj.get("ok"),
        "no_credit": obj.get("no_credit"),
        "last_key": obj.get("last_key"),
        "last_error": obj.get("last_error"),
        "outdir": obj.get("outdir"),
        "pid": obj.get("pid"),
        "progress_pct": (100.0 * done / total) if total > 0 else None,
        "stale_after_sec": STALE_AFTER_SEC,
    }
