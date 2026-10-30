"""SPXW 0DTE short iron fly live state for /hot (timeframe=if).

Day filters: skip Wednesday; VIX session open < 20.
Entry clock: 13:45 ET. L5/S3/P2 sit-out can paper-skip the day.

Approaching from 13:30 ET; ENTER / BUY NOW from 13:45 (until ~14:00).

Fridays: on ENTER NOW, CronRunner parks a paper BAG via
``scripts/trading/spxw_iron_fly_combo.py --place --no-transmit`` (manual Transmit in TWS).
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import pandas as pd

from utils.scanning.spxw_iron_fly_l5 import (
    decide_today,
    load_state,
    seed_from_natural_trades,
)

LOG = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")
ROOT = Path(__file__).resolve().parents[2]

SYMBOL = "SPX"
TIMEFRAME_IF = "if"
VIX_MAX = 20.0
ENTRY_CLOCK = dtime(13, 45)
APPROACH_CLOCK = dtime(13, 30)
WATCH_END = dtime(14, 5)
WING = 15.0
FRIDAY_PARK_MARKER = ROOT / "logs" / "scanners" / "spxw_iron_fly_friday_park.json"
COMBO_SCRIPT = ROOT / "scripts" / "trading" / "spxw_iron_fly_combo.py"


def now_et() -> datetime:
    return datetime.now(ET)


def today_et() -> date:
    return now_et().date()


def is_wednesday(day: Optional[date] = None) -> bool:
    d = day or today_et()
    return d.weekday() == 2


def is_friday(day: Optional[date] = None) -> bool:
    d = day or today_et()
    return d.weekday() == 4


def in_approach_window(now: Optional[datetime] = None) -> bool:
    n = now or now_et()
    t = n.timetz().replace(tzinfo=None) if n.tzinfo else n.time()
    return APPROACH_CLOCK <= t < ENTRY_CLOCK


def in_entry_window(now: Optional[datetime] = None) -> bool:
    n = now or now_et()
    t = n.timetz().replace(tzinfo=None) if n.tzinfo else n.time()
    return ENTRY_CLOCK <= t <= WATCH_END


def in_watch_window(now: Optional[datetime] = None) -> bool:
    n = now or now_et()
    t = n.timetz().replace(tzinfo=None) if n.tzinfo else n.time()
    return APPROACH_CLOCK <= t <= WATCH_END and n.weekday() < 5


def fetch_vix_open_yahoo(day: Optional[date] = None) -> Optional[float]:
    """Session open for VIX via Yahoo (^VIX)."""
    try:
        import yfinance as yf
    except Exception as exc:  # noqa: BLE001
        LOG.warning("yfinance missing: %s", exc)
        return None
    d = day or today_et()
    start = (d - timedelta(days=5)).isoformat()
    end = (d + timedelta(days=1)).isoformat()
    try:
        hist = yf.download("^VIX", start=start, end=end, progress=False, auto_adjust=False)
    except Exception as exc:  # noqa: BLE001
        LOG.warning("Yahoo VIX failed: %s", exc)
        return None
    if hist is None or hist.empty:
        return None
    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = hist.columns.get_level_values(0)
    hist = hist.copy()
    hist.index = pd.DatetimeIndex(pd.to_datetime(hist.index)).tz_localize(None)
    # Prefer today's row
    for ts, row in hist.iloc[::-1].iterrows():
        if pd.Timestamp(ts).date() <= d:
            op = float(row["Open"])
            if op == op and op > 0:
                return op
    return None


def fetch_spx_last_yahoo() -> Optional[float]:
    try:
        import yfinance as yf

        hist = yf.Ticker("^GSPC").history(period="5d")
        if hist is None or hist.empty:
            return None
        return float(hist["Close"].iloc[-1])
    except Exception as exc:  # noqa: BLE001
        LOG.warning("Yahoo SPX failed: %s", exc)
        return None


def day_filters_ok(
    *,
    day: Optional[date] = None,
    vix_open: Optional[float] = None,
    vix_max: float = VIX_MAX,
) -> Tuple[bool, str, Optional[float]]:
    d = day or today_et()
    if is_wednesday(d):
        return False, "skip_wednesday", vix_open
    vix = vix_open if vix_open is not None else fetch_vix_open_yahoo(d)
    if vix is None:
        return False, "no_vix", None
    if float(vix) >= float(vix_max):
        return False, f"vix_ge_{vix_max:g}", float(vix)
    return True, "ok", float(vix)


def evaluate_live(
    *,
    prev_row: Optional[Dict[str, Any]] = None,
    now: Optional[datetime] = None,
    vix_open: Optional[float] = None,
    spot: Optional[float] = None,
    seed: bool = True,
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, str]]]:
    """Build /hot row + alert list for the iron fly."""
    n = now or now_et()
    day = n.date()
    alerts: List[Dict[str, str]] = []

    if seed:
        seed_from_natural_trades()

    ok, filt_reason, vix = day_filters_ok(day=day, vix_open=vix_open)
    state = load_state()
    l5 = decide_today(state, day)
    take_l5 = bool(l5.get("take"))
    px = spot if spot is not None else fetch_spx_last_yahoo()

    prev_notify = str((prev_row or {}).get("hot_notified_on") or "")
    today_iso = day.isoformat()
    approach_key = f"{today_iso}:approach"
    arm_key = f"{today_iso}:arm"
    skip_key = f"{today_iso}:sit"

    status = "idle"
    hot = False
    buy_ok = False
    notify_tag = prev_notify

    if not ok:
        status = "blocked"
        note = filt_reason
    elif not take_l5:
        status = "sit"
        note = str(l5.get("reason") or "l5_sit")
    else:
        note = str(l5.get("reason") or "active")
        if in_entry_window(n):
            status = "armed"
            hot = True
            buy_ok = True
        elif in_approach_window(n) or in_watch_window(n):
            status = "waiting"
            hot = True
            buy_ok = False
        else:
            status = "armed" if n.timetz().replace(tzinfo=None) < APPROACH_CLOCK else "idle"
            # Pre-approach: show armed candidate for the day if filters pass
            if n.weekday() < 5 and n.timetz().replace(tzinfo=None) < APPROACH_CLOCK:
                status = "armed"
                buy_ok = False

    # Alerts
    if ok and take_l5 and in_approach_window(n) and prev_notify != approach_key:
        alerts.append(
            {
                "kind": "approach",
                "text": (
                    f"SPXW iron fly APPROACHING — entry 13:45 ET today. "
                    f"VIX open {vix:.2f} (<{VIX_MAX:g}). L5 {note}. "
                    f"Sell 50d put / 40d call + {WING:.0f}pt wings (BAG). "
                    f"Detector IF on /hot."
                ),
            }
        )
        notify_tag = approach_key
    elif ok and take_l5 and in_entry_window(n) and prev_notify != arm_key:
        friday_note = (
            " Friday: auto-parking BAG with --place --no-transmit (Transmit in TWS)."
            if is_friday(day)
            else f" Paper: python scripts\\trading\\spxw_iron_fly_combo.py --place"
        )
        alerts.append(
            {
                "kind": "arm",
                "text": (
                    f"SPXW iron fly ENTER NOW — 13:45 ET window. "
                    f"VIX open {vix:.2f}. L5 {note}. "
                    f"1 lot BAG credit; hold to cash settle."
                    f"{friday_note}"
                ),
            }
        )
        notify_tag = arm_key
    elif ok and (not take_l5) and in_watch_window(n) and prev_notify != skip_key:
        alerts.append(
            {
                "kind": "approach",
                "text": (
                    f"SPXW iron fly PAPER / SIT — L5/S3/P2 {note}. "
                    f"VIX open {vix if vix is not None else 'n/a'}. "
                    f"Do not enter today; sit/probe circuit breaker."
                ),
            }
        )
        notify_tag = skip_key

    if status in ("idle",) and not in_watch_window(n) and not (
        n.weekday() < 5 and n.timetz().replace(tzinfo=None) < APPROACH_CLOCK and ok and take_l5
    ):
        # Outside RTH watch and not a pre-arm day card
        if not ok or not take_l5:
            return None, []

    as_of = n.astimezone(timezone.utc)
    row: Dict[str, Any] = {
        "stock": SYMBOL,
        "timeframe": TIMEFRAME_IF,
        "status": status if status != "idle" else ("armed" if (ok and take_l5) else "waiting"),
        "as_of": as_of,
        "h2_time": n if buy_ok else (prev_row or {}).get("h2_time"),
        "channel_start": None,
        "channel_end": None,
        "channel_span_days": None,
        "wait_bars": int(ENTRY_CLOCK.hour * 60 + ENTRY_CLOCK.minute),
        "wait_ok": bool(buy_ok),
        "support": float(WING),  # wing width (defined-risk hint)
        "resist": float(vix) if vix is not None else None,  # VIX open on Resist
        "last_close": float(px) if px is not None else None,
        "dist_to_resist_pct": None,
        "volume_rel_20": None,
        "overshoot_prior": None,
        "fill_px": None,
        "overshoot": None,
        "support_x0": None,
        "support_y0": None,
        "support_slope": None,
        "channel_width": float(WING),
        "h2_idx": None,
        "as_of_i": None,
        "last_price": float(px) if px is not None else None,
        "last_price_ts": n,
        "dist_live_pct": None,
        "hot": bool(hot),
        "hot_notified_on": notify_tag,
        "_vix": vix,
        "_filt": filt_reason if not ok else "ok",
        "_l5": l5,
        "_take": bool(ok and take_l5),
        "_note": note if ok else filt_reason,
    }
    # Encode L5 + filter into wait_ok / status for UI
    if not ok:
        row["status"] = "waiting"
        row["hot"] = False
        row["wait_ok"] = False
    elif not take_l5:
        row["status"] = "waiting"
        row["hot"] = False
        row["wait_ok"] = False
    return row, alerts


def candidate_db_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in row.items() if not str(k).startswith("_")}


def load_friday_park_marker(path: Optional[Path] = None) -> Dict[str, Any]:
    p = path or FRIDAY_PARK_MARKER
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        LOG.warning("Friday park marker unreadable: %s", exc)
        return {}


def friday_park_already_done(
    day: Optional[date] = None, *, path: Optional[Path] = None
) -> bool:
    """True if we already parked (or are parking) for this Friday session.

    ``place_failed`` / ``timeout`` are retryable on the next cron tick.
    Stale ``in_progress`` (older than 10 minutes) is also retryable.
    """
    d = day or today_et()
    data = load_friday_park_marker(path)
    if str(data.get("date") or "") != d.isoformat():
        return False
    status = str(data.get("status") or "")
    if status in ("ok", "parked_no_transmit"):
        return True
    if status == "in_progress":
        try:
            ts = datetime.fromisoformat(str(data.get("ts") or ""))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=ET)
            age = (now_et() - ts.astimezone(ET)).total_seconds()
            return age < 600.0
        except Exception:  # noqa: BLE001
            return True
    return False


def write_friday_park_marker(
    day: date,
    *,
    status: str,
    detail: Optional[Dict[str, Any]] = None,
    path: Optional[Path] = None,
) -> Path:
    p = path or FRIDAY_PARK_MARKER
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "date": day.isoformat(),
        "status": status,
        "ts": now_et().isoformat(),
    }
    if detail:
        payload["detail"] = detail
    p.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return p


def should_auto_park_friday(
    *,
    day: Optional[date] = None,
    buy_ok: bool = False,
    path: Optional[Path] = None,
) -> bool:
    """Friday ENTER NOW and we have not already parked today."""
    d = day or today_et()
    if not is_friday(d):
        return False
    if not buy_ok:
        return False
    if friday_park_already_done(d, path=path):
        return False
    return True


def run_friday_auto_park(
    *,
    day: Optional[date] = None,
    marker_path: Optional[Path] = None,
    timeout_sec: int = 180,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Park paper BAG with ``--place --no-transmit`` (once per Friday).

    Returns a small result dict; does not transmit. Manual Transmit in TWS.
    """
    d = day or today_et()
    out: Dict[str, Any] = {
        "date": d.isoformat(),
        "status": "skipped",
        "cmd": [
            sys.executable,
            str(COMBO_SCRIPT),
            "--place",
            "--no-transmit",
        ],
    }
    if dry_run:
        out["status"] = "dry_run"
        LOG.info("[dry-run] would run: %s", " ".join(out["cmd"]))
        return out
    if not COMBO_SCRIPT.is_file():
        out["status"] = "missing_script"
        LOG.error("Combo script missing: %s", COMBO_SCRIPT)
        return out

    write_friday_park_marker(d, status="in_progress", path=marker_path)
    cmd = out["cmd"]
    LOG.info("Friday auto-park: %s", " ".join(cmd))
    env = os.environ.copy()
    # Ensure project imports resolve the same way CronRunner bats do.
    pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(ROOT) if not pp else f"{ROOT}{os.pathsep}{pp}"
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=int(timeout_sec),
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        LOG.error("Friday auto-park timed out after %ss", timeout_sec)
        out["status"] = "timeout"
        out["error"] = str(exc)
        write_friday_park_marker(d, status="timeout", detail=out, path=marker_path)
        return out
    except Exception as exc:  # noqa: BLE001
        LOG.exception("Friday auto-park failed to spawn: %s", exc)
        out["status"] = "spawn_error"
        out["error"] = str(exc)
        write_friday_park_marker(d, status="spawn_error", detail=out, path=marker_path)
        return out

    out["returncode"] = int(proc.returncode)
    # Keep logs short in marker; full stdout goes to scanner log
    if proc.stdout:
        LOG.info("Friday auto-park stdout (tail):\n%s", proc.stdout[-4000:])
        out["stdout_tail"] = proc.stdout[-2000:]
    if proc.stderr:
        LOG.warning("Friday auto-park stderr (tail):\n%s", proc.stderr[-4000:])
        out["stderr_tail"] = proc.stderr[-2000:]

    if proc.returncode == 0:
        out["status"] = "parked_no_transmit"
        write_friday_park_marker(
            d, status="parked_no_transmit", detail={"returncode": 0}, path=marker_path
        )
        LOG.info("Friday auto-park OK — Transmit the BAG in TWS Orders")
    else:
        out["status"] = "place_failed"
        write_friday_park_marker(
            d,
            status="place_failed",
            detail={"returncode": proc.returncode},
            path=marker_path,
        )
        LOG.error("Friday auto-park failed rc=%s", proc.returncode)
    return out
