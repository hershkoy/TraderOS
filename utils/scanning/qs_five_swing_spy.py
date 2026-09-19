"""QS Five-Swing SPY live state for /hot (timeframe=qs).

Combined S1->S5 priority book (same rules as utils.research.qs_five_swing).
Shared QS exit: close > prior day's high.

Live MOC path (mirrors Deep Pullback):
  - Approaching Telegram from 15:45 ET when developing OHLC arms combined
  - Sticky row + minute Alpaca last refresh; /hot shows current / target
    (green = still buy-ok)
  - BUY NOW from 15:57 only while last still at/below the sleeve close ceiling
"""
from __future__ import annotations

import logging
from datetime import datetime, time, timezone
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import pandas as pd

from utils.research.qs_five_swing import STRATEGY_IDS, STRATEGY_NAMES, prepare_frame
from utils.research.qs_five_swing_realism import (
    _entry_flags_developing,
    _priority_sid_from_flags,
)
from utils.scanning.deep_pullback_spy import (
    ET,
    MOC_APPROACH_WINDOW_MIN,
    MOC_BUY_NOW_WINDOW_MIN,
    in_moc_alert_window,
    in_moc_buy_now_window,
    load_spy_daily,
    minutes_to_rth_close,
    session_date_et,
    session_moc_watching,
)

logger = logging.getLogger(__name__)

TIMEFRAME_QS = "qs"
SYMBOL = "SPY"
RTH_CLOSE = time(16, 0)


def load_spy_1m_session(client: Any, day_et: Any) -> pd.DataFrame:
    """Load SPY 1m bars for one ET calendar day (RTH window preferred)."""
    start = datetime.combine(day_et, time(9, 30), tzinfo=ET).astimezone(timezone.utc)
    end = datetime.combine(day_et, time(16, 5), tzinfo=ET).astimezone(timezone.utc)
    q = """
    SELECT ts, open, high, low, close, volume
    FROM market_data
    WHERE symbol = 'SPY' AND timeframe = '1m'
      AND ts >= %s AND ts <= %s
    ORDER BY ts
    """
    rows = client.execute_query(q, (start, end))
    if not rows:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["close"]).reset_index(drop=True)


def build_developing_ohlc(
    *,
    prior_close: float,
    last: float,
    intraday: Optional[pd.DataFrame] = None,
) -> Dict[str, float]:
    """Developing session OHLC; close = live last."""
    c = float(last)
    if intraday is not None and not intraday.empty:
        o = float(intraday.iloc[0]["open"])
        h = float(intraday["high"].max())
        l = float(intraday["low"].min())
        h = max(h, c, o)
        l = min(l, c, o)
        return {"open": o, "high": h, "low": l, "close": c}
    o = float(prior_close)
    return {"open": o, "high": max(o, c), "low": min(o, c), "close": c}


def _band_and_priors(d: pd.DataFrame, i: int, dev: Dict[str, float]) -> Dict[str, float]:
    """Priors used by developing flags / close ceiling (mirror realism helper)."""
    rng = float(dev["high"] - dev["low"])
    if i >= 1:
        high_10 = float(d["high"].iloc[max(0, i - 9) : i].max())
        high_10 = max(high_10, float(dev["high"]))
        prior_ranges = d["range"].iloc[max(0, i - 24) : i]
        if len(prior_ranges) >= 24:
            avg_r = float((prior_ranges.sum() + rng) / 25.0)
        else:
            avg_r = float("nan")
        min_low_5 = float(d["low"].iloc[max(0, i - 5) : i].min()) if i >= 5 else float("nan")
        max_high_10 = float(d["high"].iloc[max(0, i - 10) : i].max()) if i >= 10 else float("nan")
    else:
        high_10 = float(dev["high"])
        avg_r = float("nan")
        min_low_5 = float("nan")
        max_high_10 = float("nan")
    band = high_10 - 2.5 * avg_r if pd.notna(avg_r) else float("nan")
    return {
        "rng": rng,
        "band": float(band) if pd.notna(band) else float("nan"),
        "min_low_5": float(min_low_5) if pd.notna(min_low_5) else float("nan"),
        "max_high_10": float(max_high_10) if pd.notna(max_high_10) else float("nan"),
        "prev_close": float(d.iloc[i - 1]["close"]) if i >= 1 else float("nan"),
    }


def moc_close_target(d: pd.DataFrame, i: int, dev: Dict[str, float], sid: str) -> Optional[float]:
    """Max close that still keeps sleeve `sid` armed given developing H/L."""
    eps = 1e-4
    h = float(dev["high"])
    l = float(dev["low"])
    rng = h - l
    p = _band_and_priors(d, i, dev)
    sid = str(sid or "").upper()
    if sid == "S1":
        if not pd.notna(p["band"]):
            return None
        ibs_cap = (l + 0.3 * rng) if rng > 0 else float(dev["close"])
        return float(min(p["band"], ibs_cap) - eps)
    if sid == "S2":
        if not pd.notna(p["prev_close"]):
            return None
        return float(p["prev_close"] - eps)
    if sid == "S3":
        if not pd.notna(p["min_low_5"]):
            return None
        return float(p["min_low_5"] - eps)
    if sid == "S4":
        # Close does not gate S4 (range + prior ADX); treat high as display target.
        return float(h)
    if sid == "S5":
        ibs_cap = (l + 0.15 * rng) if rng > 0 else float(dev["close"])
        return float(ibs_cap - eps)
    return None


def moc_price_buy_ok(last: float, target: Optional[float], *, sid: Optional[str], signal: bool) -> bool:
    """Green when combined still arms and last is at/below the sleeve close ceiling."""
    if not signal:
        return False
    if sid == "S4":
        return True
    if target is None:
        return False
    return float(last) <= float(target)


def evaluate_live(
    daily: pd.DataFrame,
    *,
    last_price: Optional[float],
    prev_row: Optional[Dict[str, Any]] = None,
    now: Optional[datetime] = None,
    intraday_1m: Optional[pd.DataFrame] = None,
    moc_window_min: int = MOC_APPROACH_WINDOW_MIN,
    moc_buy_now_min: int = MOC_BUY_NOW_WINDOW_MIN,
) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    """Build /hot qs row + optional Telegram payloads (approach / arm / exit)."""
    now = now or datetime.now(timezone.utc)
    now_et = now.astimezone(ET)
    today_et = now_et.date()
    today_iso = today_et.isoformat()
    alerts: List[Dict[str, str]] = []
    if daily is None or daily.empty or last_price is None or float(last_price) <= 0:
        return {}, alerts

    d = prepare_frame(daily)
    if d.empty or len(d) < 40:
        return {}, alerts

    last_completed = d.iloc[-1]
    bar_day = session_date_et(last_completed["ts"])
    # Index of "today" row: append synthetic if last bar is prior session
    if bar_day == today_et:
        i = len(d) - 1
        prior_close = float(d.iloc[i - 1]["close"]) if i >= 1 else float(d.iloc[i]["close"])
    else:
        # Append placeholder row for today (features from history; OHLC overwritten)
        prior_close = float(last_completed["close"])
        extra = last_completed.copy()
        extra["ts"] = pd.Timestamp(datetime.combine(today_et, time(16, 0), tzinfo=ET)).tz_convert("UTC")
        extra["open"] = prior_close
        extra["high"] = prior_close
        extra["low"] = prior_close
        extra["close"] = float(last_price)
        extra["weekday"] = today_et.weekday()
        d = pd.concat([d, extra.to_frame().T], ignore_index=True)
        d = prepare_frame(d)  # refresh rolling cols with placeholder
        i = len(d) - 1

    approach_window = in_moc_alert_window(now, window_min=moc_window_min)
    buy_now_window = in_moc_buy_now_window(now, window_min=moc_buy_now_min)
    mins_left = minutes_to_rth_close(now)

    live_px = float(last_price)
    dev = build_developing_ohlc(
        prior_close=prior_close,
        last=live_px,
        intraday=intraday_1m,
    )
    flags = _entry_flags_developing(d, i, dev, ibs_buffer=0.0)
    sid = _priority_sid_from_flags(flags)
    signal = sid is not None
    target = moc_close_target(d, i, dev, sid) if sid else None
    # If no sleeve armed, still show a provisional ceiling from nearest S3/S1 if useful
    if target is None and signal is False:
        # Prefer S3 then S1 ceilings for display while sticky-watching
        for cand in ("S3", "S1", "S2", "S5"):
            t = moc_close_target(d, i, dev, cand)
            if t is not None:
                target = t
                break
    buy_ok = moc_price_buy_ok(live_px, target, sid=sid, signal=signal)

    prev_status = str((prev_row or {}).get("status") or "")
    prev_fill = (prev_row or {}).get("fill_px")
    prev_notify = str((prev_row or {}).get("hot_notified_on") or "")
    in_position = prev_status in ("filled", "in_position", "exit_signal") and prev_fill is not None
    watching = session_moc_watching(prev_notify, today_iso)
    sticky_watch = watching and mins_left is not None

    status = "idle"
    hot = False
    notify_tag = prev_notify
    arm_key = "%s:arm" % today_iso
    approach_key = "%s:approach" % today_iso
    prior_high = float(d.iloc[i - 1]["high"]) if i >= 1 else None

    if in_position:
        if prior_high is not None and live_px > float(prior_high):
            status = "exit_signal"
            if prev_status != "exit_signal":
                alerts.append(
                    {
                        "kind": "exit",
                        "text": (
                            "QS Five-Swing SPY EXIT: last %.2f > prior high %.2f "
                            "(entry %.2f). Sell MOC / near close."
                            % (live_px, float(prior_high), float(prev_fill))
                        ),
                    }
                )
        else:
            status = "filled"
    elif sticky_watch or signal:
        if signal and buy_ok:
            status = "armed"
            hot = True
        else:
            status = "waiting"

        if (
            approach_window
            and not buy_now_window
            and signal
            and target is not None
            and prev_notify not in (approach_key, arm_key)
        ):
            mins_txt = "?" if mins_left is None else ("%.0f" % mins_left)
            sname = STRATEGY_NAMES.get(sid or "", sid or "?")
            alerts.append(
                {
                    "kind": "approach",
                    "text": (
                        "QS Five-Swing SPY approaching (%sm to close) — %s. "
                        "If SPY stays at or below %.2f, send MOC buy at ~15:57. "
                        "Now last=%.2f."
                        % (mins_txt, sname, float(target), live_px)
                    ),
                }
            )
            notify_tag = approach_key
            watching = True
            sticky_watch = mins_left is not None

        if (
            buy_now_window
            and buy_ok
            and prev_notify != arm_key
            and (watching or signal)
            and target is not None
        ):
            mins_txt = "?" if mins_left is None else ("%.0f" % mins_left)
            sname = STRATEGY_NAMES.get(sid or "", sid or "?")
            alerts.append(
                {
                    "kind": "arm",
                    "text": (
                        "QS Five-Swing SPY BUY NOW — send MOC (%sm to close). "
                        "%s: last=%.2f <= target %.2f."
                        % (mins_txt, sname, live_px, float(target))
                    ),
                }
            )
            notify_tag = arm_key
            status = "armed"
            hot = True

    if status == "idle":
        return {}, alerts

    fill_px = float(prev_fill) if in_position and prev_fill is not None else None
    as_of = pd.Timestamp(d.iloc[i]["ts"]).to_pydatetime()
    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=timezone.utc)

    sleeve_ord = {s: n + 1 for n, s in enumerate(STRATEGY_IDS)}
    wait_bars = int(sleeve_ord.get(sid or "", 0))

    row: Dict[str, Any] = {
        "stock": SYMBOL,
        "timeframe": TIMEFRAME_QS,
        "status": status,
        "as_of": as_of,
        "h2_time": (prev_row or {}).get("h2_time") if in_position else None,
        "channel_start": None,
        "channel_end": None,
        "channel_span_days": None,
        "wait_bars": wait_bars,
        "wait_ok": bool(buy_ok) if not in_position else False,
        "support": float(prior_high) if prior_high is not None else float(prior_close),
        "resist": float(target) if target is not None else live_px,
        "last_close": float(prior_close),
        "dist_to_resist_pct": None
        if target in (None, 0)
        else round((live_px / float(target) - 1.0) * 100.0, 4),
        "volume_rel_20": None,
        "overshoot_prior": None,
        "fill_px": fill_px,
        "overshoot": None if prior_high is None else round(float(prior_high), 4),
        "support_x0": None,
        "support_y0": prior_high,
        "support_slope": None,
        "channel_width": None,
        "h2_idx": None,
        "as_of_i": None,
        "last_price": live_px,
        "last_price_ts": now,
        "dist_live_pct": None
        if target in (None, 0)
        else round((live_px / float(target) - 1.0) * 100.0, 4),
        "hot": bool(hot),
        "hot_notified_on": notify_tag,
        "_developing": True,
        "_sid": sid,
        "_flags": flags,
        "_target": target,
        "_moc_buy_ok": buy_ok,
        "_mins_to_close": mins_left,
        "_sticky_watch": sticky_watch,
    }
    return row, alerts


def mark_entered(row: Dict[str, Any], *, entry_px: float, entry_ts: datetime) -> Dict[str, Any]:
    out = dict(row)
    out["status"] = "filled"
    out["fill_px"] = float(entry_px)
    out["h2_time"] = entry_ts
    out["hot"] = False
    return out


def candidate_db_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in row.items() if not str(k).startswith("_")}
