"""QQQ LEAPS gap live state for /hot (timeframe=lg).

Research rules (Options With Ravish / claim-check):
  Strat1: open gap <= -1% AND prior close > SMA100; buy 1x ~60d ~365 DTE call.
  Strat2: open gap <= -2%; buy 2x same; TP +50%; no stop.

Live:
  - Prior close + SMA100 from TimescaleDB QQQ 1d (completed bars).
  - Alpaca last (and session open when available) for developing gap %.
  - Soft "hot / approaching" when estimated gap gets within NEAR_* of the
    -1% / -2% triggers (premarket + early RTH).
  - "Armed / BUY" when official open (or last after open) meets the rule.

/hot mapping:
  resist  = nearest buy-trigger open (strat1 target if SMA ok else strat2)
  support = prior SMA100
  Last UI = current / target (green when gap already meets trigger)
  wait_bars = score (0-3); wait_ok = buy-ok for active mode
"""
from __future__ import annotations

import logging
from datetime import datetime, time, timezone
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from utils.scanning.channel_touch_candidates_store import approach_already_notified

logger = logging.getLogger(__name__)

TIMEFRAME_LG = "lg"
SYMBOL = "QQQ"
ET = ZoneInfo("America/New_York")

GAP_STRAT1_PCT = 1.0
GAP_STRAT2_PCT = 2.0
# Soft band: fire approaching/hot when gap_est reaches this (less negative than full)
NEAR_STRAT1_PCT = 0.70  # within ~0.3% of -1%
NEAR_STRAT2_PCT = 1.50  # within ~0.5% of -2%
SMA_LEN = 100

RTH_OPEN = time(9, 30)
RTH_CLOSE = time(16, 0)
# Watch premarket from this hour (ET) for approaching
PREMARKET_START = time(8, 0)
# After this, stop sticky approaching (gap is an open trade)
WATCH_END = time(11, 0)


def load_qqq_daily(client: Any, *, lookback_bars: int = 400) -> pd.DataFrame:
    """Load QQQ 1d bars (IB prefix + Alpaca preferred)."""
    q = """
    SELECT provider, ts, open, high, low, close, volume
    FROM market_data
    WHERE symbol = 'QQQ' AND timeframe = '1d'
    ORDER BY provider, ts
    """
    rows = client.execute_query(q)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(
        rows, columns=["provider", "ts", "open", "high", "low", "close", "volume"]
    )
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    alp = df[df["provider"] == "ALPACA"].sort_values("ts").drop_duplicates("ts")
    ib = df[df["provider"] == "IB"].sort_values("ts").drop_duplicates("ts")
    if len(alp) and len(ib):
        first_alp = alp["ts"].min()
        out = pd.concat([ib[ib["ts"] < first_alp], alp], ignore_index=True)
    elif len(alp):
        out = alp
    else:
        out = ib
    out = out.sort_values("ts").drop_duplicates("ts", keep="last").reset_index(drop=True)
    if lookback_bars and len(out) > lookback_bars:
        out = out.iloc[-lookback_bars:].reset_index(drop=True)
    return out[["ts", "open", "high", "low", "close", "volume"]]


def session_date_et(ts: Any) -> Any:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return t.tz_convert(ET).date()


def now_et(now: Optional[datetime] = None) -> datetime:
    return (now or datetime.now(timezone.utc)).astimezone(ET)


def in_watch_window(now: Optional[datetime] = None) -> bool:
    """Premarket 08:00 through 11:00 ET weekdays (gap is an open event)."""
    et = now_et(now)
    if et.weekday() >= 5:
        return False
    t = et.time()
    return PREMARKET_START <= t <= WATCH_END


def rth_opened(now: Optional[datetime] = None) -> bool:
    et = now_et(now)
    if et.weekday() >= 5:
        return False
    return et.time() >= RTH_OPEN


def gap_pct(px: float, prior_close: float) -> Optional[float]:
    if prior_close is None or prior_close <= 0 or px is None:
        return None
    return (float(px) / float(prior_close) - 1.0) * 100.0


def levels_from_completed(closes: np.ndarray) -> Dict[str, Any]:
    """Causal levels from completed daily closes (through yesterday)."""
    n = len(closes)
    if n < SMA_LEN + 1:
        return {"ok": False, "reason": "need_sma100"}
    prior_close = float(closes[-1])
    # SMA100 of last 100 completed closes; prior_sma for "prior close > SMA"
    # matches research: prior_sma100 = sma100.shift(1) at today's bar
    # At open today, SMA through yesterday = mean(closes[-100:]), compare prior_close to it.
    sma100 = float(np.mean(closes[-SMA_LEN:]))
    above_sma = prior_close > sma100
    t1 = prior_close * (1.0 - GAP_STRAT1_PCT / 100.0)
    t2 = prior_close * (1.0 - GAP_STRAT2_PCT / 100.0)
    return {
        "ok": True,
        "prior_close": prior_close,
        "sma100": sma100,
        "above_sma100_prior": above_sma,
        "strat1_target": round(t1, 4),
        "strat2_target": round(t2, 4),
        # Prefer strat1 target on /hot when SMA ok; else strat2 only
        "display_target": round(t1 if above_sma else t2, 4),
    }


def classify_gap(
    gap_est: Optional[float],
    *,
    above_sma: bool,
    near1: float = NEAR_STRAT1_PCT,
    near2: float = NEAR_STRAT2_PCT,
    full1: float = GAP_STRAT1_PCT,
    full2: float = GAP_STRAT2_PCT,
) -> Dict[str, Any]:
    """Score soft/full strat1/strat2 from a gap % (negative = gap down)."""
    if gap_est is None:
        return {
            "ok": False,
            "gap_est": None,
            "score": 0,
            "near_s1": False,
            "near_s2": False,
            "full_s1": False,
            "full_s2": False,
            "approaching": False,
            "full_signal": False,
            "active_mode": None,
            "contracts": 0,
        }
    g = float(gap_est)
    full_s2 = g <= -abs(full2)
    full_s1 = above_sma and g <= -abs(full1)
    near_s2 = (not full_s2) and g <= -abs(near2)
    near_s1 = above_sma and (not full_s1) and g <= -abs(near1)
    score = 0
    if above_sma:
        score += 1
    if near_s1 or full_s1 or near_s2 or full_s2:
        score += 1
    if full_s1 or full_s2:
        score += 1
    if abs(g) >= abs(near2) or full_s2:
        score = max(score, 2 if above_sma else 1)
    full_signal = full_s1 or full_s2
    approaching = (near_s1 or near_s2) and not full_signal
    if full_s2:
        mode, ctr = "strat2", 2
    elif full_s1:
        mode, ctr = "strat1", 1
    elif near_s2:
        mode, ctr = "strat2", 2
    elif near_s1:
        mode, ctr = "strat1", 1
    else:
        mode, ctr = None, 0
    return {
        "ok": True,
        "gap_est": round(g, 3),
        "score": int(score),
        "near_s1": near_s1,
        "near_s2": near_s2,
        "full_s1": full_s1,
        "full_s2": full_s2,
        "approaching": approaching,
        "full_signal": full_signal,
        "active_mode": mode,
        "contracts": ctr,
    }


def fetch_alpaca_qqq_marks() -> Dict[str, Optional[float]]:
    """QQQ last + optional session open from Alpaca snapshot."""
    from utils.scanning.channel_touch_15m import (
        ALPACA_CONNECT_TIMEOUT_SEC,
        ALPACA_READ_TIMEOUT_SEC,
        _alpaca_snapshot_client,
    )
    from utils.config.env_loader import get_env_var, load_env_file
    import os

    from alpaca.data.enums import DataFeed
    from alpaca.data.requests import StockSnapshotRequest

    load_env_file()
    key = (get_env_var("ALPACA_API_KEY_ID") or os.environ.get("ALPACA_API_KEY_ID") or "").strip()
    secret = (get_env_var("ALPACA_API_SECRET") or os.environ.get("ALPACA_API_SECRET") or "").strip()
    out: Dict[str, Optional[float]] = {"last": None, "open": None, "prev_close": None}
    if not key or not secret:
        logger.warning("Alpaca keys missing; QQQ marks unavailable")
        return out
    try:
        client = _alpaca_snapshot_client(
            key, secret, timeout=(ALPACA_CONNECT_TIMEOUT_SEC, ALPACA_READ_TIMEOUT_SEC)
        )
        req = StockSnapshotRequest(symbol_or_symbols=["QQQ"], feed=DataFeed.IEX)
        data = client.get_stock_snapshot(req)
        snap = None
        if hasattr(data, "items"):
            for sym, s in data.items():
                if str(sym).upper() == "QQQ":
                    snap = s
                    break
        if snap is None and isinstance(data, dict):
            snap = data.get("QQQ") or data.get("qqq")
        if snap is None:
            return out
        trade = getattr(snap, "latest_trade", None)
        px = getattr(trade, "price", None) if trade is not None else None
        if px is not None and float(px) > 0:
            out["last"] = float(px)
        daily = getattr(snap, "daily_bar", None) or getattr(snap, "minute_bar", None)
        op = getattr(daily, "open", None) if daily is not None else None
        if op is not None and float(op) > 0:
            out["open"] = float(op)
        prev = getattr(snap, "previous_daily_bar", None)
        pc = getattr(prev, "close", None) if prev is not None else None
        if pc is not None and float(pc) > 0:
            out["prev_close"] = float(pc)
    except Exception as exc:
        logger.warning("Alpaca QQQ snapshot failed: %s", exc)
    return out


def evaluate_live(
    daily: pd.DataFrame,
    *,
    last_price: Optional[float],
    session_open: Optional[float] = None,
    prev_row: Optional[Dict[str, Any]] = None,
    now: Optional[datetime] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    """Build /hot lg row + Telegram/browser alert payloads.

    alerts kinds: approach (hot near trigger), arm (gap confirmed BUY).
    """
    now = now or datetime.now(timezone.utc)
    et = now_et(now)
    today_iso = et.date().isoformat()
    alerts: List[Dict[str, str]] = []
    if daily is None or daily.empty or len(daily) < SMA_LEN + 2:
        return {}, alerts

    d = daily.copy()
    closes = d["close"].to_numpy(dtype=float)
    last_completed = d.iloc[-1]
    bar_day = session_date_et(last_completed["ts"])

    if bar_day == et.date():
        completed = closes[:-1].copy()
    else:
        completed = closes.copy()

    levels = levels_from_completed(completed)
    if not levels.get("ok"):
        return {}, alerts

    prior_close = float(levels["prior_close"])
    above_sma = bool(levels["above_sma100_prior"])
    target = float(levels["display_target"])
    t1 = float(levels["strat1_target"])
    t2 = float(levels["strat2_target"])

    live_px = float(last_price) if last_price is not None and float(last_price) > 0 else None
    # Gap for soft hot: use last (premarket or RTH). Confirm arm prefers session open.
    gap_live = gap_pct(live_px, prior_close) if live_px is not None else None
    opened = rth_opened(now)
    open_px = float(session_open) if session_open and float(session_open) > 0 else None
    if open_px is None and opened and live_px is not None:
        # Persist first post-open print as open proxy across minutes
        prev_open = (prev_row or {}).get("fill_px")  # reuse unused until armed
        # Better: store in overshoot field as open; use support_y0 below
        stored = (prev_row or {}).get("support_y0")
        if stored is not None and float(stored) > 0:
            open_px = float(stored)
        else:
            open_px = live_px
    gap_open = gap_pct(open_px, prior_close) if open_px is not None else None

    # Soft classification from live gap; full arm uses open gap once RTH opened
    soft = classify_gap(gap_live, above_sma=above_sma)
    hard = (
        classify_gap(gap_open, above_sma=above_sma)
        if opened and gap_open is not None
        else soft
    )

    prev_status = str((prev_row or {}).get("status") or "")
    prev_notify = str((prev_row or {}).get("hot_notified_on") or "")
    approach_key = "%s:approach" % today_iso
    arm_key = "%s:arm" % today_iso
    watching = approach_already_notified(prev_notify, today_iso)
    approach_sent = watching
    sticky = watching and in_watch_window(now)

    status = "idle"
    hot = False
    notify_tag = prev_notify
    buy_ok = False
    active = hard if (opened and hard.get("full_signal")) else soft

    if sticky or soft.get("approaching") or soft.get("full_signal") or hard.get("full_signal"):
        if opened and hard.get("full_signal"):
            status = "armed"
            hot = True
            buy_ok = True
            active = hard
        elif soft.get("approaching") or soft.get("full_signal"):
            status = "waiting"
            hot = True
            buy_ok = bool(soft.get("full_signal"))
            active = soft
        elif sticky:
            status = "waiting"
            hot = bool(soft.get("approaching") or soft.get("full_signal"))
            buy_ok = bool(soft.get("full_signal"))

        # Approaching / hot: once per session while soft near trigger.
        # A premarket print already through -1%/-2% is still "watch the open",
        # not BUY NOW — the official open is the arm.
        full_preopen = bool(soft.get("full_signal") and not opened)
        if approach_sent and prev_notify == today_iso:
            notify_tag = approach_key
        if (
            in_watch_window(now)
            and not approach_sent
            and (soft.get("approaching") or full_preopen)
        ):
            mode = soft.get("active_mode") or "?"
            need = GAP_STRAT2_PCT if mode == "strat2" else GAP_STRAT1_PCT
            sma_txt = "prior>SMA100" if above_sma else "SMA100 FAIL (strat1 blocked)"
            gap_txt = "%+.2f" % float(gap_live or 0)
            if full_preopen:
                lead = (
                    "QQQ LEAPS gap HOT — %s already at gap_est=%s%% "
                    "(trigger <= -%.1f%%) before the open."
                    % (mode, gap_txt, need)
                )
            else:
                lead = (
                    "QQQ LEAPS gap HOT — approaching %s (need gap <= -%.1f%%). "
                    "gap_est=%s%%."
                    % (mode, need, gap_txt)
                )
            alerts.append(
                {
                    "kind": "approach",
                    "text": (
                        "%s last=%.2f prior=%.2f target_open=%.2f. %s. "
                        "Watch for official open; strat1=1ct strat2=2ct ~60d LEAPS TP50."
                        % (
                            lead,
                            float(live_px or 0),
                            prior_close,
                            t2 if mode == "strat2" else t1,
                            sma_txt,
                        )
                    ),
                }
            )
            notify_tag = approach_key
            watching = True
            sticky = in_watch_window(now)
            status = "waiting" if status == "idle" else status
            hot = True

        # Armed / BUY once RTH open confirms full gap
        if (
            opened
            and hard.get("full_signal")
            and prev_notify != arm_key
        ):
            mode = hard.get("active_mode") or "?"
            ctr = int(hard.get("contracts") or 1)
            alerts.append(
                {
                    "kind": "arm",
                    "text": (
                        "QQQ LEAPS gap BUY NOW — %s confirmed (gap=%+.2f%% open=%.2f). "
                        "Buy %dct ~60-delta ~365 DTE call; TP +50%% premium; no stop. "
                        "prior=%.2f SMA100=%.2f (%s)."
                        % (
                            mode,
                            float(gap_open or 0),
                            float(open_px or 0),
                            ctr,
                            prior_close,
                            float(levels["sma100"]),
                            "above" if above_sma else "below",
                        )
                    ),
                }
            )
            notify_tag = arm_key
            status = "armed"
            hot = True
            buy_ok = True
            active = hard

    if status == "idle":
        return {}, alerts

    px_show = float(live_px if live_px is not None else prior_close)
    mode = active.get("active_mode")
    mode_target = t2 if mode == "strat2" else t1
    dist_pct = None
    if mode_target > 0:
        # Negative = still above trigger (need lower); positive = through trigger
        dist_pct = round((px_show / mode_target - 1.0) * 100.0, 4)

    as_of = pd.Timestamp(d.iloc[-1]["ts"]).to_pydatetime()
    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=timezone.utc)

    row: Dict[str, Any] = {
        "stock": SYMBOL,
        "timeframe": TIMEFRAME_LG,
        "status": status,
        "as_of": as_of,
        "h2_time": now if status == "armed" else (prev_row or {}).get("h2_time"),
        "channel_start": None,
        "channel_end": None,
        "channel_span_days": None,
        "wait_bars": int(active.get("score") or 0),
        "wait_ok": bool(buy_ok),
        "support": float(levels["sma100"]),
        "resist": float(mode_target),
        "last_close": prior_close,
        "dist_to_resist_pct": dist_pct,
        "volume_rel_20": float(active.get("contracts") or 0) or None,
        "overshoot_prior": soft.get("gap_est"),
        "fill_px": float(open_px) if (status == "armed" and open_px) else None,
        "overshoot": gap_open if gap_open is not None else gap_live,
        "support_x0": None,
        # Persist inferred session open across minute ticks
        "support_y0": float(open_px) if open_px is not None else None,
        "support_slope": None,
        "channel_width": round(abs(GAP_STRAT1_PCT if mode != "strat2" else GAP_STRAT2_PCT), 4),
        "h2_idx": None,
        "as_of_i": None,
        "last_price": px_show,
        "last_price_ts": now,
        "dist_live_pct": dist_pct,
        "hot": bool(hot),
        "hot_notified_on": notify_tag,
        "_levels": levels,
        "_soft": soft,
        "_hard": hard,
        "_gap_live": gap_live,
        "_gap_open": gap_open,
        "_opened": opened,
        "_sticky": sticky,
    }
    return row, alerts


def candidate_db_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in row.items() if not str(k).startswith("_")}
