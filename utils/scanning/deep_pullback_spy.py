"""Deep Pullback SPY live state for /hot (timeframe=dp).

Rules (Quantified Strategies):
  Entry at close when all true:
    1) close = lowest close of last 15 sessions
    2) 1-day % return = lowest of last 10 sessions
    3) close > SMA200
  Exit (QS): close > prior day's high.

Live / RTH: Alpaca last stands in for today's developing close.
Approaching (early arm): above SMA200 and at least two of
  {near 15d low, sharp down-day, full cond1, full cond2}.

Telegram "approaching" only in the last MOC_WINDOW_MIN minutes of RTH, with an
explicit MOC target: if close finishes at/below moc_max (and above SMA floor),
send MOC buy.
"""
from __future__ import annotations

import logging
from datetime import datetime, time, timezone
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TIMEFRAME_DP = "dp"
SYMBOL = "SPY"
ET = ZoneInfo("America/New_York")

# Early-arm thresholds (fractional)
NEAR_LOW_PCT = 0.005  # within 0.5% of the rolling 15d min close
SHARP_RET_RANK = 3  # today's ret among worst N of last 10 (1=worst)

# Approaching Telegram only inside this many minutes before 16:00 ET
MOC_WINDOW_MIN = 20
RTH_CLOSE = time(16, 0)


def load_spy_daily(client: Any, *, lookback_bars: int = 400) -> pd.DataFrame:
    """Load SPY 1d bars (IB prefix + Alpaca preferred)."""
    q = """
    SELECT provider, ts, open, high, low, close, volume
    FROM market_data
    WHERE symbol = 'SPY' AND timeframe = '1d'
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
    """Calendar date in America/New_York for a bar timestamp."""
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return t.tz_convert(ET).date()


def minutes_to_rth_close(now: Optional[datetime] = None) -> Optional[float]:
    """Minutes until 16:00 ET on the current ET session day. None if not RTH afternoon."""
    now_et = (now or datetime.now(timezone.utc)).astimezone(ET)
    if now_et.weekday() >= 5:
        return None
    close_dt = datetime.combine(now_et.date(), RTH_CLOSE, tzinfo=ET)
    open_dt = datetime.combine(now_et.date(), time(9, 30), tzinfo=ET)
    if now_et < open_dt or now_et > close_dt:
        return None
    return (close_dt - now_et).total_seconds() / 60.0


def in_moc_alert_window(now: Optional[datetime] = None, *, window_min: int = MOC_WINDOW_MIN) -> bool:
    """True in the last `window_min` minutes of the RTH cash close (15:40-16:00 ET default)."""
    m = minutes_to_rth_close(now)
    if m is None:
        return False
    return 0.0 <= m <= float(window_min)


def moc_entry_target(completed_closes: np.ndarray) -> Dict[str, Any]:
    """Price caps for a full entry if today closes at that print.

    completed_closes: bars through yesterday only (no developing today).

    full signal iff:
      close <= low_cap   (lowest of last 15 including today)
      close <= ret_cap   (1d ret worst of last 10)
      close >  sma_floor (above SMA200; equiv. close > mean of prior 199)
    so MOC buy if close finishes in (sma_floor, moc_max] with moc_max = min(low_cap, ret_cap).
    """
    n = len(completed_closes)
    if n < 200:
        return {"ok": False, "reason": "need_200_bars"}
    prior_close = float(completed_closes[-1])
    low_cap = float(np.min(completed_closes[-14:]))
    prior_rets: List[float] = []
    for i in range(max(1, n - 9), n):
        pc = float(completed_closes[i - 1])
        if pc == 0:
            continue
        prior_rets.append(float(completed_closes[i]) / pc - 1.0)
    if not prior_rets:
        return {"ok": False, "reason": "need_rets"}
    min_prior_ret = float(min(prior_rets))
    ret_cap = prior_close * (1.0 + min_prior_ret)
    sma_floor = float(np.mean(completed_closes[-199:]))
    moc_max = min(low_cap, ret_cap)
    return {
        "ok": True,
        "prior_close": prior_close,
        "low_cap": round(low_cap, 4),
        "ret_cap": round(ret_cap, 4),
        "moc_max": round(moc_max, 4),
        "sma_floor": round(sma_floor, 4),
        "min_prior_ret": min_prior_ret,
        "viable": moc_max > sma_floor,
    }


def _features(closes: np.ndarray, highs: np.ndarray) -> Dict[str, Any]:
    """Features at the last bar of closes/highs (last may be developing)."""
    n = len(closes)
    if n < 200:
        return {"ok": False, "reason": "need_200_bars"}
    close = float(closes[-1])
    prior_close = float(closes[-2]) if n >= 2 else None
    ret1 = None if prior_close in (None, 0) else close / prior_close - 1.0
    sma200 = float(np.mean(closes[-200:]))
    win15 = closes[-15:]
    min15 = float(np.min(win15))
    rets = []
    for i in range(max(1, n - 10), n):
        pc = float(closes[i - 1])
        if pc == 0:
            continue
        rets.append(float(closes[i]) / pc - 1.0)
    min_ret10 = float(min(rets)) if rets else None
    prior_high = float(highs[-2]) if n >= 2 else None

    cond_sma = close > sma200
    cond_low = close <= min15 + 1e-12
    cond_ret = (
        ret1 is not None and min_ret10 is not None and ret1 <= min_ret10 + 1e-15
    )

    near_low = close <= min15 * (1.0 + NEAR_LOW_PCT)
    sharp = False
    if ret1 is not None and rets:
        order = sorted(rets)
        cutoff = order[min(SHARP_RET_RANK, len(order)) - 1]
        sharp = ret1 <= cutoff + 1e-15

    score = int(cond_sma)
    if cond_low or near_low:
        score += 1
    if cond_ret or sharp:
        score += 1

    return {
        "ok": True,
        "close": close,
        "prior_close": prior_close,
        "ret1": ret1,
        "sma200": sma200,
        "min15": min15,
        "min_ret10": min_ret10,
        "prior_high": prior_high,
        "cond_sma": cond_sma,
        "cond_low": cond_low,
        "cond_ret": cond_ret,
        "near_low": near_low,
        "sharp": sharp,
        "score": score,
        "full_signal": bool(cond_sma and cond_low and cond_ret),
        "approaching": bool(cond_sma and score >= 2 and not (cond_sma and cond_low and cond_ret)),
    }


def evaluate_live(
    daily: pd.DataFrame,
    *,
    last_price: Optional[float],
    prev_row: Optional[Dict[str, Any]] = None,
    now: Optional[datetime] = None,
    moc_window_min: int = MOC_WINDOW_MIN,
) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    """Build /hot candidate row + optional Telegram payloads.

    Returns (row, alerts) where alerts are dicts with keys kind, text.
    kind in {approach, arm, exit}.
    Approaching alerts only inside the last moc_window_min minutes before 16:00 ET,
    and include an MOC target price.
    """
    now = now or datetime.now(timezone.utc)
    now_et = now.astimezone(ET)
    today_et = now_et.date()
    alerts: List[Dict[str, str]] = []
    if daily is None or daily.empty or len(daily) < 200:
        return {}, alerts

    d = daily.copy()
    closes = d["close"].to_numpy(dtype=float)
    highs = d["high"].to_numpy(dtype=float)
    last_completed = d.iloc[-1]
    bar_day = session_date_et(last_completed["ts"])

    # Completed history through yesterday (for MOC target)
    if bar_day == today_et:
        completed = closes[:-1].copy()
        completed_highs = highs[:-1].copy()
    else:
        completed = closes.copy()
        completed_highs = highs.copy()

    target = moc_entry_target(completed)
    moc_window = in_moc_alert_window(now, window_min=moc_window_min)
    mins_left = minutes_to_rth_close(now)

    # Inject developing bar from Alpaca last when available (NY session date)
    developing = False
    if last_price is not None and float(last_price) > 0:
        if bar_day == today_et:
            closes = closes.copy()
            closes[-1] = float(last_price)
            developing = True
        else:
            closes = np.append(closes, float(last_price))
            highs = np.append(highs, float(last_price))
            developing = True

    feat = _features(closes, highs)
    if not feat.get("ok"):
        return {}, alerts

    prev_status = str((prev_row or {}).get("status") or "")
    prev_fill = (prev_row or {}).get("fill_px")
    prev_notify = str((prev_row or {}).get("hot_notified_on") or "")
    in_position = prev_status in ("filled", "in_position", "exit_signal") and prev_fill is not None

    status = "idle"
    hot = False
    notify_tag = prev_notify

    if in_position:
        ph = feat.get("prior_high")
        if ph is not None and feat["close"] > float(ph):
            status = "exit_signal"
            if prev_status != "exit_signal":
                alerts.append(
                    {
                        "kind": "exit",
                        "text": (
                            "Deep Pullback SPY EXIT (QS): last %.2f > prior high %.2f "
                            "(entry %.2f). Sell MOC / near close."
                            % (feat["close"], float(ph), float(prev_fill))
                        ),
                    }
                )
        else:
            status = "filled"
    elif feat["full_signal"]:
        status = "armed"
        hot = True
        arm_key = "%s:arm" % today_et.isoformat()
        if moc_window and prev_notify != arm_key:
            tgt = ""
            if target.get("ok"):
                tgt = " Target was close<=%.2f (above SMA %.2f)." % (
                    float(target["moc_max"]),
                    float(target["sma_floor"]),
                )
            alerts.append(
                {
                    "kind": "arm",
                    "text": (
                        "Deep Pullback SPY ARMED — send MOC buy now. "
                        "15d low + worst 1d/10 + above SMA200. "
                        "last=%.2f sma200=%.2f ret1=%.2f%%%s"
                        % (
                            feat["close"],
                            feat["sma200"],
                            100.0 * float(feat["ret1"] or 0.0),
                            tgt,
                        )
                    ),
                }
            )
            notify_tag = arm_key
    elif feat["approaching"]:
        status = "waiting"
        approach_key = "%s:approach" % today_et.isoformat()
        if moc_window and target.get("ok") and target.get("viable") and prev_notify != approach_key:
            mins_txt = "?" if mins_left is None else ("%.0f" % mins_left)
            alerts.append(
                {
                    "kind": "approach",
                    "text": (
                        "Deep Pullback SPY approaching (%sm to close). "
                        "If SPY closes at or below %.2f (and still above SMA200 %.2f), "
                        "send MOC buy. "
                        "Now last=%.2f (need <=%.2f for 15d-low, <=%.2f for worst-1d). "
                        "score=%d/3."
                        % (
                            mins_txt,
                            float(target["moc_max"]),
                            float(target["sma_floor"]),
                            feat["close"],
                            float(target["low_cap"]),
                            float(target["ret_cap"]),
                            int(feat["score"]),
                        )
                    ),
                }
            )
            notify_tag = approach_key

    if status == "idle":
        return {}, alerts

    min15 = float(feat["min15"])
    dist = None if min15 == 0 else (feat["close"] / min15 - 1.0) * 100.0
    fill_px = float(prev_fill) if in_position and prev_fill is not None else None

    as_of = pd.Timestamp(d.iloc[-1]["ts"]).to_pydatetime()
    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=timezone.utc)

    # Surface MOC target on the row (resist stays 15d low; overshoot_prior = moc_max)
    moc_max = target.get("moc_max") if target.get("ok") else None
    sma_floor = target.get("sma_floor") if target.get("ok") else None

    row: Dict[str, Any] = {
        "stock": SYMBOL,
        "timeframe": TIMEFRAME_DP,
        "status": status,
        "as_of": as_of,
        "h2_time": (prev_row or {}).get("h2_time") if in_position else None,
        "channel_start": None,
        "channel_end": None,
        "channel_span_days": None,
        "wait_bars": int(feat["score"]),
        "wait_ok": bool(feat["full_signal"] or feat["approaching"]),
        "support": float(sma_floor if sma_floor is not None else feat["sma200"]),
        "resist": float(moc_max if moc_max is not None else feat["min15"]),
        "last_close": float(feat.get("prior_close") or feat["close"]),
        "dist_to_resist_pct": None
        if moc_max in (None, 0)
        else round((feat["close"] / float(moc_max) - 1.0) * 100.0, 4),
        "volume_rel_20": None,
        "overshoot_prior": None
        if feat.get("ret1") is None
        else round(100.0 * float(feat["ret1"]), 4),
        "fill_px": fill_px,
        "overshoot": None
        if feat.get("prior_high") is None
        else round(float(feat["prior_high"]), 4),
        "support_x0": None,
        "support_y0": float(feat["prior_high"]) if feat.get("prior_high") is not None else None,
        "support_slope": None,
        "channel_width": None
        if feat["sma200"] == 0
        else round(abs(feat["close"] / feat["sma200"] - 1.0) * 100.0, 4),
        "h2_idx": None,
        "as_of_i": None,
        "last_price": float(last_price) if last_price is not None else float(feat["close"]),
        "last_price_ts": now,
        "dist_live_pct": None
        if moc_max in (None, 0)
        else round((feat["close"] / float(moc_max) - 1.0) * 100.0, 4),
        "hot": bool(hot),
        "hot_notified_on": notify_tag,
        "_developing": developing,
        "_full_signal": feat["full_signal"],
        "_feat": feat,
        "_target": target,
        "_moc_window": moc_window,
        "_mins_to_close": mins_left,
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
    """Strip private keys before upsert."""
    return {k: v for k, v in row.items() if not str(k).startswith("_")}
