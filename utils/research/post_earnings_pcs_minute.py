"""Minute-bar open / first-hour marks for post-earnings PCS (Polygon Starter).

Uses 1-minute option aggregates for entry clocks (first RTH print, ~10:00 ET,
first-hour VWAP), then re-walks TP50 / SL 2x on subsequent day closes.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Any, Dict, List, Optional, Sequence
from zoneinfo import ZoneInfo

import pandas as pd

from utils.research.post_earnings_pcs_live import (
    align_leg_bars,
    book_stats_from_live,
    credit_vertical_mids,
    walk_vertical_tp_sl,
)
from utils.research.post_earnings_pcs_open_delta import MIN_CREDIT_FRAC
from utils.research.short_put_polygon_spotcheck import bars_from_aggs_response

ET = ZoneInfo("America/New_York")
RTH_OPEN = time(9, 30)
FIRST_HOUR_END = time(10, 30)
MARK_10 = time(10, 0)


@dataclass
class MinuteMark:
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


def minute_bars_from_aggs(resp: Dict[str, Any]) -> pd.DataFrame:
    """Polygon minute aggs -> DataFrame with ET timestamps."""
    rows = (resp or {}).get("results") or []
    out: List[Dict[str, Any]] = []
    for b in rows:
        ts_ms = b.get("t")
        if ts_ms is None:
            continue
        ts = datetime.fromtimestamp(int(ts_ms) / 1000.0, tz=ET)
        out.append(
            {
                "ts": ts,
                "date": ts.date(),
                "open": float(b.get("o") or 0),
                "high": float(b.get("h") or 0),
                "low": float(b.get("l") or 0),
                "close": float(b.get("c") or 0),
                "volume": float(b.get("v") or 0),
            }
        )
    if not out:
        return pd.DataFrame(
            columns=["ts", "date", "open", "high", "low", "close", "volume"]
        )
    return pd.DataFrame(out).sort_values("ts").reset_index(drop=True)


def _rth_slice(df: pd.DataFrame, day: date) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    hit = df[df["date"] == day].copy()
    if hit.empty:
        return hit
    hit = hit[hit["ts"].apply(lambda t: t.timetz().replace(tzinfo=None) >= RTH_OPEN)]
    return hit.reset_index(drop=True)


def first_rth_mark(df: pd.DataFrame, day: date) -> Optional[MinuteMark]:
    hit = _rth_slice(df, day)
    if hit.empty:
        return None
    r = hit.iloc[0]
    return MinuteMark(
        ts=r["ts"],
        open=float(r["open"]),
        high=float(r["high"]),
        low=float(r["low"]),
        close=float(r["close"]),
        volume=float(r["volume"]),
    )


def mark_at_or_after(
    df: pd.DataFrame,
    day: date,
    clock: time,
) -> Optional[MinuteMark]:
    hit = _rth_slice(df, day)
    if hit.empty:
        return None
    for _, r in hit.iterrows():
        t = r["ts"].timetz().replace(tzinfo=None)
        if t >= clock:
            return MinuteMark(
                ts=r["ts"],
                open=float(r["open"]),
                high=float(r["high"]),
                low=float(r["low"]),
                close=float(r["close"]),
                volume=float(r["volume"]),
            )
    return None


def first_hour_vwap(df: pd.DataFrame, day: date) -> Optional[float]:
    """Volume-weighted average of minute closes 09:30-10:30 ET (inclusive open)."""
    hit = _rth_slice(df, day)
    if hit.empty:
        return None
    win = hit[
        hit["ts"].apply(
            lambda t: RTH_OPEN
            <= t.timetz().replace(tzinfo=None)
            <= FIRST_HOUR_END
        )
    ]
    if win.empty:
        return None
    vol = win["volume"].astype(float)
    if float(vol.sum()) <= 0:
        return float(win["close"].mean())
    px = win["close"].astype(float)
    return float((px * vol).sum() / vol.sum())


def vertical_credit_from_marks(
    short_px: Optional[float],
    long_px: Optional[float],
) -> Optional[float]:
    if short_px is None or long_px is None:
        return None
    return credit_vertical_mids(float(short_px), float(long_px), haircut=False)


def evaluate_minute_vertical(
    *,
    symbol: str,
    entry_date: date,
    short_ticker: str,
    long_ticker: str,
    width: float,
    expiry: date,
    fetch_minute,  # callable(ticker, day_iso) -> dict aggs
    fetch_day,  # callable(ticker, start, end) -> DataFrame day bars
    entry_modes: Sequence[str] = ("rth_open", "t1000", "first_hour_vwap", "day_close"),
    haircut: bool = False,
    spread_floor: float = 0.05,
    spread_frac: float = 0.08,
) -> Dict[str, Any]:
    """Mark vertical at minute clocks; walk TP/SL from each entry mode.

    Default haircut=False: minute aggs are trade prints, not NBBO mids, so the
    day-bar 8% half-spread proxy double-counts and can zero the entry credit.
    """
    out: Dict[str, Any] = {
        "symbol": symbol,
        "entry_date": entry_date.isoformat(),
        "short_ticker": short_ticker,
        "long_ticker": long_ticker,
        "width": float(width),
        "expiry": expiry.isoformat(),
        "haircut": bool(haircut),
        "status": "pending",
    }
    day_s = entry_date.isoformat()
    try:
        s_min = minute_bars_from_aggs(fetch_minute(short_ticker, day_s))
        l_min = minute_bars_from_aggs(fetch_minute(long_ticker, day_s))
    except Exception as exc:  # noqa: BLE001
        out["status"] = "minute_fetch_err"
        out["error"] = str(exc)
        return out

    out["n_short_min"] = int(len(s_min))
    out["n_long_min"] = int(len(l_min))
    if s_min.empty or l_min.empty:
        out["status"] = "no_minute_bars"
        return out

    s_rth = first_rth_mark(s_min, entry_date)
    l_rth = first_rth_mark(l_min, entry_date)
    s_10 = mark_at_or_after(s_min, entry_date, MARK_10)
    l_10 = mark_at_or_after(l_min, entry_date, MARK_10)
    s_fh = first_hour_vwap(s_min, entry_date)
    l_fh = first_hour_vwap(l_min, entry_date)

    credits = {
        "rth_open": vertical_credit_from_marks(
            None if s_rth is None else s_rth.open,
            None if l_rth is None else l_rth.open,
        ),
        "t1000": vertical_credit_from_marks(
            None if s_10 is None else s_10.close,
            None if l_10 is None else l_10.close,
        ),
        "first_hour_vwap": vertical_credit_from_marks(s_fh, l_fh),
    }
    out["credit_rth_open"] = credits["rth_open"]
    out["credit_t1000"] = credits["t1000"]
    out["credit_first_hour_vwap"] = credits["first_hour_vwap"]
    if s_rth is not None:
        out["short_rth_ts"] = s_rth.ts.isoformat()
        out["short_rth_open"] = s_rth.open
    if l_rth is not None:
        out["long_rth_open"] = l_rth.open

    min_credit = MIN_CREDIT_FRAC * max(float(width), 1e-9)
    out["min_credit"] = min_credit
    for name, cred in credits.items():
        out["meets_min_%s" % name] = (
            bool(cred is not None and cred >= min_credit - 1e-9)
        )

    # Day bars for exit walk (+ entry day close for day_close mode)
    start = (entry_date - timedelta(days=1)).isoformat()
    end = (expiry + timedelta(days=2)).isoformat()
    try:
        s_day = fetch_day(short_ticker, start, end)
        l_day = fetch_day(long_ticker, start, end)
    except Exception as exc:  # noqa: BLE001
        out["status"] = "day_fetch_err"
        out["error"] = str(exc)
        return out
    aligned = align_leg_bars(s_day, l_day)
    out["n_aligned_days"] = int(len(aligned))
    if aligned.empty:
        out["status"] = "no_day_bars"
        return out

    # day_close credit from aligned entry bar
    entry_rows = aligned[aligned["date"] == entry_date]
    if not entry_rows.empty:
        er = entry_rows.iloc[0]
        credits["day_close"] = credit_vertical_mids(
            float(er["short_close"]), float(er["long_close"]), haircut=False
        )
        out["credit_day_close"] = credits["day_close"]
        out["meets_min_day_close"] = bool(
            credits["day_close"] is not None
            and credits["day_close"] >= min_credit - 1e-9
        )

    mark_pairs = {
        "rth_open": (
            None if s_rth is None else s_rth.open,
            None if l_rth is None else l_rth.open,
        ),
        "t1000": (
            None if s_10 is None else s_10.close,
            None if l_10 is None else l_10.close,
        ),
        "first_hour_vwap": (s_fh, l_fh),
        "day_close": (None, None),  # use bar closes
    }

    walks: Dict[str, Any] = {}
    hs_kw = {
        "spread_floor": float(spread_floor) if haircut else 0.0,
        "spread_frac": float(spread_frac) if haircut else 0.0,
    }
    for mode in entry_modes:
        if mode not in mark_pairs and mode != "day_close":
            continue
        short_m, long_m = mark_pairs.get(mode, (None, None))
        if mode == "day_close":
            wr = walk_vertical_tp_sl(
                aligned,
                entry_date=entry_date,
                width=float(width),
                expiry=expiry,
                include_entry_day_exit=False,
                **hs_kw,
            )
        else:
            if short_m is None or long_m is None:
                walks[mode] = {"status": "no_mark"}
                continue
            wr = walk_vertical_tp_sl(
                aligned,
                entry_date=entry_date,
                width=float(width),
                expiry=expiry,
                entry_short_mid=float(short_m),
                entry_long_mid=float(long_m),
                include_entry_day_exit=True,
                **hs_kw,
            )
        walks[mode] = {
            "status": wr.status,
            "credit_mid": wr.credit_mid,
            "credit_hs": wr.credit_hs,
            "exit_reason": wr.exit_reason,
            "exit_date": wr.exit_date.isoformat() if wr.exit_date else None,
            "hold_days": wr.hold_days,
            "pnl_mid": wr.pnl_mid,
            "pnl_hs": wr.pnl_hs,
            "ror_mid": wr.ror_mid,
            "ror_hs": wr.ror_hs,
        }
        out["walk_%s_status" % mode] = wr.status
        out["walk_%s_ror_hs" % mode] = wr.ror_hs
        out["walk_%s_exit" % mode] = wr.exit_reason
        out["walk_%s_hold" % mode] = wr.hold_days

    out["walks"] = walks
    out["status"] = "ok"
    return out


def flatten_walk_rows(
    rows: Sequence[Dict[str, Any]],
    mode: str,
) -> List[Dict[str, Any]]:
    """Rows shaped for book_stats_from_live from evaluate_minute_vertical."""
    out: List[Dict[str, Any]] = []
    for r in rows:
        w = (r.get("walks") or {}).get(mode) or {}
        if w.get("status") != "ok":
            continue
        out.append(
            {
                "symbol": r.get("symbol"),
                "entry_date": r.get("entry_date"),
                "ror_hs": w.get("ror_hs"),
                "hold_days": w.get("hold_days"),
                "exit_reason": w.get("exit_reason"),
            }
        )
    return out


def summarize_minute_pilot(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    df = pd.DataFrame(list(rows))
    summary: Dict[str, Any] = {
        "attempted": len(rows),
        "status_counts": (
            df["status"].value_counts().to_dict() if not df.empty else {}
        ),
    }
    ok = [r for r in rows if r.get("status") == "ok"]
    summary["ok_n"] = len(ok)
    for col in ("rth_open", "t1000", "first_hour_vwap", "day_close"):
        key = "credit_%s" % col
        s = pd.Series([r.get(key) for r in ok], dtype=float).dropna()
        meet = pd.Series(
            [bool(r.get("meets_min_%s" % col)) for r in ok if r.get(key) is not None]
        )
        summary[key] = {
            "n": int(len(s)),
            "median": float(s.median()) if len(s) else None,
            "mean": float(s.mean()) if len(s) else None,
            "meets_min_pct": 100.0 * float(meet.mean()) if len(meet) else None,
        }
        flat = flatten_walk_rows(ok, col)
        summary["book_%s" % col] = book_stats_from_live(flat, ror_key="ror_hs")
    # open vs close credit
    pairs = [
        (r.get("credit_rth_open"), r.get("credit_day_close"))
        for r in ok
        if r.get("credit_rth_open") is not None and r.get("credit_day_close") is not None
    ]
    if pairs:
        o_gt_c = sum(1 for o, c in pairs if o > c)
        summary["rth_open_gt_day_close_pct"] = 100.0 * o_gt_c / len(pairs)
        ratios = [o / c for o, c in pairs if c and c > 1e-9]
        summary["rth_open_vs_day_close_med"] = (
            float(pd.Series(ratios).median()) if ratios else None
        )
    return summary
