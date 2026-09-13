"""Causal joins between Deepvue weekend focus and last-15m H2 fills.

Deepvue signal_date is the completed week-end session (usually Friday).
The live buy week is the next RTH sessions (typically Mon-Fri). Last-15m
AND uses that trade week, not the Friday signal bar itself.

Occupancy is not re-walked. This is not a promote.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Dict, List, Mapping, Optional, Sequence

import pandas as pd

from utils.research.last_15m_volume_geometry import fill_day_et

TRADE_WEEK_MAX_DAYS = 7


def as_date(val: object) -> Optional[date]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, date) and not isinstance(val, pd.Timestamp):
        return val
    try:
        ts = pd.Timestamp(val)
    except (TypeError, ValueError):
        return None
    if pd.isna(ts):
        return None
    return ts.date()


def in_focus_trade_week(fill_day: Optional[date], signal_day: Optional[date]) -> bool:
    """True when fill is after the weekend signal and within the next 7 calendar days."""
    if fill_day is None or signal_day is None:
        return False
    delta = (fill_day - signal_day).days
    return 1 <= delta <= int(TRADE_WEEK_MAX_DAYS)


def signal_dates_by_stock(signals: pd.DataFrame) -> Dict[str, List[date]]:
    out: Dict[str, List[date]] = {}
    if signals is None or signals.empty:
        return out
    stocks = signals["stock"].astype(str).str.upper()
    for stock, raw in zip(stocks, signals["signal_date"]):
        d = as_date(raw)
        if d is None:
            continue
        out.setdefault(str(stock), []).append(d)
    for key, days in out.items():
        out[key] = sorted(set(days))
    return out


def fill_dates_by_stock(trades: pd.DataFrame) -> Dict[str, List[date]]:
    out: Dict[str, List[date]] = {}
    if trades is None or trades.empty:
        return out
    for _, row in trades.iterrows():
        stock = str(row.get("stock") or "").upper()
        d = fill_day_et(row)
        if not stock or d is None:
            continue
        out.setdefault(stock, []).append(d)
    for key, days in out.items():
        out[key] = sorted(set(days))
    return out


def _any_match(
    stock: str,
    fill: Optional[date],
    by_stock: Mapping[str, Sequence[date]],
    predicate,
) -> bool:
    if fill is None:
        return False
    for sig in by_stock.get(str(stock).upper(), ()):
        if predicate(fill, sig):
            return True
    return False


def keep_last15m_in_focus_trade_week(
    trades: pd.DataFrame,
    signals: pd.DataFrame,
) -> pd.Series:
    """Last-15m fill during the week after a same-name Deepvue weekend focus row."""
    by_stock = signal_dates_by_stock(signals)
    flags: List[bool] = []
    for _, row in trades.iterrows():
        flags.append(
            _any_match(
                str(row.get("stock") or ""),
                fill_day_et(row),
                by_stock,
                in_focus_trade_week,
            )
        )
    return pd.Series(flags, index=trades.index, dtype=bool)


def keep_last15m_on_signal_day(
    trades: pd.DataFrame,
    signals: pd.DataFrame,
) -> pd.Series:
    """Last-15m fill on the same calendar day as the Deepvue weekend signal."""
    by_stock = signal_dates_by_stock(signals)

    def same_day(fill: Optional[date], signal: Optional[date]) -> bool:
        return fill is not None and signal is not None and fill == signal

    flags: List[bool] = []
    for _, row in trades.iterrows():
        flags.append(
            _any_match(
                str(row.get("stock") or ""),
                fill_day_et(row),
                by_stock,
                same_day,
            )
        )
    return pd.Series(flags, index=trades.index, dtype=bool)


def keep_last15m_ever_focused(
    trades: pd.DataFrame,
    signals: pd.DataFrame,
) -> pd.Series:
    """Universe overlap: name ever appeared on the Deepvue take-4 list (not a week AND)."""
    names = set(signal_dates_by_stock(signals).keys())
    stocks = trades["stock"].astype(str).str.upper()
    return stocks.isin(names)


def keep_deepvue_if_last15m_trade_week(
    deepvue: pd.DataFrame,
    last15m: pd.DataFrame,
) -> pd.Series:
    """Deepvue row whose name also printed a last-15m fill in that trade week."""
    by_stock = fill_dates_by_stock(last15m)
    flags: List[bool] = []
    for _, row in deepvue.iterrows():
        stock = str(row.get("stock") or "").upper()
        signal = as_date(row.get("signal_date"))
        hit = False
        for fill in by_stock.get(stock, ()):
            if in_focus_trade_week(fill, signal):
                hit = True
                break
        flags.append(hit)
    return pd.Series(flags, index=deepvue.index, dtype=bool)


def overlap_counts(
    last15m: pd.DataFrame,
    signals: pd.DataFrame,
    deepvue: Optional[pd.DataFrame] = None,
) -> dict:
    l15_week = keep_last15m_in_focus_trade_week(last15m, signals)
    l15_day = keep_last15m_on_signal_day(last15m, signals)
    l15_ever = keep_last15m_ever_focused(last15m, signals)
    sig_n = int(len(signals)) if signals is not None and not signals.empty else 0
    dv = deepvue if deepvue is not None else pd.DataFrame()
    dv_week = (
        keep_deepvue_if_last15m_trade_week(dv, last15m)
        if not dv.empty
        else pd.Series(dtype=bool)
    )
    l15_names = set(last15m["stock"].astype(str).str.upper()) if not last15m.empty else set()
    sg_names = set(signal_dates_by_stock(signals).keys())
    return {
        "n_last15m": int(len(last15m)),
        "n_signals": sig_n,
        "n_deepvue": int(len(dv)),
        "n_l15_and_week": int(l15_week.sum()),
        "n_l15_and_signal_day": int(l15_day.sum()),
        "n_l15_ever_focused": int(l15_ever.sum()),
        "n_signals_with_l15_week": int(dv_week.sum()) if len(dv_week) else 0,
        "n_stock_overlap": len(l15_names & sg_names),
        "n_l15_stocks": len(l15_names),
        "n_signal_stocks": len(sg_names),
    }


def pair_rows(
    last15m: pd.DataFrame,
    signals: pd.DataFrame,
    *,
    mode: str = "trade_week",
) -> pd.DataFrame:
    """Matched last-15m rows with the Deepvue signal that justified the keep."""
    by_stock = signal_dates_by_stock(signals)
    rows: List[dict] = []
    for _, row in last15m.iterrows():
        stock = str(row.get("stock") or "").upper()
        fill = fill_day_et(row)
        matched: Optional[date] = None
        for sig in by_stock.get(stock, ()):
            if mode == "signal_day" and fill is not None and fill == sig:
                matched = sig
                break
            if mode == "trade_week" and in_focus_trade_week(fill, sig):
                matched = sig
                break
        if matched is None:
            continue
        rec = {
            "stock": stock,
            "fill_day": str(fill),
            "signal_date": str(matched),
            "gain_pct": row.get("gain_pct"),
            "buy_date": row.get("buy_date"),
            "exit_reason": row.get("exit_reason"),
        }
        rows.append(rec)
    return pd.DataFrame(rows)
