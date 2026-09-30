"""Polygon listed put-credit-spread realism for post-earnings PCS claim-check.

Fetches day bars for short + long puts, marks the vertical mid, applies
leg half-spreads, and re-walks TP50 / SL 2x on live marks.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from utils.research.short_put_live_realism import (
    half_spread,
    nearby_fridays,
    round_option_strike,
)
from utils.research.short_put_polygon_spotcheck import (
    bars_from_aggs_response,
    polygon_option_ticker,
)


def credit_vertical_mids(
    short_mid: float,
    long_mid: float,
    *,
    floor: float = 0.05,
    frac: float = 0.08,
    haircut: bool = True,
) -> float:
    """Sell short / buy long: net credit from day-bar mids."""
    s = float(short_mid)
    l = float(long_mid)
    if not haircut:
        return max(0.0, s - l)
    # Sell short at mid - hs; buy long at mid + hs
    sell = max(0.0, s - half_spread(s, floor=floor, frac=frac))
    buy = l + half_spread(l, floor=floor, frac=frac)
    return max(0.0, sell - buy)


def debit_vertical_mids(
    short_mid: float,
    long_mid: float,
    *,
    floor: float = 0.05,
    frac: float = 0.08,
    haircut: bool = True,
) -> float:
    """Buy short / sell long: net debit to close."""
    s = float(short_mid)
    l = float(long_mid)
    if not haircut:
        return max(0.0, s - l)
    buy = s + half_spread(s, floor=floor, frac=frac)
    sell = max(0.0, l - half_spread(l, floor=floor, frac=frac))
    return max(0.0, buy - sell)


def align_leg_bars(
    short_bars: pd.DataFrame,
    long_bars: pd.DataFrame,
) -> pd.DataFrame:
    """Inner-join short/long day bars on date; columns short_*/long_*."""
    if short_bars is None or short_bars.empty or long_bars is None or long_bars.empty:
        return pd.DataFrame()
    s = short_bars.copy()
    l = long_bars.copy()
    s["date"] = pd.to_datetime(s["date"]).dt.date
    l["date"] = pd.to_datetime(l["date"]).dt.date
    s = s.rename(
        columns={
            "open": "short_open",
            "high": "short_high",
            "low": "short_low",
            "close": "short_close",
            "volume": "short_volume",
        }
    )
    l = l.rename(
        columns={
            "open": "long_open",
            "high": "long_high",
            "low": "long_low",
            "close": "long_close",
            "volume": "long_volume",
        }
    )
    m = s.merge(l, on="date", how="inner")
    return m.sort_values("date").reset_index(drop=True)


@dataclass
class VerticalWalkResult:
    status: str
    entry_date: Optional[date] = None
    exit_date: Optional[date] = None
    credit_mid: Optional[float] = None
    credit_hs: Optional[float] = None
    exit_debit_mid: Optional[float] = None
    exit_debit_hs: Optional[float] = None
    exit_reason: Optional[str] = None
    hold_days: Optional[int] = None
    pnl_mid: Optional[float] = None
    pnl_hs: Optional[float] = None
    ror_mid: Optional[float] = None
    ror_hs: Optional[float] = None
    max_risk_hs: Optional[float] = None
    n_bars: int = 0


def walk_vertical_tp_sl(
    aligned: pd.DataFrame,
    *,
    entry_date: date,
    width: float,
    expiry: date,
    tp_frac: float = 0.50,
    sl_mult: float = 2.0,
    spread_floor: float = 0.05,
    spread_frac: float = 0.08,
    entry_slack_days: int = 0,
    entry_short_mid: Optional[float] = None,
    entry_long_mid: Optional[float] = None,
    include_entry_day_exit: bool = False,
) -> VerticalWalkResult:
    """Re-walk TP50 / SL 2x on aligned short/long day closes.

    Entry = first bar on/after entry_date within slack.
    Marks use close mids; half-spread applied at entry credit and exit debit.
    Optional entry_short_mid/entry_long_mid override the entry-day closes
    (e.g. first RTH / first-hour minute marks). When include_entry_day_exit,
    the entry session's day close is eligible for TP/SL (intraday entry).
    """
    if aligned is None or aligned.empty:
        return VerticalWalkResult(status="no_bars")
    entry_end = entry_date + timedelta(days=int(entry_slack_days))
    cands = aligned[
        (aligned["date"] >= entry_date) & (aligned["date"] <= entry_end)
    ]
    if cands.empty:
        # allow first print after entry within 3 calendar days
        cands = aligned[
            (aligned["date"] >= entry_date)
            & (aligned["date"] <= entry_date + timedelta(days=3))
        ]
        if cands.empty:
            return VerticalWalkResult(status="no_entry_bar")
    entry = cands.iloc[0]
    ed = entry["date"]
    if isinstance(ed, pd.Timestamp):
        ed = ed.date()

    short_e = (
        float(entry_short_mid)
        if entry_short_mid is not None
        else float(entry["short_close"])
    )
    long_e = (
        float(entry_long_mid)
        if entry_long_mid is not None
        else float(entry["long_close"])
    )
    credit_mid = credit_vertical_mids(short_e, long_e, haircut=False)
    credit_hs = credit_vertical_mids(
        short_e,
        long_e,
        floor=spread_floor,
        frac=spread_frac,
        haircut=True,
    )
    if credit_hs <= 1e-8:
        return VerticalWalkResult(
            status="no_credit",
            entry_date=ed,
            credit_mid=credit_mid,
            credit_hs=credit_hs,
            n_bars=int(len(aligned)),
        )

    width_f = float(width)
    max_risk = max(1e-9, width_f - credit_hs)
    tp_debit = float(tp_frac) * credit_hs
    sl_debit = min(float(sl_mult) * credit_hs, width_f)

    if include_entry_day_exit:
        after = aligned[aligned["date"] >= ed]
    else:
        after = aligned[aligned["date"] > ed]
    after = after[after["date"] <= expiry]
    exit_reason = None
    exit_day = None
    exit_mid = None
    exit_hs = None

    for _, row in after.iterrows():
        day = row["date"]
        if isinstance(day, pd.Timestamp):
            day = day.date()
        # Debit to close (buy vertical)
        deb_mid = debit_vertical_mids(
            float(row["short_close"]),
            float(row["long_close"]),
            haircut=False,
        )
        deb_hs = debit_vertical_mids(
            float(row["short_close"]),
            float(row["long_close"]),
            floor=spread_floor,
            frac=spread_frac,
            haircut=True,
        )
        deb_hs = min(deb_hs, width_f)
        deb_mid = min(deb_mid, width_f)

        if day >= expiry:
            exit_reason = "expiry"
            exit_day = day
            exit_mid = deb_mid
            exit_hs = deb_hs
            break
        if deb_hs <= tp_debit + 1e-12:
            exit_reason = "tp50"
            exit_day = day
            exit_mid = deb_mid
            exit_hs = tp_debit  # fill at target
            break
        if deb_hs >= sl_debit - 1e-12:
            exit_reason = "sl_2x"
            exit_day = day
            exit_mid = deb_mid
            exit_hs = sl_debit
            break

    if exit_reason is None:
        return VerticalWalkResult(
            status="no_exit",
            entry_date=ed,
            credit_mid=credit_mid,
            credit_hs=credit_hs,
            n_bars=int(len(aligned)),
        )

    pnl_mid = float(credit_mid) - float(exit_mid)
    pnl_hs = float(credit_hs) - float(exit_hs)
    ror_mid = 100.0 * pnl_mid / max(1e-9, width_f - credit_mid)
    ror_hs = 100.0 * pnl_hs / max_risk
    return VerticalWalkResult(
        status="ok",
        entry_date=ed,
        exit_date=exit_day,
        credit_mid=credit_mid,
        credit_hs=credit_hs,
        exit_debit_mid=float(exit_mid),
        exit_debit_hs=float(exit_hs),
        exit_reason=exit_reason,
        hold_days=(exit_day - ed).days if exit_day else None,
        pnl_mid=pnl_mid,
        pnl_hs=pnl_hs,
        ror_mid=ror_mid,
        ror_hs=ror_hs,
        max_risk_hs=max_risk,
        n_bars=int(len(aligned)),
    )


def select_recent_pcs_sample(
    trades: pd.DataFrame,
    *,
    n: int = 40,
    min_entry: str = "2024-10-01",
) -> pd.DataFrame:
    """Recent PCS trades, prefer diverse symbols then newest."""
    df = trades.copy()
    df["entry_date"] = pd.to_datetime(df["entry_date"], errors="coerce")
    df = df[df["entry_date"] >= pd.Timestamp(min_entry)].copy()
    df = df.dropna(subset=["symbol", "entry_date", "short_k", "long_k", "width", "dte"])
    if df.empty:
        return df
    df = df.sort_values(["entry_date"], ascending=False)
    picked = df.drop_duplicates("symbol", keep="first")
    if len(picked) < int(n):
        rest = df[~df.index.isin(picked.index)]
        picked = pd.concat([picked, rest.head(int(n) - len(picked))], axis=0)
    return picked.head(int(n)).sort_values("entry_date").reset_index(drop=True)


def resolve_vertical_tickers(
    symbol: str,
    entry_date: date,
    short_k: float,
    long_k: float,
    dte: int,
) -> Tuple[str, str, str, float, float]:
    """Nearest Friday expiry + rounded strikes -> OCC tickers."""
    target = entry_date + timedelta(days=int(dte))
    fridays = nearby_fridays(target, window_days=21)
    expiry = fridays[0].isoformat() if fridays else target.isoformat()
    sk = round_option_strike(float(short_k))
    lk = round_option_strike(float(long_k))
    if lk >= sk:
        # keep width at least one step
        step = sk - round_option_strike(sk - 1) if sk > 1 else 1.0
        # recompute step from short
        p = abs(sk)
        if p < 25:
            step = 0.5
        elif p < 100:
            step = 1.0
        elif p < 200:
            step = 2.5
        else:
            step = 5.0
        lk = round(sk - max(step, float(short_k) - float(long_k)), 4)
    short_t = polygon_option_ticker(symbol, expiry, sk, "P")
    long_t = polygon_option_ticker(symbol, expiry, lk, "P")
    return short_t, long_t, expiry, sk, lk


def book_stats_from_live(rows: Sequence[Dict[str, Any]], *, ror_key: str = "ror_hs") -> Dict[str, Any]:
    if not rows:
        return {"n": 0}
    r = pd.Series([float(x[ror_key]) for x in rows if x.get(ror_key) is not None])
    holds = pd.Series(
        [float(x["hold_days"]) for x in rows if x.get("hold_days") is not None]
    )
    reasons = pd.Series([str(x.get("exit_reason") or "") for x in rows])
    wins = r[r > 0]
    losses = r[r <= 0]
    gp = float(wins.sum()) if len(wins) else 0.0
    gl = float((-losses).sum()) if len(losses) else 0.0
    pf = (gp / gl) if gl > 1e-12 else (float("inf") if gp > 0 else None)
    return {
        "n": int(len(rows)),
        "win_rate_pct": 100.0 * float((r > 0).mean()) if len(r) else None,
        "avg_hold_days": float(holds.mean()) if len(holds) else None,
        "avg_ror_pct": float(r.mean()) if len(r) else None,
        "median_ror_pct": float(r.median()) if len(r) else None,
        "profit_factor": pf,
        "tp_pct": 100.0 * float((reasons == "tp50").mean()) if len(reasons) else None,
        "sl_pct": 100.0 * float((reasons == "sl_2x").mean()) if len(reasons) else None,
        "expiry_pct": 100.0 * float((reasons == "expiry").mean()) if len(reasons) else None,
    }
