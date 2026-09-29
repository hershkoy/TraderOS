"""Live-realism helpers for short-put research (listed contract + half-spread).

Uses Polygon day bars as mid proxies; half-spread haircut approximates
buying the ask / selling the bid without quote snapshots.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from utils.research.short_put_polygon_spotcheck import (
    closest_listed_expiry,
    polygon_option_ticker,
)


def nearby_fridays(target: date, *, window_days: int = 21) -> List[date]:
    """Fridays within +/- window of target, nearest first (weekly/monthly listed)."""
    fridays: List[date] = []
    for delta in range(-int(window_days), int(window_days) + 1):
        d = target + timedelta(days=delta)
        if d.weekday() == 4:
            fridays.append(d)
    fridays.sort(key=lambda d: (abs((d - target).days), d))
    return fridays


def half_spread(
    mid: float,
    *,
    floor: float = 0.05,
    frac: float = 0.08,
) -> float:
    """Half-spread in dollars: max(floor, frac * mid)."""
    m = float(mid)
    if m <= 0 or m != m:
        return float(floor)
    return max(float(floor), float(frac) * m)


def credit_after_spread(mid: float, *, floor: float = 0.05, frac: float = 0.08) -> float:
    """Sell-to-open fill (credit) from mid."""
    return max(0.0, float(mid) - half_spread(mid, floor=floor, frac=frac))


def debit_after_spread(mid: float, *, floor: float = 0.05, frac: float = 0.08) -> float:
    """Buy-to-close fill (debit) from mid."""
    return float(mid) + half_spread(mid, floor=floor, frac=frac)


def cash_secured_gain(
    credit: float,
    debit: float,
    strike: float,
) -> Optional[float]:
    """% return on cash-secured notional (credit - debit) / strike * 100."""
    k = float(strike)
    if k <= 0:
        return None
    return (float(credit) - float(debit)) / k * 100.0


def round_option_strike(px: float) -> float:
    """Round underlying to a common equity put strike increment."""
    p = abs(float(px))
    if p < 25:
        step = 0.5
    elif p < 100:
        step = 1.0
    elif p < 200:
        step = 2.5
    else:
        step = 5.0
    return round(p / step) * step


def strike_candidates(target_strike: float, *, n_each_side: int = 2) -> List[float]:
    """ATM-rounded strike plus nearby increments (for empty-bar retries)."""
    mid = round_option_strike(target_strike)
    p = abs(float(target_strike))
    if p < 25:
        step = 0.5
    elif p < 100:
        step = 1.0
    elif p < 200:
        step = 2.5
    else:
        step = 5.0
    out: List[float] = []
    for i in range(-int(n_each_side), int(n_each_side) + 1):
        k = mid + i * step
        if k > 0:
            out.append(round(k, 4))
    # Prefer original mid first
    seen = set()
    ordered = []
    for k in [mid] + out:
        if k not in seen:
            seen.add(k)
            ordered.append(k)
    return ordered


def pick_nearest_put_contract(
    contracts: Sequence[Dict[str, Any]],
    target_strike: float,
) -> Optional[Dict[str, Any]]:
    """Nearest put by strike_price (or strike)."""
    best = None
    best_abs = None
    tgt = float(target_strike)
    for c in contracts or []:
        ctype = str(c.get("contract_type") or c.get("type") or "").lower()
        if ctype and ctype not in ("put", "p"):
            continue
        raw = c.get("strike_price", c.get("strike"))
        if raw is None:
            continue
        try:
            k = float(raw)
        except (TypeError, ValueError):
            continue
        dist = abs(k - tgt)
        if best_abs is None or dist < best_abs:
            best_abs = dist
            best = c
    return best


@dataclass
class ListedPut:
    ticker: str
    strike: float
    expiry: str
    target_strike: float
    target_expiry: str
    strike_diff: float
    expiry_diff_days: int
    source: str = "contracts"


def resolve_listed_put(
    *,
    underlying: str,
    expiries: Sequence[str],
    contracts: Sequence[Dict[str, Any]],
    buy_date: date,
    target_dte: int,
    target_strike: float,
) -> Optional[ListedPut]:
    """Snap synthetic DTE/strike to a listed expiry + nearest put strike.

    Prefers Polygon contract metadata when present; otherwise rounds strike
    to a standard increment and builds an OCC ticker (free-tier safe).
    """
    target_exp = buy_date + timedelta(days=int(target_dte))
    expiry = closest_listed_expiry(expiries, target_exp)
    if not expiry:
        return None
    put = pick_nearest_put_contract(contracts, target_strike)
    if put is not None:
        strike = float(put.get("strike_price", put.get("strike")))
        ticker = str(put.get("ticker") or "").strip()
        if not ticker:
            ticker = polygon_option_ticker(underlying, expiry, strike, "P")
        source = "contracts"
    else:
        strike = round_option_strike(target_strike)
        ticker = polygon_option_ticker(underlying, expiry, strike, "P")
        source = "rounded"
    if not ticker.startswith("O:"):
        ticker = "O:" + ticker.lstrip("O:")
    exp_d = date.fromisoformat(expiry[:10])
    return ListedPut(
        ticker=ticker,
        strike=strike,
        expiry=expiry,
        target_strike=float(target_strike),
        target_expiry=target_exp.isoformat(),
        strike_diff=strike - float(target_strike),
        expiry_diff_days=(exp_d - target_exp).days,
        source=source,
    )


def occupancy_fcfs_put(
    trades: List[Dict[str, Any]],
    *,
    max_open: int = 8,
    unique_symbol: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """FCFS occupancy on put trades (entry_date / exit_date / symbol)."""
    if not trades:
        return [], {"n_kept": 0, "n_skipped": 0, "max_open": int(max_open)}
    rows = []
    for i, tr in enumerate(trades):
        buy = pd.Timestamp(tr["entry_date"]).normalize()
        sell = pd.Timestamp(tr["exit_date"]).normalize()
        if buy.tzinfo is not None:
            buy = buy.tz_convert("UTC").tz_localize(None)
        if sell.tzinfo is not None:
            sell = sell.tz_convert("UTC").tz_localize(None)
        rows.append(
            {
                **tr,
                "_i": i,
                "_buy": buy.normalize(),
                "_sell": sell.normalize(),
                "_sym": str(tr.get("symbol") or "").upper(),
            }
        )
    rows.sort(key=lambda r: (r["_buy"], r["_sym"], r["_i"]))

    open_pos: List[Tuple[pd.Timestamp, str]] = []
    kept: List[Dict[str, Any]] = []
    n_skip = 0
    for row in rows:
        buy = row["_buy"]
        open_pos = [(s, sym) for s, sym in open_pos if s > buy]
        if unique_symbol and any(sym == row["_sym"] for _, sym in open_pos):
            n_skip += 1
            continue
        if int(max_open) > 0 and len(open_pos) >= int(max_open):
            n_skip += 1
            continue
        kept.append({k: v for k, v in row.items() if not str(k).startswith("_")})
        open_pos.append((row["_sell"], row["_sym"]))
    return kept, {
        "n_kept": len(kept),
        "n_skipped": n_skip,
        "max_open": int(max_open),
    }


def summarize_gains(gains: Sequence[float]) -> Dict[str, float]:
    s = pd.Series([float(x) for x in gains if x == x], dtype=float)
    if s.empty:
        return {"n": 0, "E": float("nan"), "PF": float("nan"), "WR": float("nan")}
    wins = s[s > 0].sum()
    losses = (-s[s < 0]).sum()
    pf = float(wins / losses) if losses > 0 else float("inf") if wins > 0 else float("nan")
    return {
        "n": int(len(s)),
        "E": float(s.mean()),
        "PF": pf,
        "WR": float((s > 0).mean() * 100.0),
        "med": float(s.median()),
    }
