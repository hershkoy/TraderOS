"""Open-vs-close credit + mid-implied ~20-delta put vertical spot-check (Polygon aggs).

Free-tier friendly: no historical snapshots (403). Uses day OHLCV opens/closes,
solves BS IV from mid to pick ~20-delta short put, then measures vertical credit
at open vs close vs the BS claim-check credit.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
from scipy.optimize import brentq

from utils.options.greeks import black_scholes_put, calculate_greeks
from utils.research.last_15m_short_put_bs import _years
from utils.research.post_earnings_pcs_live import (
    credit_vertical_mids,
    resolve_vertical_tickers,
)
from utils.research.short_put_live_realism import nearby_fridays, round_option_strike
from utils.research.short_put_polygon_spotcheck import (
    bars_from_aggs_response,
    polygon_option_ticker,
)

DEFAULT_R = 0.05
DEFAULT_TARGET_DELTA = -0.20
MIN_CREDIT_FRAC = 0.16


def imply_put_iv(
    mid: float,
    S: float,
    K: float,
    T: float,
    r: float = DEFAULT_R,
    *,
    lo: float = 1e-4,
    hi: float = 5.0,
) -> Optional[float]:
    """Solve BS put IV from mid price. None if mid outside no-arb bounds."""
    if mid is None or S <= 0 or K <= 0 or T <= 0:
        return None
    m = float(mid)
    if m <= 0 or not math.isfinite(m):
        return None
    # intrinsic lower bound
    intrinsic = max(0.0, float(K) - float(S))
    if m < intrinsic - 1e-6:
        return None
    if m > float(K):  # absurd
        return None

    def objective(sig: float) -> float:
        return float(black_scholes_put(float(S), float(K), float(T), float(r), float(sig))) - m

    try:
        f_lo = objective(lo)
        f_hi = objective(hi)
        if f_lo * f_hi > 0:
            # mid above even high-vol put -> clamp hi; below lo -> None
            if abs(f_lo) < abs(f_hi):
                return float(lo) if abs(f_lo) < 0.05 else None
            return float(hi) if abs(f_hi) < 0.05 else None
        return float(brentq(objective, lo, hi, maxiter=80))
    except Exception:
        return None


def put_delta_from_mid(
    mid: float,
    S: float,
    K: float,
    T: float,
    r: float = DEFAULT_R,
) -> Tuple[Optional[float], Optional[float]]:
    """Return (delta, iv) from mid-implied IV."""
    iv = imply_put_iv(mid, S, K, T, r)
    if iv is None:
        return None, None
    g = calculate_greeks(float(S), float(K), float(T), float(r), float(iv), "P")
    return float(g["delta"]), float(iv)


def strike_ladder(entry_px: float, *, n: int = 10) -> List[float]:
    """OTM put strikes roughly 78%-96% of spot (20-delta hunting ground)."""
    s = float(entry_px)
    p = abs(s)
    if p < 25:
        step = 0.5
    elif p < 100:
        step = 1.0
    elif p < 200:
        step = 2.5
    else:
        step = 5.0
    lo = round_option_strike(s * 0.78)
    hi = round_option_strike(s * 0.96)
    out: List[float] = []
    k = lo
    while k <= hi + 1e-9:
        out.append(round(k, 4))
        k += step
    if len(out) <= int(n):
        return out
    # Keep n strikes nearest to 0.90*S (typical 20d zone at moderate IV)
    target = s * 0.90
    out = sorted(out, key=lambda x: abs(x - target))[: int(n)]
    return sorted(out)


@dataclass
class LegBar:
    date: date
    open: float
    close: float
    high: float
    low: float


def first_bar_on(
    bars: pd.DataFrame,
    day: date,
) -> Optional[LegBar]:
    if bars is None or bars.empty:
        return None
    dcol = pd.to_datetime(bars["date"]).dt.date
    hit = bars.loc[dcol == day]
    if hit.empty:
        return None
    r = hit.iloc[0]
    return LegBar(
        date=day,
        open=float(r["open"]),
        close=float(r["close"]),
        high=float(r["high"]),
        low=float(r["low"]),
    )


def pick_delta_short(
    candidates: Sequence[Dict[str, Any]],
    target_delta: float = DEFAULT_TARGET_DELTA,
) -> Optional[Dict[str, Any]]:
    """Nearest put by |delta - target| among candidates with delta."""
    best = None
    best_abs = None
    tgt = float(target_delta)
    for c in candidates:
        d = c.get("delta")
        if d is None:
            continue
        dist = abs(float(d) - tgt)
        if best_abs is None or dist < best_abs:
            best_abs = dist
            best = c
    return best


def vertical_credits_from_legs(
    short_open: float,
    short_close: float,
    long_open: float,
    long_close: float,
    *,
    haircut: bool = False,
) -> Dict[str, float]:
    return {
        "credit_open": credit_vertical_mids(short_open, long_open, haircut=haircut),
        "credit_close": credit_vertical_mids(short_close, long_close, haircut=haircut),
    }


def evaluate_trade_open_close(
    *,
    symbol: str,
    entry_date: date,
    entry_px: float,
    bs_short_k: float,
    bs_long_k: float,
    bs_credit: float,
    width: float,
    dte: int,
    fetch_bars,  # callable(ticker, start, end) -> DataFrame
    target_delta: float = DEFAULT_TARGET_DELTA,
    r: float = DEFAULT_R,
) -> Dict[str, Any]:
    """Compare BS strikes open/close credits vs mid-implied ~20d vertical."""
    out: Dict[str, Any] = {
        "symbol": symbol,
        "entry_date": entry_date.isoformat(),
        "entry_px": float(entry_px),
        "bs_credit": float(bs_credit),
        "bs_short_k": float(bs_short_k),
        "bs_long_k": float(bs_long_k),
        "status": "pending",
    }
    st, lt, expiry, sk, lk = resolve_vertical_tickers(
        symbol, entry_date, bs_short_k, bs_long_k, dte
    )
    width_bs = float(sk) - float(lk)
    out.update(
        {
            "bs_live_short_k": sk,
            "bs_live_long_k": lk,
            "bs_live_expiry": expiry,
            "bs_live_width": width_bs,
            "bs_short_ticker": st,
            "bs_long_ticker": lt,
        }
    )
    start = (entry_date - timedelta(days=1)).isoformat()
    end = (entry_date + timedelta(days=1)).isoformat()
    T = _years(max(1, (date.fromisoformat(expiry[:10]) - entry_date).days))

    short_bars = fetch_bars(st, start, end)
    long_bars = fetch_bars(lt, start, end)
    sb = first_bar_on(short_bars, entry_date)
    lb = first_bar_on(long_bars, entry_date)
    if sb is None or lb is None:
        out["status"] = "no_bs_strike_bars"
        out["n_short"] = 0 if short_bars is None else int(len(short_bars))
        out["n_long"] = 0 if long_bars is None else int(len(long_bars))
        # still try delta ladder below
    else:
        creds = vertical_credits_from_legs(sb.open, sb.close, lb.open, lb.close)
        out["bs_strike_credit_open"] = creds["credit_open"]
        out["bs_strike_credit_close"] = creds["credit_close"]
        out["bs_strike_open_vs_bs"] = (
            creds["credit_open"] / float(bs_credit) if bs_credit > 0 else None
        )
        out["bs_strike_close_vs_bs"] = (
            creds["credit_close"] / float(bs_credit) if bs_credit > 0 else None
        )
        d_open, iv_o = put_delta_from_mid(sb.open, entry_px, sk, T, r)
        d_close, iv_c = put_delta_from_mid(sb.close, entry_px, sk, T, r)
        out["bs_strike_delta_open"] = d_open
        out["bs_strike_delta_close"] = d_close
        out["bs_strike_iv_open"] = iv_o
        out["bs_strike_iv_close"] = iv_c

    # Mid-implied ~20-delta short from strike ladder (open marks)
    fridays = nearby_fridays(entry_date + timedelta(days=int(dte)), window_days=21)
    exp = fridays[0].isoformat() if fridays else expiry
    T2 = _years(max(1, (date.fromisoformat(exp[:10]) - entry_date).days))
    width_use = float(width) if width > 0 else float(width_bs) if width_bs > 0 else 5.0
    cands: List[Dict[str, Any]] = []
    for k in strike_ladder(entry_px):
        ticker = polygon_option_ticker(symbol, exp, k, "P")
        bars = fetch_bars(ticker, start, end)
        bar = first_bar_on(bars, entry_date)
        if bar is None:
            continue
        # prefer open for delta targeting (Brian entry after announce)
        mid_o = float(bar.open)
        delta, iv = put_delta_from_mid(mid_o, entry_px, k, T2, r)
        if delta is None:
            continue
        cands.append(
            {
                "strike": k,
                "ticker": ticker,
                "open": mid_o,
                "close": float(bar.close),
                "delta": delta,
                "iv": iv,
            }
        )
    out["ladder_n"] = len(cands)
    pick = pick_delta_short(cands, target_delta=target_delta)
    if pick is None:
        out["status"] = out.get("status") if out.get("status") != "pending" else "no_delta_pick"
        if out["status"] == "pending":
            out["status"] = "ok_bs_only" if sb is not None else "fail"
        return out

    short_k = float(pick["strike"])
    long_k = round_option_strike(short_k - width_use)
    if long_k >= short_k:
        long_k = short_k - width_use
    long_ticker = polygon_option_ticker(symbol, exp, long_k, "P")
    long_bars2 = fetch_bars(long_ticker, start, end)
    long_bar = first_bar_on(long_bars2, entry_date)
    out.update(
        {
            "d20_short_k": short_k,
            "d20_long_k": long_k,
            "d20_expiry": exp,
            "d20_short_ticker": pick["ticker"],
            "d20_long_ticker": long_ticker,
            "d20_delta_open": pick["delta"],
            "d20_iv_open": pick["iv"],
        }
    )
    if long_bar is None:
        out["status"] = "no_long_bar"
        return out

    d20 = vertical_credits_from_legs(
        pick["open"], pick["close"], long_bar.open, long_bar.close
    )
    out["d20_credit_open"] = d20["credit_open"]
    out["d20_credit_close"] = d20["credit_close"]
    min_credit = MIN_CREDIT_FRAC * max(short_k - long_k, 1e-9)
    out["d20_min_credit"] = min_credit
    out["d20_open_meets_min"] = bool(d20["credit_open"] >= min_credit - 1e-9)
    out["d20_close_meets_min"] = bool(d20["credit_close"] >= min_credit - 1e-9)
    out["d20_open_vs_bs"] = (
        d20["credit_open"] / float(bs_credit) if bs_credit > 0 else None
    )
    out["d20_close_vs_bs"] = (
        d20["credit_close"] / float(bs_credit) if bs_credit > 0 else None
    )
    out["status"] = "ok"
    return out
