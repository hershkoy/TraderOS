"""Black-Scholes short-put overlay on the last-15m H2 book (research only).

Same signal clock as ``current_best`` 1d (last-RTH 15m open above daily rail).
Instead of buying stock, sell a cash-secured put and manage:

  - Take profit when mark <= ``tp_frac`` of credit (default 0.50).
  - Stop when the underlying daily low tags the stock ATR k=2 hard-stop;
    buy the put back at BS marked at the stop price (constant IV).
  - Else hold to calendar expiry and settle intrinsic.

IV is fixed at entry from annualized ATR% (or ``--iv`` override). Constant IV
is optimistic vs live (stops often expand IV). If this book still fails the
E/PF gate, real-option API replay is unlikely to rescue it.

Occupancy is not re-walked. Primary P&L is cash-secured % = (credit - exit) / K.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, Optional

import pandas as pd

from utils.options.greeks import black_scholes_put, calculate_greeks
from utils.research.last_15m_volume_geometry import _num, atr_dollars_for_row
from utils.scanning.channel_touch_bought import (
    ATR_STOP_MULT,
    STOP_PCT,
    STOP_PCT_CEIL,
    STOP_PCT_FLOOR,
    hard_stop_price,
)

DEFAULT_DTE = 45
DEFAULT_TP_FRAC = 0.50
DEFAULT_R = 0.05
DEFAULT_STRIKE_MODE = "atm"  # atm | delta
DEFAULT_TARGET_DELTA = -0.30
IV_FLOOR = 0.15
IV_CEIL = 0.90
REASON_OK = "ok"
REASON_NO_DAILY = "no_daily"
REASON_NO_ENTRY = "no_entry"
REASON_NO_CREDIT = "no_credit"
REASON_NO_STOP = "no_stop"
REASON_BAD_IV = "bad_iv"
EXIT_TP = "tp50"
EXIT_HARD_STOP = "hard_stop"
EXIT_EXPIRY = "expiry"


@dataclass
class ShortPutResult:
    filled: bool
    reason: str
    strike: Optional[float] = None
    credit: Optional[float] = None
    iv: Optional[float] = None
    dte: Optional[int] = None
    entry_delta: Optional[float] = None
    hard_stop: Optional[float] = None
    exit_premium: Optional[float] = None
    exit_reason: Optional[str] = None
    exit_date: Optional[date] = None
    hold_days: Optional[int] = None
    gain_cash_secured_pct: Optional[float] = None
    gain_rop_pct: Optional[float] = None
    stock_gain_pct: Optional[float] = None


def _as_date(val: Any) -> Optional[date]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, date) and not isinstance(val, datetime):
        return val
    ts = pd.Timestamp(val)
    if pd.isna(ts):
        return None
    return ts.date()


def _panel_dates(panel: pd.DataFrame) -> pd.DatetimeIndex:
    idx = panel.index
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.to_datetime(idx, errors="coerce")
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    return pd.DatetimeIndex(idx).normalize()


def round_strike(px: float) -> float:
    """Coarse listed-strike grid (research approximation)."""
    p = float(px)
    if p <= 0:
        return p
    if p < 25.0:
        step = 0.5
    elif p < 200.0:
        step = 1.0
    else:
        step = 5.0
    return round(p / step) * step


def iv_from_atr_pct(atr_pct: float, *, floor: float = IV_FLOOR, ceil: float = IV_CEIL) -> float:
    """Annualize daily ATR% as a constant-IV proxy: atr% * sqrt(252)."""
    daily = float(atr_pct) / 100.0
    if not math.isfinite(daily) or daily <= 0:
        return float(floor)
    return float(max(floor, min(ceil, daily * math.sqrt(252.0))))


def strike_for_put_delta(
    S: float,
    T: float,
    r: float,
    sigma: float,
    target_delta: float = DEFAULT_TARGET_DELTA,
    *,
    lo_mult: float = 0.50,
    hi_mult: float = 1.05,
    iters: int = 40,
) -> float:
    """Binary-search strike so BS put delta ~= target_delta (e.g. -0.30)."""
    s = float(S)
    tgt = float(target_delta)
    lo = s * float(lo_mult)
    hi = s * float(hi_mult)
    for _ in range(int(iters)):
        mid = 0.5 * (lo + hi)
        g = calculate_greeks(s, mid, T, r, sigma, "P")
        d = float(g["delta"])
        # Put delta in [-1, 0]; more negative = deeper ITM = higher strike.
        if d < tgt:
            hi = mid
        else:
            lo = mid
    return round_strike(0.5 * (lo + hi))


def resolve_hard_stop(row: pd.Series, entry_px: float) -> Optional[float]:
    hs = _num(row, "hard_stop_price")
    if hs is not None and hs > 0 and hs < entry_px:
        return float(hs)
    atr = atr_dollars_for_row(row, entry_px)
    if atr is None:
        atr_pct = _num(row, "atr_1d_pct", "atr_pct")
        if atr_pct is not None and atr_pct > 0:
            atr = entry_px * atr_pct / 100.0
    if atr is None:
        return hard_stop_price(
            entry_px,
            atr_at_entry=None,
            atr_stop_mult=None,
            stop_pct=STOP_PCT,
            stop_pct_floor=STOP_PCT_FLOOR,
            stop_pct_ceil=STOP_PCT_CEIL,
        )
    return hard_stop_price(
        entry_px,
        atr_at_entry=float(atr),
        atr_stop_mult=ATR_STOP_MULT,
        stop_pct=STOP_PCT,
        stop_pct_floor=STOP_PCT_FLOOR,
        stop_pct_ceil=STOP_PCT_CEIL,
    )


def resolve_iv(
    row: pd.Series,
    entry_px: float,
    *,
    iv_override: Optional[float] = None,
) -> Optional[float]:
    if iv_override is not None:
        v = float(iv_override)
        if not math.isfinite(v) or v <= 0:
            return None
        return float(max(IV_FLOOR, min(IV_CEIL, v)))
    atr_pct = _num(row, "atr_1d_pct", "atr_pct")
    if atr_pct is None or atr_pct <= 0:
        atr = atr_dollars_for_row(row, entry_px)
        if atr is not None and entry_px > 0:
            atr_pct = 100.0 * float(atr) / float(entry_px)
    if atr_pct is None or atr_pct <= 0:
        return None
    return iv_from_atr_pct(float(atr_pct))


def _years(days: float) -> float:
    return max(float(days), 0.0) / 365.25


def _busday_on_or_before(d: date) -> date:
    """Snap calendar expiry onto the last weekday (equity options last trade)."""
    out = d
    while out.weekday() >= 5:  # Sat/Sun
        out -= timedelta(days=1)
    return out


def simulate_short_put(
    row: pd.Series,
    panel_1d: Optional[pd.DataFrame],
    *,
    dte: int = DEFAULT_DTE,
    tp_frac: float = DEFAULT_TP_FRAC,
    r: float = DEFAULT_R,
    strike_mode: str = DEFAULT_STRIKE_MODE,
    target_delta: float = DEFAULT_TARGET_DELTA,
    iv_override: Optional[float] = None,
    opt_friction_per_share: float = 0.0,
    stop_iv_mult: float = 1.0,
) -> ShortPutResult:
    """Walk daily bars after the stock fill; mark puts with constant-IV BS.

    ``stop_iv_mult`` > 1 re-marks hard-stop exits at ``iv * stop_iv_mult``
    (stress for crash IV expansion; winners still use entry IV).
    """
    entry_px = _num(row, "buy_price")
    entry_day = _as_date(row.get("buy_date"))
    stock_gain = _num(row, "gain_pct")
    if entry_px is None or entry_px <= 0 or entry_day is None:
        return ShortPutResult(filled=False, reason=REASON_NO_ENTRY, stock_gain_pct=stock_gain)
    if panel_1d is None or panel_1d.empty:
        return ShortPutResult(filled=False, reason=REASON_NO_DAILY, stock_gain_pct=stock_gain)

    hard_stop = resolve_hard_stop(row, float(entry_px))
    if hard_stop is None or hard_stop <= 0:
        return ShortPutResult(filled=False, reason=REASON_NO_STOP, stock_gain_pct=stock_gain)

    iv = resolve_iv(row, float(entry_px), iv_override=iv_override)
    if iv is None:
        return ShortPutResult(filled=False, reason=REASON_BAD_IV, stock_gain_pct=stock_gain)

    dte_i = max(1, int(dte))
    T0 = _years(dte_i)
    mode = str(strike_mode or DEFAULT_STRIKE_MODE).lower().strip()
    if mode == "delta":
        strike = strike_for_put_delta(
            float(entry_px), T0, float(r), float(iv), target_delta=float(target_delta)
        )
    else:
        strike = round_strike(float(entry_px))

    credit = float(black_scholes_put(float(entry_px), float(strike), T0, float(r), float(iv)))
    if not math.isfinite(credit) or credit <= 1e-8:
        return ShortPutResult(
            filled=False,
            reason=REASON_NO_CREDIT,
            strike=strike,
            iv=iv,
            dte=dte_i,
            hard_stop=hard_stop,
            stock_gain_pct=stock_gain,
        )

    g0 = calculate_greeks(float(entry_px), float(strike), T0, float(r), float(iv), "P")
    entry_delta = float(g0["delta"])
    tp_level = float(tp_frac) * credit
    # Calendar DTE then snap to last weekday so weekend expiry settles on Friday.
    expiry = _busday_on_or_before(entry_day + timedelta(days=dte_i))

    dates = _panel_dates(panel_1d)
    work = panel_1d.copy()
    work.index = dates
    after = work.loc[work.index.date > entry_day]
    after = after.loc[after.index.date <= expiry]

    exit_prem: Optional[float] = None
    exit_reason: Optional[str] = None
    exit_day: Optional[date] = None

    for ts, bar in after.iterrows():
        day = ts.date() if hasattr(ts, "date") else _as_date(ts)
        if day is None:
            continue
        try:
            lo = float(bar["low"])
            cl = float(bar["close"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (math.isfinite(lo) and math.isfinite(cl) and lo > 0 and cl > 0):
            continue
        days_left = (expiry - day).days
        T = _years(days_left)
        is_expiry_session = day >= expiry

        if lo <= float(hard_stop):
            # On expiry session, stop still wins if tagged; else mark at stop.
            stop_iv = float(iv) * max(1.0, float(stop_iv_mult))
            if is_expiry_session:
                exit_prem = max(0.0, float(strike) - float(hard_stop))
            else:
                exit_prem = float(
                    black_scholes_put(
                        float(hard_stop), float(strike), T, float(r), stop_iv
                    )
                )
            exit_reason = EXIT_HARD_STOP
            exit_day = day
            break

        if is_expiry_session or days_left <= 0:
            exit_prem = max(0.0, float(strike) - cl)
            exit_reason = EXIT_EXPIRY
            exit_day = day
            break

        mark = float(black_scholes_put(cl, float(strike), T, float(r), float(iv)))
        if mark <= tp_level + 1e-12:
            exit_prem = float(tp_level)
            exit_reason = EXIT_TP
            exit_day = day
            break

    if exit_prem is None:
        if after.empty:
            return ShortPutResult(
                filled=False,
                reason=REASON_NO_DAILY,
                strike=strike,
                credit=credit,
                iv=iv,
                dte=dte_i,
                entry_delta=entry_delta,
                hard_stop=hard_stop,
                stock_gain_pct=stock_gain,
            )
        # Panel truncated before expiry session: mark BS at last close (not intrinsic).
        last = after.iloc[-1]
        last_day = after.index[-1].date()
        try:
            cl = float(last["close"])
        except (TypeError, ValueError):
            return ShortPutResult(
                filled=False,
                reason=REASON_NO_DAILY,
                strike=strike,
                credit=credit,
                iv=iv,
                dte=dte_i,
                entry_delta=entry_delta,
                hard_stop=hard_stop,
                stock_gain_pct=stock_gain,
            )
        days_left = max(0, (expiry - last_day).days)
        if last_day >= expiry or days_left <= 0:
            exit_prem = max(0.0, float(strike) - cl)
        else:
            exit_prem = float(
                black_scholes_put(cl, float(strike), _years(days_left), float(r), float(iv))
            )
        exit_reason = EXIT_EXPIRY
        exit_day = last_day

    fric = max(0.0, float(opt_friction_per_share))
    pnl_per_share = (credit - float(exit_prem)) - 2.0 * fric
    gain_cs = 100.0 * pnl_per_share / float(strike)
    gain_rop = 100.0 * pnl_per_share / credit
    hold = (exit_day - entry_day).days if exit_day is not None else None

    return ShortPutResult(
        filled=True,
        reason=REASON_OK,
        strike=float(strike),
        credit=float(credit),
        iv=float(iv),
        dte=dte_i,
        entry_delta=entry_delta,
        hard_stop=float(hard_stop),
        exit_premium=float(exit_prem),
        exit_reason=exit_reason,
        exit_date=exit_day,
        hold_days=hold,
        gain_cash_secured_pct=float(gain_cs),
        gain_rop_pct=float(gain_rop),
        stock_gain_pct=stock_gain,
    )


def apply_short_put_book(
    trades: pd.DataFrame,
    panels_1d: Dict[str, pd.DataFrame],
    *,
    dte: int = DEFAULT_DTE,
    tp_frac: float = DEFAULT_TP_FRAC,
    r: float = DEFAULT_R,
    strike_mode: str = DEFAULT_STRIKE_MODE,
    target_delta: float = DEFAULT_TARGET_DELTA,
    iv_override: Optional[float] = None,
    opt_friction_per_share: float = 0.0,
    stop_iv_mult: float = 1.0,
) -> pd.DataFrame:
    rows = []
    for idx, row in trades.iterrows():
        sym = str(row.get("stock", "")).upper()
        got = simulate_short_put(
            row,
            panels_1d.get(sym),
            dte=dte,
            tp_frac=tp_frac,
            r=r,
            strike_mode=strike_mode,
            target_delta=target_delta,
            iv_override=iv_override,
            opt_friction_per_share=opt_friction_per_share,
            stop_iv_mult=stop_iv_mult,
        )
        rec = row.to_dict()
        rec["src_idx"] = idx
        rec["put_ok"] = bool(got.filled)
        rec["put_reason"] = got.reason
        rec["put_strike"] = got.strike
        rec["put_credit"] = got.credit
        rec["put_iv"] = got.iv
        rec["put_dte"] = got.dte
        rec["put_entry_delta"] = got.entry_delta
        rec["put_hard_stop"] = got.hard_stop
        rec["put_exit_premium"] = got.exit_premium
        rec["put_exit_reason"] = got.exit_reason
        rec["put_exit_date"] = got.exit_date.isoformat() if got.exit_date else None
        rec["put_hold_days"] = got.hold_days
        rec["gain_pct_stock"] = got.stock_gain_pct
        rec["gain_pct"] = got.gain_cash_secured_pct if got.filled else None
        rec["gain_rop_pct"] = got.gain_rop_pct if got.filled else None
        rec["exit_reason"] = got.exit_reason if got.filled else row.get("exit_reason")
        rows.append(rec)
    return pd.DataFrame(rows)
