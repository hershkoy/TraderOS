"""Buy-write (long stock + covered call) on last-15m H2 fills.

Buy the breakout fill, sell ~target_dte / ~target_delta call the same day,
hold to call expiry (called away or liquidate shares). Optional earnings
flatten is applied by ``last_15m_cc_earnings`` (BTC call + stock mid exit).

Gain % = pnl_per_share / buy_price * 100 (stock notional, not cash-secured).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Dict, List, Mapping, Optional

import pandas as pd

from utils.options.greeks import black_scholes_call, calculate_greeks
from utils.research.last_15m_short_put_bs import (
    _as_date,
    _busday_on_or_before,
    _num,
    _panel_dates,
    _years,
    resolve_iv,
    round_strike,
)
from utils.research.last_15m_wheel import (
    DEFAULT_DTE,
    DEFAULT_DTE_TOL,
    DEFAULT_MIN_CREDIT,
    DEFAULT_MIN_DTE,
    QuoteSource,
    _dte_window,
    _spot_on,
    select_contract,
)
from utils.research.qqq_leaps_gap import strike_for_call_delta
from utils.research.spy_wheel import entry_credit, exit_debit

DEFAULT_R = 0.05
DEFAULT_BS_FRICTION = 0.05  # $/share haircut on ask_bid BS credits

REASON_OK = "ok"
REASON_NO_ENTRY = "no_entry"
REASON_NO_DAILY = "no_daily"
REASON_NO_CALL_CHAIN = "no_call_chain"
REASON_NO_CALL_QUOTE = "no_call_quote"
REASON_NO_SPOT = "no_spot_expiry"

EXIT_CALL_ASSIGN = "call_assign"
EXIT_SHARE_LIQ = "share_liquidate"
EXIT_EARNINGS_FLATTEN = "earnings_flatten"


@dataclass
class BuyWriteLeg:
    expiration: date
    strike: float
    credit: float
    entry_date: date
    entry_delta: Optional[float]
    entry_dte: int
    source: str
    bid: float = 0.0
    ask: float = 0.0


@dataclass
class BuyWriteResult:
    filled: bool
    reason: str
    exit_reason: Optional[str] = None
    call: Optional[BuyWriteLeg] = None
    share_exit_px: Optional[float] = None
    call_close_debit: float = 0.0
    called_away: bool = False
    pnl_per_share: Optional[float] = None
    gain_pct: Optional[float] = None
    stock_gain_pct: Optional[float] = None
    mark_mode: str = "mid"
    hold_days: Optional[int] = None
    exit_date: Optional[date] = None
    notes: str = ""
    events: List[Dict[str, Any]] = field(default_factory=list)


def buy_write_pnl_per_share(
    *,
    entry_px: float,
    call_credit: float,
    called_away: bool,
    call_strike: Optional[float] = None,
    share_exit_px: Optional[float] = None,
    call_close_debit: float = 0.0,
) -> float:
    """Net $ per share: long stock + short call (+ optional BTC debit)."""
    pnl = -float(entry_px) + float(call_credit) - float(call_close_debit)
    if called_away:
        if call_strike is None:
            raise ValueError("called_away requires call_strike")
        pnl += float(call_strike)
    else:
        if share_exit_px is None:
            raise ValueError("share liquidation requires share_exit_px")
        pnl += float(share_exit_px)
    return float(pnl)


def stock_notional_pct(pnl_per_share: float, entry_px: float) -> Optional[float]:
    e = float(entry_px)
    if e <= 0:
        return None
    return float(pnl_per_share) / e * 100.0


def simulate_buy_write(
    row: pd.Series,
    panel_1d: Optional[pd.DataFrame],
    source: QuoteSource,
    *,
    mark_mode: str = "mid",
    target_dte: int = DEFAULT_DTE,
    target_delta: float = 0.30,
    dte_tol: int = DEFAULT_DTE_TOL,
    min_dte: int = DEFAULT_MIN_DTE,
    min_credit: float = DEFAULT_MIN_CREDIT,
    strike_mode: str = "delta",
    entry_slack_days: int = 0,
) -> BuyWriteResult:
    """Open covered call on the signal day; hold to call expiry."""
    stock_gain = _num(row, "gain_pct")
    # Prefer explicit stock trail gain when wheel CSV already overwrote gain_pct.
    if row.get("gain_pct_stock") is not None and pd.notna(row.get("gain_pct_stock")):
        stock_gain = float(row["gain_pct_stock"])
    entry_px = _num(row, "buy_price")
    entry_day = _as_date(row.get("buy_date"))
    sym = str(row.get("stock") or "").upper()
    if not sym or entry_px is None or entry_px <= 0 or entry_day is None:
        return BuyWriteResult(
            filled=False, reason=REASON_NO_ENTRY, stock_gain_pct=stock_gain
        )
    if panel_1d is None or panel_1d.empty:
        return BuyWriteResult(
            filled=False, reason=REASON_NO_DAILY, stock_gain_pct=stock_gain
        )

    spot0 = _spot_on(panel_1d, entry_day) or float(entry_px)
    dte_lo, dte_hi = _dte_window(target_dte, dte_tol, min_dte)

    call_open_day = entry_day
    call_pick = None
    call_quote = None
    call_src = "unknown"
    for slack in range(int(entry_slack_days) + 1):
        day = entry_day + timedelta(days=slack)
        if day.weekday() >= 5:
            continue
        chain = source.chain_on_day(
            sym, day, "C", dte_lo, dte_hi, spot_hint=float(spot0)
        )
        pick = select_contract(
            chain,
            day=day,
            spot=float(spot0),
            target_dte=int(target_dte),
            target_delta=float(target_delta),
            dte_tol=int(dte_tol),
            min_dte=int(min_dte),
            min_credit=float(min_credit),
            right="C",
            strike_mode=strike_mode,
        )
        if pick is None:
            continue
        q = source.quote_on_day(
            sym, pick["expiration"], float(pick["strike"]), "C", day
        )
        if q is None:
            if math.isfinite(float(pick.get("bid") or float("nan"))) and math.isfinite(
                float(pick.get("ask") or float("nan"))
            ):
                q = {
                    "bid": float(pick["bid"]),
                    "ask": float(pick["ask"]),
                    "mid": float(
                        pick.get("mid")
                        or 0.5 * (float(pick["bid"]) + float(pick["ask"]))
                    ),
                    "source": "chain",
                }
            else:
                continue
        call_open_day = day
        call_pick = pick
        call_quote = q
        call_src = str(q.get("source") or pick.get("source") or "source")
        break

    if call_pick is None or call_quote is None:
        return BuyWriteResult(
            filled=False,
            reason=REASON_NO_CALL_CHAIN if call_pick is None else REASON_NO_CALL_QUOTE,
            stock_gain_pct=stock_gain,
            mark_mode=mark_mode,
        )

    call_cred = entry_credit(call_quote, mark_mode)
    if call_cred < float(min_credit):
        return BuyWriteResult(
            filled=False,
            reason=REASON_NO_CALL_QUOTE,
            stock_gain_pct=stock_gain,
            mark_mode=mark_mode,
        )

    call_leg = BuyWriteLeg(
        expiration=call_pick["expiration"],
        strike=float(call_pick["strike"]),
        credit=float(call_cred),
        entry_date=call_open_day,
        entry_delta=(
            float(call_pick["delta"])
            if call_pick.get("delta") is not None
            and math.isfinite(float(call_pick["delta"]))
            else None
        ),
        entry_dte=int(call_pick["dte"]),
        source=call_src,
        bid=float(call_quote.get("bid") or 0.0),
        ask=float(call_quote.get("ask") or 0.0),
    )
    events: List[Dict[str, Any]] = [
        {
            "event": "open_call",
            "day": call_open_day.isoformat(),
            "strike": call_leg.strike,
            "exp": call_leg.expiration.isoformat(),
            "credit": call_leg.credit,
            "source": call_src,
        }
    ]

    call_exp = call_leg.expiration
    spot_call = _spot_on(panel_1d, call_exp)
    if spot_call is None:
        return BuyWriteResult(
            filled=False,
            reason=REASON_NO_SPOT,
            call=call_leg,
            stock_gain_pct=stock_gain,
            mark_mode=mark_mode,
            events=events,
        )

    if float(spot_call) >= float(call_leg.strike):
        called_away = True
        share_exit = float(call_leg.strike)
        exit_reason = EXIT_CALL_ASSIGN
        events.append(
            {
                "event": "call_assign",
                "day": call_exp.isoformat(),
                "spot": spot_call,
                "strike": call_leg.strike,
            }
        )
    else:
        called_away = False
        share_exit = float(spot_call)
        exit_reason = EXIT_SHARE_LIQ
        events.append(
            {
                "event": "share_liquidate",
                "day": call_exp.isoformat(),
                "spot": spot_call,
            }
        )

    pnl = buy_write_pnl_per_share(
        entry_px=float(entry_px),
        call_credit=call_leg.credit,
        called_away=called_away,
        call_strike=call_leg.strike,
        share_exit_px=None if called_away else share_exit,
    )
    return BuyWriteResult(
        filled=True,
        reason=REASON_OK,
        exit_reason=exit_reason,
        call=call_leg,
        share_exit_px=share_exit,
        called_away=called_away,
        pnl_per_share=pnl,
        gain_pct=stock_notional_pct(pnl, float(entry_px)),
        stock_gain_pct=stock_gain,
        mark_mode=mark_mode,
        hold_days=(call_exp - call_open_day).days,
        exit_date=call_exp,
        events=events,
    )


def result_to_record(row: pd.Series, got: BuyWriteResult) -> Dict[str, Any]:
    rec = row.to_dict() if hasattr(row, "to_dict") else dict(row)
    rec["bw_ok"] = bool(got.filled)
    rec["bw_reason"] = got.reason
    rec["bw_exit_reason"] = got.exit_reason
    rec["bw_mark_mode"] = got.mark_mode
    rec["bw_called_away"] = bool(got.called_away)
    rec["bw_pnl_ps"] = got.pnl_per_share
    rec["bw_call_close_debit"] = got.call_close_debit
    rec["gain_pct_stock"] = got.stock_gain_pct
    rec["gain_pct"] = got.gain_pct if got.filled else None
    rec["bw_hold_days"] = got.hold_days
    rec["bw_exit_date"] = got.exit_date.isoformat() if got.exit_date else None
    if got.call is not None:
        rec["call_strike"] = got.call.strike
        rec["call_credit"] = got.call.credit
        rec["call_expiry"] = got.call.expiration.isoformat()
        rec["call_entry_date"] = got.call.entry_date.isoformat()
        rec["call_delta"] = got.call.entry_delta
        rec["call_dte"] = got.call.entry_dte
        rec["call_source"] = got.call.source
    rec["share_exit_px"] = got.share_exit_px
    return rec


def close_call_debit(
    source: QuoteSource,
    *,
    symbol: str,
    expiration: date,
    strike: float,
    day: date,
    mark_mode: str,
    spot_hint: Optional[float] = None,
) -> Optional[float]:
    """Debit to buy back a short call on ``day`` (None if no quote)."""
    q = source.quote_on_day(symbol, expiration, float(strike), "C", day)
    if q is None:
        # Intrinsic floor when quotes missing (ITM call buyback).
        if spot_hint is not None and math.isfinite(float(spot_hint)):
            intrinsic = max(0.0, float(spot_hint) - float(strike))
            return float(intrinsic)
        return None
    bid = float(q.get("bid") or float("nan"))
    ask = float(q.get("ask") or float("nan"))
    if not (math.isfinite(bid) and math.isfinite(ask)):
        mid = float(q.get("mid") or float("nan"))
        if math.isfinite(mid):
            return float(mid)
        return None
    return float(exit_debit(bid, ask, mark_mode))


def simulate_buy_write_bs(
    row: pd.Series,
    panel_1d: Optional[pd.DataFrame],
    *,
    mark_mode: str = "mid",
    target_dte: int = DEFAULT_DTE,
    target_delta: float = 0.30,
    r: float = DEFAULT_R,
    iv_override: Optional[float] = None,
    ask_bid_friction: float = DEFAULT_BS_FRICTION,
) -> BuyWriteResult:
    """Constant-IV BS covered call when listed chains are unavailable."""
    stock_gain = _num(row, "gain_pct")
    if row.get("gain_pct_stock") is not None and pd.notna(row.get("gain_pct_stock")):
        stock_gain = float(row["gain_pct_stock"])
    entry_px = _num(row, "buy_price")
    entry_day = _as_date(row.get("buy_date"))
    if entry_px is None or entry_px <= 0 or entry_day is None:
        return BuyWriteResult(
            filled=False, reason=REASON_NO_ENTRY, stock_gain_pct=stock_gain
        )
    if panel_1d is None or panel_1d.empty:
        return BuyWriteResult(
            filled=False, reason=REASON_NO_DAILY, stock_gain_pct=stock_gain
        )

    iv = None
    if row.get("name_atm_iv") is not None and pd.notna(row.get("name_atm_iv")):
        try:
            iv = float(row["name_atm_iv"])
        except (TypeError, ValueError):
            iv = None
    if iv is None or iv <= 0:
        iv = resolve_iv(row, float(entry_px), iv_override=iv_override)
    if iv is None or iv <= 0:
        return BuyWriteResult(
            filled=False, reason="bad_iv", stock_gain_pct=stock_gain, mark_mode=mark_mode
        )

    dte_i = max(1, int(target_dte))
    T0 = _years(dte_i)
    strike = strike_for_call_delta(
        float(entry_px), T0, float(r), float(iv), 0.0, target_delta=float(target_delta)
    )
    strike = round_strike(float(strike))
    credit = float(
        black_scholes_call(float(entry_px), float(strike), T0, float(r), float(iv))
    )
    if str(mark_mode).lower() == "ask_bid":
        credit = max(0.0, credit - float(ask_bid_friction))
    if not math.isfinite(credit) or credit < float(DEFAULT_MIN_CREDIT):
        return BuyWriteResult(
            filled=False,
            reason=REASON_NO_CALL_QUOTE,
            stock_gain_pct=stock_gain,
            mark_mode=mark_mode,
        )

    g0 = calculate_greeks(
        float(entry_px), float(strike), T0, float(r), float(iv), "C"
    )
    expiry = _busday_on_or_before(entry_day + timedelta(days=dte_i))
    call_leg = BuyWriteLeg(
        expiration=expiry,
        strike=float(strike),
        credit=float(credit),
        entry_date=entry_day,
        entry_delta=float(g0.get("delta") or float("nan")),
        entry_dte=dte_i,
        source="bs",
    )

    spot_call = _spot_on(panel_1d, expiry)
    if spot_call is None:
        # Last available close on/before expiry
        work = panel_1d.copy()
        work.index = _panel_dates(work)
        before = work.loc[work.index.date <= expiry]
        if before.empty:
            return BuyWriteResult(
                filled=False,
                reason=REASON_NO_SPOT,
                call=call_leg,
                stock_gain_pct=stock_gain,
                mark_mode=mark_mode,
            )
        spot_call = float(before.iloc[-1]["close"])

    if float(spot_call) >= float(call_leg.strike):
        called_away = True
        share_exit = float(call_leg.strike)
        exit_reason = EXIT_CALL_ASSIGN
    else:
        called_away = False
        share_exit = float(spot_call)
        exit_reason = EXIT_SHARE_LIQ

    pnl = buy_write_pnl_per_share(
        entry_px=float(entry_px),
        call_credit=call_leg.credit,
        called_away=called_away,
        call_strike=call_leg.strike,
        share_exit_px=None if called_away else share_exit,
    )
    return BuyWriteResult(
        filled=True,
        reason=REASON_OK,
        exit_reason=exit_reason,
        call=call_leg,
        share_exit_px=share_exit,
        called_away=called_away,
        pnl_per_share=pnl,
        gain_pct=stock_notional_pct(pnl, float(entry_px)),
        stock_gain_pct=stock_gain,
        mark_mode=mark_mode,
        hold_days=(expiry - entry_day).days,
        exit_date=expiry,
        notes="bs_fallback",
        events=[{"event": "open_call_bs", "credit": credit, "iv": iv, "strike": strike}],
    )


def _row_float(row: Mapping[str, Any], *names: str) -> Optional[float]:
    for name in names:
        if name not in row:
            continue
        try:
            v = float(row[name])  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if v == v:
            return v
    return None


def apply_credit_slip(
    row: Mapping[str, Any],
    *,
    slip: float,
) -> Optional[float]:
    """Recompute stock-notional gain% after haircutting the short-call credit."""
    if "bw_ok" in row and not bool(row.get("bw_ok")):
        return None
    entry_px = _row_float(row, "buy_price")
    credit = _row_float(row, "call_credit")
    if entry_px is None or entry_px <= 0 or credit is None:
        return None
    slipped = max(0.0, float(credit) - float(slip))
    called = bool(row.get("bw_called_away"))
    strike = _row_float(row, "call_strike")
    share_exit = _row_float(row, "share_exit_px")
    try:
        pnl = buy_write_pnl_per_share(
            entry_px=float(entry_px),
            call_credit=slipped,
            called_away=called,
            call_strike=strike,
            share_exit_px=None if called else share_exit,
        )
    except ValueError:
        return None
    return stock_notional_pct(pnl, float(entry_px))


def filled_buy_write_to_occ_rows(
    df: pd.DataFrame,
    *,
    slip: float = 0.0,
    gain_col: str = "gain_pct",
) -> List[Dict[str, Any]]:
    """Normalize listed buy-write CSV rows for FCFS occupancy."""
    if df is None or df.empty:
        return []
    work = df.copy()
    if "bw_ok" in work.columns:
        work = work[work["bw_ok"] == True]  # noqa: E712
    rows: List[Dict[str, Any]] = []
    for _, r in work.iterrows():
        entry = r.get("call_entry_date") or r.get("buy_date")
        exit_ = r.get("bw_exit_date") or r.get("call_expiry")
        if entry is None or exit_ is None:
            continue
        if float(slip) > 0:
            gain = apply_credit_slip(r, slip=float(slip))
        else:
            try:
                gain = float(r[gain_col]) if pd.notna(r.get(gain_col)) else None
            except (TypeError, ValueError):
                gain = None
        if gain is None:
            continue
        half = None
        try:
            bid = float(r.get("call_bid") if pd.notna(r.get("call_bid")) else float("nan"))
            ask = float(r.get("call_ask") if pd.notna(r.get("call_ask")) else float("nan"))
            mid = 0.5 * (bid + ask) if math.isfinite(bid) and math.isfinite(ask) else float("nan")
            if math.isfinite(mid) and mid > 0 and math.isfinite(bid) and math.isfinite(ask):
                half = 100.0 * (0.5 * (ask - bid) / mid)
        except (TypeError, ValueError):
            half = None
        # Reconstruct bid/ask from credit if mark was mid (credit=mid) — optional.
        rows.append(
            {
                "symbol": str(r.get("stock") or "").upper(),
                "entry_date": pd.Timestamp(entry).normalize(),
                "exit_date": pd.Timestamp(exit_).normalize(),
                "gain_pct": float(gain),
                "buy_price": float(r["buy_price"]) if pd.notna(r.get("buy_price")) else None,
                "call_strike": (
                    float(r["call_strike"]) if pd.notna(r.get("call_strike")) else None
                ),
                "call_credit": (
                    float(r["call_credit"]) if pd.notna(r.get("call_credit")) else None
                ),
                "call_expiry": (
                    pd.Timestamp(r["call_expiry"]).normalize()
                    if pd.notna(r.get("call_expiry"))
                    else None
                ),
                "call_entry_date": (
                    pd.Timestamp(r["call_entry_date"]).normalize()
                    if pd.notna(r.get("call_entry_date"))
                    else pd.Timestamp(entry).normalize()
                ),
                "adv_20": float(r["adv_20"]) if pd.notna(r.get("adv_20")) else None,
                "half_spread_pct": half,
                "mark_mode": r.get("bw_mark_mode") or r.get("mark_mode"),
                "exit_reason": r.get("bw_exit_reason"),
                "slip": float(slip),
            }
        )
    return rows


def buy_write_unrealized_frac(
    day: date,
    *,
    entry_px: float,
    call_credit: float,
    call_strike: Optional[float],
    call_expiry: date,
    spot: Optional[float],
    call_mark: Optional[float],
) -> float:
    """Unrealized fraction of stock notional: long stock + short call."""
    e = float(entry_px)
    if e <= 0:
        return 0.0
    d = day
    ce = call_expiry
    if spot is None or not math.isfinite(float(spot)):
        return 0.0
    cc = float(call_credit)
    if d >= ce:
        # Settled at expiry marks (caller should use realized on exit day).
        if call_strike is not None and float(spot) >= float(call_strike):
            return (-e + float(call_strike) + cc) / e
        return (-e + float(spot) + cc) / e
    if call_mark is not None and math.isfinite(float(call_mark)):
        cm = float(call_mark)
    elif call_strike is not None:
        cm = max(float(spot) - float(call_strike), 0.0)
    else:
        cm = 0.0
    return (-e + float(spot) + cc - cm) / e


def simulate_buy_write_listed_or_bs(
    row: pd.Series,
    panel_1d: Optional[pd.DataFrame],
    source: Optional[QuoteSource],
    **kwargs: Any,
) -> BuyWriteResult:
    """Try listed Theta/Polygon chain; fall back to constant-IV BS."""
    mark_mode = str(kwargs.get("mark_mode") or "mid")
    if source is not None:
        got = simulate_buy_write(row, panel_1d, source, **kwargs)
        if got.filled:
            return got
    bs_kwargs = {
        "mark_mode": mark_mode,
        "target_dte": kwargs.get("target_dte", DEFAULT_DTE),
        "target_delta": kwargs.get("target_delta", 0.30),
    }
    return simulate_buy_write_bs(row, panel_1d, **bs_kwargs)


__all__ = [
    "BuyWriteLeg",
    "BuyWriteResult",
    "EXIT_CALL_ASSIGN",
    "EXIT_EARNINGS_FLATTEN",
    "EXIT_SHARE_LIQ",
    "apply_credit_slip",
    "buy_write_pnl_per_share",
    "buy_write_unrealized_frac",
    "close_call_debit",
    "filled_buy_write_to_occ_rows",
    "result_to_record",
    "simulate_buy_write",
    "simulate_buy_write_bs",
    "simulate_buy_write_listed_or_bs",
    "stock_notional_pct",
]
