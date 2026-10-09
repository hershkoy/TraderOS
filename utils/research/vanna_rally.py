"""Vanna Rally / vol-crush long call claim-check (Multi Strat Mark).

Source: youtu.be/2GqrWos6sJw — Mark Anderson ("I Backtested a Trade That
Made 24% a Year on 1% of the Account").

Video rules (mechanical translation):
  1. VIX opens below yesterday's close
  2. VIX drops another 1% after the open
  3. S&P is up at least 0.5% from the open
  4. Buy S&P calls 7-10 DTE, roughly 2-3% OTM (video cites ~$7-$10 SPX)
  5. Risk 2-3% of account per trade
  6. Sell the next session before the close
  7. Optional profit-take 700%-1000% (8x-11x entry premium)

Honesty / data limits:
  - Triggers use daily OHLC (Yahoo VIX + SPY). Same-day simultaneity of the
    three intraday conditions is not visible; we require VIX gap-down,
    VIX.low <= 0.99*VIX.open, and SPY.high >= 1.005*SPY.open.
  - Listed marks are SPY OptionsDX EOD (through 2023-12-29) then Theta EOD
    (2024+). SPX $7-$10 maps to ~$0.70-$1.00 on SPY; selection is by % OTM,
    with an optional SPY premium band. SPY spot is TimescaleDB as-traded
    (not Yahoo auto-adjust) so strikes line up with OptionsDX.
  - Entry is signal-day EOD (late vs an intraday arm). Exit is next-session
    EOD; a TP fill is booked only when the exit mark is at/above the multiple
    (intraday TP hits that fade before the close are invisible).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from utils.research.forward_factor import load_spy_daily
from utils.research.metrics import max_drawdown, perf_stats
from utils.research.spy_wheel import HybridSource, QuoteSource
from utils.research.vix_spike_call import load_vix_spot

LOG = logging.getLogger(__name__)

MULT = 100.0
DEFAULT_START_EQUITY = 100_000.0
DEFAULT_RISK_FRAC = 0.025
DEFAULT_OTM = 0.025
DEFAULT_OTM_LO = 0.02
DEFAULT_OTM_HI = 0.03
DEFAULT_DTE_LO = 7
DEFAULT_DTE_HI = 10
DEFAULT_TP_MULT = 8.0  # 700% profit => 8x premium
DEFAULT_COMM_PER_CONTRACT = 0.65
DEFAULT_TBILL_APY = 0.05

CLAIMED: Dict[str, Any] = {
    "source": "youtu.be/2GqrWos6sJw Multi Strat Mark / Mark Anderson",
    "window_claim": "2018 through Sep 2026",
    "cagr": 0.244,
    "mdd": -0.156,
    "spy_cagr": 0.144,
    "spy_mdd": -0.337,
    "win_rate": 0.40,
    "year_2022": 0.18,
    "spy_2022": -0.182,
    "risk_frac": "0.02-0.03",
    "avg_capital_deployed": 0.01,
    "tp": "700%-1000%",
    "structure": "long 7-10 DTE 2-3% OTM S&P calls, sell next day",
}

EXIT_NEXT = "next_close"
EXIT_TP = "tp"
EXIT_NO_MARK = "no_mark"

REASON_OK = "ok"
REASON_NO_CHAIN = "no_chain"
REASON_NO_STRIKE = "no_strike"
REASON_NO_ENTRY = "no_entry_mark"
REASON_NO_EXIT = "no_exit_mark"
REASON_NO_SIZE = "size_zero"
REASON_BUSY = "already_in_trade"
REASON_NO_NEXT = "no_next_session"


@dataclass
class VannaTrade:
    entry_date: date
    exit_date: Optional[date] = None
    expiration: Optional[date] = None
    strike: Optional[float] = None
    dte: Optional[int] = None
    spot: Optional[float] = None
    otm_pct: Optional[float] = None
    contracts: int = 0
    entry_px: Optional[float] = None
    exit_px: Optional[float] = None
    pnl: Optional[float] = None
    ret_on_premium: Optional[float] = None
    ret_on_equity: Optional[float] = None
    exit_reason: Optional[str] = None
    equity_before: Optional[float] = None
    mark_mode: str = "mid"
    skipped: bool = False
    skip_reason: Optional[str] = None
    vix_open: Optional[float] = None
    vix_prev_close: Optional[float] = None
    vix_low: Optional[float] = None
    spy_open: Optional[float] = None
    spy_high: Optional[float] = None


@dataclass
class BookResult:
    label: str
    mark_mode: str
    trades: List[VannaTrade] = field(default_factory=list)
    skipped: List[Dict[str, Any]] = field(default_factory=list)
    equity: pd.Series = field(default_factory=pd.Series)
    spy_equity: pd.Series = field(default_factory=pd.Series)
    notes: str = ""


def _base_mark_mode(mark_mode: str) -> str:
    m = str(mark_mode).strip().lower()
    if m.startswith("mid_slip"):
        return "mid"
    if m in ("natural", "askbid"):
        return "ask_bid"
    return m


def parse_leg_slip(mark_mode: str, leg_slip: float = 0.0) -> float:
    if leg_slip and leg_slip > 0:
        return float(leg_slip)
    m = str(mark_mode).strip().lower()
    if m.startswith("mid_slip"):
        rest = m[len("mid_slip") :]
        return float(rest) if rest else 0.05
    return 0.0


def buy_mark(bid: float, ask: float, mark_mode: str, leg_slip: float = 0.0) -> Optional[float]:
    base = _base_mark_mode(mark_mode)
    if base == "ask_bid":
        px = float(ask)
    else:
        mid = 0.5 * (float(bid) + float(ask))
        px = mid + float(leg_slip or 0.0)
    return px if px > 0 and math.isfinite(px) else None


def sell_mark(bid: float, ask: float, mark_mode: str, leg_slip: float = 0.0) -> Optional[float]:
    base = _base_mark_mode(mark_mode)
    if base == "ask_bid":
        px = float(bid)
    else:
        mid = 0.5 * (float(bid) + float(ask))
        px = mid - float(leg_slip or 0.0)
    return px if px > 0 and math.isfinite(px) else None


def load_spots(start: date, end: date) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """VIX (Yahoo) and SPY daily OHLC for triggers + strike selection.

    SPY must be as-traded (TimescaleDB / load_spy_daily), not Yahoo
    auto-adjusted closes — OptionsDX strikes are unadjusted, and dividend
    back-adjustment makes 2-3% OTM picks look like deep ITM on old years.
    """
    warm = start - timedelta(days=10)
    vix = load_vix_spot(warm, end)
    spy_raw = load_spy_daily(warm, end)
    spy = spy_raw[["open", "high", "low", "close"]].copy()
    spy.index = pd.DatetimeIndex(pd.to_datetime(spy.index).normalize())
    spy = spy[~spy.index.duplicated(keep="last")].sort_index()
    return vix, spy


def find_signal_days(
    vix: pd.DataFrame,
    spy: pd.DataFrame,
    *,
    start: date,
    end: date,
    vix_drop_pct: float = 0.01,
    spy_up_pct: float = 0.005,
) -> pd.DataFrame:
    """Daily proxy for the three video triggers.

    Returns one row per signal with VIX/SPY OHLC fields used by the filter.
    """
    if vix is None or spy is None or vix.empty or spy.empty:
        return pd.DataFrame()

    v = vix.copy()
    s = spy.copy()
    v.index = pd.DatetimeIndex(pd.to_datetime(v.index).normalize())
    s.index = pd.DatetimeIndex(pd.to_datetime(s.index).normalize())
    v = v[~v.index.duplicated(keep="last")].sort_index()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    v["prev_close"] = v["close"].shift(1)

    joined = v.join(s, how="inner", lsuffix="_vix", rsuffix="_spy")
    if joined.empty:
        return pd.DataFrame()

    open_v = joined["open_vix"]
    low_v = joined["low_vix"]
    prev = joined["prev_close"]
    open_s = joined["open_spy"]
    high_s = joined["high_spy"]

    mask = (
        prev.notna()
        & (open_v < prev)
        & (low_v <= open_v * (1.0 - float(vix_drop_pct)))
        & (high_s >= open_s * (1.0 + float(spy_up_pct)))
    )
    out = joined.loc[mask].copy()
    if out.empty:
        return out
    sessions = [pd.Timestamp(ts).date() for ts in out.index]
    out = out[[(start <= d <= end) for d in sessions]].copy()
    if out.empty:
        return out
    out["session"] = [pd.Timestamp(ts).date() for ts in out.index]
    return out


def pick_otm_call(
    chain: pd.DataFrame,
    *,
    day: date,
    spot: float,
    dte_lo: int = DEFAULT_DTE_LO,
    dte_hi: int = DEFAULT_DTE_HI,
    otm: float = DEFAULT_OTM,
    otm_lo: float = DEFAULT_OTM_LO,
    otm_hi: float = DEFAULT_OTM_HI,
    premium_lo: Optional[float] = None,
    premium_hi: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """Pick 7-10 DTE call nearest target % OTM inside the 2-3% band."""
    if chain is None or chain.empty or spot <= 0:
        return None
    df = chain.copy()
    df["dte"] = df["expiration"].map(lambda e: (e - day).days)
    df = df[(df["dte"] >= int(dte_lo)) & (df["dte"] <= int(dte_hi))]
    if df.empty:
        return None

    lo_k = float(spot) * (1.0 + float(otm_lo))
    hi_k = float(spot) * (1.0 + float(otm_hi))
    band = df[(df["strike"] >= lo_k) & (df["strike"] <= hi_k)].copy()
    if band.empty:
        # Fall back to nearest strike at target OTM even if outside tight band
        band = df.copy()

    target_k = float(spot) * (1.0 + float(otm))
    band["mid"] = 0.5 * (band["bid"].astype(float) + band["ask"].astype(float))
    band = band[(band["bid"] > 0) & (band["ask"] > 0) & (band["mid"] > 0)]
    if band.empty:
        return None

    if premium_lo is not None and premium_hi is not None:
        prem = band[
            (band["mid"] >= float(premium_lo)) & (band["mid"] <= float(premium_hi))
        ]
        if not prem.empty:
            band = prem

    # Prefer expiry nearest mid-window, then strike nearest target OTM
    mid_dte = 0.5 * (float(dte_lo) + float(dte_hi))
    band["dte_err"] = (band["dte"] - mid_dte).abs()
    band["strike_err"] = (band["strike"] - target_k).abs()
    row = band.sort_values(["dte_err", "strike_err", "strike"]).iloc[0]
    strike = float(row["strike"])
    return {
        "expiration": row["expiration"],
        "strike": strike,
        "bid": float(row["bid"]),
        "ask": float(row["ask"]),
        "mid": float(row["mid"]),
        "dte": int(row["dte"]),
        "otm_pct": (strike / float(spot)) - 1.0,
        "delta": float(row["delta"]) if "delta" in row and pd.notna(row["delta"]) else None,
    }


def next_session(sessions: Sequence[date], day: date) -> Optional[date]:
    later = [d for d in sessions if d > day]
    return later[0] if later else None


def size_contracts(
    equity: float,
    entry_px: float,
    risk_frac: float = DEFAULT_RISK_FRAC,
) -> int:
    cost = float(entry_px) * MULT
    if cost <= 0 or equity <= 0:
        return 0
    budget = float(equity) * float(risk_frac)
    return int(math.floor(budget / cost))


def simulate_book(
    *,
    source: QuoteSource,
    signals: pd.DataFrame,
    sessions: Sequence[date],
    spot_by_day: Dict[date, float],
    start: date,
    end: date,
    mark_mode: str = "mid",
    leg_slip: float = 0.0,
    risk_frac: float = DEFAULT_RISK_FRAC,
    otm: float = DEFAULT_OTM,
    otm_lo: float = DEFAULT_OTM_LO,
    otm_hi: float = DEFAULT_OTM_HI,
    dte_lo: int = DEFAULT_DTE_LO,
    dte_hi: int = DEFAULT_DTE_HI,
    tp_mult: Optional[float] = DEFAULT_TP_MULT,
    premium_lo: Optional[float] = None,
    premium_hi: Optional[float] = None,
    start_equity: float = DEFAULT_START_EQUITY,
    tbill_apy: float = 0.0,
    comm_per_contract: float = DEFAULT_COMM_PER_CONTRACT,
    label: str = "vanna",
) -> BookResult:
    """Walk signal days; one overnight call at a time; rest of cash optional T-bill."""
    slip = parse_leg_slip(mark_mode, leg_slip)
    result = BookResult(label=label, mark_mode=mark_mode)
    if signals is None or signals.empty:
        result.notes = "no signals"
        return result

    session_list = sorted({d for d in sessions if start <= d <= end})
    if not session_list:
        result.notes = "no sessions"
        return result

    cash = float(start_equity)
    equity_pts: Dict[date, float] = {}
    open_trade: Optional[VannaTrade] = None
    open_path: Optional[pd.DataFrame] = None
    signal_days = [pd.Timestamp(ts).date() for ts in signals.index]
    signal_set = set(signal_days)
    sig_lookup = {pd.Timestamp(ts).date(): signals.loc[ts] for ts in signals.index}

    daily_rf = float(tbill_apy) / 252.0 if tbill_apy and tbill_apy > 0 else 0.0

    for i, day in enumerate(session_list):
        # Accrue idle T-bill on prior cash (before trading)
        if daily_rf > 0 and i > 0:
            cash *= 1.0 + daily_rf

        # Manage open trade: exit on the session after entry
        if open_trade is not None and open_trade.entry_date < day:
            exit_px = None
            if open_path is not None and not open_path.empty:
                row = open_path[open_path["session_date"] == day]
                if not row.empty:
                    bid = float(row.iloc[0]["bid"])
                    ask = float(row.iloc[0]["ask"])
                    exit_px = sell_mark(bid, ask, mark_mode, slip)
            reason = EXIT_NEXT
            if exit_px is None:
                open_trade.exit_date = day
                open_trade.exit_reason = EXIT_NO_MARK
                open_trade.pnl = -float(open_trade.entry_px) * MULT * open_trade.contracts
                open_trade.pnl -= 2.0 * float(comm_per_contract) * open_trade.contracts
                open_trade.ret_on_premium = -1.0
                open_trade.skipped = False
                cash += 0.0  # premium already spent at entry; total loss
                # Actually at entry we deducted debit; on no-mark treat as worthless
                result.trades.append(open_trade)
                open_trade = None
                open_path = None
            else:
                fill = float(exit_px)
                if tp_mult is not None and fill >= float(open_trade.entry_px) * float(tp_mult):
                    fill = float(open_trade.entry_px) * float(tp_mult)
                    reason = EXIT_TP
                proceeds = fill * MULT * open_trade.contracts
                sell_comm = float(comm_per_contract) * open_trade.contracts
                cash += proceeds - sell_comm
                entry_comm = float(comm_per_contract) * open_trade.contracts
                pnl = (fill - float(open_trade.entry_px)) * MULT * open_trade.contracts
                pnl -= entry_comm + sell_comm
                open_trade.exit_date = day
                open_trade.exit_px = fill
                open_trade.exit_reason = reason
                open_trade.pnl = float(pnl)
                open_trade.ret_on_premium = (fill / float(open_trade.entry_px)) - 1.0
                if open_trade.equity_before:
                    open_trade.ret_on_equity = pnl / float(open_trade.equity_before)
                result.trades.append(open_trade)
                open_trade = None
                open_path = None

        # New entry
        if open_trade is None and day in signal_set:
            row = sig_lookup[day]
            spot = float(spot_by_day.get(day) or row.get("close_spy") or row.get("open_spy") or 0.0)
            if spot <= 0:
                result.skipped.append({"day": day.isoformat(), "reason": REASON_NO_ENTRY})
            else:
                chain = source.chain_on_day(day, "C", dte_lo, dte_hi)
                pick = pick_otm_call(
                    chain,
                    day=day,
                    spot=spot,
                    dte_lo=dte_lo,
                    dte_hi=dte_hi,
                    otm=otm,
                    otm_lo=otm_lo,
                    otm_hi=otm_hi,
                    premium_lo=premium_lo,
                    premium_hi=premium_hi,
                )
                if pick is None:
                    result.skipped.append(
                        {
                            "day": day.isoformat(),
                            "reason": REASON_NO_STRIKE if chain is not None and not chain.empty else REASON_NO_CHAIN,
                        }
                    )
                else:
                    entry_px = buy_mark(pick["bid"], pick["ask"], mark_mode, slip)
                    if entry_px is None:
                        result.skipped.append({"day": day.isoformat(), "reason": REASON_NO_ENTRY})
                    else:
                        nxt = next_session(session_list, day)
                        if nxt is None:
                            result.skipped.append({"day": day.isoformat(), "reason": REASON_NO_NEXT})
                        else:
                            n = size_contracts(cash, entry_px, risk_frac)
                            if n <= 0:
                                result.skipped.append({"day": day.isoformat(), "reason": REASON_NO_SIZE})
                            else:
                                debit = float(entry_px) * MULT * n
                                buy_comm = float(comm_per_contract) * n
                                if debit + buy_comm > cash:
                                    result.skipped.append(
                                        {"day": day.isoformat(), "reason": REASON_NO_SIZE}
                                    )
                                else:
                                    cash -= debit + buy_comm
                                    path = source.quote_path(
                                        pick["expiration"],
                                        pick["strike"],
                                        "C",
                                        day,
                                        nxt,
                                    )
                                    open_path = path
                                    open_trade = VannaTrade(
                                        entry_date=day,
                                        expiration=pick["expiration"],
                                        strike=pick["strike"],
                                        dte=pick["dte"],
                                        spot=spot,
                                        otm_pct=pick.get("otm_pct"),
                                        contracts=n,
                                        entry_px=float(entry_px),
                                        equity_before=cash + debit + buy_comm,
                                        mark_mode=mark_mode,
                                        vix_open=float(row.get("open_vix") or 0) or None,
                                        vix_prev_close=float(row.get("prev_close") or 0) or None,
                                        vix_low=float(row.get("low_vix") or 0) or None,
                                        spy_open=float(row.get("open_spy") or 0) or None,
                                        spy_high=float(row.get("high_spy") or 0) or None,
                                    )

        # Mark-to-market equity
        mtm = cash
        if open_trade is not None and open_path is not None and not open_path.empty:
            rowp = open_path[open_path["session_date"] == day]
            if not rowp.empty:
                bid = float(rowp.iloc[0]["bid"])
                ask = float(rowp.iloc[0]["ask"])
                px = sell_mark(bid, ask, mark_mode, slip)
                if px is not None:
                    mtm = cash + float(px) * MULT * open_trade.contracts
            elif day == open_trade.entry_date and open_trade.entry_px:
                # Just entered; mark at entry
                mtm = cash + float(open_trade.entry_px) * MULT * open_trade.contracts
        equity_pts[day] = mtm

    # Force-close leftover at last mark
    if open_trade is not None:
        open_trade.exit_date = session_list[-1]
        open_trade.exit_reason = EXIT_NO_MARK
        open_trade.pnl = -float(open_trade.entry_px or 0) * MULT * open_trade.contracts
        open_trade.pnl -= float(comm_per_contract) * open_trade.contracts
        result.trades.append(open_trade)

    if equity_pts:
        eq = pd.Series(equity_pts).sort_index()
        eq.index = pd.DatetimeIndex(pd.to_datetime(eq.index))
        result.equity = eq
        # SPY buy & hold on same calendar
        spy_closes = pd.Series(
            {pd.Timestamp(d): float(spot_by_day[d]) for d in session_list if d in spot_by_day}
        ).sort_index()
        if not spy_closes.empty:
            result.spy_equity = spy_closes / float(spy_closes.iloc[0]) * float(start_equity)

    result.notes = (
        f"signals={len(signal_days)} fills={len(result.trades)} "
        f"skipped={len(result.skipped)} slip={slip} tp={tp_mult} risk={risk_frac}"
    )
    return result


def book_stats(
    book: BookResult,
    *,
    start_equity: float = DEFAULT_START_EQUITY,
) -> Dict[str, Any]:
    closed = [t for t in book.trades if t.pnl is not None and not t.skipped]
    n = len(closed)
    pnls = [float(t.pnl) for t in closed]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    gross_win = float(sum(wins)) if wins else 0.0
    gross_loss = float(-sum(losses)) if losses else 0.0
    pf = (gross_win / gross_loss) if gross_loss > 1e-12 else (float("inf") if gross_win > 0 else 0.0)
    wr = (len(wins) / n * 100.0) if n else 0.0
    rets = [float(t.ret_on_premium) for t in closed if t.ret_on_premium is not None]

    # Max consecutive losses
    streak = max_streak = 0
    for p in pnls:
        if p < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    years: Dict[str, Dict[str, Any]] = {}
    for t in closed:
        y = str(t.entry_date.year)
        bucket = years.setdefault(y, {"n": 0, "pnl": 0.0, "wins": 0})
        bucket["n"] += 1
        bucket["pnl"] += float(t.pnl or 0)
        if (t.pnl or 0) > 0:
            bucket["wins"] += 1

    out: Dict[str, Any] = {
        "label": book.label,
        "mark_mode": book.mark_mode,
        "n": n,
        "win_rate_pct": round(wr, 2),
        "pf": round(pf, 3) if math.isfinite(pf) else None,
        "total_pnl": round(float(sum(pnls)), 2) if pnls else 0.0,
        "mean_pnl": round(float(np.mean(pnls)), 2) if pnls else 0.0,
        "med_pnl": round(float(np.median(pnls)), 2) if pnls else 0.0,
        "mean_ret_on_prem_pct": round(float(np.mean(rets)) * 100.0, 2) if rets else 0.0,
        "med_ret_on_prem_pct": round(float(np.median(rets)) * 100.0, 2) if rets else 0.0,
        "max_consec_losses": max_streak,
        "exits": _count_exits(closed),
        "years": {
            y: {
                "n": v["n"],
                "pnl": round(v["pnl"], 2),
                "wr_pct": round(v["wins"] / v["n"] * 100.0, 1) if v["n"] else 0.0,
            }
            for y, v in sorted(years.items())
        },
        "skipped": len(book.skipped),
        "skip_reasons": _count_skip_reasons(book.skipped),
        "notes": book.notes,
        "claimed": CLAIMED,
    }

    eq = book.equity.dropna()
    if len(eq) >= 2:
        # Normalize to start_equity for perf_stats
        eq_n = eq / float(eq.iloc[0]) * float(start_equity)
        spy = book.spy_equity.dropna()
        spy_stats = perf_stats("SPY", spy) if len(spy) >= 2 else None
        st = perf_stats(
            book.label,
            eq_n,
            spy_cagr=spy_stats.cagr if spy_stats else None,
            spy_sharpe=spy_stats.sharpe if spy_stats else None,
            trade_rets=[p / start_equity for p in pnls],
        )
        out["cagr"] = st.cagr
        out["sharpe"] = st.sharpe
        out["max_drawdown"] = st.max_drawdown
        out["total_return"] = st.total_return
        out["end_equity"] = round(float(eq.iloc[-1]), 2)
        if spy_stats:
            out["spy"] = {
                "cagr": spy_stats.cagr,
                "sharpe": spy_stats.sharpe,
                "max_drawdown": spy_stats.max_drawdown,
                "total_return": spy_stats.total_return,
            }
            out["mdd_vs_spy"] = st.max_drawdown - spy_stats.max_drawdown
        # Top-10 contribution
        if pnls:
            ranked = sorted(pnls, reverse=True)
            top10 = ranked[:10]
            pos = sum(p for p in pnls if p > 0)
            out["top10_pnl"] = round(float(sum(top10)), 2)
            out["top10_share_of_wins"] = (
                round(float(sum(top10)) / pos, 3) if pos > 1e-9 else None
            )
        out["mdd_curve"] = round(max_drawdown(eq_n), 4)
    else:
        out["end_equity"] = round(float(start_equity), 2)

    # Calendar year returns from equity curve
    if len(eq) >= 2:
        yr = eq.pct_change().fillna(0.0)
        ann = (1.0 + yr).groupby(yr.index.year).prod() - 1.0
        out["calendar_returns"] = {str(int(y)): round(float(r), 4) for y, r in ann.items()}

    return out


def _count_exits(trades: Sequence[VannaTrade]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for t in trades:
        r = str(t.exit_reason or "unknown")
        out[r] = out.get(r, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def _count_skip_reasons(skipped: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for s in skipped:
        r = str(s.get("reason") or "unknown")
        out[r] = out.get(r, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def trades_to_frame(trades: Sequence[VannaTrade]) -> pd.DataFrame:
    rows = []
    for t in trades:
        rows.append(
            {
                "entry_date": t.entry_date,
                "exit_date": t.exit_date,
                "expiration": t.expiration,
                "strike": t.strike,
                "dte": t.dte,
                "spot": t.spot,
                "otm_pct": t.otm_pct,
                "contracts": t.contracts,
                "entry_px": t.entry_px,
                "exit_px": t.exit_px,
                "pnl": t.pnl,
                "ret_on_premium": t.ret_on_premium,
                "ret_on_equity": t.ret_on_equity,
                "exit_reason": t.exit_reason,
                "mark_mode": t.mark_mode,
                "vix_open": t.vix_open,
                "vix_prev_close": t.vix_prev_close,
                "vix_low": t.vix_low,
                "spy_open": t.spy_open,
                "spy_high": t.spy_high,
            }
        )
    return pd.DataFrame(rows)


def spot_map_from_ohlc(spy: pd.DataFrame) -> Dict[date, float]:
    out: Dict[date, float] = {}
    if spy is None or spy.empty:
        return out
    for ts, row in spy.iterrows():
        d = pd.Timestamp(ts).date()
        out[d] = float(row["close"])
    return out


def sessions_from_ohlc(spy: pd.DataFrame, start: date, end: date) -> List[date]:
    if spy is None or spy.empty:
        return []
    days = [pd.Timestamp(ts).date() for ts in spy.index]
    return sorted(d for d in days if start <= d <= end)


def default_source(cache_dir=None) -> HybridSource:
    from pathlib import Path

    from utils.research.spy_wheel import ThetaSource

    theta = ThetaSource(cache_dir=Path(cache_dir) if cache_dir else None)
    return HybridSource(theta=theta)
