"""SPY Friday 7 DTE put credit spread claim-check (OptionsDX EOD + Theta 2024+).

Video: Nathan / optionsforall.com "Best Put Credit Spread" (HbV4sTEyoBM).

Claimed winner (mid, no commissions, Fri open, 7 DTE):
  - Short put ~40 delta, long put $5 lower
  - Take profit 25% or 50% of max credit
  - Stop loss 300% of credit (debit >= 4x credit; loss = 3x credit)
  - Window 2016-01-01 through 2025-12-31
  - Claimed: TP25 CAGR ~23.8% MDD 52.5% Sharpe 0.74 WR 85% n~519;
    TP50 CAGR ~23% MDD 30.7% Sharpe 0.73 WR 80%

Data: OptionsDX SPY EOD through 2023-12-29, Theta EOD from 2024-01-02.
EOD marks only — an intraday TP/SL touch that reverts before the close is missed.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from utils.data.ohlcv_loader import load_ohlcv_many
from utils.research.metrics import max_drawdown, passes_phase5_gates, perf_stats
from utils.research.spy_wheel import (
    HybridSource,
    ODX_END,
    OptionsDXSource,
    QuoteSource,
    THETA_START,
    ThetaSource,
)

LOG = logging.getLogger(__name__)

SYMBOL = "SPY"
MULT = 100.0

CLAIMED: Dict[str, Any] = {
    "source": "youtu.be/HbV4sTEyoBM Nathan optionsforall SPY Fri 7 DTE PCS",
    "window_claim": "2016-01-01 through 2025-12-31",
    "structure": "short ~40d put / long $5 lower; open Friday (or next session)",
    "dte": 7,
    "tp": "25% or 50% of credit (debit <= 0.75x or 0.50x credit)",
    "sl": "300% of credit lost (debit >= 4x credit)",
    "fill": "midpoint, no commissions",
    "tp25_claim": {
        "return_pct": 747.9,
        "cagr": 0.238,
        "mdd": 0.525,
        "sharpe": 0.74,
        "win_rate": 0.85,
        "n": 519,
    },
    "tp50_claim": {
        "return_pct": 690.8,
        "cagr": 0.23,
        "mdd": 0.307,
        "sharpe": 0.73,
        "win_rate": 0.803,
        "n": 519,
    },
}

DEFAULT_TARGET_DELTA = 0.40
DEFAULT_WIDTH = 5.0
DEFAULT_DTE = 7
DEFAULT_DTE_TOL = 2
DEFAULT_TP_PROFIT_FRAC = 0.25
DEFAULT_SL_MULT = 4.0  # debit >= 4x credit == 300% of credit lost
DEFAULT_MIN_CREDIT = 0.05
DEFAULT_MARGIN_BUFFER = 1.5

EXIT_TP = "tp"
EXIT_SL = "sl"
EXIT_EXPIRY = "expiry"
EXIT_NO_MARK = "no_mark"

REASON_OK = "ok"
REASON_NO_CHAIN = "no_chain"
REASON_NO_EXP = "no_expiry"
REASON_NO_DELTA = "no_delta"
REASON_NO_WING = "no_wing"
REASON_NO_CREDIT = "credit_below_min"
REASON_BUSY = "max_concurrent"
REASON_NO_SPOT = "no_spot"


def load_spy_daily_ib(start: date, end: date) -> pd.DataFrame:
    """SPY 1d from IB only.

    ``load_spy_daily`` (Alpaca + IB prefix) drops IB after the first Alpaca bar.
    Alpaca SPY in this DB has multi-month holes (all of 2019), which zeroed Friday
    entries for that year. IB covers the OptionsDX window continuously.
    """
    from datetime import datetime

    warm = start - timedelta(days=40)
    frames = load_ohlcv_many(
        ["SPY"],
        timeframe="1d",
        provider="IB",
        start=datetime(warm.year, warm.month, warm.day),
        end=datetime(end.year, end.month, end.day),
        use_cache=True,
        workers=1,
    )
    raw = frames.get("SPY")
    if raw is None or raw.empty:
        raise RuntimeError("No IB daily bars for SPY")
    df = raw.copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df.sort_index()


def _base_mark_mode(mark_mode: str) -> str:
    m = str(mark_mode).strip().lower()
    if m in ("natural", "askbid", "ask_bid"):
        return "ask_bid"
    if m.startswith("mid_slip"):
        return "mid"
    return "mid"


def parse_package_slip(mark_mode: str, package_slip: float = 0.0) -> float:
    """Package slip in index points. ``mid_slip0.05`` embeds the slip in mark_mode."""
    if package_slip and float(package_slip) > 0:
        return float(package_slip)
    m = str(mark_mode).strip().lower()
    if m.startswith("mid_slip"):
        rest = m[len("mid_slip") :]
        return float(rest) if rest else 0.05
    return 0.0


def friday_entry_days(sessions: Sequence[date], start: date, end: date) -> List[date]:
    """Each calendar Friday in range, or the next session if Friday is closed."""
    sess = sorted(d for d in sessions if start <= d <= end)
    if not sess:
        return []
    sess_set = set(sess)
    out: List[date] = []
    # Walk calendar Fridays from start..end
    d = start
    # advance to first Friday on/after start
    while d.weekday() != 4:
        d += timedelta(days=1)
    while d <= end:
        if d in sess_set:
            out.append(d)
        else:
            # next session after closed Friday
            nxt = next((s for s in sess if s > d), None)
            if nxt is not None and nxt <= end:
                # avoid duplicate if previous Friday also rolled into same Monday
                if not out or out[-1] != nxt:
                    out.append(nxt)
        d += timedelta(days=7)
    return out


def vertical_credit(
    short_bid: float,
    short_ask: float,
    long_bid: float,
    long_ask: float,
    mark_mode: str = "mid",
    package_slip: float = 0.0,
) -> Optional[float]:
    """Open credit. Mid subtracts package slip, clamped so credit is not worse than natural."""
    if min(short_bid, short_ask, long_bid, long_ask) <= 0:
        return None
    if short_ask + 1e-9 < short_bid or long_ask + 1e-9 < long_bid:
        return None
    natural = float(short_bid) - float(long_ask)
    if _base_mark_mode(mark_mode) == "ask_bid":
        return natural
    s_mid = 0.5 * (float(short_bid) + float(short_ask))
    l_mid = 0.5 * (float(long_bid) + float(long_ask))
    mid = s_mid - l_mid
    slip = parse_package_slip(mark_mode, package_slip)
    if slip <= 0:
        return mid
    # Worse credit is lower; never cross below the natural bid/ask package.
    return max(mid - slip, natural)


def vertical_debit(
    short_bid: float,
    short_ask: float,
    long_bid: float,
    long_ask: float,
    mark_mode: str = "mid",
    package_slip: float = 0.0,
) -> Optional[float]:
    """Close debit. Mid adds package slip, clamped so debit is not worse than natural."""
    if min(short_bid, short_ask, long_bid, long_ask) <= 0:
        return None
    if short_ask + 1e-9 < short_bid or long_ask + 1e-9 < long_bid:
        return None
    natural = float(short_ask) - float(long_bid)
    if _base_mark_mode(mark_mode) == "ask_bid":
        return natural
    s_mid = 0.5 * (float(short_bid) + float(short_ask))
    l_mid = 0.5 * (float(long_bid) + float(long_ask))
    mid = s_mid - l_mid
    slip = parse_package_slip(mark_mode, package_slip)
    if slip <= 0:
        return mid
    # Worse debit is higher; never cross above the natural ask/bid package.
    return min(mid + slip, natural)


def intrinsic_spread_value(spot: float, short_k: float, long_k: float) -> float:
    short_intr = max(float(short_k) - float(spot), 0.0)
    long_intr = max(float(long_k) - float(spot), 0.0)
    width = float(short_k) - float(long_k)
    return float(min(max(short_intr - long_intr, 0.0), max(width, 0.0)))


def pick_pcs(
    chain: pd.DataFrame,
    *,
    day: date,
    spot: float,
    target_dte: int = DEFAULT_DTE,
    dte_tol: int = DEFAULT_DTE_TOL,
    target_delta: float = DEFAULT_TARGET_DELTA,
    width: float = DEFAULT_WIDTH,
    min_credit: float = DEFAULT_MIN_CREDIT,
    mark_mode: str = "mid",
    package_slip: float = 0.0,
) -> Tuple[Optional[Dict[str, Any]], str]:
    if chain is None or chain.empty or spot <= 0:
        return None, REASON_NO_CHAIN
    df = chain.copy()
    df["dte"] = df["expiration"].map(lambda e: (e - day).days)
    df = df[
        (df["dte"] >= int(target_dte) - int(dte_tol))
        & (df["dte"] <= int(target_dte) + int(dte_tol))
    ]
    if df.empty:
        return None, REASON_NO_EXP
    best_err = int(df["dte"].sub(int(target_dte)).abs().min())
    df = df[df["dte"].sub(int(target_dte)).abs() == best_err]
    exp = df["expiration"].iloc[0]
    sub = df[df["expiration"] == exp].copy()
    sub = sub[sub["strike"] < float(spot)]
    if sub.empty:
        return None, REASON_NO_DELTA
    sub["abs_delta"] = sub["delta"].abs()
    sub["delta_err"] = (sub["abs_delta"] - abs(float(target_delta))).abs()
    row = sub.sort_values(["delta_err", "strike"], ascending=[True, False]).iloc[0]
    short_k = float(row["strike"])
    long_k = short_k - float(width)
    wing = sub[np.isclose(sub["strike"].astype(float), long_k, atol=0.02)]
    if wing.empty:
        # wing may be on same expiry but filtered by spot; look in full expiry chain
        full = df[df["expiration"] == exp]
        wing = full[np.isclose(full["strike"].astype(float), long_k, atol=0.02)]
    if wing.empty:
        return None, REASON_NO_WING
    wrow = wing.iloc[0]
    credit = vertical_credit(
        float(row["bid"]),
        float(row["ask"]),
        float(wrow["bid"]),
        float(wrow["ask"]),
        mark_mode,
        package_slip=package_slip,
    )
    if credit is None or credit < float(min_credit):
        return None, REASON_NO_CREDIT
    actual_width = short_k - float(wrow["strike"])
    return {
        "expiration": exp,
        "short_k": short_k,
        "long_k": float(wrow["strike"]),
        "width": float(actual_width),
        "delta": float(row["delta"]),
        "abs_delta": float(row["abs_delta"]),
        "dte": int(row["dte"]),
        "short_bid": float(row["bid"]),
        "short_ask": float(row["ask"]),
        "long_bid": float(wrow["bid"]),
        "long_ask": float(wrow["ask"]),
        "credit": float(credit),
    }, REASON_OK


@dataclass
class PCSTrade:
    entry_date: date
    exit_date: Optional[date] = None
    expiration: Optional[date] = None
    spot: Optional[float] = None
    short_k: Optional[float] = None
    long_k: Optional[float] = None
    width: float = DEFAULT_WIDTH
    entry_delta: Optional[float] = None
    credit: Optional[float] = None
    exit_debit: Optional[float] = None
    defined_risk: Optional[float] = None
    pnl: Optional[float] = None
    ror_pct: Optional[float] = None
    hold_days: Optional[int] = None
    exit_reason: Optional[str] = None
    mark_mode: str = "mid"
    dte_entry: Optional[int] = None
    skipped: bool = False
    skip_reason: Optional[str] = None


@dataclass
class BookResult:
    label: str
    mark_mode: str
    tp_profit_frac: float
    sl_mult: float
    trades: List[PCSTrade] = field(default_factory=list)
    skipped: List[Dict[str, Any]] = field(default_factory=list)
    equity: pd.Series = field(default_factory=pd.Series)
    spy_equity: pd.Series = field(default_factory=pd.Series)
    capital: float = 0.0
    notes: str = ""


def _year_stats(trades: Sequence[PCSTrade]) -> Dict[str, Any]:
    buckets: Dict[int, List[PCSTrade]] = {}
    for t in trades:
        if t.skipped or t.pnl is None:
            continue
        buckets.setdefault(int(t.entry_date.year), []).append(t)
    out: Dict[str, Any] = {}
    for year in sorted(buckets):
        group = buckets[year]
        pnls = np.array([float(t.pnl or 0.0) for t in group], dtype=float)
        wins = pnls[pnls > 0]
        losses = pnls[pnls <= 0]
        gw = float(wins.sum()) if len(wins) else 0.0
        gl = float(-losses.sum()) if len(losses) else 0.0
        pf = (gw / gl) if gl > 1e-9 else None
        out[str(year)] = {
            "n": len(group),
            "win_rate_pct": round(100.0 * float((pnls > 0).mean()), 1),
            "mean_ror_pct": round(float(np.mean([t.ror_pct or 0.0 for t in group])), 2),
            "pf": None if pf is None else round(pf, 2),
            "pnl": round(float(pnls.sum()), 2),
        }
    return out


def book_stats(
    trades: Sequence[PCSTrade],
    *,
    capital: float,
    equity: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    kept = [t for t in trades if not t.skipped and t.pnl is not None and t.exit_date is not None]
    if not kept:
        return {"n": 0}
    pnls = np.array([float(t.pnl) for t in kept], dtype=float)
    rors = np.array([float(t.ror_pct or 0.0) for t in kept], dtype=float)
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    gw = float(wins.sum()) if len(wins) else 0.0
    gl = float(-losses.sum()) if len(losses) else 0.0
    pf = (gw / gl) if gl > 1e-9 else None
    hold = float(np.mean([t.hold_days or 0 for t in kept])) or 7.0
    n_per_year = 365.0 / hold
    sharpe_proxy = None
    if len(rors) >= 5 and float(rors.std()) > 1e-12:
        sharpe_proxy = float((rors.mean() / rors.std()) * math.sqrt(n_per_year))
    reasons: Dict[str, int] = {}
    for t in kept:
        key = str(t.exit_reason or "unknown")
        reasons[key] = reasons.get(key, 0) + 1
    total_pnl = float(pnls.sum())
    ret_on_cap = (total_pnl / float(capital)) if capital > 0 else None
    mdd_eq = None
    cagr = None
    sharpe_eq = None
    if equity is not None and len(equity.dropna()) >= 2:
        eq = equity.dropna()
        mdd_eq = float(abs(max_drawdown(eq)))
        years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
        cagr = float((eq.iloc[-1] / eq.iloc[0]) ** (1.0 / years) - 1.0) if eq.iloc[0] > 0 else None
        rets = eq.pct_change().dropna()
        vol = float(rets.std(ddof=0) * math.sqrt(252)) if len(rets) else 0.0
        sharpe_eq = float(cagr / vol) if cagr is not None and vol > 1e-12 else None
    return {
        "n": len(kept),
        "win_rate_pct": round(100.0 * float((pnls > 0).mean()), 1),
        "mean_ror_pct": round(float(rors.mean()), 2),
        "med_ror_pct": round(float(np.median(rors)), 2),
        "mean_pnl": round(float(pnls.mean()), 2),
        "total_pnl": round(total_pnl, 2),
        "pf": None if pf is None else round(pf, 2),
        "avg_hold": round(hold, 1),
        "avg_credit": round(float(np.mean([t.credit for t in kept if t.credit])), 3),
        "avg_delta": round(
            float(np.mean([abs(t.entry_delta or 0) for t in kept])), 3
        ),
        "avg_defined_risk": round(
            float(np.mean([t.defined_risk for t in kept if t.defined_risk])), 2
        ),
        "capital": round(float(capital), 2),
        "return_on_capital_pct": None
        if ret_on_cap is None
        else round(100.0 * ret_on_cap, 1),
        "cagr": None if cagr is None else round(cagr, 4),
        "mdd_pct": None if mdd_eq is None else round(100.0 * mdd_eq, 1),
        "sharpe": None if sharpe_eq is None else round(sharpe_eq, 2),
        "sharpe_proxy_ror": None if sharpe_proxy is None else round(sharpe_proxy, 2),
        "exit_reasons": reasons,
        "years": _year_stats(kept),
    }


def trades_to_frame(trades: Sequence[PCSTrade]) -> pd.DataFrame:
    rows = []
    for t in trades:
        rows.append(
            {
                "entry_date": t.entry_date.isoformat() if t.entry_date else None,
                "exit_date": t.exit_date.isoformat() if t.exit_date else None,
                "expiration": t.expiration.isoformat() if t.expiration else None,
                "spot": t.spot,
                "short_k": t.short_k,
                "long_k": t.long_k,
                "width": t.width,
                "entry_delta": t.entry_delta,
                "credit": t.credit,
                "exit_debit": t.exit_debit,
                "defined_risk": t.defined_risk,
                "pnl": t.pnl,
                "ror_pct": t.ror_pct,
                "hold_days": t.hold_days,
                "exit_reason": t.exit_reason,
                "mark_mode": t.mark_mode,
                "dte_entry": t.dte_entry,
                "skipped": t.skipped,
                "skip_reason": t.skip_reason,
            }
        )
    return pd.DataFrame(rows)


def run_friday_pcs_book(
    *,
    start: date,
    end: date,
    source: QuoteSource,
    mark_mode: str = "mid",
    package_slip: float = 0.0,
    target_delta: float = DEFAULT_TARGET_DELTA,
    width: float = DEFAULT_WIDTH,
    target_dte: int = DEFAULT_DTE,
    dte_tol: int = DEFAULT_DTE_TOL,
    tp_profit_frac: float = DEFAULT_TP_PROFIT_FRAC,
    sl_mult: float = DEFAULT_SL_MULT,
    use_stop: bool = True,
    min_credit: float = DEFAULT_MIN_CREDIT,
    margin_buffer: float = DEFAULT_MARGIN_BUFFER,
    max_concurrent: int = 1,
    spots: Optional[pd.DataFrame] = None,
    label: Optional[str] = None,
) -> BookResult:
    """One-lot Friday 7 DTE SPY put credit spreads with TP/SL management."""
    warm = start - timedelta(days=10)
    spots = spots if spots is not None else load_spy_daily_ib(warm, end + timedelta(days=14))
    if spots is None or spots.empty:
        raise RuntimeError("No SPY daily bars")
    close = spots["close"].astype(float).copy()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    close = close.sort_index()
    close_by_day: Dict[date, float] = {
        ts.date(): float(px) for ts, px in close.items()
    }
    sessions = sorted(close_by_day)
    entries = friday_entry_days(sessions, start, end)

    slip = parse_package_slip(mark_mode, package_slip)
    mark_tag = _base_mark_mode(mark_mode)
    if slip > 0 and mark_tag == "mid":
        mark_tag = f"mid_slip{slip:g}"
    tag = label or (
        f"PCS_d{int(round(target_delta*100))}_w{width:g}_"
        f"tp{int(round(tp_profit_frac*100))}_sl{int(round(sl_mult*100))}pct_"
        f"{mark_tag}"
    )
    # sl label: sl_mult 4.0 => 300% loss of credit
    result = BookResult(
        label=tag,
        mark_mode=mark_tag if slip > 0 else mark_mode,
        tp_profit_frac=float(tp_profit_frac),
        sl_mult=float(sl_mult),
    )

    open_until: List[date] = []  # exit dates of still-blocking positions
    path_cache: Dict[Tuple[Any, ...], Dict[date, Tuple[float, float]]] = {}

    def _path(exp: date, strike: float, from_day: date) -> Dict[date, Tuple[float, float]]:
        key = (exp, float(strike), from_day)
        if key in path_cache:
            return path_cache[key]
        df = source.quote_path(exp, float(strike), "P", from_day, exp + timedelta(days=2))
        mp: Dict[date, Tuple[float, float]] = {}
        if df is not None and not df.empty:
            for _, r in df.iterrows():
                mp[r["session_date"]] = (float(r["bid"]), float(r["ask"]))
        path_cache[key] = mp
        return mp

    def _debit_on(
        day: date,
        exp: date,
        short_k: float,
        long_k: float,
        entry_day: date,
    ) -> Optional[float]:
        sp = _path(exp, short_k, entry_day)
        lp = _path(exp, long_k, entry_day)
        if day in sp and day in lp:
            sb, sa = sp[day]
            lb, la = lp[day]
            return vertical_debit(sb, sa, lb, la, mark_mode, package_slip=slip)
        if day >= exp and day in close_by_day:
            return intrinsic_spread_value(close_by_day[day], short_k, long_k)
        return None

    for entry in entries:
        open_until = [x for x in open_until if x > entry]
        if len(open_until) >= int(max_concurrent):
            result.skipped.append({"date": entry.isoformat(), "reason": REASON_BUSY})
            continue
        spot = close_by_day.get(entry)
        if spot is None:
            result.skipped.append({"date": entry.isoformat(), "reason": REASON_NO_SPOT})
            continue
        chain = source.chain_on_day(
            entry,
            "P",
            int(target_dte) - int(dte_tol),
            int(target_dte) + int(dte_tol),
        )
        picked, reason = pick_pcs(
            chain,
            day=entry,
            spot=float(spot),
            target_dte=target_dte,
            dte_tol=dte_tol,
            target_delta=target_delta,
            width=width,
            min_credit=min_credit,
            mark_mode=mark_mode,
            package_slip=slip,
        )
        if picked is None:
            result.skipped.append({"date": entry.isoformat(), "reason": reason})
            continue

        credit = float(picked["credit"])
        short_k = float(picked["short_k"])
        long_k = float(picked["long_k"])
        exp = picked["expiration"]
        actual_width = float(picked["width"])
        defined_risk = max(actual_width - credit, 1e-6) * MULT
        tp_debit = credit * (1.0 - float(tp_profit_frac))
        sl_debit = credit * float(sl_mult)

        manage_days = [d for d in sessions if entry < d <= exp]
        exit_day: Optional[date] = None
        exit_debit: Optional[float] = None
        exit_reason: Optional[str] = None
        for cur in manage_days:
            debit = _debit_on(cur, exp, short_k, long_k, entry)
            if debit is None or not math.isfinite(float(debit)):
                continue
            debit_f = float(debit)
            if use_stop and debit_f >= sl_debit:
                exit_day, exit_debit, exit_reason = cur, debit_f, EXIT_SL
                break
            if debit_f <= tp_debit:
                exit_day, exit_debit, exit_reason = cur, debit_f, EXIT_TP
                break
            if cur == exp:
                exit_day, exit_debit, exit_reason = cur, debit_f, EXIT_EXPIRY
                break
        if exit_day is None:
            if exp in close_by_day:
                exit_day = exp
                exit_debit = intrinsic_spread_value(close_by_day[exp], short_k, long_k)
                exit_reason = EXIT_EXPIRY
            else:
                result.trades.append(
                    PCSTrade(
                        entry_date=entry,
                        expiration=exp,
                        spot=float(spot),
                        short_k=short_k,
                        long_k=long_k,
                        width=actual_width,
                        entry_delta=float(picked["delta"]),
                        credit=credit,
                        defined_risk=defined_risk,
                        mark_mode=mark_mode,
                        dte_entry=int(picked["dte"]),
                        skipped=True,
                        skip_reason=EXIT_NO_MARK,
                    )
                )
                continue

        pnl = (credit - float(exit_debit)) * MULT
        ror = 100.0 * pnl / defined_risk
        hold = (exit_day - entry).days
        result.trades.append(
            PCSTrade(
                entry_date=entry,
                exit_date=exit_day,
                expiration=exp,
                spot=float(spot),
                short_k=short_k,
                long_k=long_k,
                width=actual_width,
                entry_delta=float(picked["delta"]),
                credit=credit,
                exit_debit=float(exit_debit),
                defined_risk=defined_risk,
                pnl=float(pnl),
                ror_pct=float(ror),
                hold_days=int(hold),
                exit_reason=exit_reason,
                mark_mode=mark_mode,
                dte_entry=int(picked["dte"]),
            )
        )
        open_until.append(exit_day)

    kept = [t for t in result.trades if not t.skipped and t.pnl is not None]
    peak_margin = max((t.defined_risk or 0.0) for t in kept) if kept else 0.0
    capital = float(peak_margin) * float(margin_buffer)
    if capital <= 0:
        capital = 1000.0
    result.capital = capital

    # Daily equity through last exit (a Fri entry near `end` can expire after `end`).
    last_exit = max((t.exit_date for t in kept if t.exit_date is not None), default=end)
    eq_end = max(end, last_exit)
    days = [d for d in sessions if start <= d <= eq_end]
    if last_exit not in set(days) and last_exit in close_by_day:
        days = sorted(set(days) | {last_exit})
    eq_map: Dict[date, float] = {}
    cash = float(capital)
    pnl_by_exit: Dict[date, float] = {}
    for t in kept:
        if t.exit_date is None:
            continue
        pnl_by_exit[t.exit_date] = pnl_by_exit.get(t.exit_date, 0.0) + float(t.pnl or 0.0)
    for d in days:
        if d in pnl_by_exit:
            cash += pnl_by_exit[d]
        eq_map[d] = cash
    if eq_map:
        result.equity = pd.Series(eq_map, dtype=float)
        result.equity.index = pd.to_datetime(list(result.equity.index))
        result.equity = result.equity.sort_index()
        # SPY buy-hold from same start capital over the same dates
        spot0 = close_by_day.get(days[0]) if days else None
        if spot0 and spot0 > 0:
            shares = capital / float(spot0)
            spy_eq = {
                d: shares * close_by_day[d]
                for d in days
                if d in close_by_day
            }
            result.spy_equity = pd.Series(spy_eq, dtype=float)
            result.spy_equity.index = pd.to_datetime(list(result.spy_equity.index))
            result.spy_equity = result.spy_equity.sort_index()

    result.notes = (
        f"capital=peak_defined_risk*{margin_buffer:g}={capital:.2f}; "
        f"tp_debit=(1-{tp_profit_frac:g})*credit; "
        f"sl_debit={sl_mult:g}*credit; package_slip={slip:g}; "
        f"commissions=0; EOD marks (slip clamped to NBBO)"
    )
    return result


def phase5_summary(book: BookResult) -> Dict[str, Any]:
    stats = book_stats(book.trades, capital=book.capital, equity=book.equity)
    gate = None
    spy_s = None
    strat_s = None
    if book.equity is not None and len(book.equity.dropna()) >= 2 and len(book.spy_equity.dropna()) >= 2:
        kept = [t for t in book.trades if not t.skipped and t.pnl is not None]
        spy_s = perf_stats("SPY_buy_hold", book.spy_equity)
        strat_s = perf_stats(
            book.label,
            book.equity,
            spy_cagr=spy_s.cagr,
            spy_sharpe=spy_s.sharpe,
            trade_rets=[float(t.pnl) for t in kept],
        )
        ok, msg = passes_phase5_gates(strat_s, spy_s)
        gate = {"pass": ok, "msg": msg}
        # 2022 window if present
        eq_2022 = book.equity.loc["2022-01-01":"2022-12-31"]
        spy_2022 = book.spy_equity.loc["2022-01-01":"2022-12-31"]
        if len(eq_2022.dropna()) >= 2 and len(spy_2022.dropna()) >= 2:
            s22 = perf_stats(book.label + "_2022", eq_2022 / float(eq_2022.iloc[0]))
            p22 = perf_stats("SPY_2022", spy_2022 / float(spy_2022.iloc[0]))
            ok22, msg22 = passes_phase5_gates(s22, p22, window="2022")
            gate["2022"] = {"pass": ok22, "msg": msg22, "strat": s22.as_dict(), "spy": p22.as_dict()}
    return {
        "label": book.label,
        "stats": stats,
        "phase5": gate,
        "strat_perf": None if strat_s is None else strat_s.as_dict(),
        "spy_perf": None if spy_s is None else spy_s.as_dict(),
        "notes": book.notes,
        "n_skipped": len(book.skipped),
    }


def make_hybrid_source(cache_dir: Optional[Path] = None) -> HybridSource:
    theta = ThetaSource(cache_dir=cache_dir)
    return HybridSource(odx=OptionsDXSource(), theta=theta)
