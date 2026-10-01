"""QQQ gap-down LEAPS claim-check (Options With Ravish / OptionsKit).

Mechanical rules from YouTube dY2ralsWdQ8 (2025-07-05):

  Strat1 (pullback): open gap <= -1% AND prior close above SMA100;
                     buy 1x ~60-delta call, ~365 DTE; TP +50%; no SL.
  Strat2 (capitulation): open gap <= -2%; buy 2x same; TP +50%; no SL.
  Optional occupancy cap: max 3 open lots (video risk note).

Marks are Black-Scholes with ATR-proxy IV (constant, or elevated-at-entry
then linear decay). TimescaleDB has QQQ option_contracts but **zero**
option_quotes; OptionsDX EOD is SPY/VIX only -- listed mid replay is out
of scope for the full book. Treat BS as an optimistic method check.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from utils.options.greeks import black_scholes_call, calculate_greeks

CLAIMED = {
    "strat1": {
        "n": 53,
        "wins": 51,
        "losses": 2,
        "win_rate_pct": 96.0,
        "total_pnl": 88400.0,
        "avg_win": 1900.0,
        "avg_loss": 4600.0,
        "pf": 10.48,
        "avg_days": 102.0,
        "max_dd": 9330.0,
        "contracts": 1,
    },
    "strat2": {
        "n": 27,
        "wins": 26,
        "losses": 1,
        "win_rate_pct": 96.0,
        "total_pnl": 91000.0,
        "avg_win": 3779.0,
        "avg_loss": 6328.0,
        "pf": 15.0,
        "avg_days": None,
        "max_dd": 6000.0,
        "contracts": 2,
    },
    "combined": {
        "total_pnl": 180000.0,
        "win_rate_pct": 96.0,
        "window": "approx 2020-01 to 2025-06 (video: last 5y; OptionsKit shot Jan03'20-Jun05'25)",
    },
}

DEFAULT_DTE = 365
DEFAULT_TARGET_DELTA = 0.60
DEFAULT_TP_FRAC = 0.50  # +50% on premium paid
DEFAULT_R = 0.05
DEFAULT_Q = 0.005  # rough QQQ dividend yield
DEFAULT_SMA = 100
DEFAULT_MAX_OPEN = 3
IV_FLOOR = 0.12
IV_CEIL = 0.80
ATR_LEN = 20
EXIT_TP = "tp50"
EXIT_EXPIRY = "expiry"
EXIT_PANEL_END = "panel_end"
REASON_OK = "ok"
REASON_SKIP_OCC = "skip_occupancy"
REASON_NO_IV = "no_iv"
REASON_NO_STRIKE = "no_strike"
REASON_NO_PREMIUM = "no_premium"


def _as_date(val: Any) -> Optional[date]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, date) and not isinstance(val, datetime):
        return val
    ts = pd.Timestamp(val)
    if pd.isna(ts):
        return None
    return ts.date()


def _busday_on_or_before(d: date) -> date:
    out = d
    while out.weekday() >= 5:
        out -= timedelta(days=1)
    return out


def _years(days: float) -> float:
    return max(float(days), 0.0) / 365.25


def round_qqq_strike(px: float) -> float:
    """QQQ listed strikes are typically $1 under ~$300 and $5 above for LEAPS."""
    p = float(px)
    if p <= 0:
        return p
    if p < 300.0:
        step = 1.0
    else:
        step = 5.0
    return round(p / step) * step


def atr_series(df: pd.DataFrame, length: int = ATR_LEN) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev).abs(), (low - prev).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(int(length), min_periods=max(5, int(length) // 2)).mean()


def iv_from_atr(atr: float, price: float) -> Optional[float]:
    if price is None or price <= 0 or atr is None or not math.isfinite(atr) or atr <= 0:
        return None
    daily = float(atr) / float(price)
    iv = daily * math.sqrt(252.0)
    if not math.isfinite(iv) or iv <= 0:
        return None
    return float(max(IV_FLOOR, min(IV_CEIL, iv)))


def strike_for_call_delta(
    S: float,
    T: float,
    r: float,
    sigma: float,
    q: float,
    target_delta: float = DEFAULT_TARGET_DELTA,
    *,
    lo_mult: float = 0.70,
    hi_mult: float = 1.05,
    iters: int = 40,
) -> float:
    """Binary-search strike so BS call delta ~= target (e.g. 0.60)."""
    s = float(S)
    tgt = float(target_delta)
    lo = s * float(lo_mult)
    hi = s * float(hi_mult)
    for _ in range(int(iters)):
        mid = 0.5 * (lo + hi)
        g = calculate_greeks(s, mid, T, r, sigma, "C", q=q)
        d = float(g["delta"])
        # Higher strike -> lower call delta.
        if d > tgt:
            lo = mid
        else:
            hi = mid
    return round_qqq_strike(0.5 * (lo + hi))


def prepare_qqq_daily(df: pd.DataFrame, *, sma_len: int = DEFAULT_SMA) -> pd.DataFrame:
    """Normalize OHLCV and add gap / SMA / ATR columns (causal)."""
    d = df.copy()
    if "ts" in d.columns:
        d["ts"] = pd.to_datetime(d["ts"], utc=True, errors="coerce")
        d = d.dropna(subset=["ts"]).sort_values("ts").drop_duplicates("ts", keep="last")
        d = d.set_index("ts")
    elif not isinstance(d.index, pd.DatetimeIndex):
        raise ValueError("QQQ daily needs ts column or DatetimeIndex")
    else:
        d = d.sort_index()
        if d.index.tz is None:
            d.index = d.index.tz_localize("UTC")
        else:
            d.index = d.index.tz_convert("UTC")

    for col in ("open", "high", "low", "close", "volume"):
        if col not in d.columns:
            raise ValueError(f"missing column {col}")
        d[col] = pd.to_numeric(d[col], errors="coerce")

    d = d.dropna(subset=["open", "high", "low", "close"])
    prev_close = d["close"].shift(1)
    d["gap_pct"] = (d["open"] - prev_close) / prev_close * 100.0
    d["sma100"] = d["close"].rolling(int(sma_len), min_periods=int(sma_len)).mean()
    # Causal at open: prior close vs prior SMA (SMA includes prior close).
    d["prior_close"] = prev_close
    d["prior_sma100"] = d["sma100"].shift(1)
    d["above_sma100_prior"] = d["prior_close"] > d["prior_sma100"]
    d["above_sma100_open"] = d["open"] > d["prior_sma100"]
    # Causal at open: ATR as of prior close (today's H/L unknown at the gap print).
    d["atr"] = atr_series(d, ATR_LEN).shift(1)
    d["session_date"] = [ts.date() for ts in d.index]
    return d


def signal_mask(
    d: pd.DataFrame,
    *,
    mode: str,
    gap_pct: float,
    sma_rule: str = "prior",
) -> pd.Series:
    """Boolean mask of entry days (index-aligned)."""
    gap_ok = d["gap_pct"] <= -abs(float(gap_pct))
    if mode == "strat1":
        if sma_rule == "open":
            sma_ok = d["above_sma100_open"].fillna(False)
        else:
            sma_ok = d["above_sma100_prior"].fillna(False)
        return gap_ok & sma_ok
    if mode == "strat2":
        return gap_ok
    raise ValueError(f"unknown mode {mode}")


@dataclass
class LeapTrade:
    mode: str
    entry_date: date
    exit_date: Optional[date]
    contracts: int
    entry_px: float
    strike: float
    entry_premium: float
    exit_premium: Optional[float]
    entry_delta: float
    entry_iv: float
    dte: int
    hold_days: Optional[int]
    exit_reason: Optional[str]
    pnl_cash: Optional[float]
    gain_pct: Optional[float]
    reason: str = REASON_OK
    skipped: bool = False


@dataclass
class BookResult:
    mode: str
    trades: List[LeapTrade] = field(default_factory=list)
    signals_raw: int = 0
    filled: int = 0
    skipped_occ: int = 0
    skipped_other: int = 0


def _mark_call(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float,
) -> float:
    if T <= 0:
        return max(0.0, float(S) - float(K))
    return float(black_scholes_call(float(S), float(K), T, r, sigma, q=q))


def _iv_on_day(
    entry_iv: float,
    base_iv: float,
    day_i: int,
    *,
    crush_days: int,
) -> float:
    """Linear IV from entry_iv -> base_iv over crush_days (1 = constant entry)."""
    cd = max(1, int(crush_days))
    if cd <= 1:
        return float(entry_iv)
    t = min(1.0, float(day_i) / float(cd))
    return float(entry_iv + (base_iv - entry_iv) * t)


def simulate_leap_trade(
    d: pd.DataFrame,
    entry_idx: int,
    *,
    contracts: int,
    dte: int = DEFAULT_DTE,
    target_delta: float = DEFAULT_TARGET_DELTA,
    tp_frac: float = DEFAULT_TP_FRAC,
    r: float = DEFAULT_R,
    q: float = DEFAULT_Q,
    iv_elev: float = 1.0,
    iv_crush_days: int = 1,
    friction_per_share: float = 0.0,
    mode: str = "strat1",
) -> LeapTrade:
    row = d.iloc[entry_idx]
    entry_day = _as_date(row["session_date"])
    assert entry_day is not None
    S0 = float(row["open"])  # gap known at open; buy at open
    atr0 = float(row["atr"]) if pd.notna(row["atr"]) else float("nan")
    base_iv = iv_from_atr(atr0, S0)
    if base_iv is None:
        prior = float(row["prior_close"]) if pd.notna(row["prior_close"]) else S0
        base_iv = iv_from_atr(atr0, prior)
    if base_iv is None:
        return LeapTrade(
            mode=mode,
            entry_date=entry_day,
            exit_date=None,
            contracts=contracts,
            entry_px=S0,
            strike=0.0,
            entry_premium=0.0,
            exit_premium=None,
            entry_delta=0.0,
            entry_iv=0.0,
            dte=dte,
            hold_days=None,
            exit_reason=None,
            pnl_cash=None,
            gain_pct=None,
            reason=REASON_NO_IV,
            skipped=True,
        )

    entry_iv = float(max(IV_FLOOR, min(IV_CEIL, base_iv * float(iv_elev))))
    dte_i = max(30, int(dte))
    T0 = _years(dte_i)
    strike = strike_for_call_delta(S0, T0, r, entry_iv, q, target_delta=target_delta)
    if strike <= 0:
        return LeapTrade(
            mode=mode,
            entry_date=entry_day,
            exit_date=None,
            contracts=contracts,
            entry_px=S0,
            strike=0.0,
            entry_premium=0.0,
            exit_premium=None,
            entry_delta=0.0,
            entry_iv=entry_iv,
            dte=dte_i,
            hold_days=None,
            exit_reason=None,
            pnl_cash=None,
            gain_pct=None,
            reason=REASON_NO_STRIKE,
            skipped=True,
        )

    debit = _mark_call(S0, strike, T0, r, entry_iv, q)
    fric = max(0.0, float(friction_per_share))
    entry_prem = debit + fric
    if not math.isfinite(entry_prem) or entry_prem <= 1e-8:
        return LeapTrade(
            mode=mode,
            entry_date=entry_day,
            exit_date=None,
            contracts=contracts,
            entry_px=S0,
            strike=strike,
            entry_premium=0.0,
            exit_premium=None,
            entry_delta=0.0,
            entry_iv=entry_iv,
            dte=dte_i,
            hold_days=None,
            exit_reason=None,
            pnl_cash=None,
            gain_pct=None,
            reason=REASON_NO_PREMIUM,
            skipped=True,
        )

    g0 = calculate_greeks(S0, strike, T0, r, entry_iv, "C", q=q)
    entry_delta = float(g0["delta"])
    tp_level = entry_prem * (1.0 + float(tp_frac))  # +50% => 1.5x debit
    expiry = _busday_on_or_before(entry_day + timedelta(days=dte_i))

    exit_prem: Optional[float] = None
    exit_reason: Optional[str] = None
    exit_day: Optional[date] = None

    for j in range(entry_idx + 1, len(d)):
        bar = d.iloc[j]
        day = _as_date(bar["session_date"])
        if day is None:
            continue
        cl = float(bar["close"])
        if not math.isfinite(cl) or cl <= 0:
            continue
        days_left = (expiry - day).days
        T = _years(max(0, days_left))
        day_i = j - entry_idx
        sigma = _iv_on_day(entry_iv, base_iv, day_i, crush_days=iv_crush_days)

        if day >= expiry or days_left <= 0:
            exit_prem = max(0.0, cl - strike)
            exit_reason = EXIT_EXPIRY
            exit_day = day
            break

        mark = _mark_call(cl, strike, T, r, sigma, q) - fric
        if mark >= tp_level - 1e-12:
            exit_prem = float(tp_level)
            exit_reason = EXIT_TP
            exit_day = day
            break

    if exit_prem is None:
        last = d.iloc[-1]
        exit_day = _as_date(last["session_date"])
        cl = float(last["close"])
        days_left = max(0, (expiry - exit_day).days) if exit_day else 0
        day_i = len(d) - 1 - entry_idx
        sigma = _iv_on_day(entry_iv, base_iv, day_i, crush_days=iv_crush_days)
        if exit_day is not None and exit_day >= expiry:
            exit_prem = max(0.0, cl - strike)
            exit_reason = EXIT_EXPIRY
        else:
            exit_prem = _mark_call(cl, strike, _years(days_left), r, sigma, q) - fric
            exit_reason = EXIT_PANEL_END

    assert exit_day is not None and exit_prem is not None
    pnl_per = (exit_prem - entry_prem) * 100.0
    pnl_cash = pnl_per * int(contracts)
    gain_pct = 100.0 * (exit_prem - entry_prem) / entry_prem
    hold = (exit_day - entry_day).days
    return LeapTrade(
        mode=mode,
        entry_date=entry_day,
        exit_date=exit_day,
        contracts=int(contracts),
        entry_px=S0,
        strike=strike,
        entry_premium=entry_prem,
        exit_premium=exit_prem,
        entry_delta=entry_delta,
        entry_iv=entry_iv,
        dte=dte_i,
        hold_days=hold,
        exit_reason=exit_reason,
        pnl_cash=pnl_cash,
        gain_pct=gain_pct,
        reason=REASON_OK,
        skipped=False,
    )


def run_book(
    d: pd.DataFrame,
    *,
    mode: str,
    gap_pct: float,
    contracts: int,
    start: date,
    end: date,
    sma_rule: str = "prior",
    max_open: Optional[int] = DEFAULT_MAX_OPEN,
    dte: int = DEFAULT_DTE,
    target_delta: float = DEFAULT_TARGET_DELTA,
    tp_frac: float = DEFAULT_TP_FRAC,
    r: float = DEFAULT_R,
    q: float = DEFAULT_Q,
    iv_elev: float = 1.0,
    iv_crush_days: int = 1,
    friction_per_share: float = 0.0,
) -> BookResult:
    mask = signal_mask(d, mode=mode, gap_pct=gap_pct, sma_rule=sma_rule)
    dates = [_as_date(x) for x in d["session_date"].tolist()]
    in_window = [
        (dt is not None and dt >= start and dt <= end) for dt in dates
    ]
    idxs = [i for i, ok in enumerate(mask.tolist()) if ok and in_window[i]]
    book = BookResult(mode=mode, signals_raw=len(idxs))
    open_lots: List[Tuple[date, LeapTrade]] = []

    for i in idxs:
        day = dates[i]
        assert day is not None
        open_lots = [(ed, t) for ed, t in open_lots if ed is None or ed > day]
        if max_open is not None and len(open_lots) >= int(max_open):
            book.skipped_occ += 1
            book.trades.append(
                LeapTrade(
                    mode=mode,
                    entry_date=day,
                    exit_date=None,
                    contracts=contracts,
                    entry_px=float(d.iloc[i]["open"]),
                    strike=0.0,
                    entry_premium=0.0,
                    exit_premium=None,
                    entry_delta=0.0,
                    entry_iv=0.0,
                    dte=dte,
                    hold_days=None,
                    exit_reason=None,
                    pnl_cash=None,
                    gain_pct=None,
                    reason=REASON_SKIP_OCC,
                    skipped=True,
                )
            )
            continue

        tr = simulate_leap_trade(
            d,
            i,
            contracts=contracts,
            dte=dte,
            target_delta=target_delta,
            tp_frac=tp_frac,
            r=r,
            q=q,
            iv_elev=iv_elev,
            iv_crush_days=iv_crush_days,
            friction_per_share=friction_per_share,
            mode=mode,
        )
        book.trades.append(tr)
        if tr.skipped:
            book.skipped_other += 1
            continue
        book.filled += 1
        open_lots.append((tr.exit_date, tr))

    return book


def book_stats(trades: Sequence[LeapTrade]) -> Dict[str, Any]:
    filled = [t for t in trades if not t.skipped and t.pnl_cash is not None]
    if not filled:
        return {
            "n": 0,
            "wins": 0,
            "losses": 0,
            "win_rate_pct": None,
            "total_pnl": 0.0,
            "avg_win": None,
            "avg_loss": None,
            "pf": None,
            "avg_days": None,
            "max_dd": None,
            "avg_gain_pct": None,
            "tp_pct": None,
            "expiry_pct": None,
            "avg_entry_premium": None,
            "avg_entry_delta": None,
            "avg_contracts": None,
        }
    pnls = [float(t.pnl_cash) for t in filled]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gp = sum(wins)
    gl = abs(sum(losses))
    holds = [t.hold_days for t in filled if t.hold_days is not None]
    exits = [t.exit_reason for t in filled]
    by_exit = sorted(filled, key=lambda t: (t.exit_date or t.entry_date, t.entry_date))
    eq = 0.0
    peak = 0.0
    max_dd = 0.0
    for t in by_exit:
        eq += float(t.pnl_cash or 0.0)
        peak = max(peak, eq)
        max_dd = max(max_dd, peak - eq)

    return {
        "n": len(filled),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": round(100.0 * len(wins) / len(filled), 1),
        "total_pnl": round(sum(pnls), 0),
        "avg_win": round(float(np.mean(wins)), 0) if wins else None,
        "avg_loss": round(float(np.mean([abs(x) for x in losses])), 0) if losses else None,
        "pf": round(gp / gl, 2) if gl > 1e-9 else None,
        "avg_days": round(float(np.mean(holds)), 1) if holds else None,
        "max_dd": round(max_dd, 0),
        "avg_gain_pct": round(
            float(np.mean([t.gain_pct for t in filled if t.gain_pct is not None])), 1
        ),
        "tp_pct": round(100.0 * sum(1 for e in exits if e == EXIT_TP) / len(filled), 1),
        "expiry_pct": round(
            100.0 * sum(1 for e in exits if e in (EXIT_EXPIRY, EXIT_PANEL_END)) / len(filled),
            1,
        ),
        "avg_entry_premium": round(float(np.mean([t.entry_premium for t in filled])), 2),
        "avg_entry_delta": round(float(np.mean([t.entry_delta for t in filled])), 3),
        "avg_contracts": round(float(np.mean([t.contracts for t in filled])), 2),
    }


def year_bucket_stats(trades: Sequence[LeapTrade]) -> List[Dict[str, Any]]:
    buckets = [
        ("2020-21", date(2020, 1, 1), date(2021, 12, 31)),
        ("2022-23", date(2022, 1, 1), date(2023, 12, 31)),
        ("2024-26", date(2024, 1, 1), date(2026, 12, 31)),
        ("2022", date(2022, 1, 1), date(2022, 12, 31)),
    ]
    out = []
    for name, a, b in buckets:
        sub = [
            t
            for t in trades
            if not t.skipped
            and t.pnl_cash is not None
            and t.entry_date is not None
            and a <= t.entry_date <= b
        ]
        s = book_stats(sub)
        s["bucket"] = name
        out.append(s)
    return out


def trades_to_frame(trades: Sequence[LeapTrade]) -> pd.DataFrame:
    rows = []
    for t in trades:
        rows.append(
            {
                "mode": t.mode,
                "entry_date": t.entry_date,
                "exit_date": t.exit_date,
                "contracts": t.contracts,
                "entry_px": t.entry_px,
                "strike": t.strike,
                "entry_premium": t.entry_premium,
                "exit_premium": t.exit_premium,
                "entry_delta": t.entry_delta,
                "entry_iv": t.entry_iv,
                "dte": t.dte,
                "hold_days": t.hold_days,
                "exit_reason": t.exit_reason,
                "pnl_cash": t.pnl_cash,
                "gain_pct": t.gain_pct,
                "reason": t.reason,
                "skipped": t.skipped,
            }
        )
    return pd.DataFrame(rows)


def count_signals(
    d: pd.DataFrame,
    *,
    start: date,
    end: date,
    sma_rule: str = "prior",
) -> Dict[str, Any]:
    """Raw signal counts (no occupancy) for claim n-match."""
    dates = [_as_date(x) for x in d["session_date"].tolist()]

    def _n(mask: pd.Series) -> int:
        return sum(
            1
            for i, ok in enumerate(mask.tolist())
            if ok and dates[i] is not None and start <= dates[i] <= end
        )

    return {
        "start": str(start),
        "end": str(end),
        "sma_rule": sma_rule,
        "gap1_only": _n(d["gap_pct"] <= -1.0),
        "strat1": _n(signal_mask(d, mode="strat1", gap_pct=1.0, sma_rule=sma_rule)),
        "strat2": _n(signal_mask(d, mode="strat2", gap_pct=2.0, sma_rule=sma_rule)),
    }
