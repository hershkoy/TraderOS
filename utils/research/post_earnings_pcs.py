"""Post-earnings bull put credit spread claim-check (Brian Terry / Theta Profits).

Video H2G8_jisZP8 (2025-07): sell ~20-delta put credit spreads *after* earnings
while residual IV is still elevated; TP 50% of credit, SL 2x credit.

Claimed (personal 2-3 month track, not a multi-year backtest):
  ~80% win rate, ~8 day avg hold, ~8-10% return on risk per trade.

This module implements a *mechanical* approximation:
  - Earnings dates via yfinance (AMC if hour>=16 else BMO).
  - Skip gap <= ``max_drop_pct`` (default -10%).
  - Optional require non-negative gap and prior-N post-earnings hold filter.
  - Short put ~20 delta, preferably below pre-earnings close; width by price.
  - Min credit = ``min_credit_frac`` * width (0.16 => $0.80 on $5).
  - Marks: Black-Scholes vertical with elevated entry IV that decays to ATR IV.

Listed Polygon replay is out of scope for the first pass (BS only).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from utils.options.greeks import black_scholes_put, calculate_greeks
from utils.research.last_15m_short_put_bs import (
    IV_CEIL,
    IV_FLOOR,
    _busday_on_or_before,
    _years,
    iv_from_atr_pct,
    round_strike,
    strike_for_put_delta,
)

LOG = logging.getLogger(__name__)

CLAIMED: Dict[str, Any] = {
    "win_rate_pct": 80.0,
    "avg_hold_days": 8.0,
    "ror_pct_lo": 8.0,
    "ror_pct_hi": 10.0,
    "sample": "2-3 months personal track (not multi-year BT)",
    "source": "Theta Profits / Brian Terry YouTube H2G8_jisZP8",
}

DEFAULT_TARGET_DELTA = -0.20
DEFAULT_DTE = 37  # mid of 30-45
DEFAULT_MIN_CREDIT_FRAC = 0.16  # 0.80 / 5.00
DEFAULT_MAX_DROP_PCT = -10.0
DEFAULT_MAX_GAP_PCT = 15.0
DEFAULT_TP_FRAC = 0.50
DEFAULT_SL_MULT = 2.0
DEFAULT_R = 0.05
DEFAULT_IV_ELEV = 1.35
DEFAULT_IV_CRUSH_DAYS = 10
DEFAULT_HIST_LOOKBACK = 8
DEFAULT_HIST_HORIZON_BARS = 21
DEFAULT_HIST_MIN_PASS = 5  # of prior lookback events
DEFAULT_REQUIRE_NONNEG_GAP = True

EXIT_TP = "tp50"
EXIT_SL = "sl_2x"
EXIT_EXPIRY = "expiry"
REASON_OK = "ok"
REASON_SKIP_DROP = "skip_drop"
REASON_SKIP_GAP = "skip_gap_overext"
REASON_SKIP_HIST = "skip_hist"
REASON_NO_CREDIT = "no_credit"
REASON_NO_STRIKE = "no_strike"
REASON_NO_BARS = "no_bars"
REASON_NO_ENTRY = "no_entry"
REASON_BELOW_PRE = "strike_not_below_pre"

TRADE_COLUMNS = (
    "symbol",
    "earn_date",
    "entry_date",
    "exit_date",
    "pre_close",
    "entry_px",
    "gap_pct",
    "short_k",
    "long_k",
    "width",
    "credit",
    "entry_delta",
    "iv_entry",
    "iv_base",
    "dte",
    "max_risk",
    "pnl",
    "ror_pct",
    "hold_days",
    "exit_reason",
    "hist_pass",
    "hist_n",
)


@dataclass
class EarnEvent:
    earn_date: date
    session: str  # "amc" | "bmo"


def _as_date(val: Any) -> Optional[date]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, date) and not isinstance(val, datetime):
        return val
    ts = pd.Timestamp(val)
    if pd.isna(ts):
        return None
    return ts.date()


def normalize_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Return OHLCV indexed by naive date (DatetimeIndex normalized)."""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "ts" in out.columns:
        out["ts"] = pd.to_datetime(out["ts"], errors="coerce", utc=True)
        out = out.dropna(subset=["ts"]).sort_values("ts")
        out["ts"] = out["ts"].dt.tz_convert("America/New_York").dt.tz_localize(None)
        out = out.set_index("ts")
    elif not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, errors="coerce")
    out = out[~out.index.isna()].sort_index()
    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_convert("America/New_York").tz_localize(None)
    out.index = pd.DatetimeIndex(out.index).normalize()
    # De-dupe same calendar day (prefer last)
    out = out[~out.index.duplicated(keep="last")]
    need = ("open", "high", "low", "close")
    for c in need:
        if c not in out.columns:
            return pd.DataFrame()
    return out


def atr_pct_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)
    prev = c.shift(1)
    tr = pd.concat([(h - l), (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    atr = tr.rolling(int(period), min_periods=max(5, period // 2)).mean()
    return 100.0 * atr / c.replace(0, np.nan)


def spread_width_for_price(px: float) -> float:
    p = float(px)
    if p < 50.0:
        return 2.5
    if p < 250.0:
        return 5.0
    return 10.0


def fetch_earnings_yf(symbol: str, limit: int = 40) -> List[EarnEvent]:
    """Pull earnings dates from yfinance. Future rows without Reported EPS kept out."""
    import yfinance as yf

    t = yf.Ticker(str(symbol).upper())
    ed = t.get_earnings_dates(limit=int(limit))
    if ed is None or ed.empty:
        return []
    out: List[EarnEvent] = []
    for ts, row in ed.iterrows():
        reported = row.get("Reported EPS")
        if reported is None or (isinstance(reported, float) and math.isnan(reported)):
            continue
        ts_ny = pd.Timestamp(ts)
        if ts_ny.tzinfo is None:
            ts_ny = ts_ny.tz_localize("America/New_York")
        else:
            ts_ny = ts_ny.tz_convert("America/New_York")
        d = ts_ny.date()
        hour = int(ts_ny.hour)
        session = "amc" if hour >= 16 else "bmo"
        out.append(EarnEvent(earn_date=d, session=session))
    # Chronological ascending
    out.sort(key=lambda e: e.earn_date)
    return out


def load_or_fetch_earnings(
    symbols: Sequence[str],
    cache_path: Optional[Path] = None,
    *,
    limit: int = 40,
    refresh: bool = False,
) -> Dict[str, List[EarnEvent]]:
    """Cache earnings as JSON {SYM: [{earn_date, session}, ...]}."""
    import json

    cache: Dict[str, Any] = {}
    if cache_path is not None and cache_path.exists() and not refresh:
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception as exc:
            LOG.warning("earnings cache read failed: %s", exc)
            cache = {}

    result: Dict[str, List[EarnEvent]] = {}
    dirty = False
    for sym in symbols:
        s = str(sym).upper()
        if s in cache and not refresh:
            rows = cache[s] or []
            result[s] = [
                EarnEvent(
                    earn_date=_as_date(r["earn_date"]) or date.min,
                    session=str(r.get("session") or "amc"),
                )
                for r in rows
                if _as_date(r.get("earn_date")) is not None
            ]
            continue
        try:
            evs = fetch_earnings_yf(s, limit=limit)
        except Exception as exc:
            LOG.warning("earnings fetch failed %s: %s", s, exc)
            evs = []
        result[s] = evs
        cache[s] = [
            {"earn_date": e.earn_date.isoformat(), "session": e.session} for e in evs
        ]
        dirty = True
        LOG.info("earnings %s n=%d", s, len(evs))

    if dirty and cache_path is not None:
        import json

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    return result


def entry_session_date(df: pd.DataFrame, ev: EarnEvent) -> Optional[date]:
    """First RTH session that fully reflects the report."""
    dates = [ts.date() for ts in df.index]
    if not dates:
        return None
    if ev.session == "bmo":
        # Same calendar day if listed; else next available.
        if ev.earn_date in dates:
            return ev.earn_date
        later = [d for d in dates if d >= ev.earn_date]
        return later[0] if later else None
    # AMC: first session *after* earn_date
    later = [d for d in dates if d > ev.earn_date]
    return later[0] if later else None


def pre_earnings_close(df: pd.DataFrame, ev: EarnEvent) -> Optional[float]:
    """Last close strictly before the reaction session's prior day logic.

    BMO: prior session close. AMC: earn_date close (pre-announce close).
    """
    dates = [ts.date() for ts in df.index]
    if ev.session == "amc":
        if ev.earn_date in dates:
            return float(df.loc[pd.Timestamp(ev.earn_date), "close"])
        prior = [d for d in dates if d < ev.earn_date]
    else:
        prior = [d for d in dates if d < ev.earn_date]
    if not prior:
        return None
    return float(df.loc[pd.Timestamp(prior[-1]), "close"])


def post_earn_hist_pass(
    df: pd.DataFrame,
    events: Sequence[EarnEvent],
    asof_idx: int,
    *,
    lookback: int = DEFAULT_HIST_LOOKBACK,
    horizon_bars: int = DEFAULT_HIST_HORIZON_BARS,
    min_pass: int = DEFAULT_HIST_MIN_PASS,
) -> Tuple[bool, int, int]:
    """Prior ``lookback`` events: reaction gap then hold >= pre-close at +horizon.

    Returns (pass, n_scored, n_ok).
    """
    if asof_idx <= 0 or lookback <= 0:
        return True, 0, 0
    prior = list(events[:asof_idx])[-int(lookback) :]
    scored = 0
    ok = 0
    for ev in prior:
        entry_d = entry_session_date(df, ev)
        pre = pre_earnings_close(df, ev)
        if entry_d is None or pre is None or pre <= 0:
            continue
        # Index location of entry
        try:
            loc = df.index.get_loc(pd.Timestamp(entry_d))
        except KeyError:
            continue
        if isinstance(loc, slice):
            continue
        if isinstance(loc, np.ndarray):
            loc = int(np.flatnonzero(loc)[0]) if loc.any() else None
        if loc is None:
            continue
        loc = int(loc)
        j = loc + int(horizon_bars)
        if j >= len(df):
            continue
        scored += 1
        horizon_close = float(df.iloc[j]["close"])
        # Brian: after a gap-up, still at/above that gap level 21d later.
        # Mechanical: horizon close >= max(pre, entry_close).
        entry_close = float(df.iloc[loc]["close"])
        floor = max(float(pre), float(entry_close))
        if horizon_close + 1e-9 >= floor:
            ok += 1
    if scored == 0:
        return True, 0, 0  # no history -> don't block
    return ok >= int(min_pass), scored, ok


def iv_on_day(
    iv_entry: float,
    iv_base: float,
    days_since_entry: int,
    crush_days: int,
) -> float:
    if crush_days <= 0:
        return float(iv_entry)
    w = max(0.0, 1.0 - float(days_since_entry) / float(crush_days))
    return float(iv_base + (iv_entry - iv_base) * w)


def put_vertical_mark(
    S: float,
    short_k: float,
    long_k: float,
    T: float,
    r: float,
    sigma: float,
) -> float:
    """Debit to close a short put vertical (= short put - long put)."""
    short_p = float(black_scholes_put(S, short_k, T, r, sigma))
    long_p = float(black_scholes_put(S, long_k, T, r, sigma))
    return max(0.0, short_p - long_p)


def pick_short_strike(
    entry_px: float,
    pre_close: float,
    T: float,
    r: float,
    sigma: float,
    target_delta: float = DEFAULT_TARGET_DELTA,
) -> Optional[float]:
    """~target_delta put, clamped at/below pre-earnings close."""
    k = strike_for_put_delta(
        float(entry_px), T, r, sigma, target_delta=float(target_delta)
    )
    # Prefer below pre-earnings price (Brian).
    max_k = round_strike(float(pre_close) - 0.01)
    if max_k <= 0:
        return None
    if k > max_k:
        k = max_k
    # Still need OTM-ish vs entry
    if k >= float(entry_px):
        k = round_strike(float(entry_px) * 0.95)
    if k <= 0 or k >= float(entry_px):
        return None
    return float(k)


@dataclass
class PcsResult:
    filled: bool
    reason: str
    row: Optional[Dict[str, Any]] = None


def simulate_pcs(
    df: pd.DataFrame,
    ev: EarnEvent,
    *,
    dte: int = DEFAULT_DTE,
    target_delta: float = DEFAULT_TARGET_DELTA,
    min_credit_frac: float = DEFAULT_MIN_CREDIT_FRAC,
    max_drop_pct: float = DEFAULT_MAX_DROP_PCT,
    max_gap_pct: float = DEFAULT_MAX_GAP_PCT,
    require_nonneg_gap: bool = DEFAULT_REQUIRE_NONNEG_GAP,
    tp_frac: float = DEFAULT_TP_FRAC,
    sl_mult: float = DEFAULT_SL_MULT,
    r: float = DEFAULT_R,
    iv_elev: float = DEFAULT_IV_ELEV,
    iv_crush_days: int = DEFAULT_IV_CRUSH_DAYS,
    hist_ok: bool = True,
    hist_n: int = 0,
    hist_pass_n: int = 0,
    symbol: str = "",
) -> PcsResult:
    """Simulate one post-earnings put credit spread from daily bars."""
    if not hist_ok:
        return PcsResult(False, REASON_SKIP_HIST)

    entry_d = entry_session_date(df, ev)
    pre = pre_earnings_close(df, ev)
    if entry_d is None or pre is None or pre <= 0:
        return PcsResult(False, REASON_NO_ENTRY)

    try:
        bar = df.loc[pd.Timestamp(entry_d)]
    except KeyError:
        return PcsResult(False, REASON_NO_BARS)

    entry_px = float(bar["close"])
    if entry_px <= 0:
        return PcsResult(False, REASON_NO_ENTRY)

    gap_pct = 100.0 * (entry_px / float(pre) - 1.0)
    if gap_pct <= float(max_drop_pct):
        return PcsResult(False, REASON_SKIP_DROP)
    if require_nonneg_gap and gap_pct < 0.0:
        return PcsResult(False, REASON_SKIP_DROP)
    if gap_pct > float(max_gap_pct):
        return PcsResult(False, REASON_SKIP_GAP)

    atr_pct = atr_pct_series(df)
    try:
        atr_v = float(atr_pct.loc[pd.Timestamp(entry_d)])
    except Exception:
        atr_v = float("nan")
    if not math.isfinite(atr_v) or atr_v <= 0:
        # Fallback: 20d realized
        loc = int(df.index.get_loc(pd.Timestamp(entry_d)))
        window = df.iloc[max(0, loc - 21) : loc + 1]["close"].astype(float).pct_change()
        vol = float(window.std() * math.sqrt(252) * 100) if len(window) > 5 else 25.0
        atr_v = max(1.0, vol / math.sqrt(252))

    iv_base = iv_from_atr_pct(atr_v)
    iv_entry = float(min(IV_CEIL, max(IV_FLOOR, iv_base * float(iv_elev))))

    dte_i = max(21, int(dte))
    T0 = _years(dte_i)
    short_k = pick_short_strike(
        entry_px, float(pre), T0, float(r), float(iv_entry), target_delta=float(target_delta)
    )
    if short_k is None:
        return PcsResult(False, REASON_NO_STRIKE)
    if short_k > float(pre) + 1e-9:
        return PcsResult(False, REASON_BELOW_PRE)

    width = spread_width_for_price(entry_px)
    long_k = round_strike(float(short_k) - float(width))
    if long_k <= 0 or long_k >= short_k:
        # Adjust width to listed grid
        long_k = round_strike(float(short_k) - float(width))
        if long_k <= 0:
            return PcsResult(False, REASON_NO_STRIKE)
    width_eff = float(short_k) - float(long_k)
    if width_eff <= 0:
        return PcsResult(False, REASON_NO_STRIKE)

    credit = put_vertical_mark(entry_px, short_k, long_k, T0, float(r), float(iv_entry))
    min_credit = float(min_credit_frac) * float(width_eff)
    if not math.isfinite(credit) or credit < min_credit - 1e-9:
        return PcsResult(False, REASON_NO_CREDIT)

    g0 = calculate_greeks(entry_px, short_k, T0, float(r), float(iv_entry), "P")
    entry_delta = float(g0["delta"])

    max_risk = float(width_eff) - float(credit)
    if max_risk <= 1e-9:
        return PcsResult(False, REASON_NO_CREDIT)

    tp_level = float(tp_frac) * float(credit)  # debit to close for TP
    sl_level = float(sl_mult) * float(credit)  # debit to close for SL
    # Cap SL debit at max width (can't lose more than width)
    sl_level = min(sl_level, float(width_eff))

    expiry = _busday_on_or_before(entry_d + timedelta(days=dte_i))
    after = df.loc[(df.index.date > entry_d) & (df.index.date <= expiry)]
    if after.empty:
        return PcsResult(False, REASON_NO_BARS)

    exit_debit: Optional[float] = None
    exit_reason: Optional[str] = None
    exit_day: Optional[date] = None

    for ts, brow in after.iterrows():
        day = ts.date() if hasattr(ts, "date") else _as_date(ts)
        if day is None:
            continue
        cl = float(brow["close"])
        if not (math.isfinite(cl) and cl > 0):
            continue
        days_left = (expiry - day).days
        days_in = (day - entry_d).days
        T = _years(max(days_left, 0))
        sig = iv_on_day(iv_entry, iv_base, days_in, int(iv_crush_days))

        if day >= expiry or days_left <= 0:
            # Intrinsic settlement of vertical
            short_i = max(0.0, short_k - cl)
            long_i = max(0.0, long_k - cl)
            exit_debit = max(0.0, short_i - long_i)
            exit_reason = EXIT_EXPIRY
            exit_day = day
            break

        mark = put_vertical_mark(cl, short_k, long_k, T, float(r), float(sig))
        if mark <= tp_level + 1e-12:
            exit_debit = float(tp_level)
            exit_reason = EXIT_TP
            exit_day = day
            break
        if mark >= sl_level - 1e-12:
            exit_debit = float(sl_level)
            exit_reason = EXIT_SL
            exit_day = day
            break

    if exit_debit is None or exit_day is None or exit_reason is None:
        return PcsResult(False, REASON_NO_BARS)

    pnl = float(credit) - float(exit_debit)
    ror = 100.0 * pnl / float(max_risk)
    row = {
        "symbol": str(symbol).upper(),
        "earn_date": ev.earn_date,
        "entry_date": entry_d,
        "exit_date": exit_day,
        "pre_close": float(pre),
        "entry_px": entry_px,
        "gap_pct": gap_pct,
        "short_k": short_k,
        "long_k": long_k,
        "width": width_eff,
        "credit": credit,
        "entry_delta": entry_delta,
        "iv_entry": iv_entry,
        "iv_base": iv_base,
        "dte": dte_i,
        "max_risk": max_risk,
        "pnl": pnl,
        "ror_pct": ror,
        "hold_days": (exit_day - entry_d).days,
        "exit_reason": exit_reason,
        "hist_pass": int(hist_pass_n),
        "hist_n": int(hist_n),
    }
    return PcsResult(True, REASON_OK, row)


def collect_symbol_trades(
    df_raw: pd.DataFrame,
    events: Sequence[EarnEvent],
    *,
    symbol: str,
    start: Optional[date] = None,
    end: Optional[date] = None,
    use_hist_filter: bool = True,
    hist_lookback: int = DEFAULT_HIST_LOOKBACK,
    hist_horizon_bars: int = DEFAULT_HIST_HORIZON_BARS,
    hist_min_pass: int = DEFAULT_HIST_MIN_PASS,
    **sim_kwargs: Any,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    df = normalize_daily(df_raw)
    counts: Dict[str, int] = {}
    rows: List[Dict[str, Any]] = []
    if df.empty or not events:
        return pd.DataFrame(columns=list(TRADE_COLUMNS)), counts

    for i, ev in enumerate(events):
        if start is not None and ev.earn_date < start:
            continue
        if end is not None and ev.earn_date > end:
            continue
        if use_hist_filter:
            ok, scored, n_ok = post_earn_hist_pass(
                df,
                events,
                i,
                lookback=hist_lookback,
                horizon_bars=hist_horizon_bars,
                min_pass=hist_min_pass,
            )
        else:
            ok, scored, n_ok = True, 0, 0
        res = simulate_pcs(
            df,
            ev,
            hist_ok=ok,
            hist_n=scored,
            hist_pass_n=n_ok,
            symbol=symbol,
            **sim_kwargs,
        )
        counts[res.reason] = counts.get(res.reason, 0) + 1
        if res.filled and res.row is not None:
            rows.append(res.row)

    trades = pd.DataFrame(rows, columns=list(TRADE_COLUMNS)) if rows else pd.DataFrame(
        columns=list(TRADE_COLUMNS)
    )
    return trades, counts


def book_stats(trades: pd.DataFrame) -> Dict[str, Any]:
    if trades is None or trades.empty:
        return {
            "n": 0,
            "win_rate_pct": None,
            "avg_hold_days": None,
            "avg_ror_pct": None,
            "median_ror_pct": None,
            "expectancy_ror_pct": None,
            "profit_factor": None,
            "avg_win_ror_pct": None,
            "avg_loss_ror_pct": None,
            "tp_pct": None,
            "sl_pct": None,
            "expiry_pct": None,
        }
    r = trades["ror_pct"].astype(float)
    wins = r[r > 0]
    losses = r[r <= 0]
    gp = float(wins.sum()) if len(wins) else 0.0
    gl = float((-losses).sum()) if len(losses) else 0.0
    pf = (gp / gl) if gl > 1e-12 else (float("inf") if gp > 0 else None)
    reasons = trades["exit_reason"].astype(str)
    n = int(len(trades))
    return {
        "n": n,
        "win_rate_pct": 100.0 * float((r > 0).mean()),
        "avg_hold_days": float(trades["hold_days"].astype(float).mean()),
        "avg_ror_pct": float(r.mean()),
        "median_ror_pct": float(r.median()),
        "expectancy_ror_pct": float(r.mean()),
        "profit_factor": pf,
        "avg_win_ror_pct": float(wins.mean()) if len(wins) else None,
        "avg_loss_ror_pct": float(losses.mean()) if len(losses) else None,
        "tp_pct": 100.0 * float((reasons == EXIT_TP).mean()),
        "sl_pct": 100.0 * float((reasons == EXIT_SL).mean()),
        "expiry_pct": 100.0 * float((reasons == EXIT_EXPIRY).mean()),
        "avg_credit": float(trades["credit"].astype(float).mean()),
        "avg_gap_pct": float(trades["gap_pct"].astype(float).mean()),
    }


def year_bucket_stats(trades: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    buckets = (
        ("2018-2019", date(2018, 1, 1), date(2019, 12, 31)),
        ("2020-2021", date(2020, 1, 1), date(2021, 12, 31)),
        ("2022-2023", date(2022, 1, 1), date(2023, 12, 31)),
        ("2024-2026", date(2024, 1, 1), date(2026, 12, 31)),
    )
    out: Dict[str, Dict[str, Any]] = {}
    if trades is None or trades.empty:
        return {name: book_stats(trades) for name, _, _ in buckets}
    ed = pd.to_datetime(trades["entry_date"]).dt.date
    for name, a, b in buckets:
        mask = (ed >= a) & (ed <= b)
        out[name] = book_stats(trades.loc[mask].copy())
    return out
