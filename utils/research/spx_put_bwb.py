"""Carl Allen 21 DTE put Broken Wing Butterfly (SPX/SPXW) claim-check.

Source: Theta Profits interview (youtu.be/8ea0DwpJtgg) — Carl Allen /
datadrivenoptions.com.

Mechanical rules (video):
  - Put BWB: long K+narrow, short 2x K, long K-wide (default 25x50 on SPX)
  - Open ~21 DTE; short put delta ~0.28-0.32 (scroll for credit)
  - Collect credit = 12-15% of narrow wing (e.g. $3.00-$3.75 on 25-wide)
  - GTC take-profit: close debit <= 2% of narrow wing ($0.50 on 25-wide)
  - Stop: close package when debit >= 3x credit (collected $3.50 -> stop $10.50)
  - Prefer exit by ~7 DTE remaining (~2 weeks); time-stop at min_dte
  - Claimed WR ~80% without stop, ~70% with 2x-premium stop

Listed marks via Theta Terminal EOD (prefer) / 15:45 NBBO fallback.
Underlying spot via Theta /v3/index/history/eod (SPX) or Yahoo ^GSPC fallback.

Follow-up (not in this harness): paper-trade 1-lot SPXW BWB combo orders to
profile true CBOE COB fill friction vs mid (calibrate package_slip).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from io import StringIO
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from utils.api.thetadata.thetadata_client import ThetaDataClient, get_thetadata_client
from utils.options.greeks import calculate_greeks
from utils.research.forward_factor import (
    estimate_iv,
    fetch_eod,
    list_expirations,
    list_strikes,
    mondays_on_index,
    pick_expiry,
    quote_on_day,
)
from utils.research.last_15m_short_put_bs import strike_for_put_delta

LOG = logging.getLogger(__name__)

CLAIMED: Dict[str, Any] = {
    "win_rate_pct_no_stop": 80.0,
    "win_rate_pct_with_stop": 70.0,
    "credit_frac_lo": 0.12,
    "credit_frac_hi": 0.15,
    "target_delta": 0.30,
    "dte_open": 21,
    "dte_exit_hope": 7,
    "tp_frac_of_narrow": 0.02,
    "sl_mult_of_credit": 3.0,  # package value = 3x credit (loss = 2x credit)
    "narrow": 25,
    "wide": 50,
    "return_claim": "10-13% of narrow-wing risk in ~2 weeks",
    "source": "Theta Profits / Carl Allen youtu.be/8ea0DwpJtgg",
}

DEFAULT_TARGET_DELTA = -0.30
DEFAULT_DELTA_LO = -0.35
DEFAULT_DELTA_HI = -0.21
DEFAULT_NARROW = 25.0
DEFAULT_WIDE = 50.0
DEFAULT_DTE = 21
DEFAULT_DTE_TOL = 4
DEFAULT_MIN_DTE_EXIT = 7
DEFAULT_CREDIT_FRAC_LO = 0.12
DEFAULT_CREDIT_FRAC_HI = 0.15
DEFAULT_TP_FRAC = 0.02
DEFAULT_SL_MULT = 3.0
DEFAULT_R = 0.05
DEFAULT_COMM_PER_LEG = 0.65
DEFAULT_MULT = 100.0
DEFAULT_PACKAGE_SLIP = 0.15  # COB-style combo slip vs mid (open credit -slip, close debit +slip)

EXIT_TP = "tp"
EXIT_SL = "sl"
EXIT_SL_SPOT = "sl_spot"
EXIT_TIME = "time_dte"
EXIT_EXPIRY = "expiry"
EXIT_NO_MARK = "no_mark"

STOP_MODE_DEBIT = "debit"  # package debit >= 3x credit
STOP_MODE_SPOT = "spot"  # underlying close <= short_k (+ optional buffer)
STOP_MODE_NONE = "none"

REASON_OK = "ok"
REASON_NO_EXP = "no_expiry"
REASON_NO_SPOT = "no_spot"
REASON_NO_STRIKES = "no_strikes"
REASON_NO_CREDIT = "credit_out_of_band"
REASON_NO_LEGS = "missing_leg_quote"
REASON_NO_SHORT = "no_short_strike"
REASON_BUSY = "max_concurrent"
REASON_BELOW_MA = "below_trend_ma"


@dataclass
class BwbTrade:
    symbol: str
    entry_date: date
    exit_date: Optional[date] = None
    expiration: Optional[date] = None
    spot: Optional[float] = None
    short_k: Optional[float] = None
    high_k: Optional[float] = None
    low_k: Optional[float] = None
    narrow: float = DEFAULT_NARROW
    wide: float = DEFAULT_WIDE
    entry_delta: Optional[float] = None
    credit: Optional[float] = None
    exit_debit: Optional[float] = None
    tp_debit: Optional[float] = None
    sl_debit: Optional[float] = None
    max_risk: Optional[float] = None
    pnl: Optional[float] = None
    ror_pct: Optional[float] = None
    hold_days: Optional[int] = None
    exit_reason: Optional[str] = None
    mark_mode: str = "mid"
    skipped: bool = False
    skip_reason: Optional[str] = None
    dte_entry: Optional[int] = None
    dte_exit: Optional[int] = None


@dataclass
class BookResult:
    label: str
    mark_mode: str
    trades: List[BwbTrade] = field(default_factory=list)
    skipped: List[Dict[str, Any]] = field(default_factory=list)


def load_index_eod(
    client: ThetaDataClient,
    symbol: str,
    start: date,
    end: date,
) -> pd.DataFrame:
    """SPX/SPY daily OHLC. Prefer Theta; SPX falls back to Yahoo ^GSPC on 403."""
    sym = str(symbol).upper()
    if sym in ("SPX", "VIX", "VIX3M", "VVIX"):
        path = "/v3/index/history/eod"
    else:
        path = "/v3/stock/history/eod"
    r = client.session.get(
        f"{client.base_url}{path}",
        params={
            "symbol": sym,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
        },
        timeout=120,
    )
    if r.status_code == 200 and r.content:
        df = pd.read_csv(StringIO(r.text))
        if not df.empty:
            if "created" in df.columns:
                ts = pd.to_datetime(df["created"], errors="coerce")
            elif "last_trade" in df.columns:
                ts = pd.to_datetime(df["last_trade"], errors="coerce")
            else:
                ts = pd.to_datetime(df.iloc[:, 0], errors="coerce")
            out = pd.DataFrame(
                {
                    "open": pd.to_numeric(df.get("open"), errors="coerce"),
                    "high": pd.to_numeric(df.get("high"), errors="coerce"),
                    "low": pd.to_numeric(df.get("low"), errors="coerce"),
                    "close": pd.to_numeric(df.get("close"), errors="coerce"),
                },
                index=pd.DatetimeIndex(ts.dt.normalize()),
            )
            out = out.dropna(subset=["close"]).sort_index()
            out = out[~out.index.duplicated(keep="last")]
            return out
        LOG.warning("Theta EOD empty for %s; trying Yahoo fallback", sym)
    else:
        LOG.warning(
            "Theta EOD %s -> %s; trying Yahoo/Timescale fallback (%s)",
            sym,
            r.status_code,
            (r.text or "")[:120],
        )

    if sym == "SPX":
        return _yahoo_ohlc("^GSPC", start, end)
    if sym == "SPY":
        try:
            from utils.data.ohlcv_loader import load_ohlcv_many

            frames = load_ohlcv_many(
                ["SPY"],
                timeframe="1d",
                provider="ALPACA",
                start=datetime(start.year, start.month, start.day),
                end=datetime(end.year, end.month, end.day),
                use_cache=True,
            )
            df = frames.get("SPY")
            if df is not None and not df.empty:
                out = df[["open", "high", "low", "close"]].copy()
                out.index = pd.DatetimeIndex(pd.to_datetime(out.index).normalize())
                return out
        except Exception as exc:  # noqa: BLE001
            LOG.warning("Timescale SPY load failed: %s", exc)
        return _yahoo_ohlc("SPY", start, end)
    raise RuntimeError(f"No spot history for {sym}")


def _yahoo_ohlc(ticker: str, start: date, end: date) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(
        ticker,
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        progress=False,
        auto_adjust=True,
    )
    if raw is None or raw.empty:
        raise RuntimeError(f"Yahoo empty for {ticker}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0].lower() if isinstance(c, tuple) else str(c).lower() for c in raw.columns]
    else:
        raw.columns = [str(c).lower() for c in raw.columns]
    out = raw[["open", "high", "low", "close"]].copy()
    out.index = pd.DatetimeIndex(pd.to_datetime(out.index).normalize())
    return out.dropna(subset=["close"])


def _nearest_listed(strikes: Sequence[float], target: float) -> Optional[float]:
    if not strikes:
        return None
    return float(min(strikes, key=lambda s: abs(float(s) - float(target))))


def _base_mark_mode(mark_mode: str) -> str:
    """Strip slip suffix: mid_slip0.15 -> mid."""
    m = str(mark_mode).strip().lower()
    if m.startswith("mid_slip"):
        return "mid"
    return m


def parse_package_slip(mark_mode: str, package_slip: float = 0.0) -> float:
    """Slip from explicit arg, or from mark_mode mid_slip0.15 / mid_slip."""
    if package_slip and package_slip > 0:
        return float(package_slip)
    m = str(mark_mode).strip().lower()
    if m.startswith("mid_slip"):
        rest = m[len("mid_slip") :]
        if not rest:
            return float(DEFAULT_PACKAGE_SLIP)
        return float(rest)
    return 0.0


def _leg_mark(q: Dict[str, float], mode: str, side: str) -> Optional[float]:
    """side 'buy' or 'sell'."""
    bid = float(q.get("bid") or 0)
    ask = float(q.get("ask") or 0)
    mid = float(q.get("mid") or (0.5 * (bid + ask) if bid > 0 and ask > 0 else 0))
    base = _base_mark_mode(mode)
    if base == "mid":
        return mid if mid > 0 else None
    if side == "buy":
        return ask if ask > 0 else None
    return bid if bid > 0 else None


def package_credit(
    high_q: Dict[str, float],
    short_q: Dict[str, float],
    low_q: Dict[str, float],
    mark_mode: str,
    package_slip: float = 0.0,
) -> Optional[float]:
    """Open credit for long high / short 2x short / long low.

    ``package_slip`` models CBOE complex-order fills: credit = mid_credit - slip
    (avoids summing worst natural on all 4 legs). Ignored for ask_bid mode.
    """
    base = _base_mark_mode(mark_mode)
    slip = parse_package_slip(mark_mode, package_slip) if base == "mid" else 0.0
    h = _leg_mark(high_q, base, "buy")
    s = _leg_mark(short_q, base, "sell")
    lo = _leg_mark(low_q, base, "buy")
    if h is None or s is None or lo is None:
        return None
    credit = float(2.0 * s - h - lo)
    if base == "mid" and slip > 0:
        credit -= float(slip)
    return credit


def package_debit(
    high_q: Dict[str, float],
    short_q: Dict[str, float],
    low_q: Dict[str, float],
    mark_mode: str,
    package_slip: float = 0.0,
) -> Optional[float]:
    """Close debit (buy back shorts, sell longs).

    With ``package_slip`` on mid: debit = mid_debit + slip (COB-style).
    """
    base = _base_mark_mode(mark_mode)
    slip = parse_package_slip(mark_mode, package_slip) if base == "mid" else 0.0
    h = _leg_mark(high_q, base, "sell")
    s = _leg_mark(short_q, base, "buy")
    lo = _leg_mark(low_q, base, "sell")
    if h is None or s is None or lo is None:
        return None
    debit = float(2.0 * s - h - lo)
    if base == "mid" and slip > 0:
        debit += float(slip)
    return debit


def _bs_put_delta(spot: float, strike: float, dte: float, iv: float, r: float = DEFAULT_R) -> float:
    t = max(float(dte), 0.25) / 365.0
    g = calculate_greeks(float(spot), float(strike), t, float(r), float(iv), "P")
    return float(g["delta"])


def pick_bwb_strikes(
    *,
    client: ThetaDataClient,
    opt_symbol: str,
    expiration: date,
    day: date,
    spot: float,
    strikes: Sequence[float],
    narrow: float,
    wide: float,
    mark_mode: str,
    cache: Dict[Tuple[Any, ...], pd.DataFrame],
    target_delta: float = DEFAULT_TARGET_DELTA,
    credit_lo: float = DEFAULT_CREDIT_FRAC_LO,
    credit_hi: float = DEFAULT_CREDIT_FRAC_HI,
    delta_lo: float = DEFAULT_DELTA_LO,
    delta_hi: float = DEFAULT_DELTA_HI,
    package_slip: float = 0.0,
) -> Optional[Dict[str, Any]]:
    """Find short/high/low with credit in band; prefer near target delta.

    Credit band is checked on **pre-slip mid** (or ask_bid package). Fill credit
    returned already includes package_slip haircut when mid-based.
    """
    dte = (expiration - day).days
    if dte <= 0 or spot <= 0:
        return None

    base_mode = _base_mark_mode(mark_mode)
    slip = parse_package_slip(mark_mode, package_slip) if base_mode == "mid" else 0.0
    # Select strikes on mid (or ask_bid); apply slip only to fill credit
    select_mode = base_mode

    # Seed IV from ~ATM put mid
    atm = _nearest_listed(strikes, spot)
    if atm is None:
        return None
    atm_q = quote_on_day(
        client,
        symbol=opt_symbol,
        expiration=expiration,
        strike=float(atm),
        right="put",
        day=day,
        cache=cache,
    )
    if not atm_q:
        return None
    iv = estimate_iv(
        mid=float(atm_q["mid"]),
        spot=spot,
        strike=float(atm),
        dte_days=float(dte),
        right="put",
        r=DEFAULT_R,
        q=0.0,
    )
    if iv is None:
        iv = 0.18

    raw_k = strike_for_put_delta(spot, max(dte, 1) / 365.0, DEFAULT_R, iv, target_delta)
    # Candidate shorts: around target, then walk OTM/ITM for credit band
    seed = _nearest_listed(strikes, raw_k)
    if seed is None:
        return None
    ordered = sorted(strikes)
    try:
        idx = ordered.index(seed)
    except ValueError:
        idx = int(np.argmin([abs(x - seed) for x in ordered]))
    # Prefer slightly OTM first (lower strike) then both directions
    cand_idx = []
    for off in range(0, 25):
        for sign in (0, -1, 1) if off == 0 else (-1, 1):
            j = idx + sign * off
            if 0 <= j < len(ordered) and j not in cand_idx:
                cand_idx.append(j)

    best: Optional[Dict[str, Any]] = None
    best_score = 1e18
    credit_lo_abs = float(narrow) * float(credit_lo)
    credit_hi_abs = float(narrow) * float(credit_hi)

    for j in cand_idx:
        short_k = float(ordered[j])
        high_k = _nearest_listed(strikes, short_k + float(narrow))
        low_k = _nearest_listed(strikes, short_k - float(wide))
        if high_k is None or low_k is None:
            continue
        if abs(high_k - short_k - narrow) > 1.0 or abs(short_k - low_k - wide) > 1.0:
            # require near-exact wing widths (SPX 5-pt grid)
            continue
        if not (low_k < short_k < high_k):
            continue

        delta = _bs_put_delta(spot, short_k, float(dte), float(iv))
        if delta < delta_lo or delta > delta_hi:
            # still allow if credit lands — Carl scrolls delta for premium
            pass

        hq = quote_on_day(
            client, symbol=opt_symbol, expiration=expiration, strike=high_k, right="put", day=day, cache=cache
        )
        sq = quote_on_day(
            client, symbol=opt_symbol, expiration=expiration, strike=short_k, right="put", day=day, cache=cache
        )
        lq = quote_on_day(
            client, symbol=opt_symbol, expiration=expiration, strike=low_k, right="put", day=day, cache=cache
        )
        if not hq or not sq or not lq:
            continue
        credit = package_credit(hq, sq, lq, select_mode, package_slip=0.0)
        if credit is None or credit < credit_lo_abs or credit > credit_hi_abs:
            continue

        fill_credit = float(credit) - float(slip) if slip > 0 else float(credit)

        # score: distance from target delta, then mid-band credit
        mid_c = 0.5 * (credit_lo_abs + credit_hi_abs)
        score = abs(delta - target_delta) * 10.0 + abs(credit - mid_c)
        if score < best_score:
            best_score = score
            best = {
                "short_k": short_k,
                "high_k": float(high_k),
                "low_k": float(low_k),
                "credit": float(fill_credit),
                "mid_credit": float(credit),
                "delta": float(delta),
                "iv": float(iv),
                "high_q": hq,
                "short_q": sq,
                "low_q": lq,
            }
        # good enough near target delta
        if abs(delta - target_delta) <= 0.03 and credit_lo_abs <= credit <= credit_hi_abs:
            break

    return best


def book_stats(trades: Sequence[BwbTrade]) -> Dict[str, Any]:
    kept = [t for t in trades if not t.skipped and t.pnl is not None]
    if not kept:
        return {"n": 0}
    rors = np.array([float(t.ror_pct or 0.0) for t in kept], dtype=float)
    pnls = np.array([float(t.pnl) for t in kept], dtype=float)
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    gross_win = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(-losses.sum()) if len(losses) else 0.0
    pf = (gross_win / gross_loss) if gross_loss > 1e-9 else None
    hold = float(np.mean([t.hold_days or 0 for t in kept])) or 10.0
    n_per_year = 365.0 / hold
    sharpe = None
    if len(rors) >= 5 and rors.std() > 1e-12:
        sharpe = float((rors.mean() / rors.std()) * math.sqrt(n_per_year))
    reasons: Dict[str, int] = {}
    for t in kept:
        r = str(t.exit_reason or "unknown")
        reasons[r] = reasons.get(r, 0) + 1
    return {
        "n": len(kept),
        "win_rate_pct": round(100.0 * float((pnls > 0).mean()), 1),
        "mean_ror_pct": round(float(rors.mean()), 2),
        "med_ror_pct": round(float(np.median(rors)), 2),
        "mean_pnl": round(float(pnls.mean()), 2),
        "total_pnl": round(float(pnls.sum()), 2),
        "pf": None if pf is None else round(pf, 2),
        "avg_hold": round(hold, 1),
        "avg_credit": round(float(np.mean([t.credit for t in kept if t.credit])), 3),
        "sharpe_proxy": None if sharpe is None else round(sharpe, 2),
        "exit_reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
    }


def trades_to_frame(trades: Sequence[BwbTrade]) -> pd.DataFrame:
    rows = []
    for t in trades:
        rows.append(
            {
                "symbol": t.symbol,
                "entry_date": t.entry_date.isoformat() if t.entry_date else None,
                "exit_date": t.exit_date.isoformat() if t.exit_date else None,
                "expiration": t.expiration.isoformat() if t.expiration else None,
                "spot": t.spot,
                "high_k": t.high_k,
                "short_k": t.short_k,
                "low_k": t.low_k,
                "narrow": t.narrow,
                "wide": t.wide,
                "entry_delta": t.entry_delta,
                "credit": t.credit,
                "exit_debit": t.exit_debit,
                "tp_debit": t.tp_debit,
                "sl_debit": t.sl_debit,
                "max_risk": t.max_risk,
                "pnl": t.pnl,
                "ror_pct": t.ror_pct,
                "hold_days": t.hold_days,
                "exit_reason": t.exit_reason,
                "mark_mode": t.mark_mode,
                "skipped": t.skipped,
                "skip_reason": t.skip_reason,
                "dte_entry": t.dte_entry,
                "dte_exit": t.dte_exit,
            }
        )
    return pd.DataFrame(rows)


def run_bwb_book(
    *,
    start: date,
    end: date,
    underlier: str = "SPX",
    opt_symbol: str = "SPXW",
    mark_mode: str = "mid",
    narrow: float = DEFAULT_NARROW,
    wide: float = DEFAULT_WIDE,
    target_dte: int = DEFAULT_DTE,
    dte_tol: int = DEFAULT_DTE_TOL,
    min_dte_exit: int = DEFAULT_MIN_DTE_EXIT,
    credit_lo: float = DEFAULT_CREDIT_FRAC_LO,
    credit_hi: float = DEFAULT_CREDIT_FRAC_HI,
    use_stop: bool = True,
    stop_mode: str = STOP_MODE_DEBIT,
    spot_stop_buffer: float = 0.0,
    down_day_only: bool = False,
    trend_ma: Optional[int] = None,
    package_slip: float = 0.0,
    scan: str = "monday",
    max_concurrent: int = 1,
    commission_per_leg: float = DEFAULT_COMM_PER_LEG,
    multiplier: float = DEFAULT_MULT,
    client: Optional[ThetaDataClient] = None,
    spots: Optional[pd.DataFrame] = None,
    exps: Optional[List[date]] = None,
    cache: Optional[Dict[Tuple[Any, ...], pd.DataFrame]] = None,
    strike_cache: Optional[Dict[date, List[float]]] = None,
) -> BookResult:
    """Walk candidate days; enter one put BWB when credit band met; manage TP/SL/time.

    ``package_slip``: COB-style mid +/- slip (open credit -slip, close debit +slip).
    ``trend_ma``: if set (50/200), skip entries when spot < SMA(trend_ma).
    ``stop_mode``: ``debit`` (3x credit package), ``spot`` (close <= short_k + buffer),
    or ``none`` / use_stop=False.
    ``scan``: ``monday`` or ``daily`` (all sessions; combine with filters to thin).
    """
    client = client or get_thetadata_client()
    if not client.is_ready():
        raise RuntimeError("Theta Terminal not ready on :25503")

    slip = parse_package_slip(mark_mode, package_slip)
    base_mode = _base_mark_mode(mark_mode)
    ma_n = int(trend_ma) if trend_ma else 0
    warm_days = max(10, ma_n + 40 if ma_n else 10)
    warm_start = start - timedelta(days=warm_days * 2)
    spots = spots if spots is not None else load_index_eod(client, underlier, warm_start, end)
    if spots is None or spots.empty:
        raise RuntimeError(f"No spot history for {underlier}")

    sma: Optional[pd.Series] = None
    if ma_n > 0:
        sma = spots["close"].astype(float).rolling(ma_n, min_periods=ma_n).mean()

    exps = exps if exps is not None else list_expirations(client, opt_symbol)
    # SPXW weeklies for ~21 DTE; SPX root is mostly monthlies
    monthly_only = str(opt_symbol).upper() == "SPX"

    smode = str(stop_mode or STOP_MODE_DEBIT).strip().lower()
    if not use_stop:
        smode = STOP_MODE_NONE
    slip_tag = f"_slip{slip:g}" if slip > 0 and base_mode == "mid" else ""
    ma_tag = f"_ma{ma_n}" if ma_n > 0 else ""
    scan_tag = f"_{scan}" if scan != "monday" else ""
    if smode == STOP_MODE_SPOT:
        stop_tag = f"_spotsl{('p' + str(int(spot_stop_buffer))) if spot_stop_buffer else ''}"
    elif smode == STOP_MODE_NONE:
        stop_tag = "_nosl"
    else:
        stop_tag = "_sl"
    label = (
        f"{opt_symbol}_put_bwb_{int(narrow)}x{int(wide)}_dte{target_dte}_"
        f"{base_mode}{slip_tag}{stop_tag}"
        f"{'_downday' if down_day_only else ''}{ma_tag}{scan_tag}"
    )
    book = BookResult(label=label, mark_mode=mark_mode if slip <= 0 else f"{base_mode}_slip{slip:g}")
    cache = cache if cache is not None else {}
    strike_cache = strike_cache if strike_cache is not None else {}

    if scan == "monday":
        days = mondays_on_index(spots.index, start, end)
    else:
        days = [ts.date() for ts in spots.index if start <= ts.date() <= end]

    # Spot lookup for management (session date -> close)
    spot_by_day: Dict[date, float] = {}
    for ts, crow in spots.iterrows():
        d = ts.date() if hasattr(ts, "date") else pd.Timestamp(ts).date()
        c = crow["close"]
        spot_by_day[d] = float(c.iloc[0] if isinstance(c, pd.Series) else c)

    open_until: Optional[date] = None

    for day in days:
        if open_until is not None and day <= open_until:
            book.skipped.append({"date": str(day), "reason": REASON_BUSY})
            continue

        # spot row
        matches = spots.index[spots.index.normalize().date == day]
        if len(matches) == 0:
            book.skipped.append({"date": str(day), "reason": REASON_NO_SPOT})
            continue
        row = spots.loc[matches[0]]
        spot = float(row["close"] if not isinstance(row["close"], pd.Series) else row["close"].iloc[0])
        if down_day_only:
            prev_ix = spots.index[spots.index.normalize() < pd.Timestamp(day)]
            if len(prev_ix) == 0:
                book.skipped.append({"date": str(day), "reason": "no_prev"})
                continue
            prev_close = float(spots.loc[prev_ix[-1], "close"])
            if spot >= prev_close:
                book.skipped.append({"date": str(day), "reason": "not_down_day"})
                continue

        if sma is not None:
            ma_val = sma.loc[matches[0]]
            if pd.isna(ma_val) or spot < float(ma_val):
                book.skipped.append(
                    {"date": str(day), "reason": REASON_BELOW_MA, "ma": ma_n, "spot": spot}
                )
                continue

        exp = pick_expiry(exps, day, target_dte, dte_tol, monthly_only=monthly_only)
        if exp is None:
            book.skipped.append({"date": str(day), "reason": REASON_NO_EXP})
            continue

        if exp not in strike_cache:
            strike_cache[exp] = list_strikes(client, opt_symbol, exp)
        strikes = strike_cache[exp]
        if not strikes:
            book.skipped.append({"date": str(day), "reason": REASON_NO_STRIKES, "exp": str(exp)})
            continue

        pick = pick_bwb_strikes(
            client=client,
            opt_symbol=opt_symbol,
            expiration=exp,
            day=day,
            spot=spot,
            strikes=strikes,
            narrow=narrow,
            wide=wide,
            mark_mode=base_mode,
            cache=cache,
            credit_lo=credit_lo,
            credit_hi=credit_hi,
            package_slip=slip,
        )
        if pick is None:
            book.skipped.append({"date": str(day), "reason": REASON_NO_CREDIT, "exp": str(exp)})
            continue

        credit = float(pick["credit"])
        short_k = float(pick["short_k"])
        tp_debit = float(narrow) * float(DEFAULT_TP_FRAC)
        sl_debit = float(DEFAULT_SL_MULT) * credit
        spot_trigger = short_k + float(spot_stop_buffer)
        max_risk = max(float(narrow) - credit, 1e-6)
        # round-trip commission: 4 legs open + 4 close
        comm = 8.0 * float(commission_per_leg)
        fill_mode = f"{base_mode}_slip{slip:g}" if slip > 0 and base_mode == "mid" else base_mode

        trade = BwbTrade(
            symbol=opt_symbol,
            entry_date=day,
            expiration=exp,
            spot=spot,
            short_k=short_k,
            high_k=float(pick["high_k"]),
            low_k=float(pick["low_k"]),
            narrow=float(narrow),
            wide=float(wide),
            entry_delta=float(pick["delta"]),
            credit=credit,
            tp_debit=tp_debit,
            sl_debit=sl_debit if smode == STOP_MODE_DEBIT else (
                spot_trigger if smode == STOP_MODE_SPOT else None
            ),
            max_risk=max_risk,
            mark_mode=fill_mode,
            dte_entry=(exp - day).days,
        )

        # Manage on session days until exit
        session_days = [
            ts.date()
            for ts in spots.index
            if day < ts.date() <= min(exp, end + timedelta(days=5))
        ]
        exit_day = None
        exit_debit = None
        exit_reason = EXIT_NO_MARK
        for cur in session_days:
            dte_left = (exp - cur).days
            cur_spot = spot_by_day.get(cur)
            hq = quote_on_day(
                client,
                symbol=opt_symbol,
                expiration=exp,
                strike=float(pick["high_k"]),
                right="put",
                day=cur,
                cache=cache,
            )
            sq = quote_on_day(
                client,
                symbol=opt_symbol,
                expiration=exp,
                strike=float(pick["short_k"]),
                right="put",
                day=cur,
                cache=cache,
            )
            lq = quote_on_day(
                client,
                symbol=opt_symbol,
                expiration=exp,
                strike=float(pick["low_k"]),
                right="put",
                day=cur,
                cache=cache,
            )
            if not (hq and sq and lq):
                continue
            debit = package_debit(hq, sq, lq, base_mode, package_slip=slip)
            if debit is None:
                continue

            # Spot stop first (structural risk zone) — bypasses quote-width noise
            if smode == STOP_MODE_SPOT and cur_spot is not None and cur_spot <= spot_trigger:
                exit_day, exit_debit, exit_reason = cur, debit, EXIT_SL_SPOT
                break
            if debit <= tp_debit:
                exit_day, exit_debit, exit_reason = cur, debit, EXIT_TP
                break
            if smode == STOP_MODE_DEBIT and debit >= sl_debit:
                exit_day, exit_debit, exit_reason = cur, debit, EXIT_SL
                break
            if dte_left <= min_dte_exit:
                exit_day, exit_debit, exit_reason = cur, debit, EXIT_TIME
                break
            if cur == exp:
                exit_day, exit_debit, exit_reason = cur, debit, EXIT_EXPIRY
                break

        if exit_day is None or exit_debit is None:
            trade.skipped = True
            trade.skip_reason = EXIT_NO_MARK
            book.trades.append(trade)
            book.skipped.append({"date": str(day), "reason": EXIT_NO_MARK})
            continue

        pnl_pts = credit - float(exit_debit)
        pnl = pnl_pts * float(multiplier) - comm
        ror = 100.0 * pnl / (max_risk * float(multiplier))
        trade.exit_date = exit_day
        trade.exit_debit = float(exit_debit)
        trade.exit_reason = exit_reason
        trade.hold_days = (exit_day - day).days
        trade.dte_exit = (exp - exit_day).days
        trade.pnl = float(pnl)
        trade.ror_pct = float(ror)
        book.trades.append(trade)
        open_until = exit_day
        LOG.info(
            "%s entry=%s exit=%s credit=%.2f debit=%.2f pnl=$%.0f ror=%.1f%% %s",
            opt_symbol,
            day,
            exit_day,
            credit,
            exit_debit,
            pnl,
            ror,
            exit_reason,
        )

    return book
