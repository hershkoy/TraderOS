"""45 DTE SPX bull put credit spread claim-check (listed Theta EOD).

Mechanical rules (retail VRP vertical, not a YouTube transcript):
  - Short put near 20-25 delta (target 0.22)
  - Long put ``width`` points lower (default 25)
  - Open on a Friday expiry with 40-50 DTE (target 45)
  - Take profit when buy-to-close debit <= 50% of fill credit
  - Stop when buy-to-close debit >= 2x fill credit
    (loss is about 1x credit, before commissions)
  - Time-stop at 21 DTE left
  - Friction: mid package slip (default 0.05 index points) plus $0.65 per leg x 4

Marks are Theta EOD NBBO. Same-day entry is the close; management starts the
next session. Intraday touches of the 50% or 2x levels are not visible.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from utils.api.thetadata.thetadata_client import ThetaDataClient, get_thetadata_client
from utils.options.greeks import calculate_greeks
from utils.research.forward_factor import (
    DEFAULT_Q,
    DEFAULT_R,
    estimate_iv,
    fetch_eod,
    list_expirations,
    list_strikes,
    mondays_on_index,
    pick_expiry,
    quote_on_day,
)
from utils.research.last_15m_short_put_bs import strike_for_put_delta
from utils.research.spx_put_bwb import load_index_eod

LOG = logging.getLogger(__name__)

CLAIMED: Dict[str, Any] = {
    "source": "retail 45 DTE index bull put (VRP); manual $100k account sketch",
    "structure": "SPX/SPXW short ~22 delta put, long 25 pts lower",
    "dte_open": "40-50",
    "tp": "buy back at 50% of fill credit",
    "sl": "buy back at 2x fill credit",
    "time_stop_dte": 21,
    "friction": "0.05 package slip + 0.65/leg x 4",
    "frequency_claim": "about 1 entry per week if stacked; one-at-a-time holds ~3 weeks",
}

DEFAULT_TARGET_DELTA = 0.22
DEFAULT_DELTA_TOL = 0.04
DEFAULT_WIDTH = 25.0
DEFAULT_DTE = 45
DEFAULT_DTE_TOL = 5
DEFAULT_MIN_DTE_EXIT = 21
DEFAULT_TP_FRAC = 0.50
DEFAULT_SL_MULT = 2.0
DEFAULT_MIN_CREDIT = 1.00
DEFAULT_COMM_PER_LEG = 0.65
DEFAULT_MULT = 100.0
DEFAULT_PACKAGE_SLIP = 0.05
DEFAULT_MAX_CONCURRENT = 1

EXIT_TP = "tp"
EXIT_SL = "sl"
EXIT_TIME = "time_dte"
EXIT_EXPIRY = "expiry"
EXIT_NO_MARK = "no_mark"

REASON_OK = "ok"
REASON_NO_EXP = "no_expiry"
REASON_NO_SPOT = "no_spot"
REASON_NO_STRIKES = "no_strikes"
REASON_NO_CREDIT = "credit_below_min"
REASON_NO_WING = "no_wing"
REASON_NO_DELTA = "no_delta"
REASON_NO_ATM = "no_atm"
REASON_BUSY = "max_concurrent"
REASON_WINDOW = "exit_after_sample"


def _base_mark_mode(mark_mode: str) -> str:
    m = str(mark_mode).strip().lower()
    if m.startswith("mid_slip"):
        return "mid"
    if m in ("natural", "askbid"):
        return "ask_bid"
    return m


def parse_package_slip(mark_mode: str, package_slip: float = 0.0) -> float:
    if package_slip and package_slip > 0:
        return float(package_slip)
    m = str(mark_mode).strip().lower()
    if m.startswith("mid_slip"):
        rest = m[len("mid_slip") :]
        if not rest:
            return float(DEFAULT_PACKAGE_SLIP)
        return float(rest)
    return 0.0


def _nearest_listed(strikes: Sequence[float], target: float) -> Optional[float]:
    if not strikes:
        return None
    return float(min(strikes, key=lambda s: abs(float(s) - float(target))))


def _quote_ok(q: Optional[Dict[str, float]]) -> bool:
    if not q:
        return False
    try:
        bid = float(q.get("bid") or 0)
        ask = float(q.get("ask") or 0)
    except (TypeError, ValueError):
        return False
    if bid <= 0 or ask <= 0 or ask + 1e-9 < bid:
        return False
    mid = float(q.get("mid") or (0.5 * (bid + ask)))
    return mid > 0


def vertical_credit(
    short_q: Dict[str, float],
    long_q: Dict[str, float],
    mark_mode: str = "mid",
    package_slip: float = 0.0,
) -> Optional[float]:
    """Open credit. Mid mode subtracts package slip. ask_bid sells the bid and buys the ask."""
    if not _quote_ok(short_q) or not _quote_ok(long_q):
        return None
    base = _base_mark_mode(mark_mode)
    if base == "ask_bid":
        credit = float(short_q["bid"]) - float(long_q["ask"])
    else:
        s_mid = float(short_q.get("mid") or 0.5 * (float(short_q["bid"]) + float(short_q["ask"])))
        l_mid = float(long_q.get("mid") or 0.5 * (float(long_q["bid"]) + float(long_q["ask"])))
        credit = s_mid - l_mid - float(package_slip or 0.0)
    return float(credit)


def vertical_debit(
    short_q: Dict[str, float],
    long_q: Dict[str, float],
    mark_mode: str = "mid",
    package_slip: float = 0.0,
) -> Optional[float]:
    """Close debit. Mid mode adds package slip. ask_bid buys the ask and sells the bid."""
    if not _quote_ok(short_q) or not _quote_ok(long_q):
        return None
    base = _base_mark_mode(mark_mode)
    if base == "ask_bid":
        debit = float(short_q["ask"]) - float(long_q["bid"])
    else:
        s_mid = float(short_q.get("mid") or 0.5 * (float(short_q["bid"]) + float(short_q["ask"])))
        l_mid = float(long_q.get("mid") or 0.5 * (float(long_q["bid"]) + float(long_q["ask"])))
        debit = s_mid - l_mid + float(package_slip or 0.0)
    return float(debit)


def half_spread_cross(short_q: Dict[str, float], long_q: Dict[str, float]) -> Optional[float]:
    """Index points given up vs mid to cross both legs (half of each bid-ask)."""
    if not _quote_ok(short_q) or not _quote_ok(long_q):
        return None
    return 0.5 * (
        (float(short_q["ask"]) - float(short_q["bid"]))
        + (float(long_q["ask"]) - float(long_q["bid"]))
    )


def trade_pnl(
    credit: float,
    debit: float,
    width: float,
    *,
    multiplier: float = DEFAULT_MULT,
    commission_per_leg: float = DEFAULT_COMM_PER_LEG,
    legs_round_turn: int = 4,
) -> Tuple[float, float, float]:
    """Return (pnl_dollars, ror_pct, max_risk_points).

    RoR is P&L / defined risk. Defined risk is (width - credit) x multiplier.
    """
    pnl = (float(credit) - float(debit)) * float(multiplier) - float(legs_round_turn) * float(
        commission_per_leg
    )
    max_risk_pts = max(float(width) - float(credit), 1e-6)
    ror = 100.0 * pnl / (max_risk_pts * float(multiplier))
    return float(pnl), float(ror), float(max_risk_pts)


def walk_spread_exit(
    *,
    session_days: Sequence[date],
    debit_on: Callable[[date], Optional[float]],
    expiration: date,
    credit: float,
    tp_frac: float = DEFAULT_TP_FRAC,
    sl_mult: float = DEFAULT_SL_MULT,
    min_dte_exit: int = DEFAULT_MIN_DTE_EXIT,
    use_stop: bool = True,
) -> Optional[Tuple[date, float, str]]:
    """First session that hits SL, then TP, then the 21 DTE time stop.

    A missing mark is skipped. The first later mark with DTE left <= min_dte_exit
    still counts as the time stop unless that mark is already through SL or TP.
    """
    tp_debit = float(credit) * float(tp_frac)
    sl_debit = float(credit) * float(sl_mult)
    for cur in session_days:
        if cur > expiration:
            break
        debit = debit_on(cur)
        if debit is None or not math.isfinite(float(debit)):
            continue
        debit_f = float(debit)
        dte_left = (expiration - cur).days
        if use_stop and debit_f >= sl_debit:
            return cur, debit_f, EXIT_SL
        if debit_f <= tp_debit:
            return cur, debit_f, EXIT_TP
        if dte_left <= int(min_dte_exit):
            return cur, debit_f, EXIT_TIME
        if cur == expiration:
            return cur, debit_f, EXIT_EXPIRY
    return None


def _abs_put_delta(spot: float, strike: float, dte_days: float, iv: float, r: float, q: float) -> float:
    t = max(float(dte_days), 0.25) / 365.0
    g = calculate_greeks(float(spot), float(strike), t, float(r), float(iv), "P", q=float(q))
    return abs(float(g["delta"]))


def pick_vertical_strikes(
    *,
    spot: float,
    strikes: Sequence[float],
    dte: int,
    width: float,
    quote_fn: Callable[[float], Optional[Dict[str, float]]],
    target_delta: float = DEFAULT_TARGET_DELTA,
    delta_tol: float = DEFAULT_DELTA_TOL,
    min_credit: float = DEFAULT_MIN_CREDIT,
    mark_mode: str = "mid",
    package_slip: float = 0.0,
    r: float = DEFAULT_R,
    q: float = DEFAULT_Q,
    max_steps: int = 8,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Snap a listed short put near ``target_delta`` and a long put ``width`` below.

    Delta uses each candidate's own mid IV (put skew), seeded from the ATM put.
    Strike choice is on delta, then the fill credit (slip or natural) must clear
    ``min_credit``.
    """
    if dte <= 0 or spot <= 0 or not strikes:
        return None, REASON_NO_STRIKES
    listed = sorted({float(k) for k in strikes})
    otm = [k for k in listed if k < float(spot)]
    if not otm:
        return None, REASON_NO_DELTA

    atm = _nearest_listed(listed, spot)
    iv = None
    if atm is not None:
        atm_q = quote_fn(float(atm))
        if _quote_ok(atm_q):
            iv = estimate_iv(
                mid=float(atm_q["mid"]),
                spot=float(spot),
                strike=float(atm),
                dte_days=float(dte),
                right="put",
                r=float(r),
                q=float(q),
            )
    if iv is None:
        iv = 0.18

    seed = strike_for_put_delta(
        float(spot), max(int(dte), 1) / 365.0, float(r), float(iv), -abs(float(target_delta))
    )
    start = _nearest_listed(otm, seed)
    if start is None:
        return None, REASON_NO_DELTA
    try:
        idx = otm.index(start)
    except ValueError:
        idx = int(np.argmin([abs(k - start) for k in otm]))

    best: Optional[Dict[str, Any]] = None
    best_err = 1e9
    j = idx
    seen: set = set()
    for _ in range(int(max_steps)):
        if j in seen:
            break
        seen.add(j)
        k = float(otm[j])
        quote = quote_fn(k)
        if not _quote_ok(quote):
            break
        ivk = estimate_iv(
            mid=float(quote["mid"]),
            spot=float(spot),
            strike=k,
            dte_days=float(dte),
            right="put",
            r=float(r),
            q=float(q),
        )
        if ivk is None:
            break
        delta = _abs_put_delta(float(spot), k, float(dte), float(ivk), float(r), float(q))
        err = abs(delta - abs(float(target_delta)))
        if err < best_err:
            best_err = err
            best = {"short_k": k, "delta": float(delta), "iv": float(ivk), "short_q": quote}
        if err <= 0.02:
            break
        # Higher strike -> larger put delta. Step OTM if we are too close.
        if delta > abs(float(target_delta)):
            if j == 0:
                break
            j -= 1
        else:
            if j >= len(otm) - 1:
                break
            j += 1

    if best is None or best_err > float(delta_tol):
        return None, REASON_NO_DELTA

    short_k = float(best["short_k"])
    wing_target = short_k - float(width)
    exact = [k for k in listed if abs(k - wing_target) < 0.02]
    long_k = float(exact[0]) if exact else _nearest_listed([k for k in listed if k < short_k], wing_target)
    if long_k is None or long_k >= short_k:
        return None, REASON_NO_WING
    actual_width = short_k - long_k
    if abs(actual_width - float(width)) > 5.01:
        return None, REASON_NO_WING
    long_q = quote_fn(float(long_k))
    if not _quote_ok(long_q):
        return None, REASON_NO_WING

    slip = 0.0 if _base_mark_mode(mark_mode) == "ask_bid" else float(package_slip or 0.0)
    mid_credit = vertical_credit(best["short_q"], long_q, "mid", 0.0)
    fill = vertical_credit(best["short_q"], long_q, mark_mode, slip)
    if mid_credit is None or fill is None or fill < float(min_credit):
        return None, REASON_NO_CREDIT
    best.update(
        {
            "long_k": float(long_k),
            "long_q": long_q,
            "width": float(actual_width),
            "mid_credit": float(mid_credit),
            "credit": float(fill),
        }
    )
    return best, REASON_OK


@dataclass
class SpreadTrade:
    symbol: str
    entry_date: date
    exit_date: Optional[date] = None
    expiration: Optional[date] = None
    spot: Optional[float] = None
    short_k: Optional[float] = None
    long_k: Optional[float] = None
    width: float = DEFAULT_WIDTH
    entry_delta: Optional[float] = None
    credit: Optional[float] = None
    mid_credit: Optional[float] = None
    exit_debit: Optional[float] = None
    cross_cost: Optional[float] = None
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
    trades: List[SpreadTrade] = field(default_factory=list)
    skipped: List[Dict[str, Any]] = field(default_factory=list)


def _year_stats(trades: Sequence[SpreadTrade]) -> Dict[str, Any]:
    buckets: Dict[int, List[SpreadTrade]] = {}
    for t in trades:
        buckets.setdefault(int(t.entry_date.year), []).append(t)
    out: Dict[str, Any] = {}
    for year in sorted(buckets):
        group = buckets[year]
        pnls = np.array([float(t.pnl or 0.0) for t in group], dtype=float)
        wins = pnls[pnls > 0]
        losses = pnls[pnls <= 0]
        gross_win = float(wins.sum()) if len(wins) else 0.0
        gross_loss = float(-losses.sum()) if len(losses) else 0.0
        pf = (gross_win / gross_loss) if gross_loss > 1e-9 else None
        out[str(year)] = {
            "n": len(group),
            "win_rate_pct": round(100.0 * float((pnls > 0).mean()), 1),
            "mean_ror_pct": round(float(np.mean([t.ror_pct or 0.0 for t in group])), 2),
            "pf": None if pf is None else round(pf, 2),
            "pnl": round(float(pnls.sum()), 2),
        }
    return out


def book_stats(trades: Sequence[SpreadTrade], *, start_equity: float = 100_000.0) -> Dict[str, Any]:
    kept = [t for t in trades if not t.skipped and t.pnl is not None and t.exit_date is not None]
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
    if len(rors) >= 5 and float(rors.std()) > 1e-12:
        sharpe = float((rors.mean() / rors.std()) * math.sqrt(n_per_year))
    reasons: Dict[str, int] = {}
    for t in kept:
        key = str(t.exit_reason or "unknown")
        reasons[key] = reasons.get(key, 0) + 1
    ordered = sorted(kept, key=lambda t: (t.exit_date, t.entry_date))
    eq = float(start_equity)
    peak = eq
    mdd = 0.0
    for t in ordered:
        eq += float(t.pnl or 0.0)
        peak = max(peak, eq)
        if peak > 0:
            mdd = max(mdd, (peak - eq) / peak)
    crosses = [float(t.cross_cost) for t in kept if t.cross_cost is not None]
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
        "avg_delta": round(float(np.mean([t.entry_delta for t in kept if t.entry_delta])), 3),
        "med_cross": None if not crosses else round(float(np.median(crosses)), 3),
        "p90_cross": None if not crosses else round(float(np.quantile(crosses, 0.9)), 3),
        "sharpe_proxy": None if sharpe is None else round(sharpe, 2),
        "exit_reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
        "years": _year_stats(kept),
        "end_equity_1lot": round(eq, 2),
        "mdd_pct_1lot": round(100.0 * mdd, 2),
        "span_days": (ordered[-1].exit_date - ordered[0].entry_date).days if ordered else 0,
    }


def trades_to_frame(trades: Sequence[SpreadTrade]) -> pd.DataFrame:
    rows = []
    for t in trades:
        rows.append(
            {
                "symbol": t.symbol,
                "entry_date": t.entry_date.isoformat() if t.entry_date else None,
                "exit_date": t.exit_date.isoformat() if t.exit_date else None,
                "expiration": t.expiration.isoformat() if t.expiration else None,
                "spot": t.spot,
                "short_k": t.short_k,
                "long_k": t.long_k,
                "width": t.width,
                "entry_delta": t.entry_delta,
                "credit": t.credit,
                "mid_credit": t.mid_credit,
                "exit_debit": t.exit_debit,
                "cross_cost": t.cross_cost,
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


def _fetch_range_marks(
    client: ThetaDataClient,
    *,
    symbol: str,
    expiration: date,
    strike: float,
    start: date,
    end: date,
    cache: Dict[Tuple[Any, ...], pd.DataFrame],
) -> Dict[date, Dict[str, float]]:
    if end < start:
        return {}
    last_err: Optional[Exception] = None
    df = pd.DataFrame()
    for attempt in range(4):
        try:
            df = fetch_eod(
                client,
                symbol=symbol,
                expiration=expiration,
                strike=float(strike),
                right="put",
                start=start,
                end=end,
                cache=cache,
            )
            last_err = None
            break
        except Exception as exc:
            last_err = exc
            msg = str(exc)
            if "429" in msg or "478" in msg or "Too Many" in msg or "timed out" in msg.lower():
                time.sleep(2.0 * (attempt + 1))
                continue
            LOG.warning("range EOD hard-fail %s %s %s: %s", symbol, expiration, strike, exc)
            return {}
    if last_err is not None:
        LOG.warning("range EOD failed %s %s %s: %s", symbol, expiration, strike, last_err)
        return {}
    out: Dict[date, Dict[str, float]] = {}
    if df is None or df.empty or "session_date" not in df.columns:
        return out
    for _, row in df.iterrows():
        d = row["session_date"]
        if not isinstance(d, date):
            d = pd.Timestamp(d).date()
        q = {
            "bid": float(row["bid"]),
            "ask": float(row["ask"]),
            "mid": float(row["mid"]),
        }
        if _quote_ok(q):
            out[d] = q
    return out


def _spot_map(spots: pd.DataFrame) -> Dict[date, float]:
    out: Dict[date, float] = {}
    for ts, crow in spots.iterrows():
        d = ts.date() if hasattr(ts, "date") else pd.Timestamp(ts).date()
        c = crow["close"]
        out[d] = float(c.iloc[0] if isinstance(c, pd.Series) else c)
    return out


def _row_on(spots: pd.DataFrame, day: date) -> Optional[pd.Series]:
    matches = spots.index[spots.index.normalize().map(lambda t: t.date()) == day]
    if len(matches) == 0:
        return None
    row = spots.loc[matches[0]]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[-1]
    return row


def run_bull_put_book(
    *,
    start: date,
    end: date,
    underlier: str = "SPX",
    opt_symbol: str = "SPXW",
    mark_mode: str = "mid",
    width: float = DEFAULT_WIDTH,
    target_dte: int = DEFAULT_DTE,
    dte_tol: int = DEFAULT_DTE_TOL,
    min_dte_exit: int = DEFAULT_MIN_DTE_EXIT,
    target_delta: float = DEFAULT_TARGET_DELTA,
    delta_tol: float = DEFAULT_DELTA_TOL,
    min_credit: float = DEFAULT_MIN_CREDIT,
    tp_frac: float = DEFAULT_TP_FRAC,
    sl_mult: float = DEFAULT_SL_MULT,
    use_stop: bool = True,
    package_slip: float = 0.0,
    scan: str = "monday",
    max_concurrent: int = DEFAULT_MAX_CONCURRENT,
    max_entries: int = 0,
    commission_per_leg: float = DEFAULT_COMM_PER_LEG,
    multiplier: float = DEFAULT_MULT,
    client: Optional[ThetaDataClient] = None,
    spots: Optional[pd.DataFrame] = None,
    exps: Optional[List[date]] = None,
    cache: Optional[Dict[Tuple[Any, ...], pd.DataFrame]] = None,
    strike_cache: Optional[Dict[date, List[float]]] = None,
    picker: Optional[Callable[..., Tuple[Optional[Dict[str, Any]], str]]] = None,
    quote_lookup: Optional[Callable[[date, float, date], Optional[Dict[str, float]]]] = None,
) -> BookResult:
    """Walk Monday (or daily) sessions. One short put vertical at a time unless stacked.

    ``quote_lookup(expiration, strike, day)`` and ``picker`` exist so tests can
    run without Theta Terminal. Live runs leave both unset.
    """
    if quote_lookup is None:
        client = client or get_thetadata_client()
        if not client.is_ready():
            raise RuntimeError("Theta Terminal not ready on :25503")

    slip = parse_package_slip(mark_mode, package_slip)
    base_mode = _base_mark_mode(mark_mode)
    fill_slip = 0.0 if base_mode == "ask_bid" else slip
    monthly_only = str(opt_symbol).upper() == "SPX"

    spots = spots if spots is not None else load_index_eod(client, underlier, start - timedelta(days=10), end)
    if spots is None or spots.empty:
        raise RuntimeError(f"No spot history for {underlier}")
    exps = exps if exps is not None else list_expirations(client, opt_symbol)
    cache = cache if cache is not None else {}
    strike_cache = strike_cache if strike_cache is not None else {}
    choose = picker or pick_vertical_strikes

    slip_tag = f"_slip{fill_slip:g}" if fill_slip > 0 and base_mode == "mid" else ""
    stop_tag = "_sl2x" if use_stop else "_nostop"
    scan_tag = "" if scan == "monday" else f"_{scan}"
    conc_tag = "" if int(max_concurrent) <= 1 else f"_c{int(max_concurrent)}"
    label = (
        f"{opt_symbol}_bullput_w{int(width)}_dte{target_dte}_{base_mode}{slip_tag}"
        f"{stop_tag}{scan_tag}{conc_tag}"
    )
    book = BookResult(
        label=label,
        mark_mode=f"{base_mode}_slip{fill_slip:g}" if fill_slip > 0 else base_mode,
    )

    if scan == "monday":
        days = mondays_on_index(spots.index, start, end)
    else:
        days = [ts.date() for ts in spots.index if start <= ts.date() <= end]

    spot_by_day = _spot_map(spots)
    open_exits: List[date] = []
    n_entries = 0

    for i, day in enumerate(days):
        if i and i % 20 == 0:
            LOG.info("scan %s/%s entries %s", i, len(days), n_entries)
        open_exits = [x for x in open_exits if x >= day]
        if len(open_exits) >= int(max_concurrent):
            book.skipped.append({"date": str(day), "reason": REASON_BUSY})
            continue
        if max_entries and n_entries >= int(max_entries):
            break

        row = _row_on(spots, day)
        if row is None or day not in spot_by_day:
            book.skipped.append({"date": str(day), "reason": REASON_NO_SPOT})
            continue
        spot = float(spot_by_day[day])

        exp = pick_expiry(exps, day, target_dte, dte_tol, friday_only=True, monthly_only=monthly_only)
        if exp is None:
            book.skipped.append({"date": str(day), "reason": REASON_NO_EXP})
            continue
        if (exp - end).days > int(min_dte_exit):
            book.skipped.append({"date": str(day), "reason": REASON_WINDOW, "exp": str(exp)})
            continue

        if exp not in strike_cache:
            if quote_lookup is not None:
                strike_cache[exp] = []
            else:
                strike_cache[exp] = list_strikes(client, opt_symbol, exp)
        strikes = strike_cache[exp]
        if picker is None and not strikes:
            book.skipped.append({"date": str(day), "reason": REASON_NO_STRIKES, "exp": str(exp)})
            continue

        def _quote_fn(strike: float, _exp: date = exp, _day: date = day) -> Optional[Dict[str, float]]:
            if quote_lookup is not None:
                return quote_lookup(_exp, float(strike), _day)
            return quote_on_day(
                client,
                symbol=opt_symbol,
                expiration=_exp,
                strike=float(strike),
                right="put",
                day=_day,
                cache=cache,
            )

        picked, reason = choose(
            spot=spot,
            strikes=strikes,
            dte=(exp - day).days,
            width=float(width),
            quote_fn=_quote_fn,
            target_delta=float(target_delta),
            delta_tol=float(delta_tol),
            min_credit=float(min_credit),
            mark_mode=base_mode,
            package_slip=fill_slip,
        )
        if not picked:
            book.skipped.append({"date": str(day), "reason": reason, "exp": str(exp)})
            continue

        credit = vertical_credit(picked["short_q"], picked["long_q"], base_mode, fill_slip)
        mid_credit = vertical_credit(picked["short_q"], picked["long_q"], "mid", 0.0)
        if credit is None or credit < float(min_credit):
            book.skipped.append({"date": str(day), "reason": REASON_NO_CREDIT, "exp": str(exp)})
            continue

        short_k = float(picked["short_k"])
        long_k = float(picked["long_k"])
        actual_width = float(picked.get("width") or (short_k - long_k))
        cross = half_spread_cross(picked["short_q"], picked["long_q"])
        fill_mode = f"{base_mode}_slip{fill_slip:g}" if fill_slip > 0 else base_mode

        session_days = [
            ts.date()
            for ts in spots.index
            if day < ts.date() <= min(exp, end + timedelta(days=5))
        ]
        if quote_lookup is not None:
            short_marks = {d: quote_lookup(exp, short_k, d) for d in session_days}
            long_marks = {d: quote_lookup(exp, long_k, d) for d in session_days}
        else:
            path_end = min(exp, end + timedelta(days=3))
            short_marks = _fetch_range_marks(
                client,
                symbol=opt_symbol,
                expiration=exp,
                strike=short_k,
                start=day + timedelta(days=1),
                end=path_end,
                cache=cache,
            )
            long_marks = _fetch_range_marks(
                client,
                symbol=opt_symbol,
                expiration=exp,
                strike=long_k,
                start=day + timedelta(days=1),
                end=path_end,
                cache=cache,
            )

        def _debit_on(cur: date) -> Optional[float]:
            sq = short_marks.get(cur)
            lq = long_marks.get(cur)
            if not sq or not lq:
                return None
            return vertical_debit(sq, lq, base_mode, fill_slip)

        walked = walk_spread_exit(
            session_days=session_days,
            debit_on=_debit_on,
            expiration=exp,
            credit=float(credit),
            tp_frac=float(tp_frac),
            sl_mult=float(sl_mult),
            min_dte_exit=int(min_dte_exit),
            use_stop=bool(use_stop),
        )
        trade = SpreadTrade(
            symbol=opt_symbol,
            entry_date=day,
            expiration=exp,
            spot=spot,
            short_k=short_k,
            long_k=long_k,
            width=actual_width,
            entry_delta=float(picked.get("delta") or 0.0) or None,
            credit=float(credit),
            mid_credit=None if mid_credit is None else float(mid_credit),
            cross_cost=cross,
            mark_mode=fill_mode,
            dte_entry=(exp - day).days,
        )
        if walked is None:
            trade.skipped = True
            trade.skip_reason = EXIT_NO_MARK
            book.trades.append(trade)
            book.skipped.append({"date": str(day), "reason": EXIT_NO_MARK, "exp": str(exp)})
            continue

        exit_day, exit_debit, exit_reason = walked
        pnl, ror, max_risk_pts = trade_pnl(
            float(credit),
            float(exit_debit),
            actual_width,
            multiplier=float(multiplier),
            commission_per_leg=float(commission_per_leg),
        )
        trade.exit_date = exit_day
        trade.exit_debit = float(exit_debit)
        trade.exit_reason = exit_reason
        trade.hold_days = (exit_day - day).days
        trade.dte_exit = (exp - exit_day).days
        trade.max_risk = float(max_risk_pts)
        trade.pnl = float(pnl)
        trade.ror_pct = float(ror)
        book.trades.append(trade)
        open_exits.append(exit_day)
        n_entries += 1
        LOG.info(
            "%s entry=%s exit=%s exp=%s credit=%.2f debit=%.2f pnl=$%.0f ror=%.1f%% %s dlt=%.2f",
            opt_symbol,
            day,
            exit_day,
            exp,
            credit,
            exit_debit,
            pnl,
            ror,
            exit_reason,
            float(picked.get("delta") or 0.0),
        )

    return book
