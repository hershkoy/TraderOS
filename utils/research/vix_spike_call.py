"""VIX spike call credit-spread claim-check (listed Theta EOD).

Mechanical rules from the 2026-10-03 sketch:
  - Enter when cash VIX closes at or above a panic level (default 32)
  - Standard Wednesday VIX monthly, 30-60 calendar DTE (nearest)
  - Short call about 10 points above the reference, long call 10 points higher
  - Reference is cash VIX (the pasted code) or the parity-implied future
  - No stop. Take profit when the buyback debit is 40% of the fill credit
    (60% of the credit has decayed)
  - Otherwise mark the spread on the first session with <= 3 calendar days
    left, or on the last session still inside the sample
  - Friction: mid package slip (default 0.10) plus $0.65 per leg x 4

VIX options settle to the special opening quotation on expiration morning.
This harness never sees that print. The expiry exit is the prior EOD NBBO.
Marks are Theta EOD. Management starts the session after the signal close.
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

from utils.api.thetadata.thetadata_client import ThetaDataClient
from utils.research.forward_factor import fetch_eod, list_strikes, quote_on_day
from utils.research.spx_bull_put import (
    half_spread_cross,
    trade_pnl,
    vertical_credit,
    vertical_debit,
)

LOG = logging.getLogger(__name__)

CLAIMED: Dict[str, Any] = {
    "source": "VIX spike call credit spread; manual $100k account sketch",
    "structure": "short call ~10 pts above VIX, long call 10 pts higher",
    "trigger": "cash VIX close >= 32, fresh cross, one at a time",
    "dte": "30-60, standard Wednesday monthly",
    "tp": "buy back when debit <= 40% of fill credit",
    "sl": "none; defined risk is width minus credit",
    "friction": "0.10 package slip + 0.65/leg x 4",
    "frequency_claim": "about 2 to 5 entries per year",
}

DEFAULT_TRIGGER = 32.0
DEFAULT_OTM = 10.0
DEFAULT_WIDTH = 10.0
DEFAULT_STRIKE_STEP = 5.0
DEFAULT_DTE_MIN = 30
DEFAULT_DTE_MAX = 60
DEFAULT_MIN_CREDIT = 0.80
DEFAULT_TP_REMAIN = 0.40
DEFAULT_EXPIRY_WEEK_DAYS = 3
DEFAULT_PACKAGE_SLIP = 0.10
DEFAULT_COMM_PER_LEG = 0.65
DEFAULT_MULT = 100.0
DEFAULT_RATE = 0.04
DEFAULT_RISK_BUDGET = 15_000.0
# A stuck 10-point ask (bid still a few dimes) makes the vertical mid negative.
DEFAULT_MAX_LEG_SPREAD = 3.0
# Call spread plus put spread used for put-call parity. Wider than this is not a future.
DEFAULT_MAX_PARITY_WIDTH = 2.0

EXIT_TP = "tp_60"
EXIT_EXPIRY = "expiry_week"
EXIT_SAMPLE = "sample_end"

ARM_CROSS = "cross"
ARM_ELEVATED = "elevated"
REF_SPOT = "spot"
REF_FORWARD = "forward"

REASON_NO_EXP = "no_expiry"
REASON_NO_STRIKES = "no_strikes"
REASON_NO_LISTED = "no_listed_strike"
REASON_NO_FORWARD = "no_forward"
REASON_NO_QUOTE = "no_entry_quote"
REASON_NO_CREDIT = "credit_below_min"
REASON_NO_PATH = "no_exit_mark"
REASON_BUSY = "position_open"

MarkFn = Callable[[date, float, str, date, date], Dict[date, Dict[str, float]]]
StrikeFn = Callable[[date], List[float]]
ForwardFn = Callable[[date, date, float, int], Optional[float]]


@dataclass
class VixSpikeTrade:
    entry_date: date
    expiration: Optional[date] = None
    exit_date: Optional[date] = None
    spot: Optional[float] = None
    ref_px: Optional[float] = None
    ref_kind: str = ""
    forward_px: Optional[float] = None
    short_k: Optional[float] = None
    long_k: Optional[float] = None
    width: float = DEFAULT_WIDTH
    credit: Optional[float] = None
    mid_credit: Optional[float] = None
    exit_debit: Optional[float] = None
    cross_cost: Optional[float] = None
    max_risk: Optional[float] = None
    pnl: Optional[float] = None
    ror_pct: Optional[float] = None
    hold_days: Optional[int] = None
    dte_entry: Optional[int] = None
    dte_exit: Optional[int] = None
    exit_reason: Optional[str] = None
    mark_mode: str = "mid"
    arm_mode: str = ARM_CROSS
    trigger: float = DEFAULT_TRIGGER
    skipped: bool = False
    skip_reason: Optional[str] = None


@dataclass
class BookResult:
    label: str
    trades: List[VixSpikeTrade] = field(default_factory=list)
    skipped: List[Dict[str, Any]] = field(default_factory=list)


def third_friday(year: int, month: int) -> date:
    """Third Friday of a calendar month (standard SPX monthly expiry)."""
    first = date(int(year), int(month), 1)
    # Monday=0 ... Friday=4
    offset = (4 - first.weekday()) % 7
    return first + timedelta(days=offset + 14)


def expected_vix_monthly(year: int, month: int) -> date:
    """Wednesday 30 days before the next month's SPX third Friday.

    A Friday holiday moves the SPX expiry to Thursday, which moves this
    date to Tuesday. Callers also accept the prior session.
    """
    ny, nm = (year + 1, 1) if month == 12 else (year, month + 1)
    return third_friday(ny, nm) - timedelta(days=30)


def monthly_expiry_set(start_year: int, end_year: int) -> set:
    """Listed dates we treat as the standard VIX monthly (plus a 1-day holiday shift)."""
    out = set()
    for year in range(int(start_year), int(end_year) + 1):
        for month in range(1, 13):
            raw = expected_vix_monthly(year, month)
            out.add(raw)
            out.add(raw - timedelta(days=1))
    return out


def filter_monthly_expirations(expirations: Sequence[date]) -> List[date]:
    if not expirations:
        return []
    years = [d.year for d in expirations]
    allowed = monthly_expiry_set(min(years) - 1, max(years) + 1)
    return sorted(d for d in expirations if d in allowed)


def pick_monthly_expiry(
    expirations: Sequence[date],
    day: date,
    *,
    dte_min: int = DEFAULT_DTE_MIN,
    dte_max: int = DEFAULT_DTE_MAX,
) -> Optional[date]:
    """Nearest standard monthly with DTE inside [dte_min, dte_max]."""
    cands = []
    for exp in expirations:
        dte = (exp - day).days
        if dte_min <= dte <= dte_max:
            cands.append((dte, exp))
    if not cands:
        return None
    cands.sort()
    return cands[0][1]


def round_to_step(price: float, step: float = DEFAULT_STRIKE_STEP) -> float:
    """Half-up round onto a strike grid. Positive prices only."""
    step_f = float(step)
    if step_f <= 0:
        raise ValueError("step must be positive")
    units = math.floor((float(price) / step_f) + 0.5)
    return float(units * step_f)


def listed_strike(strikes: Sequence[float], target: float) -> Optional[float]:
    """Exact listed strike after rounding to 3 decimals (45 matches 45.000)."""
    want = round(float(target), 3)
    for raw in strikes:
        if round(float(raw), 3) == want:
            return float(raw)
    return None


def close_by_day(spots: pd.DataFrame) -> List[Tuple[date, float]]:
    """Unique session closes, last row wins if a timestamp repeats."""
    if spots is None or spots.empty or "close" not in spots.columns:
        return []
    found: Dict[date, float] = {}
    for ts, row in spots.iterrows():
        d = pd.Timestamp(ts).date()
        raw = row["close"]
        val = float(raw.iloc[0] if isinstance(raw, pd.Series) else raw)
        if math.isfinite(val) and val > 0:
            found[d] = val
    return sorted(found.items())


def is_entry_signal(
    prev_close: Optional[float],
    close: float,
    trigger: float,
    arm_mode: str,
) -> bool:
    if not math.isfinite(close) or close < float(trigger):
        return False
    if str(arm_mode).lower() == ARM_ELEVATED:
        return True
    if prev_close is None or not math.isfinite(float(prev_close)):
        return False
    return float(prev_close) < float(trigger)


def implied_vix_forward(
    strike: float,
    call_q: Dict[str, float],
    put_q: Dict[str, float],
    dte: int,
    rate: float = DEFAULT_RATE,
) -> Optional[float]:
    """European put-call parity: C - P = exp(-rT) * (F - K).

    Returns None when either side lacks a two-sided quote.
    """
    if not _two_sided(call_q) or not _two_sided(put_q):
        return None
    call_mid = _mid(call_q)
    put_mid = _mid(put_q)
    if call_mid is None or put_mid is None:
        return None
    t = max(int(dte), 1) / 365.0
    forward = float(strike) + (call_mid - put_mid) * math.exp(float(rate) * t)
    if not math.isfinite(forward) or forward <= 0:
        return None
    return float(forward)


def _mid(q: Dict[str, float]) -> Optional[float]:
    if not _two_sided(q):
        return None
    if q.get("mid") is not None and float(q["mid"]) > 0:
        return float(q["mid"])
    return 0.5 * (float(q["bid"]) + float(q["ask"]))


def _two_sided(q: Optional[Dict[str, float]]) -> bool:
    if not q:
        return False
    try:
        bid = float(q.get("bid") or 0)
        ask = float(q.get("ask") or 0)
    except (TypeError, ValueError):
        return False
    return bid > 0 and ask > 0 and ask + 1e-9 >= bid


def leg_is_usable(q: Optional[Dict[str, float]], max_spread: float = DEFAULT_MAX_LEG_SPREAD) -> bool:
    """Two-sided quote whose bid-ask is tight enough to mark a 10-point vertical.

    VIX EOD sometimes prints ask=10 against a sub-1 bid. That mid is not a fill.
    """
    if not _two_sided(q):
        return False
    return (float(q["ask"]) - float(q["bid"])) <= float(max_spread) + 1e-9


def walk_call_spread_exit(
    *,
    session_days: Sequence[date],
    debit_on,
    expiration: date,
    credit: float,
    tp_remain: float = DEFAULT_TP_REMAIN,
    expiry_week_days: int = DEFAULT_EXPIRY_WEEK_DAYS,
) -> Optional[Tuple[date, float, str]]:
    """First later session at the profit target, else expiry week, else sample end.

    ``session_days`` must already exclude the entry session. A missing mark is
    skipped. There is no debit stop.
    """
    tp_debit = float(credit) * float(tp_remain)
    days = [d for d in session_days if d < expiration]
    if not days:
        return None
    last_mark: Optional[Tuple[date, float]] = None
    for cur in days:
        debit = debit_on(cur)
        if debit is None or not math.isfinite(float(debit)):
            continue
        debit_f = float(debit)
        last_mark = (cur, debit_f)
        if debit_f <= tp_debit:
            return cur, debit_f, EXIT_TP
        if (expiration - cur).days <= int(expiry_week_days):
            return cur, debit_f, EXIT_EXPIRY
    if last_mark is None:
        return None
    return last_mark[0], last_mark[1], EXIT_SAMPLE


def load_vix_spot(start: date, end: date) -> pd.DataFrame:
    """Cash VIX daily bars. Theta index history is 403 on this plan; use Yahoo ^VIX."""
    from utils.research.spx_put_bwb import _yahoo_ohlc

    return _yahoo_ohlc("^VIX", start, end)


def eod_mark_map(
    client: ThetaDataClient,
    cache: Dict[Tuple[Any, ...], pd.DataFrame],
    expiration: date,
    strike: float,
    right: str,
    start: date,
    end: date,
) -> Dict[date, Dict[str, float]]:
    """Session EOD bid/ask for one VIX option. Retries Theta 429/478."""
    if end < start:
        return {}
    last_err: Optional[Exception] = None
    df = pd.DataFrame()
    for attempt in range(4):
        try:
            df = fetch_eod(
                client,
                symbol="VIX",
                expiration=expiration,
                strike=float(strike),
                right=right,
                start=start,
                end=end,
                cache=cache,
            )
            last_err = None
            break
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            msg = str(exc)
            if "429" in msg or "478" in msg or "Too Many" in msg or "timed out" in msg.lower():
                time.sleep(2.0 * (attempt + 1))
                continue
            LOG.warning("VIX EOD hard-fail %s %s %s: %s", expiration, strike, right, exc)
            return {}
    if last_err is not None:
        LOG.warning("VIX EOD failed %s %s %s: %s", expiration, strike, right, last_err)
        return {}
    out: Dict[date, Dict[str, float]] = {}
    if df is None or df.empty or "session_date" not in df.columns:
        return out
    for _, row in df.iterrows():
        d = row["session_date"]
        if not isinstance(d, date):
            d = pd.Timestamp(d).date()
        q = {"bid": float(row["bid"]), "ask": float(row["ask"]), "mid": float(row["mid"])}
        if leg_is_usable(q):
            out[d] = q
    return out


def parity_forward(
    client: ThetaDataClient,
    cache: Dict[Tuple[Any, ...], pd.DataFrame],
    strikes: Sequence[float],
    expiration: date,
    day: date,
    spot: float,
    dte: int,
) -> Optional[float]:
    """Implied future from the tightest two-sided strike near cash VIX."""
    if not strikes:
        return None
    nearest = sorted(strikes, key=lambda s: abs(float(s) - float(spot)))[:6]
    best: Optional[Tuple[float, float]] = None
    for strike in nearest:
        call_q = quote_on_day(
            client,
            symbol="VIX",
            expiration=expiration,
            strike=float(strike),
            right="call",
            day=day,
            cache=cache,
        )
        put_q = quote_on_day(
            client,
            symbol="VIX",
            expiration=expiration,
            strike=float(strike),
            right="put",
            day=day,
            cache=cache,
        )
        forward = implied_vix_forward(float(strike), call_q or {}, put_q or {}, dte)
        if forward is None or not (5.0 <= forward <= 120.0):
            continue
        width = 0.0
        if call_q and put_q:
            width = (float(call_q["ask"]) - float(call_q["bid"])) + (
                float(put_q["ask"]) - float(put_q["bid"])
            )
        if width > float(DEFAULT_MAX_PARITY_WIDTH):
            continue
        if best is None or width < best[0]:
            best = (width, forward)
    if best is None:
        return None
    return float(best[1])


def run_vix_spike_book(
    spots: pd.DataFrame,
    expirations: Sequence[date],
    *,
    start: date,
    end: date,
    mark_fn: MarkFn,
    strike_fn: StrikeFn,
    forward_fn: Optional[ForwardFn] = None,
    trigger: float = DEFAULT_TRIGGER,
    otm: float = DEFAULT_OTM,
    width: float = DEFAULT_WIDTH,
    strike_step: float = DEFAULT_STRIKE_STEP,
    dte_min: int = DEFAULT_DTE_MIN,
    dte_max: int = DEFAULT_DTE_MAX,
    min_credit: float = DEFAULT_MIN_CREDIT,
    tp_remain: float = DEFAULT_TP_REMAIN,
    expiry_week_days: int = DEFAULT_EXPIRY_WEEK_DAYS,
    mark_mode: str = "mid",
    package_slip: float = DEFAULT_PACKAGE_SLIP,
    arm_mode: str = ARM_CROSS,
    ref_mode: str = REF_SPOT,
    monthly_only: bool = True,
    max_entries: int = 0,
    commission_per_leg: float = DEFAULT_COMM_PER_LEG,
    multiplier: float = DEFAULT_MULT,
    label: str = "",
) -> BookResult:
    """One-position book. ``mark_fn(exp, strike, right, start, end)`` returns EOD quotes."""
    monthlies = filter_monthly_expirations(expirations) if monthly_only else list(expirations)
    series = close_by_day(spots)
    book = BookResult(label=label or f"{arm_mode}_{ref_mode}_{trigger:g}")
    active_until = date(1900, 1, 1)
    filled = 0
    slip = float(package_slip or 0.0)

    for i, (day, spot) in enumerate(series):
        prev = series[i - 1][1] if i else None
        if day < start or day > end:
            continue
        if not is_entry_signal(prev, spot, trigger, arm_mode):
            continue
        if day < active_until:
            book.skipped.append({"date": day.isoformat(), "reason": REASON_BUSY, "spot": spot})
            continue

        exp = pick_monthly_expiry(monthlies, day, dte_min=dte_min, dte_max=dte_max)
        if exp is None:
            book.skipped.append({"date": day.isoformat(), "reason": REASON_NO_EXP, "spot": spot})
            continue
        dte = (exp - day).days
        strikes = list(strike_fn(exp) or [])
        if not strikes:
            book.skipped.append(
                {"date": day.isoformat(), "reason": REASON_NO_STRIKES, "expiration": exp.isoformat()}
            )
            continue

        forward_px = None
        if forward_fn is not None and str(ref_mode).lower() == REF_FORWARD:
            forward_px = forward_fn(exp, day, float(spot), dte)
            if forward_px is None:
                book.skipped.append(
                    {
                        "date": day.isoformat(),
                        "reason": REASON_NO_FORWARD,
                        "expiration": exp.isoformat(),
                        "spot": spot,
                    }
                )
                continue
            ref_px = float(forward_px)
            ref_kind = REF_FORWARD
        else:
            ref_px = float(spot)
            ref_kind = REF_SPOT

        target = round_to_step(ref_px + float(otm), strike_step)
        short_k = listed_strike(strikes, target)
        long_k = listed_strike(strikes, target + float(width))
        if short_k is None or long_k is None:
            book.skipped.append(
                {
                    "date": day.isoformat(),
                    "reason": REASON_NO_LISTED,
                    "target": target,
                    "expiration": exp.isoformat(),
                    "ref": ref_px,
                }
            )
            continue

        last_day = min(end, exp - timedelta(days=1))
        short_marks = mark_fn(exp, short_k, "call", day, last_day)
        long_marks = mark_fn(exp, long_k, "call", day, last_day)
        short_q = short_marks.get(day)
        long_q = long_marks.get(day)
        if not leg_is_usable(short_q) or not leg_is_usable(long_q):
            book.skipped.append(
                {
                    "date": day.isoformat(),
                    "reason": REASON_NO_QUOTE,
                    "short_k": short_k,
                    "long_k": long_k,
                    "expiration": exp.isoformat(),
                }
            )
            continue

        credit = vertical_credit(short_q, long_q, mark_mode, slip)
        mid_credit = vertical_credit(short_q, long_q, "mid", 0.0)
        if credit is None or float(credit) <= float(min_credit):
            book.skipped.append(
                {
                    "date": day.isoformat(),
                    "reason": REASON_NO_CREDIT,
                    "credit": credit,
                    "short_k": short_k,
                    "long_k": long_k,
                }
            )
            continue

        later = [d for d, _ in series if day < d <= last_day]

        def _debit_on(cur: date, _s=short_marks, _l=long_marks) -> Optional[float]:
            qs = _s.get(cur)
            ql = _l.get(cur)
            if not leg_is_usable(qs) or not leg_is_usable(ql):
                return None
            raw = vertical_debit(qs, ql, "mid", 0.0)
            # A higher strike cannot be worth more than the short call. A negative
            # mid is a stale ask, not a take-profit.
            if raw is None or float(raw) < -0.05:
                return None
            return vertical_debit(qs, ql, mark_mode, slip)

        walked = walk_call_spread_exit(
            session_days=later,
            debit_on=_debit_on,
            expiration=exp,
            credit=float(credit),
            tp_remain=tp_remain,
            expiry_week_days=expiry_week_days,
        )
        if walked is None:
            book.skipped.append(
                {"date": day.isoformat(), "reason": REASON_NO_PATH, "expiration": exp.isoformat()}
            )
            continue

        exit_day, exit_debit, reason = walked
        pnl, ror, risk_pts = trade_pnl(
            float(credit),
            float(exit_debit),
            float(width),
            multiplier=multiplier,
            commission_per_leg=commission_per_leg,
        )
        trade = VixSpikeTrade(
            entry_date=day,
            expiration=exp,
            exit_date=exit_day,
            spot=float(spot),
            ref_px=float(ref_px),
            ref_kind=ref_kind,
            forward_px=None if forward_px is None else float(forward_px),
            short_k=float(short_k),
            long_k=float(long_k),
            width=float(width),
            credit=float(credit),
            mid_credit=None if mid_credit is None else float(mid_credit),
            exit_debit=float(exit_debit),
            cross_cost=half_spread_cross(short_q, long_q),
            max_risk=float(risk_pts) * float(multiplier),
            pnl=pnl,
            ror_pct=ror,
            hold_days=(exit_day - day).days,
            dte_entry=dte,
            dte_exit=(exp - exit_day).days,
            exit_reason=reason,
            mark_mode=str(mark_mode),
            arm_mode=str(arm_mode),
            trigger=float(trigger),
        )
        book.trades.append(trade)
        active_until = exit_day + timedelta(days=1)
        filled += 1
        LOG.info(
            "VIX %s %s spot=%.2f ref=%.2f %s/%s credit=%.2f exit=%s %s pnl=%.0f",
            day.isoformat(),
            arm_mode,
            spot,
            ref_px,
            short_k,
            long_k,
            credit,
            exit_day.isoformat(),
            reason,
            pnl,
        )
        if max_entries and filled >= int(max_entries):
            break
    return book


def _year_stats(trades: Sequence[VixSpikeTrade]) -> Dict[str, Any]:
    buckets: Dict[int, List[VixSpikeTrade]] = {}
    for trade in trades:
        buckets.setdefault(int(trade.entry_date.year), []).append(trade)
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


def _budget_lots(max_risk: float, budget: float) -> int:
    risk = float(max_risk)
    cap = float(budget)
    if risk <= 0 or cap <= 0:
        return 1
    if risk > cap:
        return 1
    return max(1, int(cap // risk))


def book_stats(
    trades: Sequence[VixSpikeTrade],
    *,
    start_equity: float = 100_000.0,
    risk_budget: float = DEFAULT_RISK_BUDGET,
) -> Dict[str, Any]:
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
    ordered = sorted(kept, key=lambda t: (t.exit_date, t.entry_date))
    span_days = max((ordered[-1].exit_date - ordered[0].entry_date).days, 1)
    per_year = len(kept) / (span_days / 365.25)
    sharpe = None
    if len(rors) >= 5 and float(rors.std()) > 1e-12:
        sharpe = float((rors.mean() / rors.std()) * math.sqrt(per_year))
    reasons: Dict[str, int] = {}
    for trade in kept:
        key = str(trade.exit_reason or "unknown")
        reasons[key] = reasons.get(key, 0) + 1
    eq = float(start_equity)
    peak = eq
    mdd = 0.0
    bud_eq = float(start_equity)
    bud_peak = bud_eq
    bud_mdd = 0.0
    bud_pnls = []
    for trade in ordered:
        eq += float(trade.pnl or 0.0)
        peak = max(peak, eq)
        if peak > 0:
            mdd = max(mdd, (peak - eq) / peak)
        lots = _budget_lots(float(trade.max_risk or 0.0), risk_budget)
        bud = float(trade.pnl or 0.0) * lots
        bud_pnls.append(bud)
        bud_eq += bud
        bud_peak = max(bud_peak, bud_eq)
        if bud_peak > 0:
            bud_mdd = max(bud_mdd, (bud_peak - bud_eq) / bud_peak)
    crosses = [float(t.cross_cost) for t in kept if t.cross_cost is not None]
    risks = [float(t.max_risk) for t in kept if t.max_risk is not None]
    return {
        "n": len(kept),
        "per_year": round(per_year, 2),
        "win_rate_pct": round(100.0 * float((pnls > 0).mean()), 1),
        "mean_ror_pct": round(float(rors.mean()), 2),
        "med_ror_pct": round(float(np.median(rors)), 2),
        "mean_pnl": round(float(pnls.mean()), 2),
        "avg_win": None if len(wins) == 0 else round(float(wins.mean()), 2),
        "avg_loss": None if len(losses) == 0 else round(float(losses.mean()), 2),
        "total_pnl": round(float(pnls.sum()), 2),
        "pf": None if pf is None else round(pf, 2),
        "avg_hold": round(float(np.mean([t.hold_days or 0 for t in kept])), 1),
        "avg_credit": round(float(np.mean([t.credit for t in kept if t.credit])), 3),
        "avg_spot": round(float(np.mean([t.spot for t in kept if t.spot])), 2),
        "med_cross": None if not crosses else round(float(np.median(crosses)), 3),
        "p90_cross": None if not crosses else round(float(np.quantile(crosses, 0.9)), 3),
        "med_max_risk": None if not risks else round(float(np.median(risks)), 2),
        "sharpe_proxy": None if sharpe is None else round(sharpe, 2),
        "exit_reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
        "years": _year_stats(kept),
        "end_equity_1lot": round(eq, 2),
        "mdd_pct_1lot": round(100.0 * mdd, 2),
        "risk_budget": float(risk_budget),
        "budget_pnl": round(float(np.sum(bud_pnls)), 2),
        "budget_mdd_pct": round(100.0 * bud_mdd, 2),
        "end_equity_budget": round(bud_eq, 2),
        "span_days": span_days,
    }


def trades_to_frame(trades: Sequence[VixSpikeTrade]) -> pd.DataFrame:
    rows = []
    for trade in trades:
        rows.append(
            {
                "entry_date": trade.entry_date.isoformat(),
                "exit_date": trade.exit_date.isoformat() if trade.exit_date else None,
                "expiration": trade.expiration.isoformat() if trade.expiration else None,
                "spot": trade.spot,
                "ref_px": trade.ref_px,
                "ref_kind": trade.ref_kind,
                "forward_px": trade.forward_px,
                "short_k": trade.short_k,
                "long_k": trade.long_k,
                "width": trade.width,
                "credit": trade.credit,
                "mid_credit": trade.mid_credit,
                "exit_debit": trade.exit_debit,
                "cross_cost": trade.cross_cost,
                "max_risk": trade.max_risk,
                "pnl": trade.pnl,
                "ror_pct": trade.ror_pct,
                "hold_days": trade.hold_days,
                "dte_entry": trade.dte_entry,
                "dte_exit": trade.dte_exit,
                "exit_reason": trade.exit_reason,
                "mark_mode": trade.mark_mode,
                "arm_mode": trade.arm_mode,
                "trigger": trade.trigger,
                "skipped": trade.skipped,
                "skip_reason": trade.skip_reason,
            }
        )
    return pd.DataFrame(rows)


def cached_strikes(
    client: ThetaDataClient,
    cache: Dict[date, List[float]],
    expiration: date,
) -> List[float]:
    if expiration in cache:
        return list(cache[expiration])
    try:
        found = list_strikes(client, "VIX", expiration)
    except Exception as exc:  # noqa: BLE001
        LOG.warning("VIX strikes failed %s: %s", expiration, exc)
        found = []
    cache[expiration] = found
    return list(found)
