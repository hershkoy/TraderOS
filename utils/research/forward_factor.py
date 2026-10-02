"""Forward Factor (Volatility Vibes) ATM call calendar claim-check.

Video: https://youtu.be/6ao3uXE5KhU (2025-10-10)
FF = (IV_front - IV_forward) / IV_forward
IV_forward from variance additivity on two expiries.

This is an honest SPY listed-mark pilot of the headline ATM call calendar
60/90 DTE sleeve — not a reprint of the claimed 19y / 300k multi-name book
(ex-earn IV, capacity, Kelly sizing).
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
from utils.data.ohlcv_loader import load_ohlcv_many
from utils.options.greeks import implied_volatility

LOG = logging.getLogger(__name__)

CLAIMED = {
    "source": "Volatility Vibes — Forward Factor (6ao3uXE5KhU)",
    "structure": "ATM call calendar / 35d double calendar",
    "best": "ATM call calendar 60/90 @ quarter Kelly ~26.7% CAGR / Sharpe 2.40",
    "playbook_ff": 0.20,
    "atm_60_90_threshold": 0.41,
    "exit": "close as spread just before front expiry",
    "note": "19y multi-name ~300k instances with frictions; blind calendars lose",
}

MULT = 100.0
DEFAULT_R = 0.05
DEFAULT_Q = 0.012  # SPY approx dividend yield
DEFAULT_COMM_PER_LEG = 0.65
DEFAULT_DTE_FRONT = 60
DEFAULT_DTE_BACK = 90
DEFAULT_DTE_TOL = 12
DEFAULT_FF_THRESH = 0.20


@dataclass
class FFTrade:
    symbol: str
    entry_date: date
    exit_date: date
    front: date
    back: date
    strike: float
    spot_entry: float
    front_iv: float
    back_iv: float
    fwd_iv: float
    ff: float
    mark_mode: str
    entry_debit: float
    exit_credit: float
    pnl: float
    ret_pct: float
    hold_days: int
    commission: float
    skipped: bool = False
    reason: str = ""


@dataclass
class BookResult:
    label: str
    mark_mode: str
    ff_min: float
    trades: List[FFTrade] = field(default_factory=list)
    skipped: List[Dict[str, Any]] = field(default_factory=list)


def forward_vol(iv1: float, iv2: float, t1: float, t2: float) -> Optional[float]:
    """Implied forward vol between T1 and T2 (years)."""
    if t2 <= t1 or iv1 <= 0 or iv2 <= 0:
        return None
    num = (iv2 * iv2) * t2 - (iv1 * iv1) * t1
    den = t2 - t1
    if den <= 0 or num <= 0:
        return None
    return float(math.sqrt(num / den))


def forward_factor(iv_front: float, iv_fwd: float) -> Optional[float]:
    if iv_fwd is None or iv_fwd <= 0 or iv_front is None:
        return None
    return float((iv_front - iv_fwd) / iv_fwd)


def list_expirations(client: ThetaDataClient, symbol: str) -> List[date]:
    r = client.session.get(
        f"{client.base_url}/v3/option/list/expirations",
        params={"symbol": str(symbol).upper()},
        timeout=60,
    )
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text))
    col = "expiration" if "expiration" in df.columns else df.columns[-1]
    out: List[date] = []
    for x in df[col].tolist():
        try:
            out.append(date.fromisoformat(str(x)[:10]))
        except ValueError:
            continue
    return sorted(set(out))


def monthly_expirations(exps: Sequence[date]) -> List[date]:
    """Prefer standard monthly / month-end listings (Theta weeklies often 472)."""
    out = []
    for ed in exps:
        # Classic monthly ~3rd Friday (day 15-21) or last trading day of month
        if 15 <= ed.day <= 21:
            out.append(ed)
        elif ed.month != (ed + timedelta(days=1)).month:
            out.append(ed)
    return sorted(set(out))


def list_strikes(client: ThetaDataClient, symbol: str, exp: date) -> List[float]:
    r = client.session.get(
        f"{client.base_url}/v3/option/list/strikes",
        params={"symbol": str(symbol).upper(), "expiration": exp.isoformat()},
        timeout=60,
    )
    if r.status_code != 200 or not r.content:
        return []
    df = pd.read_csv(StringIO(r.text))
    col = "strike" if "strike" in df.columns else df.columns[-1]
    return sorted({float(x) for x in df[col].tolist() if pd.notna(x)})


def nearest_strike(strikes: Sequence[float], spot: float) -> Optional[float]:
    if not strikes:
        return None
    return float(min(strikes, key=lambda s: abs(float(s) - float(spot))))


def is_monthly_expiry(ed: date) -> bool:
    """Standard monthly (~3rd Fri day 15-21) or month-end listing."""
    if 15 <= ed.day <= 21 and ed.weekday() == 4:
        return True
    # month-end (next day is a new month)
    return ed.month != (ed + timedelta(days=1)).month


def pick_expiry(
    exps: Sequence[date],
    day: date,
    target_dte: int,
    tol: int,
    *,
    monthly_only: bool = True,
) -> Optional[date]:
    pool = [e for e in exps if (not monthly_only) or is_monthly_expiry(e)]
    if not pool:
        pool = list(exps)
    cands = []
    for ed in pool:
        dte = (ed - day).days
        if abs(dte - target_dte) <= tol and dte > 5:
            cands.append((abs(dte - target_dte), ed))
    if not cands:
        return None
    cands.sort(key=lambda x: x[0])
    return cands[0][1]


def quote_on_day(
    client: ThetaDataClient,
    *,
    symbol: str,
    expiration: date,
    strike: float,
    right: str,
    day: date,
    cache: Dict[Tuple[Any, ...], pd.DataFrame],
) -> Optional[Dict[str, float]]:
    """Prefer Theta EOD; fall back to 15:45 ET 1m NBBO."""
    import time as _time

    last_err: Optional[Exception] = None
    for attempt in range(4):
        try:
            eod = fetch_eod(
                client,
                symbol=symbol,
                expiration=expiration,
                strike=float(strike),
                right=right,
                start=day,
                end=day,
                cache=cache,
            )
            row = eod_on(eod, day)
            if row is not None:
                return {
                    "bid": float(row["bid"]),
                    "ask": float(row["ask"]),
                    "mid": float(row["mid"]),
                    "source": "eod",
                }
            from datetime import time as dtime

            nbbo = client.nbbo_at_or_after(
                symbol=symbol,
                expiration=expiration,
                strike=float(strike),
                right=right,
                day=day,
                clock=dtime(15, 45),
            )
            if nbbo is None:
                return None
            return {
                "bid": float(nbbo["bid"]),
                "ask": float(nbbo["ask"]),
                "mid": float(nbbo["mid"]),
                "source": "1545",
            }
        except Exception as exc:
            last_err = exc
            msg = str(exc)
            if "429" in msg or "Too Many" in msg or "timed out" in msg.lower():
                _time.sleep(2.0 * (attempt + 1))
                continue
            raise
    if last_err:
        LOG.warning("quote_on_day failed %s %s %s: %s", symbol, expiration, day, last_err)
    return None


def fetch_eod(
    client: ThetaDataClient,
    *,
    symbol: str,
    expiration: date,
    strike: float,
    right: str,
    start: date,
    end: date,
    cache: Dict[Tuple[Any, ...], pd.DataFrame],
) -> pd.DataFrame:
    if end < start:
        return pd.DataFrame()
    if (end - start).days > 364:
        end = start + timedelta(days=364)
    key = (
        str(symbol).upper(),
        expiration.isoformat(),
        float(strike),
        right[:1].upper(),
        start.isoformat(),
        end.isoformat(),
    )
    if key in cache:
        return cache[key].copy()

    right_s = "call" if str(right).lower().startswith("c") else "put"
    r = client.session.get(
        f"{client.base_url}/v3/option/history/eod",
        params={
            "symbol": str(symbol).upper(),
            "expiration": expiration.isoformat(),
            "strike": f"{float(strike):.3f}",
            "right": right_s,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
        },
        timeout=120,
    )
    if r.status_code == 429:
        raise RuntimeError(f"429 Too Many Requests for EOD {symbol} {expiration} {start}")
    if r.status_code == 472 or (r.content and b"No data" in r.content[:80]):
        cache[key] = pd.DataFrame()
        return pd.DataFrame()
    if r.status_code != 200 or not r.content:
        cache[key] = pd.DataFrame()
        return pd.DataFrame()

    df = pd.read_csv(StringIO(r.text))
    if df.empty:
        cache[key] = df
        return df.copy()
    if "created" in df.columns:
        ts = pd.to_datetime(df["created"], errors="coerce")
    elif "last_trade" in df.columns:
        ts = pd.to_datetime(df["last_trade"], errors="coerce")
    else:
        ts = pd.to_datetime(df.iloc[:, min(4, len(df.columns) - 1)], errors="coerce")
    df = df.copy()
    df["session_date"] = ts.dt.date
    df["bid"] = pd.to_numeric(df.get("bid"), errors="coerce")
    df["ask"] = pd.to_numeric(df.get("ask"), errors="coerce")
    df["mid"] = (df["bid"] + df["ask"]) / 2.0
    df = df.dropna(subset=["session_date", "bid", "ask"])
    df = df[(df["bid"] > 0) & (df["ask"] > 0)]
    df = df.drop_duplicates("session_date", keep="last").sort_values("session_date")
    cache[key] = df
    return df.copy()


def eod_on(df: pd.DataFrame, day: date) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    rows = df[df["session_date"] == day]
    if rows.empty:
        return None
    return rows.iloc[-1]


def estimate_iv(
    *,
    mid: float,
    spot: float,
    strike: float,
    dte_days: float,
    right: str = "call",
    r: float = DEFAULT_R,
    q: float = DEFAULT_Q,
) -> Optional[float]:
    t = max(float(dte_days), 0.25) / 365.0
    if mid <= 0 or spot <= 0 or t <= 0:
        return None
    try:
        iv = implied_volatility(
            float(mid),
            float(spot),
            float(strike),
            t,
            float(r),
            "call" if right.lower().startswith("c") else "put",
            q=float(q),
        )
    except Exception:
        return None
    if iv is None or not math.isfinite(iv) or iv <= 0.01 or iv > 2.5:
        return None
    return float(iv)


def load_daily(symbol: str, start: date, end: date) -> pd.DataFrame:
    warm = start - timedelta(days=40)
    frames = load_ohlcv_many(
        [str(symbol).upper()],
        timeframe="1d",
        provider="ALPACA",
        start=datetime(warm.year, warm.month, warm.day),
        end=datetime(end.year, end.month, end.day),
        use_cache=True,
        workers=1,
        fallback_provider="IB",
        merge_mode="prefix",
    )
    raw = frames.get(str(symbol).upper())
    if raw is None or raw.empty:
        raise RuntimeError(f"No daily bars for {symbol}")
    df = raw.copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df.sort_index()


def load_spy_daily(start: date, end: date) -> pd.DataFrame:
    return load_daily("SPY", start, end)


def mondays_on_index(idx: pd.DatetimeIndex, start: date, end: date) -> List[date]:
    out = []
    for ts in idx:
        d = ts.date()
        if d < start or d > end:
            continue
        if d.weekday() == 0:
            out.append(d)
    return out


def _pnl_calendar(
    *,
    front_bid_e: float,
    front_ask_e: float,
    front_mid_e: float,
    back_bid_e: float,
    back_ask_e: float,
    back_mid_e: float,
    front_bid_x: float,
    front_ask_x: float,
    front_mid_x: float,
    back_bid_x: float,
    back_ask_x: float,
    back_mid_x: float,
    mark_mode: str,
    commission: float,
) -> Tuple[float, float, float]:
    """Return (entry_debit, exit_credit, pnl_cash) for 1 long calendar."""
    if mark_mode == "mid":
        entry_debit = back_mid_e - front_mid_e
        exit_credit = back_mid_x - front_mid_x
        # PnL: (front_e - front_x) + (back_x - back_e) on mid = exit_credit - entry_debit
        pnl = (exit_credit - entry_debit) * MULT - commission
    else:
        # ask_bid: sell front@bid, buy back@ask; cover front@ask, sell back@bid
        entry_debit = back_ask_e - front_bid_e
        exit_credit = back_bid_x - front_ask_x
        pnl = (exit_credit - entry_debit) * MULT - commission
    return float(entry_debit), float(exit_credit), float(pnl)


def book_stats(trades: Sequence[FFTrade]) -> Dict[str, Any]:
    kept = [t for t in trades if not t.skipped and t.pnl is not None]
    if not kept:
        return {"n": 0}
    rets = np.array([t.ret_pct for t in kept], dtype=float)
    pnls = np.array([t.pnl for t in kept], dtype=float)
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    gross_win = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(-losses.sum()) if len(losses) else 0.0
    pf = (gross_win / gross_loss) if gross_loss > 1e-9 else None
    # simple equal-$1 equity for Sharpe proxy on trade returns
    eq = np.cumsum(rets / 100.0)
    sharpe = None
    if len(rets) >= 5 and rets.std() > 1e-12:
        # annualize by avg hold ~60d -> ~6 trades/yr if sequential; use trade-Sharpe * sqrt(n_per_year)
        hold = float(np.mean([t.hold_days for t in kept])) or 60.0
        n_per_year = 365.0 / hold
        sharpe = float((rets.mean() / rets.std()) * math.sqrt(n_per_year))
    return {
        "n": len(kept),
        "win_rate_pct": round(100.0 * float((pnls > 0).mean()), 1),
        "mean_ret_pct": round(float(rets.mean()), 2),
        "med_ret_pct": round(float(np.median(rets)), 2),
        "pf": None if pf is None else round(pf, 2),
        "total_pnl": round(float(pnls.sum()), 2),
        "avg_pnl": round(float(pnls.mean()), 2),
        "avg_hold": round(float(np.mean([t.hold_days for t in kept])), 1),
        "avg_ff": round(float(np.mean([t.ff for t in kept])), 3),
        "avg_debit": round(float(np.mean([t.entry_debit for t in kept])), 3),
        "sharpe_proxy": None if sharpe is None else round(sharpe, 2),
        "eq_end": round(float(1.0 + eq[-1]), 3) if len(eq) else None,
    }


def run_ff_book(
    *,
    start: date,
    end: date,
    symbol: str = "SPY",
    mark_mode: str = "mid",
    ff_min: float = DEFAULT_FF_THRESH,
    dte_front: int = DEFAULT_DTE_FRONT,
    dte_back: int = DEFAULT_DTE_BACK,
    dte_tol: int = 20,
    commission_per_leg: float = DEFAULT_COMM_PER_LEG,
    client: Optional[ThetaDataClient] = None,
    spots: Optional[pd.DataFrame] = None,
    exps: Optional[List[date]] = None,
    max_concurrent: int = 1,
    scan: str = "monday",
    monthly_only: bool = True,
) -> BookResult:
    """Walk Mondays; enter ATM call calendar when FF >= ff_min."""
    client = client or get_thetadata_client()
    if not client.is_ready():
        raise RuntimeError("Theta Terminal not ready on :25503")

    spots = spots if spots is not None else load_daily(symbol, start, end)
    exps = exps if exps is not None else list_expirations(client, symbol)
    label = f"{symbol}_atm_cal_{dte_front}_{dte_back}_ff{ff_min:g}_{mark_mode}"
    book = BookResult(label=label, mark_mode=mark_mode, ff_min=ff_min)
    cache: Dict[Tuple[Any, ...], pd.DataFrame] = {}
    strike_cache: Dict[date, List[float]] = {}

    if scan == "monday":
        days = mondays_on_index(spots.index, start, end)
    elif scan == "month_start":
        # First Monday of each calendar month (faster panel scan)
        days = []
        seen = set()
        for d in mondays_on_index(spots.index, start, end):
            key = (d.year, d.month)
            if key in seen:
                continue
            seen.add(key)
            days.append(d)
    else:
        days = [ts.date() for ts in spots.index if start <= ts.date() <= end]

    open_until: List[date] = []  # exit dates of open positions
    commission = float(commission_per_leg) * 4.0  # 2 legs open + 2 close

    close_by_day: Dict[date, float] = {}
    for ts, row in spots.iterrows():
        close_by_day[ts.date() if hasattr(ts, "date") else pd.Timestamp(ts).date()] = float(
            row["close"]
        )

    for day in days:
        # drop finished
        open_until = [x for x in open_until if x > day]
        if len(open_until) >= max_concurrent:
            continue

        spot = close_by_day.get(day)
        if spot is None or spot <= 0:
            continue

        front = pick_expiry(
            exps, day, dte_front, dte_tol, monthly_only=monthly_only
        )
        back = pick_expiry(
            exps, day, dte_back, dte_tol, monthly_only=monthly_only
        )
        if front is None or back is None or back <= front:
            book.skipped.append({"date": str(day), "reason": "no_expiry_pair"})
            continue

        if front not in strike_cache:
            strike_cache[front] = list_strikes(client, symbol, front)
        if back not in strike_cache:
            strike_cache[back] = list_strikes(client, symbol, back)
        common = sorted(set(strike_cache[front]) & set(strike_cache[back]))
        strike = nearest_strike(common if common else strike_cache[front], spot)
        if strike is None:
            book.skipped.append({"date": str(day), "reason": "no_strike"})
            continue
        if strike not in strike_cache[back]:
            if not common:
                book.skipped.append({"date": str(day), "reason": "no_common_strike"})
                continue
            strike = nearest_strike(common, spot)

        exit_day = front - timedelta(days=1)
        while exit_day.weekday() >= 5:
            exit_day -= timedelta(days=1)
        if exit_day <= day:
            book.skipped.append({"date": str(day), "reason": "exit_before_entry"})
            continue

        fe = quote_on_day(
            client,
            symbol=symbol,
            expiration=front,
            strike=float(strike),
            right="call",
            day=day,
            cache=cache,
        )
        be = quote_on_day(
            client,
            symbol=symbol,
            expiration=back,
            strike=float(strike),
            right="call",
            day=day,
            cache=cache,
        )
        if fe is None or be is None:
            book.skipped.append({"date": str(day), "reason": "no_entry_quote"})
            continue

        dte1 = float((front - day).days)
        dte2 = float((back - day).days)
        iv1 = estimate_iv(
            mid=float(fe["mid"]), spot=spot, strike=float(strike), dte_days=dte1
        )
        iv2 = estimate_iv(
            mid=float(be["mid"]), spot=spot, strike=float(strike), dte_days=dte2
        )
        if iv1 is None or iv2 is None:
            book.skipped.append({"date": str(day), "reason": "iv_fail"})
            continue
        t1 = dte1 / 365.0
        t2 = dte2 / 365.0
        ivf = forward_vol(iv1, iv2, t1, t2)
        ff = forward_factor(iv1, ivf) if ivf else None
        if ff is None:
            book.skipped.append(
                {"date": str(day), "reason": "ff_undefined", "iv1": iv1, "iv2": iv2}
            )
            continue
        if ff < ff_min:
            book.skipped.append({"date": str(day), "reason": "ff_below", "ff": round(ff, 4)})
            continue

        fx = quote_on_day(
            client,
            symbol=symbol,
            expiration=front,
            strike=float(strike),
            right="call",
            day=exit_day,
            cache=cache,
        )
        bx = quote_on_day(
            client,
            symbol=symbol,
            expiration=back,
            strike=float(strike),
            right="call",
            day=exit_day,
            cache=cache,
        )
        if fx is None or bx is None:
            # walk back a few sessions for exit quotes
            found = False
            for back_days in range(1, 6):
                alt = exit_day - timedelta(days=back_days)
                if alt.weekday() >= 5 or alt <= day:
                    continue
                fx = quote_on_day(
                    client,
                    symbol=symbol,
                    expiration=front,
                    strike=float(strike),
                    right="call",
                    day=alt,
                    cache=cache,
                )
                bx = quote_on_day(
                    client,
                    symbol=symbol,
                    expiration=back,
                    strike=float(strike),
                    right="call",
                    day=alt,
                    cache=cache,
                )
                if fx is not None and bx is not None:
                    exit_day = alt
                    found = True
                    break
            if not found:
                book.skipped.append({"date": str(day), "reason": "no_exit_quote"})
                continue

        entry_debit, exit_credit, pnl = _pnl_calendar(
            front_bid_e=float(fe["bid"]),
            front_ask_e=float(fe["ask"]),
            front_mid_e=float(fe["mid"]),
            back_bid_e=float(be["bid"]),
            back_ask_e=float(be["ask"]),
            back_mid_e=float(be["mid"]),
            front_bid_x=float(fx["bid"]),
            front_ask_x=float(fx["ask"]),
            front_mid_x=float(fx["mid"]),
            back_bid_x=float(bx["bid"]),
            back_ask_x=float(bx["ask"]),
            back_mid_x=float(bx["mid"]),
            mark_mode=mark_mode,
            commission=commission,
        )
        if entry_debit <= 0.05:
            book.skipped.append(
                {
                    "date": str(day),
                    "reason": "entry_not_debit",
                    "debit": round(entry_debit, 3),
                }
            )
            continue

        ret_pct = 100.0 * pnl / (entry_debit * MULT)
        trade = FFTrade(
            symbol=symbol,
            entry_date=day,
            exit_date=exit_day,
            front=front,
            back=back,
            strike=float(strike),
            spot_entry=spot,
            front_iv=iv1,
            back_iv=iv2,
            fwd_iv=float(ivf),
            ff=float(ff),
            mark_mode=mark_mode,
            entry_debit=entry_debit,
            exit_credit=exit_credit,
            pnl=pnl,
            ret_pct=ret_pct,
            hold_days=(exit_day - day).days,
            commission=commission,
        )
        book.trades.append(trade)
        open_until.append(trade.exit_date)
        LOG.info(
            "%s %s %s FF=%.2f debit=%.2f pnl=$%.0f ret=%.1f%%",
            symbol,
            day,
            mark_mode,
            ff,
            entry_debit,
            pnl,
            ret_pct,
        )

    return book


def run_ff_panel(
    *,
    symbols: Sequence[str],
    start: date,
    end: date,
    mark_mode: str = "mid",
    ff_min: float = DEFAULT_FF_THRESH,
    dte_front: int = DEFAULT_DTE_FRONT,
    dte_back: int = DEFAULT_DTE_BACK,
    dte_tol: int = 20,
    client: Optional[ThetaDataClient] = None,
    monthly_only: bool = True,
    scan: str = "monday",
) -> BookResult:
    """Independent per-symbol sequential books, concatenated (panel pool)."""
    client = client or get_thetadata_client()
    label = f"panel_atm_cal_{dte_front}_{dte_back}_ff{ff_min:g}_{mark_mode}"
    out = BookResult(label=label, mark_mode=mark_mode, ff_min=ff_min)
    for sym in symbols:
        try:
            book = run_ff_book(
                start=start,
                end=end,
                symbol=sym,
                mark_mode=mark_mode,
                ff_min=ff_min,
                dte_front=dte_front,
                dte_back=dte_back,
                dte_tol=dte_tol,
                client=client,
                monthly_only=monthly_only,
                scan=scan,
            )
        except Exception as exc:
            LOG.warning("%s failed: %s", sym, exc)
            out.skipped.append({"symbol": sym, "reason": f"symbol_fail:{exc}"})
            continue
        out.trades.extend(book.trades)
        for s in book.skipped:
            s2 = dict(s)
            s2["symbol"] = sym
            out.skipped.append(s2)
    out.trades.sort(key=lambda t: (t.entry_date, t.symbol))
    return out


def trades_to_frame(trades: Sequence[FFTrade]) -> pd.DataFrame:
    rows = []
    for t in trades:
        rows.append(
            {
                "symbol": t.symbol,
                "entry_date": t.entry_date.isoformat(),
                "exit_date": t.exit_date.isoformat(),
                "front": t.front.isoformat(),
                "back": t.back.isoformat(),
                "strike": t.strike,
                "spot_entry": t.spot_entry,
                "front_iv": t.front_iv,
                "back_iv": t.back_iv,
                "fwd_iv": t.fwd_iv,
                "ff": t.ff,
                "mark_mode": t.mark_mode,
                "entry_debit": t.entry_debit,
                "exit_credit": t.exit_credit,
                "pnl": t.pnl,
                "ret_pct": t.ret_pct,
                "hold_days": t.hold_days,
                "commission": t.commission,
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "CLAIMED",
    "BookResult",
    "FFTrade",
    "book_stats",
    "forward_factor",
    "forward_vol",
    "list_expirations",
    "load_daily",
    "load_spy_daily",
    "run_ff_book",
    "run_ff_panel",
    "trades_to_frame",
]
