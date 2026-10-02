"""Forward Factor (Volatility Vibes) calendar claim-check — v2 harness.

Video: https://youtu.be/6ao3uXE5KhU (2025-10-10)
FF = (IV_front - IV_forward) / IV_forward via variance additivity.

v2 vs v1 (2026-10-01):
  * Friday expiries (not monthly-only) — shorter 30/60 (or 20/40) pairs
  * Skip single-name entries when earnings fall in [entry, front]
  * Dynamic exits: TP / SL / FF normalize / front DTE time-stop
  * mid_slip combo friction (mid +/- fraction of spread width)
  * Optional +/-35d double calendar
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from utils.api.thetadata.thetadata_client import ThetaDataClient, get_thetadata_client
from utils.data.ohlcv_loader import load_ohlcv_many
from utils.options.greeks import calculate_greeks, implied_volatility

LOG = logging.getLogger(__name__)

CLAIMED = {
    "source": "Volatility Vibes — Forward Factor (6ao3uXE5KhU)",
    "structure": "ATM call calendar / 35d double calendar",
    "best": "ATM call calendar 60/90 @ quarter Kelly ~26.7% CAGR / Sharpe 2.40",
    "playbook_ff": 0.20,
    "exit": "close as spread before front expiry (video); v2 uses dynamic exits",
}

MULT = 100.0
DEFAULT_R = 0.05
DEFAULT_Q = 0.012
DEFAULT_COMM_PER_LEG = 0.65
DEFAULT_DTE_FRONT = 30
DEFAULT_DTE_BACK = 60
DEFAULT_DTE_TOL = 10
DEFAULT_FF_THRESH = 0.20
DEFAULT_FF_EXIT = 0.05
DEFAULT_TP = 0.25
DEFAULT_SL = 0.40
DEFAULT_TIME_STOP_DTE = 15
DEFAULT_SLIP_FRAC = 0.10
DEFAULT_TARGET_DELTA = 0.35

NO_EARNINGS_ROOTS = {
    "SPY",
    "QQQ",
    "IWM",
    "DIA",
    "XLK",
    "XLF",
    "XLE",
    "XLI",
    "XLV",
    "XLY",
    "XLP",
    "XLU",
    "XLB",
    "XLRE",
    "VIX",
}


@dataclass
class FFTrade:
    symbol: str
    entry_date: date
    exit_date: date
    front: date
    back: date
    strike: float  # call strike (ATM or 35d call); put strike if double
    put_strike: Optional[float]
    spot_entry: float
    front_iv: float
    back_iv: float
    fwd_iv: float
    ff: float
    mark_mode: str
    structure: str
    entry_debit: float
    exit_credit: float
    pnl: float
    ret_pct: float
    hold_days: int
    commission: float
    exit_reason: str = ""
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


def list_strikes(client: ThetaDataClient, symbol: str, exp: date) -> List[float]:
    r = client.session.get(
        f"{client.base_url}/v3/option/list/strikes",
        params={"symbol": str(symbol).upper(), "expiration": exp.isoformat()},
        timeout=60,
    )
    if r.status_code == 429:
        raise RuntimeError(f"429 list_strikes {symbol} {exp}")
    if r.status_code != 200 or not r.content:
        return []
    df = pd.read_csv(StringIO(r.text))
    col = "strike" if "strike" in df.columns else df.columns[-1]
    return sorted({float(x) for x in df[col].tolist() if pd.notna(x)})


def nearest_strike(strikes: Sequence[float], spot: float) -> Optional[float]:
    if not strikes:
        return None
    return float(min(strikes, key=lambda s: abs(float(s) - float(spot))))


def is_friday_expiry(ed: date) -> bool:
    return ed.weekday() == 4


def is_monthly_expiry(ed: date) -> bool:
    if 15 <= ed.day <= 21 and ed.weekday() == 4:
        return True
    return ed.month != (ed + timedelta(days=1)).month


def pick_expiry(
    exps: Sequence[date],
    day: date,
    target_dte: int,
    tol: int,
    *,
    friday_only: bool = True,
    monthly_only: bool = False,
) -> Optional[date]:
    pool = list(exps)
    if monthly_only:
        pool = [e for e in pool if is_monthly_expiry(e)]
    elif friday_only:
        pool = [e for e in pool if is_friday_expiry(e)]
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


def strike_for_abs_delta(
    *,
    spot: float,
    strikes: Sequence[float],
    dte_days: float,
    iv: float,
    right: str,
    target_abs_delta: float = DEFAULT_TARGET_DELTA,
    r: float = DEFAULT_R,
    q: float = DEFAULT_Q,
) -> Optional[float]:
    if not strikes or spot <= 0 or iv <= 0:
        return None
    T = max(float(dte_days), 0.25) / 365.0
    right_c = "C" if str(right).upper().startswith("C") else "P"
    best = None
    best_err = 1e9
    # Narrow candidates around ATM
    ranked = sorted(strikes, key=lambda k: abs(float(k) - float(spot)))[:40]
    for k in ranked:
        g = calculate_greeks(float(spot), float(k), T, float(r), float(iv), right_c, q=float(q))
        err = abs(abs(float(g["delta"])) - float(target_abs_delta))
        if err < best_err:
            best_err = err
            best = float(k)
    return best


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


def eod_on(df: pd.DataFrame, day: date) -> Optional[Dict[str, float]]:
    if df is None or df.empty:
        return None
    rows = df[df["session_date"] == day]
    if rows.empty:
        return None
    row = rows.iloc[-1]
    return {
        "bid": float(row["bid"]),
        "ask": float(row["ask"]),
        "mid": float(row["mid"]),
        "spread": float(row["ask"]) - float(row["bid"]),
    }


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


def session_days_between(
    close_by_day: Dict[date, float], start: date, end: date
) -> List[date]:
    return sorted(d for d in close_by_day if start < d <= end)


def fetch_earnings_dates(symbol: str, *, limit: int = 80) -> List[date]:
    import yfinance as yf

    t = yf.Ticker(str(symbol).upper())
    ed = t.get_earnings_dates(limit=int(limit))
    if ed is None or ed.empty:
        return []
    out: List[date] = []
    for ts, row in ed.iterrows():
        reported = row.get("Reported EPS")
        if reported is None or (isinstance(reported, float) and math.isnan(reported)):
            continue
        ts_ny = pd.Timestamp(ts)
        if ts_ny.tzinfo is None:
            ts_ny = ts_ny.tz_localize("America/New_York")
        else:
            ts_ny = ts_ny.tz_convert("America/New_York")
        out.append(ts_ny.date())
    return sorted(set(out))


def load_earnings_map(
    symbols: Sequence[str],
    cache_path: Optional[Path] = None,
) -> Dict[str, List[date]]:
    cache: Dict[str, Any] = {}
    if cache_path is not None and cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception as exc:
            LOG.warning("earnings cache read failed: %s", exc)
    out: Dict[str, List[date]] = {}
    dirty = False
    for sym in symbols:
        s = str(sym).upper()
        if s in NO_EARNINGS_ROOTS:
            out[s] = []
            continue
        if s in cache:
            out[s] = [date.fromisoformat(x) for x in cache[s]]
            continue
        try:
            dates = fetch_earnings_dates(s)
        except Exception as exc:
            LOG.warning("earnings fetch failed %s: %s", s, exc)
            dates = []
        out[s] = dates
        cache[s] = [d.isoformat() for d in dates]
        dirty = True
        LOG.info("earnings %s n=%d", s, len(dates))
    if dirty and cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    return out


def has_earnings_in_window(
    symbol: str,
    start_date: date,
    end_date: date,
    earnings_map: Optional[Dict[str, List[date]]] = None,
) -> bool:
    s = str(symbol).upper()
    if s in NO_EARNINGS_ROOTS:
        return False
    dates = (earnings_map or {}).get(s) or []
    for d in dates:
        if start_date <= d <= end_date:
            return True
    return False


def _calendar_debit_credit(
    front: Dict[str, float],
    back: Dict[str, float],
    *,
    mark_mode: str,
    slip_frac: float,
    side: str,
) -> float:
    """Package mark for one calendar (long back / short front).

    side='entry' -> debit paid; side='exit' -> credit received.
    """
    fmid, bmid = front["mid"], back["mid"]
    fs, bs = max(front["spread"], 0.0), max(back["spread"], 0.0)
    if mark_mode == "mid":
        raw = bmid - fmid
        return float(raw)
    if mark_mode == "mid_slip":
        raw = bmid - fmid
        haircut = float(slip_frac) * (fs + bs)
        return float(raw + haircut) if side == "entry" else float(raw - haircut)
    # ask_bid natural
    if side == "entry":
        return float(back["ask"] - front["bid"])
    return float(back["bid"] - front["ask"])


def package_debit(
    legs: Dict[str, Dict[str, float]],
    *,
    mark_mode: str,
    slip_frac: float,
    structure: str,
) -> float:
    call = _calendar_debit_credit(
        legs["front_call"], legs["back_call"], mark_mode=mark_mode, slip_frac=slip_frac, side="entry"
    )
    if structure == "atm_call":
        return call
    put = _calendar_debit_credit(
        legs["front_put"], legs["back_put"], mark_mode=mark_mode, slip_frac=slip_frac, side="entry"
    )
    return call + put


def package_credit(
    legs: Dict[str, Dict[str, float]],
    *,
    mark_mode: str,
    slip_frac: float,
    structure: str,
) -> float:
    call = _calendar_debit_credit(
        legs["front_call"], legs["back_call"], mark_mode=mark_mode, slip_frac=slip_frac, side="exit"
    )
    if structure == "atm_call":
        return call
    put = _calendar_debit_credit(
        legs["front_put"], legs["back_put"], mark_mode=mark_mode, slip_frac=slip_frac, side="exit"
    )
    return call + put


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
    eq = np.cumsum(rets / 100.0)
    sharpe = None
    if len(rets) >= 5 and rets.std() > 1e-12:
        hold = float(np.mean([t.hold_days for t in kept])) or 30.0
        n_per_year = 365.0 / max(hold, 1.0)
        sharpe = float((rets.mean() / rets.std()) * math.sqrt(n_per_year))
    reasons: Dict[str, int] = {}
    for t in kept:
        reasons[t.exit_reason or "unknown"] = reasons.get(t.exit_reason or "unknown", 0) + 1
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
        "exit_reasons": reasons,
    }


def _legs_on_day(
    frames: Dict[str, pd.DataFrame],
    day: date,
    structure: str,
) -> Optional[Dict[str, Dict[str, float]]]:
    need = ["front_call", "back_call"]
    if structure == "double_cal":
        need += ["front_put", "back_put"]
    out: Dict[str, Dict[str, float]] = {}
    for k in need:
        q = eod_on(frames[k], day)
        if q is None:
            return None
        out[k] = q
    return out


def run_ff_book(
    *,
    start: date,
    end: date,
    symbol: str = "SPY",
    mark_mode: str = "mid_slip",
    ff_min: float = DEFAULT_FF_THRESH,
    dte_front: int = DEFAULT_DTE_FRONT,
    dte_back: int = DEFAULT_DTE_BACK,
    dte_tol: int = DEFAULT_DTE_TOL,
    commission_per_leg: float = DEFAULT_COMM_PER_LEG,
    client: Optional[ThetaDataClient] = None,
    spots: Optional[pd.DataFrame] = None,
    exps: Optional[List[date]] = None,
    max_concurrent: int = 1,
    scan: str = "monday",
    friday_only: bool = True,
    monthly_only: bool = False,
    structure: str = "atm_call",
    slip_frac: float = DEFAULT_SLIP_FRAC,
    tp_pct: float = DEFAULT_TP,
    sl_pct: float = DEFAULT_SL,
    ff_exit: float = DEFAULT_FF_EXIT,
    time_stop_dte: int = DEFAULT_TIME_STOP_DTE,
    skip_earnings: bool = True,
    earnings_map: Optional[Dict[str, List[date]]] = None,
    target_delta: float = DEFAULT_TARGET_DELTA,
    dynamic_exit: bool = True,
) -> BookResult:
    """Walk scan days; enter calendar when FF >= ff_min; manage with dynamic exits."""
    client = client or get_thetadata_client()
    if not client.is_ready():
        raise RuntimeError("Theta Terminal not ready on :25503")

    structure = str(structure).lower().strip()
    if structure not in ("atm_call", "double_cal"):
        raise ValueError(f"unknown structure {structure}")

    spots = spots if spots is not None else load_daily(symbol, start, end)
    exps = exps if exps is not None else list_expirations(client, symbol)
    label = (
        f"{symbol}_{structure}_{dte_front}_{dte_back}_ff{ff_min:g}_{mark_mode}"
    )
    book = BookResult(label=label, mark_mode=mark_mode, ff_min=ff_min)
    cache: Dict[Tuple[Any, ...], pd.DataFrame] = {}
    strike_cache: Dict[date, List[float]] = {}

    if scan == "monday":
        days = mondays_on_index(spots.index, start, end)
    elif scan == "month_start":
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

    open_until: List[date] = []
    n_legs = 4 if structure == "atm_call" else 8
    commission = float(commission_per_leg) * float(n_legs)

    close_by_day: Dict[date, float] = {}
    for ts, row in spots.iterrows():
        close_by_day[ts.date() if hasattr(ts, "date") else pd.Timestamp(ts).date()] = float(
            row["close"]
        )

    for day in days:
        open_until = [x for x in open_until if x > day]
        if len(open_until) >= max_concurrent:
            continue

        spot = close_by_day.get(day)
        if spot is None or spot <= 0:
            continue

        front = pick_expiry(
            exps,
            day,
            dte_front,
            dte_tol,
            friday_only=friday_only,
            monthly_only=monthly_only,
        )
        back = pick_expiry(
            exps,
            day,
            dte_back,
            dte_tol,
            friday_only=friday_only,
            monthly_only=monthly_only,
        )
        if front is None or back is None or back <= front:
            book.skipped.append({"date": str(day), "reason": "no_expiry_pair"})
            continue

        if skip_earnings and has_earnings_in_window(symbol, day, front, earnings_map):
            book.skipped.append({"date": str(day), "reason": "earnings_in_window"})
            continue

        if front not in strike_cache:
            strike_cache[front] = list_strikes(client, symbol, front)
        if back not in strike_cache:
            strike_cache[back] = list_strikes(client, symbol, back)
        common = sorted(set(strike_cache[front]) & set(strike_cache[back]))
        if not common:
            book.skipped.append({"date": str(day), "reason": "no_common_strike"})
            continue

        # Seed IV from ATM call mids on both expiries for FF + delta snaps
        atm = nearest_strike(common, spot)
        if atm is None:
            book.skipped.append({"date": str(day), "reason": "no_strike"})
            continue

        hard_exit = front - timedelta(days=1)
        while hard_exit.weekday() >= 5:
            hard_exit -= timedelta(days=1)
        if hard_exit <= day:
            book.skipped.append({"date": str(day), "reason": "exit_before_entry"})
            continue

        # Prefetch ATM call path for FF gate (and atm_call structure)
        fc_atm = fetch_eod(
            client,
            symbol=symbol,
            expiration=front,
            strike=float(atm),
            right="call",
            start=day,
            end=hard_exit,
            cache=cache,
        )
        bc_atm = fetch_eod(
            client,
            symbol=symbol,
            expiration=back,
            strike=float(atm),
            right="call",
            start=day,
            end=hard_exit,
            cache=cache,
        )
        fe0 = eod_on(fc_atm, day)
        be0 = eod_on(bc_atm, day)
        if fe0 is None or be0 is None:
            book.skipped.append({"date": str(day), "reason": "no_entry_quote"})
            continue

        dte1 = float((front - day).days)
        dte2 = float((back - day).days)
        iv1 = estimate_iv(mid=fe0["mid"], spot=spot, strike=float(atm), dte_days=dte1)
        iv2 = estimate_iv(mid=be0["mid"], spot=spot, strike=float(atm), dte_days=dte2)
        if iv1 is None or iv2 is None:
            book.skipped.append({"date": str(day), "reason": "iv_fail"})
            continue
        ivf = forward_vol(iv1, iv2, dte1 / 365.0, dte2 / 365.0)
        ff = forward_factor(iv1, ivf) if ivf else None
        if ff is None:
            book.skipped.append({"date": str(day), "reason": "ff_undefined"})
            continue
        if ff < ff_min:
            book.skipped.append({"date": str(day), "reason": "ff_below", "ff": round(ff, 4)})
            continue

        put_strike: Optional[float] = None
        call_strike = float(atm)
        frames: Dict[str, pd.DataFrame] = {}

        if structure == "atm_call":
            frames["front_call"] = fc_atm
            frames["back_call"] = bc_atm
        else:
            # +/- target_delta double calendar
            call_strike = strike_for_abs_delta(
                spot=spot,
                strikes=common,
                dte_days=dte1,
                iv=iv1,
                right="call",
                target_abs_delta=target_delta,
            )
            put_strike = strike_for_abs_delta(
                spot=spot,
                strikes=common,
                dte_days=dte1,
                iv=iv1,
                right="put",
                target_abs_delta=target_delta,
            )
            if call_strike is None or put_strike is None:
                book.skipped.append({"date": str(day), "reason": "no_delta_strike"})
                continue
            frames["front_call"] = fetch_eod(
                client,
                symbol=symbol,
                expiration=front,
                strike=float(call_strike),
                right="call",
                start=day,
                end=hard_exit,
                cache=cache,
            )
            frames["back_call"] = fetch_eod(
                client,
                symbol=symbol,
                expiration=back,
                strike=float(call_strike),
                right="call",
                start=day,
                end=hard_exit,
                cache=cache,
            )
            frames["front_put"] = fetch_eod(
                client,
                symbol=symbol,
                expiration=front,
                strike=float(put_strike),
                right="put",
                start=day,
                end=hard_exit,
                cache=cache,
            )
            frames["back_put"] = fetch_eod(
                client,
                symbol=symbol,
                expiration=back,
                strike=float(put_strike),
                right="put",
                start=day,
                end=hard_exit,
                cache=cache,
            )

        entry_legs = _legs_on_day(frames, day, structure)
        if entry_legs is None:
            book.skipped.append({"date": str(day), "reason": "no_entry_package"})
            continue
        entry_debit = package_debit(
            entry_legs, mark_mode=mark_mode, slip_frac=slip_frac, structure=structure
        )
        if entry_debit <= 0.05:
            book.skipped.append(
                {"date": str(day), "reason": "entry_not_debit", "debit": round(entry_debit, 3)}
            )
            continue

        # Dynamic management path
        exit_day = hard_exit
        exit_reason = "hard_front"
        exit_credit = 0.0
        path_days = session_days_between(close_by_day, day, hard_exit)
        if not path_days:
            book.skipped.append({"date": str(day), "reason": "no_path_days"})
            continue

        closed = False
        for xd in path_days:
            legs = _legs_on_day(frames, xd, structure)
            if legs is None:
                continue
            credit = package_credit(
                legs, mark_mode=mark_mode, slip_frac=slip_frac, structure=structure
            )
            pnl_now = (credit - entry_debit) * MULT - commission
            ret_now = pnl_now / (entry_debit * MULT)
            dte_left = (front - xd).days

            reason = None
            if dynamic_exit:
                if ret_now >= float(tp_pct):
                    reason = "tp"
                elif ret_now <= -float(sl_pct):
                    reason = "sl"
                elif dte_left <= int(time_stop_dte):
                    reason = "time_stop"
                elif float(ff) >= max(float(ff_exit) + 0.05, 0.10):
                    # FF normalize only if entry was an elevated spike
                    fe = eod_on(frames["front_call"], xd)
                    be = eod_on(frames["back_call"], xd)
                    spot_x = close_by_day.get(xd)
                    if fe and be and spot_x:
                        iv1x = estimate_iv(
                            mid=fe["mid"],
                            spot=spot_x,
                            strike=float(call_strike),
                            dte_days=max(dte_left, 1),
                        )
                        iv2x = estimate_iv(
                            mid=be["mid"],
                            spot=spot_x,
                            strike=float(call_strike),
                            dte_days=max((back - xd).days, 1),
                        )
                        if iv1x and iv2x:
                            fvx = forward_vol(
                                iv1x,
                                iv2x,
                                max(dte_left, 1) / 365.0,
                                max((back - xd).days, 1) / 365.0,
                            )
                            ffx = forward_factor(iv1x, fvx) if fvx else None
                            if ffx is not None and ffx < float(ff_exit):
                                reason = "ff_normalize"

            if reason is not None or xd == path_days[-1]:
                exit_day = xd
                exit_credit = credit
                exit_reason = reason or "hard_front"
                closed = True
                break

        if not closed:
            book.skipped.append({"date": str(day), "reason": "no_exit_quote"})
            continue

        pnl = (exit_credit - entry_debit) * MULT - commission
        ret_pct = 100.0 * pnl / (entry_debit * MULT)
        trade = FFTrade(
            symbol=symbol,
            entry_date=day,
            exit_date=exit_day,
            front=front,
            back=back,
            strike=float(call_strike),
            put_strike=put_strike,
            spot_entry=spot,
            front_iv=iv1,
            back_iv=iv2,
            fwd_iv=float(ivf),
            ff=float(ff),
            mark_mode=mark_mode,
            structure=structure,
            entry_debit=entry_debit,
            exit_credit=exit_credit,
            pnl=pnl,
            ret_pct=ret_pct,
            hold_days=(exit_day - day).days,
            commission=commission,
            exit_reason=exit_reason,
        )
        book.trades.append(trade)
        open_until.append(trade.exit_date)
        LOG.info(
            "%s %s %s %s FF=%.2f debit=%.2f pnl=$%.0f ret=%.1f%% exit=%s hold=%d",
            symbol,
            day,
            structure,
            mark_mode,
            ff,
            entry_debit,
            pnl,
            ret_pct,
            exit_reason,
            trade.hold_days,
        )

    return book


def run_ff_panel(
    *,
    symbols: Sequence[str],
    start: date,
    end: date,
    mark_mode: str = "mid_slip",
    ff_min: float = DEFAULT_FF_THRESH,
    dte_front: int = DEFAULT_DTE_FRONT,
    dte_back: int = DEFAULT_DTE_BACK,
    dte_tol: int = DEFAULT_DTE_TOL,
    client: Optional[ThetaDataClient] = None,
    friday_only: bool = True,
    monthly_only: bool = False,
    scan: str = "monday",
    structure: str = "atm_call",
    slip_frac: float = DEFAULT_SLIP_FRAC,
    tp_pct: float = DEFAULT_TP,
    sl_pct: float = DEFAULT_SL,
    ff_exit: float = DEFAULT_FF_EXIT,
    time_stop_dte: int = DEFAULT_TIME_STOP_DTE,
    skip_earnings: bool = True,
    earnings_map: Optional[Dict[str, List[date]]] = None,
    dynamic_exit: bool = True,
) -> BookResult:
    client = client or get_thetadata_client()
    label = f"panel_{structure}_{dte_front}_{dte_back}_ff{ff_min:g}_{mark_mode}"
    out = BookResult(label=label, mark_mode=mark_mode, ff_min=ff_min)
    if earnings_map is None and skip_earnings:
        earnings_map = load_earnings_map(symbols)
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
                friday_only=friday_only,
                monthly_only=monthly_only,
                scan=scan,
                structure=structure,
                slip_frac=slip_frac,
                tp_pct=tp_pct,
                sl_pct=sl_pct,
                ff_exit=ff_exit,
                time_stop_dte=time_stop_dte,
                skip_earnings=skip_earnings,
                earnings_map=earnings_map,
                dynamic_exit=dynamic_exit,
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
                "structure": t.structure,
                "entry_date": t.entry_date.isoformat(),
                "exit_date": t.exit_date.isoformat(),
                "exit_reason": t.exit_reason,
                "front": t.front.isoformat(),
                "back": t.back.isoformat(),
                "strike": t.strike,
                "put_strike": t.put_strike,
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
    "has_earnings_in_window",
    "list_expirations",
    "load_daily",
    "load_earnings_map",
    "load_spy_daily",
    "pick_expiry",
    "run_ff_book",
    "run_ff_panel",
    "trades_to_frame",
]
