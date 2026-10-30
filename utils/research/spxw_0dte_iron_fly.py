"""Jason Rutschel 0DTE SPXW short iron-fly claim-check (listed Theta NBBO).

Transcript rules (Theta Profits interview):
  - Enter 13:45 ET on same-day SPXW expiry
  - Sell put nearest 50 delta, sell call nearest 40 delta
  - Buy wings 15 points further (10-pt book is a smaller-account variant)
  - Skip Wednesday; prefer Mon/Tue/Thu/Fri
  - Only when VIX is below 20 (prefer ~15-16)
  - Hold to cash settle (automated book); optional 50% TP is discretionary

Marks: ``mid`` sells/buys at mid; ``natural`` sells shorts at bid and buys
wings at ask. Settlement uses SPX EOD close (SPXW is PM-settled).
"""
from __future__ import annotations

import logging
import math
import time
from datetime import date, datetime, time as dtime, timedelta
from io import StringIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)

CLAIMED: Dict[str, Any] = {
    "source": "Jason Rutschel / Theta Profits interview (0DTE short iron fly)",
    "window": "about 2 years ending interview date",
    "n": 299,
    "win_rate": 0.571,
    "avg_win": 513.0,
    "avg_loss": 492.0,
    "start_equity": 25_000.0,
    "cagr": 0.44,
    "structure": "short 50d put / 40d call, 15-pt wings, 13:45 ET, VIX<20, skip Wed",
}

FILL_MID = "mid"
FILL_NATURAL = "natural"

OPT_SYMBOL = "SPXW"
ENTRY_CLOCK = dtime(13, 45)
DEFAULT_PUT_DELTA = 0.50
DEFAULT_CALL_DELTA = 0.40
DEFAULT_WING = 15.0
DEFAULT_DELTA_TOL = 0.12
DEFAULT_MULT = 100.0
DEFAULT_VIX_MAX = 20.0
# Mon=0 ... Fri=4; skip Wednesday (2)
DEFAULT_WEEKDAYS = (0, 1, 3, 4)


def _penny(x: float) -> float:
    return float(np.round(float(x) + 1e-12, 2))


def _theta_csv(
    client: Any,
    url_path: str,
    params: Dict[str, Any],
    *,
    timeout: float = 90.0,
    attempts: int = 4,
) -> Optional[pd.DataFrame]:
    last = ""
    for attempt in range(attempts):
        try:
            r = client.session.get(
                f"{client.base_url}{url_path}", params=params, timeout=timeout
            )
        except Exception as exc:  # noqa: BLE001
            last = str(exc)
            time.sleep(1.5 * (attempt + 1))
            continue
        if r.status_code in (429, 478) or r.status_code >= 500:
            last = f"{r.status_code}"
            time.sleep(1.5 * (attempt + 1))
            continue
        if r.status_code == 472:
            return pd.DataFrame()
        if r.status_code == 400:
            last = (r.text or "")[:160]
            return pd.DataFrame()
        if r.status_code != 200 or not r.content:
            LOG.info("Theta %s %s %s", r.status_code, url_path, params)
            return pd.DataFrame()
        try:
            return pd.read_csv(StringIO(r.text))
        except Exception as exc:  # noqa: BLE001
            last = str(exc)
            time.sleep(1.0)
    LOG.warning("Theta failed %s (%s)", url_path, last)
    return None


def _read_cached(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        if path.suffixes[-2:] == [".csv", ".gz"] or str(path).endswith(".csv.gz"):
            return pd.read_csv(path, compression="gzip")
        return pd.read_csv(path)
    except Exception:  # noqa: BLE001
        return None


def _write_cached(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if str(path).endswith(".csv.gz"):
        df.to_csv(path, index=False, compression="gzip")
    else:
        df.to_csv(path, index=False)


def _month_chunks(start: date, end: date) -> List[Tuple[date, date]]:
    out: List[Tuple[date, date]] = []
    cur = date(start.year, start.month, 1)
    while cur <= end:
        if cur.month == 12:
            nxt = date(cur.year + 1, 1, 1)
        else:
            nxt = date(cur.year, cur.month + 1, 1)
        lo = max(cur, start)
        hi = min(end, nxt - timedelta(days=1))
        if lo <= hi:
            out.append((lo, hi))
        cur = nxt
    return out


def _yahoo_index_eod(tick: str, start: date, end: date) -> pd.DataFrame:
    import yfinance as yf

    hist = yf.download(
        tick,
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        progress=False,
        auto_adjust=False,
    )
    if hist is None or hist.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close"])
    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = hist.columns.get_level_values(0)
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(hist.index).date,
            "open": pd.to_numeric(hist["Open"], errors="coerce"),
            "high": pd.to_numeric(hist["High"], errors="coerce"),
            "low": pd.to_numeric(hist["Low"], errors="coerce"),
            "close": pd.to_numeric(hist["Close"], errors="coerce"),
        }
    )
    return out.dropna(subset=["date", "close"]).drop_duplicates("date").reset_index(drop=True)


def load_index_eod(
    client: Any,
    symbol: str,
    start: date,
    end: date,
    *,
    yahoo: Optional[str] = None,
    prefer_yahoo: bool = False,
) -> pd.DataFrame:
    """Daily OHLC for an index. Theta FREE needs short windows; Yahoo fallback.

    Partial Theta coverage (e.g. 403s through 2023, then 2024+ OK) must not
    short-circuit Yahoo — otherwise expand-to-2022 silently starts in 2024.
    """
    tick = yahoo or ("^VIX" if symbol.upper() == "VIX" else "^GSPC")
    if prefer_yahoo:
        LOG.info("prefer_yahoo: %s via %s", symbol, tick)
        return _yahoo_index_eod(tick, start, end)

    frames: List[pd.DataFrame] = []
    for lo, hi in _month_chunks(start, end):
        df = _theta_csv(
            client,
            "/v3/index/history/eod",
            {
                "symbol": str(symbol).upper(),
                "start_date": lo.isoformat(),
                "end_date": hi.isoformat(),
            },
            timeout=120.0,
        )
        if df is None or df.empty:
            continue
        frames.append(df)
    if frames:
        raw = pd.concat(frames, ignore_index=True)
        raw.columns = [str(c).strip().lower() for c in raw.columns]
        stamp = "created" if "created" in raw.columns else raw.columns[0]
        out = pd.DataFrame(
            {
                "date": pd.to_datetime(raw[stamp], errors="coerce").dt.date,
                "open": pd.to_numeric(raw.get("open"), errors="coerce"),
                "high": pd.to_numeric(raw.get("high"), errors="coerce"),
                "low": pd.to_numeric(raw.get("low"), errors="coerce"),
                "close": pd.to_numeric(raw.get("close"), errors="coerce"),
            }
        )
        out = out.dropna(subset=["date", "close"]).drop_duplicates("date").sort_values("date")
        if not out.empty:
            first = out["date"].iloc[0]
            # Tolerate a few sessions of holiday/weekend lag at the left edge.
            if first <= start + timedelta(days=7):
                return out.reset_index(drop=True)
            LOG.warning(
                "Theta %s EOD starts %s (requested %s); Yahoo %s for full window",
                symbol,
                first,
                start,
                tick,
            )
            return _yahoo_index_eod(tick, start, end)

    LOG.warning("Theta %s EOD empty/short; Yahoo %s", symbol, tick)
    return _yahoo_index_eod(tick, start, end)


def fetch_chain(
    client: Any,
    day: date,
    cache_dir: Path,
    *,
    clock: dtime = ENTRY_CLOCK,
) -> Optional[pd.DataFrame]:
    """1m first-order greeks for both rights at the entry minute."""
    stamp = f"{int(clock.hour):02d}{int(clock.minute):02d}"
    path = cache_dir / "chains" / f"{day.isoformat()}_{stamp}.csv.gz"
    cached = _read_cached(path)
    if cached is not None:
        return cached
    start_s = f"{int(clock.hour):02d}:{int(clock.minute):02d}:00"
    end_minute = (datetime.combine(day, clock) + timedelta(minutes=1)).time()
    end_s = f"{int(end_minute.hour):02d}:{int(end_minute.minute):02d}:00"
    params = {
        "symbol": OPT_SYMBOL,
        "expiration": day.isoformat(),
        "strike": "*",
        "right": "both",
        "date": day.isoformat(),
        "interval": "1m",
        "start_time": start_s,
        "end_time": end_s,
    }
    df = _theta_csv(client, "/v3/option/history/greeks/first_order", params)
    if df is None:
        return None
    keep = [
        c
        for c in (
            "timestamp",
            "strike",
            "right",
            "bid",
            "ask",
            "delta",
            "underlying_price",
            "implied_vol",
            "iv_error",
        )
        if c in df.columns or df.empty
    ]
    if not df.empty:
        df = df[[c for c in keep if c in df.columns]].copy()
    _write_cached(path, df)
    return df


def _coerce_chain(chain: pd.DataFrame, clock: dtime = ENTRY_CLOCK) -> pd.DataFrame:
    if chain is None or chain.empty:
        return pd.DataFrame()
    df = chain.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "timestamp" not in df.columns:
        return pd.DataFrame()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    if df.empty:
        return df
    if df["timestamp"].dt.tz is not None:
        df["timestamp"] = df["timestamp"].dt.tz_convert("America/New_York")
    at_entry = (df["timestamp"].dt.hour == clock.hour) & (
        df["timestamp"].dt.minute == clock.minute
    )
    if bool(at_entry.any()):
        df = df.loc[at_entry].copy()
    else:
        t0 = df["timestamp"].min()
        df = df.loc[df["timestamp"] == t0].copy()
    for col in ("bid", "ask", "delta", "strike", "underlying_price", "iv_error", "implied_vol"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "right" in df.columns:
        df["right"] = df["right"].astype(str).str.upper().str[0]
    return df


def _valid_quotes(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.dropna(subset=["bid", "ask", "strike"]).copy()
    if "iv_error" in out.columns:
        err = out["iv_error"]
        out = out.loc[err.isna() | (err.abs() < 1.0)].copy()
    out = out.loc[(out["bid"] > 0) & (out["ask"] > 0) & (out["ask"] >= out["bid"])].copy()
    if "underlying_price" in out.columns:
        spot = out["underlying_price"]
        out = out.loc[spot.notna() & (spot > 0)].copy()
        out = out.loc[(out["strike"] - spot).abs() <= 200.0].copy()
    return out


def chain_spot(chain: pd.DataFrame, clock: dtime = ENTRY_CLOCK) -> Optional[float]:
    df = _valid_quotes(_coerce_chain(chain, clock))
    if df.empty or "underlying_price" not in df.columns:
        return None
    spot = float(df["underlying_price"].median())
    if not math.isfinite(spot) or spot <= 0:
        return None
    return spot


def _pick_by_delta(
    side: pd.DataFrame,
    *,
    target_abs_delta: float,
    delta_tol: float,
) -> Optional[pd.Series]:
    if side.empty or "delta" not in side.columns:
        return None
    work = side.dropna(subset=["delta"]).copy()
    if work.empty:
        return None
    work["abs_delta"] = work["delta"].abs()
    work["score"] = (work["abs_delta"] - float(target_abs_delta)).abs()
    work = work.sort_values(["score", "strike"])
    if float(work.iloc[0]["score"]) > float(delta_tol):
        return None
    return work.iloc[0]


def _quote_at_strike(side: pd.DataFrame, strike: float) -> Optional[pd.Series]:
    if side.empty:
        return None
    hits = side.loc[(side["strike"] - float(strike)).abs() < 0.01]
    if hits.empty:
        return None
    return hits.iloc[0]


def select_iron_fly(
    chain: pd.DataFrame,
    *,
    put_delta: float = DEFAULT_PUT_DELTA,
    call_delta: float = DEFAULT_CALL_DELTA,
    wing: float = DEFAULT_WING,
    delta_tol: float = DEFAULT_DELTA_TOL,
    clock: dtime = ENTRY_CLOCK,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Pick short 50d put / 40d call and 15-pt wings from the entry chain."""
    df = _valid_quotes(_coerce_chain(chain, clock))
    if df.empty or "delta" not in df.columns or "right" not in df.columns:
        return None, "no_chain"
    spot = chain_spot(chain, clock)
    if spot is None:
        return None, "no_spot"
    puts = df.loc[df["right"] == "P"].copy()
    calls = df.loc[df["right"] == "C"].copy()
    sp = _pick_by_delta(puts, target_abs_delta=put_delta, delta_tol=delta_tol)
    sc = _pick_by_delta(calls, target_abs_delta=call_delta, delta_tol=delta_tol)
    if sp is None:
        return None, "no_put"
    if sc is None:
        return None, "no_call"
    short_put = float(sp["strike"])
    short_call = float(sc["strike"])
    long_put_k = short_put - float(wing)
    long_call_k = short_call + float(wing)
    lp = _quote_at_strike(puts, long_put_k)
    lc = _quote_at_strike(calls, long_call_k)
    if lp is None:
        return None, "no_long_put"
    if lc is None:
        return None, "no_long_call"

    def _leg(row: pd.Series) -> Dict[str, float]:
        bid = float(row["bid"])
        ask = float(row["ask"])
        return {
            "strike": float(row["strike"]),
            "delta": float(row["delta"]) if pd.notna(row.get("delta")) else float("nan"),
            "bid": bid,
            "ask": ask,
            "mid": 0.5 * (bid + ask),
        }

    short_put_leg = _leg(sp)
    short_call_leg = _leg(sc)
    long_put_leg = _leg(lp)
    long_call_leg = _leg(lc)
    credit_mid = (
        short_put_leg["mid"]
        + short_call_leg["mid"]
        - long_put_leg["mid"]
        - long_call_leg["mid"]
    )
    credit_nat = (
        short_put_leg["bid"]
        + short_call_leg["bid"]
        - long_put_leg["ask"]
        - long_call_leg["ask"]
    )
    if credit_mid <= 0:
        return None, "no_credit"
    width = float(wing)
    max_loss_mid = (width - credit_mid) * DEFAULT_MULT
    return (
        {
            "spot": float(spot),
            "wing": width,
            "short_put": short_put_leg,
            "short_call": short_call_leg,
            "long_put": long_put_leg,
            "long_call": long_call_leg,
            "credit_mid": float(credit_mid),
            "credit_natural": float(credit_nat),
            "max_loss_mid": float(max_loss_mid),
        },
        "ok",
    )


def settle_short_iron_fly(
    *,
    settle: float,
    short_put: float,
    short_call: float,
    wing: float,
    credit: float,
    mult: float = DEFAULT_MULT,
) -> Dict[str, float]:
    """Cash-settle a short iron fly / narrow iron condor."""
    put_val = max(0.0, min(float(wing), float(short_put) - float(settle)))
    call_val = max(0.0, min(float(wing), float(settle) - float(short_call)))
    pnl_pts = float(credit) - put_val - call_val
    return {
        "put_value": put_val,
        "call_value": call_val,
        "pnl_points": pnl_pts,
        "pnl": pnl_pts * float(mult),
    }


def simulate_day(
    chain: pd.DataFrame,
    *,
    settle: float,
    fill: str = FILL_MID,
    put_delta: float = DEFAULT_PUT_DELTA,
    call_delta: float = DEFAULT_CALL_DELTA,
    wing: float = DEFAULT_WING,
    delta_tol: float = DEFAULT_DELTA_TOL,
    clock: dtime = ENTRY_CLOCK,
) -> Tuple[Optional[Dict[str, Any]], str]:
    picked, reason = select_iron_fly(
        chain,
        put_delta=put_delta,
        call_delta=call_delta,
        wing=wing,
        delta_tol=delta_tol,
        clock=clock,
    )
    if picked is None:
        return None, reason
    credit = (
        float(picked["credit_natural"])
        if fill == FILL_NATURAL
        else float(picked["credit_mid"])
    )
    if credit <= 0:
        return None, "no_credit"
    settled = settle_short_iron_fly(
        settle=float(settle),
        short_put=float(picked["short_put"]["strike"]),
        short_call=float(picked["short_call"]["strike"]),
        wing=float(picked["wing"]),
        credit=credit,
    )
    row = {
        "fill": fill,
        "spot": picked["spot"],
        "settle": float(settle),
        "wing": picked["wing"],
        "short_put": picked["short_put"]["strike"],
        "short_call": picked["short_call"]["strike"],
        "long_put": picked["long_put"]["strike"],
        "long_call": picked["long_call"]["strike"],
        "put_delta": picked["short_put"]["delta"],
        "call_delta": picked["short_call"]["delta"],
        "credit": _penny(credit),
        "credit_mid": _penny(picked["credit_mid"]),
        "credit_natural": _penny(picked["credit_natural"]),
        "max_loss": _penny((picked["wing"] - credit) * DEFAULT_MULT),
        "put_value": settled["put_value"],
        "call_value": settled["call_value"],
        "pnl": settled["pnl"],
        "won": settled["pnl"] > 0,
    }
    return row, "ok"


def eligible_days(
    expirations: Sequence[date],
    *,
    start: date,
    end: date,
    weekdays: Sequence[int] = DEFAULT_WEEKDAYS,
    vix_by_day: Optional[Dict[date, float]] = None,
    vix_max: float = DEFAULT_VIX_MAX,
) -> List[date]:
    """0DTE sessions that pass weekday + VIX filters."""
    out: List[date] = []
    for d in expirations:
        if d < start or d > end:
            continue
        if int(d.weekday()) not in set(int(x) for x in weekdays):
            continue
        if vix_by_day is not None:
            v = vix_by_day.get(d)
            if v is None or not math.isfinite(float(v)) or float(v) >= float(vix_max):
                continue
        out.append(d)
    return out


def vix_open_map(vix: pd.DataFrame) -> Dict[date, float]:
    """Prefer session open; fall back to close."""
    out: Dict[date, float] = {}
    if vix is None or vix.empty:
        return out
    for _, row in vix.iterrows():
        d = row["date"]
        if isinstance(d, datetime):
            d = d.date()
        open_ = row.get("open")
        close = row.get("close")
        val = open_ if pd.notna(open_) and float(open_) > 0 else close
        if pd.isna(val):
            continue
        out[d] = float(val)
    return out


def spx_close_map(spx: pd.DataFrame) -> Dict[date, float]:
    out: Dict[date, float] = {}
    if spx is None or spx.empty:
        return out
    for _, row in spx.iterrows():
        d = row["date"]
        if isinstance(d, datetime):
            d = d.date()
        c = row.get("close")
        if pd.isna(c):
            continue
        out[d] = float(c)
    return out


def summarize_pnls(pnls: Sequence[float]) -> Dict[str, Any]:
    arr = np.asarray([float(x) for x in pnls], dtype=float)
    if arr.size == 0:
        return {
            "n": 0,
            "win_rate": None,
            "avg_win": None,
            "avg_loss": None,
            "expectancy": None,
            "total": 0.0,
            "pf": None,
        }
    wins = arr[arr > 0]
    losses = arr[arr <= 0]
    gross_win = float(wins.sum()) if wins.size else 0.0
    gross_loss = float((-losses).sum()) if losses.size else 0.0
    pf = (gross_win / gross_loss) if gross_loss > 0 else None
    return {
        "n": int(arr.size),
        "win_rate": float((arr > 0).mean()),
        "avg_win": float(wins.mean()) if wins.size else None,
        "avg_loss": float((-losses).mean()) if losses.size else None,
        "expectancy": float(arr.mean()),
        "total": float(arr.sum()),
        "pf": pf,
        "max_dd_one_lot": float(_max_dd(arr)),
    }


def _max_dd(pnls: np.ndarray) -> float:
    equity = np.cumsum(pnls)
    peak = np.maximum.accumulate(equity)
    dd = peak - equity
    return float(dd.max()) if dd.size else 0.0


def compound_fixed_contracts(
    trades: Sequence[Dict[str, Any]],
    *,
    start_equity: float = 25_000.0,
    contracts: int = 1,
) -> Dict[str, Any]:
    """Simple 1-lot (or N-lot) equity path; risk is capped by the spread."""
    equity = float(start_equity)
    peak = equity
    max_dd = 0.0
    for t in trades:
        equity += float(t["pnl"]) * int(contracts)
        peak = max(peak, equity)
        if peak > 0:
            max_dd = max(max_dd, (peak - equity) / peak)
    years = None
    if trades:
        d0 = trades[0]["date"]
        d1 = trades[-1]["date"]
        days = max((d1 - d0).days, 1)
        years = days / 365.25
    cagr = None
    if years and years > 0 and equity > 0 and start_equity > 0:
        cagr = (equity / float(start_equity)) ** (1.0 / years) - 1.0
    return {
        "start_equity": float(start_equity),
        "end_equity": float(equity),
        "profit": float(equity - start_equity),
        "cagr": cagr,
        "max_dd": float(max_dd),
        "contracts": int(contracts),
        "years": years,
    }


def claimed_one_lot_pnl() -> float:
    """Implied one-lot total from claimed averages (approx)."""
    n = int(CLAIMED["n"])
    wr = float(CLAIMED["win_rate"])
    winners = int(round(n * wr))
    losers = n - winners
    return winners * float(CLAIMED["avg_win"]) - losers * float(CLAIMED["avg_loss"])
