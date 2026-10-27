"""QQQ 0-DTE 9:45 short-put claim-check (listed Theta NBBO).

Video rules (Options With Ravish, 7S-mGV4D-fI, published 2026-10-24):
  - Each morning a same-day QQQ expiry exists, at 09:45 ET
  - Sell the put nearest 30 delta (the video also says "2-3 points below")
  - Take profit: keep 95% of the credit (buy back at 5% of the credit)
  - Stop: loss of 200% of the credit (buy back at 3x the credit).
    The video's example is sell $100, stop at $300, loss capped at $200.
  - If neither prints, buy back at 15:45 ET

Marks are 1-minute NBBO snapshots, not ticks. A stop that spikes and reverts
inside a minute is invisible. ``target`` fills at the bracket price when the
mid crosses it (the no-slippage backtest the video says it ran). ``mid`` fills
at the mid, so a gap through the stop is a worse fill. ``natural`` sells the
bid and buys the ask.
"""
from __future__ import annotations

import logging
import math
import time
from datetime import date, datetime, time as dtime
from io import StringIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)

CLAIMED: Dict[str, Any] = {
    "source": "Options With Ravish, YouTube 7S-mGV4D-fI (2026-10-24)",
    "window": "May 2022 through 2026",
    "n": 1067,
    "winners": 782,
    "win_rate": 0.733,
    "avg_win": 85.0,
    "avg_loss": 174.0,
    "start_equity": 100_000.0,
    "profit": 556_000.0,
    "cagr": 0.52,
    "max_dd": 0.13,
    "bp_per_contract": 10_000.0,
}

FILL_TARGET = "target"
FILL_MID = "mid"
FILL_NATURAL = "natural"

DEFAULT_TARGET_DELTA = -0.30
DEFAULT_DELTA_TOL = 0.08
DEFAULT_TP_KEEP = 0.95
DEFAULT_SL_LOSS_MULT = 2.0
DEFAULT_POINTS = 2.5
DEFAULT_MULT = 100.0
DEFAULT_FEE_SIDE = 0.65
ENTRY_CLOCK = dtime(9, 45)
FLAT_CLOCK = dtime(15, 45)


def _penny(x: float) -> float:
    return float(np.round(float(x) + 1e-12, 2))


def approx_put_margin(spot: float, strike: float, credit: float) -> float:
    """Reg-T style naked-put buying power, in dollars per contract.

    max(20% of spot minus OTM, 10% of spot) plus the credit, times 100.
    Not portfolio margin and not a broker's 0-DTE shock.
    """
    spot_f = float(spot)
    otm = max(spot_f - float(strike), 0.0)
    per_share = max(0.20 * spot_f - otm, 0.10 * spot_f, 0.0) + max(float(credit), 0.0)
    return float(per_share * DEFAULT_MULT)


def _coerce_chain(chain: pd.DataFrame) -> pd.DataFrame:
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
    at_entry = (df["timestamp"].dt.hour == ENTRY_CLOCK.hour) & (
        df["timestamp"].dt.minute == ENTRY_CLOCK.minute
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
        right = df["right"].astype(str).str.lower()
        df = df.loc[right.str.startswith("p")].copy()
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
        # Drop strikes that are not a real near-the-money quote.
        out = out.loc[(out["strike"] - spot).abs() <= spot * 0.20].copy()
    return out


def chain_spot(chain: pd.DataFrame) -> Optional[float]:
    df = _valid_quotes(_coerce_chain(chain))
    if df.empty or "underlying_price" not in df.columns:
        return None
    spot = float(df["underlying_price"].median())
    if not math.isfinite(spot) or spot <= 0:
        return None
    return spot


def select_put(
    chain: pd.DataFrame,
    *,
    target_delta: float = DEFAULT_TARGET_DELTA,
    delta_tol: float = DEFAULT_DELTA_TOL,
    points_below: Optional[float] = None,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Pick the 09:45 put.

    ``points_below`` set: strike nearest to spot minus that many points.
    Otherwise: strike whose delta is nearest ``target_delta`` (negative).
    """
    df = _valid_quotes(_coerce_chain(chain))
    if df.empty or "delta" not in df.columns:
        return None, "no_chain"
    spot = chain_spot(chain)
    if spot is None:
        return None, "no_spot"
    deltas = df["delta"].dropna()
    if deltas.empty:
        return None, "no_delta"
    if float((deltas > 0).mean()) > 0.8:
        df = df.copy()
        df["delta"] = -df["delta"].abs()
    df = df.loc[df["delta"] < 0].copy()
    if df.empty:
        return None, "no_delta"
    if points_below is not None:
        target_k = float(spot) - float(points_below)
        df = df.copy()
        df["score"] = (df["strike"] - target_k).abs()
        df = df.sort_values(["score", "strike"])
        # $1 strikes sit 0.5 away from an x.5 target. Farther than that is a different trade.
        if float(df.iloc[0]["score"]) > 1.5:
            return None, "no_strike"
    else:
        df = df.copy()
        df["score"] = (df["delta"] - float(target_delta)).abs()
        df = df.sort_values(["score", "strike"])
        if float(df.iloc[0]["score"]) > float(delta_tol):
            return None, "no_delta"
    row = df.iloc[0]
    bid = float(row["bid"])
    ask = float(row["ask"])
    strike = float(row["strike"])
    delta = float(row["delta"])
    return (
        {
            "strike": strike,
            "spot": float(spot),
            "delta": delta,
            "points_otm": float(spot) - strike,
            "bid": bid,
            "ask": ask,
            "mid": 0.5 * (bid + ask),
            "spread": ask - bid,
            "iv": float(row["implied_vol"]) if "implied_vol" in row and pd.notna(row["implied_vol"]) else None,
        },
        "ok",
    )


def _as_et_time(ts: Any) -> Optional[dtime]:
    if ts is None or (isinstance(ts, float) and math.isnan(ts)):
        return None
    stamp = pd.Timestamp(ts)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("America/New_York")
    return stamp.time()


def _prepare_path(path: pd.DataFrame, strike: float) -> pd.DataFrame:
    if path is None or path.empty:
        return pd.DataFrame()
    df = path.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "timestamp" not in df.columns:
        return pd.DataFrame()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["bid"] = pd.to_numeric(df.get("bid"), errors="coerce")
    df["ask"] = pd.to_numeric(df.get("ask"), errors="coerce")
    df = df.dropna(subset=["timestamp", "bid", "ask"])
    df = df.loc[(df["bid"] >= 0) & (df["ask"] > 0) & (df["ask"] >= df["bid"])].copy()
    cap = float(strike) + 1.0
    df = df.loc[df["ask"] <= cap].copy()
    if df.empty:
        return df
    df = df.sort_values("timestamp")
    times = df["timestamp"].map(_as_et_time)
    df = df.loc[times.map(lambda t: t is not None and t >= ENTRY_CLOCK)].copy()
    return df.reset_index(drop=True)


def simulate_short_put(
    path: pd.DataFrame,
    *,
    strike: float,
    mode: str = FILL_TARGET,
    tp_keep: float = DEFAULT_TP_KEEP,
    sl_loss_mult: float = DEFAULT_SL_LOSS_MULT,
    fee_per_side: float = 0.0,
    flat_clock: dtime = FLAT_CLOCK,
) -> Dict[str, Any]:
    """Walk one short put from the first 09:45 quote.

    Returns pnl_1lot in dollars. Missing path -> reason ``no_path`` and pnl None.
    """
    df = _prepare_path(path, strike)
    empty = {
        "credit": None,
        "exit_px": None,
        "exit_reason": "no_path",
        "exit_time": None,
        "pnl_1lot": None,
        "entry_bid": None,
        "entry_ask": None,
    }
    if df.empty:
        return empty
    entry = df.iloc[0]
    bid = float(entry["bid"])
    ask = float(entry["ask"])
    if bid <= 0 or ask <= 0:
        empty["exit_reason"] = "no_credit"
        return empty
    use_natural = str(mode).lower() == FILL_NATURAL
    credit = bid if use_natural else 0.5 * (bid + ask)
    if credit <= 0:
        empty["exit_reason"] = "no_credit"
        return empty
    tp_px = credit * (1.0 - float(tp_keep))
    sl_px = credit * (1.0 + float(sl_loss_mult))
    if use_natural:
        tp_px = max(_penny(tp_px), 0.01)
        sl_px = max(_penny(sl_px), tp_px + 0.01)

    def _close(exit_px: float, reason: str, ts: Any) -> Dict[str, Any]:
        gross = (credit - float(exit_px)) * DEFAULT_MULT
        fee = 2.0 * float(fee_per_side)
        stamp = pd.Timestamp(ts)
        return {
            "credit": float(credit),
            "exit_px": float(exit_px),
            "exit_reason": reason,
            "exit_time": stamp.isoformat(),
            "pnl_1lot": float(gross - fee),
            "entry_bid": bid,
            "entry_ask": ask,
        }

    if len(df) < 2:
        return _close(ask if use_natural else 0.5 * (bid + ask), "flat", entry["timestamp"])

    last_usable: Optional[Tuple[float, Any, str]] = None
    for _, row in df.iloc[1:].iterrows():
        rb = float(row["bid"])
        ra = float(row["ask"])
        mid = 0.5 * (rb + ra)
        clock = _as_et_time(row["timestamp"])
        if clock is None:
            continue
        past_flat = clock >= flat_clock
        if use_natural:
            mark = ra
            if mark >= sl_px:
                return _close(mark, "sl", row["timestamp"])
            if mark <= tp_px:
                return _close(mark, "tp", row["timestamp"])
            if past_flat:
                return _close(mark, "flat", row["timestamp"])
        else:
            mark = mid
            fill_at_touch = str(mode).lower() == FILL_TARGET
            if mark >= sl_px:
                return _close(sl_px if fill_at_touch else mark, "sl", row["timestamp"])
            if mark <= tp_px:
                return _close(tp_px if fill_at_touch else mark, "tp", row["timestamp"])
            if past_flat:
                return _close(mark, "flat", row["timestamp"])
        # Early close: remember the last afternoon quote and keep walking
        # until the tape ends. A later stop still wins.
        if clock >= dtime(12, 30):
            last_usable = (ra if use_natural else mid, row["timestamp"], "flat")
    if last_usable is None:
        tail = df.iloc[-1]
        rb = float(tail["bid"])
        ra = float(tail["ask"])
        mark = ra if use_natural else 0.5 * (rb + ra)
        return _close(mark, "flat", tail["timestamp"])
    return _close(last_usable[0], last_usable[2], last_usable[1])


def summarize_pnls(pnls: Sequence[float]) -> Dict[str, Any]:
    arr = np.asarray([float(p) for p in pnls if p is not None and math.isfinite(float(p))], dtype=float)
    if arr.size == 0:
        return {
            "n": 0,
            "winners": 0,
            "win_rate": None,
            "avg_win": None,
            "avg_loss": None,
            "pf": None,
            "pnl": 0.0,
            "expectancy": None,
        }
    wins = arr[arr > 0]
    losses = arr[arr <= 0]
    gross_win = float(wins.sum()) if wins.size else 0.0
    gross_loss = float(-losses.sum()) if losses.size else 0.0
    pf = None
    if gross_loss > 0:
        pf = gross_win / gross_loss
    elif gross_win > 0:
        pf = math.inf
    return {
        "n": int(arr.size),
        "winners": int(wins.size),
        "win_rate": float(wins.size / arr.size),
        "avg_win": float(wins.mean()) if wins.size else None,
        "avg_loss": float((-losses).mean()) if losses.size else None,
        "pf": None if pf is None else float(pf),
        "pnl": float(arr.sum()),
        "expectancy": float(arr.mean()),
    }


def _contracts_for(
    equity: float,
    trade: Dict[str, Any],
    *,
    mode: str,
    bp: float,
) -> int:
    if equity <= 0:
        return 0
    m = str(mode).lower()
    if m in ("one", "1", "1lot"):
        return 1
    if m in ("reg_t", "regt"):
        spot = float(trade.get("spot") or 0)
        strike = float(trade.get("strike") or 0)
        credit = float(trade.get("credit") or 0)
        if spot <= 0 or strike <= 0:
            return 0
        margin = approx_put_margin(spot, strike, credit)
    else:
        margin = float(bp)
    if margin <= 0:
        return 0
    return int(math.floor(float(equity) / margin))


def compound_book(
    trades: Sequence[Dict[str, Any]],
    *,
    pnl_key: str = "pnl_1lot",
    start: float = 100_000.0,
    bp: float = 10_000.0,
    mode: str = "flat_bp",
) -> Dict[str, Any]:
    """Size each day from equity at entry and compound closed P&L.

    ``flat_bp`` uses a constant dollar buying-power per contract (the video's
    "$10,000"). ``reg_t`` uses :func:`approx_put_margin`. ``one`` is a single lot.
    Days that do not fit a contract are skipped and leave equity unchanged.
    """
    rows: List[Dict[str, Any]] = []
    equity = float(start)
    peak = float(start)
    max_dd = 0.0
    skipped = 0
    ordered = sorted(trades, key=lambda t: str(t.get("date")))
    for trade in ordered:
        pnl_1 = trade.get(pnl_key)
        if pnl_1 is None or not math.isfinite(float(pnl_1)):
            continue
        n = _contracts_for(equity, trade, mode=mode, bp=bp)
        if n <= 0:
            skipped += 1
            rows.append(
                {
                    "date": trade.get("date"),
                    "contracts": 0,
                    "pnl": 0.0,
                    "equity": equity,
                    "ret": 0.0,
                }
            )
            continue
        equity_before = equity
        pnl = float(pnl_1) * int(n)
        equity = equity_before + pnl
        peak = max(peak, equity)
        if peak > 0:
            max_dd = max(max_dd, (peak - equity) / peak)
        rows.append(
            {
                "date": trade.get("date"),
                "contracts": int(n),
                "pnl": pnl,
                "equity": equity,
                "ret": pnl / equity_before if equity_before else 0.0,
            }
        )
    end = equity
    if rows:
        d0 = _as_date(rows[0]["date"])
        d1 = _as_date(rows[-1]["date"])
        years = max((d1 - d0).days / 365.25, 1e-9) if d0 and d1 else None
    else:
        years = None
    cagr = None
    if years and start > 0 and end > 0:
        cagr = float((end / float(start)) ** (1.0 / years) - 1.0)
    curve = pd.DataFrame(rows)
    return {
        "start": float(start),
        "end": float(end),
        "profit": float(end - float(start)),
        "cagr": cagr,
        "max_dd": float(max_dd),
        "skipped": int(skipped),
        "years": years,
        "curve": curve,
    }


def _as_date(value: Any) -> Optional[date]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value)[:10]
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def max_losing_streak(pnls: Sequence[float]) -> int:
    streak = 0
    best = 0
    for pnl in pnls:
        if pnl is None or not math.isfinite(float(pnl)):
            continue
        if float(pnl) <= 0:
            streak += 1
            best = max(best, streak)
        else:
            streak = 0
    return int(best)


def sharpe_ratio(rets: Sequence[float]) -> Optional[float]:
    arr = np.asarray([float(r) for r in rets if r is not None and math.isfinite(float(r))], dtype=float)
    if arr.size < 2:
        return None
    sd = float(arr.std(ddof=1))
    if sd <= 0:
        return None
    return float(arr.mean() / sd * math.sqrt(252.0))


def daily_returns_on_sessions(
    curve: pd.DataFrame,
    sessions: Iterable[Any],
    *,
    start: float,
) -> pd.Series:
    """Map compounded trade P&L onto session dates. Missing sessions are flat."""
    pnl_by_day: Dict[date, float] = {}
    if curve is not None and not curve.empty:
        for _, row in curve.iterrows():
            d = _as_date(row["date"])
            if d is None:
                continue
            pnl_by_day[d] = pnl_by_day.get(d, 0.0) + float(row["pnl"])
    equity = float(start)
    out: Dict[date, float] = {}
    for raw in sessions:
        d = _as_date(raw)
        if d is None:
            continue
        pnl = float(pnl_by_day.get(d, 0.0))
        ret = pnl / equity if equity else 0.0
        equity = equity + pnl
        out[d] = ret
    return pd.Series(out, dtype=float)


def buyhold_stats(closes: pd.Series, start_equity: float = 100_000.0) -> Dict[str, Any]:
    """``closes`` indexed by session date."""
    s = closes.dropna().astype(float).sort_index()
    if s.size < 2 or float(s.iloc[0]) <= 0:
        return {"cagr": None, "max_dd": None, "sharpe": None, "end": None}
    eq = float(start_equity) * s / float(s.iloc[0])
    peak = eq.cummax()
    dd = eq / peak - 1.0
    d0 = _as_date(s.index[0])
    d1 = _as_date(s.index[-1])
    years = max((d1 - d0).days / 365.25, 1e-9) if d0 and d1 else None
    cagr = None
    if years and float(eq.iloc[-1]) > 0:
        cagr = float((float(eq.iloc[-1]) / float(start_equity)) ** (1.0 / years) - 1.0)
    rets = s.pct_change().dropna()
    return {
        "cagr": cagr,
        "max_dd": float(-dd.min()) if len(dd) else 0.0,
        "sharpe": sharpe_ratio(rets.tolist()),
        "end": float(eq.iloc[-1]),
        "years": years,
    }


def claimed_one_lot_pnl() -> float:
    """Dollar P&L implied by the video's per-trade averages, one contract."""
    n = int(CLAIMED["n"])
    winners = int(CLAIMED["winners"])
    losers = n - winners
    return float(winners * CLAIMED["avg_win"] - losers * CLAIMED["avg_loss"])


# --- Theta fetch (cached). Unit tests do not call these. ---


def _read_cached(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        LOG.warning("cache read failed %s: %s", path, exc)
        return None


def _write_cached(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, compression="gzip")


def _theta_csv(client: Any, url_path: str, params: Dict[str, Any], timeout: float = 90.0) -> Optional[pd.DataFrame]:
    """GET a Theta CSV. None means a retryable failure (do not cache)."""
    url = f"{client.base_url}{url_path}"
    last = ""
    for attempt in range(4):
        try:
            r = client.session.get(url, params=params, timeout=timeout)
        except Exception as exc:
            last = str(exc)
            time.sleep(1.5 * (attempt + 1))
            continue
        if r.status_code in (429, 478) or r.status_code >= 500:
            last = f"{r.status_code}"
            time.sleep(1.5 * (attempt + 1))
            continue
        if r.status_code == 472:
            return pd.DataFrame()
        if r.status_code != 200 or not r.content:
            LOG.info("Theta %s %s %s", r.status_code, url_path, params.get("date") or params.get("expiration"))
            return pd.DataFrame()
        try:
            return pd.read_csv(StringIO(r.text))
        except Exception as exc:
            last = str(exc)
            time.sleep(1.0)
    LOG.warning("Theta failed %s %s (%s)", url_path, params.get("date"), last)
    return None


def fetch_chain(client: Any, day: date, cache_dir: Path) -> Optional[pd.DataFrame]:
    path = cache_dir / "chains" / f"{day.isoformat()}.csv.gz"
    cached = _read_cached(path)
    if cached is not None:
        return cached
    params = {
        "symbol": "QQQ",
        "expiration": day.isoformat(),
        "strike": "*",
        "right": "put",
        "date": day.isoformat(),
        "interval": "1m",
        "start_time": "09:45:00",
        "end_time": "09:46:00",
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


def fetch_path(client: Any, day: date, strike: float, cache_dir: Path) -> Optional[pd.DataFrame]:
    name = f"{day.isoformat()}_{float(strike):.3f}.csv.gz"
    path = cache_dir / "paths" / name
    cached = _read_cached(path)
    if cached is not None:
        return cached
    params = {
        "symbol": "QQQ",
        "expiration": day.isoformat(),
        "strike": f"{float(strike):.3f}",
        "right": "put",
        "date": day.isoformat(),
        "interval": "1m",
        "start_time": "09:45:00",
        "end_time": "16:00:00",
    }
    df = _theta_csv(client, "/v3/option/history/quote", params)
    if df is None:
        return None
    if not df.empty:
        keep = [c for c in ("timestamp", "bid", "ask", "strike") if c in df.columns]
        df = df[keep].copy()
    _write_cached(path, df)
    return df


def fetch_stock_eod(client: Any, symbol: str, start: date, end: date) -> pd.DataFrame:
    params = {
        "symbol": str(symbol).upper(),
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
    }
    df = _theta_csv(client, "/v3/stock/history/eod", params, timeout=120.0)
    if df is None or df.empty:
        return pd.DataFrame()
    df.columns = [str(c).strip().lower() for c in df.columns]
    stamp_col = "created" if "created" in df.columns else df.columns[0]
    df["date"] = pd.to_datetime(df[stamp_col], errors="coerce").dt.date
    df["close"] = pd.to_numeric(df.get("close"), errors="coerce")
    df = df.dropna(subset=["date", "close"]).sort_values("date")
    return df[["date", "close"]].drop_duplicates("date")
