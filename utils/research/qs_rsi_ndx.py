"""Quantified Strategies RSI + IBS mean-reversion on Nasdaq-100 stocks (claim-check).

Video / Substack claim (2026-09):
  Entry (signal on close, fill next open):
    close > SMA200
    RSI(3) < 20
    IBS < 0.3
  Exit (signal on close, fill next open):
    close > prior day's high  (QS exit)
  Portfolio: max 5 open positions (equal-dollar slots).

Claimed (NDX stocks, no survivorship bias / Norgate):
  n=1321, E=+1.1%, WR=68%, PF=2.0, CAGR~11%, exposure~16%, MDD=17%.

Source: YouTube tjL4wcI403E / Substack "RSI Trading Strategy for Nasdaq Stocks".
Rules taken from the public video summary (article rules are paywalled).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from utils.research.choppy_regime import occupancy_ranked, rsi_wilder_wide
from utils.research.qs_five_swing import ibs as ibs_series

YEAR_BUCKETS: Sequence[Tuple[str, str, str]] = (
    ("2018-2019", "2018-01-01", "2019-12-31"),
    ("2020-2021", "2020-01-01", "2021-12-31"),
    ("2022-2023", "2022-01-01", "2023-12-31"),
    ("2024-2026", "2024-01-01", "2026-12-31"),
)

CLAIMED: Dict[str, Any] = {
    "n": 1321,
    "expectancy_pct": 1.1,
    "avg_win_pct": 3.2,
    "avg_loss_pct": 3.4,
    "win_rate_pct": 68.0,
    "profit_factor": 2.0,
    "cagr_pct": 11.0,
    "exposure_pct": 16.0,
    "mdd_pct": 17.0,
    "max_open": 5,
    "source": "QS Substack 2026-09-14 / YouTube tjL4wcI403E",
}

TRADE_COLUMNS = (
    "symbol",
    "buy_date",
    "sell_date",
    "buy_price",
    "sell_price",
    "gain_pct",
    "exit_reason",
    "hold_bars",
    "entry_score",
    "rsi3",
    "ibs",
)


def _empty_trades() -> pd.DataFrame:
    return pd.DataFrame(columns=list(TRADE_COLUMNS))


def collect_symbol_trades(
    df: pd.DataFrame,
    *,
    symbol: str,
    rsi_period: int = 3,
    rsi_entry: float = 20.0,
    ibs_max: float = 0.3,
    sma_trend: int = 200,
    friction_pct: float = 0.0,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Walk one symbol: signal on close, fill next open; one position at a time."""
    if df is None or df.empty:
        return _empty_trades()

    d = df.copy()
    if "ts" not in d.columns:
        if isinstance(d.index, pd.DatetimeIndex):
            d = d.reset_index()
            if "ts" not in d.columns:
                d = d.rename(columns={d.columns[0]: "ts"})
        else:
            return _empty_trades()

    d["ts"] = pd.to_datetime(d["ts"], utc=True, errors="coerce")
    for col in ("open", "high", "low", "close"):
        if col not in d.columns:
            return _empty_trades()
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d = (
        d.dropna(subset=["ts", "open", "high", "low", "close"])
        .sort_values("ts")
        .drop_duplicates("ts", keep="last")
        .reset_index(drop=True)
    )
    min_bars = int(sma_trend) + int(rsi_period) + 5
    if len(d) < min_bars:
        return _empty_trades()

    close = d["close"].astype(float)
    high = d["high"].astype(float)
    low = d["low"].astype(float)
    open_ = d["open"].astype(float)
    ts = pd.to_datetime(d["ts"], utc=True, errors="coerce")
    dates = pd.DatetimeIndex(ts.dt.tz_localize(None).dt.normalize())

    sma = close.rolling(int(sma_trend), min_periods=int(sma_trend)).mean()
    # Wide helper expects DataFrame; reuse series path via 1-col frame.
    rsi = rsi_wilder_wide(close.to_frame(name=symbol), int(rsi_period))[symbol]
    ibs = ibs_series(high, low, close)

    start_ts = pd.Timestamp(start) if start else dates[0]
    end_ts = pd.Timestamp(end) if end else dates[-1]

    fric = float(friction_pct) / 100.0
    rows: List[Dict[str, Any]] = []
    in_pos = False
    pending_entry = False
    pending_exit = False
    entry_i: Optional[int] = None
    entry_px: Optional[float] = None
    entry_rsi = np.nan
    entry_ibs = np.nan
    n = len(d)

    for i in range(n):
        dt = dates[i]

        if pending_entry and not in_pos:
            px = float(open_.iloc[i])
            if np.isfinite(px) and px > 0:
                in_pos = True
                entry_i = i
                entry_px = px * (1.0 + fric)
            pending_entry = False

        if pending_exit and in_pos and entry_i is not None and entry_px is not None:
            px = float(open_.iloc[i])
            if np.isfinite(px) and px > 0 and i > entry_i:
                sell_px = px * (1.0 - fric)
                rows.append(
                    {
                        "symbol": str(symbol).upper(),
                        "buy_date": dates[entry_i],
                        "sell_date": dt,
                        "buy_price": float(entry_px),
                        "sell_price": float(sell_px),
                        "gain_pct": (sell_px / entry_px - 1.0) * 100.0,
                        "exit_reason": "qs_exit",
                        "hold_bars": int(i - entry_i),
                        "entry_score": float(entry_rsi) if np.isfinite(entry_rsi) else np.nan,
                        "rsi3": float(entry_rsi) if np.isfinite(entry_rsi) else np.nan,
                        "ibs": float(entry_ibs) if np.isfinite(entry_ibs) else np.nan,
                    }
                )
                in_pos = False
                entry_i = None
                entry_px = None
            pending_exit = False

        if i >= n - 1:
            continue

        # Exit signal on completed close while long.
        if in_pos and not pending_exit and entry_i is not None and i > entry_i:
            prior_high = float(high.iloc[i - 1])
            c = float(close.iloc[i])
            if np.isfinite(c) and np.isfinite(prior_high) and c > prior_high:
                pending_exit = True

        # Entry signal only when flat and not already pending.
        if in_pos or pending_entry or pending_exit:
            continue
        if dt < start_ts or dt > end_ts:
            continue

        c = float(close.iloc[i])
        ma = float(sma.iloc[i])
        r = float(rsi.iloc[i])
        b = float(ibs.iloc[i])
        if not (
            np.isfinite(c)
            and np.isfinite(ma)
            and np.isfinite(r)
            and np.isfinite(b)
            and c > ma
            and r < float(rsi_entry)
            and b < float(ibs_max)
        ):
            continue
        pending_entry = True
        entry_rsi = r
        entry_ibs = b

    if not rows:
        return _empty_trades()
    return pd.DataFrame(rows)


def collect_panel_trades(
    frames: Dict[str, pd.DataFrame],
    *,
    rsi_period: int = 3,
    rsi_entry: float = 20.0,
    ibs_max: float = 0.3,
    sma_trend: int = 200,
    friction_pct: float = 0.0,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Per-symbol unconstrained trades (no portfolio cap)."""
    parts: List[pd.DataFrame] = []
    for sym, df in frames.items():
        t = collect_symbol_trades(
            df,
            symbol=str(sym),
            rsi_period=rsi_period,
            rsi_entry=rsi_entry,
            ibs_max=ibs_max,
            sma_trend=sma_trend,
            friction_pct=friction_pct,
            start=start,
            end=end,
        )
        if not t.empty:
            parts.append(t)
    if not parts:
        return _empty_trades()
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values(["buy_date", "symbol"]).reset_index(drop=True)


def book_stats(trades: pd.DataFrame) -> Dict[str, Any]:
    """Pooled trade-list metrics (n / WR / E / PF / avg win-loss)."""
    if trades is None or trades.empty:
        return {
            "n": 0,
            "win_rate_pct": None,
            "expectancy_pct": None,
            "profit_factor": None,
            "avg_win_pct": None,
            "avg_loss_pct": None,
            "median_gain_pct": None,
            "geo_mean_year_pf": None,
        }
    g = pd.to_numeric(trades["gain_pct"], errors="coerce").dropna()
    n = int(len(g))
    if n == 0:
        return {
            "n": 0,
            "win_rate_pct": None,
            "expectancy_pct": None,
            "profit_factor": None,
            "avg_win_pct": None,
            "avg_loss_pct": None,
            "median_gain_pct": None,
            "geo_mean_year_pf": None,
        }
    wins = g[g > 0]
    losses = g[g < 0]
    gross_w = float(wins.sum()) if len(wins) else 0.0
    gross_l = float((-losses).sum()) if len(losses) else 0.0
    pf = (gross_w / gross_l) if gross_l > 1e-12 else (float("inf") if gross_w > 0 else 0.0)
    by_year = year_bucket_stats(trades)
    geo = geom_mean_year_pf(by_year)
    return {
        "n": n,
        "win_rate_pct": round(100.0 * float((g > 0).mean()), 2),
        "expectancy_pct": round(float(g.mean()), 3),
        "profit_factor": None if not np.isfinite(pf) else round(float(pf), 3),
        "avg_win_pct": None if wins.empty else round(float(wins.mean()), 3),
        "avg_loss_pct": None if losses.empty else round(float((-losses).mean()), 3),
        "median_gain_pct": round(float(g.median()), 3),
        "geo_mean_year_pf": None if geo is None else round(float(geo), 3),
        "by_year": by_year,
    }


def year_bucket_stats(trades: pd.DataFrame) -> List[Dict[str, Any]]:
    if trades is None or trades.empty:
        return []
    buy = pd.to_datetime(trades["buy_date"])
    out: List[Dict[str, Any]] = []
    for label, s, e in YEAR_BUCKETS:
        m = (buy >= pd.Timestamp(s)) & (buy <= pd.Timestamp(e))
        sub = trades.loc[m]
        g = pd.to_numeric(sub["gain_pct"], errors="coerce").dropna()
        n = int(len(g))
        if n == 0:
            out.append(
                {
                    "bucket": label,
                    "n": 0,
                    "expectancy_pct": None,
                    "win_rate_pct": None,
                    "profit_factor": None,
                }
            )
            continue
        wins = g[g > 0]
        losses = g[g < 0]
        gw = float(wins.sum()) if len(wins) else 0.0
        gl = float((-losses).sum()) if len(losses) else 0.0
        pf = (gw / gl) if gl > 1e-12 else (float("inf") if gw > 0 else 0.0)
        out.append(
            {
                "bucket": label,
                "n": n,
                "expectancy_pct": round(float(g.mean()), 3),
                "win_rate_pct": round(100.0 * float((g > 0).mean()), 2),
                "profit_factor": None if not np.isfinite(pf) else round(float(pf), 3),
            }
        )
    return out


def geom_mean_year_pf(
    by_year: Sequence[Dict[str, Any]],
    *,
    min_n: int = 30,
) -> Optional[float]:
    pfs: List[float] = []
    for row in by_year:
        n = int(row.get("n") or 0)
        pf = row.get("profit_factor")
        if n < int(min_n) or pf is None:
            continue
        try:
            v = float(pf)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(v) or v <= 0:
            continue
        pfs.append(v)
    if not pfs:
        return None
    return float(np.exp(np.mean(np.log(pfs))))


def portfolio_equity(
    trades: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    close_panel: pd.DataFrame,
    *,
    max_open: int = 5,
) -> Dict[str, Any]:
    """Equal-slot daily equity from occupancy trades (cash earns 0).

    Each new name deploys 1/max_open of equity at the buy open; mark-to-market
    on close. Exposure is mean(open_slots / max_open), matching QS "time invested".
    """
    cal = pd.DatetimeIndex(pd.to_datetime(calendar)).tz_localize(None).normalize().unique().sort_values()
    if trades is None or trades.empty or len(cal) == 0:
        eq = pd.Series(1.0, index=cal, dtype=float)
        inv = pd.Series(0.0, index=cal, dtype=float)
        return {
            "equity": eq,
            "invested_frac": inv,
            "cagr_pct": 0.0,
            "mdd_pct": 0.0,
            "exposure_pct": 0.0,
            "sharpe": None,
        }

    close = close_panel.copy()
    if not isinstance(close.index, pd.DatetimeIndex):
        close.index = pd.to_datetime(close.index)
    close.index = pd.DatetimeIndex(close.index).tz_localize(None).normalize()
    close = close[~close.index.duplicated(keep="last")].sort_index()
    close.columns = [str(c).upper() for c in close.columns]

    work = trades.copy()
    work["symbol"] = work["symbol"].astype(str).str.upper()
    work["buy_date"] = pd.to_datetime(work["buy_date"]).dt.tz_localize(None).dt.normalize()
    work["sell_date"] = pd.to_datetime(work["sell_date"]).dt.tz_localize(None).dt.normalize()
    work["buy_price"] = pd.to_numeric(work["buy_price"], errors="coerce")
    work["sell_price"] = pd.to_numeric(work["sell_price"], errors="coerce")

    cap = max(1, int(max_open))
    slot_w = 1.0 / float(cap)

    buys_by_day: Dict[pd.Timestamp, List[Any]] = {}
    sells_by_day: Dict[pd.Timestamp, List[Any]] = {}
    for _, row in work.iterrows():
        buys_by_day.setdefault(row["buy_date"], []).append(row)
        sells_by_day.setdefault(row["sell_date"], []).append(row)

    open_pos: List[Dict[str, Any]] = []
    eq_vals: List[float] = []
    inv_vals: List[float] = []
    cash = 1.0
    equity = 1.0

    for dt in cal:
        if dt in sells_by_day:
            remain: List[Dict[str, Any]] = []
            sold_syms = {str(r["symbol"]) for r in sells_by_day[dt]}
            for pos in open_pos:
                if pos["symbol"] not in sold_syms:
                    remain.append(pos)
                    continue
                row = next(r for r in sells_by_day[dt] if r["symbol"] == pos["symbol"])
                sell_px = float(row["sell_price"])
                shares = float(pos["shares"])
                if np.isfinite(sell_px) and shares > 0:
                    cash += shares * sell_px
            open_pos = remain

        if dt in buys_by_day and len(open_pos) < cap:
            mtm = cash
            for pos in open_pos:
                px = _close_on(close, pos["symbol"], dt)
                if px is None:
                    px = float(pos["entry_px"])
                mtm += float(pos["shares"]) * px
            equity_now = mtm if mtm > 0 else cash
            ranked = sorted(
                buys_by_day[dt],
                key=lambda r: (float(r["entry_score"]) if pd.notna(r.get("entry_score")) else 999.0, r["symbol"]),
            )
            for row in ranked:
                if len(open_pos) >= cap:
                    break
                if any(p["symbol"] == row["symbol"] for p in open_pos):
                    continue
                buy_px = float(row["buy_price"])
                if not np.isfinite(buy_px) or buy_px <= 0:
                    continue
                alloc_cash = min(cash, equity_now * slot_w)
                if alloc_cash <= 0:
                    continue
                shares = alloc_cash / buy_px
                cash -= alloc_cash
                open_pos.append(
                    {
                        "symbol": str(row["symbol"]),
                        "entry_px": buy_px,
                        "shares": shares,
                    }
                )

        mtm = cash
        for pos in open_pos:
            px = _close_on(close, pos["symbol"], dt)
            if px is None:
                px = float(pos["entry_px"])
            mtm += float(pos["shares"]) * px
        equity = mtm if mtm > 0 else equity
        eq_vals.append(float(equity))
        inv_vals.append(float(len(open_pos)) / float(cap))

    eq = pd.Series(eq_vals, index=cal, dtype=float)
    inv = pd.Series(inv_vals, index=cal, dtype=float)
    return _equity_summary(eq, inv)


def _close_on(close: pd.DataFrame, symbol: str, dt: pd.Timestamp) -> Optional[float]:
    sym = str(symbol).upper()
    if sym not in close.columns:
        return None
    if dt not in close.index:
        # nearest prior close
        prior = close.index[close.index <= dt]
        if len(prior) == 0:
            return None
        dt = prior[-1]
    v = close.at[dt, sym]
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(fv) or fv <= 0:
        return None
    return fv


def _equity_summary(eq: pd.Series, inv: pd.Series) -> Dict[str, Any]:
    if eq is None or eq.empty:
        return {
            "equity": eq,
            "invested_frac": inv,
            "cagr_pct": 0.0,
            "mdd_pct": 0.0,
            "exposure_pct": 0.0,
            "sharpe": None,
        }
    peak = eq.cummax()
    dd = eq / peak - 1.0
    mdd = float(dd.min()) if len(dd) else 0.0
    n_days = max(1, int(len(eq)))
    years = n_days / 252.0
    final = float(eq.iloc[-1])
    cagr = (final ** (1.0 / years) - 1.0) if years > 0 and final > 0 else 0.0
    rets = eq.pct_change().fillna(0.0)
    vol = float(rets.std()) * np.sqrt(252.0) if len(rets) > 2 else 0.0
    sharpe = (cagr / vol) if vol > 1e-12 else None
    return {
        "equity": eq,
        "invested_frac": inv,
        "cagr_pct": round(100.0 * float(cagr), 2),
        "mdd_pct": round(100.0 * abs(float(mdd)), 2),
        "exposure_pct": round(100.0 * float(inv.mean()), 2),
        "sharpe": None if sharpe is None else round(float(sharpe), 3),
        "final_equity": round(float(final), 4),
    }
