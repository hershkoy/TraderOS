"""Animesh / Dhan HA + 55 EMA channel + MACD swing (daily claim-check).

Source: Dhan masterclass https://www.youtube.com/watch?v=2YBmiyVmNNw (transcript
unavailable; encoded from the published summary).

Signals use Heikin-Ashi + EMA(55) of High/Low + MACD(12,26,9) on HA close.
Fills and stops use regular OHLC (HA is not a tradable price).

Long entry on a completed daily bar:
  HA close breaks above EMA(55) High (prior HA close was at/below that rail)
  HA candle is green (close > open) after a red HA bar (color flip)
  MACD line > 0
Fill is the next session open. Stop is the signal bar's EMA(55) Low (below the
band). Target is entry + rr * (entry - stop) with default rr=3.

Exits (first hit): hard stop, 3R target, Friday close, MACD line cross below
signal (next open). Same-bar stop beats target.

This is a research translation, not a live scanner.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from utils.research.costs import one_way_from_rt_bps
from utils.research.panel import adv_dollar, top_n_liquid_mask

DEFAULT_EMA_SPAN = 55
DEFAULT_MACD_FAST = 12
DEFAULT_MACD_SLOW = 26
DEFAULT_MACD_SIGNAL = 9
DEFAULT_RR = 3.0
DEFAULT_COST_BPS_RT = 0.0
DEFAULT_LIQUID_N = 500
DEFAULT_MIN_PRICE = 10.0
DEFAULT_MIN_ADV = 1_000_000.0

TRADE_COLUMNS: Tuple[str, ...] = (
    "symbol",
    "buy_date",
    "sell_date",
    "buy_price",
    "sell_price",
    "gain_pct",
    "exit_reason",
    "hold_bars",
    "entry_score",
    "stop_price",
    "target_price",
    "stop_pct",
    "signal_date",
    "ema_high",
    "ema_low",
    "macd",
    "macd_hist",
)


def _empty_trades() -> pd.DataFrame:
    return pd.DataFrame(columns=list(TRADE_COLUMNS))


def heikin_ashi_arrays(
    open_px: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Causal HA OHLC. Seed HA open at the first finite regular (O+C)/2."""
    o = np.asarray(open_px, dtype=float)
    h = np.asarray(high, dtype=float)
    lo = np.asarray(low, dtype=float)
    c = np.asarray(close, dtype=float)
    n = len(c)
    ha_c = (o + h + lo + c) / 4.0
    ha_o = np.full(n, np.nan, dtype=float)
    started = False
    prev_o = np.nan
    prev_c = np.nan
    for i in range(n):
        if not started:
            if np.isfinite(o[i]) and np.isfinite(c[i]):
                ha_o[i] = (o[i] + c[i]) / 2.0
                started = True
                prev_o = ha_o[i]
                prev_c = ha_c[i]
            continue
        ha_o[i] = (prev_o + prev_c) / 2.0
        if np.isfinite(ha_c[i]):
            prev_o = ha_o[i]
            prev_c = ha_c[i]
    ha_h = np.maximum(h, np.maximum(ha_o, ha_c))
    ha_l = np.minimum(lo, np.minimum(ha_o, ha_c))
    return ha_o, ha_h, ha_l, ha_c


def ema_array(x: np.ndarray, span: int) -> np.ndarray:
    """TradingView-style EMA (adjust=False), NaN until ``span`` finite seeds."""
    s = pd.Series(np.asarray(x, dtype=float))
    return s.ewm(span=int(span), adjust=False, min_periods=int(span)).mean().to_numpy(dtype=float)


def macd_arrays(
    x: np.ndarray,
    *,
    fast: int = DEFAULT_MACD_FAST,
    slow: int = DEFAULT_MACD_SLOW,
    signal: int = DEFAULT_MACD_SIGNAL,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    line = ema_array(x, fast) - ema_array(x, slow)
    sig = ema_array(line, signal)
    hist = line - sig
    return line, sig, hist


def resample_weekly_ohlc(
    open_px: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    volume: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """Friday-ending weekly bars from a daily wide panel."""
    rule = "W-FRI"
    out = {
        "open": open_px.resample(rule).first(),
        "high": high.resample(rule).max(),
        "low": low.resample(rule).min(),
        "close": close.resample(rule).last(),
        "volume": volume.resample(rule).sum(),
    }
    idx = out["close"].index
    for key in out:
        out[key] = out[key].reindex(idx)
    return out


def _is_friday(ts: pd.Timestamp) -> bool:
    return int(pd.Timestamp(ts).dayofweek) == 4


def collect_ha_ema_macd_trades(
    open_px: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    volume: pd.DataFrame,
    *,
    eval_start: str = "2018-01-01",
    eval_end: str = "2026-09-18",
    ema_span: int = DEFAULT_EMA_SPAN,
    macd_fast: int = DEFAULT_MACD_FAST,
    macd_slow: int = DEFAULT_MACD_SLOW,
    macd_signal: int = DEFAULT_MACD_SIGNAL,
    rr: float = DEFAULT_RR,
    friday_exit: bool = True,
    require_zero_cross: bool = False,
    require_color_flip: bool = True,
    require_prior_inside: bool = True,
    liquid_n: int = DEFAULT_LIQUID_N,
    min_price: float = DEFAULT_MIN_PRICE,
    min_adv: float = DEFAULT_MIN_ADV,
    cost_bps_rt: float = DEFAULT_COST_BPS_RT,
    adv_window: int = 30,
) -> pd.DataFrame:
    """Per-symbol sequential longs. Signal on completed HA close; fill next open."""
    close = close.astype(float).sort_index()
    open_px = open_px.astype(float).reindex_like(close)
    high = high.astype(float).reindex_like(close)
    low = low.astype(float).reindex_like(close)
    volume = volume.astype(float).reindex_like(close)
    liquid = top_n_liquid_mask(
        adv_dollar(close, volume, window=adv_window),
        top_n=liquid_n,
        min_adv=min_adv,
        close=close,
        min_price=min_price,
    ).reindex_like(close).fillna(False)
    one_way = one_way_from_rt_bps(cost_bps_rt)
    start = pd.Timestamp(eval_start)
    end = pd.Timestamp(eval_end)
    dates = close.index
    n = len(dates)
    if n < int(ema_span) + 2:
        return _empty_trades()

    rows: List[dict] = []
    rr = float(rr)

    for sym in close.columns:
        o = open_px[sym].to_numpy(dtype=float)
        h = high[sym].to_numpy(dtype=float)
        lo = low[sym].to_numpy(dtype=float)
        c = close[sym].to_numpy(dtype=float)
        liq = liquid[sym].to_numpy(dtype=bool)
        ha_o, _ha_h, ha_l, ha_c = heikin_ashi_arrays(o, h, lo, c)
        ema_h = ema_array(h, ema_span)
        ema_l = ema_array(lo, ema_span)
        macd, macd_sig, macd_hist = macd_arrays(
            ha_c, fast=macd_fast, slow=macd_slow, signal=macd_signal
        )

        in_pos = False
        pending_buy = False
        pending_sell = False
        pending_reason = "macd"
        entry_px = 0.0
        stop_px = 0.0
        target_px = 0.0
        entry_i = -1
        signal_i = -1
        entry_score = 0.0
        sig_ema_h = np.nan
        sig_ema_l = np.nan
        sig_macd = np.nan
        sig_hist = np.nan

        def _close_trade(exit_i: int, exit_px: float, reason: str) -> None:
            nonlocal in_pos
            if not in_pos:
                return
            buy_dt = dates[entry_i]
            if start <= buy_dt <= end and np.isfinite(exit_px) and exit_px > 0 and entry_px > 0:
                ret = (exit_px * (1.0 - one_way)) / (entry_px * (1.0 + one_way)) - 1.0
                rows.append(
                    {
                        "symbol": str(sym),
                        "buy_date": buy_dt,
                        "sell_date": dates[exit_i],
                        "buy_price": float(entry_px),
                        "sell_price": float(exit_px),
                        "gain_pct": float(ret * 100.0),
                        "exit_reason": reason,
                        "hold_bars": int(exit_i - entry_i),
                        "entry_score": float(entry_score),
                        "stop_price": float(stop_px),
                        "target_price": float(target_px),
                        "stop_pct": float((entry_px - stop_px) / entry_px * 100.0) if entry_px else np.nan,
                        "signal_date": dates[signal_i],
                        "ema_high": float(sig_ema_h),
                        "ema_low": float(sig_ema_l),
                        "macd": float(sig_macd),
                        "macd_hist": float(sig_hist),
                    }
                )
            in_pos = False

        for i in range(n):
            if pending_buy:
                pending_buy = False
                if np.isfinite(o[i]) and o[i] > 0 and np.isfinite(stop_px) and stop_px > 0:
                    fill = float(o[i])
                    if fill <= stop_px:
                        # Gap through the band stop: take the loss at the open.
                        in_pos = True
                        entry_px = fill
                        entry_i = i
                        _close_trade(i, fill, "gap_stop")
                        continue
                    in_pos = True
                    entry_px = fill
                    entry_i = i
                    target_px = entry_px + rr * (entry_px - stop_px)

            if pending_sell and in_pos:
                pending_sell = False
                if np.isfinite(o[i]) and o[i] > 0:
                    _close_trade(i, float(o[i]), pending_reason)
                    continue

            if in_pos:
                # Intrabar stop then target on regular OHLC.
                if np.isfinite(lo[i]) and lo[i] <= stop_px:
                    px = float(o[i]) if np.isfinite(o[i]) and o[i] < stop_px else float(stop_px)
                    _close_trade(i, px, "stop")
                    continue
                if np.isfinite(h[i]) and h[i] >= target_px:
                    px = float(o[i]) if np.isfinite(o[i]) and o[i] > target_px else float(target_px)
                    _close_trade(i, px, "target")
                    continue
                if friday_exit and _is_friday(dates[i]) and np.isfinite(c[i]) and c[i] > 0:
                    _close_trade(i, float(c[i]), "friday")
                    continue
                if (
                    i > 0
                    and np.isfinite(macd[i])
                    and np.isfinite(macd_sig[i])
                    and np.isfinite(macd[i - 1])
                    and np.isfinite(macd_sig[i - 1])
                    and macd[i] < macd_sig[i]
                    and macd[i - 1] >= macd_sig[i - 1]
                ):
                    pending_sell = True
                    pending_reason = "macd"
                continue

            if i >= n - 1:
                break
            if not bool(liq[i]):
                continue
            if not (
                np.isfinite(ha_c[i])
                and np.isfinite(ha_o[i])
                and np.isfinite(ema_h[i])
                and np.isfinite(ema_l[i])
                and np.isfinite(macd[i])
            ):
                continue
            if ha_c[i] <= ema_h[i]:
                continue
            if require_prior_inside:
                if i < 1 or not np.isfinite(ha_c[i - 1]) or not np.isfinite(ema_h[i - 1]):
                    continue
                if ha_c[i - 1] > ema_h[i - 1]:
                    continue
            if ha_c[i] <= ha_o[i]:
                continue
            if require_color_flip:
                if i < 1 or not np.isfinite(ha_c[i - 1]) or not np.isfinite(ha_o[i - 1]):
                    continue
                if ha_c[i - 1] > ha_o[i - 1]:
                    continue
            if macd[i] <= 0:
                continue
            if require_zero_cross:
                if i < 1 or not np.isfinite(macd[i - 1]) or macd[i - 1] > 0:
                    continue
            stop_cand = float(ema_l[i])
            if not np.isfinite(stop_cand) or stop_cand <= 0:
                continue
            # Next-open fill is unknown here; skip if the stop is already at/above HA close.
            if stop_cand >= float(ha_c[i]):
                continue

            pending_buy = True
            signal_i = i
            stop_px = stop_cand
            entry_score = float(macd_hist[i]) if np.isfinite(macd_hist[i]) else 0.0
            sig_ema_h = float(ema_h[i])
            sig_ema_l = float(ema_l[i])
            sig_macd = float(macd[i])
            sig_hist = float(macd_hist[i]) if np.isfinite(macd_hist[i]) else np.nan

        if in_pos:
            last_px = float(c[-1]) if np.isfinite(c[-1]) and c[-1] > 0 else entry_px
            _close_trade(n - 1, last_px, "eod")

    if not rows:
        return _empty_trades()
    out = pd.DataFrame(rows)
    out["buy_date"] = pd.to_datetime(out["buy_date"])
    out["sell_date"] = pd.to_datetime(out["sell_date"])
    out["signal_date"] = pd.to_datetime(out["signal_date"])
    return out.sort_values(["buy_date", "symbol"]).reset_index(drop=True)


def occupancy_equity(
    trades: pd.DataFrame,
    close: pd.DataFrame,
    *,
    max_open: int = 15,
    eval_start: Optional[str] = None,
    eval_end: Optional[str] = None,
) -> Tuple[pd.Series, pd.Series, str]:
    """Equal 1/max_open of equity, MTM on daily close, using occupancy-kept fills."""
    if trades is None or trades.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float), "empty"
    close = close.astype(float).sort_index()
    s = pd.Timestamp(eval_start) if eval_start else close.index[0]
    e = pd.Timestamp(eval_end) if eval_end else close.index[-1]
    calendar = close.index[(close.index >= s) & (close.index <= e)]
    if calendar.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float), "no calendar"

    t = trades.copy()
    t["buy_date"] = pd.to_datetime(t["buy_date"])
    t["sell_date"] = pd.to_datetime(t["sell_date"])
    by_buy: Dict[pd.Timestamp, List[int]] = {}
    for idx, row in t.iterrows():
        by_buy.setdefault(pd.Timestamp(row["buy_date"]).normalize(), []).append(int(idx))

    cash = 1.0
    positions: Dict[str, dict] = {}
    eq_pts: List[Tuple[pd.Timestamp, float]] = []
    inv_pts: List[Tuple[pd.Timestamp, float]] = []
    alloc = 1.0 / float(max(1, int(max_open)))

    def _mark() -> float:
        eq = cash
        for pos in positions.values():
            eq += pos["shares"] * pos["last_px"]
        return eq

    for dt in calendar:
        dtn = pd.Timestamp(dt).normalize()
        to_close = [sym for sym, pos in positions.items() if pos["sell_date"] < dtn]
        for sym in to_close:
            pos = positions.pop(sym)
            cash += pos["shares"] * float(pos["sell_px"])
        if dtn in by_buy:
            for idx in by_buy[dtn]:
                if len(positions) >= int(max_open):
                    break
                row = t.loc[idx]
                sym = str(row["symbol"])
                if sym in positions:
                    continue
                entry_px = float(row["buy_price"])
                if entry_px <= 0:
                    continue
                eq_now = _mark()
                spend = min(eq_now * alloc, cash)
                if spend <= 0:
                    continue
                shares = spend / entry_px
                cash -= spend
                last_px = entry_px
                if dt in close.index and sym in close.columns and pd.notna(close.loc[dt, sym]):
                    last_px = float(close.loc[dt, sym])
                positions[sym] = {
                    "shares": shares,
                    "last_px": last_px,
                    "sell_date": pd.Timestamp(row["sell_date"]).normalize(),
                    "sell_px": float(row["sell_price"]),
                }
        same_day = [sym for sym, pos in positions.items() if pos["sell_date"] <= dtn]
        for sym in same_day:
            pos = positions.pop(sym)
            cash += pos["shares"] * float(pos["sell_px"])
        for sym, pos in positions.items():
            if dt in close.index and sym in close.columns and pd.notna(close.loc[dt, sym]):
                pos["last_px"] = float(close.loc[dt, sym])
        eq_pts.append((dt, _mark()))
        inv_pts.append((dt, 1.0 if positions else 0.0))

    if positions:
        last_dt = eq_pts[-1][0]
        for pos in list(positions.values()):
            cash += pos["shares"] * float(pos["sell_px"])
        positions.clear()
        eq_pts[-1] = (last_dt, cash)

    eq = pd.Series({d: v for d, v in eq_pts}).sort_index()
    inv = pd.Series({d: v for d, v in inv_pts}).sort_index()
    if not eq.empty and float(eq.iloc[0]) != 0:
        eq = eq / float(eq.iloc[0])
    notes = "ha_ema_macd occupancy max_open=%s n=%s" % (max_open, len(t))
    return eq, inv, notes


def max_concurrent(trades: pd.DataFrame) -> int:
    if trades is None or trades.empty:
        return 0
    events: Dict[pd.Timestamp, int] = {}
    for _, row in trades.iterrows():
        b = pd.Timestamp(row["buy_date"]).normalize()
        s = pd.Timestamp(row["sell_date"]).normalize()
        events[b] = events.get(b, 0) + 1
        events[s] = events.get(s, 0) - 1
    cur = 0
    peak = 0
    for dt in sorted(events):
        cur += events[dt]
        if cur > peak:
            peak = cur
    return int(peak)
