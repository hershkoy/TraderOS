"""Bollinger mean-reversion matching ``strategies/mean_reversion_strategy.py``.

Signal on completed daily close; Backtrader default fill is the next session open.
Buy when close <= lower band; exit when close >= SMA (upper-band exit is redundant
because SMA is always at or below the upper band).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from utils.research.costs import one_way_from_rt_bps
from utils.research.panel import adv_dollar, top_n_liquid_mask

DEFAULT_PERIOD = 20
DEFAULT_DEVFACTOR = 2.0
# Strategy broker commission is 0.001 per side = 20 bps round-trip.
DEFAULT_COST_BPS_RT = 20.0


def bollinger_bands(
    close: pd.DataFrame,
    period: int = DEFAULT_PERIOD,
    devfactor: float = DEFAULT_DEVFACTOR,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """SMA, upper, lower. Std uses ddof=0 (population), matching typical TA/Backtrader."""
    c = close.astype(float)
    sma = c.rolling(int(period), min_periods=int(period)).mean()
    std = c.rolling(int(period), min_periods=int(period)).std(ddof=0)
    upper = sma + std * float(devfactor)
    lower = sma - std * float(devfactor)
    return sma, upper, lower


def collect_bb_trades(
    open_px: pd.DataFrame,
    close: pd.DataFrame,
    volume: pd.DataFrame,
    *,
    period: int = DEFAULT_PERIOD,
    devfactor: float = DEFAULT_DEVFACTOR,
    eval_start: str = "2022-01-01",
    eval_end: str = "2022-12-31",
    liquid_n: int = 500,
    min_price: float = 10.0,
    min_adv: float = 1_000_000.0,
    cost_bps_rt: float = DEFAULT_COST_BPS_RT,
    stop_loss_pct: float = 0.0,
    adv_window: int = 30,
) -> pd.DataFrame:
    """
    Per-symbol sequential trades (one position at a time). Next-open fills.

    ``gain_pct`` is in percent (channel-touch convention). ``stop_loss_pct`` 0
    means no stop (faithful strategy). A stop uses the fill-day low vs entry.
    """
    close = close.astype(float).sort_index()
    open_px = open_px.astype(float).reindex_like(close)
    volume = volume.astype(float).reindex_like(close)
    sma, _upper, lower = bollinger_bands(close, period, devfactor)
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
    rows: List[dict] = []

    for sym in close.columns:
        c = close[sym].to_numpy(dtype=float)
        o = open_px[sym].to_numpy(dtype=float)
        ma = sma[sym].to_numpy(dtype=float)
        lo = lower[sym].to_numpy(dtype=float)
        liq = liquid[sym].to_numpy(dtype=bool)
        in_pos = False
        pending_buy = False
        pending_sell = False
        entry_px = 0.0
        entry_i = -1
        reason = "sma"

        for i in range(n):
            if pending_buy:
                if np.isfinite(o[i]) and o[i] > 0:
                    in_pos = True
                    entry_px = float(o[i])
                    entry_i = i
                    reason = "sma"
                pending_buy = False
            if pending_sell and in_pos:
                if np.isfinite(o[i]) and o[i] > 0:
                    exit_px = float(o[i])
                    buy_dt = dates[entry_i]
                    sell_dt = dates[i]
                    if start <= buy_dt <= end:
                        ret = (exit_px * (1.0 - one_way)) / (entry_px * (1.0 + one_way)) - 1.0
                        rows.append(
                            {
                                "symbol": str(sym),
                                "buy_date": buy_dt,
                                "sell_date": sell_dt,
                                "buy_price": entry_px,
                                "sell_price": exit_px,
                                "gain_pct": ret * 100.0,
                                "exit_reason": reason,
                                "hold_bars": int(i - entry_i),
                            }
                        )
                    in_pos = False
                pending_sell = False

            if stop_loss_pct > 0 and in_pos and not pending_sell:
                # Use close as a conservative same-bar stop proxy when low is absent.
                stop_px = entry_px * (1.0 - stop_loss_pct)
                if np.isfinite(c[i]) and c[i] <= stop_px:
                    buy_dt = dates[entry_i]
                    sell_dt = dates[i]
                    if start <= buy_dt <= end:
                        ret = (stop_px * (1.0 - one_way)) / (entry_px * (1.0 + one_way)) - 1.0
                        rows.append(
                            {
                                "symbol": str(sym),
                                "buy_date": buy_dt,
                                "sell_date": sell_dt,
                                "buy_price": entry_px,
                                "sell_price": stop_px,
                                "gain_pct": ret * 100.0,
                                "exit_reason": "stop",
                                "hold_bars": int(i - entry_i),
                            }
                        )
                    in_pos = False
                    continue

            if i >= n - 1:
                break
            if (not in_pos) and (not pending_buy):
                if (
                    np.isfinite(c[i])
                    and np.isfinite(lo[i])
                    and bool(liq[i])
                    and c[i] <= lo[i]
                ):
                    pending_buy = True
            elif in_pos and (not pending_sell):
                if np.isfinite(c[i]) and np.isfinite(ma[i]) and c[i] >= ma[i]:
                    pending_sell = True
                    reason = "sma"

        if in_pos:
            exit_px = float(c[-1]) if np.isfinite(c[-1]) else entry_px
            buy_dt = dates[entry_i]
            sell_dt = dates[-1]
            if start <= buy_dt <= end:
                ret = (exit_px * (1.0 - one_way)) / (entry_px * (1.0 + one_way)) - 1.0
                rows.append(
                    {
                        "symbol": str(sym),
                        "buy_date": buy_dt,
                        "sell_date": sell_dt,
                        "buy_price": entry_px,
                        "sell_price": exit_px,
                        "gain_pct": ret * 100.0,
                        "exit_reason": "eod",
                        "hold_bars": int(n - 1 - entry_i),
                    }
                )

    if not rows:
        return pd.DataFrame(
            columns=[
                "symbol",
                "buy_date",
                "sell_date",
                "buy_price",
                "sell_price",
                "gain_pct",
                "exit_reason",
                "hold_bars",
            ]
        )
    out = pd.DataFrame(rows)
    out["buy_date"] = pd.to_datetime(out["buy_date"])
    out["sell_date"] = pd.to_datetime(out["sell_date"])
    return out.sort_values(["buy_date", "symbol"]).reset_index(drop=True)


def simulate_bb_occupancy(
    open_px: pd.DataFrame,
    close: pd.DataFrame,
    volume: pd.DataFrame,
    *,
    period: int = DEFAULT_PERIOD,
    devfactor: float = DEFAULT_DEVFACTOR,
    eval_start: str = "2022-01-01",
    eval_end: str = "2022-12-31",
    max_positions: int = 15,
    liquid_n: int = 500,
    min_price: float = 10.0,
    min_adv: float = 1_000_000.0,
    cost_bps_rt: float = DEFAULT_COST_BPS_RT,
    stop_loss_pct: float = 0.0,
    adv_window: int = 30,
) -> Tuple[pd.Series, pd.Series, List[float], Dict[str, int], str]:
    """Equal 1/max_positions of equity. Next-open entry after close <= lower BB."""
    close = close.astype(float).sort_index()
    open_px = open_px.astype(float).reindex_like(close)
    volume = volume.astype(float).reindex_like(close)
    sma, _upper, lower = bollinger_bands(close, period, devfactor)
    liquid = top_n_liquid_mask(
        adv_dollar(close, volume, window=adv_window),
        top_n=liquid_n,
        min_adv=min_adv,
        close=close,
        min_price=min_price,
    ).reindex_like(close).fillna(False)

    signal = (close <= lower) & liquid & close.notna() & lower.notna()
    score = (lower - close) / close.replace(0, np.nan)
    exec_mask = signal.astype(bool).shift(1, fill_value=False)
    exec_score = score.shift(1)

    s = pd.Timestamp(eval_start)
    e = pd.Timestamp(eval_end)
    calendar = close.index[(close.index >= s) & (close.index <= e)]
    one_way = one_way_from_rt_bps(cost_bps_rt)
    alloc_frac = 1.0 / float(max(1, max_positions))

    cash = 1.0
    positions: Dict[str, dict] = {}
    equity_points: List[Tuple[pd.Timestamp, float]] = []
    invested_points: List[Tuple[pd.Timestamp, float]] = []
    trade_rets: List[float] = []
    exits = {"sma": 0, "stop": 0, "eod": 0}

    for dt in calendar:
        to_close: List[Tuple[str, float, str]] = []
        for sym, pos in positions.items():
            if dt not in close.index or pd.isna(close.loc[dt, sym]):
                continue
            px_close = float(close.loc[dt, sym])
            pos["bars"] = int(pos.get("bars", 0)) + 1
            ma_v = float(sma.loc[dt, sym]) if dt in sma.index and pd.notna(sma.loc[dt, sym]) else None
            if stop_loss_pct > 0:
                stop_px = pos["entry_px"] * (1.0 - stop_loss_pct)
                if px_close <= stop_px:
                    to_close.append((sym, stop_px, "stop"))
                    continue
            if ma_v is not None and px_close >= ma_v and pos["bars"] >= 1:
                to_close.append((sym, px_close, "sma"))

        for sym, px, reason in to_close:
            pos = positions.pop(sym)
            cash += pos["shares"] * px * (1.0 - one_way)
            ret = (px * (1.0 - one_way)) / (pos["entry_px"] * (1.0 + one_way)) - 1.0
            trade_rets.append(float(ret))
            exits[reason] = exits.get(reason, 0) + 1

        for sym, pos in positions.items():
            if dt in close.index and pd.notna(close.loc[dt, sym]):
                pos["last_px"] = float(close.loc[dt, sym])

        if dt in exec_mask.index:
            row = exec_mask.loc[dt].fillna(False).astype(bool)
            cands = [c for c in row.index[row].tolist() if c not in positions]
            if cands and dt in exec_score.index:
                sc = exec_score.loc[dt, cands].astype(float)
                cands = sc.sort_values(ascending=False).index.tolist()
            for sym in cands:
                if len(positions) >= max_positions:
                    break
                if dt not in open_px.index or pd.isna(open_px.loc[dt, sym]):
                    continue
                entry_px = float(open_px.loc[dt, sym])
                if entry_px <= 0:
                    continue
                eq_now = cash
                for p2 in positions.values():
                    eq_now += p2["shares"] * p2["last_px"]
                spend = min(eq_now * alloc_frac, cash)
                if spend <= 0 or cash < spend * 0.5:
                    continue
                gross_px = entry_px * (1.0 + one_way)
                shares = spend / gross_px
                cash -= spend
                positions[sym] = {
                    "entry_px": entry_px,
                    "shares": shares,
                    "last_px": entry_px,
                    "bars": 0,
                }

        eq = cash
        for pos in positions.values():
            eq += pos["shares"] * pos["last_px"]
        equity_points.append((dt, eq))
        invested_points.append((dt, 1.0 if positions else 0.0))

    if positions and equity_points:
        last_dt = equity_points[-1][0]
        for pos in list(positions.values()):
            px = pos["last_px"]
            cash += pos["shares"] * px * (1.0 - one_way)
            ret = (px * (1.0 - one_way)) / (pos["entry_px"] * (1.0 + one_way)) - 1.0
            trade_rets.append(float(ret))
            exits["eod"] = exits.get("eod", 0) + 1
        positions.clear()
        equity_points[-1] = (last_dt, cash)

    eq_s = pd.Series({d: v for d, v in equity_points}).sort_index()
    inv_s = pd.Series({d: v for d, v in invested_points}).sort_index()
    if not eq_s.empty and float(eq_s.iloc[0]) != 0:
        eq_s = eq_s / float(eq_s.iloc[0])
    notes = (
        f"bb period={period} k={devfactor} max_pos={max_positions} "
        f"stop={stop_loss_pct:.0%} cost={cost_bps_rt:.0f}bps RT liquid={liquid_n}; "
        f"exits={exits}"
    )
    return eq_s, inv_s, trade_rets, exits, notes
