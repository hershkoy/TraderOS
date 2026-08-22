"""Swing / multi-day long-only mean reversion on a daily panel."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from utils.research.costs import one_way_from_rt_bps
from utils.research.panel import adv_dollar, align_spy_regime, top_n_liquid_mask
from utils.research.signals import dump_signal, rsi_wide, sma_wide

VARIANT_DUMP_STOCK = "dump3_stock"
VARIANT_DUMP_SPY200 = "dump3_spy200"
VARIANT_RSI_STOCK = "rsi2_stock"
VARIANT_RSI_SPY200 = "rsi2_spy200"
ALL_VARIANTS = (
    VARIANT_DUMP_STOCK,
    VARIANT_DUMP_SPY200,
    VARIANT_RSI_STOCK,
    VARIANT_RSI_SPY200,
)


def _bool_row(df: pd.DataFrame, dt: pd.Timestamp) -> pd.Series:
    if dt not in df.index:
        return pd.Series(dtype=bool)
    return df.loc[dt].fillna(False).astype(bool)


def entry_mask(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    spy_close: pd.Series,
    variant: str,
    *,
    liquid_n: int = 500,
    min_price: float = 10.0,
    min_adv: float = 1_000_000.0,
    dump_n: int = 3,
    dump_thresh: float = -0.05,
    rsi_period: int = 2,
    rsi_thresh: float = 10.0,
    stock_sma: int = 20,
    spy_sma: int = 200,
    adv_window: int = 30,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Boolean Date x Symbol entry signals (as of close t; execute next open).

    dump*: 3-day dump AND close < stock SMA20 (revert up to SMA20).
    rsi*:  RSI oversold AND close > stock SMA20 (pullback in uptrend).
    *_spy200 adds SPY > SMA200.
    """
    close = close.astype(float)
    sma20 = sma_wide(close, stock_sma)
    liquid = top_n_liquid_mask(
        adv_dollar(close, volume, window=adv_window),
        top_n=liquid_n,
        min_adv=min_adv,
        close=close,
        min_price=min_price,
    )
    _, _, spy_on = align_spy_regime(spy_close, close.index, sma_period=spy_sma)

    below = close < sma20
    above = close > sma20
    if variant in (VARIANT_DUMP_STOCK, VARIANT_DUMP_SPY200):
        core = dump_signal(close, n=dump_n, thresh=dump_thresh) & below
        score = close.pct_change(dump_n, fill_method=None)
    elif variant in (VARIANT_RSI_STOCK, VARIANT_RSI_SPY200):
        rsi = rsi_wide(close, rsi_period)
        core = (rsi <= float(rsi_thresh)) & above
        score = rsi
    else:
        raise ValueError(f"Unknown variant: {variant}")

    mask = core.fillna(False) & liquid.reindex_like(core).fillna(False)
    if variant in (VARIANT_DUMP_SPY200, VARIANT_RSI_SPY200):
        spy_flag = spy_on.reindex(mask.index).fillna(False).astype(bool)
        mask = mask.mul(spy_flag, axis=0)
    mask = mask.fillna(False).astype(bool)
    return mask, score, spy_on


def simulate_swing_mr(
    open_px: pd.DataFrame,
    close: pd.DataFrame,
    low: pd.DataFrame,
    volume: pd.DataFrame,
    spy_close: pd.Series,
    *,
    variant: str,
    eval_start: str,
    eval_end: str,
    max_positions: int = 15,
    hold_days: int = 8,
    stop_loss_pct: float = 0.10,
    cost_bps_rt: float = 10.0,
    liquid_n: int = 500,
    min_price: float = 10.0,
    min_adv: float = 1_000_000.0,
    dump_n: int = 3,
    dump_thresh: float = -0.05,
    rsi_period: int = 2,
    rsi_thresh: float = 10.0,
    stock_sma: int = 20,
) -> Tuple[pd.Series, pd.Series, List[float], Dict[str, int], str]:
    """
    Next-open entry, equal 1/max_positions of equity, exits: SMA20 / time / stop.

    Returns (equity, invested, trade_rets, exit_counts, notes).
    """
    close = close.astype(float).sort_index()
    open_px = open_px.astype(float).reindex_like(close)
    low = low.astype(float).reindex_like(close)
    volume = volume.astype(float).reindex_like(close)

    mask, score, _spy_on = entry_mask(
        close,
        volume,
        spy_close,
        variant,
        liquid_n=liquid_n,
        min_price=min_price,
        min_adv=min_adv,
        dump_n=dump_n,
        dump_thresh=dump_thresh,
        rsi_period=rsi_period,
        rsi_thresh=rsi_thresh,
        stock_sma=stock_sma,
    )
    sma20 = sma_wide(close, stock_sma)
    # Execute next session after signal close
    exec_mask = mask.astype(bool).shift(1, fill_value=False)
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
    exits = {"sma20": 0, "time": 0, "stop": 0, "eod": 0}

    is_dump = variant in (VARIANT_DUMP_STOCK, VARIANT_DUMP_SPY200)

    for dt in calendar:
        to_close: List[Tuple[str, float, str]] = []
        for sym, pos in positions.items():
            if dt not in close.index or pd.isna(close.loc[dt, sym]):
                continue
            px_low = float(low.loc[dt, sym]) if pd.notna(low.loc[dt, sym]) else float(close.loc[dt, sym])
            px_close = float(close.loc[dt, sym])
            stop_px = pos["entry_px"] * (1.0 - stop_loss_pct)
            pos["bars"] = int(pos.get("bars", 0)) + 1
            ma_v = float(sma20.loc[dt, sym]) if dt in sma20.index and pd.notna(sma20.loc[dt, sym]) else None
            if px_low <= stop_px:
                to_close.append((sym, stop_px, "stop"))
            elif is_dump and ma_v is not None and px_close >= ma_v and pos["bars"] >= 1:
                to_close.append((sym, px_close, "sma20"))
            elif (not is_dump) and ma_v is not None and px_close < ma_v and pos["bars"] >= 1:
                to_close.append((sym, px_close, "sma20"))
            elif pos["bars"] >= int(hold_days):
                to_close.append((sym, px_close, "time"))

        for sym, px, reason in to_close:
            pos = positions.pop(sym)
            proceeds = pos["shares"] * px * (1.0 - one_way)
            cash += proceeds
            ret = (px * (1.0 - one_way)) / (pos["entry_px"] * (1.0 + one_way)) - 1.0
            trade_rets.append(float(ret))
            exits[reason] = exits.get(reason, 0) + 1

        # Mark-to-market before entries
        mtm = cash
        for sym, pos in positions.items():
            if dt in close.index and pd.notna(close.loc[dt, sym]):
                pos["last_px"] = float(close.loc[dt, sym])
            mtm += pos["shares"] * pos["last_px"]

        if dt in exec_mask.index:
            row = _bool_row(exec_mask, dt)
            cands = [c for c in row.index[row].tolist() if c not in positions]
            if cands and dt in exec_score.index:
                sc = exec_score.loc[dt, cands]
                sc = sc.astype(float)
                cands = sc.sort_values(ascending=True).index.tolist()
            for sym in cands:
                if len(positions) >= max_positions:
                    break
                if dt not in open_px.index or pd.isna(open_px.loc[dt, sym]):
                    continue
                entry_px = float(open_px.loc[dt, sym])
                if entry_px <= 0:
                    continue
                eq_now = cash
                for s2, p2 in positions.items():
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
                    "entry_date": dt,
                    "last_px": entry_px,
                    "bars": 0,
                }

        eq = cash
        for sym, pos in positions.items():
            eq += pos["shares"] * pos["last_px"]
        equity_points.append((dt, eq))
        invested_points.append((dt, 1.0 if positions else 0.0))

    if positions and equity_points:
        last_dt = equity_points[-1][0]
        for sym, pos in list(positions.items()):
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
        f"variant={variant} max_pos={max_positions} hold={hold_days}d stop={stop_loss_pct:.0%} "
        f"cost={cost_bps_rt:.0f}bps RT liquid={liquid_n} min_px={min_price:g}; "
        f"exits={exits}"
    )
    return eq_s, inv_s, trade_rets, exits, notes
