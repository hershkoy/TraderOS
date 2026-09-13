"""Choppy-regime mean-reversion rules from docs/strategies/choppy_regime/ideas.md.

Strategy 1 -- RSI-2 MOC (2026-09-13 refine)
  Default trend filter is SMA50 (SMA200 sat out 2022). Wilder RSI(2) < 5.
  VIX>30 is not a sit-out; prior-session VIX scales size (ref 20, floor 0.25x).
  Optional sma_trend=0 drops the MA and pairs with a more extreme RSI
  (research default for that book: RSI<2). CPI-eve skip stays. Scan is 15:45 ET
  on a developing daily close from IB 15m (last RTH bar, prefer 15:45), not
  Alpaca 17:00 EOD. Research fill is that session's official close (MOC proxy).
  Exit: close > SMA5 or RSI(2) >= 70, else time stop 8 sessions.

Strategy 2 -- 3-day pullback limit
  Setup on a completed close: close > SMA100, exactly three consecutive lower
  lows, close < SMA5. Next session GTC limit = close - 0.5 * ATR(10). Fill at
  the limit, or at the open if the open gaps through. Miss cancels (one day).
  First later close higher than the prior close sells the next morning open.
  Default time stop 6 sessions (next open) so falling knives cannot sit forever.
  Optional min_close_loc requires the setup bar close in the upper fraction of
  that day's range (intraday rejection; 0 disables).

RSI cross-back (research overlay, not the threshold default): arm when RSI(2)
drops below rsi_entry, buy the later close where RSI(2) reclaims rsi_cross_back
(5 or 10). Same-bar RSI cannot both arm and fire. Cancel if close loses SMA or
the arm exceeds max_arm_bars.

Phase 5 rsi2_stock / dump3_stock are different books. Do not mix first-run
(SMA200 / VIX skip / 0.25%), refine (SMA50 / no skip / 0.10%), or confirm
(cross-back / close-loc) numbers.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.research.costs import one_way_from_rt_bps
from utils.research.panel import adv_dollar, top_n_liquid_mask

# BLS CPI news-release dates (08:30 ET). ALFRED duplicates two days apart are
# revision stamps; keep the later date in those pairs. 2025-10-24 / 2025-12-18
# are the delayed shutdown prints (no Nov-2025 slot).
CPI_RELEASE_DATES: Tuple[str, ...] = (
    "2018-01-12", "2018-02-14", "2018-03-13", "2018-04-11", "2018-05-10",
    "2018-06-12", "2018-07-12", "2018-08-10", "2018-09-13", "2018-10-11",
    "2018-11-14", "2018-12-12",
    "2019-01-11", "2019-02-13", "2019-03-12", "2019-04-10", "2019-05-10",
    "2019-06-12", "2019-07-11", "2019-08-13", "2019-09-12", "2019-10-10",
    "2019-11-13", "2019-12-11",
    "2020-01-14", "2020-02-13", "2020-03-11", "2020-04-10", "2020-05-12",
    "2020-06-10", "2020-07-14", "2020-08-12", "2020-09-11", "2020-10-13",
    "2020-11-12", "2020-12-10",
    "2021-01-13", "2021-02-10", "2021-03-10", "2021-04-13", "2021-05-12",
    "2021-06-10", "2021-07-13", "2021-08-11", "2021-09-14", "2021-10-13",
    "2021-11-10", "2021-12-10",
    "2022-01-12", "2022-02-10", "2022-03-10", "2022-04-12", "2022-05-11",
    "2022-06-10", "2022-07-13", "2022-08-10", "2022-09-13", "2022-10-13",
    "2022-11-10", "2022-12-13",
    "2023-01-12", "2023-02-14", "2023-03-14", "2023-04-12", "2023-05-10",
    "2023-06-13", "2023-07-12", "2023-08-10", "2023-09-13", "2023-10-12",
    "2023-11-14", "2023-12-12",
    "2024-01-11", "2024-02-13", "2024-03-12", "2024-04-10", "2024-05-15",
    "2024-06-12", "2024-07-11", "2024-08-14", "2024-09-11", "2024-10-10",
    "2024-11-13", "2024-12-11",
    "2025-01-15", "2025-02-12", "2025-03-12", "2025-04-10", "2025-05-13",
    "2025-06-11", "2025-07-15", "2025-08-12", "2025-09-11", "2025-10-24",
    "2025-12-18",
    "2026-01-13", "2026-02-13", "2026-03-11", "2026-04-10", "2026-05-12",
    "2026-06-10", "2026-07-14", "2026-08-12", "2026-09-11", "2026-10-14",
    "2026-11-10", "2026-12-10",
)

DEFAULT_RSI_ENTRY = 5.0
DEFAULT_RSI_EXIT = 70.0
DEFAULT_TIME_STOP = 8
DEFAULT_VIX_MAX = None
DEFAULT_VIX_SKIP = 30.0
DEFAULT_SMA_TREND_RSI = 50
DEFAULT_PULLBACK_TIME_STOP = 6
DEFAULT_VIX_SIZE_REF = 20.0
DEFAULT_VIX_SIZE_MIN = 0.25
DEFAULT_ATR_MULT = 0.5
DEFAULT_ATR_PERIOD = 10
DEFAULT_COST_BPS_RT = 0.0
DEFAULT_RSI_CROSS_BACK = None
DEFAULT_MAX_ARM_BARS = 8
DEFAULT_MIN_CLOSE_LOC = 0.0
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
    "strategy",
    "size_mult",
    "vix_asof",
)


def rsi_wilder_wide(close: pd.DataFrame, period: int = 2) -> pd.DataFrame:
    """Wilder RSI (EWM alpha=1/period). Connors RSI-2 uses this, not SMA-RSI."""
    delta = close.astype(float).diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    alpha = 1.0 / float(period)
    avg_gain = gain.ewm(alpha=alpha, min_periods=int(period), adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, min_periods=int(period), adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def atr_wide(
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    period: int = DEFAULT_ATR_PERIOD,
) -> pd.DataFrame:
    """SMA of true range (matches indicators.trend.ATR)."""
    h = high.astype(float).reindex_like(close)
    lo = low.astype(float).reindex_like(close)
    c = close.astype(float)
    prev = c.shift(1)
    tr = np.maximum(h - lo, np.maximum((h - prev).abs(), (lo - prev).abs()))
    return tr.rolling(int(period), min_periods=int(period)).mean()


def bar_close_loc(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    """(close-low)/(high-low). 1.0 = close at the high (intraday rejection)."""
    h = high.astype(float).reindex_like(close)
    lo = low.astype(float).reindex_like(close)
    c = close.astype(float)
    rng = (h - lo).replace(0.0, np.nan)
    return ((c - lo) / rng).clip(0.0, 1.0)


def consecutive_lower_lows(
    low: pd.DataFrame,
    n: int = 3,
    *,
    exactly: bool = True,
) -> pd.DataFrame:
    """True when the last n lows are each strictly below the prior low.

    ``exactly`` also requires the (n+1)th lookback is *not* a lower low, so a
    four-day cascade does not count as an exact 3-day pullback.
    """
    lo = low.astype(float)
    ok = pd.DataFrame(True, index=lo.index, columns=lo.columns)
    for k in range(int(n)):
        ok = ok & (lo.shift(k) < lo.shift(k + 1))
    if exactly:
        ok = ok & ~(lo.shift(int(n)) < lo.shift(int(n) + 1))
    return ok.fillna(False)


def cpi_eve_mask(
    calendar: pd.DatetimeIndex,
    cpi_dates: Iterable[str] = CPI_RELEASE_DATES,
) -> pd.Series:
    """True on the last session strictly before each CPI 08:30 ET print."""
    cal = pd.DatetimeIndex(calendar).tz_localize(None).normalize().unique().sort_values()
    out = pd.Series(False, index=cal)
    cpi = pd.DatetimeIndex(pd.to_datetime(list(cpi_dates))).tz_localize(None).normalize()
    for day in cpi:
        prior = cal[cal < day]
        if len(prior) == 0:
            continue
        out.loc[prior[-1]] = True
    return out


def vix_skip_mask(
    calendar: pd.DatetimeIndex,
    vix: Optional[pd.Series],
    *,
    vix_max: Optional[float] = DEFAULT_VIX_MAX,
    use_prior_session: bool = True,
) -> pd.Series:
    """True when VIX (prior close by default) is above ``vix_max``.

    RSI-2 MOC is decided ~15:45 ET; cash VIX is not closed yet, so the causal
    print is the prior session. Missing VIX => never skip.
    """
    cal = pd.DatetimeIndex(calendar).tz_localize(None).normalize()
    if vix is None or vix_max is None:
        return pd.Series(False, index=cal)
    vx = vix.copy()
    if not isinstance(vx.index, pd.DatetimeIndex):
        vx.index = pd.to_datetime(vx.index)
    if vx.index.tz is not None:
        vx.index = vx.index.tz_convert(None)
    vx.index = pd.DatetimeIndex(vx.index).normalize()
    vx = vx[~vx.index.duplicated(keep="last")].sort_index().astype(float)
    aligned = vx.reindex(cal).ffill()
    if use_prior_session:
        aligned = aligned.shift(1)
    return (aligned > float(vix_max)).fillna(False)


def _align_vix(
    calendar: pd.DatetimeIndex,
    vix: Optional[pd.Series],
    *,
    use_prior_session: bool = True,
) -> pd.Series:
    cal = pd.DatetimeIndex(calendar).tz_localize(None).normalize()
    if vix is None:
        return pd.Series(np.nan, index=cal)
    vx = vix.copy()
    if not isinstance(vx.index, pd.DatetimeIndex):
        vx.index = pd.to_datetime(vx.index)
    if vx.index.tz is not None:
        vx.index = vx.index.tz_convert(None)
    vx.index = pd.DatetimeIndex(vx.index).normalize()
    vx = vx[~vx.index.duplicated(keep="last")].sort_index().astype(float)
    aligned = vx.reindex(cal).ffill()
    if use_prior_session:
        aligned = aligned.shift(1)
    return aligned


def vix_size_mult(
    calendar: pd.DatetimeIndex,
    vix: Optional[pd.Series],
    *,
    vix_ref: float = DEFAULT_VIX_SIZE_REF,
    min_mult: float = DEFAULT_VIX_SIZE_MIN,
    use_prior_session: bool = True,
) -> pd.Series:
    """Full size at VIX<=ref; size = ref/VIX floored at min_mult. Missing VIX => 1."""
    cal = pd.DatetimeIndex(calendar).tz_localize(None).normalize()
    aligned = _align_vix(cal, vix, use_prior_session=use_prior_session)
    ref = float(vix_ref)
    floor = float(min_mult)
    mult = (ref / aligned.replace(0.0, np.nan)).clip(upper=1.0)
    mult = mult.clip(lower=floor)
    return mult.reindex(cal).fillna(1.0)


def last_rth_15m_closes(
    frames_15m: Dict[str, pd.DataFrame],
    *,
    session: Optional[pd.Timestamp] = None,
) -> pd.Series:
    """Last RTH 15m close per symbol. Prefer the 15:45 ET bar, else last RTH print."""
    from utils.research.realistic_purchaser import as_et, is_last_rth_15m_bar, is_rth_15m_bar_start

    session_d = None
    if session is not None:
        session_d = pd.Timestamp(session).tz_localize(None).normalize().date()
    out: Dict[str, float] = {}
    for sym, df in frames_15m.items():
        if df is None or df.empty or "close" not in df.columns:
            continue
        work = df
        if "ts" in work.columns and not isinstance(work.index, pd.DatetimeIndex):
            work = work.set_index("ts")
        last_any = None
        last_1545 = None
        for ts, row in work.iterrows():
            try:
                if not is_rth_15m_bar_start(ts):
                    continue
                et = as_et(ts)
            except (TypeError, ValueError):
                continue
            if session_d is not None and et.date() != session_d:
                continue
            try:
                px = float(row["close"])
            except (TypeError, ValueError, KeyError):
                continue
            if not np.isfinite(px) or px <= 0:
                continue
            last_any = px
            if is_last_rth_15m_bar(ts):
                last_1545 = px
        pick = last_1545 if last_1545 is not None else last_any
        if pick is not None:
            out[str(sym).upper()] = float(pick)
    if not out:
        return pd.Series(dtype=float)
    return pd.Series(out, dtype=float)


def overlay_developing_close(
    daily_close: pd.DataFrame,
    session_closes: pd.Series,
    session: pd.Timestamp,
) -> pd.DataFrame:
    """Write IB 15m last-RTH closes onto the daily row for ``session`` (15:45 proxy)."""
    close = daily_close.astype(float).copy()
    close.index = pd.DatetimeIndex(close.index).tz_localize(None).normalize()
    session = pd.Timestamp(session).tz_localize(None).normalize()
    if session not in close.index:
        close.loc[session] = np.nan
        close = close.sort_index()
    for sym, px in session_closes.items():
        name = str(sym).upper()
        if name not in close.columns:
            continue
        try:
            val = float(px)
        except (TypeError, ValueError):
            continue
        if np.isfinite(val) and val > 0:
            close.loc[session, name] = val
    return close


def _liquid_mask(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    *,
    liquid_n: int,
    min_price: float,
    min_adv: float,
    adv_window: int = 30,
    lag: int = 1,
) -> pd.DataFrame:
    """PIT ADV/price floors. ``lag=1`` drops same-bar volume (3:45 does not have it)."""
    raw = top_n_liquid_mask(
        adv_dollar(close, volume, window=adv_window),
        top_n=liquid_n,
        min_adv=min_adv,
        close=close,
        min_price=min_price,
    ).reindex_like(close).fillna(False)
    if int(lag) <= 0:
        return raw.astype(bool)
    return raw.shift(int(lag), fill_value=False).astype(bool)


def _empty_trades() -> pd.DataFrame:
    return pd.DataFrame(columns=list(TRADE_COLUMNS))


def _finish_trades(rows: List[dict], cost_bps_rt: float) -> pd.DataFrame:
    if not rows:
        return _empty_trades()
    one_way = one_way_from_rt_bps(cost_bps_rt)
    out = pd.DataFrame(rows)
    buy = pd.to_numeric(out["buy_price"], errors="coerce")
    sell = pd.to_numeric(out["sell_price"], errors="coerce")
    ret = (sell * (1.0 - one_way)) / (buy * (1.0 + one_way)) - 1.0
    out["gain_pct"] = ret * 100.0
    out["buy_date"] = pd.to_datetime(out["buy_date"])
    out["sell_date"] = pd.to_datetime(out["sell_date"])
    return out.sort_values(["buy_date", "symbol"]).reset_index(drop=True)


def collect_rsi2_moc_trades(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    *,
    vix: Optional[pd.Series] = None,
    eval_start: str = "2018-01-01",
    eval_end: str = "2026-12-31",
    liquid_n: int = 500,
    min_price: float = 10.0,
    min_adv: float = 1_000_000.0,
    rsi_entry: float = DEFAULT_RSI_ENTRY,
    rsi_exit: float = DEFAULT_RSI_EXIT,
    time_stop: int = DEFAULT_TIME_STOP,
    vix_max: Optional[float] = DEFAULT_VIX_MAX,
    skip_cpi: bool = True,
    cost_bps_rt: float = DEFAULT_COST_BPS_RT,
    sma_trend: int = DEFAULT_SMA_TREND_RSI,
    sma_exit: int = 5,
    rsi_period: int = 2,
    vix_size_ref: float = DEFAULT_VIX_SIZE_REF,
    vix_size_min: float = DEFAULT_VIX_SIZE_MIN,
    rsi_cross_back: Optional[float] = DEFAULT_RSI_CROSS_BACK,
    max_arm_bars: int = DEFAULT_MAX_ARM_BARS,
) -> pd.DataFrame:
    """Per-symbol sequential RSI-2 MOC trades. Fill = signal-day close.

    ``sma_trend`` 0 disables the MA filter (use with a stricter ``rsi_entry``).
    ``rsi_cross_back`` None buys the oversold close. A number (5 or 10) waits
    until RSI reclaims that level after a prior bar with RSI < rsi_entry.
    """
    close = close.astype(float).sort_index()
    volume = volume.astype(float).reindex_like(close)
    use_sma = int(sma_trend) > 0
    sma_tr = (
        close.rolling(int(sma_trend), min_periods=int(sma_trend)).mean()
        if use_sma
        else pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    )
    sma5 = close.rolling(int(sma_exit), min_periods=int(sma_exit)).mean()
    rsi = rsi_wilder_wide(close, rsi_period)
    liquid = _liquid_mask(
        close, volume, liquid_n=liquid_n, min_price=min_price, min_adv=min_adv
    )
    cpi = cpi_eve_mask(close.index) if skip_cpi else pd.Series(False, index=close.index)
    vix_off = vix_skip_mask(close.index, vix, vix_max=vix_max, use_prior_session=True)
    vix_lvl = _align_vix(close.index, vix, use_prior_session=True)
    size_s = vix_size_mult(
        close.index,
        vix,
        vix_ref=vix_size_ref,
        min_mult=vix_size_min,
        use_prior_session=True,
    )
    start = pd.Timestamp(eval_start)
    end = pd.Timestamp(eval_end)
    dates = close.index
    n = len(dates)
    rows: List[dict] = []
    cross = None if rsi_cross_back is None else float(rsi_cross_back)
    max_arm = max(1, int(max_arm_bars))

    for sym in close.columns:
        c = close[sym].to_numpy(dtype=float)
        ma_tr = sma_tr[sym].to_numpy(dtype=float)
        ma_ex = sma5[sym].to_numpy(dtype=float)
        r = rsi[sym].to_numpy(dtype=float)
        liq = liquid[sym].to_numpy(dtype=bool)
        in_pos = False
        entry_px = 0.0
        entry_i = -1
        hold = 0
        score = np.nan
        size_m = 1.0
        vix_a = np.nan
        wait_bars = 0
        armed = False
        arm_i = -1

        for i in range(n):
            if in_pos:
                hold += 1
                reason = None
                if hold >= 1 and np.isfinite(c[i]) and np.isfinite(ma_ex[i]) and c[i] > ma_ex[i]:
                    reason = "sma5"
                elif hold >= 1 and np.isfinite(r[i]) and r[i] >= float(rsi_exit):
                    reason = "rsi"
                elif hold >= int(time_stop):
                    reason = "time"
                if reason is not None and np.isfinite(c[i]) and c[i] > 0:
                    buy_dt = dates[entry_i]
                    if start <= buy_dt <= end:
                        rows.append(
                            {
                                "symbol": str(sym),
                                "buy_date": buy_dt,
                                "sell_date": dates[i],
                                "buy_price": entry_px,
                                "sell_price": float(c[i]),
                                "exit_reason": reason,
                                "hold_bars": int(hold),
                                "entry_score": float(score) if np.isfinite(score) else np.nan,
                                "strategy": "rsi2_xback" if cross is not None else "rsi2_moc",
                                "size_mult": float(size_m),
                                "vix_asof": float(vix_a) if np.isfinite(vix_a) else np.nan,
                                "wait_bars": int(wait_bars),
                            }
                        )
                    in_pos = False
                continue

            if i >= n - 1:
                break
            dt = dates[i]
            if dt < start or dt > end:
                if cross is not None:
                    armed = False
                continue

            if cross is not None:
                trend_ok = True
                if use_sma:
                    trend_ok = bool(np.isfinite(ma_tr[i]) and c[i] > ma_tr[i])
                if armed and (not trend_ok or (i - arm_i) > max_arm):
                    armed = False
                skip_day = (
                    (not bool(liq[i]))
                    or (bool(cpi.iloc[i]) if dt in cpi.index else False)
                    or (bool(vix_off.iloc[i]) if dt in vix_off.index else False)
                )
                if (
                    armed
                    and (not skip_day)
                    and trend_ok
                    and np.isfinite(c[i])
                    and np.isfinite(r[i])
                    and r[i] >= cross
                    and c[i] > 0
                ):
                    in_pos = True
                    entry_px = float(c[i])
                    entry_i = i
                    hold = 0
                    score = float(r[i])
                    wait_bars = int(i - arm_i)
                    size_m = float(size_s.iloc[i]) if dt in size_s.index else 1.0
                    vix_a = float(vix_lvl.iloc[i]) if dt in vix_lvl.index else np.nan
                    armed = False
                    continue
                if (
                    (not armed)
                    and (not skip_day)
                    and trend_ok
                    and np.isfinite(c[i])
                    and np.isfinite(r[i])
                    and r[i] < float(rsi_entry)
                ):
                    armed = True
                    arm_i = i
                continue

            if not bool(liq[i]):
                continue
            if bool(cpi.iloc[i]) if dt in cpi.index else False:
                continue
            if bool(vix_off.iloc[i]) if dt in vix_off.index else False:
                continue
            rsi_ok = np.isfinite(c[i]) and np.isfinite(r[i]) and r[i] < float(rsi_entry)
            if use_sma:
                rsi_ok = rsi_ok and np.isfinite(ma_tr[i]) and c[i] > ma_tr[i]
            if not rsi_ok:
                continue
            in_pos = True
            entry_px = float(c[i])
            entry_i = i
            hold = 0
            score = float(r[i])
            wait_bars = 0
            size_m = float(size_s.iloc[i]) if dt in size_s.index else 1.0
            vix_a = float(vix_lvl.iloc[i]) if dt in vix_lvl.index else np.nan

        if in_pos:
            buy_dt = dates[entry_i]
            if start <= buy_dt <= end and np.isfinite(c[-1]) and c[-1] > 0:
                rows.append(
                    {
                        "symbol": str(sym),
                        "buy_date": buy_dt,
                        "sell_date": dates[-1],
                        "buy_price": entry_px,
                        "sell_price": float(c[-1]),
                        "exit_reason": "eod",
                        "hold_bars": int(n - 1 - entry_i),
                        "entry_score": float(score) if np.isfinite(score) else np.nan,
                        "strategy": "rsi2_xback" if cross is not None else "rsi2_moc",
                        "size_mult": float(size_m),
                        "vix_asof": float(vix_a) if np.isfinite(vix_a) else np.nan,
                        "wait_bars": int(wait_bars),
                    }
                )

    return _finish_trades(rows, cost_bps_rt)


def collect_pullback_trades(
    open_px: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    volume: pd.DataFrame,
    *,
    vix: Optional[pd.Series] = None,
    eval_start: str = "2018-01-01",
    eval_end: str = "2026-12-31",
    liquid_n: int = 500,
    min_price: float = 10.0,
    min_adv: float = 1_000_000.0,
    atr_mult: float = DEFAULT_ATR_MULT,
    atr_period: int = DEFAULT_ATR_PERIOD,
    exactly_three_lows: bool = True,
    vix_max: Optional[float] = DEFAULT_VIX_MAX,
    skip_cpi: bool = True,
    time_stop: int = DEFAULT_PULLBACK_TIME_STOP,
    cost_bps_rt: float = DEFAULT_COST_BPS_RT,
    sma_trend: int = 100,
    sma_pullback: int = 5,
    vix_size_ref: float = DEFAULT_VIX_SIZE_REF,
    vix_size_min: float = DEFAULT_VIX_SIZE_MIN,
    min_close_loc: float = DEFAULT_MIN_CLOSE_LOC,
) -> pd.DataFrame:
    """Per-symbol 3-day pullback: evening limit, next-day fill, next-open after up-close.

    ``min_close_loc`` > 0 requires the setup close in the upper fraction of that
    day's range (0.7 = close in the top 30% of the bar).
    """
    close = close.astype(float).sort_index()
    open_px = open_px.astype(float).reindex_like(close)
    high = high.astype(float).reindex_like(close)
    low = low.astype(float).reindex_like(close)
    volume = volume.astype(float).reindex_like(close)
    sma100 = close.rolling(int(sma_trend), min_periods=int(sma_trend)).mean()
    sma5 = close.rolling(int(sma_pullback), min_periods=int(sma_pullback)).mean()
    atr = atr_wide(high, low, close, atr_period)
    three = consecutive_lower_lows(low, n=3, exactly=exactly_three_lows)
    liquid = _liquid_mask(
        close, volume, liquid_n=liquid_n, min_price=min_price, min_adv=min_adv
    )
    cpi = cpi_eve_mask(close.index) if skip_cpi else pd.Series(False, index=close.index)
    # Setup is after the close, so same-session VIX is known.
    vix_off = vix_skip_mask(
        close.index, vix, vix_max=vix_max, use_prior_session=False
    )
    vix_lvl = _align_vix(close.index, vix, use_prior_session=False)
    size_s = vix_size_mult(
        close.index,
        vix,
        vix_ref=vix_size_ref,
        min_mult=vix_size_min,
        use_prior_session=False,
    )
    start = pd.Timestamp(eval_start)
    end = pd.Timestamp(eval_end)
    dates = close.index
    n = len(dates)
    rows: List[dict] = []

    for sym in close.columns:
        o = open_px[sym].to_numpy(dtype=float)
        h = high[sym].to_numpy(dtype=float)
        lo = low[sym].to_numpy(dtype=float)
        c = close[sym].to_numpy(dtype=float)
        ma_tr = sma100[sym].to_numpy(dtype=float)
        ma_pb = sma5[sym].to_numpy(dtype=float)
        a = atr[sym].to_numpy(dtype=float)
        ll = three[sym].to_numpy(dtype=bool)
        liq = liquid[sym].to_numpy(dtype=bool)
        in_pos = False
        pending_limit = False
        pending_moo = False
        pending_reason = "next_open"
        limit_px = np.nan
        setup_score = np.nan
        size_m = 1.0
        vix_a = np.nan
        entry_px = 0.0
        entry_i = -1
        hold = 0

        for i in range(n):
            if pending_moo and in_pos:
                if np.isfinite(o[i]) and o[i] > 0:
                    buy_dt = dates[entry_i]
                    if start <= buy_dt <= end:
                        rows.append(
                            {
                                "symbol": str(sym),
                                "buy_date": buy_dt,
                                "sell_date": dates[i],
                                "buy_price": entry_px,
                                "sell_price": float(o[i]),
                                "exit_reason": pending_reason,
                                "hold_bars": int(hold),
                                "entry_score": (
                                    float(setup_score) if np.isfinite(setup_score) else np.nan
                                ),
                                "strategy": "pullback3",
                                "size_mult": float(size_m),
                                "vix_asof": float(vix_a) if np.isfinite(vix_a) else np.nan,
                            }
                        )
                    in_pos = False
                pending_moo = False
                pending_reason = "next_open"

            if pending_limit and not in_pos:
                pending_limit = False
                fill_px = None
                if np.isfinite(o[i]) and o[i] > 0 and o[i] <= limit_px:
                    fill_px = float(o[i])
                elif np.isfinite(lo[i]) and lo[i] <= limit_px and np.isfinite(limit_px):
                    fill_px = float(limit_px)
                if fill_px is not None and fill_px > 0:
                    in_pos = True
                    entry_px = fill_px
                    entry_i = i
                    hold = 0

            if in_pos and not pending_moo:
                hold += 1
                up_close = (
                    i > 0
                    and np.isfinite(c[i])
                    and np.isfinite(c[i - 1])
                    and c[i] > c[i - 1]
                )
                timed = int(time_stop) > 0 and hold >= int(time_stop)
                if up_close or timed:
                    if i >= n - 1:
                        buy_dt = dates[entry_i]
                        if start <= buy_dt <= end and np.isfinite(c[i]) and c[i] > 0:
                            rows.append(
                                {
                                    "symbol": str(sym),
                                    "buy_date": buy_dt,
                                    "sell_date": dates[i],
                                    "buy_price": entry_px,
                                    "sell_price": float(c[i]),
                                    "exit_reason": "eod",
                                    "hold_bars": int(hold),
                                    "entry_score": (
                                        float(setup_score) if np.isfinite(setup_score) else np.nan
                                    ),
                                    "strategy": "pullback3",
                                    "size_mult": float(size_m),
                                    "vix_asof": float(vix_a) if np.isfinite(vix_a) else np.nan,
                                }
                            )
                        in_pos = False
                    else:
                        pending_moo = True
                        pending_reason = "next_open" if up_close else "time"

            dt = dates[i]
            if in_pos or pending_limit or pending_moo:
                continue
            if dt < start or dt > end or i >= n - 1:
                continue
            if not bool(liq[i]):
                continue
            if bool(cpi.iloc[i]) if dt in cpi.index else False:
                continue
            if bool(vix_off.iloc[i]) if dt in vix_off.index else False:
                continue
            if not (
                bool(ll[i])
                and np.isfinite(c[i])
                and np.isfinite(ma_tr[i])
                and np.isfinite(ma_pb[i])
                and np.isfinite(a[i])
                and a[i] > 0
                and c[i] > ma_tr[i]
                and c[i] < ma_pb[i]
            ):
                continue
            if float(min_close_loc) > 0:
                rng = float(h[i] - lo[i]) if np.isfinite(h[i]) and np.isfinite(lo[i]) else 0.0
                if rng <= 0:
                    continue
                loc = float((c[i] - lo[i]) / rng)
                if not np.isfinite(loc) or loc < float(min_close_loc):
                    continue
            pending_limit = True
            limit_px = float(c[i] - float(atr_mult) * a[i])
            setup_score = float((ma_pb[i] - c[i]) / c[i]) if c[i] else np.nan
            size_m = float(size_s.iloc[i]) if dt in size_s.index else 1.0
            vix_a = float(vix_lvl.iloc[i]) if dt in vix_lvl.index else np.nan

        if in_pos:
            buy_dt = dates[entry_i]
            px = float(c[-1]) if np.isfinite(c[-1]) and c[-1] > 0 else entry_px
            if start <= buy_dt <= end:
                rows.append(
                    {
                        "symbol": str(sym),
                        "buy_date": buy_dt,
                        "sell_date": dates[-1],
                        "buy_price": entry_px,
                        "sell_price": px,
                        "exit_reason": "eod",
                        "hold_bars": int(n - 1 - entry_i),
                        "entry_score": (
                            float(setup_score) if np.isfinite(setup_score) else np.nan
                        ),
                        "strategy": "pullback3",
                        "size_mult": float(size_m),
                        "vix_asof": float(vix_a) if np.isfinite(vix_a) else np.nan,
                    }
                )

    return _finish_trades(rows, cost_bps_rt)


def occupancy_ranked(
    trades: pd.DataFrame,
    *,
    max_open: int,
    score_ascending: bool = True,
) -> pd.DataFrame:
    """Unique-symbol occupancy. Same-day fills rank by ``entry_score``.

    Not ``weekly_deepvue.occupancy_take`` — that re-sorts by symbol name and
    would discard the oversold ranking.
    """
    if trades is None or trades.empty:
        return _empty_trades()
    cap = int(max_open)
    if cap <= 0:
        return trades.copy().reset_index(drop=True)
    work = trades.copy()
    buy_ts = pd.to_datetime(work["buy_date"])
    sell_ts = pd.to_datetime(work["sell_date"])
    stock = work["symbol"].astype(str)
    sc = pd.to_numeric(work["entry_score"], errors="coerce")
    rank = sc if score_ascending else -sc
    order = pd.DataFrame({"buy_ts": buy_ts, "rank": rank, "stock": stock, "i": work.index})
    order = order.sort_values(["buy_ts", "rank", "stock"], kind="mergesort")
    open_pos: List[Tuple[pd.Timestamp, str]] = []
    kept_i: List[int] = []
    for _, rec in order.iterrows():
        b = rec["buy_ts"]
        sym = str(rec["stock"])
        open_pos = [p for p in open_pos if p[0] > b]
        if any(s == sym for _, s in open_pos):
            continue
        if len(open_pos) >= cap:
            continue
        idx = int(rec["i"])
        kept_i.append(idx)
        open_pos.append((sell_ts.loc[idx], sym))
    if not kept_i:
        return _empty_trades()
    out = work.loc[kept_i].copy()
    return out.sort_values(["buy_date", "symbol"]).reset_index(drop=True)


def sized_gain_pct(trades: pd.DataFrame) -> pd.Series:
    """Equal-dollar gain scaled by VIX size_mult (1.0 if the column is missing)."""
    if trades is None or trades.empty:
        return pd.Series(dtype=float)
    g = pd.to_numeric(trades["gain_pct"], errors="coerce")
    if "size_mult" not in trades.columns:
        return g
    m = pd.to_numeric(trades["size_mult"], errors="coerce").fillna(1.0)
    return g * m


def scan_rsi2_moc(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    *,
    vix: Optional[pd.Series] = None,
    asof: Optional[pd.Timestamp] = None,
    liquid_n: int = 500,
    min_price: float = 10.0,
    min_adv: float = 1_000_000.0,
    rsi_entry: float = DEFAULT_RSI_ENTRY,
    vix_max: Optional[float] = DEFAULT_VIX_MAX,
    skip_cpi: bool = True,
    sma_trend: int = DEFAULT_SMA_TREND_RSI,
    rsi_period: int = 2,
    rsi_cross_back: Optional[float] = None,
    max_arm_bars: int = DEFAULT_MAX_ARM_BARS,
) -> pd.DataFrame:
    """Names that qualify for a same-session MOC on ``asof`` (default last bar).

    With ``rsi_cross_back``, fire only if RSI reclaimed that level after a
    prior oversold bar (RSI < rsi_entry) within ``max_arm_bars``, still above SMA.
    """
    close = close.astype(float).sort_index()
    if close.empty:
        return pd.DataFrame()
    asof = pd.Timestamp(asof) if asof is not None else close.index[-1]
    asof = asof.tz_localize(None).normalize() if getattr(asof, "tzinfo", None) else pd.Timestamp(asof).normalize()
    if asof not in close.index:
        return pd.DataFrame()
    use_sma = int(sma_trend) > 0
    sma_tr = (
        close.rolling(int(sma_trend), min_periods=int(sma_trend)).mean()
        if use_sma
        else None
    )
    rsi = rsi_wilder_wide(close, rsi_period)
    liquid = _liquid_mask(
        close, volume.reindex_like(close), liquid_n=liquid_n, min_price=min_price, min_adv=min_adv
    )
    cpi = cpi_eve_mask(close.index) if skip_cpi else pd.Series(False, index=close.index)
    vix_off = vix_skip_mask(close.index, vix, vix_max=vix_max, use_prior_session=True)
    row_c = close.loc[asof]
    row_r = rsi.loc[asof]
    row_l = liquid.loc[asof]
    skip = bool(cpi.loc[asof]) if asof in cpi.index else False
    skip = skip or (bool(vix_off.loc[asof]) if asof in vix_off.index else False)
    cols = ["symbol", "close", "sma_trend", "rsi2", "asof", "size_mult", "status"]
    if skip:
        return pd.DataFrame(columns=cols)
    loc = close.index.get_loc(asof)
    if not isinstance(loc, (int, np.integer)):
        return pd.DataFrame(columns=cols)
    i_asof = int(loc)
    row_sma = sma_tr.loc[asof] if use_sma and sma_tr is not None else None
    if rsi_cross_back is None:
        ok = row_l.fillna(False).astype(bool) & (row_r < float(rsi_entry))
        if use_sma and sma_tr is not None:
            ok = ok & (row_c > sma_tr.loc[asof])
        names = [s for s, v in ok.items() if bool(v) and np.isfinite(row_c[s]) and np.isfinite(row_r[s])]
        status = {s: "fire" for s in names}
    else:
        names = []
        status = {}
        cross = float(rsi_cross_back)
        max_arm = max(1, int(max_arm_bars))
        start_j = max(0, i_asof - max_arm)
        for s in close.columns:
            if not bool(row_l.get(s, False)):
                continue
            px = row_c.get(s)
            rv = row_r.get(s)
            if not (np.isfinite(px) and np.isfinite(rv)):
                continue
            if use_sma and row_sma is not None:
                ma = row_sma.get(s)
                if not (np.isfinite(ma) and float(px) > float(ma)):
                    continue
            r_path = rsi[s].to_numpy(dtype=float)
            c_path = close[s].to_numpy(dtype=float)
            ma_path = sma_tr[s].to_numpy(dtype=float) if use_sma and sma_tr is not None else None
            armed = False
            for j in range(start_j, i_asof + 1):
                trend_ok = True
                if use_sma and ma_path is not None:
                    trend_ok = bool(np.isfinite(ma_path[j]) and c_path[j] > ma_path[j])
                if not trend_ok:
                    armed = False
                    continue
                if armed and j == i_asof and float(rv) >= cross:
                    names.append(s)
                    status[s] = "fire"
                    break
                if (not armed) and np.isfinite(r_path[j]) and r_path[j] < float(rsi_entry):
                    if j < i_asof:
                        armed = True
                    elif j == i_asof:
                        names.append(s)
                        status[s] = "armed"
                        break
            else:
                if armed and float(rv) < cross:
                    names.append(s)
                    status[s] = "armed"
    if not names:
        return pd.DataFrame(columns=cols)
    size_s = vix_size_mult(close.index, vix, use_prior_session=True)
    sm = float(size_s.loc[asof]) if asof in size_s.index else 1.0
    out = pd.DataFrame(
        {
            "symbol": names,
            "close": [float(row_c[s]) for s in names],
            "sma_trend": [
                None if not use_sma else float(sma_tr.loc[asof, s]) for s in names
            ],
            "rsi2": [float(row_r[s]) for s in names],
            "asof": asof,
            "size_mult": sm,
            "status": [status.get(s, "fire") for s in names],
        }
    )
    return out.sort_values("rsi2").reset_index(drop=True)


def scan_pullback_limits(
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    volume: pd.DataFrame,
    *,
    vix: Optional[pd.Series] = None,
    asof: Optional[pd.Timestamp] = None,
    liquid_n: int = 500,
    min_price: float = 10.0,
    min_adv: float = 1_000_000.0,
    atr_mult: float = DEFAULT_ATR_MULT,
    atr_period: int = DEFAULT_ATR_PERIOD,
    exactly_three_lows: bool = True,
    vix_max: Optional[float] = DEFAULT_VIX_MAX,
    skip_cpi: bool = True,
    sma_trend: int = 100,
    sma_pullback: int = 5,
    min_close_loc: float = DEFAULT_MIN_CLOSE_LOC,
) -> pd.DataFrame:
    """Evening limit orders to rest on the next session."""
    close = close.astype(float).sort_index()
    if close.empty:
        return pd.DataFrame()
    asof = pd.Timestamp(asof) if asof is not None else close.index[-1]
    asof = pd.Timestamp(asof).tz_localize(None).normalize() if getattr(asof, "tzinfo", None) else pd.Timestamp(asof).normalize()
    if asof not in close.index:
        return pd.DataFrame()
    high = high.astype(float).reindex_like(close)
    low = low.astype(float).reindex_like(close)
    sma100 = close.rolling(int(sma_trend), min_periods=int(sma_trend)).mean()
    sma5 = close.rolling(int(sma_pullback), min_periods=int(sma_pullback)).mean()
    atr = atr_wide(high, low, close, atr_period)
    three = consecutive_lower_lows(low, n=3, exactly=exactly_three_lows)
    liquid = _liquid_mask(
        close, volume.reindex_like(close), liquid_n=liquid_n, min_price=min_price, min_adv=min_adv
    )
    cpi = cpi_eve_mask(close.index) if skip_cpi else pd.Series(False, index=close.index)
    vix_off = vix_skip_mask(close.index, vix, vix_max=vix_max, use_prior_session=False)
    skip = bool(cpi.loc[asof]) if asof in cpi.index else False
    skip = skip or (bool(vix_off.loc[asof]) if asof in vix_off.index else False)
    if skip:
        return pd.DataFrame(columns=["symbol", "close", "limit", "atr10", "sma100", "sma5", "asof"])
    ok = (
        liquid.loc[asof].fillna(False).astype(bool)
        & three.loc[asof].fillna(False).astype(bool)
        & (close.loc[asof] > sma100.loc[asof])
        & (close.loc[asof] < sma5.loc[asof])
        & (atr.loc[asof] > 0)
    )
    if float(min_close_loc) > 0:
        loc = bar_close_loc(high, low, close).loc[asof]
        ok = ok & (loc >= float(min_close_loc))
    names = [s for s, v in ok.items() if bool(v) and np.isfinite(close.loc[asof, s])]
    if not names:
        return pd.DataFrame(columns=["symbol", "close", "limit", "atr10", "sma100", "sma5", "asof"])
    rows = []
    for s in names:
        c = float(close.loc[asof, s])
        a = float(atr.loc[asof, s])
        rows.append(
            {
                "symbol": s,
                "close": c,
                "limit": c - float(atr_mult) * a,
                "atr10": a,
                "sma100": float(sma100.loc[asof, s]),
                "sma5": float(sma5.loc[asof, s]),
                "asof": asof,
            }
        )
    return pd.DataFrame(rows).sort_values("limit").reset_index(drop=True)
