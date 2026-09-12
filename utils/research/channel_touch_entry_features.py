"""Point-in-time entry features for channel-touch trades (no look-ahead).

Stock series are computed once per symbol. Snapshot at the last completed
bar before a wick fill (fill_i-1). SPY-regime columns use completed_asof
so a daily bar's close is not treated as known mid-session.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from indicators.momentum import RSI
from indicators.moving_averages import SMA
from indicators.trend import ATR, BollingerBands

FEATURE_COLS: Sequence[str] = (
    "rsi_14",
    "dist_sma50_pct",
    "dist_sma200_pct",
    "sma50_gt_sma200",
    "rs_spy_21d",
    "spy_ret_20d",
    "spy_above_sma50",
    "spy_atr_pct",
    "bb_pctb",
    "range_pct",
    "close_loc",
    "volume_rel_20",
    "squeeze_mom",
    "squeeze_mom_rising",
    "max_beyond_width",
    "formation_beyond_width",
    "channel_span_days",
    "channel_age_at_buy_days",
    "dow",
    "month",
)

CATEGORICAL_FEATURES: Sequence[str] = (
    "sma50_gt_sma200",
    "squeeze_mom_rising",
    "spy_above_sma50",
    "dow",
    "month",
)


def _as_float(v: object) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return x if np.isfinite(x) else float("nan")


def _round_or_none(v: float, ndigits: int = 4) -> Optional[float]:
    x = _as_float(v)
    if not np.isfinite(x):
        return None
    return round(x, ndigits)


def max_beyond_width(
    high: np.ndarray,
    support_y0: float,
    support_x0: int,
    support_slope: float,
    width: float,
    start_i: int,
    end_i: int,
) -> float:
    """Max (high - resist) / width over [start_i, end_i]. Below-rail bars count as 0."""
    if not np.isfinite(width) or width <= 0:
        return float("nan")
    if not np.isfinite(support_y0) or not np.isfinite(support_slope):
        return float("nan")
    n = len(high)
    if n == 0:
        return float("nan")
    lo = max(0, int(start_i))
    hi = min(n - 1, int(end_i))
    if hi < lo:
        return float("nan")
    best = 0.0
    any_bar = False
    x0 = int(support_x0)
    for i in range(lo, hi + 1):
        h = float(high[i])
        if not np.isfinite(h):
            continue
        resist = float(support_y0) + float(support_slope) * (i - x0) + float(width)
        if not np.isfinite(resist):
            continue
        any_bar = True
        overshoot = (h - resist) / float(width)
        if np.isfinite(overshoot) and overshoot > best:
            best = float(overshoot)
    return float(best) if any_bar else float("nan")


def stock_entry_feature_series(
    df: pd.DataFrame,
    *,
    squeeze_mom: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Vectorized per-bar stock features. Length matches df."""
    n = len(df)
    nan = np.full(n, np.nan, dtype=float)
    if n == 0 or "close" not in df.columns:
        return {
            "rsi_14": nan,
            "dist_sma50_pct": nan,
            "dist_sma200_pct": nan,
            "sma50_gt_sma200": nan,
            "bb_pctb": nan,
            "range_pct": nan,
            "close_loc": nan,
            "volume_rel_20": nan,
            "squeeze_mom": nan if squeeze_mom is None else np.asarray(squeeze_mom, dtype=float),
            "squeeze_mom_rising": nan,
        }

    close_s = df["close"].astype(float)
    high_s = df["high"].astype(float) if "high" in df.columns else close_s
    low_s = df["low"].astype(float) if "low" in df.columns else close_s
    close = close_s.to_numpy(dtype=float)
    high = high_s.to_numpy(dtype=float)
    low = low_s.to_numpy(dtype=float)

    rsi = RSI(close_s, period=14).to_numpy(dtype=float)
    sma50 = SMA(close_s, period=50).to_numpy(dtype=float)
    sma200 = SMA(close_s, period=200).to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        dist50 = np.where(sma50 > 0, (close / sma50 - 1.0) * 100.0, np.nan)
        dist200 = np.where(sma200 > 0, (close / sma200 - 1.0) * 100.0, np.nan)
        sma_cross = np.where(
            np.isfinite(sma50) & np.isfinite(sma200),
            (sma50 > sma200).astype(float),
            np.nan,
        )

    bb = BollingerBands(close_s, period=20, std_dev=2)
    band = (bb["upper"] - bb["lower"]).to_numpy(dtype=float)
    lower = bb["lower"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        pctb = np.where(band > 0, (close - lower) / band, np.nan)

    with np.errstate(divide="ignore", invalid="ignore"):
        range_pct = np.where(close > 0, (high - low) / close * 100.0, np.nan)
        hl = high - low
        close_loc = np.where(hl > 0, (close - low) / hl, np.nan)

    if "volume" in df.columns:
        vol = df["volume"].astype(float)
        vol_ma = vol.rolling(20, min_periods=20).mean().to_numpy(dtype=float)
        vol_np = vol.to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            volume_rel = np.where(vol_ma > 0, vol_np / vol_ma, np.nan)
    else:
        volume_rel = nan.copy()

    if squeeze_mom is None or len(squeeze_mom) != n:
        sq = nan.copy()
        sq_rising = nan.copy()
    else:
        sq = np.asarray(squeeze_mom, dtype=float)
        sq_rising = np.full(n, np.nan, dtype=float)
        if n >= 2:
            prev = sq[:-1]
            cur = sq[1:]
            ok = np.isfinite(prev) & np.isfinite(cur)
            sq_rising[1:] = np.where(ok, (cur >= prev).astype(float), np.nan)

    return {
        "rsi_14": rsi,
        "dist_sma50_pct": dist50,
        "dist_sma200_pct": dist200,
        "sma50_gt_sma200": sma_cross,
        "bb_pctb": pctb,
        "range_pct": range_pct,
        "close_loc": close_loc,
        "volume_rel_20": volume_rel,
        "squeeze_mom": sq,
        "squeeze_mom_rising": sq_rising,
    }


def snapshot_stock_features(series: Dict[str, np.ndarray], entry_i: int) -> dict:
    """Scalar snapshot at entry_i (pass fill_i-1 so the fill bar's close is unused). Missing/OOB -> None."""
    out: dict = {}
    for key, arr in series.items():
        if arr is None or entry_i < 0 or entry_i >= len(arr):
            out[key] = None
            continue
        ndigits = 6 if key == "squeeze_mom" else 4
        if key in ("sma50_gt_sma200", "squeeze_mom_rising"):
            v = _as_float(arr[entry_i])
            out[key] = None if not np.isfinite(v) else int(v)
        else:
            out[key] = _round_or_none(float(arr[entry_i]), ndigits)
    return out


def completed_asof(row: object, *, series_is_daily: bool) -> Optional[pd.Timestamp]:
    """Last completed bar timestamp that is legal at a wick fill.

    Stock/SPY features may only use bars with timestamp < fill time.
    Daily bars are stamped at session midnight and already contain that
    session's close, so an intra-day fill must use the prior session.
    """

    def _get(key: str) -> object:
        if isinstance(row, dict):
            return row.get(key)
        try:
            return row[key]
        except Exception:
            return None

    def _ts(val: object) -> Optional[pd.Timestamp]:
        if val is None or (isinstance(val, float) and not np.isfinite(val)):
            return None
        if pd.isna(val):
            return None
        text = str(val).strip()
        if text in ("", "nan", "None", "NaT"):
            return None
        try:
            return pd.Timestamp(val)
        except Exception:
            return None

    feat = _ts(_get("feature_asof"))
    buy_time = _ts(_get("buy_time"))
    buy_date = _ts(_get("buy_date"))
    if feat is not None:
        ts = feat
    elif buy_time is not None:
        ts = buy_time
        if not series_is_daily:
            ts = ts - pd.Timedelta(milliseconds=1)
    elif buy_date is not None:
        ts = buy_date
        if series_is_daily:
            return ts
    else:
        return None
    if series_is_daily:
        session = pd.Timestamp(ts).normalize()
        intraday = bool(ts.hour or ts.minute or ts.second or ts.microsecond) or buy_time is not None
        if intraday:
            return session - pd.Timedelta(days=1)
        return ts
    return ts


def enrich_spy_entry_features(trades: pd.DataFrame, spy_df: pd.DataFrame) -> pd.DataFrame:
    """Attach SPY 20d return, SMA50 regime, and ATR%% at last completed bar before fill."""
    if trades.empty:
        return trades
    out = trades.copy()
    if spy_df is None or spy_df.empty or "close" not in spy_df.columns:
        for col in ("spy_ret_20d", "spy_above_sma50", "spy_above_sma200", "spy_atr_pct"):
            if col not in out.columns:
                out[col] = np.nan
        return out

    spy = spy_df.copy()
    if not isinstance(spy.index, pd.DatetimeIndex):
        spy.index = pd.DatetimeIndex(spy.index)
    if spy.index.tz is not None:
        spy.index = spy.index.tz_convert(None)
    spy = spy.sort_index()
    close = spy["close"].astype(float)
    sma50 = close.rolling(50, min_periods=50).mean()
    sma200 = close.rolling(200, min_periods=200).mean()
    if {"high", "low"}.issubset(spy.columns):
        atr = ATR(spy["high"].astype(float), spy["low"].astype(float), close, period=14)
    else:
        atr = pd.Series(np.nan, index=spy.index)
    atr_pct = (atr / close * 100.0).replace([np.inf, -np.inf], np.nan)
    spy_is_daily = True
    if len(spy.index) >= 3:
        deltas = pd.Series(spy.index).diff().dropna()
        med = deltas.median()
        if pd.notna(med) and med < pd.Timedelta(hours=20):
            spy_is_daily = False

    ret20: List[Optional[float]] = []
    above: List[Optional[int]] = []
    above200: List[Optional[int]] = []
    atr_list: List[Optional[float]] = []
    for _, row in out.iterrows():
        asof = completed_asof(row, series_is_daily=spy_is_daily)
        if asof is None:
            ret20.append(None)
            above.append(None)
            above200.append(None)
            atr_list.append(None)
            continue
        hist_c = close.loc[:asof]
        if len(hist_c) < 21:
            ret20.append(None)
        else:
            end_px = float(hist_c.iloc[-1])
            start_px = float(hist_c.iloc[-21])
            if end_px > 0 and start_px > 0 and np.isfinite(end_px) and np.isfinite(start_px):
                ret20.append(round((end_px / start_px - 1.0) * 100.0, 4))
            else:
                ret20.append(None)
        hist_s = sma50.loc[:asof]
        if hist_c.empty or hist_s.empty or not np.isfinite(float(hist_s.iloc[-1])):
            above.append(None)
        else:
            above.append(int(float(hist_c.iloc[-1]) > float(hist_s.iloc[-1])))
        hist_s200 = sma200.loc[:asof]
        if hist_c.empty or hist_s200.empty or not np.isfinite(float(hist_s200.iloc[-1])):
            above200.append(None)
        else:
            above200.append(int(float(hist_c.iloc[-1]) > float(hist_s200.iloc[-1])))
        hist_a = atr_pct.loc[:asof]
        if hist_a.empty or not np.isfinite(float(hist_a.iloc[-1])):
            atr_list.append(None)
        else:
            atr_list.append(round(float(hist_a.iloc[-1]), 4))

    out["spy_ret_20d"] = ret20
    out["spy_above_sma50"] = above
    out["spy_above_sma200"] = above200
    out["spy_atr_pct"] = atr_list
    return out
