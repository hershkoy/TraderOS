"""Causal Deepvue-style weekend swing scan (Moglen notes, not Deepvue).

Weekend signal uses completed daily bars through the week's last session
(usually Friday). Live fill is the next RTH 09:30 ET 15m mid. Sells: 3% hard
stop on 15m (decision bar N, fill N+1 mid) or two daily closes below the 21 EMA
then the next 15m mid. See docs/strategies/weekly_deepvue.md and
docs/features/realistic_purchasing.md.

This is research, not a promote.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from utils.research.realistic_exits import (
    EXIT_EOD,
    EXIT_HARD_STOP,
    ExitResult,
    REASON_EOD,
    REASON_FILLED,
    REASON_NO_15M,
    REASON_NO_ENTRY_BAR,
    REASON_NO_NEXT_BAR,
    find_bar_index,
    flatten_rth_sessions,
)
from utils.research.realistic_purchaser import as_et, bar_mid, is_last_rth_15m_bar

EXIT_MA_2CLOSE = "ma_2close"

DEFAULT_STOP_PCT = 0.03
DEFAULT_ADR_MIN = 3.0
DEFAULT_ADV_MIN = 20_000_000.0
DEFAULT_RS_PCT = 85.0
DEFAULT_RMV_MAX = 45.0
DEFAULT_NEAR_MA_PCT = 0.02
DEFAULT_THEME_TOP = 6
DEFAULT_MAX_PER_SECTOR = 2
DEFAULT_WEEKLY_FOCUS = 15
DEFAULT_DAILY_FOCUS = 4
DEFAULT_MAX_OPEN = 4
DEFAULT_MIN_PRICE = 5.0


@dataclass(frozen=True)
class DeepvueParams:
    stop_pct: float = DEFAULT_STOP_PCT
    adr_min: float = DEFAULT_ADR_MIN
    adv_min: float = DEFAULT_ADV_MIN
    rs_pct: float = DEFAULT_RS_PCT
    rmv_max: float = DEFAULT_RMV_MAX
    near_ma_pct: float = DEFAULT_NEAR_MA_PCT
    theme_top: int = DEFAULT_THEME_TOP
    max_per_sector: int = DEFAULT_MAX_PER_SECTOR
    weekly_focus: int = DEFAULT_WEEKLY_FOCUS
    daily_focus: int = DEFAULT_DAILY_FOCUS
    max_open: int = DEFAULT_MAX_OPEN
    min_price: float = DEFAULT_MIN_PRICE
    ema_fast: int = 10
    ema_slow: int = 21
    rmv_short: int = 5
    rmv_long: int = 15
    rs_1m: int = 21
    rs_3m: int = 63
    adr_len: int = 14
    adv_len: int = 20
    ma_closes: int = 2


def ema(close: pd.DataFrame | pd.Series, span: int) -> pd.DataFrame | pd.Series:
    return close.astype(float).ewm(span=int(span), adjust=False, min_periods=int(span)).mean()


def week_end_mask(index: pd.DatetimeIndex) -> pd.Series:
    """True on the last session of each W-FRI week (Thursday if Friday is missing)."""
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_convert(None)
    idx = idx.normalize()
    s = pd.Series(idx, index=idx)
    last = s.groupby(s.dt.to_period("W-FRI")).transform("max")
    return s.eq(last)


def qqq_risk_on(close: pd.Series, span: int = 21) -> pd.Series:
    """Aggressive stance: close above a rising 21 EMA (completed bar)."""
    c = close.astype(float).copy()
    if not isinstance(c.index, pd.DatetimeIndex):
        c.index = pd.to_datetime(c.index)
    c.index = pd.DatetimeIndex(c.index).tz_localize(None).normalize()
    c = c[~c.index.duplicated(keep="last")].sort_index()
    ma = ema(c, span)
    rising = ma > ma.shift(1)
    return (c > ma) & rising


def adr_pct(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    rng = (high.astype(float) - low.astype(float)).abs()
    c = close.astype(float).replace(0.0, np.nan)
    return (100.0 * rng / c).rolling(int(window), min_periods=int(window)).mean()


def rmv_ratio(high: pd.DataFrame, low: pd.DataFrame, short: int = 5, long: int = 15) -> pd.DataFrame:
    """0-100-ish tightness: 5d range / 15d range. Low = contracted (Deepvue RMV 0)."""
    h = high.astype(float)
    l = low.astype(float)
    short_n = max(1, int(short))
    long_n = max(short_n + 1, int(long))
    rng_s = h.rolling(short_n, min_periods=short_n).max() - l.rolling(short_n, min_periods=short_n).min()
    rng_l = h.rolling(long_n, min_periods=long_n).max() - l.rolling(long_n, min_periods=long_n).min()
    out = 100.0 * rng_s / rng_l.replace(0.0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def rs_percentile(close: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Cross-sectional percentile of lookback returns (0-100). Point-in-time."""
    ret = close.astype(float).pct_change(int(lookback), fill_method=None)
    return ret.rank(axis=1, pct=True, method="average") * 100.0


def near_ma_pct(close: pd.DataFrame, ma: pd.DataFrame) -> pd.DataFrame:
    c = close.astype(float)
    m = ma.astype(float).replace(0.0, np.nan)
    return (c / m - 1.0).abs()


def build_signal_frame(
    *,
    open_px: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    volume: pd.DataFrame,
    qqq_on: pd.Series,
    sectors: Mapping[str, str],
    params: DeepvueParams,
) -> pd.DataFrame:
    """One row per symbol per week-end session that passes the Leaders + tightness screen."""
    c = close.astype(float)
    h = high.astype(float).reindex_like(c)
    l = low.astype(float).reindex_like(c)
    v = volume.astype(float).reindex_like(c)
    o = open_px.astype(float).reindex_like(c)
    ema10 = ema(c, params.ema_fast)
    ema21 = ema(c, params.ema_slow)
    adv = (c * v).rolling(params.adv_len, min_periods=max(5, params.adv_len // 2)).mean()
    adr = adr_pct(h, l, c, params.adr_len)
    rmv = rmv_ratio(h, l, params.rmv_short, params.rmv_long)
    rs1 = rs_percentile(c, params.rs_1m)
    rs3 = rs_percentile(c, params.rs_3m)
    near = np.minimum(near_ma_pct(c, ema10), near_ma_pct(c, ema21))
    liquid = (
        (adv >= float(params.adv_min))
        & (c >= float(params.min_price))
        & (adr >= float(params.adr_min))
        & (c > ema21)
    )
    qon = qqq_on.reindex(c.index).fillna(False).astype(bool)
    ends = week_end_mask(c.index)
    rows: List[dict] = []
    for ts in c.index[ends & qon]:
        liq = liquid.loc[ts]
        if not bool(liq.any()):
            continue
        names = [str(s) for s in liq.index[liq.fillna(False).to_numpy()]]
        if not names:
            continue
        sec_ret = _sector_returns(c.loc[ts], c, names, sectors, lookback=params.rs_1m)
        lead = set(_top_sectors(sec_ret, params.theme_top))
        day_rows: List[dict] = []
        for sym in names:
            rmv_v = _num(rmv.at[ts, sym] if sym in rmv.columns else np.nan)
            rs1_v = _num(rs1.at[ts, sym] if sym in rs1.columns else np.nan)
            rs3_v = _num(rs3.at[ts, sym] if sym in rs3.columns else np.nan)
            near_v = _num(near.at[ts, sym] if sym in near.columns else np.nan)
            if rmv_v is None or rs1_v is None or rs3_v is None or near_v is None:
                continue
            if rs1_v < params.rs_pct or rs3_v < params.rs_pct:
                continue
            if rmv_v > params.rmv_max or near_v > params.near_ma_pct:
                continue
            sec = str(sectors.get(sym, "") or "Unknown")
            if lead and sec not in lead and sec != "Unknown":
                continue
            day_rows.append(
                {
                    "signal_date": pd.Timestamp(ts).normalize(),
                    "stock": sym,
                    "sector": sec,
                    "close": float(c.at[ts, sym]),
                    "ema21": _num(ema21.at[ts, sym]),
                    "rmv": float(rmv_v),
                    "rs_1m": float(rs1_v),
                    "rs_3m": float(rs3_v),
                    "near_ma_pct": float(near_v),
                    "adr_pct": _num(adr.at[ts, sym]),
                    "adv": _num(adv.at[ts, sym]),
                    "open": _num(o.at[ts, sym]),
                }
            )
        rows.extend(_pick_focus(day_rows, params))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _pick_focus(day_rows: Sequence[dict], params: DeepvueParams) -> List[dict]:
    if not day_rows:
        return []
    df = pd.DataFrame(list(day_rows)).sort_values(["rmv", "near_ma_pct"], kind="mergesort")
    kept: List[dict] = []
    per: Dict[str, int] = {}
    for rec in df.to_dict("records"):
        sec = str(rec.get("sector") or "Unknown")
        if per.get(sec, 0) >= int(params.max_per_sector):
            continue
        kept.append(rec)
        per[sec] = per.get(sec, 0) + 1
        if len(kept) >= int(params.weekly_focus):
            break
    kept.sort(key=lambda r: (float(r["rmv"]), float(r["near_ma_pct"])))
    out = kept[: int(params.daily_focus)]
    for i, rec in enumerate(out):
        rec = dict(rec)
        rec["focus_rank"] = i + 1
        out[i] = rec
    return out


def _sector_returns(
    close_row: pd.Series,
    close: pd.DataFrame,
    names: Sequence[str],
    sectors: Mapping[str, str],
    lookback: int,
) -> Dict[str, float]:
    loc = close.index.get_loc(close_row.name)
    if isinstance(loc, slice) or isinstance(loc, np.ndarray):
        loc = int(np.max(np.asarray(loc)))
    i = int(loc)
    j = i - int(lookback)
    if j < 0:
        return {}
    prev = close.iloc[j]
    buckets: Dict[str, List[float]] = {}
    for sym in names:
        if sym not in close_row.index or sym not in prev.index:
            continue
        now = _num(close_row[sym])
        then = _num(prev[sym])
        if now is None or then is None or then <= 0:
            continue
        sec = str(sectors.get(sym, "") or "Unknown")
        buckets.setdefault(sec, []).append(now / then - 1.0)
    return {k: float(np.median(v)) for k, v in buckets.items() if v}


def _top_sectors(sec_ret: Mapping[str, float], top_n: int) -> List[str]:
    ranked = sorted(
        ((k, v) for k, v in sec_ret.items() if k and k != "Unknown"),
        key=lambda kv: kv[1],
        reverse=True,
    )
    return [k for k, _ in ranked[: max(1, int(top_n))]]


def _num(val: Any) -> Optional[float]:
    try:
        x = float(val)
    except (TypeError, ValueError):
        return None
    if x != x:
        return None
    return x


def daily_below_ema21(
    close: pd.Series,
    ema21: pd.Series,
) -> Dict[date, bool]:
    out: Dict[date, bool] = {}
    c = close.astype(float)
    m = ema21.astype(float).reindex(c.index)
    for ts, px in c.items():
        ma = m.at[ts] if ts in m.index else np.nan
        if px != px or ma != ma:
            continue
        out[pd.Timestamp(ts).date()] = bool(float(px) < float(ma))
    return out


def simulate_deepvue_exit(
    by_day: Dict[date, Sequence[dict]],
    *,
    entry_ts: Any,
    entry_px: float,
    below_ema: Mapping[date, bool],
    stop_pct: float = DEFAULT_STOP_PCT,
    ma_closes: int = 2,
) -> ExitResult:
    """3% hard stop on 15m N+1 mid, or two session closes below 21 EMA then next 15m mid."""
    bars = flatten_rth_sessions(by_day)
    px = float(entry_px)
    if not bars:
        return ExitResult(filled=False, reason=REASON_NO_15M)
    if px != px or px <= 0:
        return ExitResult(filled=False, reason=REASON_NO_ENTRY_BAR)
    entry_i = find_bar_index(bars, entry_ts)
    if entry_i is None:
        return ExitResult(filled=False, reason=REASON_NO_ENTRY_BAR)
    hard = px * (1.0 - float(stop_pct))
    last_i = len(bars) - 1
    if entry_i >= last_i:
        return ExitResult(
            filled=False,
            reason=REASON_NO_NEXT_BAR,
            hard_stop=hard,
            peak_px=px,
        )
    below_run = 0
    peak = px
    need = max(1, int(ma_closes))
    for i in range(entry_i + 1, len(bars)):
        hi = _ohlc(bars[i], "high")
        lo = _ohlc(bars[i], "low")
        if hi is not None:
            peak = max(peak, hi)
        if lo is not None and lo <= hard + 1e-12:
            return _next_mid(
                bars,
                decision_i=i,
                peak=peak,
                hard=hard,
                exit_reason=EXIT_HARD_STOP,
            )
        ts = bars[i].get("ts")
        if ts is not None and is_last_rth_15m_bar(ts):
            sess = as_et(ts).date()
            if bool(below_ema.get(sess, False)):
                below_run += 1
            else:
                below_run = 0
            if below_run >= need:
                return _next_mid(
                    bars,
                    decision_i=i,
                    peak=peak,
                    hard=hard,
                    exit_reason=EXIT_MA_2CLOSE,
                )
    last = bars[-1]
    cl = _ohlc(last, "close")
    if cl is None:
        return ExitResult(
            filled=False,
            reason=REASON_EOD,
            peak_px=peak,
            hard_stop=hard,
        )
    return ExitResult(
        filled=True,
        reason=REASON_EOD,
        sell_px=float(cl),
        decision_ts=last.get("ts"),
        exec_bar_ts=last.get("ts"),
        exit_reason=EXIT_EOD,
        peak_px=peak,
        hard_stop=hard,
        stop_at_decision=hard,
    )


def occupancy_take(
    trades: Sequence[dict],
    *,
    max_open: int,
) -> List[dict]:
    """Chronological unique-symbol occupancy. ``sell_ts`` must be comparable."""
    cap = max(1, int(max_open))
    open_pos: List[Tuple[Any, str]] = []
    kept: List[dict] = []
    ordered = sorted(trades, key=lambda r: (r["buy_ts"], r["stock"]))
    for rec in ordered:
        buy_ts = rec["buy_ts"]
        sym = str(rec["stock"])
        open_pos = [p for p in open_pos if p[0] > buy_ts]
        if any(s == sym for _, s in open_pos):
            continue
        if len(open_pos) >= cap:
            continue
        kept.append(rec)
        open_pos.append((rec["sell_ts"], sym))
    return kept


def _next_mid(
    bars: Sequence[dict],
    *,
    decision_i: int,
    peak: float,
    hard: float,
    exit_reason: str,
) -> ExitResult:
    nxt_i = int(decision_i) + 1
    if nxt_i >= len(bars):
        return ExitResult(
            filled=False,
            reason=REASON_NO_NEXT_BAR,
            decision_ts=bars[decision_i].get("ts"),
            exit_reason=exit_reason,
            peak_px=peak,
            hard_stop=hard,
            stop_at_decision=hard,
        )
    nxt = bars[nxt_i]
    hi = _ohlc(nxt, "high")
    lo = _ohlc(nxt, "low")
    if hi is None or lo is None:
        return ExitResult(
            filled=False,
            reason=REASON_NO_NEXT_BAR,
            decision_ts=bars[decision_i].get("ts"),
            peak_px=peak,
            hard_stop=hard,
        )
    return ExitResult(
        filled=True,
        reason=REASON_FILLED,
        sell_px=float(bar_mid(hi, lo)),
        decision_ts=bars[decision_i].get("ts"),
        exec_bar_ts=nxt.get("ts"),
        exit_reason=exit_reason,
        peak_px=peak,
        hard_stop=hard,
        stop_at_decision=hard,
    )


def _ohlc(bar: dict, key: str) -> Optional[float]:
    try:
        val = float(bar[key])
    except (KeyError, TypeError, ValueError):
        return None
    if val != val:
        return None
    return val


def load_symbol_sectors(rows: Iterable[Mapping[str, Any]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for row in rows:
        sym = str(row.get("symbol") or "").upper().strip()
        sec = str(row.get("sector") or "").strip()
        if not sym:
            continue
        if sym not in out or (sec and out[sym] in ("", "Unknown")):
            out[sym] = sec or "Unknown"
    return out
