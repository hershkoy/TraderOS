"""Patrick Nil P/B range claim-check on daily bars.

Objective translation of the pasted Nil sketch (15m+4h live) onto 1d:

- Impulse (P up / B down) then a horizontal consolidation ("fair value").
- Range valid only after >=3 boundary touches.
- Weekly close vs SMA50 is the HTF trend filter (4h analog).
- Three books: range fade, confirmed breakout, fake breakout.
- Signal on completed daily close; fill next session open; structure stops.

Not a promote until E/PF/geo, friction, and occupancy vs SPY clear the gate.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from utils.db.timescaledb_client import get_timescaledb_client
from utils.research.vwap_ema_reclaim import drop_top_n_expectancy, max_concurrent
from utils.research.weekly_deepvue import occupancy_take

BOOK_RANGE = "range_fade"
BOOK_CONFIRMED = "confirmed_breakout"
BOOK_FAKE = "fake_breakout"

SIDE_LONG = "long"
SIDE_SHORT = "short"

PATTERN_P = "P"
PATTERN_B = "B"

EXIT_TARGET = "target"
EXIT_STOP = "stop"
EXIT_TIME = "time_stop"

HTF_BULL = "bull"
HTF_BEAR = "bear"
HTF_NEUTRAL = "neutral"

CAP_SMALL = "small"
CAP_MID = "mid"
CAP_LARGE = "large"

SMALL_MAX = 2_000_000_000.0
MID_MAX = 10_000_000_000.0

YEAR_BUCKETS: Sequence[Tuple[str, str, str]] = (
    ("2018-2019", "2018-01-01", "2019-12-31"),
    ("2020-2021", "2020-01-01", "2021-12-31"),
    ("2022-2023", "2022-01-01", "2023-12-31"),
    ("2024-2026", "2024-01-01", "2026-12-31"),
)

CLAIMED: Dict[str, Any] = {
    "source": "Patrick Nil / Robins Cup summary (user paste 2026-10-31)",
    "title": "P/B fair-value ranges",
    "timeframe": "1d (weekly HTF)",
    "notes": (
        "Live process uses 15m decisions + 4h trend. This claim-check maps "
        "structure to daily bars and HTF to weekly SMA50. Three entry books "
        "scored separately on small/mid/large SEC mkt_cap panels."
    ),
}

TRADE_COLUMNS = (
    "stock",
    "cap_bucket",
    "book",
    "pattern",
    "side",
    "htf",
    "buy_date",
    "buy_ts",
    "buy_price",
    "sell_date",
    "sell_ts",
    "sell_price",
    "gain_pct",
    "stop_px",
    "target_px",
    "range_high",
    "range_low",
    "range_width",
    "touches",
    "exit_reason",
    "hold_days",
    "rr_planned",
)


@dataclass(frozen=True)
class NilParams:
    impulse_lookback: int = 10
    impulse_atr_mult: float = 1.5
    atr_len: int = 14
    min_range_bars: int = 5
    max_range_bars: int = 40
    max_range_atr: float = 2.0
    min_touches: int = 3
    touch_tol_pct: float = 0.15
    stop_buffer_pct: float = 0.05
    weekly_sma: int = 50
    htf_neutral_pct: float = 0.5
    time_stop_bars: int = 40
    near_extreme_frac: float = 0.25


def params_dict(params: NilParams) -> Dict[str, Any]:
    return asdict(params)


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, length: int) -> np.ndarray:
    n = len(close)
    tr = np.full(n, np.nan, dtype=float)
    atr = np.full(n, np.nan, dtype=float)
    if n == 0:
        return atr
    tr[0] = float(high[0] - low[0])
    for i in range(1, n):
        hl = float(high[i] - low[i])
        hc = abs(float(high[i] - close[i - 1]))
        lc = abs(float(low[i] - close[i - 1]))
        tr[i] = max(hl, hc, lc)
    L = max(1, int(length))
    if n < L:
        return atr
    atr[L - 1] = float(np.nanmean(tr[:L]))
    alpha = 1.0 / float(L)
    for i in range(L, n):
        atr[i] = atr[i - 1] * (1.0 - alpha) + tr[i] * alpha
    return atr


def _prepare(df: pd.DataFrame) -> Dict[str, Any]:
    d = df.copy()
    if "ts" not in d.columns:
        if isinstance(d.index, pd.DatetimeIndex):
            d = d.reset_index().rename(columns={d.index.name or "index": "ts"})
        else:
            raise ValueError("OHLCV needs ts or DatetimeIndex")
    d["ts"] = pd.to_datetime(d["ts"], utc=True, errors="coerce")
    if getattr(d["ts"].dt, "tz", None) is not None:
        d["ts"] = d["ts"].dt.tz_convert("UTC").dt.tz_localize(None)
    d = d.dropna(subset=["ts"]).sort_values("ts")
    d = d[~d["ts"].duplicated(keep="last")].reset_index(drop=True)
    for col in ("open", "high", "low", "close"):
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    if d.empty:
        raise ValueError("empty OHLCV")
    o = d["open"].to_numpy(dtype=float)
    h = d["high"].to_numpy(dtype=float)
    lo = d["low"].to_numpy(dtype=float)
    c = d["close"].to_numpy(dtype=float)
    ts = pd.DatetimeIndex(d["ts"]).normalize()
    return {
        "open": o,
        "high": h,
        "low": lo,
        "close": c,
        "ts": ts,
        "n": len(c),
    }


def weekly_htf_bias(
    ts: pd.DatetimeIndex,
    close: np.ndarray,
    *,
    sma_len: int = 50,
    neutral_pct: float = 0.5,
) -> np.ndarray:
    """Causal weekly SMA bias mapped onto each daily bar (prior completed week)."""
    n = len(close)
    out = np.array([HTF_NEUTRAL] * n, dtype=object)
    if n < 5:
        return out
    s = pd.Series(close, index=pd.DatetimeIndex(ts))
    weekly = s.resample("W-FRI").last().dropna()
    if len(weekly) < int(sma_len) + 1:
        return out
    sma = weekly.rolling(int(sma_len), min_periods=int(sma_len)).mean()
    # As-of each daily bar use last *completed* weekly close/SMA (shift 1).
    w_close = weekly.shift(1)
    w_sma = sma.shift(1)
    wdf = pd.DataFrame({"w_close": w_close, "w_sma": w_sma}).dropna()
    if wdf.empty:
        return out
    # Forward-fill weekly values onto daily calendar via asof merge.
    daily = pd.DataFrame({"ts": pd.DatetimeIndex(ts), "i": np.arange(n)})
    daily = daily.sort_values("ts")
    wdf = wdf.reset_index().rename(columns={"index": "wts"})
    if "wts" not in wdf.columns:
        wdf = wdf.rename(columns={wdf.columns[0]: "wts"})
    merged = pd.merge_asof(
        daily,
        wdf.sort_values("wts"),
        left_on="ts",
        right_on="wts",
        direction="backward",
    )
    tol = float(neutral_pct) / 100.0
    for _, row in merged.iterrows():
        i = int(row["i"])
        wc = row.get("w_close")
        ws = row.get("w_sma")
        if wc is None or ws is None or not np.isfinite(wc) or not np.isfinite(ws) or ws == 0:
            continue
        rel = (float(wc) - float(ws)) / float(ws)
        if rel > tol:
            out[i] = HTF_BULL
        elif rel < -tol:
            out[i] = HTF_BEAR
        else:
            out[i] = HTF_NEUTRAL
    return out


def _touch_tol(range_high: float, range_low: float, touch_tol_pct: float) -> float:
    mid = 0.5 * (range_high + range_low)
    width = max(range_high - range_low, 1e-9)
    return max(mid * float(touch_tol_pct) / 100.0, 0.05 * width)


def _touches_high(high: float, range_high: float, tol: float) -> bool:
    return abs(float(high) - float(range_high)) <= tol or (
        float(high) >= float(range_high) - tol and float(high) <= float(range_high) + tol
    )


def _touches_low(low: float, range_low: float, tol: float) -> bool:
    return abs(float(low) - float(range_low)) <= tol or (
        float(low) <= float(range_low) + tol and float(low) >= float(range_low) - tol
    )


def detect_impulse_at(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    atr: np.ndarray,
    end_i: int,
    params: NilParams,
) -> Optional[str]:
    """Return P/B if an impulse ends at ``end_i``, else None.

    Requires a pause bar after the thrust (``end_i + 1`` does not make a new
    extreme) so mid-impulse bars are not treated as completed P/B bases.
    """
    L = int(params.impulse_lookback)
    if end_i < L or end_i + 1 >= len(close):
        return None
    a = atr[end_i]
    if not np.isfinite(a) or a <= 0:
        return None
    start = end_i - L
    net = float(close[end_i] - close[start])
    hurdle = float(params.impulse_atr_mult) * float(a)
    win_high = float(np.nanmax(high[start : end_i + 1]))
    win_low = float(np.nanmin(low[start : end_i + 1]))
    span = max(win_high - win_low, 1e-9)
    near = float(params.near_extreme_frac) * span
    nxt = float(close[end_i + 1])
    if net >= hurdle and float(close[end_i]) >= win_high - near:
        # Pause: next close does not push a fresh high beyond the thrust close.
        if nxt <= float(close[end_i]) + 1e-9:
            return PATTERN_P
    if net <= -hurdle and float(close[end_i]) <= win_low + near:
        if nxt >= float(close[end_i]) - 1e-9:
            return PATTERN_B
    return None


def count_boundary_touches(
    high: np.ndarray,
    low: np.ndarray,
    start_i: int,
    end_i: int,
    range_high: float,
    range_low: float,
    touch_tol_pct: float,
) -> int:
    """Count bars in [start_i, end_i] that tag either rail (max 1 per bar)."""
    tol = _touch_tol(range_high, range_low, touch_tol_pct)
    n = 0
    for i in range(int(start_i), int(end_i) + 1):
        th = _touches_high(float(high[i]), range_high, tol)
        tl = _touches_low(float(low[i]), range_low, tol)
        if th or tl:
            n += 1
    return n


def _htf_allows(side: str, htf: str) -> bool:
    if htf == HTF_NEUTRAL:
        return True
    if side == SIDE_LONG and htf == HTF_BULL:
        return True
    if side == SIDE_SHORT and htf == HTF_BEAR:
        return True
    # Mild allowance: bullish HTF still allows shorts on fake upside break (Nil favorite);
    # keep strict for range fade / confirmed. Caller can override.
    return False


def _resolve_exit(
    side: str,
    entry_i: int,
    entry_px: float,
    stop_px: float,
    target_px: float,
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    ts: pd.DatetimeIndex,
    time_stop_bars: int,
) -> Dict[str, Any]:
    n = len(close)
    for j in range(entry_i, n):
        hi = float(high[j])
        lo = float(low[j])
        # Same-bar entry: use post-open path; stop wins if both hit.
        if side == SIDE_LONG:
            hit_stop = lo <= stop_px
            hit_tgt = hi >= target_px
            if hit_stop and hit_tgt:
                sell = float(stop_px)
                reason = EXIT_STOP
            elif hit_stop:
                sell = float(stop_px)
                reason = EXIT_STOP
            elif hit_tgt:
                sell = float(target_px)
                reason = EXIT_TARGET
            elif j - entry_i >= int(time_stop_bars):
                sell = float(close[j])
                reason = EXIT_TIME
            else:
                continue
        else:
            hit_stop = hi >= stop_px
            hit_tgt = lo <= target_px
            if hit_stop and hit_tgt:
                sell = float(stop_px)
                reason = EXIT_STOP
            elif hit_stop:
                sell = float(stop_px)
                reason = EXIT_STOP
            elif hit_tgt:
                sell = float(target_px)
                reason = EXIT_TARGET
            elif j - entry_i >= int(time_stop_bars):
                sell = float(close[j])
                reason = EXIT_TIME
            else:
                continue
        if side == SIDE_LONG:
            gain = (sell / entry_px - 1.0) * 100.0
        else:
            gain = (entry_px / sell - 1.0) * 100.0 if sell > 0 else 0.0
        return {
            "sell_i": j,
            "sell_price": sell,
            "sell_ts": ts[j],
            "sell_date": str(ts[j].date()),
            "exit_reason": reason,
            "gain_pct": round(float(gain), 6),
            "hold_days": int(j - entry_i),
        }
    # Flatten last close
    j = n - 1
    sell = float(close[j])
    if side == SIDE_LONG:
        gain = (sell / entry_px - 1.0) * 100.0
    else:
        gain = (entry_px / sell - 1.0) * 100.0 if sell > 0 else 0.0
    return {
        "sell_i": j,
        "sell_price": sell,
        "sell_ts": ts[j],
        "sell_date": str(ts[j].date()),
        "exit_reason": EXIT_TIME,
        "gain_pct": round(float(gain), 6),
        "hold_days": int(j - entry_i),
    }


def _trade_row(
    *,
    symbol: str,
    cap_bucket: str,
    book: str,
    pattern: str,
    side: str,
    htf: str,
    entry_i: int,
    entry_px: float,
    stop_px: float,
    target_px: float,
    range_high: float,
    range_low: float,
    touches: int,
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    ts: pd.DatetimeIndex,
    time_stop_bars: int,
) -> dict:
    risk = abs(entry_px - stop_px)
    reward = abs(target_px - entry_px)
    rr = (reward / risk) if risk > 1e-12 else None
    ex = _resolve_exit(
        side,
        entry_i,
        entry_px,
        stop_px,
        target_px,
        open_,
        high,
        low,
        close,
        ts,
        time_stop_bars,
    )
    return {
        "stock": str(symbol).upper(),
        "cap_bucket": cap_bucket,
        "book": book,
        "pattern": pattern,
        "side": side,
        "htf": htf,
        "buy_date": str(ts[entry_i].date()),
        "buy_ts": ts[entry_i],
        "buy_price": round(float(entry_px), 6),
        "sell_date": ex["sell_date"],
        "sell_ts": ex["sell_ts"],
        "sell_price": round(float(ex["sell_price"]), 6),
        "gain_pct": ex["gain_pct"],
        "stop_px": round(float(stop_px), 6),
        "target_px": round(float(target_px), 6),
        "range_high": round(float(range_high), 6),
        "range_low": round(float(range_low), 6),
        "range_width": round(float(range_high - range_low), 6),
        "touches": int(touches),
        "exit_reason": ex["exit_reason"],
        "hold_days": ex["hold_days"],
        "rr_planned": None if rr is None else round(float(rr), 3),
        "_sell_i": ex["sell_i"],
    }


def collect_symbol_trades(
    df: pd.DataFrame,
    *,
    symbol: str,
    cap_bucket: str = "",
    params: Optional[NilParams] = None,
    books: Optional[Sequence[str]] = None,
) -> List[dict]:
    """Walk one symbol; emit trades for requested books (default all three)."""
    params = params or NilParams()
    want = set(books or (BOOK_RANGE, BOOK_CONFIRMED, BOOK_FAKE))
    try:
        d = _prepare(df)
    except ValueError:
        return []
    o = d["open"]
    h = d["high"]
    lo = d["low"]
    c = d["close"]
    ts = d["ts"]
    n = d["n"]
    atr = _atr(h, lo, c, params.atr_len)
    htf_arr = weekly_htf_bias(
        ts, c, sma_len=params.weekly_sma, neutral_pct=params.htf_neutral_pct
    )

    trades: List[dict] = []
    busy_until = -1
    # At bar i we may confirm an impulse that ended at i-1 (pause = bar i).
    i = int(params.impulse_lookback) + int(params.atr_len) + 1
    while i < n - 2:
        if i <= busy_until:
            i += 1
            continue
        pattern = detect_impulse_at(c, h, lo, atr, i - 1, params)
        if pattern is None:
            i += 1
            continue

        # Consolidation starts at the pause bar (i), causally known now.
        start_r = i
        impulse_i = i - 1
        if start_r + int(params.min_range_bars) >= n:
            break
        a0 = atr[impulse_i]
        if not np.isfinite(a0) or a0 <= 0:
            i += 1
            continue
        max_w = float(params.max_range_atr) * float(a0)
        end_seed = start_r + int(params.min_range_bars) - 1
        rh = float(np.nanmax(h[start_r : end_seed + 1]))
        rl = float(np.nanmin(lo[start_r : end_seed + 1]))
        if rh - rl > max_w or rh <= rl:
            i += 1
            continue

        j = end_seed
        valid_at: Optional[int] = None
        touches = 0
        while j < n - 1 and (j - start_r + 1) <= int(params.max_range_bars):
            # Expand range extremes only while still building / within width.
            cand_h = max(rh, float(h[j]))
            cand_l = min(rl, float(lo[j]))
            if cand_h - cand_l > max_w:
                if valid_at is not None:
                    break
                # Abort unvalidated range.
                valid_at = None
                break
            rh, rl = cand_h, cand_l
            touches = count_boundary_touches(
                h, lo, start_r, j, rh, rl, params.touch_tol_pct
            )
            if touches >= int(params.min_touches) and valid_at is None:
                valid_at = j
                break
            j += 1

        if valid_at is None:
            # Do not re-arm the same impulse on the next bar.
            i = max(i + 1, start_r)
            continue

        # Active range from valid_at; scan signals until breakout resolves or max bars.
        range_end = min(n - 2, start_r + int(params.max_range_bars) - 1)
        tol = _touch_tol(rh, rl, params.touch_tol_pct)
        buf = float(params.stop_buffer_pct) / 100.0
        width = rh - rl

        # Breakout state machine shared across confirmed + fake.
        bo_side: Optional[str] = None  # direction of first close outside
        bo_extreme: Optional[float] = None
        bo_bar: Optional[int] = None
        retested = False
        confirmed = False
        fake_done = False
        range_fade_used_low = False
        range_fade_used_high = False

        k = int(valid_at)
        while k < range_end:
            if k <= busy_until:
                k += 1
                continue
            htf = str(htf_arr[k])
            close_k = float(c[k])
            high_k = float(h[k])
            low_k = float(lo[k])

            # --- Book 1: range fade on rail touch (HTF aligned) ---
            if BOOK_RANGE in want and k + 1 < n:
                if (
                    not range_fade_used_low
                    and _touches_low(low_k, rl, tol)
                    and close_k >= rl - tol
                ):
                    side = SIDE_LONG
                    if _htf_allows(side, htf) or htf == HTF_NEUTRAL:
                        entry_i = k + 1
                        entry_px = float(o[entry_i])
                        stop_px = rl * (1.0 - buf)
                        target_px = rh
                        if entry_px > stop_px:
                            tr = _trade_row(
                                symbol=symbol,
                                cap_bucket=cap_bucket,
                                book=BOOK_RANGE,
                                pattern=pattern,
                                side=side,
                                htf=htf,
                                entry_i=entry_i,
                                entry_px=entry_px,
                                stop_px=stop_px,
                                target_px=target_px,
                                range_high=rh,
                                range_low=rl,
                                touches=touches,
                                open_=o,
                                high=h,
                                low=lo,
                                close=c,
                                ts=ts,
                                time_stop_bars=params.time_stop_bars,
                            )
                            trades.append(tr)
                            busy_until = int(tr.pop("_sell_i"))
                            range_fade_used_low = True
                            k = busy_until + 1
                            continue
                if (
                    not range_fade_used_high
                    and _touches_high(high_k, rh, tol)
                    and close_k <= rh + tol
                ):
                    side = SIDE_SHORT
                    if _htf_allows(side, htf) or htf == HTF_NEUTRAL:
                        entry_i = k + 1
                        entry_px = float(o[entry_i])
                        stop_px = rh * (1.0 + buf)
                        target_px = rl
                        if entry_px < stop_px:
                            tr = _trade_row(
                                symbol=symbol,
                                cap_bucket=cap_bucket,
                                book=BOOK_RANGE,
                                pattern=pattern,
                                side=side,
                                htf=htf,
                                entry_i=entry_i,
                                entry_px=entry_px,
                                stop_px=stop_px,
                                target_px=target_px,
                                range_high=rh,
                                range_low=rl,
                                touches=touches,
                                open_=o,
                                high=h,
                                low=lo,
                                close=c,
                                ts=ts,
                                time_stop_bars=params.time_stop_bars,
                            )
                            trades.append(tr)
                            busy_until = int(tr.pop("_sell_i"))
                            range_fade_used_high = True
                            k = busy_until + 1
                            continue

            # --- Breakout / fake path ---
            outside_up = close_k > rh + tol
            outside_dn = close_k < rl - tol
            inside = (close_k <= rh + tol) and (close_k >= rl - tol)

            if bo_side is None:
                if outside_up:
                    bo_side = "up"
                    bo_extreme = high_k
                    bo_bar = k
                    retested = False
                    confirmed = False
                elif outside_dn:
                    bo_side = "down"
                    bo_extreme = low_k
                    bo_bar = k
                    retested = False
                    confirmed = False
                k += 1
                continue

            # Already saw a breakout bar.
            if bo_side == "up":
                bo_extreme = max(float(bo_extreme or high_k), high_k)
                # Fake: close back inside after breakout (favorite).
                if (
                    BOOK_FAKE in want
                    and not fake_done
                    and inside
                    and bo_bar is not None
                    and k > bo_bar
                    and k + 1 < n
                ):
                    side = SIDE_SHORT
                    entry_i = k + 1
                    entry_px = float(o[entry_i])
                    stop_px = float(bo_extreme) * (1.0 + buf)
                    target_px = rl
                    if entry_px < stop_px:
                        tr = _trade_row(
                            symbol=symbol,
                            cap_bucket=cap_bucket,
                            book=BOOK_FAKE,
                            pattern=pattern,
                            side=side,
                            htf=htf,
                            entry_i=entry_i,
                            entry_px=entry_px,
                            stop_px=stop_px,
                            target_px=target_px,
                            range_high=rh,
                            range_low=rl,
                            touches=touches,
                            open_=o,
                            high=h,
                            low=lo,
                            close=c,
                            ts=ts,
                            time_stop_bars=params.time_stop_bars,
                        )
                        trades.append(tr)
                        busy_until = int(tr.pop("_sell_i"))
                        fake_done = True
                        # Range consumed after fake resolution.
                        i = max(i + 1, busy_until + 1)
                        break
                # Confirmed path: retest then close outside again.
                if BOOK_CONFIRMED in want and not confirmed:
                    if not retested and (
                        _touches_high(high_k, rh, tol) or abs(close_k - rh) <= tol
                    ):
                        retested = True
                    elif retested and outside_up and k + 1 < n:
                        side = SIDE_LONG
                        if _htf_allows(side, htf) or htf != HTF_BEAR:
                            entry_i = k + 1
                            entry_px = float(o[entry_i])
                            stop_px = min(low_k, rl) * (1.0 - buf)
                            target_px = entry_px + width
                            if entry_px > stop_px:
                                tr = _trade_row(
                                    symbol=symbol,
                                    cap_bucket=cap_bucket,
                                    book=BOOK_CONFIRMED,
                                    pattern=pattern,
                                    side=side,
                                    htf=htf,
                                    entry_i=entry_i,
                                    entry_px=entry_px,
                                    stop_px=stop_px,
                                    target_px=target_px,
                                    range_high=rh,
                                    range_low=rl,
                                    touches=touches,
                                    open_=o,
                                    high=h,
                                    low=lo,
                                    close=c,
                                    ts=ts,
                                    time_stop_bars=params.time_stop_bars,
                                )
                                trades.append(tr)
                                busy_until = int(tr.pop("_sell_i"))
                                confirmed = True
                                i = max(i + 1, busy_until + 1)
                                break
            else:  # down breakout
                bo_extreme = min(float(bo_extreme or low_k), low_k)
                if (
                    BOOK_FAKE in want
                    and not fake_done
                    and inside
                    and bo_bar is not None
                    and k > bo_bar
                    and k + 1 < n
                ):
                    side = SIDE_LONG
                    entry_i = k + 1
                    entry_px = float(o[entry_i])
                    stop_px = float(bo_extreme) * (1.0 - buf)
                    target_px = rh
                    if entry_px > stop_px:
                        tr = _trade_row(
                            symbol=symbol,
                            cap_bucket=cap_bucket,
                            book=BOOK_FAKE,
                            pattern=pattern,
                            side=side,
                            htf=htf,
                            entry_i=entry_i,
                            entry_px=entry_px,
                            stop_px=stop_px,
                            target_px=target_px,
                            range_high=rh,
                            range_low=rl,
                            touches=touches,
                            open_=o,
                            high=h,
                            low=lo,
                            close=c,
                            ts=ts,
                            time_stop_bars=params.time_stop_bars,
                        )
                        trades.append(tr)
                        busy_until = int(tr.pop("_sell_i"))
                        fake_done = True
                        i = max(i + 1, busy_until + 1)
                        break
                if BOOK_CONFIRMED in want and not confirmed:
                    if not retested and (
                        _touches_low(low_k, rl, tol) or abs(close_k - rl) <= tol
                    ):
                        retested = True
                    elif retested and outside_dn and k + 1 < n:
                        side = SIDE_SHORT
                        if _htf_allows(side, htf) or htf != HTF_BULL:
                            entry_i = k + 1
                            entry_px = float(o[entry_i])
                            stop_px = max(high_k, rh) * (1.0 + buf)
                            target_px = entry_px - width
                            if entry_px < stop_px:
                                tr = _trade_row(
                                    symbol=symbol,
                                    cap_bucket=cap_bucket,
                                    book=BOOK_CONFIRMED,
                                    pattern=pattern,
                                    side=side,
                                    htf=htf,
                                    entry_i=entry_i,
                                    entry_px=entry_px,
                                    stop_px=stop_px,
                                    target_px=target_px,
                                    range_high=rh,
                                    range_low=rl,
                                    touches=touches,
                                    open_=o,
                                    high=h,
                                    low=lo,
                                    close=c,
                                    ts=ts,
                                    time_stop_bars=params.time_stop_bars,
                                )
                                trades.append(tr)
                                busy_until = int(tr.pop("_sell_i"))
                                confirmed = True
                                i = max(i + 1, busy_until + 1)
                                break
            k += 1
        else:
            # Finished scanning this range without consuming via BO trade.
            i = max(i + 1, int(valid_at) + 1)
            continue
        # break from inner while advanced i already
        continue

    # Strip internal keys if any remain
    for tr in trades:
        tr.pop("_sell_i", None)
    return trades


def load_latest_mkt_caps() -> pd.DataFrame:
    """Latest non-null mkt_cap per symbol from fundamentals_quarterly."""
    db = get_timescaledb_client()
    if not db.connect():
        raise RuntimeError("TimescaleDB connection failed")
    sql = """
    SELECT DISTINCT ON (symbol)
           symbol, mkt_cap, filed, period_end
    FROM fundamentals_quarterly
    WHERE mkt_cap IS NOT NULL
      AND mkt_cap > 0
      AND exclude_sector IS NOT TRUE
      AND period_end <= CURRENT_DATE
    ORDER BY symbol, filed DESC NULLS LAST, period_end DESC
    """
    rows = db.execute_query(sql)
    cols = ["symbol", "mkt_cap", "filed", "period_end"]
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows, columns=cols)
    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["mkt_cap"] = pd.to_numeric(df["mkt_cap"], errors="coerce")
    return df.dropna(subset=["mkt_cap"])


def load_adv_90d(min_bars: int = 20) -> pd.DataFrame:
    """Trailing ~90d average dollar volume from ALPACA daily bars."""
    db = get_timescaledb_client()
    if not db.ensure_connection():
        raise RuntimeError("TimescaleDB connection failed")
    sql = """
    SELECT symbol,
           AVG(close * volume) AS adv,
           COUNT(*) AS n_bars
    FROM market_data
    WHERE provider = 'ALPACA'
      AND timeframe = '1d'
      AND ts >= NOW() - INTERVAL '90 days'
    GROUP BY symbol
    HAVING COUNT(*) >= %s
    ORDER BY adv DESC NULLS LAST
    """
    rows = db.execute_query(sql, (int(min_bars),))
    cols = ["symbol", "adv", "n_bars"]
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows, columns=cols)
    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["adv"] = pd.to_numeric(df["adv"], errors="coerce")
    return df.dropna(subset=["adv"])


def cap_bucket_for(mkt_cap: float) -> str:
    v = float(mkt_cap)
    if v < SMALL_MAX:
        return CAP_SMALL
    if v < MID_MAX:
        return CAP_MID
    return CAP_LARGE


def build_cap_universe(
    *,
    n_per_bucket: int = 33,
    min_adv: float = 1_000_000.0,
) -> pd.DataFrame:
    """Pick ~n_per_bucket names per small/mid/large by SEC mkt_cap, ranked by ADV.

    If a SEC bucket cannot fill, top up from ADV-ranked residual names and set
    ``cap_proxy='adv'`` (never silently re-label SEC buckets).
    """
    caps = load_latest_mkt_caps()
    adv = load_adv_90d()
    if caps.empty and adv.empty:
        return pd.DataFrame(
            columns=["symbol", "cap_bucket", "mkt_cap", "adv", "cap_proxy"]
        )

    merged = adv.merge(caps, on="symbol", how="left")
    merged = merged.loc[merged["adv"] >= float(min_adv)].copy()
    merged["cap_bucket"] = pd.Series(pd.NA, index=merged.index, dtype="object")
    has_cap = merged["mkt_cap"].notna()
    merged.loc[has_cap, "cap_bucket"] = (
        merged.loc[has_cap, "mkt_cap"].map(cap_bucket_for).astype(object)
    )
    merged["cap_proxy"] = "sec"
    merged.loc[~has_cap, "cap_proxy"] = "adv"

    want = max(1, int(n_per_bucket))
    picked: List[pd.DataFrame] = []
    used: set = set()
    for bucket in (CAP_SMALL, CAP_MID, CAP_LARGE):
        pool = merged.loc[
            (merged["cap_bucket"] == bucket) & (~merged["symbol"].isin(used))
        ].sort_values("adv", ascending=False)
        take = pool.head(want).copy()
        picked.append(take)
        used.update(take["symbol"].tolist())

    # Top up short buckets from ADV residual; label cap_proxy=adv and assign
    # bucket by ADV tercile of the residual (explicit proxy, not SEC).
    for bucket, frame in zip((CAP_SMALL, CAP_MID, CAP_LARGE), picked):
        short = want - len(frame)
        if short <= 0:
            continue
        residual = merged.loc[~merged["symbol"].isin(used)].sort_values(
            "adv", ascending=False
        )
        if residual.empty:
            continue
        # Map residual ADV terciles: lowest third -> small, mid, large.
        q = residual["adv"].quantile([1 / 3, 2 / 3])
        lo, hi = float(q.iloc[0]), float(q.iloc[1])

        def _proxy_bucket(a: float) -> str:
            if a <= lo:
                return CAP_SMALL
            if a <= hi:
                return CAP_MID
            return CAP_LARGE

        residual = residual.copy()
        residual["cap_bucket"] = residual["adv"].map(_proxy_bucket)
        residual["cap_proxy"] = "adv"
        add = residual.loc[residual["cap_bucket"] == bucket].head(short)
        if add.empty:
            # Last resort: take next ADV names and force-label as this bucket proxy.
            add = residual.head(short).copy()
            add["cap_bucket"] = bucket
            add["cap_proxy"] = "adv"
        picked.append(add)
        used.update(add["symbol"].tolist())

    out = pd.concat(picked, ignore_index=True) if picked else pd.DataFrame()
    if out.empty:
        return out
    out = out.drop_duplicates(subset=["symbol"], keep="first")
    cols = ["symbol", "cap_bucket", "mkt_cap", "adv", "cap_proxy", "filed", "period_end"]
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    return out[cols].sort_values(["cap_bucket", "adv"], ascending=[True, False]).reset_index(
        drop=True
    )


def apply_friction(trades: pd.DataFrame, round_trip_pct: float) -> pd.DataFrame:
    out = trades.copy()
    out["gain_pct_net"] = pd.to_numeric(out["gain_pct"], errors="coerce") - float(
        round_trip_pct
    )
    return out


def summarize_by_year(
    trades: pd.DataFrame,
    *,
    gain_col: str = "gain_pct",
    buckets: Sequence[Tuple[str, str, str]] = YEAR_BUCKETS,
) -> pd.DataFrame:
    from utils.research.last_15m_volume_geometry import book_stats

    rows: List[dict] = []
    if trades is None or trades.empty or "buy_date" not in trades.columns:
        return pd.DataFrame(rows)
    bd = pd.to_datetime(trades["buy_date"], errors="coerce")
    for name, start, end in buckets:
        m = (bd >= pd.Timestamp(start)) & (bd <= pd.Timestamp(end))
        s = book_stats(pd.to_numeric(trades.loc[m, gain_col], errors="coerce"))
        rows.append(
            {
                "bucket": name,
                "n_trades": s.get("n"),
                "win_rate_pct": s.get("win_rate_pct"),
                "expectancy_pct": s.get("expectancy_pct"),
                "profit_factor": s.get("profit_factor"),
            }
        )
    s_full = book_stats(pd.to_numeric(trades[gain_col], errors="coerce"))
    rows.append(
        {
            "bucket": "FULL",
            "n_trades": s_full.get("n"),
            "win_rate_pct": s_full.get("win_rate_pct"),
            "expectancy_pct": s_full.get("expectancy_pct"),
            "profit_factor": s_full.get("profit_factor"),
        }
    )
    return pd.DataFrame(rows)


def occupancy_equity_from_trades(
    trades: pd.DataFrame,
    close_panel: pd.DataFrame,
    *,
    max_open: int = 8,
) -> Tuple[pd.Series, pd.Series]:
    """Equal-slot daily equity (cash earns 0). Long and short via signed notional.

    Each new fill deploys ``1/max_open`` of current equity. Shorts mark as
    ``notional * (2*entry - px) / entry`` so price rises hurt equity.
    """
    if trades is None or trades.empty or close_panel is None or close_panel.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    close = close_panel.astype(float).copy()
    if not isinstance(close.index, pd.DatetimeIndex):
        close.index = pd.to_datetime(close.index)
    close.index = pd.DatetimeIndex(close.index).tz_localize(None).normalize()
    close = close[~close.index.duplicated(keep="last")].sort_index()
    close.columns = [str(c).upper() for c in close.columns]

    t = trades.copy()
    sym_col = "stock" if "stock" in t.columns else "symbol"
    t["symbol"] = t[sym_col].astype(str).str.upper()
    t["buy_date"] = pd.to_datetime(t["buy_date"]).dt.tz_localize(None).dt.normalize()
    t["sell_date"] = pd.to_datetime(t["sell_date"]).dt.tz_localize(None).dt.normalize()
    t["buy_price"] = pd.to_numeric(t["buy_price"], errors="coerce")
    t["sell_price"] = pd.to_numeric(t["sell_price"], errors="coerce")
    if "side" not in t.columns:
        t["side"] = SIDE_LONG

    by_buy: Dict[pd.Timestamp, List[int]] = {}
    for idx, row in t.iterrows():
        by_buy.setdefault(pd.Timestamp(row["buy_date"]).normalize(), []).append(int(idx))

    cash = 1.0
    positions: Dict[str, dict] = {}
    eq_pts: List[Tuple[pd.Timestamp, float]] = []
    inv_pts: List[Tuple[pd.Timestamp, float]] = []
    alloc = 1.0 / float(max(1, int(max_open)))

    def _pos_value(pos: dict, px: float) -> float:
        notional = float(pos["notional"])
        entry = float(pos["entry_px"])
        if entry <= 0:
            return notional
        if pos["side"] == SIDE_SHORT:
            return notional * (2.0 * entry - float(px)) / entry
        return notional * float(px) / entry

    def _mark() -> float:
        eq = cash
        for pos in positions.values():
            eq += _pos_value(pos, float(pos["last_px"]))
        return eq

    for dt in close.index:
        dtn = pd.Timestamp(dt).normalize()
        for sym in [s for s, p in positions.items() if p["sell_date"] < dtn]:
            pos = positions.pop(sym)
            cash += _pos_value(pos, float(pos["sell_px"]))

        if dtn in by_buy:
            for idx in by_buy[dtn]:
                if len(positions) >= int(max_open):
                    break
                row = t.loc[idx]
                sym = str(row["symbol"])
                if sym in positions:
                    continue
                entry_px = float(row["buy_price"])
                if not np.isfinite(entry_px) or entry_px <= 0:
                    continue
                eq_now = _mark()
                spend = min(eq_now * alloc, cash)
                if spend <= 0:
                    continue
                cash -= spend
                last_px = entry_px
                if sym in close.columns and pd.notna(close.loc[dt, sym]):
                    last_px = float(close.loc[dt, sym])
                positions[sym] = {
                    "notional": spend,
                    "entry_px": entry_px,
                    "last_px": last_px,
                    "sell_date": pd.Timestamp(row["sell_date"]).normalize(),
                    "sell_px": float(row["sell_price"]),
                    "side": str(row.get("side") or SIDE_LONG),
                }

        for sym in [s for s, p in positions.items() if p["sell_date"] <= dtn]:
            pos = positions.pop(sym)
            cash += _pos_value(pos, float(pos["sell_px"]))

        for sym, pos in positions.items():
            if sym in close.columns and pd.notna(close.loc[dt, sym]):
                pos["last_px"] = float(close.loc[dt, sym])

        eq_pts.append((dt, _mark()))
        inv_pts.append((dt, 1.0 if positions else 0.0))

    if positions:
        for pos in list(positions.values()):
            cash += _pos_value(pos, float(pos["sell_px"]))
        positions.clear()
        if eq_pts:
            eq_pts[-1] = (eq_pts[-1][0], cash)

    eq = pd.Series({d: v for d, v in eq_pts}).sort_index()
    inv = pd.Series({d: v for d, v in inv_pts}).sort_index()
    if not eq.empty and float(eq.iloc[0]) != 0:
        eq = eq / float(eq.iloc[0])
    return eq, inv


def sitout_after_loss_streak(
    trades: Sequence[Mapping[str, Any]],
    *,
    max_losses: int = 3,
) -> List[dict]:
    """Per-symbol hygiene: after max_losses consecutive losses, skip next signal."""
    if not trades:
        return []
    ordered = sorted(trades, key=lambda r: (r["buy_ts"], r["stock"]))
    streak: Dict[str, int] = {}
    sitting: Dict[str, int] = {}
    kept: List[dict] = []
    for rec in ordered:
        sym = str(rec["stock"])
        if sitting.get(sym, 0) > 0:
            sitting[sym] -= 1
            continue
        kept.append(dict(rec))
        g = float(rec.get("gain_pct") or 0.0)
        if g < 0:
            streak[sym] = streak.get(sym, 0) + 1
            if streak[sym] >= int(max_losses):
                sitting[sym] = 1  # skip next
                streak[sym] = 0
        else:
            streak[sym] = 0
    return kept


__all__ = [
    "BOOK_CONFIRMED",
    "BOOK_FAKE",
    "BOOK_RANGE",
    "CAP_LARGE",
    "CAP_MID",
    "CAP_SMALL",
    "CLAIMED",
    "NilParams",
    "TRADE_COLUMNS",
    "YEAR_BUCKETS",
    "apply_friction",
    "build_cap_universe",
    "cap_bucket_for",
    "collect_symbol_trades",
    "count_boundary_touches",
    "detect_impulse_at",
    "drop_top_n_expectancy",
    "max_concurrent",
    "occupancy_equity_from_trades",
    "occupancy_take",
    "params_dict",
    "sitout_after_loss_streak",
    "summarize_by_year",
    "weekly_htf_bias",
]
