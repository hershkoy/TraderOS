"""Same-list volume-delta / geometry overlays on the last-15m-open-mid book.

Overlay A: post-fill Evening Doji Star or two seller-heavy RTH sessions; sell
the next 15m mid if that is strictly earlier than the book's ATR/trail exit.

Overlay B: skip formation_beyond_width > 0.25 and/or channel_pos above a cap.

Overlay C: for channel_pos skips only, wait for a close back at/below the rail
then a 2nd close above with session buy_pct >= 50. Occupancy is not re-walked.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from utils.research.evening_doji_star import find_evening_doji_star
from utils.research.realistic_exits import (
    EXIT_HARD_STOP,
    flatten_rth_sessions,
    simulate_exit_15m_next_mid,
)
from utils.research.realistic_purchaser import as_et, bar_mid
from utils.research.session_volume_delta import (
    SessionDelta,
    VOLUME_MODE_15M_SUM,
    consecutive_seller_sessions,
    sessions_from_by_day,
)
from utils.scanning.channel_touch_bought import ATR_STOP_MULT

FRICTION_PCT = 0.25
FORM_CAP = 0.25
POS_CAPS: Tuple[float, ...] = (1.25, 1.50)
SELLER_PCT_MIN = 55.0
SELLER_N = 2
EXIT_DOJI_STAR = "doji_star"
EXIT_SELLER_SESSIONS = "seller_sessions"
EXIT_FAILED_BREAKOUT = "failed_breakout"
MIN_YEAR_N = 30
BLUNT_WINNER_FRAC = 0.35
BLUNT_DOLLAR_RATIO = 0.50
C_MAX_WAIT_SESSIONS = 20
C_BUY_PCT_MIN = 50.0


def _num(row: pd.Series, *names: str) -> Optional[float]:
    for name in names:
        if name not in row.index:
            continue
        try:
            val = float(row[name])
        except (TypeError, ValueError):
            continue
        if val == val:
            return val
    return None


def atr_dollars_for_row(row: pd.Series, entry_px: float) -> Optional[float]:
    atr_pct = _num(row, "atr_1d_pct", "atr_pct")
    ref = _num(row, "buy_price_before", "orig_buy_price")
    if ref is None:
        ref = entry_px
    if atr_pct is None or ref is None or atr_pct <= 0 or ref <= 0:
        return None
    return ref * atr_pct / 100.0


def _ts_et(ts: Any) -> Optional[datetime]:
    if ts is None or (isinstance(ts, float) and pd.isna(ts)):
        return None
    try:
        return as_et(ts)
    except (TypeError, ValueError):
        return None


def next_mid_after_session(
    bars: Sequence[dict],
    session_day: date,
) -> Optional[Tuple[float, datetime, datetime]]:
    """Fill the RTH 15m mid after the last bar of ``session_day``."""
    last_i = None
    for i, bar in enumerate(bars):
        ts = _ts_et(bar.get("ts"))
        if ts is None:
            continue
        if ts.date() == session_day:
            last_i = i
    if last_i is None:
        return None
    nxt_i = last_i + 1
    if nxt_i >= len(bars):
        return None
    nxt = bars[nxt_i]
    hi = nxt.get("high")
    lo = nxt.get("low")
    try:
        hi_f = float(hi)
        lo_f = float(lo)
    except (TypeError, ValueError):
        return None
    if hi_f != hi_f or lo_f != lo_f:
        return None
    dec_ts = bars[last_i].get("ts")
    exec_ts = nxt.get("ts")
    if not isinstance(dec_ts, datetime) or not isinstance(exec_ts, datetime):
        return None
    return float(bar_mid(hi_f, lo_f)), dec_ts, exec_ts


def fill_day_et(row: pd.Series) -> Optional[date]:
    ts = _ts_et(row.get("buy_time"))
    if ts is not None:
        return ts.date()
    raw = row.get("buy_date")
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    t = pd.Timestamp(raw)
    return t.date()


@dataclass
class EarlyExitResult:
    used_overlay: bool
    exit_reason: str
    sell_px: float
    sell_ts: Optional[datetime]
    decision_ts: Optional[datetime]
    doji_day: Optional[date]
    seller_day: Optional[date]
    failed_breakout_day: Optional[date]
    gain_pct: float


def find_failed_breakout_day(
    row: pd.Series,
    sessions: Sequence[SessionDelta],
    *,
    fill_day: date,
    until_day: Optional[date] = None,
) -> Optional[date]:
    """First post-fill session whose close is at/below the projected resist rail."""
    after = [s for s in sessions if s.session_date > fill_day]
    if until_day is not None:
        after = [s for s in after if s.session_date <= until_day]
    for i, sess in enumerate(after):
        resist = _resist_at_offset(row, bars_after_fill=i + 1)
        if resist is None:
            return None
        if float(sess.close) <= resist + 1e-12:
            return sess.session_date
    return None


def apply_early_exit(
    row: pd.Series,
    by_day: Dict[date, Sequence[dict]],
    *,
    require_gap: bool = True,
    seller_pct_min: float = SELLER_PCT_MIN,
    enable_doji: bool = True,
    enable_seller: bool = True,
    enable_failed_breakout: bool = False,
    volume_mode: str = VOLUME_MODE_15M_SUM,
) -> EarlyExitResult:
    """Keep the book's ATR/trail exit unless an overlay fill is strictly earlier."""
    entry_px = float(row["buy_price"])
    orig_px = float(row["sell_price"])
    orig_reason = str(row.get("exit_reason") or EXIT_HARD_STOP)
    orig_ts = _ts_et(row.get("sell_time")) or _ts_et(row.get("sell_date"))
    orig_gain = (orig_px / entry_px - 1.0) * 100.0
    stored = _num(row, "gain_pct")
    if stored is not None:
        orig_gain = float(stored)
    fill_day = fill_day_et(row)
    until_day = orig_ts.date() if orig_ts is not None else None
    empty = EarlyExitResult(
        used_overlay=False,
        exit_reason=orig_reason,
        sell_px=orig_px,
        sell_ts=orig_ts,
        decision_ts=None,
        doji_day=None,
        seller_day=None,
        failed_breakout_day=None,
        gain_pct=float(orig_gain),
    )
    if fill_day is None or not by_day:
        return empty
    sessions = sessions_from_by_day(by_day, volume_mode=volume_mode)
    doji_day = (
        find_evening_doji_star(
            sessions, fill_day=fill_day, require_gap=require_gap, until_day=until_day
        )
        if enable_doji
        else None
    )
    seller_day = (
        consecutive_seller_sessions(
            sessions,
            fill_day=fill_day,
            seller_pct_min=seller_pct_min,
            n_needed=SELLER_N,
            until_day=until_day,
        )
        if enable_seller
        else None
    )
    fail_day = (
        find_failed_breakout_day(
            row, sessions, fill_day=fill_day, until_day=until_day
        )
        if enable_failed_breakout
        else None
    )
    bars = flatten_rth_sessions(by_day)
    candidates: List[Tuple[datetime, float, str, datetime]] = []
    for day, reason in (
        (doji_day, EXIT_DOJI_STAR),
        (seller_day, EXIT_SELLER_SESSIONS),
        (fail_day, EXIT_FAILED_BREAKOUT),
    ):
        if day is None:
            continue
        got = next_mid_after_session(bars, day)
        if got is None:
            continue
        px, dec_ts, exec_ts = got
        exec_et = _ts_et(exec_ts)
        if exec_et is None:
            continue
        candidates.append((exec_et, px, reason, dec_ts))

    def _keep(
        *,
        used: bool,
        reason: str,
        px: float,
        sell_ts: Optional[datetime],
        decision_ts: Optional[datetime],
        gain: float,
    ) -> EarlyExitResult:
        return EarlyExitResult(
            used_overlay=used,
            exit_reason=reason,
            sell_px=px,
            sell_ts=sell_ts,
            decision_ts=decision_ts,
            doji_day=doji_day,
            seller_day=seller_day,
            failed_breakout_day=fail_day,
            gain_pct=float(gain),
        )

    if not candidates:
        return _keep(
            used=False,
            reason=orig_reason,
            px=orig_px,
            sell_ts=orig_ts,
            decision_ts=None,
            gain=orig_gain,
        )
    # Prefer the earlier fill; doji_star, then failed_breakout, then seller.
    rank = {EXIT_DOJI_STAR: 0, EXIT_FAILED_BREAKOUT: 1, EXIT_SELLER_SESSIONS: 2}

    def _key(item: Tuple[datetime, float, str, datetime]) -> Tuple[datetime, int]:
        return (item[0], rank.get(item[2], 9))

    exec_et, px, reason, dec_ts = min(candidates, key=_key)
    if orig_ts is not None and exec_et >= orig_ts:
        return _keep(
            used=False,
            reason=orig_reason,
            px=orig_px,
            sell_ts=orig_ts,
            decision_ts=None,
            gain=orig_gain,
        )
    gain = (px / entry_px - 1.0) * 100.0
    return _keep(
        used=True,
        reason=reason,
        px=float(px),
        sell_ts=exec_et,
        decision_ts=dec_ts,
        gain=gain,
    )


def skip_mask(
    trades: pd.DataFrame,
    *,
    max_formation_beyond: Optional[float] = None,
    max_channel_pos: Optional[float] = None,
) -> pd.Series:
    """True = keep. Missing geometry fails the cap (treated as too large)."""
    keep = pd.Series(True, index=trades.index)
    if max_formation_beyond is not None and "formation_beyond_width" in trades.columns:
        form = pd.to_numeric(trades["formation_beyond_width"], errors="coerce")
        keep &= form.fillna(999.0) <= float(max_formation_beyond)
    if max_channel_pos is not None and "channel_pos" in trades.columns:
        pos = pd.to_numeric(trades["channel_pos"], errors="coerce")
        keep &= pos.fillna(999.0) <= float(max_channel_pos)
    return keep


def _profit_factor(gains: pd.Series) -> float:
    g = pd.to_numeric(gains, errors="coerce").dropna()
    wins = float(g[g > 0].sum())
    losses = float((-g[g < 0]).sum())
    if losses <= 0:
        return float("inf") if wins > 0 else float("nan")
    return float(wins / losses)


def book_stats(gains: pd.Series) -> dict:
    g = pd.to_numeric(gains, errors="coerce").dropna()
    n = int(len(g))
    if n == 0:
        return {
            "n": 0,
            "win_rate_pct": None,
            "expectancy_pct": None,
            "profit_factor": None,
            "median_pct": None,
        }
    pf = _profit_factor(g)
    return {
        "n": n,
        "win_rate_pct": round(float((g > 0).mean() * 100.0), 2),
        "expectancy_pct": round(float(g.mean()), 4),
        "profit_factor": None if not np.isfinite(pf) else round(float(pf), 3),
        "median_pct": round(float(g.median()), 4),
    }


def splice_calendar_year(
    base: pd.DataFrame,
    replacement: pd.DataFrame,
    *,
    year: int,
    date_col: str = "buy_date",
    src_base: str = "base",
    src_repl: str = "replacement",
) -> pd.DataFrame:
    """Keep ``base`` except calendar ``year``, which comes from ``replacement``.

    Occupancy is not re-walked. Column union; missing values stay NA.
    """
    if date_col not in base.columns or date_col not in replacement.columns:
        raise ValueError("splice needs %s on both frames" % date_col)

    def _year_mask(df: pd.DataFrame) -> pd.Series:
        d = pd.to_datetime(df[date_col], errors="coerce")
        return d.dt.year == int(year)

    left = base.loc[~_year_mask(base)].copy()
    right = replacement.loc[_year_mask(replacement)].copy()
    left["splice_src"] = src_base
    right["splice_src"] = src_repl
    out = pd.concat([left, right], axis=0, ignore_index=True, sort=False)
    out["_splice_sort"] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.sort_values("_splice_sort", kind="mergesort").drop(columns=["_splice_sort"])
    return out.reset_index(drop=True)


def geom_mean_year_pf(year_df: pd.DataFrame, *, min_n: int = MIN_YEAR_N) -> Optional[float]:
    if year_df is None or year_df.empty:
        return None
    pfs: List[float] = []
    for _, row in year_df.iterrows():
        if str(row.get("bucket") or "") == "FULL":
            continue
        try:
            n = int(row.get("n_trades") or 0)
        except (TypeError, ValueError):
            continue
        pf = row.get("profit_factor")
        if n < int(min_n) or pf is None:
            continue
        try:
            pf_f = float(pf)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(pf_f) or pf_f <= 0:
            continue
        pfs.append(pf_f)
    if not pfs:
        return None
    return float(math.exp(sum(math.log(p) for p in pfs) / len(pfs)))


def winner_cut_early_exit(
    baseline_gain: pd.Series,
    overlay_gain: pd.Series,
) -> dict:
    b = pd.to_numeric(baseline_gain, errors="coerce")
    o = pd.to_numeric(overlay_gain, errors="coerce")
    m = b.notna() & o.notna()
    b = b[m]
    o = o[m]
    bw = b > 0
    ow = o > 0
    delta = o - b
    return {
        "n": int(len(b)),
        "baseline_winners": int(bw.sum()),
        "winners_stay": int((bw & ow).sum()),
        "winners_flip_to_loser": int((bw & ~ow).sum()),
        "losers_stay": int((~bw & ~ow).sum()),
        "losers_flip_to_winner": int((~bw & ow).sum()),
        "winner_gain_delta_sum": round(float(delta[bw].sum()), 4),
        "loser_gain_delta_sum": round(float(delta[~bw].sum()), 4),
        "n_cut_earlier": int((delta.abs() > 1e-9).sum()),
    }


def winner_cut_skip(
    baseline_gain: pd.Series,
    keep: pd.Series,
) -> dict:
    b = pd.to_numeric(baseline_gain, errors="coerce")
    k = keep.reindex(b.index).fillna(False).astype(bool)
    dropped = ~k
    bw = b > 0
    win_drop = dropped & bw & b.notna()
    lose_drop = dropped & ~bw & b.notna()
    win_sum = float(b[win_drop].sum()) if win_drop.any() else 0.0
    lose_sum = float(b[lose_drop].sum()) if lose_drop.any() else 0.0
    n_drop = int(dropped.sum())
    winner_frac = (float(win_drop.sum()) / n_drop) if n_drop else 0.0
    saved = -lose_sum
    dollar_ratio = (win_sum / saved) if saved > 1e-12 else (float("inf") if win_sum > 0 else 0.0)
    blunt = bool(
        n_drop > 0
        and winner_frac >= BLUNT_WINNER_FRAC
        and (not np.isfinite(dollar_ratio) or dollar_ratio >= BLUNT_DOLLAR_RATIO)
    )
    return {
        "n": int(len(b)),
        "n_kept": int(k.sum()),
        "n_dropped": n_drop,
        "winners_dropped": int(win_drop.sum()),
        "losers_dropped": int(lose_drop.sum()),
        "winner_drop_gain_sum": round(win_sum, 4),
        "loser_drop_gain_sum": round(lose_sum, 4),
        "winner_frac_of_drops": round(winner_frac, 4),
        "dollar_ratio": None if not np.isfinite(dollar_ratio) else round(float(dollar_ratio), 4),
        "blunt": blunt,
    }


def _resist_at_offset(
    row: pd.Series,
    *,
    bars_after_fill: int,
) -> Optional[float]:
    width = _num(row, "channel_width")
    pos = _num(row, "channel_pos")
    buy = _num(row, "buy_price")
    if width is None or pos is None or buy is None or width <= 0:
        return None
    support0 = buy - pos * width
    resist0 = support0 + width
    slope_pct = _num(row, "slope_pct_per_bar")
    l1 = _num(row, "l1_price")
    slope = 0.0
    if slope_pct is not None and l1 is not None and l1 > 0:
        slope = (slope_pct / 100.0) * l1
    return float(resist0 + slope * float(bars_after_fill))


def delayed_second_close(
    row: pd.Series,
    by_day: Dict[date, Sequence[dict]],
    *,
    max_wait_sessions: int = C_MAX_WAIT_SESSIONS,
    buy_pct_min: float = C_BUY_PCT_MIN,
) -> Optional[dict]:
    """After skipping an extended first print, wait for pullback then 2nd close above.

    Fill is the last-15m mid of the confirm session. Exit is ATR k=2 + 10% trail
    from that new entry (15m N+1 mid). Occupancy is not re-walked.
    """
    fill_day = fill_day_et(row)
    if fill_day is None or not by_day:
        return None
    sessions = sessions_from_by_day(by_day)
    after = [s for s in sessions if s.session_date > fill_day]
    if not after:
        return None
    inside = False
    confirm: Optional[SessionDelta] = None
    offset = 0
    for i, sess in enumerate(after):
        if i >= int(max_wait_sessions):
            break
        offset = i + 1
        resist = _resist_at_offset(row, bars_after_fill=offset)
        if resist is None:
            return None
        if not inside:
            if float(sess.close) <= resist + 1e-12:
                inside = True
            continue
        if float(sess.close) > resist and float(sess.buy_pct) >= float(buy_pct_min):
            confirm = sess
            break
    if confirm is None:
        return None
    bars = flatten_rth_sessions(by_day)
    day_bars = [b for b in bars if _ts_et(b.get("ts")) and _ts_et(b.get("ts")).date() == confirm.session_date]
    if not day_bars:
        return None
    last = day_bars[-1]
    try:
        hi = float(last["high"])
        lo = float(last["low"])
    except (KeyError, TypeError, ValueError):
        return None
    entry_px = float(bar_mid(hi, lo))
    entry_ts = last.get("ts")
    atr = atr_dollars_for_row(row, entry_px)
    got = simulate_exit_15m_next_mid(
        bars,
        entry_ts=entry_ts,
        entry_px=entry_px,
        atr_at_entry=atr,
        atr_stop_mult=ATR_STOP_MULT,
    )
    if not got.filled or got.sell_px is None:
        return None
    gain = (float(got.sell_px) / entry_px - 1.0) * 100.0
    return {
        "buy_date": confirm.session_date.isoformat(),
        "buy_time": str(entry_ts) if entry_ts is not None else "",
        "buy_price": round(entry_px, 4),
        "sell_price": round(float(got.sell_px), 4),
        "sell_time": str(got.exec_bar_ts) if got.exec_bar_ts is not None else "",
        "gain_pct": round(float(gain), 4),
        "exit_reason": str(got.exit_reason or ""),
        "wait_sessions": offset,
        "confirm_buy_pct": round(float(confirm.buy_pct), 2),
    }
