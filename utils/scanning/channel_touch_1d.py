"""Daily (1d) armed H2 rows for the /hot dashboard.

Nightly still Telegrams fills. This writes the armed/waiting/filled book so
the dashboard can show 1d names next to the 15m list. Detector params match
LIVE_DEFAULTS (H2 resist-break, wait 6, span 365, window 504/252,
shakeout-breakout re-arm after the first fill has exited).
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import pandas as pd

from utils.scanning.channel_touch import LIVE_DEFAULTS, _remap_channel_kwargs
from utils.scanning.channel_touch_15m import armed_rows_for_symbol

logger = logging.getLogger(__name__)

TIMEFRAME_1D = "1d"


def daily_channel_kwargs() -> dict:
    d = LIVE_DEFAULTS
    return _remap_channel_kwargs(
        {
            "pivot_len": int(d["pivot_len"]),
            "error_pct": float(d["error_pct"]),
            "flat_pct": float(d["flat_pct"]),
            "min_bars_apart": int(d["min_bars_apart"]),
            "min_rally_pct": float(d["min_rally_pct"]),
            "min_pullback_pct": float(d["min_pullback_pct"]),
            "min_total_rise_pct": float(d["min_total_rise_pct"]),
            "max_low_pivots": int(d["max_low_pivots"]),
        }
    )


def build_1d_watchlist_rows(
    panels: Dict[str, pd.DataFrame],
    symbols: Sequence[str],
    *,
    as_of: Optional[pd.Timestamp] = None,
    min_wait: Optional[int] = None,
    max_wait: Optional[int] = None,
    max_span: Optional[float] = None,
    slip: Optional[float] = None,
    window_bars: Optional[int] = None,
    window_step_bars: Optional[int] = None,
) -> List[dict]:
    d = LIVE_DEFAULTS
    ck = daily_channel_kwargs()
    min_w = int(d["min_l3_wait_bars"] if min_wait is None else min_wait)
    max_w = int(d["max_l3_wait_bars"] if max_wait is None else max_wait)
    span = float(d["max_channel_span_days"] if max_span is None else max_span)
    slip_f = float(d["entry_slip_pct"] if slip is None else slip)
    win = int(d["window_bars"] if window_bars is None else window_bars)
    step = int(d["window_step_bars"] if window_step_bars is None else window_step_bars)
    err = float(d["error_pct"])
    rows: List[dict] = []
    n_sym = len(symbols)
    for i, sym in enumerate(symbols, start=1):
        df = panels.get(sym)
        if df is None or df.empty:
            continue
        got = armed_rows_for_symbol(
            str(sym),
            df,
            as_of=as_of,
            channel_kwargs=ck,
            min_wait=min_w,
            max_wait=max_w,
            max_span_days=span,
            error_pct=err,
            slip=slip_f,
            window_bars=win,
            window_step_bars=step,
            shakeout_breakout=bool(d.get("shakeout_breakout", False)),
            shakeout_breakout_min_inside=int(d.get("shakeout_breakout_min_inside", 1)),
        )
        for row in got:
            item = dict(row)
            item["timeframe"] = TIMEFRAME_1D
            rows.append(item)
        if i % 100 == 0:
            logger.info("1d watchlist scanned %d/%d symbols, rows=%d", i, n_sym, len(rows))
    return rows
