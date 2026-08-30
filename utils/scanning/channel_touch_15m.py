"""
15m channel-touch live helpers (research H5 stack; not the daily nightly).

Armed H2 resist-break setups on stored IB 15m bars, Alpaca last-price
proximity, then a completed-bar fill check:

  wait >= 12 after H2, close above resistance, unique-symbol/day,
  prior-bar volume_rel_20 >= 2, fill overshoot >= frozen p80 (~0.08).

Do not stream the IB 15m universe. Do not use Alpaca IEX 15m volume for the
H5 gate (research volume is IB). Last price is a proximity screen, not a fill.
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESEARCH = ROOT / "scripts" / "research"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(RESEARCH) not in sys.path:
    sys.path.insert(0, str(RESEARCH))

from find_ascending_channels import find_h2_l3_setups_windowed  # noqa: E402
from backtest_channel_touch_trades import (  # noqa: E402
    _l3_rail_touch,
    _limit_fill_at_support,
    _line_at,
)
from utils.research.channel_touch_scale import PRESET_15M  # noqa: E402
from utils.scanning.channel_touch import _remap_channel_kwargs  # noqa: E402

logger = logging.getLogger(__name__)

# Frozen live floor approximating expanding-year train p80 on the unique H2 book
# (top quintile of overshoot started at 0.088). Not recomputed each bar.
FROZEN_OVERSHOOT_MIN = 0.08

LIVE_15M_DEFAULTS: Dict[str, Any] = {
    "timeframe": "15m",
    "provider": "IB",
    "min_l3_wait_bars": 12,
    "max_l3_wait_bars": 252,
    "entry_slip_pct": 0.001,
    "h2_resist_break": True,
    "max_channel_span_days": 10.0,
    "require_in_channel": False,
    "volume_rel_min": 2.0,
    "overshoot_min": FROZEN_OVERSHOOT_MIN,
    "error_pct": float(PRESET_15M["error_pct"]),
    "pivot_len": int(PRESET_15M["pivot_len"]),
    "min_bars_apart": int(PRESET_15M["min_bars_apart"]),
    "min_rally_pct": float(PRESET_15M["min_rally_pct"]),
    "min_pullback_pct": float(PRESET_15M["min_pullback_pct"]),
    "min_total_rise_pct": float(PRESET_15M["min_total_rise_pct"]),
    "flat_pct": float(PRESET_15M["flat_pct"]),
    "max_low_pivots": int(PRESET_15M["max_low_pivots"]),
    "window_bars": int(PRESET_15M["window_bars"]),
    "window_step_bars": int(PRESET_15M["window_step_bars"]),
    "friction_pct": float(PRESET_15M["friction_pct"]),
    "proximity_below_pct": 0.0,
    "lookback_sessions": 40,
}


def setup_channel_kwargs(overrides: Optional[dict] = None) -> dict:
    d = LIVE_15M_DEFAULTS
    raw = {
        "pivot_len": int(d["pivot_len"]),
        "error_pct": float(d["error_pct"]),
        "flat_pct": float(d["flat_pct"]),
        "min_bars_apart": int(d["min_bars_apart"]),
        "min_rally_pct": float(d["min_rally_pct"]),
        "min_pullback_pct": float(d["min_pullback_pct"]),
        "min_total_rise_pct": float(d["min_total_rise_pct"]),
        "max_low_pivots": int(d["max_low_pivots"]),
    }
    if overrides:
        raw.update(overrides)
    return _remap_channel_kwargs(raw)


def _naive_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df
    if not isinstance(out.index, pd.DatetimeIndex):
        out = out.copy()
        out.index = pd.DatetimeIndex(out.index)
    if out.index.tz is not None:
        out = out.copy()
        out.index = out.index.tz_convert(None)
    return out.sort_index()


def _asof_i(dates: pd.DatetimeIndex, as_of: Optional[pd.Timestamp], n: int) -> Optional[int]:
    if n < 1:
        return None
    if as_of is None:
        return n - 1
    asof_ts = pd.Timestamp(as_of)
    if asof_ts.tzinfo is not None:
        asof_ts = asof_ts.tz_convert(None)
    hist = dates[dates <= asof_ts]
    if len(hist) == 0:
        return None
    loc = dates.get_loc(hist[-1])
    if isinstance(loc, slice):
        return None
    return int(loc)


def _span_days(ch: dict) -> Optional[float]:
    try:
        start = pd.Timestamp(ch["start_date"])
        end = pd.Timestamp(ch.get("h2_date") or ch["end_date"])
        return float((end - start).days)
    except Exception:
        return None


def rails_at(
    *,
    support_x0: int,
    support_y0: float,
    support_slope: float,
    width: float,
    i: int,
) -> tuple:
    sup = _line_at(support_y0, support_x0, support_slope, i)
    resist = float(sup) + float(width or 0.0)
    return float(sup), float(resist)


def channel_pos_at(px: float, support: float, width: float) -> float:
    if not np.isfinite(px) or not np.isfinite(support) or not np.isfinite(width) or width <= 0:
        return float("nan")
    return float((px - support) / width)


def walk_h2_resist_asof(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    *,
    support_x0: int,
    support_y0: float,
    support_slope: float,
    width: float,
    h2_idx: int,
    as_of_i: int,
    min_wait: int = 12,
    max_wait: int = 252,
    error_pct: float = 0.24,
    slip: float = 0.001,
) -> dict:
    """State of one H2 resist-break setup at ``as_of_i`` (inclusive).

    Mirrors ``_h2_rail_tag_fills(..., h2_resist_break=True)`` but returns
    armed/waiting/cancelled/expired when there is no fill on this bar.
    """
    n = len(close)
    h2 = int(h2_idx)
    asof = int(as_of_i)
    min_w = max(1, int(min_wait))
    wait_n = max(1, int(max_wait))
    err = float(error_pct)
    if h2 < 0 or asof < 0 or asof >= n or h2 >= n:
        return {"status": "invalid", "wait_bars": None}
    wait_bars = int(asof - h2)
    sup_now, resist_now = rails_at(
        support_x0=support_x0,
        support_y0=support_y0,
        support_slope=support_slope,
        width=width,
        i=asof,
    )
    last_close = float(close[asof]) if np.isfinite(close[asof]) else float("nan")
    dist_pct = (
        (last_close - resist_now) / resist_now * 100.0
        if np.isfinite(last_close) and np.isfinite(resist_now) and resist_now > 0
        else float("nan")
    )
    base = {
        "wait_bars": wait_bars,
        "support": round(float(sup_now), 6) if np.isfinite(sup_now) else None,
        "resist": round(float(resist_now), 6) if np.isfinite(resist_now) else None,
        "last_close": round(float(last_close), 6) if np.isfinite(last_close) else None,
        "dist_to_resist_pct": round(float(dist_pct), 4) if np.isfinite(dist_pct) else None,
        "fill_i": None,
        "fill_px": None,
        "overshoot": None,
    }
    if wait_bars > wait_n:
        base["status"] = "expired"
        return base

    h2_px = float(high[h2]) if np.isfinite(high[h2]) else float("nan")
    end = min(asof, h2 + wait_n)
    for i in range(h2 + 1, end + 1):
        sup, resist = rails_at(
            support_x0=support_x0,
            support_y0=support_y0,
            support_slope=support_slope,
            width=width,
            i=i,
        )
        if i > 0:
            sup_prev, _ = rails_at(
                support_x0=support_x0,
                support_y0=support_y0,
                support_slope=support_slope,
                width=width,
                i=i - 1,
            )
            if float(close[i - 1]) < sup_prev * (1.0 - err / 100.0):
                base["status"] = "cancelled"
                return base
        close_i = float(close[i])
        if np.isfinite(resist) and resist > 0 and close_i > resist * (1.0 + err / 100.0):
            if i >= h2 + min_w:
                fill = _limit_fill_at_support(resist, float(low[i]), float(high[i]), slip)
                if fill is None:
                    if i == asof:
                        base["status"] = "cancelled"
                        return base
                    base["status"] = "cancelled"
                    return base
                pos = channel_pos_at(float(fill), sup, float(width))
                overshoot = pos - 1.0 if np.isfinite(pos) else float("nan")
                if i < asof:
                    base["status"] = "filled_earlier"
                    base["fill_i"] = int(i)
                    base["fill_px"] = float(fill)
                    return base
                base["status"] = "filled"
                base["fill_i"] = int(i)
                base["fill_px"] = round(float(fill), 6)
                base["overshoot"] = round(float(overshoot), 4) if np.isfinite(overshoot) else None
                return base
            continue
        touched = _l3_rail_touch(float(high[i]), float(low[i]), close_i, sup, err)
        broke = close_i < sup * (1.0 - err / 100.0)
        if i < h2 + min_w:
            if touched or broke:
                base["status"] = "cancelled"
                return base
            continue
        if broke:
            base["status"] = "cancelled"
            return base
        _ = h2_px  # H2 high does not cancel on resist-break (matches take_break)

    if wait_bars < min_w:
        base["status"] = "waiting"
        return base
    base["status"] = "armed"
    return base


def _volume_rel_series(volume: np.ndarray) -> np.ndarray:
    vol = pd.Series(volume, dtype=float)
    ma = vol.rolling(20, min_periods=20).mean().to_numpy(dtype=float)
    raw = vol.to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(ma > 0, raw / ma, np.nan)


def armed_rows_for_symbol(
    symbol: str,
    df: pd.DataFrame,
    *,
    as_of: Optional[pd.Timestamp] = None,
    channel_kwargs: Optional[dict] = None,
    min_wait: int = 12,
    max_wait: int = 252,
    max_span_days: float = 10.0,
    error_pct: float = 0.24,
    slip: float = 0.001,
    window_bars: int = 390,
    window_step_bars: int = 130,
    setups: Optional[List[dict]] = None,
) -> List[dict]:
    """Armed or waiting H2 resist-break setups as of the last bar."""
    if df is None or df.empty:
        return []
    out = _naive_index(df)
    n = len(out)
    asof_i = _asof_i(out.index, as_of, n)
    if asof_i is None:
        return []
    hist = out.iloc[: asof_i + 1]
    kw = setup_channel_kwargs(channel_kwargs)
    if setups is None:
        setups = find_h2_l3_setups_windowed(
            hist,
            window_bars=int(window_bars),
            step_bars=int(window_step_bars),
            **kw,
        )
    high = hist["high"].to_numpy(dtype=float)
    low = hist["low"].to_numpy(dtype=float)
    close = hist["close"].to_numpy(dtype=float)
    volume = hist["volume"].to_numpy(dtype=float) if "volume" in hist.columns else np.ones(len(hist))
    vol_rel = _volume_rel_series(volume)
    dates = hist.index
    asof_i_hist = len(hist) - 1
    rows: List[dict] = []
    for ch in setups:
        span = _span_days(ch)
        if span is not None and span > float(max_span_days):
            continue
        h2 = int(ch.get("h2_idx", -1))
        if h2 < 0 or h2 >= asof_i_hist:
            continue
        st = walk_h2_resist_asof(
            high,
            low,
            close,
            support_x0=int(ch["support_x0"]),
            support_y0=float(ch["support_y0"]),
            support_slope=float(ch["support_slope"]),
            width=float(ch["channel_width"]),
            h2_idx=h2,
            as_of_i=asof_i_hist,
            min_wait=int(min_wait),
            max_wait=int(max_wait),
            error_pct=float(error_pct),
            slip=float(slip),
        )
        status = st.get("status")
        if status not in ("armed", "waiting", "filled"):
            continue
        prior_i = asof_i_hist - 1
        vol_prior = float(vol_rel[prior_i]) if prior_i >= 0 else float("nan")
        if prior_i >= 0:
            sup_p, resist_p = rails_at(
                support_x0=int(ch["support_x0"]),
                support_y0=float(ch["support_y0"]),
                support_slope=float(ch["support_slope"]),
                width=float(ch["channel_width"]),
                i=prior_i,
            )
            over_prior = channel_pos_at(float(close[prior_i]), sup_p, float(ch["channel_width"])) - 1.0
        else:
            over_prior = float("nan")
        h2_ts = dates[h2]
        asof_ts = dates[asof_i_hist]
        row = {
            "stock": str(symbol).upper(),
            "status": status,
            "as_of": asof_ts.strftime("%Y-%m-%d %H:%M:%S"),
            "h2_time": h2_ts.strftime("%Y-%m-%d %H:%M:%S"),
            "channel_start": ch.get("start_date"),
            "channel_end": ch.get("h2_date") or ch.get("end_date"),
            "channel_span_days": span,
            "wait_bars": st.get("wait_bars"),
            "wait_ok": bool(int(st.get("wait_bars") or 0) >= int(min_wait)),
            "support": st.get("support"),
            "resist": st.get("resist"),
            "last_close": st.get("last_close"),
            "dist_to_resist_pct": st.get("dist_to_resist_pct"),
            "volume_rel_20": round(vol_prior, 4) if np.isfinite(vol_prior) else None,
            "overshoot_prior": round(float(over_prior), 4) if np.isfinite(over_prior) else None,
            "fill_px": st.get("fill_px"),
            "overshoot": st.get("overshoot"),
            "support_x0": int(ch["support_x0"]),
            "support_y0": float(ch["support_y0"]),
            "support_slope": float(ch["support_slope"]),
            "channel_width": float(ch["channel_width"]),
            "h2_idx": h2,
            "as_of_i": asof_i_hist,
        }
        rows.append(row)
    if not rows:
        return []
    # One live setup per symbol: prefer a fill on this bar, else latest H2.
    rows.sort(key=lambda r: (0 if r["status"] == "filled" else 1, -int(r["h2_idx"])))
    return [rows[0]]


def passes_h5_stack(
    row: dict,
    *,
    volume_rel_min: float = 2.0,
    overshoot_min: float = FROZEN_OVERSHOOT_MIN,
    require_wait: bool = True,
) -> bool:
    """H5 + overshoot p80 + vol>=2. Volume is prior-bar; overshoot is fill-bar."""
    if require_wait and not row.get("wait_ok"):
        return False
    vol = row.get("volume_rel_20")
    if vol is None or not np.isfinite(float(vol)) or float(vol) < float(volume_rel_min):
        return False
    over = row.get("overshoot")
    if over is None:
        over = row.get("overshoot_prior")
    if over is None or not np.isfinite(float(over)) or float(over) < float(overshoot_min):
        return False
    return True


def is_hot_proximity(
    last_price: Optional[float],
    resist: Optional[float],
    *,
    below_pct: float = 0.0,
) -> bool:
    """True if last is at or above resist, or within ``below_pct`` percent below.

    Default below_pct=0 keeps the H5 quality book (already through the rail).
    Tight-break-from-below is the rejected H2 unique-filter.
    """
    if last_price is None or resist is None:
        return False
    px = float(last_price)
    lvl = float(resist)
    if not np.isfinite(px) or not np.isfinite(lvl) or lvl <= 0 or px <= 0:
        return False
    floor = lvl * (1.0 - max(0.0, float(below_pct)) / 100.0)
    return px >= floor


def attach_last_prices(
    rows: Sequence[dict],
    last_prices: Dict[str, float],
    *,
    below_pct: float = 0.0,
) -> List[dict]:
    out: List[dict] = []
    for row in rows:
        item = dict(row)
        px = last_prices.get(str(item.get("stock", "")).upper())
        item["last_price"] = round(float(px), 6) if px is not None and np.isfinite(float(px)) else None
        resist = item.get("resist")
        if item["last_price"] is not None and resist is not None and float(resist) > 0:
            item["dist_live_pct"] = round(
                (float(item["last_price"]) - float(resist)) / float(resist) * 100.0, 4
            )
        else:
            item["dist_live_pct"] = None
        item["hot"] = is_hot_proximity(item["last_price"], resist, below_pct=float(below_pct))
        out.append(item)
    return out


def rank_hot(rows: Sequence[dict], *, below_pct: float = 0.0) -> List[dict]:
    ranked: List[dict] = []
    for row in rows:
        item = dict(row)
        item["hot"] = is_hot_proximity(
            item.get("last_price") if item.get("last_price") is not None else item.get("last_close"),
            item.get("resist"),
            below_pct=float(below_pct),
        )
        ranked.append(item)
    ranked.sort(
        key=lambda r: (
            0 if r.get("hot") else 1,
            -(float(r["dist_live_pct"]) if r.get("dist_live_pct") is not None else -999.0),
        )
    )
    return ranked


def fetch_alpaca_last_prices(
    symbols: Sequence[str],
    *,
    batch_size: int = 200,
    feed: str = "iex",
) -> Dict[str, float]:
    """Last trade price via Alpaca snapshot. Daily volume is IEX, not IB 15m."""
    from alpaca.data.enums import DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockSnapshotRequest
    from utils.config.env_loader import get_env_var, load_env_file
    import os

    load_env_file()
    key = (get_env_var("ALPACA_API_KEY_ID") or os.environ.get("ALPACA_API_KEY_ID") or "").strip()
    secret = (get_env_var("ALPACA_API_SECRET") or os.environ.get("ALPACA_API_SECRET") or "").strip()
    if not key or not secret:
        raise ValueError("ALPACA_API_KEY_ID / ALPACA_API_SECRET missing")
    feed_enum = DataFeed.IEX if str(feed).lower() == "iex" else DataFeed.SIP
    client = StockHistoricalDataClient(api_key=key, secret_key=secret)
    clean = [str(s).upper() for s in symbols if str(s).strip()]
    out: Dict[str, float] = {}
    size = max(1, int(batch_size))
    for i in range(0, len(clean), size):
        part = clean[i : i + size]
        req = StockSnapshotRequest(symbol_or_symbols=part, feed=feed_enum)
        data = client.get_stock_snapshot(req)
        for sym, snap in data.items():
            trade = getattr(snap, "latest_trade", None)
            px = getattr(trade, "price", None) if trade is not None else None
            if px is not None and np.isfinite(float(px)) and float(px) > 0:
                out[str(sym).upper()] = float(px)
    return out


def unique_symbol_day_ok(stock: str, as_of: str, filled_today: Iterable[str]) -> bool:
    key = str(stock).upper()
    day = str(as_of)[:10]
    for item in filled_today:
        text = str(item)
        if text.upper().startswith(key + "|") and text[len(key) + 1 :].startswith(day):
            return False
        if text.upper() == key:
            return False
    return True


def format_15m_message(
    *,
    as_of: str,
    n_armed: int,
    n_hot: int,
    fills: pd.DataFrame,
    n_universe: int = 0,
    stale_warning: Optional[str] = None,
) -> str:
    lines = [
        "Channel-touch 15m (research H5 stack)",
        f"as_of={as_of}",
        (
            "mode=h2_resist_break min_wait=12 span<=10 "
            f"vol>=2 overshoot>={FROZEN_OVERSHOOT_MIN} unique-symbol/day"
        ),
        f"universe={n_universe} armed={n_armed} hot={n_hot} fills={0 if fills is None else len(fills)}",
    ]
    if stale_warning:
        lines.append(f"WARNING: {stale_warning}")
    if fills is None or fills.empty:
        lines.append("No new 15m fills.")
        return "\n".join(lines)
    lines.append("")
    for _, row in fills.iterrows():
        vol = row.get("volume_rel_20")
        vol_s = f"{float(vol):.2f}" if vol is not None and np.isfinite(float(vol)) else "n/a"
        over = row.get("overshoot")
        over_s = f"{float(over):.3f}" if over is not None and np.isfinite(float(over)) else "n/a"
        lines.append(
            "{stock} fill={px} resist={resist} | vol_rel={vol} overshoot={over} wait={wait}".format(
                stock=row.get("stock"),
                px=row.get("fill_px") or row.get("buy_price"),
                resist=row.get("resist"),
                vol=vol_s,
                over=over_s,
                wait=row.get("wait_bars"),
            )
        )
    return "\n".join(lines)


def lookback_start(*, sessions: int = 40, now: Optional[datetime] = None) -> datetime:
    """Calendar start covering ``sessions`` RTH days plus weekend slack."""
    ts = pd.Timestamp(now or datetime.now(timezone.utc))
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    days = int(max(1, sessions) * 7 / 5) + 10
    return (ts - pd.Timedelta(days=days)).to_pydatetime()
