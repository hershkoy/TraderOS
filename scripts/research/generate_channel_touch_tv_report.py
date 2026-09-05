#!/usr/bin/env python3
"""
Generate a TradingView Strategy Tester-style HTML report from channel-touch trades CSV.

Interactive controls in the HTML (client-side recalc):
  - Portfolio size (initial capital)
  - Position sizing: fixed $, %% of initial, or %% of equity
  - Round-trip friction %%
  - Max entries per day (0=all; 1+=rank by RS vs SPY when available)
  - Distributions tab: orders/day, P&L histogram, hold, weekday/hour, concurrent, exits

Portfolio model:
  - Equity marks on trade exit dates (calendar day; SPY overlay stays daily)
  - Trade table uses buy_time/sell_time converted to RTH (America/New_York)
    when the CSV has them (IB 15m bars are stored UTC)
  - Max entries/day still groups by calendar buy_date
  - SPY buy-and-hold overlay for comparison

Usage (Windows CMD):
  venv\\Scripts\\activate && set PYTHONPATH=. && python scripts\\research\\generate_channel_touch_tv_report.py
  venv\\Scripts\\activate && set PYTHONPATH=. && python scripts\\research\\generate_channel_touch_tv_report.py --trades reports\\ascending_channels\\YYYY-MM-DD\\channel_touch_trades_XXXX.csv --rs-top1
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.research.backtest_channel_touch_trades import filter_trades
from utils.data.ohlcv_loader import load_ohlcv_many
from utils.research.report_paths import dated_outdir, latest_matching

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("generate_channel_touch_tv_report")

RTH_TZ = "America/New_York"

DETECTOR_META: Dict[str, str] = {
    "name": "Classical ascending channel (Edwards / Magee heuristics)",
    "script": "scripts/research/find_ascending_channels.py",
    "pine": "indicators/pine/ascending_channel_3touch.pine",
    "version": "v1 unchanged (quality via post-filters, not detector rewrite)",
    "notes": (
        "Distinct swing lows with meaningful intervening rallies; "
        ">=2 confirmed resistance touches; support respected inside window; "
        "not Trendoscope ACP."
    ),
}

DETECTOR_PARAM_KEYS = frozenset(
    {
        "error_pct",
        "min_rally_pct",
        "min_total_rise_pct",
        "pivot_len",
        "entry_touch",
        "preset",
        "window_bars",
        "window_step_bars",
    }
)

DATA_PARAM_KEYS = frozenset(
    {
        "provider",
        "timeframe",
        "start",
        "end",
        "symbols",
        "fallback_provider",
        "merge_mode",
        "bars_per_session",
        "rs_source",
        "rs_symbol",
        "elapsed_sec",
    }
)

RESULT_PARAM_KEYS = frozenset(
    {
        "n_trades",
        "n_symbols",
        "win_rate_pct",
        "avg_gain_pct",
        "median_gain_pct",
        "avg_win_pct",
        "avg_loss_pct",
        "expectancy_pct",
        "profit_factor",
        "avg_hold_days",
        "hard_stop_exits",
        "trail_stop_exits",
        "trail_stop_wide_exits",
        "trail_stop_tight_exits",
        "resist_exits",
        "time_stop_exits",
        "eod_exits",
    }
)


def _git_info() -> Dict[str, str]:
    def _run(*cmd: str) -> str:
        try:
            r = subprocess.run(
                list(cmd),
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=8,
                check=False,
            )
            return r.stdout.strip() if r.returncode == 0 else ""
        except Exception:
            return ""

    branch = _run("git", "rev-parse", "--abbrev-ref", "HEAD") or "unknown"
    commit = _run("git", "rev-parse", "--short", "HEAD") or "unknown"
    dirty = _run("git", "status", "--porcelain")
    return {"branch": branch, "commit": commit, "dirty": "yes" if dirty else "no"}


def _summary_path_for_trades(trades_path: Path) -> Optional[Path]:
    name = trades_path.name
    if "_trades_raw_" in name:
        name = name.replace("_trades_raw_", "_trades_", 1)
    if "_trades_" not in name:
        return None
    summary_name = name.replace("_trades_", "_trades_summary_", 1).replace(".csv", ".txt")
    candidate = trades_path.parent / summary_name
    return candidate if candidate.exists() else None


def _raw_sibling_csv(trades_path: Path) -> Optional[Path]:
    """Map filtered `*_trades_{stamp}.csv` to `*_trades_raw_{stamp}.csv` (or the path itself if raw)."""
    name = trades_path.name
    if "_trades_raw_" in name:
        return trades_path if trades_path.exists() else None
    if "_trades_" not in name:
        return None
    candidate = trades_path.parent / name.replace("_trades_", "_trades_raw_", 1)
    return candidate if candidate.exists() else None


def _opt_float(raw: Optional[str]) -> Optional[float]:
    text = (raw or "").strip()
    if text in ("", "None", "none", "null"):
        return None
    return float(text)


def _opt_bool(raw: Optional[str]) -> bool:
    return (raw or "").strip().lower() in {"true", "1", "yes"}


def parse_summary_tokens(path: Optional[Path]) -> Dict[str, str]:
    """Parse all key=value tokens in a summary sidecar (lines may hold several pairs)."""
    kv: Dict[str, str] = {}
    if path is None or not path.exists():
        return kv
    for line in path.read_text(encoding="utf-8").splitlines():
        for match in re.finditer(r"([A-Za-z_][A-Za-z0-9_]*)=(\S+)", line):
            kv[match.group(1)] = match.group(2)
    return kv


def filter_kwargs_from_summary(path: Optional[Path]) -> Dict[str, Any]:
    kv = parse_summary_tokens(path)
    return {
        "min_adv": _opt_float(kv.get("min_adv")),
        "min_atr_pct": _opt_float(kv.get("min_atr_pct")),
        "require_in_channel": _opt_bool(kv.get("require_in_channel")),
        "max_channel_span_days": _opt_float(kv.get("max_channel_span_days")),
        "max_channel_age_days": _opt_float(kv.get("max_channel_age_days")),
        "max_beyond_width": _opt_float(kv.get("max_beyond_width")),
        "max_rsi": _opt_float(kv.get("max_rsi")),
        "min_close_loc": _opt_float(kv.get("min_close_loc")),
        "require_spy_above_sma": _opt_bool(kv.get("spy_regime")) or _opt_bool(kv.get("require_spy_above_sma")),
    }


def load_trades_for_report(trades_path: Path) -> pd.DataFrame:
    """
    Embed quality-filtered trades WITHOUT the same-day RS cap.

    The backtest CSV is often already `max_entries_per_day=1`. The HTML control can only
    subset the embedded set, so we expand from the raw sibling and re-apply summary filters.
    """
    raw_path = _raw_sibling_csv(trades_path)
    if raw_path is None:
        logger.warning(
            "No raw sibling CSV for %s; Max entries/day cannot add same-day fills",
            trades_path.name,
        )
        return pd.read_csv(trades_path)

    df = pd.read_csv(raw_path)
    kwargs = filter_kwargs_from_summary(_summary_path_for_trades(trades_path))
    filtered = filter_trades(df, **kwargs)
    if filtered.empty and not df.empty:
        logger.warning("Quality filters dropped all raw trades; embedding unfiltered raw set")
        return df
    logger.info(
        "Report embed from %s: raw=%d quality-filtered=%d (no RS/day cap)",
        raw_path.name,
        len(df),
        len(filtered),
    )
    return filtered


def _parse_summary_txt(path: Optional[Path]) -> Tuple[Dict[str, str], List[str], List[str]]:
    """Return (key_values, config_lines, notes)."""
    kv: Dict[str, str] = {}
    config_lines: List[str] = []
    notes: List[str] = []
    if path is None or not path.exists():
        return kv, config_lines, notes
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if ">=" in line and "=" not in line.split(">=", 1)[0]:
            key, _, val = line.partition(">=")
            kv[key.strip()] = val.strip()
        elif "=" in line:
            key, _, val = line.partition("=")
            kv[key.strip()] = val.strip()
        elif line.startswith("Exit:") or line.startswith("Features:"):
            notes.append(line)
        elif line[0].isupper() and "backtest" in line.lower():
            config_lines.append(line)
        else:
            config_lines.append(line)
    return kv, config_lines, notes


def _split_param_rows(
    kv: Dict[str, str],
    config_lines: List[str],
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    detector: List[Dict[str, str]] = []
    backtest: List[Dict[str, str]] = []
    data: List[Dict[str, str]] = []

    for line in config_lines:
        if ">=" in line:
            detector.append({"key": line.split(">=")[0].strip(), "value": line.split(">=", 1)[1].strip()})
        else:
            backtest.append({"key": "config", "value": line})

    for key, val in sorted(kv.items()):
        row = {"key": key, "value": val}
        if key in RESULT_PARAM_KEYS:
            continue
        if key in DETECTOR_PARAM_KEYS or key.startswith("error_") or key.startswith("min_"):
            detector.append(row)
        elif key in DATA_PARAM_KEYS or key in {"start", "end"}:
            data.append(row)
        else:
            backtest.append(row)
    return detector, backtest, data


def build_run_meta(trades_path: Path) -> Dict[str, Any]:
    summary_path = _summary_path_for_trades(trades_path)
    kv, config_lines, notes = _parse_summary_txt(summary_path)
    det_rows, bt_rows, data_rows = _split_param_rows(kv, config_lines)
    return {
        "git": _git_info(),
        "detector": {**DETECTOR_META, "params": det_rows},
        "backtest_params": bt_rows,
        "data_params": data_rows,
        "results": {k: kv[k] for k in RESULT_PARAM_KEYS if k in kv},
        "notes": notes,
        "summary_file": summary_path.name if summary_path else None,
        "trades_file": trades_path.name,
    }


def _latest_trades_csv(outdir: Path) -> Path:
    found = latest_matching(
        outdir,
        "channel_touch_trades_*.csv",
        exclude_substr=("_trades_raw_",),
    )
    if found is None:
        raise FileNotFoundError(f"No channel_touch_trades_*.csv in {outdir}")
    return found


def _apply_rs_topn(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if n <= 0 or "rs_spy_126d" not in df.columns:
        return df
    ranked = df.sort_values(["buy_date", "rs_spy_126d"], ascending=[True, False], na_position="last")
    return ranked.groupby("buy_date", sort=False).head(int(n)).reset_index(drop=True)


def load_spy_close(
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    provider: str = "ALPACA",
    fallback_provider: str = "IB",
    merge_mode: str = "prefix",
) -> pd.Series:
    """SPY close for the overlay. Default IB prefix so the line covers pre-Alpaca trade years."""
    panels = load_ohlcv_many(
        ["SPY"],
        timeframe="1d",
        provider=provider,
        start=datetime(start.year, start.month, start.day),
        end=datetime(end.year, end.month, end.day),
        use_cache=True,
        workers=1,
        fallback_provider=fallback_provider or None,
        merge_mode=merge_mode or None,
    )
    spy = panels.get("SPY")
    if spy is None or spy.empty or "close" not in spy.columns:
        raise RuntimeError("SPY OHLCV unavailable for S&P 500 comparison")
    close = spy["close"].astype(float).copy()
    if not isinstance(close.index, pd.DatetimeIndex):
        close.index = pd.DatetimeIndex(close.index)
    if close.index.tz is not None:
        close.index = close.index.tz_convert(None)
    return close.sort_index()


def _series_clock_ts(df: pd.DataFrame, time_col: str, date_col: str) -> Tuple[pd.Series, pd.Series]:
    """Prefer buy_time/sell_time (15m bar); fall back to the calendar date."""
    dates = pd.to_datetime(df[date_col])
    if time_col not in df.columns:
        return dates, pd.Series(False, index=df.index)
    times = pd.to_datetime(df[time_col], errors="coerce")
    has_clock = times.notna()
    return times.where(has_clock, dates), has_clock


def _to_rth_clock(ts: object) -> pd.Timestamp:
    """IB 15m CSV clocks are UTC; show US RTH (America/New_York, EST/EDT)."""
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return t.tz_convert(RTH_TZ)


def _fmt_stamp(ts: object, has_clock: bool) -> str:
    if has_clock:
        return _to_rth_clock(ts).strftime("%Y-%m-%d %H:%M")
    stamp = pd.Timestamp(ts)
    return stamp.strftime("%Y-%m-%d")


def _cell_bool(val: object) -> bool:
    """CSV/object-safe bool (string 'False' must not become True)."""
    if val is None:
        return False
    try:
        if pd.isna(val):
            return False
    except (TypeError, ValueError):
        pass
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    s = str(val).strip().lower()
    if s in ("true", "1", "yes"):
        return True
    if s in ("false", "0", "no", "", "nan"):
        return False
    return False


def _cell_float(val: object) -> Optional[float]:
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    try:
        x = float(val)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def _cell_int(val: object) -> Optional[int]:
    x = _cell_float(val)
    if x is None:
        return None
    return int(round(x))


def _iso_utc_ms_from_cell(val: object) -> Optional[Tuple[str, int]]:
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    text = str(val).strip()
    if text in ("", "None", "nan"):
        return None
    t = pd.to_datetime(text, utc=True, errors="coerce")
    if t is None or pd.isna(t):
        t = pd.to_datetime(text, errors="coerce")
        if t is None or pd.isna(t):
            return None
        t = pd.Timestamp(t)
        if t.tzinfo is None:
            t = t.tz_localize("UTC")
        else:
            t = t.tz_convert("UTC")
    else:
        t = pd.Timestamp(t)
    ms = int(round(t.timestamp() * 1000))
    return t.strftime("%Y-%m-%dT%H:%M:%SZ"), ms


def _stamp_pair(row: pd.Series, time_col: str, ms_col: str, date_fallback: str = "") -> Optional[Tuple[str, int]]:
    ms_v = _cell_int(row[ms_col]) if ms_col in row.index else None
    raw = row[time_col] if time_col in row.index else None
    parsed = _iso_utc_ms_from_cell(raw)
    if ms_v is not None and parsed is not None:
        return parsed[0], ms_v
    if parsed is not None:
        return parsed
    if ms_v is not None:
        t = pd.Timestamp(ms_v, unit="ms", tz="UTC")
        return t.strftime("%Y-%m-%dT%H:%M:%SZ"), ms_v
    if date_fallback:
        return _iso_utc_ms_from_cell(row[date_fallback] if date_fallback in row.index else None)
    return None


def channel_json_for_row(row: pd.Series) -> Optional[dict]:
    """Compact CTF paste payload (Pine has no JSON.parse; keys are scanned as text).

    Prefer exact L1/L2/H2 rails from the backtest CSV. Older CSVs reconstruct a
    support-line second point at H2 from slope/width/channel_pos (src=approx).
    """
    sym = str(row.get("stock", "") or "").upper()
    fill = None
    if "buy_time" in row.index:
        fill = _iso_utc_ms_from_cell(row["buy_time"])
    if fill is None and "buy_date" in row.index:
        fill = _iso_utc_ms_from_cell(row["buy_date"])
    l1 = _stamp_pair(row, "l1_time", "l1_ms", "channel_start")
    l2 = _stamp_pair(row, "l2_time", "l2_ms")
    h2 = _stamp_pair(row, "h2_time", "h2_ms", "channel_end")
    l1p = _cell_float(row["l1_price"]) if "l1_price" in row.index else None
    l2p = _cell_float(row["l2_price"]) if "l2_price" in row.index else None
    width = _cell_float(row["channel_width"]) if "channel_width" in row.index else None
    src = "rails"
    if l1p is None or l2p is None or width is None or width <= 0 or l1 is None or l2 is None:
        recon = _reconstruct_channel_rails(row, fill)
        if recon is None:
            return None
        l1, l1p, l2, l2p, h2, width = recon
        src = "approx"
    if fill is None or l1 is None or l2 is None or l1p is None or l2p is None or width is None:
        return None
    entry_px = _cell_float(row["buy_price"]) if "buy_price" in row.index else None
    exit_px = _cell_float(row["sell_price"]) if "sell_price" in row.index else None
    exit_ts = None
    if "sell_time" in row.index:
        exit_ts = _iso_utc_ms_from_cell(row["sell_time"])
    if exit_ts is None and "sell_date" in row.index:
        exit_ts = _iso_utc_ms_from_cell(row["sell_date"])
    out: Dict[str, Any] = {
        "v": 1,
        "sym": sym,
        "src": src,
        "fill": fill[0],
        "fill_ms": fill[1],
        "en": fill[0],
        "en_ms": fill[1],
        "l1t": l1[0],
        "l1_ms": l1[1],
        "l1p": round(float(l1p), 6),
        "l2t": l2[0],
        "l2_ms": l2[1],
        "l2p": round(float(l2p), 6),
        "w": round(float(width), 6),
    }
    if entry_px is not None:
        out["enp"] = round(float(entry_px), 6)
    if exit_ts is not None:
        out["ex"] = exit_ts[0]
        out["ex_ms"] = exit_ts[1]
    if exit_px is not None:
        out["exp"] = round(float(exit_px), 6)
    if h2 is not None:
        out["h2t"] = h2[0]
        out["h2_ms"] = h2[1]
    return out


def _reconstruct_channel_rails(
    row: pd.Series,
    fill: Optional[Tuple[str, int]],
) -> Optional[Tuple[Tuple[str, int], float, Tuple[str, int], float, Optional[Tuple[str, int]], float]]:
    """Build L1 + H2-on-support from older CSVs that only stored pct geometry.

    Prefer ``channel_width_pct`` (width / L1 support * 100) + ``channel_pos`` —
    stable for L3 and H2 resist-break (pos near/above 1). Fall back to
    room/(1-pos) only when |1-pos| is large enough (in-channel L3).
    """
    entry = _cell_float(row["buy_price"]) if "buy_price" in row.index else None
    pos = _cell_float(row["channel_pos"]) if "channel_pos" in row.index else None
    room = _cell_float(row["room_to_resist_pct"]) if "room_to_resist_pct" in row.index else None
    slope_pct = _cell_float(row["slope_pct_per_bar"]) if "slope_pct_per_bar" in row.index else None
    width_pct = (
        _cell_float(row["channel_width_pct"]) if "channel_width_pct" in row.index else None
    )
    bars_span = _cell_int(row["bars_span"]) if "bars_span" in row.index else None
    wait_bars = _cell_int(row["wait_bars"]) if "wait_bars" in row.index else None
    l1 = _stamp_pair(row, "l1_time", "l1_ms", "channel_start")
    h2 = _stamp_pair(row, "h2_time", "h2_ms", "channel_end")
    if (
        fill is None
        or l1 is None
        or h2 is None
        or entry is None
        or entry <= 0
        or pos is None
        or slope_pct is None
        or bars_span is None
        or wait_bars is None
    ):
        return None
    bars_l1_fill = int(bars_span) + int(wait_bars)
    slope_term = slope_pct / 100.0 * float(bars_l1_fill)
    y1: Optional[float] = None
    width: Optional[float] = None
    if width_pct is not None and width_pct > 0:
        # entry = y1 * (1 + slope*bars_l1_fill) + pos * (y1 * width_pct/100)
        denom = 1.0 + slope_term + float(pos) * (width_pct / 100.0)
        if denom > 0:
            y1 = float(entry) / denom
            width = y1 * (width_pct / 100.0)
    if (y1 is None or width is None or not np.isfinite(y1) or not np.isfinite(width) or width <= 0) and (
        room is not None and abs(1.0 - float(pos)) >= 0.001
    ):
        resist = entry * (1.0 + room / 100.0)
        width = (resist - entry) / (1.0 - pos)
        if not np.isfinite(width) or width <= 0:
            return None
        support_fill = entry - pos * width
        denom = 1.0 + slope_term
        if denom <= 0:
            return None
        y1 = support_fill / denom
    if y1 is None or width is None or not np.isfinite(y1) or not np.isfinite(width) or y1 <= 0 or width <= 0:
        return None
    y_h2 = y1 * (1.0 + slope_pct / 100.0 * float(bars_span))
    if not np.isfinite(y_h2) or y_h2 <= 0:
        return None
    return l1, float(y1), h2, float(y_h2), h2, float(width)


def trades_to_raw(df: pd.DataFrame) -> List[dict]:
    """Compact trade rows for client-side recalculation.

    ``ord`` is the quality-filtered row order *before* the display sort, matching
    pandas ``sort_values(..., na_position='last')`` stability for same-day RS.
    Always emit ``rs`` (JSON null if missing) so the UI does not treat omitted
    keys as a broken ``-Infinity`` sort (NaN comparators scramble the pick).
    """
    out: List[dict] = []
    t = df.copy()
    t["buy_date"] = pd.to_datetime(t["buy_date"])
    t["sell_date"] = pd.to_datetime(t["sell_date"])
    t = t.reset_index(drop=True)
    t["_ord"] = np.arange(len(t), dtype=int)
    t["_buy_at"], t["_buy_clock"] = _series_clock_ts(t, "buy_time", "buy_date")
    t["_sell_at"], t["_sell_clock"] = _series_clock_ts(t, "sell_time", "sell_date")
    t = t.sort_values(["_sell_at", "_buy_at", "stock"]).reset_index(drop=True)
    has_rs = "rs_spy_126d" in t.columns
    for _, row in t.iterrows():
        item = {
            "symbol": str(row["stock"]),
            "buy": row["buy_date"].strftime("%Y-%m-%d"),
            "sell": row["sell_date"].strftime("%Y-%m-%d"),
            "entry": float(row["buy_price"]),
            "exit": float(row["sell_price"]),
            "gain": float(row["gain_pct"]),
            "hold": int(row["hold_days"]) if "hold_days" in row and pd.notna(row["hold_days"]) else None,
            "reason": str(row.get("exit_reason", "") or ""),
            "touch": int(row["touch_num"]) if "touch_num" in row and pd.notna(row["touch_num"]) else None,
            "ord": int(row["_ord"]),
        }
        if bool(row["_buy_clock"]):
            item["buy_at"] = _fmt_stamp(row["_buy_at"], True)
        if bool(row["_sell_clock"]):
            item["sell_at"] = _fmt_stamp(row["_sell_at"], True)
        if has_rs:
            v = row["rs_spy_126d"]
            item["rs"] = None if pd.isna(v) else float(v)
        if "resist_break" in t.columns:
            item["resist_break"] = _cell_bool(row["resist_break"])
        ch = channel_json_for_row(row)
        if ch:
            item["ch"] = ch
        out.append(item)
    return out


def _rs_missing(val: object) -> bool:
    if val is None:
        return True
    try:
        x = float(val)
    except (TypeError, ValueError):
        return True
    return not np.isfinite(x)


def filter_max_per_day_raw(trades: List[dict], max_per_day: int) -> List[dict]:
    """Same-day RS top-N: missing RS last, higher RS first, then ``ord`` (stable)."""
    if not max_per_day or max_per_day <= 0:
        return list(trades)
    by_day: Dict[str, List[dict]] = {}
    for t in trades:
        by_day.setdefault(str(t["buy"]), []).append(t)
    out: List[dict] = []
    for day in sorted(by_day):
        arr = list(by_day[day])
        arr.sort(
            key=lambda t: (
                _rs_missing(t.get("rs")),
                -(float(t["rs"]) if not _rs_missing(t.get("rs")) else 0.0),
                int(t["ord"]) if t.get("ord") is not None else 0,
            )
        )
        out.extend(arr[: int(max_per_day)])
    out.sort(key=lambda t: (str(t["sell"]), str(t["buy"]), str(t["symbol"])))
    return out


def spy_to_raw(spy_close: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> List[dict]:
    hist = spy_close.loc[start:end].dropna()
    if hist.empty:
        hist = spy_close[(spy_close.index >= start) & (spy_close.index <= end)].dropna()
    return [{"x": d.strftime("%Y-%m-%d"), "c": float(v)} for d, v in hist.items()]


def load_comparison_json(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError("Comparison JSON not found: %s" % path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data.get("rows"):
        raise ValueError("Comparison JSON must be an object with a non-empty rows list")
    return data


def render_html(
    *,
    raw_trades: List[dict],
    spy_closes: List[dict],
    defaults: Dict[str, Any],
    run_meta: Dict[str, Any],
    title: str,
    source: str,
    comparison: Optional[Dict[str, Any]] = None,
) -> str:
    payload = json.dumps(
        {
            "trades": raw_trades,
            "spy": spy_closes,
            "defaults": defaults,
            "runMeta": run_meta,
            "source": source,
            "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "comparison": comparison or {},
        },
        separators=(",", ":"),
    )
    # Escape </script> in JSON
    payload = payload.replace("</", "<\\/")

    has_rs = any(t.get("rs") is not None for t in raw_trades)
    rs_note = (
        "RS vs SPY (126d) available for same-day ranking (missing RS sorts last, stable)."
        if has_rs
        else "No RS column in trades CSV; max/day keeps first N by original row order."
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>{title}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  :root {{
    --bg:#0d1421; --panel:#131722; --panel2:#1e222d; --border:#2a2e39;
    --text:#d1d4dc; --muted:#787b86; --accent:#2962ff; --pos:#26a69a; --neg:#ef5350; --spy:#f7931a;
  }}
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ font-family:-apple-system,BlinkMacSystemFont,"Trebuchet MS",Roboto,Ubuntu,sans-serif; background:var(--bg); color:var(--text); line-height:1.45; }}
  .wrap {{ max-width:1280px; margin:0 auto; padding:20px; }}
  .header {{ background:var(--panel2); border:1px solid var(--border); border-radius:6px; padding:18px 20px; margin-bottom:12px; }}
  .header h1 {{ font-size:20px; font-weight:600; color:#fff; }}
  .header .sub {{ color:var(--muted); font-size:13px; margin-top:6px; }}
  .controls {{
    background:var(--panel2); border:1px solid var(--border); border-radius:6px;
    padding:14px 16px; margin-bottom:14px; display:grid;
    grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:12px 14px; align-items:end;
  }}
  .controls label {{ display:block; font-size:11px; color:var(--muted); text-transform:uppercase; letter-spacing:.04em; margin-bottom:4px; }}
  .controls input, .controls select {{
    width:100%; background:var(--bg); border:1px solid var(--border); color:var(--text);
    border-radius:4px; padding:8px 10px; font-size:13px;
  }}
  .controls .actions {{ display:flex; gap:8px; align-items:end; }}
  .btn {{
    background:var(--accent); color:#fff; border:none; border-radius:4px; padding:9px 14px;
    font-size:13px; font-weight:600; cursor:pointer;
  }}
  .btn.secondary {{ background:#2a2e39; color:var(--text); }}
  .btn:hover {{ filter:brightness(1.08); }}
  .hint {{ grid-column:1/-1; color:var(--muted); font-size:12px; }}
  .tabs {{ display:flex; flex-wrap:wrap; gap:4px; margin:12px 0; border-bottom:1px solid var(--border); }}
  .chart-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(380px,1fr)); gap:12px; margin-bottom:8px; }}
  .chart-box.dist {{ height:280px; }}
  .tab {{ background:transparent; border:none; color:var(--muted); padding:10px 14px; cursor:pointer; font-size:13px; font-weight:500; border-bottom:2px solid transparent; margin-bottom:-1px; }}
  .tab.active {{ color:#fff; border-bottom-color:var(--accent); }}
  .panel {{ display:none; background:var(--panel); border:1px solid var(--border); border-radius:6px; padding:16px; }}
  .panel.active {{ display:block; }}
  .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:10px; margin-bottom:16px; }}
  .card {{ background:var(--panel2); border:1px solid var(--border); border-radius:6px; padding:12px 14px; }}
  .card .label {{ color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.04em; }}
  .card .value {{ font-size:18px; font-weight:600; margin-top:4px; }}
  .pos {{ color:var(--pos); }} .neg {{ color:var(--neg); }}
  .muted {{ color:var(--muted); font-size:12px; margin-top:8px; }}
  .chart-box {{ background:var(--panel2); border:1px solid var(--border); border-radius:6px; padding:12px; margin-bottom:12px; height:320px; }}
  table {{ width:100%; border-collapse:collapse; font-size:12px; }}
  th, td {{ padding:8px 10px; border-bottom:1px solid var(--border); text-align:left; }}
  th {{ color:var(--muted); font-weight:500; position:sticky; top:0; background:var(--panel); }}
  th.sortable {{ cursor:pointer; user-select:none; white-space:nowrap; }}
  th.sortable:hover {{ color:var(--text); }}
  th.sortable .sort-ind {{ color:var(--accent); margin-left:4px; font-size:10px; }}
  td.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
  td.when {{ white-space:nowrap; font-variant-numeric:tabular-nums; }}
  .table-scroll {{ max-height:560px; overflow:auto; border:1px solid var(--border); border-radius:6px; }}
  .trade-filters {{
    display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:10px 12px;
    margin:10px 0 12px; align-items:end;
  }}
  .trade-filters label {{ display:block; font-size:11px; color:var(--muted); text-transform:uppercase; letter-spacing:.04em; margin-bottom:4px; }}
  .trade-filters input, .trade-filters select {{
    width:100%; background:var(--bg); border:1px solid var(--border); color:var(--text);
    border-radius:4px; padding:7px 9px; font-size:12px;
  }}
  .trade-filters .actions {{ display:flex; gap:8px; }}
  .metric-grid {{ display:grid; grid-template-columns:1fr 1fr 1fr 1fr; border:1px solid var(--border); border-radius:6px; overflow:hidden; margin-bottom:14px; }}
  .metric-grid .mh,.metric-grid .mc {{ padding:9px 12px; border-bottom:1px solid var(--border); border-right:1px solid var(--border); font-size:12px; }}
  .metric-grid .mh {{ background:var(--panel2); color:var(--muted); }}
  .metric-grid .mc:nth-child(4n) {{ border-right:none; }}
  h2 {{ font-size:15px; margin:8px 0 12px; font-weight:600; }}
  .note {{ color:var(--muted); font-size:12px; margin-top:12px; }}
  .meta-block {{ background:var(--panel2); border:1px solid var(--border); border-radius:6px; padding:12px 14px; margin-bottom:12px; font-size:13px; }}
  .meta-block .meta-label {{ color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.04em; }}
  .meta-block .meta-value {{ margin-top:4px; color:var(--text); word-break:break-word; }}
  .meta-kv td:first-child {{ color:var(--muted); width:38%; }}
  tr.book-hl td {{ background:rgba(38,166,154,0.12); }}
  #filterBooks {{ margin-bottom:16px; }}
  td.ch-cell {{ text-align:center; width:44px; }}
  .ch-info {{
    display:inline-flex; align-items:center; justify-content:center;
    width:18px; height:18px; border-radius:50%;
    border:1px solid var(--muted); color:var(--muted);
    font-size:11px; font-style:italic; font-weight:700;
    cursor:pointer; background:transparent; line-height:1;
  }}
  .ch-info:hover {{ border-color:var(--accent); color:#fff; }}
  .ch-info.copied {{ border-color:var(--pos); color:var(--pos); }}
  #copyToast {{
    position:fixed; bottom:18px; left:50%; transform:translateX(-50%);
    background:var(--panel2); border:1px solid var(--pos); color:var(--text);
    padding:8px 14px; border-radius:6px; font-size:12px; display:none; z-index:20;
  }}
  #chCopyModal {{
    display:none; position:fixed; inset:0; background:rgba(0,0,0,.55); z-index:30;
    align-items:center; justify-content:center;
  }}
  #chCopyModal.open {{ display:flex; }}
  #chCopyModal .box {{
    background:var(--panel); border:1px solid var(--border); border-radius:8px;
    padding:14px 16px; width:min(560px,92vw);
  }}
  #chCopyModal textarea {{
    width:100%; height:180px; margin:10px 0; background:var(--bg); color:var(--text);
    border:1px solid var(--border); border-radius:4px; font-family:ui-monospace,Consolas,monospace;
    font-size:11px; padding:8px;
  }}
</style>
</head>
<body>
<div class="wrap">
  <div class="header">
    <h1>{title}</h1>
    <div class="sub" id="subtitle">Loading…</div>
    <div class="sub">Source: {source} · Generated <span id="genAt"></span></div>
  </div>

  <div class="controls">
    <div>
      <label for="capital">Portfolio size ($)</label>
      <input id="capital" type="number" min="1000" step="1000" />
    </div>
    <div>
      <label for="sizeMode">Position sizing</label>
      <select id="sizeMode">
        <option value="fixed">Fixed $ per trade</option>
        <option value="pct_initial">% of initial capital</option>
        <option value="pct_equity">% of equity (compound)</option>
      </select>
    </div>
    <div>
      <label for="sizeVal" id="sizeValLabel">Size ($)</label>
      <input id="sizeVal" type="number" min="0" step="100" />
    </div>
    <div>
      <label for="friction">Friction / round-trip (%)</label>
      <input id="friction" type="number" min="0" max="5" step="0.05" />
    </div>
    <div>
      <label for="maxPerDay">Max entries / day (0 = all)</label>
      <input id="maxPerDay" type="number" min="0" max="50" step="1" />
    </div>
    <div>
      <label for="maxOpen">Max concurrent opens (0 = none)</label>
      <input id="maxOpen" type="number" min="0" max="100" step="1" />
    </div>
    <div>
      <label for="winCap">Win cap P&amp;L % (0 = none)</label>
      <input id="winCap" type="number" min="0" max="500" step="1" title="Winsorize: cap each trade gain at this %% before sizing" />
    </div>
    <div>
      <label for="excludeSym">Exclude symbols</label>
      <input id="excludeSym" type="text" placeholder="e.g. BETR, SANM" />
    </div>
    <div class="actions">
      <button class="btn" id="btnApply" type="button">Apply</button>
      <button class="btn secondary" id="btnReset" type="button">Reset</button>
    </div>
    <div class="hint">{rs_note} <b>Max entries/day</b> ranks same-day fills by RS among the embedded quality-filtered set (0 = all). Use <b>Win cap</b> / <b>Exclude</b> / <b>Max concurrent</b> for robustness &amp; capacity stress. Trade-table Max P&amp;L also has &quot;Push to equity&quot;.</div>
  </div>

  <div class="tabs">
    <button class="tab active" data-tab="overview">Overview</button>
    <button class="tab" data-tab="compare">Compare to S&amp;P 500</button>
    <button class="tab" data-tab="performance">Performance</button>
    <button class="tab" data-tab="robustness">Robustness</button>
    <button class="tab" data-tab="distributions">Distributions</button>
    <button class="tab" data-tab="trades">List of trades</button>
    <button class="tab" data-tab="monthly">Monthly</button>
    <button class="tab" data-tab="runinfo">Run info</button>
  </div>

  <div id="overview" class="panel active">
    <div id="filterBooks" style="display:none">
      <h2 id="filterBooksTitle">Filter books</h2>
      <div class="table-scroll" style="max-height:320px;margin-bottom:8px">
        <table>
          <thead><tr><th>Book</th><th>n</th><th>WR</th><th>E</th><th>PF</th></tr></thead>
          <tbody id="filterBooksBody"></tbody>
        </table>
      </div>
      <p class="muted" id="filterBooksNote"></p>
    </div>
    <div class="cards" id="overviewCards"></div>
    <h2>Equity curve vs S&amp;P 500</h2>
    <div class="chart-box"><canvas id="eqChart"></canvas></div>
    <h2>Drawdown</h2>
    <div class="chart-box"><canvas id="ddChart"></canvas></div>
    <p class="note" id="modelNote"></p>
  </div>

  <div id="compare" class="panel">
    <h2>Compare to S&amp;P 500 (SPY buy &amp; hold)</h2>
    <div class="cards" id="compareCards"></div>
    <h2>Normalized equity (start = 100)</h2>
    <div class="chart-box"><canvas id="cmpChart"></canvas></div>
    <div class="metric-grid" id="compareGrid"></div>
    <p class="note">SPY buy &amp; hold uses the same portfolio size over the strategy window. Strategy equity is forward-filled onto trading days for the overlay.</p>
  </div>

  <div id="performance" class="panel">
    <h2>Performance summary</h2>
    <div class="metric-grid" id="perfGrid"></div>
    <p class="muted" id="exitMix"></p>
  </div>

  <div id="robustness" class="panel">
    <h2>Robustness &amp; capacity</h2>
    <div class="cards" id="robustCards"></div>
    <h2>Outlier stripping / winsorize (trade P&amp;L % after friction)</h2>
    <div class="table-scroll" style="max-height:320px;margin-bottom:12px">
      <table>
        <thead><tr><th>Scenario</th><th>n</th><th>E%</th><th>Median%</th><th>Trimmed5%</th><th>PF</th><th>Win%</th></tr></thead>
        <tbody id="robustScenBody"></tbody>
      </table>
    </div>
    <h2>Bootstrap (with replacement)</h2>
    <div class="cards" id="bootCards"></div>
    <p class="muted" id="bootNote"></p>
    <h2>Capacity (greedy max concurrent)</h2>
    <div class="table-scroll" style="max-height:240px;margin-bottom:12px">
      <table>
        <thead><tr><th>Max open</th><th>n kept</th><th>E%</th><th>PF</th></tr></thead>
        <tbody id="capBody"></tbody>
      </table>
    </div>
    <p class="note">Criterion: if drop-top-1 or winsorized E flips negative / PF &lt; 1, treat edge as tail-fragile. Tail dependency = top-3 wins / gross profit of winners. Bootstrap uses trade P&amp;L %% (not path-dependent equity).</p>
  </div>

  <div id="distributions" class="panel">
    <h2>Activity</h2>
    <div class="cards" id="distActivityCards"></div>
    <div class="chart-grid">
      <div>
        <h2>Orders initiated per calendar day</h2>
        <div class="chart-box dist"><canvas id="distDayHistChart"></canvas></div>
      </div>
      <div>
        <h2>Daily fill count over time</h2>
        <div class="chart-box dist"><canvas id="distDayTsChart"></canvas></div>
      </div>
    </div>
    <h2>Trade P&amp;L % (after friction / win cap)</h2>
    <div class="cards" id="distPnlCards"></div>
    <div class="chart-grid">
      <div>
        <h2>Histogram of trade P&amp;L %</h2>
        <div class="chart-box dist"><canvas id="distPnlHistChart"></canvas></div>
      </div>
      <div>
        <h2>Hold bars</h2>
        <div class="chart-box dist"><canvas id="distHoldChart"></canvas></div>
      </div>
    </div>
    <div class="chart-grid">
      <div>
        <h2>Entries by weekday</h2>
        <div class="chart-box dist"><canvas id="distDowChart"></canvas></div>
      </div>
      <div id="distHourWrap">
        <h2>Entries by hour (RTH, America/New_York)</h2>
        <div class="chart-box dist"><canvas id="distHourChart"></canvas></div>
      </div>
    </div>
    <div class="chart-grid">
      <div>
        <h2>Concurrent opens at entry</h2>
        <div class="chart-box dist"><canvas id="distConcChart"></canvas></div>
      </div>
      <div>
        <h2>Mean P&amp;L % by exit reason</h2>
        <div class="chart-box dist"><canvas id="distExitChart"></canvas></div>
      </div>
    </div>
    <h2>Trade P&amp;L quantiles</h2>
    <div class="metric-grid" id="distQuantGrid"></div>
    <h2>Trades per symbol (top 20 by count)</h2>
    <div class="cards" id="distSymCards"></div>
    <div class="table-scroll" style="max-height:320px;margin-bottom:12px">
      <table>
        <thead><tr><th>Symbol</th><th>n</th><th>WR</th><th>E%</th><th>PF</th></tr></thead>
        <tbody id="distSymBody"></tbody>
      </table>
    </div>
    <p class="note">Distributions use the current Apply filters (sizing does not change counts; friction / win cap change P&amp;L %). Orders/day is calendar buy_date. Histogram x-range is p01–p99 with overflow bars. Hour uses buy_time converted to RTH (America/New_York; CSV is UTC). Zero-fill days use SPY trading dates when available.</p>
  </div>

  <div id="trades" class="panel">
    <h2>List of trades</h2>
    <div class="trade-filters">
      <div>
        <label for="tfSymbol">Symbol</label>
        <input id="tfSymbol" type="text" placeholder="e.g. GLD, AAPL" />
      </div>
      <div>
        <label for="tfOutcome">Outcome</label>
        <select id="tfOutcome">
          <option value="all">All</option>
          <option value="win">Winners</option>
          <option value="loss">Losers</option>
        </select>
      </div>
      <div>
        <label for="tfExit">Exit reason</label>
        <select id="tfExit"><option value="all">All</option></select>
      </div>
      <div>
        <label for="tfMinPnl">Min P&amp;L %</label>
        <input id="tfMinPnl" type="number" step="0.1" placeholder="any" />
      </div>
      <div>
        <label for="tfMaxPnl">Max P&amp;L %</label>
        <input id="tfMaxPnl" type="number" step="0.1" placeholder="any" />
      </div>
      <div>
        <label for="tfFrom">Entry from</label>
        <input id="tfFrom" type="date" />
      </div>
      <div>
        <label for="tfTo">Entry to</label>
        <input id="tfTo" type="date" />
      </div>
      <div class="actions">
        <button class="btn secondary" id="tfClear" type="button">Clear filters</button>
        <button class="btn" id="tfPushEquity" type="button" title="Copy Max P&amp;L into Win cap and re-run equity">Push Max P&amp;L to equity</button>
      </div>
    </div>
    <p class="muted" id="tradesNote"></p>
    <div class="table-scroll">
      <table id="tradesTable">
        <thead>
          <tr>
            <th class="sortable" data-sort="n">#</th>
            <th class="sortable" data-sort="symbol">Symbol</th>
            <th>Channel</th>
            <th class="sortable" data-sort="signal">Signal</th>
            <th class="sortable" data-sort="entry_date">Entry</th>
            <th class="sortable" data-sort="exit_date">Exit</th>
            <th class="sortable" data-sort="entry_price">Entry px</th>
            <th class="sortable" data-sort="exit_price">Exit px</th>
            <th class="sortable" data-sort="pnl">P&amp;L $</th>
            <th class="sortable" data-sort="pnl_pct">P&amp;L %</th>
            <th class="sortable" data-sort="cum_pnl">Cum. P&amp;L</th>
            <th class="sortable" data-sort="hold_days">Bars</th>
            <th class="sortable" data-sort="exit_reason">Exit</th>
          </tr>
        </thead>
        <tbody id="tradesBody"></tbody>
      </table>
    </div>
  </div>

  <div id="monthly" class="panel">
    <h2>Monthly net profit (by exit month)</h2>
    <div class="table-scroll" style="max-height:640px">
      <table>
        <thead><tr><th>Month</th><th>Net P&amp;L</th></tr></thead>
        <tbody id="monthBody"></tbody>
      </table>
    </div>
  </div>

  <div id="runinfo" class="panel">
    <div id="filterBooksRun" style="display:none">
      <h2 id="filterBooksRunTitle">Filter books</h2>
      <div class="table-scroll" style="max-height:280px;margin-bottom:12px">
        <table>
          <thead><tr><th>Book</th><th>n</th><th>WR</th><th>E</th><th>PF</th></tr></thead>
          <tbody id="filterBooksRunBody"></tbody>
        </table>
      </div>
    </div>
    <h2>Git branch</h2>
    <div class="metric-grid" id="gitGrid"></div>
    <h2>Detector</h2>
    <div id="detectorBlock"></div>
    <h2>Parameters</h2>
    <h3 style="font-size:13px;color:var(--muted);margin:8px 0 8px;">Backtest / execution</h3>
    <div class="table-scroll" style="max-height:360px;margin-bottom:12px">
      <table class="meta-kv"><tbody id="backtestParamsBody"></tbody></table>
    </div>
    <h3 style="font-size:13px;color:var(--muted);margin:8px 0 8px;">Data / universe</h3>
    <div class="table-scroll" style="max-height:240px;margin-bottom:12px">
      <table class="meta-kv"><tbody id="dataParamsBody"></tbody></table>
    </div>
    <p class="note" id="runMetaNote"></p>
  </div>
</div>
<div id="copyToast"></div>
<div id="chCopyModal" role="dialog" aria-modal="true" aria-labelledby="chCopyTitle">
  <div class="box">
    <h2 id="chCopyTitle">Channel JSON (CTF paste)</h2>
    <p class="muted">Paste into Channel Touch Fill Viewer → Channel JSON.</p>
    <textarea id="chCopyText" readonly></textarea>
    <div class="trade-filters" style="margin:0">
      <div class="actions">
        <button class="btn" type="button" id="chCopyAgain">Copy</button>
        <button class="btn secondary" type="button" id="chCopyClose">Close</button>
      </div>
    </div>
  </div>
</div>

<script>
const RAW = {payload};

function stampBuy(t) {{ return t.buy_at || t.buy; }}
function stampSell(t) {{ return t.sell_at || t.sell; }}
function dayKey(t) {{ return String(t.buy || '').slice(0, 10); }}

function money(x, signed) {{
  const a = Math.abs(x);
  const s = a.toLocaleString(undefined, {{minimumFractionDigits:2, maximumFractionDigits:2}});
  if (!signed) return '$' + s;
  return (x >= 0 ? '+$' : '-$') + s;
}}
function pct(x, signed) {{
  const s = x.toFixed(2) + '%';
  if (!signed) return s;
  return (x >= 0 ? '+' : '') + s;
}}
function cls(x) {{ return x >= 0 ? 'pos' : 'neg'; }}
function setHTML(id, html) {{ const el = document.getElementById(id); if (el) el.innerHTML = html; }}

function readParams() {{
  return {{
    capital: Math.max(1000, Number(document.getElementById('capital').value) || 100000),
    sizeMode: document.getElementById('sizeMode').value,
    sizeVal: Math.max(0, Number(document.getElementById('sizeVal').value) || 0),
    friction: Math.max(0, Number(document.getElementById('friction').value) || 0),
    maxPerDay: Math.max(0, Math.floor(Number(document.getElementById('maxPerDay').value) || 0)),
    maxOpen: Math.max(0, Math.floor(Number(document.getElementById('maxOpen').value) || 0)),
    winCap: Math.max(0, Number(document.getElementById('winCap').value) || 0),
    excludeSym: (document.getElementById('excludeSym').value || '').trim().toUpperCase(),
  }};
}}

function syncSizeLabel() {{
  const mode = document.getElementById('sizeMode').value;
  document.getElementById('sizeValLabel').textContent =
    mode === 'fixed' ? 'Size ($)' : 'Size (%)';
}}

function parseExclude(p) {{
  if (!p.excludeSym) return new Set();
  return new Set(p.excludeSym.split(/[,\\s]+/).filter(Boolean));
}}

function rsMissing(t) {{
  return t.rs === undefined || t.rs === null || (typeof t.rs === 'number' && Number.isNaN(t.rs));
}}
function cmpRsThenOrd(a, b) {{
  const aMiss = rsMissing(a);
  const bMiss = rsMissing(b);
  if (aMiss !== bMiss) return aMiss ? 1 : -1;
  if (!aMiss && a.rs !== b.rs) return b.rs - a.rs;
  const oa = (a.ord === undefined || a.ord === null) ? 0 : a.ord;
  const ob = (b.ord === undefined || b.ord === null) ? 0 : b.ord;
  return oa - ob;
}}

function filterMaxPerDay(trades, maxPerDay) {{
  if (!maxPerDay || maxPerDay <= 0) return trades.slice();
  const byDay = {{}};
  for (const t of trades) {{
    const day = dayKey(t);
    if (!byDay[day]) byDay[day] = [];
    byDay[day].push(t);
  }}
  const out = [];
  Object.keys(byDay).sort().forEach(day => {{
    const arr = byDay[day].slice();
    arr.sort(cmpRsThenOrd);
    out.push(...arr.slice(0, maxPerDay));
  }});
  out.sort((a, b) => (a.sell < b.sell ? -1 : a.sell > b.sell ? 1 : a.buy < b.buy ? -1 : a.symbol.localeCompare(b.symbol)));
  return out;
}}

function filterExclude(trades, excludeSet) {{
  if (!excludeSet || !excludeSet.size) return trades.slice();
  return trades.filter(t => !excludeSet.has(String(t.symbol).toUpperCase()));
}}

function filterMaxOpen(trades, maxOpen) {{
  if (!maxOpen || maxOpen <= 0) return trades.slice();
  const ordered = trades.slice().sort((a, b) => {{
    const ab = stampBuy(a), bb = stampBuy(b);
    if (ab !== bb) return ab < bb ? -1 : 1;
    return cmpRsThenOrd(a, b);
  }});
  const kept = [];
  const openExits = [];
  for (const t of ordered) {{
    const buy = stampBuy(t);
    while (openExits.length && openExits[0] < buy) openExits.shift();
    if (openExits.length >= maxOpen) continue;
    kept.push(t);
    openExits.push(stampSell(t));
    openExits.sort();
  }}
  kept.sort((a, b) => {{
    const as = stampSell(a), bs = stampSell(b);
    if (as !== bs) return as < bs ? -1 : 1;
    const ab = stampBuy(a), bb = stampBuy(b);
    if (ab !== bb) return ab < bb ? -1 : 1;
    return a.symbol.localeCompare(b.symbol);
  }});
  return kept;
}}

function prepareTrades(raw, p) {{
  let t = filterExclude(raw, parseExclude(p));
  t = filterMaxPerDay(t, p.maxPerDay);
  t = filterMaxOpen(t, p.maxOpen);
  return t;
}}

function notionalFor(equity, capital, p) {{
  if (p.sizeMode === 'pct_equity') return equity * (p.sizeVal / 100);
  if (p.sizeMode === 'pct_initial') return capital * (p.sizeVal / 100);
  return p.sizeVal;
}}

function cappedGain(t, p) {{
  let g = t.gain - p.friction;
  if (p.winCap > 0 && g > p.winCap) g = p.winCap;
  return g;
}}

function simulate(trades, p) {{
  let equity = p.capital;
  let peak = p.capital;
  let cum = 0;
  const rows = [];
  const eqCurve = [];
  const ddCurve = [];
  const monthly = {{}};
  const exits = {{}};
  if (trades.length) {{
    const firstBuy = trades.reduce((a,t)=>t.buy<a?t.buy:a, trades[0].buy);
    eqCurve.push({{x: firstBuy, y: p.capital}});
    ddCurve.push({{x: firstBuy, y: 0}});
  }}
  let wins = 0, losses = 0, gp = 0, gl = 0;
  let sumHold = 0, nHold = 0;
  let largestWin = 0, largestLoss = 0;
  let avgWinSum = 0, avgLossSum = 0;
  const gainPcts = [];

  for (let i = 0; i < trades.length; i++) {{
    const t = trades[i];
    const notion = Math.max(0, notionalFor(equity, p.capital, p));
    const gainPct = cappedGain(t, p);
    gainPcts.push(gainPct);
    const pnl = notion * gainPct / 100;
    cum += pnl;
    equity = p.capital + cum;
    if (equity > peak) peak = equity;
    const dd = peak > 0 ? (peak - equity) / peak * 100 : 0;
    eqCurve.push({{x: t.sell, y: equity}});
    ddCurve.push({{x: t.sell, y: -dd}});
    const month = t.sell.slice(0, 7);
    monthly[month] = (monthly[month] || 0) + pnl;
    exits[t.reason || 'unknown'] = (exits[t.reason || 'unknown'] || 0) + 1;
    if (pnl > 0) {{ wins++; gp += pnl; avgWinSum += pnl; largestWin = Math.max(largestWin, pnl); }}
    else {{ losses++; gl += -pnl; avgLossSum += pnl; largestLoss = Math.min(largestLoss, pnl); }}
    if (t.hold != null) {{ sumHold += t.hold; nHold++; }}
    rows.push({{
      n: i + 1, symbol: t.symbol, signal: t.resist_break ? 'Resist-break' : (t.touch != null ? ('Touch ' + t.touch) : 'Long'),
      entry_date: stampBuy(t), exit_date: stampSell(t), entry_price: t.entry, exit_price: t.exit,
      pnl, pnl_pct: gainPct, cum_pnl: cum, hold_days: t.hold, exit_reason: t.reason,
      ch: t.ch || null
    }});
  }}

  const n = trades.length;
  const net = equity - p.capital;
  const ret = p.capital > 0 ? net / p.capital * 100 : 0;
  const wr = n ? wins / n * 100 : 0;
  const pf = gl > 1e-12 ? gp / gl : (gp > 0 ? Infinity : 0);
  let maxDd = 0;
  for (const pnt of ddCurve) maxDd = Math.max(maxDd, -pnt.y);

  return {{
    metrics: {{
      capital: p.capital, final: equity, net, ret, n, wins, losses, wr, pf,
      gp, gl,
      avgTrade: n ? cum / n : 0,
      avgWin: wins ? avgWinSum / wins : 0,
      avgLoss: losses ? avgLossSum / losses : 0,
      avgBars: nHold ? sumHold / nHold : 0,
      maxDd, largestWin, largestLoss,
      symbols: new Set(trades.map(t => t.symbol)).size,
      from: trades.length ? trades.reduce((a,t)=>t.buy<a?t.buy:a, trades[0].buy) : '',
      to: trades.length ? trades.reduce((a,t)=>t.sell>a?t.sell:a, trades[0].sell) : '',
      exits, gainPcts
    }},
    rows, eqCurve, ddCurve, monthly, params: p, trades
  }};
}}

function mean(arr) {{ return arr.length ? arr.reduce((a,b)=>a+b,0)/arr.length : 0; }}
function median(arr) {{
  if (!arr.length) return 0;
  const s = arr.slice().sort((a,b)=>a-b);
  const m = Math.floor(s.length/2);
  return s.length % 2 ? s[m] : (s[m-1]+s[m])/2;
}}
function trimmedMean(arr, frac) {{
  if (!arr.length) return 0;
  const s = arr.slice().sort((a,b)=>a-b);
  const lo = Math.floor(frac * s.length);
  const hi = Math.ceil((1-frac) * s.length);
  if (hi <= lo) return mean(s);
  return mean(s.slice(lo, hi));
}}
function pfOf(arr) {{
  let gp=0, gl=0;
  for (const x of arr) {{ if (x > 0) gp += x; else gl += -x; }}
  if (gl < 1e-12) return gp > 0 ? Infinity : 0;
  return gp / gl;
}}
function dropTopN(arr, n) {{
  const s = arr.slice().sort((a,b)=>b-a);
  const drop = new Set(s.slice(0, n));
  // drop by value multiset: remove n largest occurrences
  const sortedIdx = arr.map((v,i)=>[v,i]).sort((a,b)=>b[0]-a[0]).slice(0,n).map(x=>x[1]);
  const kill = new Set(sortedIdx);
  return arr.filter((_,i)=>!kill.has(i));
}}
function winsor(arr, cap) {{
  if (!(cap > 0)) return arr.slice();
  return arr.map(x => x > cap ? cap : x);
}}
function concurrentStats(trades) {{
  if (!trades.length) return {{max:0,p95:0,median:0}};
  const events = [];
  for (const t of trades) {{
    events.push({{d:stampBuy(t), dn:1}});
    events.push({{d:stampSell(t), dn:-1}});
  }}
  events.sort((a,b)=> a.d < b.d ? -1 : a.d > b.d ? 1 : a.dn - b.dn);
  let cur=0, max=0;
  const samples=[];
  for (const e of events) {{
    cur += e.dn;
    if (e.dn > 0) {{ samples.push(cur); if (cur > max) max = cur; }}
  }}
  samples.sort((a,b)=>a-b);
  const med = samples.length ? samples[Math.floor(samples.length/2)] : 0;
  const p95 = samples.length ? samples[Math.min(samples.length-1, Math.floor(0.95*(samples.length-1)))] : 0;
  return {{max, p95, median: med}};
}}
function bootstrapGains(arr, nIter) {{
  if (!arr.length) return {{pctNegMean:0, pctNegSum:0, pctPfLt1:0, meanP05:0, meanP50:0, meanP95:0}};
  const means=[], sums=[], pfs=[];
  for (let i=0;i<nIter;i++) {{
    const sample = new Array(arr.length);
    for (let j=0;j<arr.length;j++) sample[j] = arr[(Math.random()*arr.length)|0];
    const m = mean(sample);
    const s = sample.reduce((a,b)=>a+b,0);
    means.push(m); sums.push(s); pfs.push(pfOf(sample));
  }}
  means.sort((a,b)=>a-b); sums.sort((a,b)=>a-b);
  const q = (a,p) => a[Math.min(a.length-1, Math.floor(p*(a.length-1)))];
  return {{
    pctNegMean: 100 * means.filter(x=>x<0).length / nIter,
    pctNegSum: 100 * sums.filter(x=>x<0).length / nIter,
    pctPfLt1: 100 * pfs.filter(x=>x<1).length / nIter,
    meanP05: q(means,0.05), meanP50: q(means,0.5), meanP95: q(means,0.95)
  }};
}}
function scenRow(label, arr) {{
  const wins = arr.filter(x=>x>0);
  return {{
    label, n: arr.length,
    e: mean(arr), med: median(arr), trim: trimmedMean(arr, 0.05),
    pf: pfOf(arr), wr: arr.length ? 100*wins.length/arr.length : 0
  }};
}}
function renderRobustness(sim) {{
  const gains = sim.metrics.gainPcts || [];
  const trades = sim.trades || [];
  const base = scenRow('baseline', gains);
  const rows = [
    base,
    scenRow('drop top-1', dropTopN(gains, 1)),
    scenRow('drop top-3', dropTopN(gains, 3)),
    scenRow('drop top-5', dropTopN(gains, 5)),
    scenRow('drop top 5% winners', dropTopN(gains, Math.max(1, Math.ceil(0.05*gains.length)))),
    scenRow('winsor 50%', winsor(gains, 50)),
    scenRow('winsor 30%', winsor(gains, 30)),
    scenRow('winsor 20%', winsor(gains, 20)),
  ];
  const sortedWins = gains.filter(x=>x>0).slice().sort((a,b)=>b-a);
  const gross = sortedWins.reduce((a,b)=>a+b,0);
  const top3 = sortedWins.slice(0,3).reduce((a,b)=>a+b,0);
  const tdr = gross > 0 ? top3/gross : 0;
  const expo = concurrentStats(trades);
  const boot = bootstrapGains(gains, 2000);
  const d1 = rows[1];
  let verdict = '';
  if (d1.e <= 0 || d1.pf < 1) verdict = 'FRAGILE: edge fails drop-top-1 criterion';
  else if (tdr >= 0.4) verdict = 'CAUTION: top-3 tail dependency >= 40% (still positive after drop-top-1)';
  else verdict = 'SOFT PASS: survives drop-top-1; check winsor / bootstrap';

  setHTML('robustCards', `
    <div class="card"><div class="label">Verdict</div><div class="value" style="font-size:14px">${{verdict}}</div></div>
    <div class="card"><div class="label">Mean trade %</div><div class="value ${{cls(base.e)}}">${{pct(base.e,true)}}</div></div>
    <div class="card"><div class="label">Median trade %</div><div class="value ${{cls(base.med)}}">${{pct(base.med,true)}}</div></div>
    <div class="card"><div class="label">Trimmed mean 5%</div><div class="value ${{cls(base.trim)}}">${{pct(base.trim,true)}}</div></div>
    <div class="card"><div class="label">Top-3 / gross wins</div><div class="value">${{(tdr*100).toFixed(1)}}%</div></div>
    <div class="card"><div class="label">Concurrent open max</div><div class="value">${{expo.max}}</div></div>
    <div class="card"><div class="label">Concurrent p95 / med</div><div class="value">${{expo.p95}} / ${{expo.median}}</div></div>
    <div class="card"><div class="label">Trades in sim</div><div class="value">${{trades.length}}</div></div>
  `);
  setHTML('robustScenBody', rows.map(r => `
    <tr>
      <td>${{r.label}}</td><td class="num">${{r.n}}</td>
      <td class="num ${{cls(r.e)}}">${{pct(r.e,true)}}</td>
      <td class="num ${{cls(r.med)}}">${{pct(r.med,true)}}</td>
      <td class="num ${{cls(r.trim)}}">${{pct(r.trim,true)}}</td>
      <td class="num">${{Number.isFinite(r.pf)?r.pf.toFixed(3):'inf'}}</td>
      <td class="num">${{pct(r.wr,false)}}</td>
    </tr>`).join(''));
  setHTML('bootCards', `
    <div class="card"><div class="label">P(mean &lt; 0)</div><div class="value">${{boot.pctNegMean.toFixed(1)}}%</div></div>
    <div class="card"><div class="label">P(sum &lt; 0)</div><div class="value">${{boot.pctNegSum.toFixed(1)}}%</div></div>
    <div class="card"><div class="label">P(PF &lt; 1)</div><div class="value">${{boot.pctPfLt1.toFixed(1)}}%</div></div>
    <div class="card"><div class="label">Mean E p05 / p50 / p95</div><div class="value" style="font-size:14px">${{pct(boot.meanP05,true)}} / ${{pct(boot.meanP50,true)}} / ${{pct(boot.meanP95,true)}}</div></div>
  `);
  setHTML('bootNote', '2000 bootstrap resamples of the current filtered trade P&amp;L %% list (after friction / win-cap / excludes / capacity).');
  const capRows = [0,5,10,15,20].map(c => {{
    const pp = Object.assign({{}}, sim.params, {{maxOpen: c}});
    const tt = prepareTrades(RAW.trades, pp);
    const gg = tt.map(x => cappedGain(x, pp));
    return scenRow(c === 0 ? 'unlimited' : String(c), gg);
  }});
  setHTML('capBody', capRows.map(r => `
    <tr><td>${{r.label}}</td><td class="num">${{r.n}}</td>
    <td class="num ${{cls(r.e)}}">${{pct(r.e,true)}}</td>
    <td class="num">${{Number.isFinite(r.pf)?r.pf.toFixed(3):'inf'}}</td></tr>`).join(''));
}}

function quantile(arr, p) {{
  if (!arr.length) return 0;
  const s = arr.slice().sort((a,b)=>a-b);
  const i = (s.length - 1) * p;
  const lo = Math.floor(i), hi = Math.ceil(i);
  if (lo === hi) return s[lo];
  return s[lo] + (s[hi] - s[lo]) * (i - lo);
}}
function stdev(arr) {{
  if (arr.length < 2) return 0;
  const m = mean(arr);
  return Math.sqrt(arr.reduce((s,x)=>s+(x-m)*(x-m),0) / (arr.length - 1));
}}
function parseYmd(s) {{
  const p = String(s || '').slice(0, 10).split('-');
  if (p.length < 3) return null;
  const y = Number(p[0]), mo = Number(p[1]), d = Number(p[2]);
  if (!y || !mo || !d) return null;
  return new Date(y, mo - 1, d);
}}
function buyHour(t) {{
  const s = String(stampBuy(t) || '');
  const sp = s.indexOf(' ');
  if (sp < 0) return null;
  const rest = s.slice(sp + 1);
  const colon = rest.indexOf(':');
  if (colon < 0) return null;
  const h = Number(rest.slice(0, colon));
  return Number.isFinite(h) ? h : null;
}}
function intCounts(values, minV, maxV) {{
  const labels = [], counts = [];
  const bag = {{}};
  for (const v of values) bag[v] = (bag[v] || 0) + 1;
  for (let k = minV; k <= maxV; k++) {{
    labels.push(String(k));
    counts.push(bag[k] || 0);
  }}
  return {{ labels, counts }};
}}
function autoHist(values, maxBins) {{
  const s = values.filter(x => Number.isFinite(x)).slice().sort((a,b)=>a-b);
  if (!s.length) return {{ labels: [], counts: [], colors: [], nLo: 0, nHi: 0, lo: 0, hi: 0 }};
  const lo = quantile(s, 0.01), hi = quantile(s, 0.99);
  const span = hi - lo;
  const q1 = quantile(s, 0.25), q3 = quantile(s, 0.75);
  const iqr = q3 - q1;
  let h = iqr > 0 ? 2 * iqr / Math.cbrt(s.length) : 0;
  if (!(h > 0) || !(span > 0)) h = span > 0 ? span / 30 : 0.1;
  let nBins = span > 0 ? Math.ceil(span / h) : 12;
  nBins = Math.max(12, Math.min(maxBins || 40, nBins));
  const edges = [];
  for (let i = 0; i <= nBins; i++) edges.push(lo + (span || 1) * i / nBins);
  const counts = new Array(nBins).fill(0);
  let nLo = 0, nHi = 0;
  for (const v of s) {{
    if (v < lo) {{ nLo++; continue; }}
    if (v > hi) {{ nHi++; continue; }}
    let b = Math.floor((v - lo) / (span || 1) * nBins);
    if (b >= nBins) b = nBins - 1;
    if (b < 0) b = 0;
    counts[b]++;
  }}
  const labels = [], colors = [];
  for (let i = 0; i < nBins; i++) {{
    const mid = (edges[i] + edges[i+1]) / 2;
    labels.push(edges[i].toFixed(1) + ' to ' + edges[i+1].toFixed(1));
    colors.push(mid >= 0 ? 'rgba(38,166,154,0.75)' : 'rgba(239,83,80,0.75)');
  }}
  if (nLo) {{ labels.unshift('< p01'); counts.unshift(nLo); colors.unshift('rgba(239,83,80,0.45)'); }}
  if (nHi) {{ labels.push('> p99'); counts.push(nHi); colors.push('rgba(38,166,154,0.45)'); }}
  return {{ labels, counts, colors, nLo, nHi, lo, hi }};
}}
function dayFillCounts(trades, spy) {{
  const byDay = {{}};
  for (const t of trades) {{
    const d = dayKey(t);
    if (!d) continue;
    byDay[d] = (byDay[d] || 0) + 1;
  }}
  const fillDays = Object.keys(byDay).sort();
  const fillCounts = fillDays.map(d => byDay[d]);
  let calendar = [];
  if (spy && spy.length) {{
    const from = fillDays[0], to = fillDays[fillDays.length - 1];
    calendar = spy.map(p => p.x).filter(x => (!from || x >= from) && (!to || x <= to));
  }}
  const calCounts = calendar.length
    ? calendar.map(d => byDay[d] || 0)
    : fillCounts.slice();
  return {{ byDay, fillDays, fillCounts, calendar, calCounts }};
}}
function concurrentAtEntry(trades) {{
  if (!trades.length) return [];
  const events = [];
  for (const t of trades) {{
    events.push({{ d: stampBuy(t), dn: 1 }});
    events.push({{ d: stampSell(t), dn: -1 }});
  }}
  events.sort((a,b)=> a.d < b.d ? -1 : a.d > b.d ? 1 : a.dn - b.dn);
  let cur = 0;
  const samples = [];
  for (const e of events) {{
    cur += e.dn;
    if (e.dn > 0) samples.push(cur);
  }}
  return samples;
}}
function symbolStats(trades, gains) {{
  const bag = {{}};
  trades.forEach((t, i) => {{
    const s = t.symbol;
    if (!bag[s]) bag[s] = [];
    bag[s].push(gains[i]);
  }});
  return Object.keys(bag).map(sym => {{
    const arr = bag[sym];
    return {{ symbol: sym, n: arr.length, e: mean(arr), pf: pfOf(arr), wr: 100 * arr.filter(x=>x>0).length / arr.length }};
  }}).sort((a,b)=> b.n - a.n || b.e - a.e);
}}

const distChartObjs = {{}};
const distChartOpts = {{
  responsive: true, maintainAspectRatio: false,
  plugins: {{ legend: {{ display: false }}, tooltip: {{ callbacks: {{
    title: (items) => items.length ? items[0].label : '',
    label: (item) => ' n = ' + item.raw
  }} }} }},
  scales: {{
    x: {{ ticks: {{ color:'#787b86', maxRotation: 45, minRotation: 0, autoSkip: true, maxTicksLimit: 16 }}, grid: {{ color:'#2a2e39' }} }},
    y: {{ ticks: {{ color:'#787b86' }}, grid: {{ color:'#2a2e39' }}, beginAtZero: true }}
  }}
}};
function upsertBarChart(key, canvasId, labels, data, colors, extra) {{
  const el = document.getElementById(canvasId);
  if (!el) return;
  if (distChartObjs[key]) {{ distChartObjs[key].destroy(); distChartObjs[key] = null; }}
  distChartObjs[key] = new Chart(el, {{
    type: 'bar',
    data: {{ labels, datasets: [{{ label: extra && extra.seriesLabel ? extra.seriesLabel : 'count', data, backgroundColor: colors, borderWidth: 0 }}] }},
    options: extra && extra.options ? extra.options : distChartOpts
  }});
}}
function upsertLineChart(key, canvasId, labels, data, color) {{
  const el = document.getElementById(canvasId);
  if (!el) return;
  if (distChartObjs[key]) {{ distChartObjs[key].destroy(); distChartObjs[key] = null; }}
  distChartObjs[key] = new Chart(el, {{
    type: 'line',
    data: {{ labels, datasets: [{{ label: 'Fills / day', data, borderColor: color, pointRadius: 0, borderWidth: 1.2, tension: 0.05 }}] }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{ legend: {{ display: false }} }},
      scales: {{
        x: {{ ticks: {{ color:'#787b86', maxTicksLimit: 10 }}, grid: {{ color:'#2a2e39' }} }},
        y: {{ ticks: {{ color:'#787b86' }}, grid: {{ color:'#2a2e39' }}, beginAtZero: true }}
      }}
    }}
  }});
}}
function resizeDistCharts() {{
  Object.keys(distChartObjs).forEach(k => {{ try {{ distChartObjs[k].resize(); }} catch (e) {{}} }});
}}

function renderDistributions(sim) {{
  const trades = sim.trades || [];
  const gains = sim.metrics.gainPcts || [];
  const day = dayFillCounts(trades, RAW.spy);
  const fillN = day.fillCounts;
  const calN = day.calCounts;
  const maxDay = fillN.length ? Math.max.apply(null, fillN) : 0;
  const dayHist = intCounts(fillN, 1, Math.max(1, maxDay));
  const conc = concurrentAtEntry(trades);
  const maxConc = conc.length ? Math.max.apply(null, conc) : 0;
  const concHist = intCounts(conc, 1, Math.max(1, maxConc));
  const holds = trades.map(t => t.hold).filter(x => x != null && Number.isFinite(x));
  const holdCap = holds.length ? Math.max(1, Math.round(quantile(holds, 0.99))) : 1;
  const holdClipped = holds.map(h => h > holdCap ? holdCap : h);
  const holdHist = intCounts(holdClipped, 0, holdCap);
  if (holds.some(h => h > holdCap)) {{
    holdHist.labels[holdHist.labels.length - 1] = holdCap + '+';
  }}
  const dowNames = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'];
  const dowCounts = [0,0,0,0,0,0,0];
  const hourBag = {{}};
  let nHour = 0;
  for (const t of trades) {{
    const dt = parseYmd(t.buy);
    if (dt) dowCounts[dt.getDay()]++;
    const hr = buyHour(t);
    if (hr != null) {{ hourBag[hr] = (hourBag[hr] || 0) + 1; nHour++; }}
  }}
  const hourKeys = Object.keys(hourBag).map(Number).sort((a,b)=>a-b);
  const hourLabels = hourKeys.map(h => String(h).padStart(2,'0') + ':00');
  const hourCounts = hourKeys.map(h => hourBag[h]);
  const hourWrap = document.getElementById('distHourWrap');
  if (hourWrap) hourWrap.style.display = nHour ? '' : 'none';

  const byReason = {{}};
  trades.forEach((t, i) => {{
    const r = t.reason || 'unknown';
    if (!byReason[r]) byReason[r] = [];
    byReason[r].push(gains[i]);
  }});
  const reasonKeys = Object.keys(byReason).sort((a,b)=> byReason[b].length - byReason[a].length);
  const reasonMeans = reasonKeys.map(k => mean(byReason[k]));
  const reasonColors = reasonMeans.map(v => v >= 0 ? 'rgba(38,166,154,0.8)' : 'rgba(239,83,80,0.8)');
  const pnlH = autoHist(gains, 40);
  const sd = stdev(gains);
  const med = median(gains);
  const mn = mean(gains);
  const skew = sd > 1e-12 ? (mn - med) / sd : 0;
  const p05 = quantile(gains, 0.05), p95 = quantile(gains, 0.95);
  const spyDays = day.calendar.length;
  const zeroDays = spyDays ? calN.filter(x => x === 0).length : null;
  const meanFillDay = mean(fillN);
  const meanCalDay = spyDays ? mean(calN) : meanFillDay;
  const syms = symbolStats(trades, gains);

  setHTML('distActivityCards', `
    <div class="card"><div class="label">Days with a fill</div><div class="value">${{fillN.length}}</div></div>
    <div class="card"><div class="label">Mean orders / fill-day</div><div class="value">${{meanFillDay.toFixed(2)}}</div></div>
    <div class="card"><div class="label">Median / max / day</div><div class="value">${{median(fillN).toFixed(0)}} / ${{maxDay}}</div></div>
    <div class="card"><div class="label">Mean orders / session</div><div class="value">${{meanCalDay.toFixed(2)}}</div></div>
    ${{spyDays ? `<div class="card"><div class="label">Sessions with 0 fills</div><div class="value">${{zeroDays}} / ${{spyDays}}</div></div>
    <div class="card"><div class="label">% sessions with a fill</div><div class="value">${{(100*(spyDays-zeroDays)/spyDays).toFixed(1)}}%</div></div>` : ''}}
  `);
  setHTML('distPnlCards', `
    <div class="card"><div class="label">Mean / median %</div><div class="value ${{cls(mn)}}">${{pct(mn,true)}} / ${{pct(med,true)}}</div></div>
    <div class="card"><div class="label">p05 / p95 %</div><div class="value">${{pct(p05,true)}} / ${{pct(p95,true)}}</div></div>
    <div class="card"><div class="label">Stdev %</div><div class="value">${{sd.toFixed(2)}}</div></div>
    <div class="card"><div class="label">Skew (mean-med)/sd</div><div class="value">${{skew.toFixed(2)}}</div></div>
    <div class="card"><div class="label">Min / max %</div><div class="value">${{gains.length ? pct(Math.min.apply(null,gains),true) : '—'}} / ${{gains.length ? pct(Math.max.apply(null,gains),true) : '—'}}</div></div>
    <div class="card"><div class="label">% trades &gt; +5%</div><div class="value">${{gains.length ? (100*gains.filter(x=>x>5).length/gains.length).toFixed(1) : '0'}}%</div></div>
    <div class="card"><div class="label">% trades &lt; -3%</div><div class="value">${{gains.length ? (100*gains.filter(x=>x<-3).length/gains.length).toFixed(1) : '0'}}%</div></div>
  `);

  const qps = [
    ['p01', 0.01], ['p05', 0.05], ['p10', 0.10], ['p25', 0.25],
    ['p50', 0.50], ['p75', 0.75], ['p90', 0.90], ['p95', 0.95],
    ['p99', 0.99]
  ];
  const qExtra = [
    ['Min', gains.length ? Math.min.apply(null, gains) : 0],
    ['Max', gains.length ? Math.max.apply(null, gains) : 0],
    ['IQR', quantile(gains, 0.75) - quantile(gains, 0.25)]
  ];
  setHTML('distQuantGrid',
    '<div class="mh">Metric</div><div class="mh">Value</div><div class="mh">Metric</div><div class="mh">Value</div>' +
    qps.map(pair => {{
      const v = quantile(gains, pair[1]);
      return `<div class="mc">${{pair[0]}}</div><div class="mc ${{cls(v)}}">${{pct(v,true)}}</div>`;
    }}).join('') +
    qExtra.map(pair => `<div class="mc">${{pair[0]}}</div><div class="mc ${{cls(pair[1])}}">${{pct(pair[1],true)}}</div>`).join('')
  );

  setHTML('distSymCards', `
    <div class="card"><div class="label">Symbols</div><div class="value">${{syms.length}}</div></div>
    <div class="card"><div class="label">Median trades / symbol</div><div class="value">${{median(syms.map(s=>s.n)).toFixed(0)}}</div></div>
    <div class="card"><div class="label">Max trades / symbol</div><div class="value">${{syms.length ? syms[0].n : 0}}</div></div>
    <div class="card"><div class="label">Symbols with 1 trade</div><div class="value">${{syms.filter(s=>s.n===1).length}}</div></div>
  `);
  setHTML('distSymBody', syms.slice(0, 20).map(s => `
    <tr>
      <td>${{s.symbol}}</td><td class="num">${{s.n}}</td>
      <td class="num">${{pct(s.wr,false)}}</td>
      <td class="num ${{cls(s.e)}}">${{pct(s.e,true)}}</td>
      <td class="num">${{Number.isFinite(s.pf)?s.pf.toFixed(2):'inf'}}</td>
    </tr>`).join(''));

  const accent = 'rgba(41,98,255,0.85)';
  upsertBarChart('dayHist', 'distDayHistChart', dayHist.labels, dayHist.counts, accent, {{ seriesLabel: 'days' }});
  const tsLabels = spyDays ? day.calendar : day.fillDays;
  const tsData = spyDays ? day.calCounts : day.fillCounts;
  upsertLineChart('dayTs', 'distDayTsChart', tsLabels, tsData, '#2962ff');
  upsertBarChart('pnlHist', 'distPnlHistChart', pnlH.labels, pnlH.counts, pnlH.colors, {{ seriesLabel: 'trades' }});
  upsertBarChart('hold', 'distHoldChart', holdHist.labels, holdHist.counts, 'rgba(41,98,255,0.8)', {{ seriesLabel: 'trades' }});
  upsertBarChart('dow', 'distDowChart', dowNames, dowCounts, 'rgba(41,98,255,0.8)', {{ seriesLabel: 'fills' }});
  if (nHour) upsertBarChart('hour', 'distHourChart', hourLabels, hourCounts, 'rgba(247,147,26,0.85)', {{ seriesLabel: 'fills' }});
  upsertBarChart('conc', 'distConcChart', concHist.labels, concHist.counts, 'rgba(41,98,255,0.8)', {{ seriesLabel: 'entries' }});
  upsertBarChart('exit', 'distExitChart', reasonKeys, reasonMeans, reasonColors, {{
    seriesLabel: 'mean %',
    options: Object.assign({{}}, distChartOpts, {{
      plugins: {{ legend: {{ display: false }}, tooltip: {{ callbacks: {{
        label: (item) => {{
          const k = reasonKeys[item.dataIndex];
          const arr = byReason[k] || [];
          return ' mean ' + (item.raw>=0?'+':'') + Number(item.raw).toFixed(2) + '%  (n=' + arr.length + ')';
        }}
      }} }} }}
    }})
  }});
}}

function simulateSpy(spy, capital, from, to) {{
  if (!spy || !spy.length || !from || !to) return null;
  const bars = spy.filter(p => p.x >= from && p.x <= to);
  if (!bars.length) return null;
  const start = bars[0].c;
  if (!(start > 0)) return null;
  const eq = bars.map(p => ({{x: p.x, y: capital * p.c / start}}));
  let peak = eq[0].y, maxDd = 0;
  for (const p of eq) {{
    if (p.y > peak) peak = p.y;
    const dd = peak > 0 ? (peak - p.y) / peak * 100 : 0;
    if (dd > maxDd) maxDd = dd;
  }}
  const final = eq[eq.length - 1].y;
  const net = final - capital;
  const ret = capital > 0 ? net / capital * 100 : 0;
  const days = (Date.parse(eq[eq.length-1].x) - Date.parse(eq[0].x)) / 86400000;
  const years = Math.max(days / 365.25, 1e-9);
  const cagr = (Math.pow(final / capital, 1 / years) - 1) * 100;
  return {{ eq, ret, net, final, maxDd, cagr, from: eq[0].x, to: eq[eq.length-1].x }};
}}

function buildOverlay(stratEq, spyEq, capital) {{
  if (!spyEq || !spyEq.length) return [];
  const map = {{}};
  for (const p of stratEq) map[p.x] = p.y;
  let last = capital;
  const out = [];
  for (const p of spyEq) {{
    if (map[p.x] !== undefined) last = map[p.x];
    out.push({{
      x: p.x,
      strategy: 100 * last / capital,
      spy: 100 * p.y / capital
    }});
  }}
  return out;
}}

let eqChart, ddChart, cmpChart;
let tradeRowsAll = [];
let tradeSort = {{ key: 'exit_date', dir: 'asc' }};

function ensureCharts() {{
  const common = {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ labels: {{ color: '#d1d4dc' }} }} }},
    scales: {{
      x: {{ ticks: {{ color:'#787b86', maxTicksLimit:10 }}, grid: {{ color:'#2a2e39' }} }},
      y: {{ ticks: {{ color:'#787b86' }}, grid: {{ color:'#2a2e39' }} }}
    }}
  }};
  if (!eqChart) {{
    eqChart = new Chart(document.getElementById('eqChart'), {{
      type: 'line', data: {{ labels: [], datasets: [
        {{ label:'Strategy', data: [], borderColor:'#2962ff', pointRadius:0, borderWidth:1.5, tension:0.05, spanGaps:true }},
        {{ label:'S&P 500 (SPY)', data: [], borderColor:'#f7931a', pointRadius:0, borderWidth:1.5, tension:0.05, spanGaps:true }}
      ]}}, options: common
    }});
  }}
  if (!ddChart) {{
    ddChart = new Chart(document.getElementById('ddChart'), {{
      type: 'line', data: {{ labels: [], datasets: [
        {{ label:'Drawdown %', data: [], borderColor:'#ef5350', backgroundColor:'rgba(239,83,80,0.15)', fill:true, pointRadius:0, borderWidth:1.5, tension:0.05 }}
      ]}}, options: {{ ...common, plugins: {{ legend: {{ display:false }} }} }}
    }});
  }}
  if (!cmpChart) {{
    cmpChart = new Chart(document.getElementById('cmpChart'), {{
      type: 'line', data: {{ labels: [], datasets: [
        {{ label:'Strategy (norm)', data: [], borderColor:'#2962ff', pointRadius:0, borderWidth:1.5, tension:0.05 }},
        {{ label:'S&P 500 (norm)', data: [], borderColor:'#f7931a', pointRadius:0, borderWidth:1.5, tension:0.05 }}
      ]}}, options: common
    }});
  }}
}}

function alignSeries(dates, points) {{
  const m = Object.fromEntries(points.map(p => [p.x, p.y]));
  let last = null;
  return dates.map(d => {{ if (m[d] !== undefined) last = m[d]; return last; }});
}}

function render(sim, spy) {{
  const m = sim.metrics;
  const p = sim.params;
  const excess = spy ? m.ret - spy.ret : null;
  document.getElementById('subtitle').textContent =
    m.from + ' to ' + m.to + ' · ' + m.n + ' trades · sizing ' + p.sizeMode + ' ' +
    (p.sizeMode === 'fixed' ? money(p.sizeVal,false) : (p.sizeVal.toFixed(2) + '%')) +
    ' · friction ' + p.friction.toFixed(2) + '% · max/day ' + (p.maxPerDay || 'all') +
    (RAW.trades.length > m.n ? (' (' + RAW.trades.length + ' embedded)') : '') +
    ' · maxOpen ' + (p.maxOpen || 'none') +
    (p.winCap > 0 ? (' · winCap ' + p.winCap + '%') : '') +
    (p.excludeSym ? (' · excl ' + p.excludeSym) : '');

  setHTML('overviewCards', `
    <div class="card"><div class="label">Net profit</div><div class="value ${{cls(m.net)}}">${{money(m.net,true)}}</div></div>
    <div class="card"><div class="label">Total return</div><div class="value ${{cls(m.ret)}}">${{pct(m.ret,true)}}</div></div>
    <div class="card"><div class="label">Max equity DD</div><div class="value neg">${{pct(m.maxDd,false)}}</div></div>
    <div class="card"><div class="label">Total closed trades</div><div class="value">${{m.n}}</div></div>
    <div class="card"><div class="label">Percent profitable</div><div class="value">${{pct(m.wr,false)}}</div></div>
    <div class="card"><div class="label">Profit factor</div><div class="value">${{Number.isFinite(m.pf) ? m.pf.toFixed(3) : 'inf'}}</div></div>
    <div class="card"><div class="label">Avg trade</div><div class="value">${{money(m.avgTrade,true)}}</div></div>
    <div class="card"><div class="label">Avg bars in trade</div><div class="value">${{m.avgBars.toFixed(1)}}</div></div>
    ${{spy ? `<div class="card"><div class="label">S&P 500 (SPY) return</div><div class="value">${{pct(spy.ret,true)}}</div></div>
    <div class="card"><div class="label">Excess vs SPY</div><div class="value ${{cls(excess)}}">${{pct(excess,true)}}</div></div>` : ''}}
  `);

  setHTML('modelNote',
    'Model: ' + (p.sizeMode === 'fixed' ? ('fixed ' + money(p.sizeVal,false) + '/trade') :
      (p.sizeMode === 'pct_initial' ? (p.sizeVal + '% of initial') : (p.sizeVal + '% of equity'))) +
    '; equity marks on exit dates. Friction ' + pct(p.friction,false) +
    ' round-trip. Orange = SPY buy & hold with same portfolio size.' +
    (p.maxOpen > 0
      ? (' Max concurrent opens=' + p.maxOpen + ' (greedy by buy date / RS).')
      : ' Overlapping multi-symbol fills are not capital-constrained (set Max concurrent).') +
    (p.winCap > 0 ? (' Win cap=' + p.winCap + '%.') : '') +
    (RAW.trades.length > m.n
      ? (' Showing ' + m.n + ' of ' + RAW.trades.length + ' embedded quality-filtered trades after Max entries/day.')
      : (p.maxPerDay > 0
        ? ' Embedded trades are already unique by entry date, so Max entries/day cannot add fills (regenerate from *_trades_raw_*.csv).'
        : ''))
  );

  renderRobustness(sim);
  renderDistributions(sim);

  // Compare
  if (spy) {{
    setHTML('compareCards', `
      <div class="card"><div class="label">Strategy return</div><div class="value ${{cls(m.ret)}}">${{pct(m.ret,true)}}</div></div>
      <div class="card"><div class="label">SPY buy & hold</div><div class="value">${{pct(spy.ret,true)}}</div></div>
      <div class="card"><div class="label">Excess return</div><div class="value ${{cls(excess)}}">${{pct(excess,true)}}</div></div>
      <div class="card"><div class="label">Beats SPY?</div><div class="value ${{m.ret > spy.ret ? 'pos' : 'neg'}}">${{m.ret > spy.ret ? 'Yes' : 'No'}}</div></div>
      <div class="card"><div class="label">Strategy max DD</div><div class="value neg">${{pct(m.maxDd,false)}}</div></div>
      <div class="card"><div class="label">SPY max DD</div><div class="value neg">${{pct(spy.maxDd,false)}}</div></div>
      <div class="card"><div class="label">Lower DD than SPY?</div><div class="value ${{m.maxDd < spy.maxDd ? 'pos' : 'neg'}}">${{m.maxDd < spy.maxDd ? 'Yes' : 'No'}}</div></div>
      <div class="card"><div class="label">SPY CAGR (approx)</div><div class="value">${{pct(spy.cagr,true)}}</div></div>
    `);
    setHTML('compareGrid', `
      <div class="mh">Metric</div><div class="mh">Strategy</div><div class="mh">S&P 500 (SPY)</div><div class="mh">Diff</div>
      <div class="mc">Total return</div><div class="mc ${{cls(m.ret)}}">${{pct(m.ret,true)}}</div><div class="mc">${{pct(spy.ret,true)}}</div><div class="mc ${{cls(excess)}}">${{pct(excess,true)}}</div>
      <div class="mc">Net profit</div><div class="mc ${{cls(m.net)}}">${{money(m.net,true)}}</div><div class="mc">${{money(spy.net,true)}}</div><div class="mc ${{cls(m.net-spy.net)}}">${{money(m.net-spy.net,true)}}</div>
      <div class="mc">Final equity</div><div class="mc">${{money(m.final,false)}}</div><div class="mc">${{money(spy.final,false)}}</div><div class="mc ${{cls(m.final-spy.final)}}">${{money(m.final-spy.final,true)}}</div>
      <div class="mc">Max equity drawdown</div><div class="mc neg">${{pct(m.maxDd,false)}}</div><div class="mc neg">${{pct(spy.maxDd,false)}}</div><div class="mc">${{pct(m.maxDd-spy.maxDd,true)}}</div>
    `);
  }} else {{
    setHTML('compareCards', '<p class="muted">SPY data unavailable.</p>');
    setHTML('compareGrid', '');
  }}

  const pfS = Number.isFinite(m.pf) ? m.pf.toFixed(3) : 'inf';
  setHTML('perfGrid', `
    <div class="mh">Metric</div><div class="mh">All</div><div class="mh">Long</div><div class="mh">Short</div>
    <div class="mc">Net profit</div><div class="mc ${{cls(m.net)}}">${{money(m.net,true)}}</div><div class="mc ${{cls(m.net)}}">${{money(m.net,true)}}</div><div class="mc">—</div>
    <div class="mc">Gross profit</div><div class="mc">${{money(m.gp,false)}}</div><div class="mc">${{money(m.gp,false)}}</div><div class="mc">—</div>
    <div class="mc">Gross loss</div><div class="mc">${{money(m.gl,false)}}</div><div class="mc">${{money(m.gl,false)}}</div><div class="mc">—</div>
    <div class="mc">Max equity drawdown</div><div class="mc neg">${{pct(m.maxDd,false)}}</div><div class="mc neg">${{pct(m.maxDd,false)}}</div><div class="mc">—</div>
    <div class="mc">Total closed trades</div><div class="mc">${{m.n}}</div><div class="mc">${{m.n}}</div><div class="mc">—</div>
    <div class="mc">Winning / losing</div><div class="mc">${{m.wins}} / ${{m.losses}}</div><div class="mc">${{m.wins}} / ${{m.losses}}</div><div class="mc">—</div>
    <div class="mc">Percent profitable</div><div class="mc">${{pct(m.wr,false)}}</div><div class="mc">${{pct(m.wr,false)}}</div><div class="mc">—</div>
    <div class="mc">Profit factor</div><div class="mc">${{pfS}}</div><div class="mc">${{pfS}}</div><div class="mc">—</div>
    <div class="mc">Avg winning trade</div><div class="mc pos">${{money(m.avgWin,true)}}</div><div class="mc pos">${{money(m.avgWin,true)}}</div><div class="mc">—</div>
    <div class="mc">Avg losing trade</div><div class="mc neg">${{money(m.avgLoss,true)}}</div><div class="mc neg">${{money(m.avgLoss,true)}}</div><div class="mc">—</div>
    <div class="mc">Largest winning trade</div><div class="mc pos">${{money(m.largestWin,true)}}</div><div class="mc pos">${{money(m.largestWin,true)}}</div><div class="mc">—</div>
    <div class="mc">Largest losing trade</div><div class="mc neg">${{money(m.largestLoss,true)}}</div><div class="mc neg">${{money(m.largestLoss,true)}}</div><div class="mc">—</div>
    <div class="mc">Avg bars in trade</div><div class="mc">${{m.avgBars.toFixed(1)}}</div><div class="mc">${{m.avgBars.toFixed(1)}}</div><div class="mc">—</div>
    <div class="mc">Symbols traded</div><div class="mc">${{m.symbols}}</div><div class="mc">${{m.symbols}}</div><div class="mc">—</div>
    <div class="mc">Initial capital</div><div class="mc">${{money(m.capital,false)}}</div><div class="mc">${{money(m.capital,false)}}</div><div class="mc">—</div>
    <div class="mc">Final equity</div><div class="mc">${{money(m.final,false)}}</div><div class="mc">${{money(m.final,false)}}</div><div class="mc">—</div>
  `);
  setHTML('exitMix', 'Exit mix: ' + Object.entries(m.exits).map(([k,v]) => k + ': ' + v).join(' · '));

  tradeRowsAll = sim.rows.slice();
  // Populate exit-reason filter options from current sim
  const exitSel = document.getElementById('tfExit');
  const prevExit = exitSel.value || 'all';
  const reasons = Object.keys(m.exits).sort();
  exitSel.innerHTML = '<option value="all">All</option>' +
    reasons.map(r => '<option value="' + r.replace(/"/g, '&quot;') + '">' + r + '</option>').join('');
  if ([...exitSel.options].some(o => o.value === prevExit)) exitSel.value = prevExit;
  else exitSel.value = 'all';
  paintTradesTable();

  const months = Object.keys(sim.monthly).sort();
  setHTML('monthBody', months.map(mo => {{
    const v = sim.monthly[mo];
    return `<tr><td>${{mo}}</td><td class="num ${{cls(v)}}">${{money(v,true)}}</td></tr>`;
  }}).join(''));

  ensureCharts();
  const spyEq = spy ? spy.eq : [];
  const dates = Array.from(new Set([...sim.eqCurve.map(p=>p.x), ...spyEq.map(p=>p.x)])).sort();
  eqChart.data.labels = dates;
  eqChart.data.datasets[0].data = alignSeries(dates, sim.eqCurve);
  eqChart.data.datasets[1].data = alignSeries(dates, spyEq);
  eqChart.update();
  ddChart.data.labels = sim.ddCurve.map(p => p.x);
  ddChart.data.datasets[0].data = sim.ddCurve.map(p => p.y);
  ddChart.update();
  const overlay = spy ? buildOverlay(sim.eqCurve, spy.eq, p.capital) : [];
  cmpChart.data.labels = overlay.map(p => p.x);
  cmpChart.data.datasets[0].data = overlay.map(p => p.strategy);
  cmpChart.data.datasets[1].data = overlay.map(p => p.spy);
  cmpChart.update();
}}

function readTradeFilters() {{
  const minRaw = document.getElementById('tfMinPnl').value;
  const maxRaw = document.getElementById('tfMaxPnl').value;
  return {{
    symbol: (document.getElementById('tfSymbol').value || '').trim().toUpperCase(),
    outcome: document.getElementById('tfOutcome').value,
    exit: document.getElementById('tfExit').value,
    minPnl: minRaw === '' ? null : Number(minRaw),
    maxPnl: maxRaw === '' ? null : Number(maxRaw),
    from: document.getElementById('tfFrom').value || '',
    to: document.getElementById('tfTo').value || '',
  }};
}}

function filterTradeRows(rows) {{
  const f = readTradeFilters();
  return rows.filter(t => {{
    if (f.symbol) {{
      const toks = f.symbol.split(/[,\\s]+/).filter(Boolean);
      if (toks.length && !toks.some(s => t.symbol.includes(s))) return false;
    }}
    if (f.outcome === 'win' && !(t.pnl > 0)) return false;
    if (f.outcome === 'loss' && !(t.pnl <= 0)) return false;
    if (f.exit !== 'all' && t.exit_reason !== f.exit) return false;
    if (f.minPnl != null && !Number.isNaN(f.minPnl) && t.pnl_pct < f.minPnl) return false;
    if (f.maxPnl != null && !Number.isNaN(f.maxPnl) && t.pnl_pct > f.maxPnl) return false;
    const entryDay = String(t.entry_date || '').slice(0, 10);
    if (f.from && entryDay < f.from) return false;
    if (f.to && entryDay > f.to) return false;
    return true;
  }});
}}

function sortTradeRows(rows) {{
  const {{ key, dir }} = tradeSort;
  const mul = dir === 'asc' ? 1 : -1;
  const out = rows.slice();
  out.sort((a, b) => {{
    let va = a[key], vb = b[key];
    if (va == null && vb == null) return 0;
    if (va == null) return 1;
    if (vb == null) return -1;
    if (typeof va === 'string') return mul * va.localeCompare(vb);
    return mul * (va - vb);
  }});
  return out;
}}

function updateSortHeaders() {{
  const labels = {{
    n:'#', symbol:'Symbol', signal:'Signal', entry_date:'Entry', exit_date:'Exit',
    entry_price:'Entry px', exit_price:'Exit px', pnl:'P&amp;L $', pnl_pct:'P&amp;L %',
    cum_pnl:'Cum. P&amp;L', hold_days:'Bars', exit_reason:'Exit'
  }};
  document.querySelectorAll('#tradesTable th.sortable').forEach(th => {{
    const key = th.dataset.sort;
    const label = labels[key] || key;
    th.innerHTML = label + (tradeSort.key === key
      ? ('<span class="sort-ind">' + (tradeSort.dir === 'asc' ? '▲' : '▼') + '</span>')
      : '');
  }});
}}

function paintTradesTable() {{
  const filtered = filterTradeRows(tradeRowsAll);
  const sorted = sortTradeRows(filtered);
  const maxRows = 2000;
  const slice = sorted.slice(0, maxRows);
  let note = filtered.length + ' of ' + tradeRowsAll.length + ' trades';
  if (filtered.length !== tradeRowsAll.length) note += ' (filtered)';
  note += ' · sort: ' + tradeSort.key + ' ' + tradeSort.dir;
  if (sorted.length > maxRows) note += ' · showing first ' + maxRows;
  if ((RAW.trades || []).some(t => t.buy_at || t.sell_at)) {{
    note += ' · entry/exit are RTH bar times (America/New_York, YYYY-MM-DD HH:MM)';
  }}
  if ((RAW.trades || []).some(t => t.ch)) {{
    note += ' · Channel i copies JSON for CTF Channel JSON (one paste)';
  }}
  setHTML('tradesNote', note);
  updateSortHeaders();
  setHTML('tradesBody', slice.map(t => `
    <tr>
      <td>${{t.n}}</td><td>${{t.symbol}}</td>
      <td class="ch-cell">${{t.ch ? '<button type="button" class="ch-info" data-copy-ch="' + encodeURIComponent(JSON.stringify(t.ch)) + '" title="Copy channel JSON for CTF">i</button>' : ''}}</td>
      <td>${{t.signal}}</td>
      <td class="when">${{t.entry_date}}</td><td class="when">${{t.exit_date}}</td>
      <td class="num">${{t.entry_price.toFixed(4)}}</td><td class="num">${{t.exit_price.toFixed(4)}}</td>
      <td class="num ${{cls(t.pnl)}}">${{money(t.pnl,true)}}</td>
      <td class="num ${{cls(t.pnl_pct)}}">${{pct(t.pnl_pct,true)}}</td>
      <td class="num">${{money(t.cum_pnl,true)}}</td>
      <td class="num">${{t.hold_days == null ? '' : t.hold_days}}</td>
      <td>${{t.exit_reason}}</td>
    </tr>`).join(''));
}}

function clearTradeFilters() {{
  document.getElementById('tfSymbol').value = '';
  document.getElementById('tfOutcome').value = 'all';
  document.getElementById('tfExit').value = 'all';
  document.getElementById('tfMinPnl').value = '';
  document.getElementById('tfMaxPnl').value = '';
  document.getElementById('tfFrom').value = '';
  document.getElementById('tfTo').value = '';
  paintTradesTable();
}}

function apply() {{
  const p = readParams();
  const filtered = prepareTrades(RAW.trades, p);
  const sim = simulate(filtered, p);
  const spy = simulateSpy(RAW.spy, p.capital, sim.metrics.from, sim.metrics.to);
  render(sim, spy);
}}

function esc(s) {{
  return String(s == null ? '' : s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}}

function renderFilterBooks() {{
  const c = RAW.comparison || {{}};
  const rows = c.rows || [];
  const box = document.getElementById('filterBooks');
  const runBox = document.getElementById('filterBooksRun');
  if (!rows.length) {{
    if (box) box.style.display = 'none';
    if (runBox) runBox.style.display = 'none';
    return;
  }}
  if (box) box.style.display = 'block';
  if (runBox) runBox.style.display = 'block';
  const title = c.title || 'Filter books';
  const tEl = document.getElementById('filterBooksTitle');
  const tRun = document.getElementById('filterBooksRunTitle');
  if (tEl) tEl.textContent = title;
  if (tRun) tRun.textContent = title;
  const body = rows.map(r => {{
    const hl = r.highlight ? ' class="book-hl"' : '';
    const n = (r.n == null) ? '—' : Number(r.n).toLocaleString();
    const wr = (r.wr == null) ? '—' : Number(r.wr).toFixed(1) + '%';
    const eNum = Number(r.e);
    const e = Number.isFinite(eNum) ? ((eNum >= 0 ? '+' : '') + eNum.toFixed(2) + '%') : '—';
    const pf = (r.pf == null) ? '—' : Number(r.pf).toFixed(2);
    return `<tr${{hl}}><td>${{esc(r.book || r.key || '')}}</td><td class="num">${{n}}</td><td class="num">${{wr}}</td><td class="num ${{cls(eNum)}}">${{e}}</td><td class="num">${{pf}}</td></tr>`;
  }}).join('');
  setHTML('filterBooksBody', body);
  setHTML('filterBooksRunBody', body);
  const noteEl = document.getElementById('filterBooksNote');
  if (noteEl) noteEl.textContent = c.note || '';
}}

function renderRunInfo() {{
  const m = RAW.runMeta || {{}};
  const g = m.git || {{}};
  document.getElementById('gitGrid').innerHTML = [
    ['Branch', g.branch || '—'],
    ['Commit', g.commit || '—'],
    ['Working tree', g.dirty === 'yes' ? 'dirty (uncommitted changes)' : 'clean'],
  ].map(([h, v]) => `<div class="mh">${{esc(h)}}</div><div class="mc">${{esc(v)}}</div>`).join('');

  const det = m.detector || {{}};
  const detParams = (det.params || []).map(r =>
    `<tr><td>${{esc(r.key)}}</td><td>${{esc(r.value)}}</td></tr>`
  ).join('');
  document.getElementById('detectorBlock').innerHTML = `
    <div class="meta-block"><div class="meta-label">Name</div><div class="meta-value">${{esc(det.name || '—')}}</div></div>
    <div class="meta-block"><div class="meta-label">Script</div><div class="meta-value">${{esc(det.script || '—')}}</div></div>
    <div class="meta-block"><div class="meta-label">Pine</div><div class="meta-value">${{esc(det.pine || '—')}}</div></div>
    <div class="meta-block"><div class="meta-label">Version</div><div class="meta-value">${{esc(det.version || '—')}}</div></div>
    <div class="meta-block"><div class="meta-label">Notes</div><div class="meta-value">${{esc(det.notes || '—')}}</div></div>
    ${{detParams ? '<div class="table-scroll" style="max-height:200px;margin-top:8px"><table class="meta-kv"><tbody>' + detParams + '</tbody></table></div>' : ''}}
  `;

  const bt = m.backtest_params || [];
  document.getElementById('backtestParamsBody').innerHTML = bt.length
    ? bt.map(r => `<tr><td>${{esc(r.key)}}</td><td>${{esc(r.value)}}</td></tr>`).join('')
    : '<tr><td colspan="2">No backtest params in summary sidecar</td></tr>';

  const dp = m.data_params || [];
  document.getElementById('dataParamsBody').innerHTML = dp.length
    ? dp.map(r => `<tr><td>${{esc(r.key)}}</td><td>${{esc(r.value)}}</td></tr>`).join('')
    : '<tr><td colspan="2">No data params in summary sidecar</td></tr>';

  const bits = [];
  if (m.trades_file) bits.push('Trades: ' + m.trades_file);
  if (m.summary_file) bits.push('Summary: ' + m.summary_file);
  if ((m.notes || []).length) bits.push(m.notes.join(' · '));
  document.getElementById('runMetaNote').textContent = bits.join(' · ') || '';
}}

function resetDefaults() {{
  const d = RAW.defaults;
  document.getElementById('capital').value = d.capital;
  document.getElementById('sizeMode').value = d.sizeMode;
  document.getElementById('sizeVal').value = d.sizeVal;
  document.getElementById('friction').value = d.friction;
  document.getElementById('maxPerDay').value = d.maxPerDay;
  document.getElementById('maxOpen').value = d.maxOpen != null ? d.maxOpen : 0;
  document.getElementById('winCap').value = d.winCap != null ? d.winCap : 0;
  document.getElementById('excludeSym').value = d.excludeSym || '';
  syncSizeLabel();
  apply();
}}

function pushMaxPnlToEquity() {{
  const maxRaw = document.getElementById('tfMaxPnl').value;
  if (maxRaw === '' || Number.isNaN(Number(maxRaw))) {{
    alert('Set Max P&L % in the trade filters first (e.g. 30 or 50).');
    return;
  }}
  document.getElementById('winCap').value = Number(maxRaw);
  apply();
  document.querySelectorAll('.tab').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.querySelector('.tab[data-tab="overview"]').classList.add('active');
  document.getElementById('overview').classList.add('active');
}}

document.querySelectorAll('.tab').forEach(btn => {{
  btn.addEventListener('click', () => {{
    document.querySelectorAll('.tab').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById(btn.dataset.tab).classList.add('active');
    if (btn.dataset.tab === 'distributions') resizeDistCharts();
  }});
}});
document.getElementById('sizeMode').addEventListener('change', syncSizeLabel);
document.getElementById('btnApply').addEventListener('click', apply);
document.getElementById('btnReset').addEventListener('click', resetDefaults);
document.querySelector('.controls').addEventListener('keydown', e => {{
  if (e.key === 'Enter') {{ e.preventDefault(); apply(); }}
}});

// Trade table sort + filter
document.querySelectorAll('#tradesTable th.sortable').forEach(th => {{
  th.addEventListener('click', () => {{
    const key = th.dataset.sort;
    if (tradeSort.key === key) tradeSort.dir = tradeSort.dir === 'asc' ? 'desc' : 'asc';
    else {{ tradeSort.key = key; tradeSort.dir = (key === 'pnl' || key === 'pnl_pct' || key === 'cum_pnl') ? 'desc' : 'asc'; }}
    paintTradesTable();
  }});
}});
['tfSymbol','tfOutcome','tfExit','tfMinPnl','tfMaxPnl','tfFrom','tfTo'].forEach(id => {{
  const el = document.getElementById(id);
  el.addEventListener('input', paintTradesTable);
  el.addEventListener('change', paintTradesTable);
}});
document.getElementById('tfClear').addEventListener('click', clearTradeFilters);
document.getElementById('tfPushEquity').addEventListener('click', pushMaxPnlToEquity);

function showCopyToast(msg) {{
  const el = document.getElementById('copyToast');
  el.textContent = msg;
  el.style.display = 'block';
  clearTimeout(showCopyToast._t);
  showCopyToast._t = setTimeout(() => {{ el.style.display = 'none'; }}, 2200);
}}

function openChModal(text) {{
  const modal = document.getElementById('chCopyModal');
  const ta = document.getElementById('chCopyText');
  ta.value = text;
  modal.classList.add('open');
  ta.focus();
  ta.select();
}}

function closeChModal() {{
  document.getElementById('chCopyModal').classList.remove('open');
}}

function copyText(text) {{
  if (navigator.clipboard && window.isSecureContext) {{
    return navigator.clipboard.writeText(text);
  }}
  return new Promise((resolve, reject) => {{
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.setAttribute('readonly', '');
    ta.style.position = 'fixed';
    ta.style.left = '-9999px';
    document.body.appendChild(ta);
    ta.select();
    try {{
      const ok = document.execCommand('copy');
      document.body.removeChild(ta);
      if (ok) resolve();
      else reject(new Error('execCommand copy failed'));
    }} catch (err) {{
      document.body.removeChild(ta);
      reject(err);
    }}
  }});
}}

function copyChannelJson(text, btn) {{
  openChModal(text);
  copyText(text).then(() => {{
    if (btn) {{
      btn.classList.add('copied');
      setTimeout(() => btn.classList.remove('copied'), 1200);
    }}
    showCopyToast('Copied channel JSON — paste into CTF Channel JSON');
  }}).catch(() => {{
    showCopyToast('Select the JSON and copy (Ctrl+C)');
  }});
}}

document.getElementById('tradesTable').addEventListener('click', (e) => {{
  const btn = e.target.closest('[data-copy-ch]');
  if (!btn) return;
  e.preventDefault();
  const text = decodeURIComponent(btn.getAttribute('data-copy-ch') || '');
  if (!text) return;
  copyChannelJson(text, btn);
}});
document.getElementById('chCopyAgain').addEventListener('click', () => {{
  copyChannelJson(document.getElementById('chCopyText').value, null);
}});
document.getElementById('chCopyClose').addEventListener('click', closeChModal);
document.getElementById('chCopyModal').addEventListener('click', (e) => {{
  if (e.target.id === 'chCopyModal') closeChModal();
}});
document.addEventListener('keydown', (e) => {{
  if (e.key === 'Escape') closeChModal();
}});

document.getElementById('genAt').textContent = RAW.generated;
renderFilterBooks();
renderRunInfo();
resetDefaults();
</script>
</body>
</html>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description="TradingView-style strategy report for channel-touch trades")
    ap.add_argument("--trades", type=Path, default=None, help="Trades CSV path")
    ap.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "reports" / "ascending_channels",
        help="Base report folder; a YYYY-MM-DD subfolder is created automatically",
    )
    ap.add_argument("--initial-capital", type=float, default=100_000.0)
    ap.add_argument("--notional", type=float, default=10_000.0, help="Default fixed $ per trade")
    ap.add_argument("--size-mode", choices=("fixed", "pct_initial", "pct_equity"), default="fixed")
    ap.add_argument("--size-pct", type=float, default=10.0, help="Default %% when size-mode is pct_*")
    ap.add_argument("--friction-pct", type=float, default=0.0)
    ap.add_argument("--rs-top1", action="store_true", help="Default max entries/day = 1")
    ap.add_argument("--max-per-day", type=int, default=None, help="Default max entries/day (overrides --rs-top1)")
    ap.add_argument("--provider", default="ALPACA")
    ap.add_argument("--tag", default="")
    ap.add_argument(
        "--title",
        default="Ascending Channel Touch Long — Strategy Report",
        help="HTML page title / header",
    )
    ap.add_argument(
        "--comparison-json",
        type=Path,
        default=None,
        help="Optional JSON with filter-book comparison rows (title/note/rows)",
    )
    args = ap.parse_args()

    trades_path = args.trades or _latest_trades_csv(args.outdir)
    logger.info("Loading trades: %s", trades_path)
    df = load_trades_for_report(trades_path)
    if df.empty:
        logger.error("Empty trades CSV")
        return 1

    # Quality-filtered full list (no RS/day cap); UI filterMaxPerDay applies the cap
    raw_trades = trades_to_raw(df)
    max_per_day = args.max_per_day if args.max_per_day is not None else (1 if args.rs_top1 else 0)

    buy_min = pd.to_datetime(df["buy_date"]).min()
    sell_max = pd.to_datetime(df["sell_date"]).max()
    spy_closes: List[dict] = []
    try:
        spy_close = load_spy_close(
            buy_min - pd.Timedelta(days=5),
            sell_max + pd.Timedelta(days=5),
            provider=args.provider,
        )
        spy_closes = spy_to_raw(spy_close, buy_min, sell_max)
        logger.info(
            "SPY bars for compare: %d (%s to %s)",
            len(spy_closes),
            spy_closes[0]["x"] if spy_closes else "n/a",
            spy_closes[-1]["x"] if spy_closes else "n/a",
        )
        if spy_closes and spy_closes[0]["x"] > buy_min.strftime("%Y-%m-%d"):
            logger.warning(
                "SPY overlay starts %s after first trade %s; IB prefix may be missing",
                spy_closes[0]["x"],
                buy_min.strftime("%Y-%m-%d"),
            )
    except Exception as exc:
        logger.warning("S&P 500 data skipped: %s", exc)

    if args.size_mode == "fixed":
        size_val = float(args.notional)
    else:
        size_val = float(args.size_pct)

    defaults = {
        "capital": float(args.initial_capital),
        "sizeMode": args.size_mode,
        "sizeVal": size_val,
        "friction": float(args.friction_pct),
        "maxPerDay": int(max_per_day),
        "maxOpen": 0,
        "winCap": 0,
        "excludeSym": "",
    }

    label_bits = ["interactive"]
    if max_per_day == 1:
        label_bits.append("rs_top1_default")
    elif max_per_day > 1:
        label_bits.append(f"max{max_per_day}")
    if args.friction_pct:
        label_bits.append(f"fric{args.friction_pct:.2f}")
    if args.tag:
        label_bits.append(args.tag)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = "_".join(label_bits)
    args.outdir = dated_outdir(args.outdir)
    out_html = args.outdir / f"channel_touch_tv_report_{tag}_{stamp}.html"
    out_json = args.outdir / f"channel_touch_tv_report_{tag}_{stamp}.json"

    run_meta = build_run_meta(trades_path)
    comparison = load_comparison_json(args.comparison_json)
    logger.info(
        "Run meta: branch=%s commit=%s summary=%s comparison_rows=%s",
        run_meta["git"]["branch"],
        run_meta["git"]["commit"],
        run_meta.get("summary_file"),
        0 if not comparison else len(comparison.get("rows") or []),
    )

    html = render_html(
        raw_trades=raw_trades,
        spy_closes=spy_closes,
        defaults=defaults,
        run_meta=run_meta,
        title=args.title,
        source=str(trades_path.name),
        comparison=comparison,
    )
    out_html.write_text(html, encoding="utf-8")
    out_json.write_text(
        json.dumps(
            {
                "defaults": defaults,
                "n_trades_embedded": len(raw_trades),
                "n_spy_bars": len(spy_closes),
                "source": str(trades_path),
                "source_raw": str(_raw_sibling_csv(trades_path) or ""),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    logger.info("Wrote %s (%d trades embedded)", out_html, len(raw_trades))
    print(f"\nInteractive TradingView-style report")
    print(f"  Trades embedded: {len(raw_trades)} (quality-filtered, no RS/day cap)")
    print(f"  Default capital: ${args.initial_capital:,.0f}")
    print(f"  Default sizing:  {args.size_mode} = {size_val}")
    print(f"  Default max/day: {max_per_day or 'all'}")
    print(f"  Friction:        {args.friction_pct:.2f}%")
    print(f"\nOpen: {out_html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
