"""Count H2 support-cancels that later break the same rails (TARS Jan-2024).

Symbols from the current_best last-15m 1d book. Windowed H2 504/252, span365,
error 1.2%, min_wait 6, max_wait 252. Occupancy is not re-walked.

Shallow = cancel-bar close undershoot / width <= 0.25.
TARS-like = shallow and next close back at/above support.
Later breakout = first close above resist inside remaining wait from H2.
Last-15m sleeve = that session's last RTH 15m mid if open > rail, then
15m N+1 ATR k=2 + 10% trail.

Usage (Windows CMD):
  venv\\Scripts\\activate
  set PYTHONPATH=.
  python scripts\\research\\analyze_h2_cancel_reclaim.py --symbols TARS
  python scripts\\research\\analyze_h2_cancel_reclaim.py --workers 8
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

from backtest_channel_touch_trades import YEAR_BUCKETS, summarize_by_year  # noqa: E402
from find_ascending_channels import find_h2_l3_setups_windowed  # noqa: E402
from utils.data.ohlcv_loader import load_ohlcv_many  # noqa: E402
from utils.research.channel_outside_area import stamp_date  # noqa: E402
from utils.research.channel_touch_scale import (  # noqa: E402
    DAILY_WINDOW_BARS,
    DAILY_WINDOW_STEP_BARS,
)
from utils.research.h2_cancel_reclaim import (  # noqa: E402
    ERROR_PCT,
    MAX_WAIT,
    MIN_WAIT,
    OUTCOME_CANCELLED,
    SHALLOW_WIDTH,
    SPAN_DAYS,
    count_bucket,
    rail_at,
    setup_row_ok,
    walk_h2_cancel_reclaim,
    wilder_atr,
)
from utils.research.last_15m_volume_geometry import book_stats, geom_mean_year_pf  # noqa: E402
from utils.research.realistic_exits import (  # noqa: E402
    flatten_rth_sessions,
    simulate_exit_15m_next_mid,
)
from utils.research.realistic_purchaser import (  # noqa: E402
    index_rth_15m_by_session,
    purchase_last_rth_open_above_mid,
)
from utils.research.report_paths import dated_outdir  # noqa: E402
from utils.scanning.channel_touch_bought import ATR_STOP_MULT  # noqa: E402

LOG = logging.getLogger("analyze_h2_cancel_reclaim")

DEFAULT_TRADES = ROOT / "reports" / "ascending_channels" / "channel_touch_1d_last_15m.csv"
FRICTION_PCT = 0.25


def _idx_date(dates: List, i: int) -> Optional[str]:
    if i is None or i < 0 or i >= len(dates):
        return None
    d = dates[i]
    return d.isoformat() if d is not None else None


def walk_symbol(
    sym: str,
    df: pd.DataFrame,
) -> List[dict]:
    if df is None or df.empty:
        return []
    setups = find_h2_l3_setups_windowed(
        df,
        window_bars=DAILY_WINDOW_BARS,
        step_bars=DAILY_WINDOW_STEP_BARS,
        pivot_len=15,
        error_pct=ERROR_PCT,
    )
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    dates = [stamp_date(t) for t in df.index]
    atr = wilder_atr(high, low, close, length=14)
    rows: List[dict] = []
    for ch in setups:
        if not setup_row_ok(ch, max_span_days=SPAN_DAYS):
            continue
        h2_i = int(ch["h2_idx"])
        x0 = int(ch["support_x0"])
        y0 = float(ch["support_y0"])
        slope = float(ch["support_slope"])
        width = float(ch["channel_width"])
        walked = walk_h2_cancel_reclaim(
            close,
            low,
            h2_i=h2_i,
            support_x0=x0,
            support_y0=y0,
            support_slope=slope,
            width=width,
            error_pct=ERROR_PCT,
            min_wait=MIN_WAIT,
            max_wait=MAX_WAIT,
            shallow_width=SHALLOW_WIDTH,
        )
        reclaim_i = walked.get("reclaim_i")
        resist_reclaim = None
        atr_pct = None
        if reclaim_i is not None:
            sup = rail_at(y0, x0, slope, int(reclaim_i))
            resist_reclaim = sup + width
            a = float(atr[int(reclaim_i)]) if int(reclaim_i) < len(atr) else float("nan")
            c = float(close[int(reclaim_i)])
            if np.isfinite(a) and np.isfinite(c) and c > 0:
                atr_pct = a / c * 100.0
        span = None
        try:
            span = (
                pd.Timestamp(ch["h2_date"]) - pd.Timestamp(ch["start_date"])
            ).days
        except (TypeError, ValueError):
            span = None
        rows.append(
            {
                "stock": sym,
                "l1_date": ch.get("start_date"),
                "l2_date": ch.get("end_date"),
                "h2_date": ch.get("h2_date"),
                "channel_width": width,
                "channel_width_pct": ch.get("channel_width_pct"),
                "channel_span_days": span,
                "outcome": walked["outcome"],
                "cancel_date": _idx_date(dates, walked.get("cancel_i")),
                "cancel_undershoot_close_width": walked.get(
                    "cancel_undershoot_close_width"
                ),
                "cancel_undershoot_low_width": walked.get(
                    "cancel_undershoot_low_width"
                ),
                "shallow": walked.get("shallow"),
                "recovered_next": walked.get("recovered_next"),
                "tars_like": walked.get("tars_like"),
                "reclaim_date": _idx_date(dates, reclaim_i),
                "reclaim_within_wait": walked.get("reclaim_within_wait"),
                "reclaim_resist": resist_reclaim,
                "atr_pct": None if atr_pct is None else round(float(atr_pct), 4),
                "wait_bars_cancel": walked.get("wait_bars_cancel"),
                "wait_bars_reclaim": walked.get("wait_bars_reclaim"),
            }
        )
    return rows


def _open_overlap(
    book: pd.DataFrame,
    stock: str,
    fill_day: str,
) -> bool:
    if book is None or book.empty or not fill_day:
        return False
    sub = book.loc[book["stock"].astype(str).str.upper() == str(stock).upper()]
    if sub.empty:
        return False
    day = pd.Timestamp(fill_day)
    buys = pd.to_datetime(sub["buy_date"], errors="coerce")
    sells = pd.to_datetime(sub["sell_date"], errors="coerce")
    return bool(((buys <= day) & (sells >= day)).any())


def _slice_15m(panel: pd.DataFrame, lo: pd.Timestamp, hi: pd.Timestamp) -> pd.DataFrame:
    if panel is None or panel.empty:
        return panel
    idx = panel.index
    if getattr(idx, "tz", None) is not None:
        lo_ts = pd.Timestamp(lo).tz_localize("UTC")
        hi_ts = pd.Timestamp(hi).tz_localize("UTC")
    else:
        lo_ts = pd.Timestamp(lo)
        hi_ts = pd.Timestamp(hi)
    return panel.loc[(idx >= lo_ts) & (idx < hi_ts)]


def _symbol_15m_windows(cancels: pd.DataFrame) -> Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]:
    out: Dict[str, Tuple[pd.Timestamp, pd.Timestamp]] = {}
    work = cancels.loc[cancels["reclaim_within_wait"].fillna(False)].copy()
    if work.empty:
        return out
    work["_d"] = pd.to_datetime(work["reclaim_date"], errors="coerce")
    for sym, part in work.groupby(work["stock"].astype(str).str.upper()):
        days = part["_d"].dropna()
        if days.empty:
            continue
        lo = days.min() - pd.Timedelta(days=7)
        hi = days.max() + pd.Timedelta(days=400)
        out[str(sym)] = (pd.Timestamp(lo), pd.Timestamp(hi))
    return out


def apply_last_15m_sleeve(
    cancels: pd.DataFrame,
    panels_15m: Dict[str, pd.DataFrame],
    book: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[dict] = []
    windows = _symbol_15m_windows(cancels)
    index_cache: Dict[str, dict] = {}
    flat_cache: Dict[str, list] = {}
    for _, row in cancels.iterrows():
        if not bool(row.get("reclaim_within_wait")):
            continue
        day = row.get("reclaim_date")
        rail = row.get("reclaim_resist")
        sym = str(row["stock"]).upper()
        rec = dict(row)
        rec["last15m_status"] = "no_reclaim"
        rec["buy_price"] = None
        rec["sell_price"] = None
        rec["gain_pct"] = None
        rec["gain_pct_net"] = None
        rec["exit_reason"] = None
        rec["book_open_overlap"] = False
        if pd.isna(day) or pd.isna(rail):
            rec["last15m_status"] = "no_rail"
            rows.append(rec)
            continue
        rec["book_open_overlap"] = _open_overlap(book, sym, str(day))
        df15 = panels_15m.get(sym)
        if df15 is None or df15.empty:
            rec["last15m_status"] = "no_15m"
            rows.append(rec)
            continue
        if sym not in index_cache:
            lo, hi = windows.get(
                sym,
                (
                    pd.Timestamp(day) - pd.Timedelta(days=7),
                    pd.Timestamp(day) + pd.Timedelta(days=400),
                ),
            )
            sliced = _slice_15m(df15, lo, hi)
            if sliced is None or sliced.empty:
                rec["last15m_status"] = "no_15m"
                rows.append(rec)
                continue
            index_cache[sym] = index_rth_15m_by_session(sliced)
            flat_cache[sym] = flatten_rth_sessions(index_cache[sym])
        got = purchase_last_rth_open_above_mid(
            None,
            signal_session_date=day,
            rail=float(rail),
            session_index=index_cache[sym],
        )
        if not got.filled or got.fill_px is None:
            rec["last15m_status"] = got.reason or "no_fill"
            rows.append(rec)
            continue
        atr_d = None
        atr_pct = row.get("atr_pct")
        if pd.notna(atr_pct):
            try:
                atr_d = float(got.fill_px) * float(atr_pct) / 100.0
            except (TypeError, ValueError):
                atr_d = None
        exited = simulate_exit_15m_next_mid(
            flat_cache[sym],
            entry_ts=got.exec_bar_ts or got.hit_bar_ts,
            entry_px=float(got.fill_px),
            atr_at_entry=atr_d,
            atr_stop_mult=ATR_STOP_MULT,
        )
        rec["last15m_status"] = "filled"
        rec["buy_date"] = str(day)
        rec["buy_price"] = round(float(got.fill_px), 4)
        rec["buy_time"] = str(got.exec_bar_ts or got.hit_bar_ts or "")
        if exited.filled and exited.sell_px is not None:
            gain = (float(exited.sell_px) / float(got.fill_px) - 1.0) * 100.0
            rec["sell_price"] = round(float(exited.sell_px), 4)
            rec["sell_time"] = str(exited.exec_bar_ts or "")
            rec["gain_pct"] = round(float(gain), 4)
            rec["gain_pct_net"] = round(float(gain) - FRICTION_PCT, 4)
            rec["exit_reason"] = str(exited.exit_reason or "")
        else:
            rec["last15m_status"] = exited.reason or "no_exit"
        rows.append(rec)
    return pd.DataFrame(rows)


def _print_counts(title: str, c: dict) -> None:
    n_c = max(1, int(c.get("n_cancelled") or 0))
    n_s = max(1, int(c.get("n_shallow") or 0))
    n_t = max(1, int(c.get("n_tars_like") or 0))
    print("---- %s ----" % title, flush=True)
    print(json.dumps(c, indent=2), flush=True)
    print(
        "reclaim rates: all_cancel %.1f%%  shallow %.1f%%  tars_like %.1f%%"
        % (
            100.0 * float(c.get("n_cancel_reclaim") or 0) / n_c,
            100.0 * float(c.get("n_shallow_reclaim") or 0) / n_s,
            100.0 * float(c.get("n_tars_like_reclaim") or 0) / n_t,
        ),
        flush=True,
    )


def _print_sleeve(title: str, df: pd.DataFrame) -> dict:
    filled = df.loc[df["last15m_status"] == "filled"].copy() if not df.empty else df
    if filled.empty or "gain_pct" not in filled.columns:
        print("---- %s ---- n=0" % title)
        return {"label": title, "n": 0}
    g = pd.to_numeric(filled["gain_pct"], errors="coerce")
    years = summarize_by_year(filled, gain_col="gain_pct", buckets=YEAR_BUCKETS)
    s = book_stats(g)
    geo = geom_mean_year_pf(years)
    print("---- %s ----" % title)
    print(years.to_string(index=False))
    print(
        "n=%d E=%s PF=%s geo=%s WR=%s overlap_open=%d"
        % (
            s["n"],
            "n/a" if s["expectancy_pct"] is None else "%.3f" % s["expectancy_pct"],
            "n/a" if s["profit_factor"] is None else "%.3f" % s["profit_factor"],
            "n/a" if geo is None else "%.3f" % geo,
            "n/a" if s["win_rate_pct"] is None else "%.1f" % s["win_rate_pct"],
            int(filled["book_open_overlap"].fillna(False).sum())
            if "book_open_overlap" in filled.columns
            else 0,
        )
    )
    out = dict(s)
    out["label"] = title
    out["geo_year_pf"] = geo
    out["n_overlap_open"] = int(
        filled["book_open_overlap"].fillna(False).sum()
    ) if "book_open_overlap" in filled.columns else 0
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trades", type=Path, default=DEFAULT_TRADES)
    ap.add_argument("--outdir", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--symbols", type=str, default="")
    ap.add_argument("--skip-15m", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    t_all = time.perf_counter()
    trades = pd.read_csv(args.trades)
    trades["stock"] = trades["stock"].astype(str).str.upper()
    smoke = [s.strip().upper() for s in str(args.symbols).split(",") if s.strip()]
    symbols = sorted(set(smoke) if smoke else {str(s).upper() for s in trades["stock"].tolist()})
    if not symbols:
        LOG.error("No symbols")
        return 1

    end = pd.to_datetime(trades["buy_date"], errors="coerce").max()
    if pd.isna(end):
        end = pd.Timestamp("2026-09-12")
    end = pd.Timestamp(end) + pd.Timedelta(days=400)
    LOG.info("Symbols %d daily through %s", len(symbols), end.strftime("%Y-%m-%d"))

    t0 = time.perf_counter()
    panels = load_ohlcv_many(
        symbols,
        timeframe="1d",
        provider="ALPACA",
        start=datetime(2006, 1, 1),
        end=end.to_pydatetime(),
        fallback_provider="IB",
        merge_mode="prefix",
        workers=max(1, int(args.workers)),
        use_cache=True,
    )
    load_s = time.perf_counter() - t0
    n_have = sum(1 for s in symbols if panels.get(s) is not None and not panels[s].empty)
    LOG.info("Loaded ALPACA+IB 1d %d/%d in %.1fs", n_have, len(symbols), load_s)

    t1 = time.perf_counter()
    rows: List[dict] = []
    for sym in symbols:
        rows.extend(walk_symbol(sym, panels.get(sym)))
    setups = pd.DataFrame(rows)
    LOG.info("Walked %d setups in %.1fs", len(setups), time.perf_counter() - t1)
    if setups.empty:
        LOG.error("No setups")
        return 1

    counts = count_bucket(setups.to_dict("records"))
    _print_counts("all span365 H2", counts)
    outdir = dated_outdir(args.outdir)
    setups_path = outdir / "h2_cancel_reclaim_setups.csv"
    setups.to_csv(setups_path, index=False)
    LOG.info("Wrote setups %s", setups_path)
    if smoke:
        show = setups.loc[setups["outcome"] == OUTCOME_CANCELLED]
        cols = [
            "stock",
            "h2_date",
            "cancel_date",
            "cancel_undershoot_close_width",
            "shallow",
            "recovered_next",
            "tars_like",
            "reclaim_date",
            "wait_bars_cancel",
            "wait_bars_reclaim",
        ]
        print(show[cols].to_string(index=False), flush=True)

    cancels = setups.loc[setups["outcome"] == OUTCOME_CANCELLED].copy()
    sleeve_all = pd.DataFrame()
    sleeve_tars = pd.DataFrame()
    sleeve_stats: Dict[str, Any] = {}
    if not args.skip_15m and not cancels.empty:
        reclaim = cancels.loc[cancels["reclaim_within_wait"].fillna(False)].copy()
        need = sorted({str(s).upper() for s in reclaim["stock"].tolist()}) if not reclaim.empty else []
        LOG.info("Reclaim candidates %d setups / %d symbols for 15m", len(reclaim), len(need))
        panels_15m: Dict[str, pd.DataFrame] = {}
        if need:
            t2 = time.perf_counter()
            panels_15m = load_ohlcv_many(
                need,
                timeframe="15m",
                provider="IB",
                start=datetime(2018, 1, 1),
                end=end.to_pydatetime(),
                workers=max(1, int(args.workers)),
                use_cache=True,
            )
            LOG.info("Loaded IB 15m %d/%d in %.1fs", len(panels_15m), len(need), time.perf_counter() - t2)
        t3 = time.perf_counter()
        sleeve_all = apply_last_15m_sleeve(reclaim, panels_15m, trades)
        tars_like = reclaim.loc[reclaim["tars_like"].fillna(False)].copy()
        sleeve_tars = apply_last_15m_sleeve(tars_like, panels_15m, trades)
        LOG.info("15m sleeve apply in %.1fs", time.perf_counter() - t3)
        sleeve_stats["all_cancel_reclaim"] = _print_sleeve(
            "last15m all-cancel later-breakout", sleeve_all
        )
        sleeve_stats["tars_like_reclaim"] = _print_sleeve(
            "last15m TARS-like shallow+recovered later-breakout", sleeve_tars
        )
        if smoke and not sleeve_tars.empty:
            print(sleeve_tars.to_string(index=False), flush=True)

    if not sleeve_all.empty:
        sleeve_all.to_csv(outdir / "h2_cancel_reclaim_last15m_all.csv", index=False)
    if not sleeve_tars.empty:
        sleeve_tars.to_csv(outdir / "h2_cancel_reclaim_last15m_tars_like.csv", index=False)
    summary = {
        "n_symbols": len(symbols),
        "n_setups": int(len(setups)),
        "load_s": round(load_s, 1),
        "wall_s": round(time.perf_counter() - t_all, 1),
        "counts": counts,
        "sleeves": sleeve_stats,
        "params": {
            "error_pct": ERROR_PCT,
            "min_wait": MIN_WAIT,
            "max_wait": MAX_WAIT,
            "shallow_width": SHALLOW_WIDTH,
            "span_days": SPAN_DAYS,
            "window_bars": DAILY_WINDOW_BARS,
            "window_step_bars": DAILY_WINDOW_STEP_BARS,
        },
    }
    (outdir / "h2_cancel_reclaim_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    LOG.info("Wrote %s (wall %.1fs)", setups_path, time.perf_counter() - t_all)
    print("Wrote %s" % setups_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
