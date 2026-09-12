"""Last-15m shakeout-confirm diagnostics and same-list entry gates.

--diag-extras: split occupancy-surviving extras vs first-break parent on the
current_best last-15m N+1 book. Occupancy is not re-walked.

--gates: same-list runaway pos / buy_pct / morning-star / engulf / hammer on a
confirm-only (or any) last-15m trades CSV. Occupancy is not re-walked.

Usage (Windows CMD):
  venv\\Scripts\\activate
  set PYTHONPATH=.
  python scripts\\research\\overlay_last_15m_shakeout_confirm.py --diag-extras
  python scripts\\research\\overlay_last_15m_shakeout_confirm.py --gates --trades PATH --workers 8
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

from backtest_channel_touch_trades import YEAR_BUCKETS, summarize_by_year  # noqa: E402
from compare_1d_last_15m_realistic_sells import build_hold_session_index  # noqa: E402
from utils.data.ohlcv_loader import load_ohlcv_many  # noqa: E402
from utils.research.evening_doji_star import candle_from_session  # noqa: E402
from utils.research.last_15m_volume_geometry import (  # noqa: E402
    FRICTION_PCT,
    book_stats,
    fill_day_et,
    geom_mean_year_pf,
    skip_mask,
    winner_cut_skip,
)
from utils.research.morning_doji_star import (  # noqa: E402
    find_morning_doji_star,
    is_bullish_engulfing,
    is_dragonfly_doji,
    is_hammer,
)
from utils.research.report_paths import dated_outdir  # noqa: E402
from utils.research.session_volume_delta import sessions_from_by_day  # noqa: E402

LOG = logging.getLogger("overlay_sbo_confirm")

DEFAULT_TRADES = (
    ROOT
    / "reports"
    / "ascending_channels"
    / "2026-09-11"
    / "channel_touch_h2_last_15m_open_mid_sell_15m_next_mid_span365.csv"
)
GCOL = "gain_pct"
BUY_PCT_MIN = 50.0
STAR_LOOKBACK = 20
SMOKE_DEFAULT = ("AMPL", "VST", "TARS")


def _sbo_mask(trades: pd.DataFrame) -> pd.Series:
    if "shakeout_breakout" not in trades.columns:
        return pd.Series(False, index=trades.index)
    return trades["shakeout_breakout"].fillna(False).astype(str).str.lower().isin(
        ("true", "1", "yes")
    )


def _dump(label: str, df: pd.DataFrame) -> dict:
    s = book_stats(df[GCOL] if not df.empty else pd.Series(dtype=float))
    years = (
        summarize_by_year(df, gain_col=GCOL, buckets=YEAR_BUCKETS)
        if not df.empty
        else pd.DataFrame()
    )
    geo = geom_mean_year_pf(years)
    gtxt = "n/a" if geo is None else "%.3f" % geo
    ptxt = "n/a" if s["profit_factor"] is None else "%.3f" % s["profit_factor"]
    LOG.info(
        "%s n=%d E=%s PF=%s geo=%s WR=%s",
        label,
        s["n"],
        "%.4f" % s["expectancy_pct"] if s["expectancy_pct"] is not None else "n/a",
        ptxt,
        gtxt,
        s["win_rate_pct"],
    )
    if not years.empty:
        print(years.to_string(index=False))
    rec = dict(s)
    rec["label"] = label
    rec["geo_year_pf"] = geo
    return rec


def run_diag_extras(trades: pd.DataFrame) -> List[dict]:
    sbo = _sbo_mask(trades)
    parent = trades.loc[~sbo]
    extra = trades.loc[sbo]
    rows = [
        _dump("ALL last-15m N+1", trades),
        _dump("PARENT first-break", parent),
        _dump("EXTRAS occupancy-surviving", extra),
    ]
    cut = winner_cut_skip(trades[GCOL], sbo)
    print("==== extras-only (skip parent) winner$ ====")
    for k, v in cut.items():
        print("  %s=%s" % (k, v))
    rows.append({"label": "extras_only_cut", **cut})
    return rows


def _prior_and_fill(sessions, fill_day: date):
    before = [s for s in sessions if s.session_date < fill_day]
    fill = next((s for s in sessions if s.session_date == fill_day), None)
    prior = before[-1] if before else None
    return before, prior, fill


def gate_masks(
    trades: pd.DataFrame,
    indexed: Dict[str, dict],
) -> Dict[str, pd.Series]:
    pos_125 = skip_mask(trades, max_channel_pos=1.25)
    pos_150 = skip_mask(trades, max_channel_pos=1.50)
    buy_ok = pd.Series(False, index=trades.index)
    star_ok = pd.Series(False, index=trades.index)
    engulf_ok = pd.Series(False, index=trades.index)
    hammer_ok = pd.Series(False, index=trades.index)
    for idx, row in trades.iterrows():
        fill_day = fill_day_et(row)
        by_day = indexed.get(str(row["stock"]).upper()) or {}
        if fill_day is None or not by_day:
            continue
        sessions = sessions_from_by_day(by_day)
        before, prior, fill = _prior_and_fill(sessions, fill_day)
        if fill is not None and float(fill.buy_pct) >= BUY_PCT_MIN:
            buy_ok.loc[idx] = True
        if before:
            start = before[max(0, len(before) - STAR_LOOKBACK)].session_date
            end = before[-1].session_date
            if find_morning_doji_star(sessions, start_day=start, end_day=end, require_gap=True):
                star_ok.loc[idx] = True
        if prior is not None and fill is not None:
            pc = candle_from_session(prior)
            fc = candle_from_session(fill)
            if is_bullish_engulfing(pc, fc):
                engulf_ok.loc[idx] = True
        if prior is not None:
            pc = candle_from_session(prior)
            if is_hammer(pc) or is_dragonfly_doji(pc):
                hammer_ok.loc[idx] = True
    return {
        "pos_1.25": pos_125,
        "pos_1.50": pos_150,
        "buy_pct_ge_50": buy_ok,
        "morning_star": star_ok,
        "engulfing": engulf_ok,
        "hammer_dragonfly": hammer_ok,
        "pos_1.25_and_buy50": pos_125 & buy_ok,
    }


def run_gates(trades: pd.DataFrame, indexed: Dict[str, dict]) -> List[dict]:
    baseline = pd.to_numeric(trades[GCOL], errors="coerce")
    _dump("baseline", trades)
    rows = []
    masks = gate_masks(trades, indexed)
    for label, keep in masks.items():
        part = trades.loc[keep].copy()
        rec = _dump(label, part)
        cut = winner_cut_skip(baseline, keep)
        print("---- %s winner-cut ----" % label)
        for k, v in cut.items():
            print("  %s=%s" % (k, v))
        rec.update({("cut_" + k): v for k, v in cut.items()})
        rows.append(rec)
    return rows


def _smoke_print(row: pd.Series, by_day: dict) -> None:
    fill_day = fill_day_et(row)
    print("")
    print(
        "==== SMOKE %s fill=%s px=%.4f pos=%s form=%s sbo=%s gain=%.3f ===="
        % (
            str(row["stock"]).upper(),
            row.get("buy_time") or row.get("buy_date"),
            float(row["buy_price"]),
            row.get("channel_pos"),
            row.get("formation_beyond_width"),
            row.get("shakeout_breakout"),
            float(row["gain_pct"]),
        )
    )
    if fill_day is None or not by_day:
        print("  no 15m / fill_day")
        return
    sessions = sessions_from_by_day(by_day)
    before, prior, fill = _prior_and_fill(sessions, fill_day)
    if fill is None:
        print("  no fill session")
        return
    print("  fill buy_pct=%.1f close=%.4f" % (float(fill.buy_pct), float(fill.close)))
    if prior is not None:
        pc = candle_from_session(prior)
        fc = candle_from_session(fill)
        print(
            "  prior hammer=%s dragon=%s engulf=%s"
            % (is_hammer(pc), is_dragonfly_doji(pc), is_bullish_engulfing(pc, fc))
        )
    if before:
        start = before[max(0, len(before) - STAR_LOOKBACK)].session_date
        end = before[-1].session_date
        star = find_morning_doji_star(sessions, start_day=start, end_day=end, require_gap=True)
        print("  morning_star=%s" % star)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trades", type=Path, default=DEFAULT_TRADES)
    ap.add_argument("--outdir", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--pad-before", type=int, default=21)
    ap.add_argument("--pad-after", type=int, default=5)
    ap.add_argument("--symbols", type=str, default="")
    ap.add_argument("--diag-extras", action="store_true")
    ap.add_argument("--gates", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    trades = pd.read_csv(args.trades)
    trades["stock"] = trades["stock"].astype(str).str.upper()
    want = str(args.symbols or "").strip().upper()
    smoke_syms = set()
    if want:
        smoke_syms = {s.strip() for s in want.split(",") if s.strip()}
        trades = trades[trades["stock"].isin(smoke_syms)].copy()
        LOG.info("Restricted to %s n=%d", sorted(smoke_syms), len(trades))
    if trades.empty:
        LOG.error("No trades in %s", args.trades)
        return 1

    outdir = Path(args.outdir) if args.outdir is not None else dated_outdir()
    outdir.mkdir(parents=True, exist_ok=True)
    do_diag = bool(args.diag_extras) or not bool(args.gates)
    if do_diag:
        rows = run_diag_extras(trades)
        pd.DataFrame(rows).to_csv(outdir / "last_15m_shakeout_confirm_diag.csv", index=False)

    if args.gates or smoke_syms:
        symbols = sorted({str(s).upper() for s in trades["stock"].tolist()})
        buy_min = pd.to_datetime(trades["buy_date"], errors="coerce").min() - timedelta(
            days=int(args.pad_before)
        )
        sell_max = pd.to_datetime(trades["sell_date"], errors="coerce").max() + timedelta(
            days=int(args.pad_after)
        )
        t0 = time.perf_counter()
        panels = load_ohlcv_many(
            symbols,
            timeframe="15m",
            provider="IB",
            start=buy_min,
            end=sell_max,
            workers=int(args.workers),
            use_cache=True,
        )
        LOG.info(
            "Loaded IB 15m %d/%d in %.1fs",
            sum(1 for s in symbols if panels.get(s) is not None and not panels[s].empty),
            len(symbols),
            time.perf_counter() - t0,
        )
        indexed = build_hold_session_index(
            panels,
            trades,
            pad_before=int(args.pad_before),
            pad_after=int(args.pad_after),
        )
        if smoke_syms:
            for _, row in trades.iterrows():
                _smoke_print(row, indexed.get(str(row["stock"]).upper()) or {})
        if args.gates:
            g_rows = run_gates(trades, indexed)
            pd.DataFrame(g_rows).to_csv(
                outdir / "last_15m_shakeout_confirm_gates.csv", index=False
            )
    print("outdir:", outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
