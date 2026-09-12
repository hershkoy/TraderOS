"""Reprice current_best 1d hot-cross sells with realistic 15m / next-open-mid exits.

current_best/1d_hot_cross.html keeps daily occupancy sells (same-bar ATR clip).
A buy-now 15m lerp85 fill cannot sell that stop on the next daily candle.

After (two modes, occupancy not re-walked):
  1) 15m-next-mid: ATR k=2 + 10% trail on RTH 15m. Decision on bar N, fill N+1 mid.
     Same-session bars after the fill ARE in play (unlike skip_entry_bar_stop on daily).
  2) daily-close-next-open-mid: same stop on the session; decide at the close,
     fill next session 09:30 ET 15m mid.
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

from backtest_channel_touch_trades import YEAR_BUCKETS, summarize_by_year  # noqa: E402
from compare_1d_last_15m_realistic_sells import (  # noqa: E402
    FRICTION_PCT,
    _log_book,
    apply_realistic_sells,
    build_hold_session_index,
    trades_from_compare,
)
from utils.data.ohlcv_loader import load_ohlcv_many  # noqa: E402
from utils.research.realistic_exits import (  # noqa: E402
    EXIT_MODE_15M_NEXT_MID,
    EXIT_MODE_DAILY_CLOSE_NEXT_OPEN_MID,
    EXIT_MODES,
    normalize_exit_mode,
)
from utils.research.report_paths import dated_outdir  # noqa: E402

LOG = logging.getLogger("hot_cross_realistic_sells")

DEFAULT_TRADES = (
    ROOT / "reports" / "ascending_channels" / "channel_touch_1d_hot_cross.csv"
)
DEFAULT_OUT_NAME = "hot_cross_realistic_sells_compare.csv"

STEM_MAP = {
    EXIT_MODE_15M_NEXT_MID: "channel_touch_1d_hot_cross_sell_15m_next_mid_span365",
    EXIT_MODE_DAILY_CLOSE_NEXT_OPEN_MID: (
        "channel_touch_1d_hot_cross_sell_eod_next_open_mid_span365"
    ),
}


def _profit_factor(gains: pd.Series) -> float:
    g = pd.to_numeric(gains, errors="coerce").dropna()
    wins = g[g > 0].sum()
    losses = (-g[g < 0]).sum()
    if losses <= 0:
        return float("inf") if wins > 0 else float("nan")
    return float(wins / losses)


def _geo_mean_year_pf(years: pd.DataFrame, *, min_n: int = 30) -> float:
    pfs = []
    for _, row in years.iterrows():
        if str(row.get("bucket") or "") == "FULL":
            continue
        try:
            n = int(row["n_trades"])
            pf = float(row["profit_factor"])
        except (TypeError, ValueError):
            continue
        if n < min_n or pf != pf or pf <= 0:
            continue
        pfs.append(pf)
    if not pfs:
        return float("nan")
    log_sum = sum(math.log(p) for p in pfs)
    return math.exp(log_sum / len(pfs))


def _write_summary(path: Path, *, mode: str, trades: pd.DataFrame, skipped: int) -> None:
    g = pd.to_numeric(trades["gain_pct"], errors="coerce")
    g_net = g - FRICTION_PCT
    lines = [
        "H2 resistance-break unique-symbol/day (span<=365)",
        "intraday_trigger=hot-cross",
        "hot_cross_fill=lerp85",
        "realistic_exit=%s" % mode,
        "Note: Before book kept daily occupancy sells (unrealistic same-bar clip).",
        "Note: Occupancy not re-walked. ATR k=2 clamp 1.5-6% + 10% trail.",
        "Note: 15m-next-mid can stop out later the same session (daily skip_entry_bar_stop cannot).",
        "",
        "n_trades=%d" % len(trades),
        "n_skipped=%d" % skipped,
        "win_rate_pct=%.2f" % float((g > 0).mean() * 100.0),
        "avg_gain_pct=%.2f" % float(g.mean()),
        "median_gain_pct=%.2f" % float(g.median()),
        "expectancy_pct=%.2f" % float(g.mean()),
        "profit_factor=%.3f" % _profit_factor(g),
        "expectancy_net_0.25=%.2f" % float(g_net.mean()),
        "profit_factor_net_0.25=%.3f" % _profit_factor(g_net),
        "",
        "Exit: %s" % mode,
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _log_years(label: str, ok: pd.DataFrame, *, gain_col: str) -> None:
    if ok.empty:
        return
    years = summarize_by_year(ok, gain_col=gain_col, buckets=YEAR_BUCKETS)
    LOG.info("%s year buckets:\n%s", label, years.to_string(index=False))
    LOG.info("%s geo-mean year PF (n>=30)=%.3f", label, _geo_mean_year_pf(years))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trades", type=Path, default=DEFAULT_TRADES)
    ap.add_argument("--outdir", type=Path, default=None)
    ap.add_argument("--out-name", type=str, default=DEFAULT_OUT_NAME)
    ap.add_argument(
        "--mode",
        default="both",
        help="15m-next-mid, daily-close-next-open-mid, or both",
    )
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--pad-before", type=int, default=5)
    ap.add_argument("--pad-after", type=int, default=21)
    ap.add_argument("--symbols", type=str, default="", help="Optional comma list")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    trades = pd.read_csv(args.trades)
    trades["stock"] = trades["stock"].astype(str).str.upper()
    want = str(args.symbols or "").strip().upper()
    if want:
        keep = {s.strip() for s in want.split(",") if s.strip()}
        trades = trades[trades["stock"].isin(keep)].copy()
        LOG.info("Restricted to %s n=%d", sorted(keep), len(trades))
    if trades.empty:
        LOG.error("No trades in %s", args.trades)
        return 1

    modes: Sequence[str]
    raw_mode = str(args.mode or "both").strip().lower()
    if raw_mode in ("both", "all"):
        modes = EXIT_MODES
    else:
        modes = (normalize_exit_mode(args.mode),)

    symbols = sorted({str(s).upper() for s in trades["stock"].tolist()})
    buy_min = pd.to_datetime(trades["buy_date"], errors="coerce").min() - timedelta(
        days=int(args.pad_before)
    )
    sell_max = pd.to_datetime(trades["sell_date"], errors="coerce").max() + timedelta(
        days=int(args.pad_after)
    )
    LOG.info(
        "Trades n=%d symbols=%d 15m window %s -> %s",
        len(trades),
        len(symbols),
        buy_min.strftime("%Y-%m-%d"),
        sell_max.strftime("%Y-%m-%d"),
    )
    kept = pd.to_numeric(trades["gain_pct"], errors="coerce")
    LOG.info(
        "kept daily occupancy n=%d E_gross=%.3f PF_gross=%.3f E_net0.25=%.3f PF_net0.25=%.3f",
        len(trades),
        float(kept.mean()),
        _profit_factor(kept),
        float((kept - FRICTION_PCT).mean()),
        _profit_factor(kept - FRICTION_PCT),
    )
    _log_years("kept-sell", trades, gain_col="gain_pct")

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
    n_have = sum(1 for s in symbols if panels.get(s) is not None and not panels[s].empty)
    LOG.info("Loaded IB 15m %d/%d in %.1fs", n_have, len(symbols), time.perf_counter() - t0)

    indexed = build_hold_session_index(
        panels,
        trades,
        pad_before=int(args.pad_before),
        pad_after=int(args.pad_after),
    )
    LOG.info("Indexed 15m sessions for %d symbols", len(indexed))

    outdir = Path(args.outdir) if args.outdir is not None else dated_outdir()
    outdir.mkdir(parents=True, exist_ok=True)
    pieces = []
    for mode in modes:
        part = apply_realistic_sells(trades, indexed, mode=mode)
        pieces.append(part)
        ok = part[part["status"] == "ok"]
        skipped = int((part["status"] != "ok").sum())
        _log_book(mode, ok)
        years_ok = ok.copy()
        if not years_ok.empty:
            years_ok["gain_pct"] = pd.to_numeric(years_ok["gain_after"], errors="coerce")
            years_ok["gain_pct_net"] = years_ok["gain_pct"] - FRICTION_PCT
            _log_years(mode + " gross", years_ok, gain_col="gain_pct")
            _log_years(mode + " net0.25", years_ok, gain_col="gain_pct_net")
        bad = part.loc[part["status"] != "ok", "skip_reason"]
        if not bad.empty:
            print("%s skip_reason:" % mode)
            print(bad.value_counts().head(10).to_string())
        if ok.empty:
            continue
        book = trades_from_compare(trades, ok)
        trades_path = outdir / (STEM_MAP[mode] + ".csv")
        book.to_csv(trades_path, index=False)
        _write_summary(
            outdir / (STEM_MAP[mode] + "_summary.txt"),
            mode=mode,
            trades=book,
            skipped=skipped,
        )
        LOG.info("Wrote %s n=%d skipped=%d", trades_path, len(book), skipped)
        hcc = ok[ok["stock"] == "HCC"]
        if not hcc.empty:
            r = hcc.iloc[0]
            LOG.info(
                "HCC %s buy=%s sell_before=%s @ %s -> sell_after=%s @ %s gain %.3f -> %.3f",
                mode,
                r["buy_price"],
                r["sell_price_before"],
                r["sell_datetime_before"],
                r["sell_price_after"],
                r["sell_time_after"],
                r["gain_before"],
                r["gain_after"],
            )

    out = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    out_path = outdir / args.out_name
    out.to_csv(out_path, index=False)
    print("out:", out_path)
    LOG.info("Wall-clock %.1fs", time.perf_counter() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
