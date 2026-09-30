#!/usr/bin/env python3
"""Claim-check Brian Terry post-earnings put credit spreads (Theta Profits).

Mechanical approximation of YouTube H2G8_jisZP8:
  post-earnings bull put credit ~20d, 30-45 DTE, TP50 / SL 2x credit.

Usage (Windows CMD):
  venv\\Scripts\\activate
  set PYTHONPATH=.
  python scripts\\research\\claimcheck_post_earnings_pcs.py --symbols AAPL,MSFT,NVDA,AMZN,META --limit-earn 20
  python scripts\\research\\claimcheck_post_earnings_pcs.py --n-symbols 50
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.data.ohlcv_loader import load_ohlcv_many  # noqa: E402
from utils.research.panel import adv_dollar, list_daily_symbols  # noqa: E402
from utils.research.post_earnings_pcs import (  # noqa: E402
    CLAIMED,
    book_stats,
    collect_symbol_trades,
    load_or_fetch_earnings,
    year_bucket_stats,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOG = logging.getLogger("claimcheck_post_earnings_pcs")
logging.getLogger("utils.db.timescaledb_client").setLevel(logging.WARNING)
logging.getLogger("yfinance").setLevel(logging.WARNING)

DEFAULT_OUT = ROOT / "reports" / "research" / "post_earnings_pcs"


def _pick_adv_symbols(n: int, provider: str = "ALPACA") -> List[str]:
    symbols = [s.upper() for s in list_daily_symbols(provider=provider, timeframe="1d")]
    # Load recent window for ADV ranking
    end = datetime.utcnow()
    start = datetime(end.year - 1, end.month, end.day) if end.month != 2 or end.day != 29 else datetime(end.year - 1, 2, 28)
    frames = load_ohlcv_many(
        symbols,
        timeframe="1d",
        provider=provider,
        start=start,
        end=end,
        use_cache=True,
        workers=8,
        chunk_size=80,
        fallback_provider="IB",
        merge_mode="prefix",
    )
    # Build close/volume panels lightly
    closes = {}
    vols = {}
    for sym, df in frames.items():
        if df is None or df.empty:
            continue
        x = df.copy()
        if "ts" in x.columns:
            x = x.set_index(pd.to_datetime(x["ts"], utc=True))
        if "close" not in x.columns or "volume" not in x.columns:
            continue
        closes[sym] = x["close"].astype(float)
        vols[sym] = x["volume"].astype(float)
    if not closes:
        return symbols[:n]
    close = pd.DataFrame(closes).sort_index()
    vol = pd.DataFrame(vols).reindex(close.index)
    adv = adv_dollar(close, vol)
    last = adv.iloc[-1].dropna().sort_values(ascending=False)
    return [str(s).upper() for s in last.head(int(n)).index.tolist()]


def run(args: argparse.Namespace) -> Dict[str, Any]:
    t_wall = time.perf_counter()
    outdir = Path(args.outdir) if args.outdir else DEFAULT_OUT / date.today().isoformat()
    outdir.mkdir(parents=True, exist_ok=True)

    if args.symbols.strip():
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        LOG.info("Picking top-%d ADV symbols...", int(args.n_symbols))
        symbols = _pick_adv_symbols(int(args.n_symbols), provider=args.provider)
    LOG.info("Universe n=%d: %s", len(symbols), ",".join(symbols[:20]) + ("..." if len(symbols) > 20 else ""))

    start = datetime.strptime(args.start, "%Y-%m-%d") if args.start else datetime(2018, 1, 1)
    end = datetime.strptime(args.end, "%Y-%m-%d") if args.end else None
    start_d = start.date()
    end_d = end.date() if end else None

    t0 = time.perf_counter()
    frames = load_ohlcv_many(
        symbols,
        timeframe="1d",
        provider=args.provider,
        start=start,
        end=end,
        use_cache=not args.no_cache,
        workers=int(args.workers),
        chunk_size=int(args.chunk_size),
        fallback_provider="IB",
        merge_mode="prefix",
    )
    t_load = time.perf_counter() - t0
    LOG.info("OHLCV loaded %d/%d in %.1fs", len(frames), len(symbols), t_load)

    earn_cache = outdir / "earnings_cache.json"
    t0 = time.perf_counter()
    earnings = load_or_fetch_earnings(
        list(frames.keys()),
        cache_path=earn_cache,
        limit=int(args.earn_limit),
        refresh=bool(args.refresh_earnings),
    )
    t_earn = time.perf_counter() - t0
    LOG.info("Earnings fetched/cached in %.1fs", t_earn)

    all_trades: List[pd.DataFrame] = []
    reason_counts: Dict[str, int] = {}
    t0 = time.perf_counter()
    for sym, df in frames.items():
        evs = earnings.get(sym.upper(), [])
        trades, counts = collect_symbol_trades(
            df,
            evs,
            symbol=sym,
            start=start_d,
            end=end_d,
            use_hist_filter=not args.no_hist_filter,
            hist_lookback=int(args.hist_lookback),
            hist_min_pass=int(args.hist_min_pass),
            dte=int(args.dte),
            target_delta=float(args.target_delta),
            min_credit_frac=float(args.min_credit_frac),
            max_drop_pct=float(args.max_drop_pct),
            max_gap_pct=float(args.max_gap_pct),
            require_nonneg_gap=not args.allow_down_gap,
            iv_elev=float(args.iv_elev),
            iv_crush_days=int(args.iv_crush_days),
        )
        for k, v in counts.items():
            reason_counts[k] = reason_counts.get(k, 0) + int(v)
        if not trades.empty:
            all_trades.append(trades)
    t_sim = time.perf_counter() - t0

    book = (
        pd.concat(all_trades, ignore_index=True)
        if all_trades
        else pd.DataFrame()
    )
    if not book.empty:
        book = book.sort_values(["entry_date", "symbol"]).reset_index(drop=True)

    stats = book_stats(book)
    by_year = year_bucket_stats(book)
    wall = time.perf_counter() - t_wall

    summary: Dict[str, Any] = {
        "claimed": CLAIMED,
        "params": {
            "symbols_n": len(symbols),
            "start": args.start,
            "end": args.end,
            "dte": args.dte,
            "target_delta": args.target_delta,
            "min_credit_frac": args.min_credit_frac,
            "max_drop_pct": args.max_drop_pct,
            "max_gap_pct": args.max_gap_pct,
            "require_nonneg_gap": not args.allow_down_gap,
            "hist_filter": not args.no_hist_filter,
            "hist_lookback": args.hist_lookback,
            "hist_min_pass": args.hist_min_pass,
            "iv_elev": args.iv_elev,
            "iv_crush_days": args.iv_crush_days,
            "pricing": "BS put vertical + IV decay (not listed quotes)",
        },
        "stats": stats,
        "by_year": by_year,
        "reason_counts": reason_counts,
        "timings": {
            "wall_s": wall,
            "ohlcv_s": t_load,
            "earnings_s": t_earn,
            "sim_s": t_sim,
        },
        "match_notes": {
            "wr_vs_80": None
            if stats.get("win_rate_pct") is None
            else abs(float(stats["win_rate_pct"]) - 80.0) <= 10.0,
            "hold_vs_8": None
            if stats.get("avg_hold_days") is None
            else abs(float(stats["avg_hold_days"]) - 8.0) <= 4.0,
            "ror_in_8_10": None
            if stats.get("avg_ror_pct") is None
            else 8.0 <= float(stats["avg_ror_pct"]) <= 10.0,
        },
    }

    trades_path = outdir / "trades.csv"
    if not book.empty:
        book.to_csv(trades_path, index=False)
    summary_path = outdir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    txt = outdir / "summary.txt"
    lines = [
        "Post-earnings PCS claim-check (Brian Terry / Theta Profits H2G8_jisZP8)",
        f"Claimed: WR~80% hold~8d RoR 8-10% (2-3mo personal track)",
        f"Ours: n={stats.get('n')} WR={stats.get('win_rate_pct')} "
        f"hold={stats.get('avg_hold_days')} avg_RoR={stats.get('avg_ror_pct')} "
        f"PF={stats.get('profit_factor')}",
        f"TP%={stats.get('tp_pct')} SL%={stats.get('sl_pct')} expiry%={stats.get('expiry_pct')}",
        f"Reasons: {reason_counts}",
        f"Wall {wall:.1f}s  out={outdir}",
        "",
        "Caveat: BS+IV-decay marks, mechanical filters only (no OptionSlam/fundamentals).",
        "Winner TP50 on $0.80/$5 naturally prints ~9.5% RoR; pooled avg RoR includes stops.",
    ]
    for y, st in by_year.items():
        lines.append(
            f"  {y}: n={st.get('n')} WR={st.get('win_rate_pct')} "
            f"E_RoR={st.get('avg_ror_pct')} PF={st.get('profit_factor')}"
        )
    txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOG.info("\n".join(lines))
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="Post-earnings PCS claim-check")
    ap.add_argument("--symbols", default="", help="Comma list (overrides ADV pick)")
    ap.add_argument("--n-symbols", type=int, default=40)
    ap.add_argument("--provider", default="ALPACA")
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default="")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--chunk-size", type=int, default=50)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--earn-limit", type=int, default=40)
    ap.add_argument("--refresh-earnings", action="store_true")
    ap.add_argument("--no-hist-filter", action="store_true")
    ap.add_argument("--hist-lookback", type=int, default=8)
    ap.add_argument("--hist-min-pass", type=int, default=5)
    ap.add_argument("--dte", type=int, default=37)
    ap.add_argument("--target-delta", type=float, default=-0.20)
    ap.add_argument("--min-credit-frac", type=float, default=0.16)
    ap.add_argument("--max-drop-pct", type=float, default=-10.0)
    ap.add_argument("--max-gap-pct", type=float, default=15.0)
    ap.add_argument("--allow-down-gap", action="store_true")
    ap.add_argument("--iv-elev", type=float, default=1.35)
    ap.add_argument("--iv-crush-days", type=int, default=10)
    ap.add_argument("--outdir", default="")
    args = ap.parse_args()
    if not args.end:
        args.end = None
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
