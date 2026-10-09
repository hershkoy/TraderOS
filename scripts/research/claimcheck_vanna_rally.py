#!/usr/bin/env python3
"""Claim-check Multi Strat Mark Vanna Rally long-call overnight.

Data: Yahoo VIX/SPY daily triggers; SPY OptionsDX EOD through 2023-12-29,
Theta Terminal EOD from 2024 (requires :25503 for the recent sleeve).

Usage (Windows CMD):
  venv\\Scripts\\activate
  set PYTHONPATH=.
  python scripts\\pipeline\\thetadata_terminal.py
  python scripts\\research\\claimcheck_vanna_rally.py
  python scripts\\research\\claimcheck_vanna_rally.py --variants baseline --end 2023-12-29
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.research.metrics import format_elapsed  # noqa: E402
from utils.research.vanna_rally import (  # noqa: E402
    CLAIMED,
    DEFAULT_RISK_FRAC,
    DEFAULT_START_EQUITY,
    DEFAULT_TP_MULT,
    book_stats,
    default_source,
    find_signal_days,
    load_spots,
    sessions_from_ohlc,
    simulate_book,
    spot_map_from_ohlc,
    trades_to_frame,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOG = logging.getLogger("claimcheck_vanna_rally")

DEFAULT_OUT = ROOT / "reports" / "research" / "vanna_rally"

VARIANT_SUITES: Dict[str, List[Dict[str, Any]]] = {
    "baseline": [
        {"marks": "mid", "leg_slip": 0.0, "tp_mult": 8.0, "tbill_apy": 0.0},
        {"marks": "mid", "leg_slip": 0.05, "tp_mult": 8.0, "tbill_apy": 0.0},
        {"marks": "ask_bid", "leg_slip": 0.0, "tp_mult": 8.0, "tbill_apy": 0.0},
        {"marks": "mid", "leg_slip": 0.0, "tp_mult": 10.0, "tbill_apy": 0.0},
        {"marks": "mid", "leg_slip": 0.0, "tp_mult": None, "tbill_apy": 0.0},
        {"marks": "mid", "leg_slip": 0.0, "tp_mult": 8.0, "tbill_apy": 0.05},
        {
            "marks": "mid",
            "leg_slip": 0.0,
            "tp_mult": 8.0,
            "tbill_apy": 0.05,
            "premium_lo": 0.70,
            "premium_hi": 1.00,
        },
    ],
}


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _tag(cfg: Dict[str, Any]) -> str:
    mark = str(cfg.get("marks") or "mid")
    slip = float(cfg.get("leg_slip") or 0.0)
    if mark == "mid" and slip > 0:
        base = f"mid_slip{slip:g}"
    else:
        base = mark
    tp = cfg.get("tp_mult", DEFAULT_TP_MULT)
    tp_s = "notp" if tp is None else f"tp{float(tp):g}x"
    parts = [base, tp_s]
    tbill = float(cfg.get("tbill_apy") or 0.0)
    if tbill > 0:
        parts.append(f"tbill{tbill:g}")
    if cfg.get("premium_lo") is not None:
        parts.append("premband")
    return "_".join(parts)


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, float) and (obj == float("inf") or obj == float("-inf")):
        return None
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    return obj


def _fmt(label: str, s: Dict[str, Any]) -> str:
    spy = s.get("spy") or {}
    return (
        f"{label}: n={s.get('n')} WR={s.get('win_rate_pct')}% PF={s.get('pf')} "
        f"PnL=${s.get('total_pnl')} meanRoP={s.get('mean_ret_on_prem_pct')}% "
        f"CAGR={s.get('cagr')} Sharpe={s.get('sharpe')} MDD={s.get('max_drawdown')} "
        f"end=${s.get('end_equity')} maxLL={s.get('max_consec_losses')} "
        f"SPY[cagr={spy.get('cagr')} sharpe={spy.get('sharpe')} mdd={spy.get('max_drawdown')}] "
        f"top10share={s.get('top10_share_of_wins')} years={s.get('years')}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default="2026-09-30")
    ap.add_argument("--marks", default="mid")
    ap.add_argument("--leg-slip", type=float, default=0.0)
    ap.add_argument("--tp-mult", type=float, default=DEFAULT_TP_MULT)
    ap.add_argument("--no-tp", action="store_true")
    ap.add_argument("--risk-frac", type=float, default=DEFAULT_RISK_FRAC)
    ap.add_argument("--tbill-apy", type=float, default=0.0)
    ap.add_argument("--otm", type=float, default=0.025)
    ap.add_argument("--premium-lo", type=float, default=None)
    ap.add_argument("--premium-hi", type=float, default=None)
    ap.add_argument("--variants", default="baseline", help="Named suite or empty for single")
    ap.add_argument("--outdir", default="", help="Default reports/research/vanna_rally/YYYY-MM-DD")
    ap.add_argument("--cache-dir", default="", help="Theta chain/eod cache")
    ap.add_argument("--max-signals", type=int, default=0, help="Debug cap on signals")
    args = ap.parse_args()

    start = _parse_date(args.start)
    end = _parse_date(args.end)
    outdir = Path(args.outdir) if args.outdir else DEFAULT_OUT / date.today().isoformat()
    outdir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else outdir / "theta_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    LOG.info("Claimed: %s", CLAIMED)
    LOG.info("Loading VIX/SPY spots %s -> %s", start, end)
    vix, spy = load_spots(start, end)
    signals = find_signal_days(vix, spy, start=start, end=end)
    LOG.info("Signal days: %d", len(signals))
    if args.max_signals and len(signals) > args.max_signals:
        signals = signals.iloc[: int(args.max_signals)]
        LOG.info("Truncated to %d signals", len(signals))

    sessions = sessions_from_ohlc(spy, start, end)
    spots = spot_map_from_ohlc(spy)
    source = default_source(cache_dir=cache_dir)

    if args.variants and args.variants in VARIANT_SUITES:
        configs = VARIANT_SUITES[args.variants]
    else:
        configs = [
            {
                "marks": args.marks,
                "leg_slip": args.leg_slip,
                "tp_mult": None if args.no_tp else args.tp_mult,
                "tbill_apy": args.tbill_apy,
                "premium_lo": args.premium_lo,
                "premium_hi": args.premium_hi,
            }
        ]

    summaries: Dict[str, Any] = {
        "claimed": CLAIMED,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "n_signals": int(len(signals)),
        "books": {},
        "wall_clock_s": None,
    }

    for cfg in configs:
        tag = _tag(cfg)
        LOG.info("=== Book %s ===", tag)
        book = simulate_book(
            source=source,
            signals=signals,
            sessions=sessions,
            spot_by_day=spots,
            start=start,
            end=end,
            mark_mode=str(cfg.get("marks") or "mid"),
            leg_slip=float(cfg.get("leg_slip") or 0.0),
            risk_frac=float(args.risk_frac),
            otm=float(args.otm),
            tp_mult=cfg.get("tp_mult", DEFAULT_TP_MULT),
            premium_lo=cfg.get("premium_lo"),
            premium_hi=cfg.get("premium_hi"),
            start_equity=DEFAULT_START_EQUITY,
            tbill_apy=float(cfg.get("tbill_apy") or 0.0),
            label=tag,
        )
        stats = book_stats(book, start_equity=DEFAULT_START_EQUITY)
        LOG.info(_fmt(tag, stats))
        summaries["books"][tag] = stats

        tf = trades_to_frame(book.trades)
        if not tf.empty:
            tf.to_csv(outdir / f"trades_{tag}.csv", index=False)
        if not book.equity.empty:
            book.equity.to_csv(outdir / f"equity_{tag}.csv", header=["equity"])
        if book.skipped:
            pd.DataFrame(book.skipped).to_csv(outdir / f"skipped_{tag}.csv", index=False)

    # Signal inventory
    if not signals.empty:
        inv = signals.copy()
        inv["session"] = inv.index.map(lambda ts: pd.Timestamp(ts).date())
        cols = [
            c
            for c in (
                "session",
                "open_vix",
                "prev_close",
                "low_vix",
                "close_vix",
                "open_spy",
                "high_spy",
                "close_spy",
            )
            if c in inv.columns
        ]
        inv[cols].to_csv(outdir / "signals.csv", index=False)

    elapsed = time.time() - t0
    summaries["wall_clock_s"] = round(elapsed, 1)
    summaries["wall_clock"] = format_elapsed(elapsed)
    with open(outdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(_jsonable(summaries), f, indent=2)

    LOG.info("Wrote %s (wall %s)", outdir, format_elapsed(elapsed))
    _print_verdict(summaries)
    return 0


def _print_verdict(summaries: Dict[str, Any]) -> None:
    books = summaries.get("books") or {}
    if not books:
        LOG.info("Verdict: no books")
        return
    # Prefer mid tp8x as headline
    key = next((k for k in books if k.startswith("mid_tp8")), next(iter(books)))
    s = books[key]
    spy = s.get("spy") or {}
    cagr = s.get("cagr")
    sharpe = s.get("sharpe")
    mdd = s.get("max_drawdown")
    claim = CLAIMED
    LOG.info(
        "Headline %s vs claim CAGR %.1f%% (claim %.1f%%) MDD %.1f%% (claim %.1f%%) "
        "WR %.1f%% (claim %.0f%%); SPY CAGR %.1f%% Sharpe %.2f MDD %.1f%%",
        key,
        (cagr or 0) * 100,
        claim["cagr"] * 100,
        (mdd or 0) * 100,
        claim["mdd"] * 100,
        s.get("win_rate_pct") or 0,
        claim["win_rate"] * 100,
        (spy.get("cagr") or 0) * 100,
        spy.get("sharpe") or 0,
        (spy.get("max_drawdown") or 0) * 100,
    )
    gate = (
        sharpe is not None
        and spy.get("sharpe") is not None
        and sharpe > 1.0
        and sharpe >= spy["sharpe"]
        and mdd is not None
        and mdd > spy.get("max_drawdown", -1)  # less negative is better
    )
    LOG.info("Promote gate (Sharpe>1, Sharpe>=SPY, MDD better than SPY): %s", gate)


if __name__ == "__main__":
    raise SystemExit(main())
