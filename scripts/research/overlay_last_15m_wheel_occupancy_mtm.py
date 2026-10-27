"""Option-path MTM occupancy for last-15m wheel vs mark-at-cost and SPY.

Default: liquid ADV>=$25M & half<=15% ask/bid book, max_open=8.

Usage (Windows CMD):
  venv\\Scripts\\activate
  set PYTHONPATH=.
  python scripts\\research\\overlay_last_15m_wheel_occupancy_mtm.py
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.data.ohlcv_loader import load_ohlcv_many  # noqa: E402
from utils.research.last_15m_wheel import (  # noqa: E402
    ThetaEquitySource,
    filter_wheel_by_liquidity,
    filled_trades_to_mtm_rows,
    load_option_mark_paths,
    occupancy_fcfs_wheel,
    spy_buyhold_equity,
    wheel_slot_equity,
    wheel_slot_equity_mtm,
)
from utils.research.metrics import (  # noqa: E402
    passes_phase5_gates,
    perf_stats,
    rebase_equity,
    window_equity,
)
from utils.research.report_paths import dated_outdir  # noqa: E402

LOG = logging.getLogger("overlay_last_15m_wheel_mtm")

DEFAULT_MID = (
    ROOT
    / "reports/ascending_channels/2026-10-09/last_15m_wheel_mid_repaired_20261010_112646.csv"
)
DEFAULT_ASKBID = (
    ROOT
    / "reports/ascending_channels/2026-10-09/last_15m_wheel_ask_bid_20261010_112516.csv"
)
WINDOWS = (
    ("FULL", None, None),
    ("2022", "2022-01-01", "2022-12-31"),
    ("2023", "2023-01-01", "2023-12-31"),
    ("2022-23", "2022-01-01", "2023-12-31"),
    ("2024-26", "2024-01-01", "2026-12-31"),
)


def _stats(st) -> Dict[str, Any]:
    return {
        "cagr": round(float(st.cagr), 4),
        "sharpe": round(float(st.sharpe), 3),
        "max_drawdown": round(float(st.max_drawdown), 4),
        "pct_time_invested": round(float(st.pct_time_invested), 3),
        "n_trades": int(st.n_trades),
        "expectancy": round(float(st.expectancy), 4),
        "profit_factor": round(float(st.profit_factor), 3),
        "win_rate": round(float(st.win_rate), 4),
    }


def _score(
    label: str,
    eq: pd.Series,
    inv: pd.Series,
    spy_eq: pd.Series,
    trades: List[Dict[str, Any]],
    *,
    start: Optional[str],
    end: Optional[str],
    window: Optional[str],
) -> Dict[str, Any]:
    if start and end:
        eq_w, inv_w = window_equity(eq, inv, start, end)
        spy_w, _ = window_equity(spy_eq, None, start, end)
    else:
        eq_w = rebase_equity(eq)
        inv_w = inv.reindex(eq_w.index).fillna(0.0)
        spy_w = rebase_equity(spy_eq.reindex(eq_w.index).ffill().bfill())
    common = eq_w.index.intersection(spy_w.index)
    if len(common) < 2:
        return {"label": label, "gate_pass": False, "gate_reason": "insufficient equity"}
    eq_w = eq_w.reindex(common).ffill()
    inv_w = inv_w.reindex(common).fillna(0.0)
    spy_w = spy_w.reindex(common).ffill()
    eq_w = eq_w / float(eq_w.iloc[0])
    spy_w = spy_w / float(spy_w.iloc[0])
    if start and end:
        t_win = [
            t
            for t in trades
            if start <= pd.Timestamp(t["entry_date"]).strftime("%Y-%m-%d") <= end
        ]
    else:
        t_win = list(trades)
    rets = [float(t["gain_pct"]) / 100.0 for t in t_win]
    spy_st = perf_stats("SPY", spy_w, notes="price NAV", trade_rets=[])
    sleeve = perf_stats(
        label,
        eq_w,
        inv_w,
        spy_cagr=spy_st.cagr,
        spy_sharpe=spy_st.sharpe,
        notes="wheel MTM or mark-at-cost",
        trade_rets=rets,
    )
    ok, reason = passes_phase5_gates(sleeve, spy_st, window=window)
    LOG.info(
        "%s Sharpe=%.2f CAGR=%.1f%% MDD=%.1f%% | SPY Sharpe=%.2f | %s",
        label,
        sleeve.sharpe,
        100 * sleeve.cagr,
        100 * sleeve.max_drawdown,
        spy_st.sharpe,
        reason,
    )
    return {
        "label": label,
        "gate_pass": bool(ok),
        "gate_reason": reason,
        "sleeve": _stats(sleeve),
        "spy": _stats(spy_st),
        "occ_n": len(t_win),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trades-mid", type=Path, default=DEFAULT_MID)
    ap.add_argument("--trades-askbid", type=Path, default=DEFAULT_ASKBID)
    ap.add_argument("--max-open", type=int, default=8)
    ap.add_argument("--min-adv", type=float, default=25e6)
    ap.add_argument("--max-half-pct-of-mid", type=float, default=15.0)
    ap.add_argument("--mark-mode", choices=("mid", "ask_bid"), default="ask_bid")
    ap.add_argument("--outdir", type=Path, default=None)
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    mid_df = pd.read_csv(args.trades_mid)
    ab_df = pd.read_csv(args.trades_askbid)
    mid_f, ab_f, filt_meta = filter_wheel_by_liquidity(
        mid_df,
        ab_df,
        min_adv=args.min_adv,
        max_half_pct_of_mid=args.max_half_pct_of_mid,
    )
    book_df = ab_f if args.mark_mode == "ask_bid" else mid_f
    rows = filled_trades_to_mtm_rows(book_df)
    LOG.info(
        "liquid filter kept=%s mark_mode=%s occ_rows=%s meta=%s",
        filt_meta.get("n_kept"),
        args.mark_mode,
        len(rows),
        filt_meta,
    )
    kept, occ_meta = occupancy_fcfs_wheel(
        rows, max_open=int(args.max_open), unique_symbol=True
    )
    LOG.info("FCFS max_open=%s kept=%s skip=%s", args.max_open, occ_meta["n_kept"], occ_meta["n_skipped"])
    if not kept:
        LOG.error("no kept trades")
        return 2

    start = min(pd.Timestamp(r["entry_date"]) for r in kept) - pd.Timedelta(days=5)
    end = max(pd.Timestamp(r["exit_date"]) for r in kept) + pd.Timedelta(days=5)
    syms = sorted({r["symbol"] for r in kept} | {"SPY"})
    LOG.info("loading panels n=%s %s..%s", len(syms), start.date(), end.date())
    panels = load_ohlcv_many(
        syms,
        timeframe="1d",
        start=start.date(),
        end=end.date(),
        provider="ALPACA",
        fallback_provider="IB",
        merge_mode="prefix",
        use_cache=True,
    )
    spy = panels.get("SPY")
    if spy is None or spy.empty:
        LOG.error("SPY missing")
        return 2
    close = spy["close"].astype(float).copy()
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    calendar = close.index[
        (close.index >= start.normalize()) & (close.index <= end.normalize())
    ]
    spy_eq = spy_buyhold_equity(close, calendar)

    outdir = Path(args.outdir) if args.outdir else dated_outdir(
        ROOT / "reports" / "ascending_channels"
    )
    outdir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else (
        outdir / "theta_wheel_mtm_cache"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    source = ThetaEquitySource(cache_dir=cache_dir)

    t0 = time.time()
    mark_paths = load_option_mark_paths(kept, source, mark_mode=args.mark_mode)
    path_s = time.time() - t0
    LOG.info("mark paths wall=%.1fs", path_s)

    eq_cost, inv_cost = wheel_slot_equity(kept, calendar, max_open=int(args.max_open))
    eq_mtm, inv_mtm, mtm_meta = wheel_slot_equity_mtm(
        kept,
        calendar,
        max_open=int(args.max_open),
        mark_mode=args.mark_mode,
        spot_panels=panels,
        mark_paths=mark_paths,
    )
    LOG.info("mtm meta %s", mtm_meta)

    summary: Dict[str, Any] = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "max_open": int(args.max_open),
        "mark_mode": args.mark_mode,
        "filter": filt_meta,
        "occupancy": occ_meta,
        "mtm_meta": mtm_meta,
        "path_wall_s": round(path_s, 1),
        "books": {"mark_at_cost": {}, "option_mtm": {}},
    }
    for book_name, eq, inv in (
        ("mark_at_cost", eq_cost, inv_cost),
        ("option_mtm", eq_mtm, inv_mtm),
    ):
        windows = {}
        for wlabel, wstart, wend in WINDOWS:
            windows[wlabel] = _score(
                "%s_%s" % (book_name, wlabel),
                eq,
                inv,
                spy_eq,
                kept,
                start=wstart,
                end=wend,
                window=wlabel,
            )
        summary["books"][book_name] = {"windows": windows}

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = outdir / ("last_15m_wheel_occupancy_mtm_%s.json" % stamp)
    path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    LOG.info("wrote %s", path)

    # Compact compare table
    rows_out = []
    for book_name, blob in summary["books"].items():
        for wlabel, w in blob["windows"].items():
            rows_out.append(
                {
                    "book": book_name,
                    "window": wlabel,
                    "pass": w.get("gate_pass"),
                    "gate": w.get("gate_reason"),
                    "sharpe": (w.get("sleeve") or {}).get("sharpe"),
                    "cagr": (w.get("sleeve") or {}).get("cagr"),
                    "mdd": (w.get("sleeve") or {}).get("max_drawdown"),
                    "spy_sharpe": (w.get("spy") or {}).get("sharpe"),
                    "n": w.get("occ_n"),
                }
            )
    csv_path = outdir / ("last_15m_wheel_occupancy_mtm_%s.csv" % stamp)
    pd.DataFrame(rows_out).to_csv(csv_path, index=False)
    LOG.info("wrote %s", csv_path)
    print(pd.DataFrame(rows_out).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
