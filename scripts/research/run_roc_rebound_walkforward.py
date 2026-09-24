#!/usr/bin/env python3
"""Expanding walk-forward for ROC rebound ETF overlays.

Two protocols (friction default 0.10% RT):

1) Frozen A3 — rules fixed; score each OOS calendar year / holdout block.
2) Variant-select WF — on expanding IS (entry < OOS year) pick best of
   {A0,A1,A2,A3} by IS PF (min n), apply that variant to the OOS year.
   Honest check that A3 was not free after seeing the full sample.

Usage (Windows CMD):
  venv\\Scripts\\activate
  set PYTHONPATH=.
  python scripts\\research\\run_roc_rebound_walkforward.py
  python scripts\\research\\run_roc_rebound_walkforward.py --friction 0.10 --oos-start-year 2005
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.research.run_roc_rebound_etf_grid import (  # noqa: E402
    DEFAULT_OUT,
    ETF_UNIVERSE,
    GRID,
    load_etf_frames,
)
from utils.research.roc_rebound import (  # noqa: E402
    YEAR_BUCKETS,
    backtest_roc_rebound,
    book_from_trades,
    drop_top_n_book,
    geom_mean_year_pf,
    year_bucket_stats,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOG = logging.getLogger("roc_rebound_wf")
logging.getLogger("utils.db.timescaledb_client").setLevel(logging.WARNING)

# Selection set only — time-stops excluded from IS picker.
SELECT_VARIANTS: Sequence[str] = (
    "A0_baseline",
    "A1_sma200",
    "A2_vol_scale",
    "A3_sma200_vol",
)


def _overlays() -> Dict[str, Dict[str, Any]]:
    return {name: dict(ov) for name, ov in GRID if name in SELECT_VARIANTS}


def _frame_to_bt(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "ts" not in out.columns and isinstance(out.index, pd.DatetimeIndex):
        out = out.reset_index()
        if out.columns[0] != "ts":
            out = out.rename(columns={out.columns[0]: "ts"})
    return out


def collect_variant_trades(
    frames: Dict[str, pd.DataFrame],
    *,
    friction_pct: float,
) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {name: [] for name in SELECT_VARIANTS}
    overlays = _overlays()
    for name, overlay in overlays.items():
        t0 = time.perf_counter()
        for sym, raw in sorted(frames.items()):
            res = backtest_roc_rebound(
                _frame_to_bt(raw),
                symbol=sym,
                friction_pct=friction_pct,
                **overlay,
            )
            for tr in res.get("trades") or []:
                row = dict(tr)
                row["variant"] = name
                out[name].append(row)
        LOG.info(
            "Collected %s: %d trades (%.1fs)",
            name,
            len(out[name]),
            time.perf_counter() - t0,
        )
    return out


def _filter_trades(
    trades: List[Dict[str, Any]],
    *,
    start: Optional[str] = None,
    end: Optional[str] = None,
    year: Optional[int] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for tr in trades:
        ed = pd.Timestamp(tr["entry_date"])
        if year is not None and ed.year != int(year):
            continue
        if start is not None and ed < pd.Timestamp(start):
            continue
        if end is not None and ed > pd.Timestamp(end):
            continue
        rows.append(tr)
    return rows


def _pick_variant_is(
    trades_by_var: Dict[str, List[Dict[str, Any]]],
    *,
    is_end_exclusive: str,
    min_n: int,
) -> Tuple[str, Dict[str, Any]]:
    """Pick IS winner by profit factor (tie-break: expectancy, then n)."""
    scored: List[Tuple[str, Dict[str, Any]]] = []
    for name in SELECT_VARIANTS:
        is_tr = _filter_trades(trades_by_var[name], end=_day_before(is_end_exclusive))
        book = book_from_trades(is_tr)
        book["variant"] = name
        scored.append((name, book))
    eligible = [(n, b) for n, b in scored if int(b.get("n") or 0) >= int(min_n)]
    pool = eligible if eligible else scored

    def _key(item: Tuple[str, Dict[str, Any]]) -> Tuple[float, float, int]:
        b = item[1]
        pf = b.get("profit_factor")
        e = b.get("expectancy_pct")
        pf_f = float(pf) if pf is not None else -1.0
        e_f = float(e) if e is not None else -999.0
        return (pf_f, e_f, int(b.get("n") or 0))

    pool.sort(key=_key, reverse=True)
    best_name, best_book = pool[0]
    return best_name, best_book


def _day_before(iso_date: str) -> str:
    t = pd.Timestamp(iso_date) - pd.Timedelta(days=1)
    return str(t.date())


def expanding_year_folds(
    trades_by_var: Dict[str, List[Dict[str, Any]]],
    *,
    oos_start_year: int,
    oos_end_year: int,
    min_is_n: int,
) -> Dict[str, Any]:
    frozen_folds: List[Dict[str, Any]] = []
    select_folds: List[Dict[str, Any]] = []
    frozen_oos_trades: List[Dict[str, Any]] = []
    select_oos_trades: List[Dict[str, Any]] = []

    for year in range(int(oos_start_year), int(oos_end_year) + 1):
        oos_start = "%d-01-01" % year
        a3_oos = _filter_trades(trades_by_var["A3_sma200_vol"], year=year)
        a3_book = book_from_trades(a3_oos)
        frozen_folds.append(
            {
                "oos_year": year,
                "n": a3_book.get("n"),
                "PF": a3_book.get("profit_factor"),
                "E": a3_book.get("expectancy_pct"),
                "WR": a3_book.get("win_rate_pct"),
                "median": a3_book.get("median_pct"),
            }
        )
        frozen_oos_trades.extend(a3_oos)

        picked, is_book = _pick_variant_is(
            trades_by_var, is_end_exclusive=oos_start, min_n=min_is_n
        )
        picked_oos = _filter_trades(trades_by_var[picked], year=year)
        picked_book = book_from_trades(picked_oos)
        select_folds.append(
            {
                "oos_year": year,
                "picked": picked,
                "is_n": is_book.get("n"),
                "is_PF": is_book.get("profit_factor"),
                "is_E": is_book.get("expectancy_pct"),
                "n": picked_book.get("n"),
                "PF": picked_book.get("profit_factor"),
                "E": picked_book.get("expectancy_pct"),
                "WR": picked_book.get("win_rate_pct"),
                "median": picked_book.get("median_pct"),
            }
        )
        select_oos_trades.extend(picked_oos)

    return {
        "frozen_a3_folds": frozen_folds,
        "select_folds": select_folds,
        "frozen_a3_oos_book": _enrich_book(frozen_oos_trades),
        "select_oos_book": _enrich_book(select_oos_trades),
        "frozen_a3_oos_trades": frozen_oos_trades,
        "select_oos_trades": select_oos_trades,
    }


def _enrich_book(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    book = book_from_trades(trades)
    years = year_bucket_stats(trades, YEAR_BUCKETS)
    return {
        **book,
        "by_year_bucket": years,
        "geo_pf_year_min10": geom_mean_year_pf(years, min_n=10),
        "geo_pf_year_min30": geom_mean_year_pf(years, min_n=30),
        "drop_top3": drop_top_n_book(trades, 3),
        "drop_top5": drop_top_n_book(trades, 5),
    }


def holdout_block(
    trades_by_var: Dict[str, List[Dict[str, Any]]],
    *,
    is_end: str = "2015-12-31",
    oos_start: str = "2016-01-01",
    min_is_n: int,
) -> Dict[str, Any]:
    """Single pre-declared holdout: train <=2015, test 2016+."""
    out: Dict[str, Any] = {"is_end": is_end, "oos_start": oos_start, "variants": {}}
    for name in SELECT_VARIANTS:
        is_tr = _filter_trades(trades_by_var[name], end=is_end)
        oos_tr = _filter_trades(trades_by_var[name], start=oos_start)
        out["variants"][name] = {
            "is": _enrich_book(is_tr),
            "oos": _enrich_book(oos_tr),
        }
    # IS pick then OOS of winner
    picked, is_book = _pick_variant_is(
        trades_by_var,
        is_end_exclusive=oos_start,
        min_n=min_is_n,
    )
    oos_tr = _filter_trades(trades_by_var[picked], start=oos_start)
    out["selected"] = {
        "picked": picked,
        "is": is_book,
        "oos": _enrich_book(oos_tr),
    }
    return out


def _fold_hit_rate(folds: List[Dict[str, Any]], *, pf_key: str = "PF") -> Dict[str, Any]:
    pfs = []
    pos = 0
    n_empty = 0
    for f in folds:
        n = int(f.get("n") or 0)
        if n <= 0:
            n_empty += 1
            continue
        pf = f.get(pf_key)
        if pf is None:
            continue
        pfs.append(float(pf))
        if float(pf) >= 1.0:
            pos += 1
    return {
        "n_folds_with_trades": len(pfs),
        "n_empty_folds": n_empty,
        "pct_folds_pf_ge_1": None if not pfs else round(100.0 * pos / len(pfs), 1),
        "median_fold_pf": None if not pfs else round(float(pd.Series(pfs).median()), 3),
        "mean_fold_pf": None if not pfs else round(float(pd.Series(pfs).mean()), 3),
        "worst_fold_pf": None if not pfs else round(float(min(pfs)), 3),
    }


def _log_book(label: str, book: Dict[str, Any]) -> None:
    LOG.info(
        "%s  n=%s WR=%s PF=%s E=%s med=%s MDD=%s geo10=%s drop3_PF=%s drop3_E=%s",
        label,
        book.get("n"),
        book.get("win_rate_pct"),
        book.get("profit_factor"),
        book.get("expectancy_pct"),
        book.get("median_pct"),
        book.get("mdd_trade_eq_pct"),
        None
        if book.get("geo_pf_year_min10") is None
        else round(float(book["geo_pf_year_min10"]), 3),
        (book.get("drop_top3") or {}).get("profit_factor"),
        (book.get("drop_top3") or {}).get("expectancy_pct"),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="ROC rebound ETF expanding walk-forward")
    ap.add_argument("--friction", type=float, default=0.10)
    ap.add_argument("--oos-start-year", type=int, default=2005)
    ap.add_argument("--oos-end-year", type=int, default=2026)
    ap.add_argument("--min-is-n", type=int, default=30)
    ap.add_argument("--start", type=str, default="1993-01-01")
    ap.add_argument("--end", type=str, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--outdir", type=str, default=str(DEFAULT_OUT))
    ap.add_argument("--symbols", type=str, default=",".join(ETF_UNIVERSE))
    args = ap.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    start = datetime.fromisoformat(args.start)
    end = datetime.fromisoformat(args.end) if args.end else None

    t0 = time.perf_counter()
    frames = load_etf_frames(symbols, start=start, end=end, workers=args.workers)
    LOG.info("Loaded %d ETF frames", sum(1 for d in frames.values() if d is not None and not d.empty))

    trades_by_var = collect_variant_trades(frames, friction_pct=float(args.friction))

    LOG.info(
        "=== Expanding annual WF OOS %d..%d (friction=%.2f) ===",
        args.oos_start_year,
        args.oos_end_year,
        args.friction,
    )
    wf = expanding_year_folds(
        trades_by_var,
        oos_start_year=args.oos_start_year,
        oos_end_year=args.oos_end_year,
        min_is_n=args.min_is_n,
    )

    for f in wf["frozen_a3_folds"]:
        LOG.info(
            "  frozen A3 OOS %d  n=%s PF=%s E=%s WR=%s",
            f["oos_year"],
            f["n"],
            f["PF"],
            f["E"],
            f["WR"],
        )
    for f in wf["select_folds"]:
        LOG.info(
            "  select   OOS %d  picked=%s is_PF=%s -> n=%s PF=%s E=%s",
            f["oos_year"],
            f["picked"],
            f["is_PF"],
            f["n"],
            f["PF"],
            f["E"],
        )

    frozen_stats = _fold_hit_rate(wf["frozen_a3_folds"])
    select_stats = _fold_hit_rate(wf["select_folds"])
    LOG.info("Frozen A3 fold stats: %s", frozen_stats)
    LOG.info("Select WF fold stats: %s", select_stats)
    _log_book("Frozen A3 pooled OOS", wf["frozen_a3_oos_book"])
    _log_book("Select   pooled OOS", wf["select_oos_book"])

    # How often was A3 picked?
    picks = [f["picked"] for f in wf["select_folds"]]
    pick_counts = {p: picks.count(p) for p in sorted(set(picks))}
    LOG.info("IS pick counts: %s", pick_counts)

    LOG.info("=== Holdout block IS<=2015 / OOS>=2016 ===")
    hold = holdout_block(trades_by_var, min_is_n=args.min_is_n)
    for name in SELECT_VARIANTS:
        oos = hold["variants"][name]["oos"]
        isb = hold["variants"][name]["is"]
        LOG.info(
            "  %-14s IS n=%s PF=%s | OOS n=%s PF=%s E=%s geo10=%s",
            name,
            isb.get("n"),
            isb.get("profit_factor"),
            oos.get("n"),
            oos.get("profit_factor"),
            oos.get("expectancy_pct"),
            None
            if oos.get("geo_pf_year_min10") is None
            else round(float(oos["geo_pf_year_min10"]), 3),
        )
    sel = hold["selected"]
    LOG.info(
        "  IS-selected=%s -> OOS PF=%s E=%s n=%s",
        sel["picked"],
        sel["oos"].get("profit_factor"),
        sel["oos"].get("expectancy_pct"),
        sel["oos"].get("n"),
    )

    # Verdict heuristics
    a3_oos = hold["variants"]["A3_sma200_vol"]["oos"]
    a3_pf = a3_oos.get("profit_factor") or 0
    fold_med = frozen_stats.get("median_fold_pf") or 0
    fold_worst = frozen_stats.get("worst_fold_pf") or 0
    pct_ge1 = frozen_stats.get("pct_folds_pf_ge_1") or 0
    drop3_pf = (wf["frozen_a3_oos_book"].get("drop_top3") or {}).get("profit_factor") or 0
    if a3_pf >= 1.5 and fold_med >= 1.3 and pct_ge1 >= 70 and drop3_pf >= 1.4:
        verdict = "SOFT_PASS_OOS"
    elif a3_pf >= 1.2 and fold_med >= 1.0:
        verdict = "MARGINAL_OOS"
    else:
        verdict = "FAIL_OOS"
    if fold_worst is not None and fold_worst < 0.7:
        verdict = verdict + "_FRAGILE_YEAR"
    LOG.info("Verdict: %s", verdict)

    elapsed = time.perf_counter() - t0
    payload = {
        "friction_pct": args.friction,
        "symbols": symbols,
        "oos_start_year": args.oos_start_year,
        "oos_end_year": args.oos_end_year,
        "min_is_n": args.min_is_n,
        "expanding": {
            "frozen_a3_folds": wf["frozen_a3_folds"],
            "select_folds": wf["select_folds"],
            "frozen_a3_oos_book": {
                k: v for k, v in wf["frozen_a3_oos_book"].items()
            },
            "select_oos_book": {k: v for k, v in wf["select_oos_book"].items()},
            "frozen_fold_stats": frozen_stats,
            "select_fold_stats": select_stats,
            "pick_counts": pick_counts,
        },
        "holdout_2016": {
            "selected": hold["selected"],
            "variants": {
                k: {"is": v["is"], "oos": v["oos"]} for k, v in hold["variants"].items()
            },
        },
        "verdict": verdict,
        "elapsed_sec": round(elapsed, 2),
    }

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    summary_path = outdir / "roc_rebound_walkforward_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(wf["frozen_a3_folds"]).to_csv(
        outdir / "roc_rebound_wf_frozen_a3_folds.csv", index=False
    )
    pd.DataFrame(wf["select_folds"]).to_csv(
        outdir / "roc_rebound_wf_select_folds.csv", index=False
    )
    pd.DataFrame(wf["frozen_a3_oos_trades"]).to_csv(
        outdir / "roc_rebound_wf_frozen_a3_oos_trades.csv", index=False
    )
    LOG.info("Wrote %s", summary_path)
    LOG.info("Wall-clock %.1fs", elapsed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
