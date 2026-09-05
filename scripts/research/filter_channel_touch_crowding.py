#!/usr/bin/env python3
"""Portfolio crowding / max-concurrent on the current_best 1d unique book.

Pre-registered (not fitted) rules. Gate: kept E and PF beat baseline, 2020-21
and 2024-26 both stay positive, drop-top-3 PF>1.

Hypotheses:
  H1: High n_open-at-entry trades have worse E in every year bucket.
  H2: Crowded calendar days (many unique-symbol fills) have worse E in every bucket.
  H3: Greedy max_open 5/8/10 (wait tie-break, not RS) beats baseline in both eras.
  H4: Skip days with n_day > 5 (EOD full-day count — nightly-honest, next-mid leak).
  H5: Causal same-day cap 2/3 FIFO-wait beats baseline in both eras.

Usage (Windows CMD):
  venv\\Scripts\\activate && set PYTHONPATH=. && python scripts\\research\\filter_channel_touch_crowding.py
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

from backtest_channel_touch_trades import YEAR_BUCKETS, summarize_by_year  # noqa: E402
from channel_touch_robustness import (  # noqa: E402
    apply_max_open,
    cap_same_day,
    concurrent_open_stats,
    n_open_at_entry,
    same_day_fill_count,
    skip_crowded_days,
    summarize_gains,
)
from utils.research.channel_touch_entry_model import trade_metrics  # noqa: E402
from utils.research.report_paths import dated_outdir, resolve_artifact  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("filter_channel_touch_crowding")

DEFAULT_TRADES = "channel_touch_full_h2_break_span365_unique_20260905_135126.csv"
ERA_2020 = "2020-2021"
ERA_2024 = "2024-2026"


def _gains(df: pd.DataFrame) -> np.ndarray:
    return pd.to_numeric(df["gain_pct_net"], errors="coerce").to_numpy(dtype=float)


def _drop3(df: pd.DataFrame) -> dict:
    g = _gains(df)
    if len(g) <= 3:
        return trade_metrics(np.array([]))
    keep = np.ones(len(g), dtype=bool)
    keep[np.argsort(g)[::-1][:3]] = False
    return trade_metrics(g[keep])


def _era_row(year_df: pd.DataFrame, bucket: str) -> dict:
    if year_df.empty or "bucket" not in year_df.columns:
        return {}
    hit = year_df.loc[year_df["bucket"] == bucket]
    if hit.empty:
        return {}
    return hit.iloc[0].to_dict()


def _verdict(base: dict, kept: dict, year_df: pd.DataFrame, drop3: dict) -> str:
    e_k = kept.get("expectancy_pct")
    e_b = base.get("expectancy_pct")
    pf_k = kept.get("profit_factor")
    pf_b = base.get("profit_factor")
    if e_k is None or e_b is None or pf_k is None or pf_b is None:
        return "no_promote"
    beat = float(e_k) > float(e_b) and float(pf_k) > float(pf_b)
    r20 = _era_row(year_df, ERA_2020)
    r24 = _era_row(year_df, ERA_2024)
    def _era_ok(row: dict) -> bool:
        if not row or row.get("expectancy_pct") is None or row.get("profit_factor") is None:
            return False
        return float(row["expectancy_pct"]) > 0.0 and float(row["profit_factor"]) > 1.0

    eras = _era_ok(r20) and _era_ok(r24)
    d3_ok = drop3.get("profit_factor") is not None and float(drop3["profit_factor"]) > 1.0
    if beat and eras and d3_ok:
        return "research_only"
    if eras and d3_ok and kept.get("n") and not beat:
        return "capital_only"
    return "no_promote"


def _pack(name: str, df: pd.DataFrame, base: dict) -> Tuple[dict, pd.DataFrame]:
    g = _gains(df)
    stats = summarize_gains(g, name)
    years = summarize_by_year(df, gain_col="gain_pct_net", buckets=YEAR_BUCKETS)
    d3 = _drop3(df)
    r20 = _era_row(years, ERA_2020)
    r24 = _era_row(years, ERA_2024)
    row = {
        "name": name,
        "n": stats["n"],
        "E": stats["expectancy_pct"],
        "PF": stats["profit_factor"],
        "WR": stats["win_rate_pct"],
        "med": stats["median_pct"],
        "drop3_E": d3.get("expectancy_pct"),
        "drop3_PF": d3.get("profit_factor"),
        "e2020": r20.get("expectancy_pct"),
        "pf2020": r20.get("profit_factor"),
        "n2020": r20.get("n_trades"),
        "e2024": r24.get("expectancy_pct"),
        "pf2024": r24.get("profit_factor"),
        "n2024": r24.get("n_trades"),
        "verdict": _verdict(base, stats, years, d3),
    }
    return row, years


def _bucket_table(df: pd.DataFrame, col: str, edges: List[int]) -> pd.DataFrame:
    work = df.copy()
    x = pd.to_numeric(work[col], errors="coerce")
    labels = []
    bins = [-0.5] + [float(e) + 0.5 for e in edges] + [1e9]
    for i, e in enumerate(edges):
        lo = 0 if i == 0 else edges[i - 1] + 1
        labels.append("%d-%d" % (lo, e) if lo != e else str(e))
    labels.append("%d+" % (edges[-1] + 1))
    work["_b"] = pd.cut(x, bins=bins, labels=labels)
    rows = []
    for lab, part in work.groupby("_b", observed=False):
        s = summarize_gains(_gains(part), str(lab))
        years = summarize_by_year(part, gain_col="gain_pct_net", buckets=YEAR_BUCKETS)
        r20 = _era_row(years, ERA_2020)
        r24 = _era_row(years, ERA_2024)
        rows.append(
            {
                "bucket": str(lab),
                "n": s["n"],
                "E": s["expectancy_pct"],
                "PF": s["profit_factor"],
                "e2020": r20.get("expectancy_pct"),
                "pf2020": r20.get("profit_factor"),
                "n2020": r20.get("n_trades"),
                "e2024": r24.get("expectancy_pct"),
                "pf2024": r24.get("profit_factor"),
                "n2024": r24.get("n_trades"),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="1d unique crowding / max-concurrent A/B")
    ap.add_argument("--trades", type=Path, default=None)
    ap.add_argument("--outdir", type=Path, default=ROOT / "reports" / "ascending_channels")
    args = ap.parse_args()
    args.outdir = dated_outdir(args.outdir)

    path = Path(args.trades) if args.trades is not None else resolve_artifact(DEFAULT_TRADES)
    if not path.exists():
        logger.error("Trades CSV not found: %s", path)
        return 1
    df = pd.read_csv(path)
    df["buy_date"] = pd.to_datetime(df["buy_date"], errors="coerce")
    df["sell_date"] = pd.to_datetime(df["sell_date"], errors="coerce")
    if "gain_pct_net" not in df.columns:
        logger.error("need gain_pct_net")
        return 1
    df = df.loc[df["gain_pct_net"].notna()].copy()
    df["n_open_at_entry"] = n_open_at_entry(df)
    df["n_day"] = same_day_fill_count(df)
    logger.info("Loaded n=%d from %s", len(df), path.name)

    expo = concurrent_open_stats(df)
    print("\n==== concurrent open (calendar) ====")
    print(expo)
    print("n_open_at_entry: median=%s p95=%s max=%s" % (
        int(df["n_open_at_entry"].median()),
        int(df["n_open_at_entry"].quantile(0.95)),
        int(df["n_open_at_entry"].max()),
    ))
    print("n_day: median=%s p90=%s max=%s" % (
        int(df["n_day"].median()),
        int(df["n_day"].quantile(0.90)),
        int(df["n_day"].max()),
    ))

    print("\n==== H1 n_open_at_entry buckets ====")
    open_tbl = _bucket_table(df, "n_open_at_entry", [0, 2, 5, 8, 12])
    print(open_tbl.to_string(index=False))

    print("\n==== H2 same-day fill-count buckets ====")
    day_tbl = _bucket_table(df, "n_day", [1, 2, 3, 5, 8])
    print(day_tbl.to_string(index=False))

    base_stats = summarize_gains(_gains(df), "baseline")
    books: List[Tuple[str, pd.DataFrame]] = [("baseline", df)]
    for cap in (5, 8, 10):
        books.append(("max_open_%d_wait" % cap, apply_max_open(df, cap, tie_break="wait")))
        books.append(("max_open_%d_rs" % cap, apply_max_open(df, cap, tie_break="rs")))
    for k in (2, 3):
        books.append(("cap_day_%d_wait" % k, cap_same_day(df, k, tie_break="wait")))
        books.append(("cap_day_%d_rs" % k, cap_same_day(df, k, tie_break="rs")))
    for mx in (3, 5, 8):
        books.append(("skip_day_gt_%d" % mx, skip_crowded_days(df, mx)))

    rows: List[dict] = []
    year_blocks: Dict[str, pd.DataFrame] = {}
    for name, part in books:
        row, years = _pack(name, part, base_stats)
        rows.append(row)
        year_blocks[name] = years
        print("\n==== %s verdict=%s ====" % (name, row["verdict"]))
        print(
            "n=%s E=%s PF=%s WR=%s  2020-21 E=%s PF=%s  2024-26 E=%s PF=%s  drop3 PF=%s"
            % (
                row["n"],
                row["E"],
                row["PF"],
                row["WR"],
                row["e2020"],
                row["pf2020"],
                row["e2024"],
                row["pf2024"],
                row["drop3_PF"],
            )
        )
        print(years.to_string(index=False))

    verd = pd.DataFrame(rows)
    print("\n==== summary ====")
    show = [
        "name", "n", "E", "PF", "e2020", "pf2020", "e2024", "pf2024",
        "drop3_PF", "verdict",
    ]
    print(verd[show].to_string(index=False))

    args.outdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    verd_path = args.outdir / ("channel_touch_1d_crowding_%s.csv" % stamp)
    open_path = args.outdir / ("channel_touch_1d_crowding_nopen_%s.csv" % stamp)
    day_path = args.outdir / ("channel_touch_1d_crowding_nday_%s.csv" % stamp)
    verd.to_csv(verd_path, index=False)
    open_tbl.to_csv(open_path, index=False)
    day_tbl.to_csv(day_path, index=False)
    print("wrote", verd_path)
    print("wrote", open_path)
    print("wrote", day_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
