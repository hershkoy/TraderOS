"""Occupancy walk for last-15m H2 + 15m L3 blend.

Unified FCFS max_open vs dedicated L3/H2 sleeves. Unique-symbol single account
unless --independent (two books may double a name). Timestamp occupancy
(buy_time / sell_time). Equal-dollar per taken fill; L3 and H2 winner scales
differ (~2.5% vs ~9%) so pooled E is mixed.

Pre-registered:
  unified unique-symbol: 8, 15, 20
  dedicated reserved unique-across: (15,5), (12,3), (8,4)
  dedicated independent (same caps)

Grade vs last-15m H2 n=2297 E +0.27 PF 1.09 geo 1.10. 2022 leftover, L3 taken
share, 2020-21 winner dollars. Occupancy is greedy on stored exits.

Usage (Windows CMD):
  venv\\Scripts\\activate
  set PYTHONPATH=.
  python scripts\\research\\overlay_h2_l3_blend_occupancy.py
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

from backtest_channel_touch_trades import YEAR_BUCKETS, summarize_by_year  # noqa: E402
from utils.research.h2_l3_blend_occupancy import (  # noqa: E402
    SLEEVE_H2,
    SLEEVE_L3,
    occupancy_dedicated_sleeves,
    occupancy_unified_fcfs,
    prepare_blend,
    skip_counts,
    sleeve_counts,
    year_sleeve_counts,
)
from utils.research.last_15m_volume_geometry import (  # noqa: E402
    book_stats,
    geom_mean_year_pf,
    winner_cut_skip,
)
from utils.research.report_paths import dated_outdir  # noqa: E402

LOG = logging.getLogger("overlay_h2_l3_blend_occupancy")

DEFAULT_H2 = (
    ROOT
    / "reports"
    / "ascending_channels"
    / "2026-09-11"
    / "channel_touch_h2_last_15m_open_mid_sell_15m_next_mid_span365.csv"
)
DEFAULT_L3 = ROOT / "reports" / "ascending_channels" / "channel_touch_15m_l3_wait12.csv"

UNIFIED_CAPS = (8, 15, 20)
DEDICATED: Tuple[Tuple[int, int], ...] = ((15, 5), (12, 3), (8, 4))


def _in_range(df: pd.DataFrame, start: str, end: str) -> pd.Series:
    d = pd.to_datetime(df["buy_date"], errors="coerce")
    return (d >= start) & (d <= end)


def _print_block(label: str, df: pd.DataFrame, skip: Optional[pd.DataFrame] = None) -> dict:
    g = pd.to_numeric(df["_gain"] if "_gain" in df.columns else df.get("gain_pct"), errors="coerce")
    if df.empty:
        g = pd.Series(dtype=float)
    years = summarize_by_year(df, gain_col="gain_pct", buckets=YEAR_BUCKETS) if not df.empty else pd.DataFrame()
    s = book_stats(g)
    geo = geom_mean_year_pf(years)
    y22 = df.loc[_in_range(df, "2022-01-01", "2022-12-31")] if not df.empty else df
    y23 = df.loc[_in_range(df, "2023-01-01", "2023-12-31")] if not df.empty else df
    both = df.loc[_in_range(df, "2022-01-01", "2023-12-31")] if not df.empty else df
    sleeves = sleeve_counts(df)
    y22s = year_sleeve_counts(df, 2022)
    y23s = year_sleeve_counts(df, 2023)
    skips = skip_counts(skip) if skip is not None else {}
    rec = {
        "label": label,
        "n": s.get("n"),
        "expectancy_pct": s.get("expectancy_pct"),
        "profit_factor": s.get("profit_factor"),
        "geo_year_pf": geo,
        "n_h2": sleeves.get(SLEEVE_H2, 0),
        "n_l3": sleeves.get(SLEEVE_L3, 0),
        "n_2022": int(len(y22)),
        "e_2022": book_stats(pd.to_numeric(y22["gain_pct"], errors="coerce") if not y22.empty else pd.Series(dtype=float)).get("expectancy_pct"),
        "pf_2022": book_stats(pd.to_numeric(y22["gain_pct"], errors="coerce") if not y22.empty else pd.Series(dtype=float)).get("profit_factor"),
        "n_h2_2022": y22s.get(SLEEVE_H2, 0),
        "n_l3_2022": y22s.get(SLEEVE_L3, 0),
        "n_2023": int(len(y23)),
        "e_2023": book_stats(pd.to_numeric(y23["gain_pct"], errors="coerce") if not y23.empty else pd.Series(dtype=float)).get("expectancy_pct"),
        "pf_2023": book_stats(pd.to_numeric(y23["gain_pct"], errors="coerce") if not y23.empty else pd.Series(dtype=float)).get("profit_factor"),
        "n_h2_2023": y23s.get(SLEEVE_H2, 0),
        "n_l3_2023": y23s.get(SLEEVE_L3, 0),
        "e_2022_23": book_stats(pd.to_numeric(both["gain_pct"], errors="coerce") if not both.empty else pd.Series(dtype=float)).get("expectancy_pct"),
        "pf_2022_23": book_stats(pd.to_numeric(both["gain_pct"], errors="coerce") if not both.empty else pd.Series(dtype=float)).get("profit_factor"),
    }
    rec.update(skips)
    print("---- %s ----" % label)
    print(
        "n=%s H2=%s L3=%s E=%s PF=%s geo=%s | 2022 n=%s H2=%s L3=%s E=%s PF=%s | 2023 n=%s H2=%s L3=%s E=%s PF=%s"
        % (
            rec["n"],
            rec["n_h2"],
            rec["n_l3"],
            rec["expectancy_pct"],
            rec["profit_factor"],
            None if geo is None else round(geo, 3),
            rec["n_2022"],
            rec["n_h2_2022"],
            rec["n_l3_2022"],
            rec["e_2022"],
            rec["pf_2022"],
            rec["n_2023"],
            rec["n_h2_2023"],
            rec["n_l3_2023"],
            rec["e_2023"],
            rec["pf_2023"],
        )
    )
    if skips:
        print("  skips %s" % skips)
    if not years.empty:
        print(years.to_string(index=False))
    return rec


def _winner_vs_h2(h2: pd.DataFrame, kept: pd.DataFrame, rec: dict) -> None:
    """H2 winner-$ vs unconstrained H2 (L3 fills ignored)."""
    h2_keep = pd.Series(False, index=h2.index)
    if kept is None or kept.empty:
        cut = winner_cut_skip(pd.to_numeric(h2["gain_pct"], errors="coerce"), h2_keep)
    else:
        taken = kept.loc[kept["sleeve"] == SLEEVE_H2]
        keys = set(zip(taken["_stock"].astype(str), taken["_buy"].astype("int64")))
        buy = pd.to_datetime(h2["buy_time"] if "buy_time" in h2.columns else h2["buy_date"], errors="coerce")
        if getattr(buy.dt, "tz", None) is not None:
            buy = buy.dt.tz_convert("UTC").dt.tz_localize(None)
        stock = h2["stock"].astype(str).str.upper()
        h2_keep = pd.Series(
            [(str(s), int(t.value) if t == t else -1) in keys for s, t in zip(stock, buy)],
            index=h2.index,
        )
        cut = winner_cut_skip(pd.to_numeric(h2["gain_pct"], errors="coerce"), h2_keep)
    rec["h2_cut_n"] = cut["n_dropped"]
    rec["h2_cut_win$"] = cut["winner_drop_gain_sum"]
    rec["h2_cut_lose$"] = cut["loser_drop_gain_sum"]
    rec["h2_cut_blunt"] = cut["blunt"]
    print(
        "  H2 winner-cut dropped=%s win$=%s lose$=%s blunt=%s"
        % (cut["n_dropped"], cut["winner_drop_gain_sum"], cut["loser_drop_gain_sum"], cut["blunt"])
    )
    d = pd.to_datetime(h2["buy_date"], errors="coerce")
    for name, start, end in (("2020-2021", "2020-01-01", "2021-12-31"), ("2024-2026", "2024-01-01", "2026-12-31")):
        m = (d >= start) & (d <= end)
        c2 = winner_cut_skip(pd.to_numeric(h2["gain_pct"], errors="coerce").loc[m], h2_keep.loc[m])
        rec["h2_cut_%s_win$" % name] = c2["winner_drop_gain_sum"]
        rec["h2_cut_%s_lose$" % name] = c2["loser_drop_gain_sum"]
        rec["h2_cut_%s_blunt" % name] = c2["blunt"]
        print(
            "  H2 winner-cut %s dropped=%s win$=%s lose$=%s blunt=%s"
            % (name, c2["n_dropped"], c2["winner_drop_gain_sum"], c2["loser_drop_gain_sum"], c2["blunt"])
        )


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--h2", type=Path, default=DEFAULT_H2)
    ap.add_argument("--l3", type=Path, default=DEFAULT_L3)
    ap.add_argument("--outdir", type=Path, default=None)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    h2 = pd.read_csv(args.h2)
    l3 = pd.read_csv(args.l3)
    h2["stock"] = h2["stock"].astype(str).str.upper()
    l3["stock"] = l3["stock"].astype(str).str.upper()
    blend = prepare_blend(h2, l3)
    rows: List[dict] = []

    rows.append(_print_block("H2_unconstrained", prepare_blend(h2, h2.iloc[0:0])))
    rows.append(_print_block("L3_unconstrained", prepare_blend(h2.iloc[0:0], l3)))
    rows.append(_print_block("overlay_concat_no_occ", blend))

    uniq, skip_u = occupancy_unified_fcfs(blend, max_open=0, unique_symbol=True)
    rec = _print_block("overlay_unique_symbol_nocap", uniq, skip_u)
    _winner_vs_h2(h2, uniq, rec)
    rows.append(rec)

    for cap in UNIFIED_CAPS:
        kept, skip = occupancy_unified_fcfs(blend, max_open=cap, unique_symbol=True)
        rec = _print_block("UNIFIED_FCFS_%d" % cap, kept, skip)
        _winner_vs_h2(h2, kept, rec)
        rows.append(rec)

    for h2_cap, l3_cap in DEDICATED:
        kept, skip = occupancy_dedicated_sleeves(
            blend, h2_cap=h2_cap, l3_cap=l3_cap, unique_across=True
        )
        rec = _print_block("DEDICATED_%d_%d_unique" % (h2_cap, l3_cap), kept, skip)
        _winner_vs_h2(h2, kept, rec)
        rows.append(rec)
        kept_i, skip_i = occupancy_dedicated_sleeves(
            blend, h2_cap=h2_cap, l3_cap=l3_cap, unique_across=False
        )
        rec_i = _print_block("DEDICATED_%d_%d_indep" % (h2_cap, l3_cap), kept_i, skip_i)
        _winner_vs_h2(h2, kept_i, rec_i)
        rows.append(rec_i)

    outdir = args.outdir or dated_outdir(ROOT / "reports" / "ascending_channels")
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "h2_l3_blend_occupancy_summary.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    LOG.info("Wrote %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
