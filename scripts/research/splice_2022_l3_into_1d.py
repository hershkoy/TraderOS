"""Synthetic calendar splice: 1d book except 2022 = 15m L3 wait-12.

No live switch signal. Occupancy is not re-walked.

Usage (Windows CMD):
  venv\\Scripts\\activate
  set PYTHONPATH=.
  python scripts\\research\\splice_2022_l3_into_1d.py
  python scripts\\research\\generate_channel_touch_tv_report.py --trades reports\\ascending_channels\\2026-09-12\\synthetic_1d_last15m_2022_l3.csv --friction-pct 0.25 --tag synthetic_2022_l3 --summary reports\\ascending_channels\\2026-09-12\\synthetic_1d_last15m_2022_l3_summary.txt --comparison-json reports\\ascending_channels\\2026-09-12\\synthetic_1d_last15m_2022_l3_compare.json --title "Synthetic: last-15m 1d H2, 2022 replaced by 15m L3"
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

from backtest_channel_touch_trades import YEAR_BUCKETS, summarize_by_year  # noqa: E402
from utils.research.last_15m_volume_geometry import (  # noqa: E402
    book_stats,
    geom_mean_year_pf,
    splice_calendar_year,
)
from utils.research.report_paths import dated_outdir  # noqa: E402

LOG = logging.getLogger("splice_2022_l3")

LAST15M = (
    ROOT
    / "reports"
    / "ascending_channels"
    / "2026-09-11"
    / "channel_touch_h2_last_15m_open_mid_sell_15m_next_mid_span365.csv"
)
HOT_CROSS = ROOT / "reports" / "ascending_channels" / "channel_touch_1d_hot_cross.csv"
L3 = ROOT / "reports" / "ascending_channels" / "channel_touch_15m_l3_wait12.csv"


def _year_n(df: pd.DataFrame, year: int) -> int:
    d = pd.to_datetime(df["buy_date"], errors="coerce")
    return int((d.dt.year == int(year)).sum())


def _block(label: str, df: pd.DataFrame) -> dict:
    g = pd.to_numeric(df["gain_pct"], errors="coerce")
    years = summarize_by_year(df, gain_col="gain_pct", buckets=YEAR_BUCKETS)
    s = book_stats(g)
    geo = geom_mean_year_pf(years)
    y22 = df.loc[pd.to_datetime(df["buy_date"], errors="coerce").dt.year == 2022]
    s22 = book_stats(pd.to_numeric(y22["gain_pct"], errors="coerce"))
    LOG.info(
        "%s n=%d E=%s PF=%s geo=%s WR=%s | 2022 n=%d E=%s PF=%s",
        label,
        s["n"],
        s["expectancy_pct"],
        s["profit_factor"],
        None if geo is None else round(geo, 3),
        s["win_rate_pct"],
        s22["n"],
        s22["expectancy_pct"],
        s22["profit_factor"],
    )
    print("---- %s ----" % label)
    print(years.to_string(index=False))
    return {
        "book": label,
        "n": s["n"],
        "wr": s["win_rate_pct"],
        "e": s["expectancy_pct"],
        "pf": s["profit_factor"],
        "geo": geo,
        "n_2022": s22["n"],
        "e_2022": s22["expectancy_pct"],
        "pf_2022": s22["profit_factor"],
        "years": years,
    }


def _cmp_row(rec: dict, *, highlight: bool = False) -> dict:
    return {
        "book": rec["book"],
        "n": rec["n"],
        "wr": rec["wr"],
        "e": rec["e"],
        "pf": rec["pf"],
        "highlight": highlight,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, default=LAST15M)
    ap.add_argument("--hot-cross", type=Path, default=HOT_CROSS)
    ap.add_argument("--l3", type=Path, default=L3)
    ap.add_argument("--year", type=int, default=2022)
    ap.add_argument("--outdir", type=Path, default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    base = pd.read_csv(args.base)
    l3 = pd.read_csv(args.l3)
    recs = []
    recs.append(_block("last15m_1d_H2", base))
    recs.append(_block("15m_L3_wait12_all_years", l3))
    spliced = splice_calendar_year(
        base,
        l3,
        year=int(args.year),
        src_base="last15m_1d",
        src_repl="l3_wait12",
    )
    recs.append(_block("synthetic_last15m_2022_L3", spliced))

    if args.hot_cross.is_file():
        hot = pd.read_csv(args.hot_cross)
        recs.append(_block("hot_cross_1d", hot))
        hot_sp = splice_calendar_year(
            hot, l3, year=int(args.year), src_base="hot_cross", src_repl="l3_wait12"
        )
        recs.append(_block("synthetic_hot_cross_2022_L3", hot_sp))
    else:
        hot_sp = None

    outdir = Path(args.outdir) if args.outdir is not None else dated_outdir()
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "synthetic_1d_last15m_2022_l3.csv"
    spliced.to_csv(csv_path, index=False)
    if hot_sp is not None:
        hot_sp.to_csv(outdir / "synthetic_1d_hot_cross_2022_l3.csv", index=False)

    cmp = {
        "title": "Calendar splice (not a live switch)",
        "note": (
            "2022 trades from 15m L3 wait-12 signal-close; all other years from the "
            "last-15m 1d H2 book (15m N+1 mid sells). Occupancy not re-walked. "
            "No regime signal. Gross gain_pct (HTML default friction 0.25)."
        ),
        "rows": [_cmp_row(r, highlight=(r["book"] == "synthetic_last15m_2022_L3")) for r in recs],
    }
    cmp_path = outdir / "synthetic_1d_last15m_2022_l3_compare.json"
    cmp_path.write_text(json.dumps(cmp, indent=2), encoding="utf-8")

    lines = [
        "synthetic calendar splice: last-15m 1d H2 except %d = 15m L3 wait-12" % args.year,
        "base=%s n=%d (2022 n=%d)" % (args.base, len(base), _year_n(base, args.year)),
        "l3=%s n=%d (2022 n=%d)" % (args.l3, len(l3), _year_n(l3, args.year)),
        "spliced n=%d" % len(spliced),
        "occupancy=not_re-walked",
        "switch_signal=none (calendar year only)",
        "friction=gross in CSV; HTML default 0.25 (L3 keeper was 0.10)",
    ]
    for rec in recs:
        lines.append(
            "%s n=%s E=%s PF=%s geo=%s WR=%s 2022 n=%s E=%s PF=%s"
            % (
                rec["book"],
                rec["n"],
                rec["e"],
                rec["pf"],
                rec["geo"],
                rec["wr"],
                rec["n_2022"],
                rec["e_2022"],
                rec["pf_2022"],
            )
        )
        ydf = rec.get("years")
        if ydf is not None:
            lines.append(ydf.to_string(index=False))
    sum_path = outdir / "synthetic_1d_last15m_2022_l3_summary.txt"
    sum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("csv:", csv_path)
    print("summary:", sum_path)
    LOG.info("Wrote %s %s %s", csv_path, sum_path, cmp_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
