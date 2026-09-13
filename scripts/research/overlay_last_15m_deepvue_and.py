"""AND last-15m H2 fills with Deepvue weekend focus (same-list, no occupancy re-walk).

Pre-registered:

  L15_and_week         last-15m fill in the week after a same-name Deepvue take-4
  L15_and_signal_day   last-15m fill on the Friday signal session itself
  L15_ever_focused     last-15m name ever appeared on the Deepvue take-4 (universe)
  DV_and_l15_week      Deepvue occupancy trade whose name also printed last-15m that week

Grade vs last-15m n=2297 E +0.27 PF 1.09: 2022 itself non-red, geo-PF lift
without clipping 2020-21 / 2024-26 winner dollars. Do not grade vs clip books.

Usage (Windows CMD):
  venv\\Scripts\\activate
  set PYTHONPATH=.
  python scripts\\research\\overlay_last_15m_deepvue_and.py
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

from overlay_last_15m_hmm_vix_skip import (  # noqa: E402
    DEFAULT_TRADES,
    _print_block,
    _score_keep,
)
from utils.research.deepvue_l15_and import (  # noqa: E402
    keep_deepvue_if_last15m_trade_week,
    keep_last15m_ever_focused,
    keep_last15m_in_focus_trade_week,
    keep_last15m_on_signal_day,
    overlap_counts,
    pair_rows,
)
from utils.research.report_paths import dated_outdir  # noqa: E402

LOG = logging.getLogger("overlay_last_15m_deepvue")

DEFAULT_SIGNALS = (
    ROOT / "reports" / "weekly_deepvue" / "2026-09-12" / "weekly_deepvue_signals.csv"
)
DEFAULT_DEEPVUE = (
    ROOT / "reports" / "weekly_deepvue" / "2026-09-12" / "weekly_deepvue_trades.csv"
)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trades", type=Path, default=DEFAULT_TRADES)
    ap.add_argument("--signals", type=Path, default=DEFAULT_SIGNALS)
    ap.add_argument("--deepvue", type=Path, default=DEFAULT_DEEPVUE)
    ap.add_argument("--outdir", type=Path, default=None)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    last15m = pd.read_csv(args.trades)
    last15m["stock"] = last15m["stock"].astype(str).str.upper()
    signals = pd.read_csv(args.signals)
    signals["stock"] = signals["stock"].astype(str).str.upper()
    deepvue = pd.read_csv(args.deepvue)
    deepvue["stock"] = deepvue["stock"].astype(str).str.upper()

    counts = overlap_counts(last15m, signals, deepvue)
    LOG.info("overlap %s", counts)
    print("---- overlap ----")
    for key, val in counts.items():
        print("%s=%s" % (key, val))

    rows = []
    baseline_gain = pd.to_numeric(last15m["gain_pct"], errors="coerce")
    rows.append(_print_block("baseline_last15m", last15m))
    _score_keep(
        "L15_and_week",
        last15m,
        keep_last15m_in_focus_trade_week(last15m, signals),
        baseline_gain,
        rows,
    )
    _score_keep(
        "L15_and_signal_day",
        last15m,
        keep_last15m_on_signal_day(last15m, signals),
        baseline_gain,
        rows,
    )
    _score_keep(
        "L15_ever_focused",
        last15m,
        keep_last15m_ever_focused(last15m, signals),
        baseline_gain,
        rows,
    )

    dv_gain = pd.to_numeric(deepvue["gain_pct"], errors="coerce")
    rows.append(_print_block("baseline_deepvue", deepvue))
    _score_keep(
        "DV_and_l15_week",
        deepvue,
        keep_deepvue_if_last15m_trade_week(deepvue, last15m),
        dv_gain,
        rows,
    )

    outdir = Path(args.outdir) if args.outdir is not None else dated_outdir()
    outdir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    for key, val in counts.items():
        summary[key] = val
    sum_csv = outdir / "last_15m_deepvue_and_summary.csv"
    summary.to_csv(sum_csv, index=False)
    week_pairs = pair_rows(last15m, signals, mode="trade_week")
    day_pairs = pair_rows(last15m, signals, mode="signal_day")
    if not week_pairs.empty:
        week_pairs.to_csv(outdir / "last_15m_deepvue_and_week_pairs.csv", index=False)
    if not day_pairs.empty:
        day_pairs.to_csv(outdir / "last_15m_deepvue_and_signal_day_pairs.csv", index=False)

    lines = [
        "last-15m AND Deepvue weekend focus (same-list, occupancy not re-walked)",
        "overlap %s" % counts,
    ]
    for rec in rows:
        lines.append(
            "%s n=%s E=%s PF=%s geo=%s skipped=%s 2022 n=%s E=%s PF=%s 2023 n=%s E=%s PF=%s 2022-23 E=%s PF=%s blunt=%s"
            % (
                rec.get("label"),
                rec.get("n"),
                rec.get("expectancy_pct"),
                rec.get("profit_factor"),
                rec.get("geo_year_pf"),
                rec.get("n_skipped"),
                rec.get("n_2022"),
                rec.get("e_2022"),
                rec.get("pf_2022"),
                rec.get("n_2023"),
                rec.get("e_2023"),
                rec.get("pf_2023"),
                rec.get("e_2022_23"),
                rec.get("pf_2022_23"),
                rec.get("cut_blunt"),
            )
        )
    txt = outdir / "last_15m_deepvue_and_summary.txt"
    txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("out:", outdir)
    LOG.info("Wrote %s %s", sum_csv, txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
