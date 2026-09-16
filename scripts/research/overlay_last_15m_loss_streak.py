"""Causal consecutive-loss sit-out on last-15m H2 (locked params).

Diagnose streak clustering by YEAR_BUCKETS, then score pre-registered rules
only (no full-sample grid). Occupancy not re-walked.

Locked:
  count L=5 S=3 P=2 (primary)
  count L=5 S=3 P=3
  dollar sum<=-15 S=3 P=2
Exploratory (report-only, not promotion): L in {4,5,6} S=3 P=2

Usage (Windows CMD):
  venv\\Scripts\\activate
  set PYTHONPATH=.
  python scripts\\research\\overlay_last_15m_loss_streak.py
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
from utils.research.last_15m_loss_streak import (  # noqa: E402
    apply_loss_streak_mask,
    baseline_loss_streak_diag,
    streak_longer_in_hostile,
)
from utils.research.last_15m_volume_geometry import (  # noqa: E402
    book_stats,
    geom_mean_year_pf,
    winner_cut_skip,
)
from utils.research.report_paths import dated_outdir  # noqa: E402

LOG = logging.getLogger("overlay_last_15m_loss_streak")

DEFAULT_H2 = (
    ROOT
    / "reports"
    / "ascending_channels"
    / "2026-09-11"
    / "channel_touch_h2_last_15m_open_mid_sell_15m_next_mid_span365.csv"
)

WINNER_WINDOWS: Tuple[Tuple[str, str, str], ...] = (
    ("2020-2021", "2020-01-01", "2021-12-31"),
    ("2024-2026", "2024-01-01", "2026-12-31"),
)


def _in_range(df: pd.DataFrame, start: str, end: str) -> pd.Series:
    d = pd.to_datetime(df["buy_date"], errors="coerce")
    return (d >= start) & (d <= end)


def _print_block(label: str, df: pd.DataFrame) -> dict:
    g = pd.to_numeric(df["gain_pct"], errors="coerce") if not df.empty else pd.Series(dtype=float)
    years = (
        summarize_by_year(df, gain_col="gain_pct", buckets=YEAR_BUCKETS)
        if not df.empty
        else pd.DataFrame()
    )
    s = book_stats(g)
    geo = geom_mean_year_pf(years)
    y22 = df.loc[_in_range(df, "2022-01-01", "2022-12-31")] if not df.empty else df
    y23 = df.loc[_in_range(df, "2023-01-01", "2023-12-31")] if not df.empty else df
    both = df.loc[_in_range(df, "2022-01-01", "2023-12-31")] if not df.empty else df
    hold = df.loc[_in_range(df, "2022-01-01", "2026-12-31")] if not df.empty else df
    pre = df.loc[_in_range(df, "2018-01-01", "2021-12-31")] if not df.empty else df
    s22 = book_stats(pd.to_numeric(y22["gain_pct"], errors="coerce"))
    s23 = book_stats(pd.to_numeric(y23["gain_pct"], errors="coerce"))
    s223 = book_stats(pd.to_numeric(both["gain_pct"], errors="coerce"))
    s_hold = book_stats(pd.to_numeric(hold["gain_pct"], errors="coerce"))
    s_pre = book_stats(pd.to_numeric(pre["gain_pct"], errors="coerce"))
    LOG.info(
        "%s n=%s E=%s PF=%s geo=%s | 2022 n=%s E=%s PF=%s | 2023 n=%s E=%s PF=%s | 2022-23 E=%s PF=%s",
        label,
        s.get("n"),
        s.get("expectancy_pct"),
        s.get("profit_factor"),
        None if geo is None else round(geo, 3),
        s22.get("n"),
        s22.get("expectancy_pct"),
        s22.get("profit_factor"),
        s23.get("n"),
        s23.get("expectancy_pct"),
        s23.get("profit_factor"),
        s223.get("expectancy_pct"),
        s223.get("profit_factor"),
    )
    print("---- %s ----" % label)
    if not years.empty:
        print(years.to_string(index=False))
    print(
        "  pre<=2021 n=%s E=%s PF=%s | holdout 2022-26 n=%s E=%s PF=%s"
        % (
            s_pre.get("n"),
            s_pre.get("expectancy_pct"),
            s_pre.get("profit_factor"),
            s_hold.get("n"),
            s_hold.get("expectancy_pct"),
            s_hold.get("profit_factor"),
        )
    )
    return {
        "label": label,
        "n": s.get("n"),
        "expectancy_pct": s.get("expectancy_pct"),
        "profit_factor": s.get("profit_factor"),
        "geo_year_pf": geo,
        "n_2022": s22.get("n"),
        "e_2022": s22.get("expectancy_pct"),
        "pf_2022": s22.get("profit_factor"),
        "n_2023": s23.get("n"),
        "e_2023": s23.get("expectancy_pct"),
        "pf_2023": s23.get("profit_factor"),
        "n_2022_23": s223.get("n"),
        "e_2022_23": s223.get("expectancy_pct"),
        "pf_2022_23": s223.get("profit_factor"),
        "n_pre_2021": s_pre.get("n"),
        "e_pre_2021": s_pre.get("expectancy_pct"),
        "pf_pre_2021": s_pre.get("profit_factor"),
        "n_holdout_2022_26": s_hold.get("n"),
        "e_holdout_2022_26": s_hold.get("expectancy_pct"),
        "pf_holdout_2022_26": s_hold.get("profit_factor"),
    }


def _winner_windows(trades: pd.DataFrame, keep: pd.Series, rec: dict) -> None:
    g = pd.to_numeric(trades["gain_pct"], errors="coerce")
    for name, start, end in WINNER_WINDOWS:
        m = _in_range(trades, start, end)
        cut = winner_cut_skip(g.loc[m], keep.loc[m])
        rec["cut_%s_winners" % name] = cut["winners_dropped"]
        rec["cut_%s_losers" % name] = cut["losers_dropped"]
        rec["cut_%s_win_sum" % name] = cut["winner_drop_gain_sum"]
        rec["cut_%s_lose_sum" % name] = cut["loser_drop_gain_sum"]
        rec["cut_%s_blunt" % name] = cut["blunt"]
        print(
            "  winner-cut %s dropped=%s winners=%s losers=%s win$=%s lose$=%s blunt=%s"
            % (
                name,
                cut["n_dropped"],
                cut["winners_dropped"],
                cut["losers_dropped"],
                cut["winner_drop_gain_sum"],
                cut["loser_drop_gain_sum"],
                cut["blunt"],
            )
        )


def _score_keep(
    label: str,
    trades: pd.DataFrame,
    keep: pd.Series,
    baseline_gain: pd.Series,
    rows: List[dict],
    *,
    diag: Optional[dict] = None,
    exploratory: bool = False,
) -> None:
    keep = keep.reindex(trades.index).fillna(True).astype(bool)
    part = trades.loc[keep].copy()
    rec = _print_block(label, part)
    rec["exploratory"] = bool(exploratory)
    cut = winner_cut_skip(baseline_gain, keep)
    rec.update({("cut_" + k): v for k, v in cut.items()})
    print(
        "  winner-cut ALL dropped=%s winners=%s losers=%s win$=%s lose$=%s blunt=%s"
        % (
            cut["n_dropped"],
            cut["winners_dropped"],
            cut["losers_dropped"],
            cut["winner_drop_gain_sum"],
            cut["loser_drop_gain_sum"],
            cut["blunt"],
        )
    )
    _winner_windows(trades, keep, rec)
    if diag:
        rec["n_sit_episodes"] = diag.get("n_sit_episodes")
        rec["skips_by_bucket"] = str(diag.get("skips_by_bucket"))
        print(
            "  sit_episodes=%s skips_by_bucket=%s"
            % (diag.get("n_sit_episodes"), diag.get("skips_by_bucket"))
        )
        d = pd.to_datetime(trades["buy_date"], errors="coerce")
        print(
            "  2022 skipped %d / taken %d | 2023 skipped %d / taken %d"
            % (
                int(((d.dt.year == 2022) & (~keep)).sum()),
                int(((d.dt.year == 2022) & keep).sum()),
                int(((d.dt.year == 2023) & (~keep)).sum()),
                int(((d.dt.year == 2023) & keep).sum()),
            )
        )
    rec["n_skipped"] = int((~keep).sum())
    rows.append(rec)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--h2", type=Path, default=DEFAULT_H2)
    ap.add_argument("--outdir", type=Path, default=None)
    ap.add_argument(
        "--skip-exploratory",
        action="store_true",
        help="Skip L in {4,5,6} sensitivity (report-only grid)",
    )
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    h2 = pd.read_csv(args.h2)
    h2["stock"] = h2["stock"].astype(str).str.upper()
    baseline_gain = pd.to_numeric(h2["gain_pct"], errors="coerce")
    rows: List[dict] = []

    print("==== DIAGNOSTIC: baseline consecutive-loss runs ====")
    diag_df = baseline_loss_streak_diag(h2)
    print(diag_df.to_string(index=False))
    longer = streak_longer_in_hostile(diag_df)
    print("streak_longer_in_hostile: %s" % longer)

    outdir = args.outdir or dated_outdir(ROOT / "reports" / "ascending_channels")
    outdir.mkdir(parents=True, exist_ok=True)
    diag_path = outdir / "last_15m_loss_streak_diag.csv"
    diag_df.to_csv(diag_path, index=False)
    LOG.info("Wrote %s", diag_path)

    rows.append(_print_block("baseline_last15m", h2))
    rows[-1]["longer_tailed_2022_23"] = longer.get("longer_tailed")
    rows[-1]["diag_hostile_max"] = longer.get("hostile_max")
    rows[-1]["diag_compare_max"] = longer.get("compare_max")

    if not longer.get("longer_tailed"):
        LOG.warning(
            "2022-23 is NOT longer-tailed in loss streaks vs 2020-21; "
            "filter is unlikely to be the slaughter sensor (still scoring locked rules)."
        )

    locked = [
        ("count_L5_S3_P2", dict(L=5, S=3, P=2, mode="count")),
        ("count_L5_S3_P3", dict(L=5, S=3, P=3, mode="count")),
        ("dollar_sum15_S3_P2", dict(L=5, S=3, P=2, mode="dollar", dollar_sum=-15.0)),
    ]
    print("==== LOCKED RULES ====")
    for label, kwargs in locked:
        keep, d = apply_loss_streak_mask(h2, **kwargs)
        _score_keep(label, h2, keep, baseline_gain, rows, diag=d, exploratory=False)

    if not args.skip_exploratory:
        print("==== EXPLORATORY (report-only; not for promotion) ====")
        for L in (4, 5, 6):
            label = "exploratory_L%d_S3_P2" % L
            keep, d = apply_loss_streak_mask(h2, L=L, S=3, P=2, mode="count")
            _score_keep(label, h2, keep, baseline_gain, rows, diag=d, exploratory=True)

    path = outdir / "last_15m_loss_streak_summary.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    LOG.info("Wrote %s", path)
    print("longer_tailed=%s -> promote only if locked rules also clear geo/2022/winner-$ gates" % longer.get("longer_tailed"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
