"""Causal last-15m H2 vs 15m L3 switch: rolling WR/PF, TLT, UUP.

Pre-registered (occupancy not re-walked). Paper H2 rolling stats include
skipped names. Macro joins are prior-session. Not a 2022 calendar splice.

Grade: 2022 itself non-red without clipping 2020-21 / 2024-26 winner dollars.

Usage (Windows CMD):
  venv\\Scripts\\activate
  set PYTHONPATH=.
  python scripts\\research\\overlay_last_15m_regime_switch.py
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

from backtest_channel_touch_trades import YEAR_BUCKETS, summarize_by_year  # noqa: E402
from utils.research.cta_trend import load_macro_closes  # noqa: E402
from utils.research.last_15m_regime_switch import (  # noqa: E402
    attach_h2_regime,
    attach_macro_asof,
    combine_h2_l3,
    keep_unless_tlt_below_sma,
    keep_unless_tlt_roc_neg,
    keep_unless_uup_above_sma,
    keep_unless_uup_roc_pos,
    rolling_h2_regime,
)
from utils.research.last_15m_volume_geometry import (  # noqa: E402
    book_stats,
    geom_mean_year_pf,
    winner_cut_skip,
)
from utils.research.report_paths import dated_outdir  # noqa: E402

LOG = logging.getLogger("overlay_last_15m_regime_switch")

DEFAULT_H2 = (
    ROOT
    / "reports"
    / "ascending_channels"
    / "2026-09-11"
    / "channel_touch_h2_last_15m_open_mid_sell_15m_next_mid_span365.csv"
)
DEFAULT_L3 = ROOT / "reports" / "ascending_channels" / "channel_touch_15m_l3_wait12.csv"

WINNER_WINDOWS: Tuple[Tuple[str, str, str], ...] = (
    ("2020-2021", "2020-01-01", "2021-12-31"),
    ("2024-2026", "2024-01-01", "2026-12-31"),
)


def _in_range(df: pd.DataFrame, start: str, end: str) -> pd.Series:
    d = pd.to_datetime(df["buy_date"], errors="coerce")
    return (d >= start) & (d <= end)


def _print_block(label: str, df: pd.DataFrame) -> dict:
    g = pd.to_numeric(df["gain_pct"], errors="coerce") if not df.empty else pd.Series(dtype=float)
    years = summarize_by_year(df, gain_col="gain_pct", buckets=YEAR_BUCKETS) if not df.empty else pd.DataFrame()
    s = book_stats(g)
    geo = geom_mean_year_pf(years)
    y22 = df.loc[_in_range(df, "2022-01-01", "2022-12-31")] if not df.empty else df
    y23 = df.loc[_in_range(df, "2023-01-01", "2023-12-31")] if not df.empty else df
    both = df.loc[_in_range(df, "2022-01-01", "2023-12-31")] if not df.empty else df
    s22 = book_stats(pd.to_numeric(y22["gain_pct"], errors="coerce"))
    s23 = book_stats(pd.to_numeric(y23["gain_pct"], errors="coerce"))
    s223 = book_stats(pd.to_numeric(both["gain_pct"], errors="coerce"))
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


def _score_h2_keep(
    label: str,
    trades: pd.DataFrame,
    keep: pd.Series,
    baseline_gain: pd.Series,
    rows: List[dict],
) -> pd.Series:
    keep = keep.reindex(trades.index).fillna(True).astype(bool)
    part = trades.loc[keep].copy()
    rec = _print_block(label, part)
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
    rec["n_skipped"] = int((~keep).sum())
    rows.append(rec)
    return keep


def _regime_span(h2: pd.DataFrame, keep: pd.Series) -> None:
    d = pd.to_datetime(h2["buy_date"], errors="coerce")
    off = h2.loc[~keep]
    if off.empty:
        print("  H2-off: never")
        return
    off_d = pd.to_datetime(off["buy_date"], errors="coerce")
    print(
        "  H2-off n=%d first=%s last=%s | 2022 skipped %d / taken %d"
        % (
            int((~keep).sum()),
            str(off_d.min().date()),
            str(off_d.max().date()),
            int(((d.dt.year == 2022) & (~keep)).sum()),
            int(((d.dt.year == 2022) & keep).sum()),
        )
    )


def _score_switch(
    label: str,
    h2: pd.DataFrame,
    l3: pd.DataFrame,
    h2_on: pd.Series,
    regime: pd.DataFrame,
    rows: List[dict],
) -> None:
    l3_on = ~attach_h2_regime(l3, regime)
    switched = combine_h2_l3(h2, l3, h2_on, l3_on)
    rec = _print_block(label, switched)
    if "sleeve" in switched.columns and not switched.empty:
        print(switched["sleeve"].value_counts().to_string())
    rec["n_h2_sleeve"] = int((switched["sleeve"] == "last15m_h2").sum()) if not switched.empty else 0
    rec["n_l3_sleeve"] = int((switched["sleeve"] == "l3_wait12").sum()) if not switched.empty else 0
    rows.append(rec)


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
    baseline_gain = pd.to_numeric(h2["gain_pct"], errors="coerce")
    rows: List[dict] = []
    rows.append(_print_block("baseline_last15m", h2))
    rows.append(_print_block("baseline_l3_all", l3))

    wr20 = rolling_h2_regime(h2, window=20, off_wr=0.28, on_wr=0.32, mode="wr")
    wr30 = rolling_h2_regime(h2, window=30, off_wr=0.28, on_wr=0.32, mode="wr")
    pf20 = rolling_h2_regime(h2, window=20, mode="pf", pf_off=1.0, pf_on=1.2)
    h2_on_wr20 = attach_h2_regime(h2, wr20)
    h2_on_wr30 = attach_h2_regime(h2, wr30)
    h2_on_pf20 = attach_h2_regime(h2, pf20)
    print("WR20 events n=%d off_frac=%.3f" % (len(wr20), 1.0 - float(wr20["h2_on"].mean())))
    _score_h2_keep("H2_WR20_skip", h2, h2_on_wr20, baseline_gain, rows)
    _regime_span(h2, h2_on_wr20)
    _score_switch("SWITCH_WR20_L3", h2, l3, h2_on_wr20, wr20, rows)
    _score_h2_keep("H2_WR30_skip", h2, h2_on_wr30, baseline_gain, rows)
    _regime_span(h2, h2_on_wr30)
    _score_switch("SWITCH_WR30_L3", h2, l3, h2_on_wr30, wr30, rows)
    _score_h2_keep("H2_PF20_skip", h2, h2_on_pf20, baseline_gain, rows)
    _regime_span(h2, h2_on_pf20)
    _score_switch("SWITCH_PF20_L3", h2, l3, h2_on_pf20, pf20, rows)

    macros = load_macro_closes(
        ["TLT", "UUP"],
        start=datetime(2017, 1, 1),
        end=datetime(2026, 9, 12),
        preferred_provider="IB",
        fallback_provider="ALPACA",
    )
    tlt = macros.get("TLT")
    uup = macros.get("UUP")
    if tlt is None or tlt.empty or uup is None or uup.empty:
        LOG.error("Need TLT and UUP daily closes")
        return 1
    LOG.info("TLT n=%d %s->%s | UUP n=%d %s->%s", len(tlt), tlt.index.min().date(), tlt.index.max().date(), len(uup), uup.index.min().date(), uup.index.max().date())
    h2_tlt100 = attach_macro_asof(h2, tlt, prefix="tlt", sma_span=100)
    h2_tlt200 = attach_macro_asof(h2, tlt, prefix="tlt", sma_span=200)
    h2_uup50 = attach_macro_asof(h2, uup, prefix="uup", sma_span=50)
    h2_uup100 = attach_macro_asof(h2, uup, prefix="uup", sma_span=100)
    l3_tlt100 = attach_macro_asof(l3, tlt, prefix="tlt", sma_span=100)
    l3_uup50 = attach_macro_asof(l3, uup, prefix="uup", sma_span=50)

    tlt100_keep = keep_unless_tlt_below_sma(h2_tlt100)
    tlt200_keep = keep_unless_tlt_below_sma(h2_tlt200)
    tlt_roc_keep = keep_unless_tlt_roc_neg(h2_tlt100)
    uup50_keep = keep_unless_uup_above_sma(h2_uup50)
    uup100_keep = keep_unless_uup_above_sma(h2_uup100)
    uup_roc_keep = keep_unless_uup_roc_pos(h2_uup50)

    _score_h2_keep("H2_TLT_lt_SMA100_skip", h2, tlt100_keep, baseline_gain, rows)
    _regime_span(h2, tlt100_keep)
    _score_switch_macro("SWITCH_TLT100_L3", h2, l3, tlt100_keep, keep_unless_tlt_below_sma(l3_tlt100), rows)
    _score_h2_keep("H2_TLT_lt_SMA200_skip", h2, tlt200_keep, baseline_gain, rows)
    _score_h2_keep("H2_TLT_ROC63_neg_skip", h2, tlt_roc_keep, baseline_gain, rows)
    _score_h2_keep("H2_UUP_gt_SMA50_skip", h2, uup50_keep, baseline_gain, rows)
    _regime_span(h2, uup50_keep)
    _score_switch_macro("SWITCH_UUP50_L3", h2, l3, uup50_keep, keep_unless_uup_above_sma(l3_uup50), rows)
    _score_h2_keep("H2_UUP_gt_SMA100_skip", h2, uup100_keep, baseline_gain, rows)
    _score_h2_keep("H2_UUP_ROC63_pos_skip", h2, uup_roc_keep, baseline_gain, rows)

    outdir = args.outdir or dated_outdir(ROOT / "reports" / "ascending_channels")
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "last_15m_regime_switch_summary.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    LOG.info("Wrote %s", path)
    return 0


def _score_switch_macro(
    label: str,
    h2: pd.DataFrame,
    l3: pd.DataFrame,
    h2_keep: pd.Series,
    l3_h2_keep: pd.Series,
    rows: List[dict],
) -> None:
    l3_on = ~l3_h2_keep.reindex(l3.index).fillna(True)
    switched = combine_h2_l3(h2, l3, h2_keep, l3_on)
    rec = _print_block(label, switched)
    if "sleeve" in switched.columns and not switched.empty:
        print(switched["sleeve"].value_counts().to_string())
    rec["n_h2_sleeve"] = int((switched["sleeve"] == "last15m_h2").sum()) if not switched.empty else 0
    rec["n_l3_sleeve"] = int((switched["sleeve"] == "l3_wait12").sum()) if not switched.empty else 0
    rows.append(rec)


if __name__ == "__main__":
    raise SystemExit(main())
