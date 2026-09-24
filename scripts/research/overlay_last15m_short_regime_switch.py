"""Causal last-15m H2 long vs 15m short H3+ regime switch.

Same pre-registered sensors as the L3 switch (WR/PF, TLT, UUP) plus SPY SMA.
When the long sensor is OFF, take short H3+ instead of sitting in cash / L3.

Occupancy not re-walked. Grade calendar 2022/2023 and geo-year PF. Do not
promote on pooled E alone (mixed payoff scales).

Usage (Windows CMD):
  venv\\Scripts\\activate
  set PYTHONPATH=.
  python scripts\\research\\overlay_last15m_short_regime_switch.py
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

from backtest_channel_touch_trades import (  # noqa: E402
    YEAR_BUCKETS,
    summarize_by_year,
)
from utils.research.cta_trend import load_macro_closes  # noqa: E402
from utils.research.h2_l3_blend_occupancy import SLEEVE_SHORT  # noqa: E402
from utils.research.last_15m_regime_switch import (  # noqa: E402
    attach_h2_regime,
    attach_macro_asof,
    combine_h2_alt,
    keep_unless_spy_below_sma,
    keep_unless_tlt_below_sma,
    keep_unless_uup_above_sma,
    keep_when_spy_below_sma,
    keep_when_tlt_below_sma,
    keep_when_uup_above_sma,
    rolling_h2_regime,
)
from utils.research.last_15m_volume_geometry import (  # noqa: E402
    book_stats,
    geom_mean_year_pf,
    winner_cut_skip,
)
from utils.research.report_paths import dated_outdir  # noqa: E402

LOG = logging.getLogger("overlay_last15m_short_regime_switch")

DEFAULT_H2 = ROOT / "reports" / "ascending_channels" / "channel_touch_1d_last_15m.csv"
DEFAULT_SHORT = (
    ROOT
    / "reports"
    / "ascending_channels"
    / "2026-09-16"
    / "channel_touch_resist_short_unique_20260916_211434.csv"
)

WINNER_WINDOWS: Tuple[Tuple[str, str, str], ...] = (
    ("2020-2021", "2020-01-01", "2021-12-31"),
    ("2024-2026", "2024-01-01", "2026-12-31"),
)


def _in_range(df: pd.DataFrame, start: str, end: str) -> pd.Series:
    d = pd.to_datetime(df["buy_date"], errors="coerce")
    return (d >= start) & (d <= end)


def _ensure_gain(df: pd.DataFrame, *, friction: float) -> pd.DataFrame:
    out = df.copy()
    if friction and friction > 0:
        g = pd.to_numeric(out["gain_pct"], errors="coerce")
        out["gain_pct"] = g - float(friction)
    else:
        out["gain_pct"] = pd.to_numeric(out["gain_pct"], errors="coerce")
    return out


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
    n_long = int((df["sleeve"] == "last15m_h2").sum()) if "sleeve" in df.columns and not df.empty else (
        int(len(df)) if not df.empty else 0
    )
    n_short = int((df["sleeve"] == SLEEVE_SHORT).sum()) if "sleeve" in df.columns and not df.empty else 0
    print("---- %s ----" % label)
    print(
        "n=%s long=%s short=%s E=%s PF=%s geo=%s | "
        "2022 n=%s E=%s PF=%s | 2023 n=%s E=%s PF=%s | 2022-23 E=%s PF=%s"
        % (
            s.get("n"),
            n_long,
            n_short,
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
    )
    if not years.empty:
        print(years.to_string(index=False))
    if "sleeve" in df.columns and not df.empty:
        print(df["sleeve"].value_counts().to_string())
    return {
        "label": label,
        "n": s.get("n"),
        "n_long": n_long,
        "n_short": n_short,
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
        rec["cut_%s_win$" % name] = cut["winner_drop_gain_sum"]
        rec["cut_%s_lose$" % name] = cut["loser_drop_gain_sum"]
        rec["cut_%s_blunt" % name] = cut["blunt"]
        print(
            "  winner-cut %s dropped=%s win$=%s lose$=%s blunt=%s"
            % (
                name,
                cut["n_dropped"],
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
    part["sleeve"] = "last15m_h2"
    rec = _print_block(label, part)
    cut = winner_cut_skip(baseline_gain, keep)
    rec["cut_all_blunt"] = cut["blunt"]
    print(
        "  winner-cut ALL dropped=%s win$=%s lose$=%s blunt=%s"
        % (cut["n_dropped"], cut["winner_drop_gain_sum"], cut["loser_drop_gain_sum"], cut["blunt"])
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


def _score_switch_rolling(
    label: str,
    h2: pd.DataFrame,
    short: pd.DataFrame,
    h2_on: pd.Series,
    regime: pd.DataFrame,
    rows: List[dict],
) -> None:
    short_on = ~attach_h2_regime(short, regime)
    switched = combine_h2_alt(h2, short, h2_on, short_on, alt_sleeve=SLEEVE_SHORT)
    rec = _print_block(label, switched)
    rows.append(rec)


def _score_switch_macro(
    label: str,
    h2: pd.DataFrame,
    short: pd.DataFrame,
    h2_on: pd.Series,
    short_on: pd.Series,
    rows: List[dict],
) -> None:
    switched = combine_h2_alt(h2, short, h2_on, short_on, alt_sleeve=SLEEVE_SHORT)
    rec = _print_block(label, switched)
    rows.append(rec)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--h2", type=Path, default=DEFAULT_H2)
    ap.add_argument("--short", type=Path, default=DEFAULT_SHORT)
    ap.add_argument("--long-friction", type=float, default=0.0)
    ap.add_argument("--short-friction", type=float, default=0.10)
    ap.add_argument("--outdir", type=Path, default=None)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not args.h2.exists() or not args.short.exists():
        LOG.error("Missing trades: h2=%s short=%s", args.h2, args.short)
        return 1

    h2 = _ensure_gain(pd.read_csv(args.h2), friction=float(args.long_friction))
    short = _ensure_gain(pd.read_csv(args.short), friction=float(args.short_friction))
    h2["stock"] = h2["stock"].astype(str).str.upper()
    short["stock"] = short["stock"].astype(str).str.upper()
    baseline_gain = pd.to_numeric(h2["gain_pct"], errors="coerce")
    rows: List[dict] = []

    h2_base = h2.copy()
    h2_base["sleeve"] = "last15m_h2"
    short_base = short.copy()
    short_base["sleeve"] = SLEEVE_SHORT
    rows.append(_print_block("baseline_last15m", h2_base))
    rows.append(_print_block("baseline_short_all", short_base))

    # --- Rolling paper H2 health ---
    wr20 = rolling_h2_regime(h2, window=20, off_wr=0.28, on_wr=0.32, mode="wr")
    wr30 = rolling_h2_regime(h2, window=30, off_wr=0.28, on_wr=0.32, mode="wr")
    pf20 = rolling_h2_regime(h2, window=20, mode="pf", pf_off=1.0, pf_on=1.2)
    h2_on_wr20 = attach_h2_regime(h2, wr20)
    h2_on_wr30 = attach_h2_regime(h2, wr30)
    h2_on_pf20 = attach_h2_regime(h2, pf20)
    print("WR20 events n=%d off_frac=%.3f" % (len(wr20), 1.0 - float(wr20["h2_on"].mean())))
    _score_h2_keep("H2_WR20_skip", h2, h2_on_wr20, baseline_gain, rows)
    _regime_span(h2, h2_on_wr20)
    _score_switch_rolling("SWITCH_WR20_SHORT", h2, short, h2_on_wr20, wr20, rows)
    _score_h2_keep("H2_WR30_skip", h2, h2_on_wr30, baseline_gain, rows)
    _regime_span(h2, h2_on_wr30)
    _score_switch_rolling("SWITCH_WR30_SHORT", h2, short, h2_on_wr30, wr30, rows)
    _score_h2_keep("H2_PF20_skip", h2, h2_on_pf20, baseline_gain, rows)
    _regime_span(h2, h2_on_pf20)
    _score_switch_rolling("SWITCH_PF20_SHORT", h2, short, h2_on_pf20, pf20, rows)

    # --- Macro ---
    macros = load_macro_closes(
        ["TLT", "UUP", "SPY"],
        start=datetime(2017, 1, 1),
        end=datetime(2026, 9, 19),
        preferred_provider="IB",
        fallback_provider="ALPACA",
    )
    tlt = macros.get("TLT")
    uup = macros.get("UUP")
    spy = macros.get("SPY")
    if tlt is None or tlt.empty or uup is None or uup.empty or spy is None or spy.empty:
        LOG.error("Need TLT, UUP, and SPY daily closes")
        return 1
    LOG.info(
        "TLT n=%d | UUP n=%d | SPY n=%d",
        len(tlt),
        len(uup),
        len(spy),
    )

    h2_tlt100 = attach_macro_asof(h2, tlt, prefix="tlt", sma_span=100)
    short_tlt100 = attach_macro_asof(short, tlt, prefix="tlt", sma_span=100)
    tlt100_keep = keep_unless_tlt_below_sma(h2_tlt100)
    _score_h2_keep("H2_TLT_lt_SMA100_skip", h2, tlt100_keep, baseline_gain, rows)
    _regime_span(h2, tlt100_keep)
    _score_switch_macro(
        "SWITCH_TLT100_SHORT",
        h2,
        short,
        tlt100_keep,
        keep_when_tlt_below_sma(short_tlt100),
        rows,
    )

    h2_tlt200 = attach_macro_asof(h2, tlt, prefix="tlt", sma_span=200)
    short_tlt200 = attach_macro_asof(short, tlt, prefix="tlt", sma_span=200)
    tlt200_keep = keep_unless_tlt_below_sma(h2_tlt200)
    _score_h2_keep("H2_TLT_lt_SMA200_skip", h2, tlt200_keep, baseline_gain, rows)
    _score_switch_macro(
        "SWITCH_TLT200_SHORT",
        h2,
        short,
        tlt200_keep,
        keep_when_tlt_below_sma(short_tlt200),
        rows,
    )

    h2_uup50 = attach_macro_asof(h2, uup, prefix="uup", sma_span=50)
    short_uup50 = attach_macro_asof(short, uup, prefix="uup", sma_span=50)
    uup50_keep = keep_unless_uup_above_sma(h2_uup50)
    _score_h2_keep("H2_UUP_gt_SMA50_skip", h2, uup50_keep, baseline_gain, rows)
    _regime_span(h2, uup50_keep)
    _score_switch_macro(
        "SWITCH_UUP50_SHORT",
        h2,
        short,
        uup50_keep,
        keep_when_uup_above_sma(short_uup50),
        rows,
    )

    h2_uup100 = attach_macro_asof(h2, uup, prefix="uup", sma_span=100)
    short_uup100 = attach_macro_asof(short, uup, prefix="uup", sma_span=100)
    uup100_keep = keep_unless_uup_above_sma(h2_uup100)
    _score_h2_keep("H2_UUP_gt_SMA100_skip", h2, uup100_keep, baseline_gain, rows)
    _score_switch_macro(
        "SWITCH_UUP100_SHORT",
        h2,
        short,
        uup100_keep,
        keep_when_uup_above_sma(short_uup100),
        rows,
    )

    h2_spy200 = attach_macro_asof(h2, spy, prefix="spy", sma_span=200)
    short_spy200 = attach_macro_asof(short, spy, prefix="spy", sma_span=200)
    spy200_keep = keep_unless_spy_below_sma(h2_spy200)
    _score_h2_keep("H2_SPY_lt_SMA200_skip", h2, spy200_keep, baseline_gain, rows)
    _regime_span(h2, spy200_keep)
    _score_switch_macro(
        "SWITCH_SPY200_SHORT",
        h2,
        short,
        spy200_keep,
        keep_when_spy_below_sma(short_spy200),
        rows,
    )

    h2_spy100 = attach_macro_asof(h2, spy, prefix="spy", sma_span=100)
    short_spy100 = attach_macro_asof(short, spy, prefix="spy", sma_span=100)
    spy100_keep = keep_unless_spy_below_sma(h2_spy100)
    _score_h2_keep("H2_SPY_lt_SMA100_skip", h2, spy100_keep, baseline_gain, rows)
    _regime_span(h2, spy100_keep)
    _score_switch_macro(
        "SWITCH_SPY100_SHORT",
        h2,
        short,
        spy100_keep,
        keep_when_spy_below_sma(short_spy100),
        rows,
    )

    outdir = args.outdir or dated_outdir(ROOT / "reports" / "ascending_channels")
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "last15m_short_regime_switch_summary.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    LOG.info("Wrote %s", path)
    print("Wrote %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
