"""Occupancy walk: current_best last-15m H2 long + 15m short H3+ resist fade.

Portfolio overlay (not same-channel long-then-short). Unified FCFS vs dedicated
long/short sleeves. Unique-symbol single account unless --independent.

Equal-dollar per taken fill; H2 avg wins ~9% vs short ~1.6% so pooled E is
mixed-scale — do not promote on pooled E alone. Grade calendar 2022/2023 and
geo-year PF. Number to beat: last-15m alone n=2297 E +0.27 PF 1.09.

Friction: long research path often fric0.25; short research default fric0.10.
Report research frictions, then one equalized fric0.25 stress.

Usage (Windows CMD):
  venv\\Scripts\\activate
  set PYTHONPATH=.
  python scripts\\research\\overlay_last15m_resist_short.py
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

from backtest_channel_touch_trades import (  # noqa: E402
    YEAR_BUCKETS,
    apply_friction,
    summarize_by_year,
)
from utils.research.h2_l3_blend_occupancy import (  # noqa: E402
    SLEEVE_H2,
    SLEEVE_SHORT,
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

LOG = logging.getLogger("overlay_last15m_resist_short")

DEFAULT_LONG = ROOT / "reports" / "ascending_channels" / "channel_touch_1d_last_15m.csv"
DEFAULT_SHORT = (
    ROOT
    / "reports"
    / "ascending_channels"
    / "2026-09-16"
    / "channel_touch_resist_short_unique_20260916_211434.csv"
)

UNIFIED_CAPS = (8, 15, 20)
DEDICATED: Tuple[Tuple[int, int], ...] = ((15, 5), (12, 3), (8, 4))


def _in_range(df: pd.DataFrame, start: str, end: str) -> pd.Series:
    d = pd.to_datetime(df["buy_date"], errors="coerce")
    return (d >= start) & (d <= end)


def _ensure_gain(df: pd.DataFrame, *, friction: float, label: str) -> pd.DataFrame:
    """Return a copy with gain_pct set to research-friction net (or gross if fric=0)."""
    out = df.copy()
    if friction and friction > 0:
        if "gain_pct_net" in out.columns and "gain_pct" in out.columns:
            # Recompute from gross so we do not double-apply.
            g = pd.to_numeric(out["gain_pct"], errors="coerce")
            out["gain_pct"] = g - float(friction)
        elif "gain_pct" in out.columns:
            out = apply_friction(out, float(friction))
            out["gain_pct"] = pd.to_numeric(out["gain_pct_net"], errors="coerce")
        else:
            raise ValueError("%s missing gain_pct" % label)
    else:
        if "gain_pct" not in out.columns:
            raise ValueError("%s missing gain_pct" % label)
        out["gain_pct"] = pd.to_numeric(out["gain_pct"], errors="coerce")
    return out


def _print_block(
    label: str,
    df: pd.DataFrame,
    skip: Optional[pd.DataFrame] = None,
) -> dict:
    g = pd.to_numeric(df["_gain"] if "_gain" in df.columns else df.get("gain_pct"), errors="coerce")
    if df.empty:
        g = pd.Series(dtype=float)
    # summarize_by_year expects gain_pct column
    work = df.copy()
    if not work.empty:
        if "_gain" in work.columns:
            work["gain_pct"] = pd.to_numeric(work["_gain"], errors="coerce")
        years = summarize_by_year(work, gain_col="gain_pct", buckets=YEAR_BUCKETS)
    else:
        years = pd.DataFrame()
    s = book_stats(g)
    geo = geom_mean_year_pf(years)
    y22 = work.loc[_in_range(work, "2022-01-01", "2022-12-31")] if not work.empty else work
    y23 = work.loc[_in_range(work, "2023-01-01", "2023-12-31")] if not work.empty else work
    both = work.loc[_in_range(work, "2022-01-01", "2023-12-31")] if not work.empty else work
    sleeves = sleeve_counts(work, sleeve_a=SLEEVE_H2, sleeve_b=SLEEVE_SHORT)
    y22s = year_sleeve_counts(work, 2022, sleeve_a=SLEEVE_H2, sleeve_b=SLEEVE_SHORT)
    y23s = year_sleeve_counts(work, 2023, sleeve_a=SLEEVE_H2, sleeve_b=SLEEVE_SHORT)
    skips = skip_counts(skip) if skip is not None else {}

    def _yr(part: pd.DataFrame) -> dict:
        gg = pd.to_numeric(part["gain_pct"], errors="coerce") if not part.empty else pd.Series(dtype=float)
        return book_stats(gg)

    s22 = _yr(y22)
    s23 = _yr(y23)
    sboth = _yr(both)
    rec = {
        "label": label,
        "n": s.get("n"),
        "expectancy_pct": s.get("expectancy_pct"),
        "profit_factor": s.get("profit_factor"),
        "geo_year_pf": geo,
        "n_long": sleeves.get(SLEEVE_H2, 0),
        "n_short": sleeves.get(SLEEVE_SHORT, 0),
        "n_2022": int(len(y22)),
        "e_2022": s22.get("expectancy_pct"),
        "pf_2022": s22.get("profit_factor"),
        "n_long_2022": y22s.get(SLEEVE_H2, 0),
        "n_short_2022": y22s.get(SLEEVE_SHORT, 0),
        "n_2023": int(len(y23)),
        "e_2023": s23.get("expectancy_pct"),
        "pf_2023": s23.get("profit_factor"),
        "n_long_2023": y23s.get(SLEEVE_H2, 0),
        "n_short_2023": y23s.get(SLEEVE_SHORT, 0),
        "e_2022_23": sboth.get("expectancy_pct"),
        "pf_2022_23": sboth.get("profit_factor"),
    }
    rec.update(skips)
    print("---- %s ----" % label)
    print(
        "n=%s long=%s short=%s E=%s PF=%s geo=%s | "
        "2022 n=%s L=%s S=%s E=%s PF=%s | "
        "2023 n=%s L=%s S=%s E=%s PF=%s | "
        "2022-23 E=%s PF=%s"
        % (
            rec["n"],
            rec["n_long"],
            rec["n_short"],
            rec["expectancy_pct"],
            rec["profit_factor"],
            None if geo is None else round(geo, 3),
            rec["n_2022"],
            rec["n_long_2022"],
            rec["n_short_2022"],
            rec["e_2022"],
            rec["pf_2022"],
            rec["n_2023"],
            rec["n_long_2023"],
            rec["n_short_2023"],
            rec["e_2023"],
            rec["pf_2023"],
            rec["e_2022_23"],
            rec["pf_2022_23"],
        )
    )
    if skips:
        print("  skips %s" % skips)
    if not years.empty:
        print(years.to_string(index=False))
    return rec


def _winner_vs_long(long_df: pd.DataFrame, kept: pd.DataFrame, rec: dict) -> None:
    """Long-sleeve winner-$ vs unconstrained long (short fills ignored)."""
    long_keep = pd.Series(False, index=long_df.index)
    if kept is None or kept.empty:
        cut = winner_cut_skip(pd.to_numeric(long_df["gain_pct"], errors="coerce"), long_keep)
    else:
        taken = kept.loc[kept["sleeve"] == SLEEVE_H2]
        keys = set(zip(taken["_stock"].astype(str), taken["_buy"].astype("int64")))
        buy = pd.to_datetime(
            long_df["buy_time"] if "buy_time" in long_df.columns else long_df["buy_date"],
            errors="coerce",
        )
        if getattr(buy.dt, "tz", None) is not None:
            buy = buy.dt.tz_convert("UTC").dt.tz_localize(None)
        stock = long_df["stock"].astype(str).str.upper()
        long_keep = pd.Series(
            [(str(s), int(t.value) if t == t else -1) in keys for s, t in zip(stock, buy)],
            index=long_df.index,
        )
        cut = winner_cut_skip(pd.to_numeric(long_df["gain_pct"], errors="coerce"), long_keep)
    rec["long_cut_n"] = cut["n_dropped"]
    rec["long_cut_win$"] = cut["winner_drop_gain_sum"]
    rec["long_cut_lose$"] = cut["loser_drop_gain_sum"]
    rec["long_cut_blunt"] = cut["blunt"]
    print(
        "  long winner-cut dropped=%s win$=%s lose$=%s blunt=%s"
        % (cut["n_dropped"], cut["winner_drop_gain_sum"], cut["loser_drop_gain_sum"], cut["blunt"])
    )


def _run_suite(
    long_df: pd.DataFrame,
    short_df: pd.DataFrame,
    *,
    tag: str,
) -> List[dict]:
    blend = prepare_blend(
        long_df,
        short_df,
        sleeve_a=SLEEVE_H2,
        sleeve_b=SLEEVE_SHORT,
    )
    rows: List[dict] = []
    rows.append(
        _print_block(
            "%s_long_unconstrained" % tag,
            prepare_blend(long_df, long_df.iloc[0:0], sleeve_a=SLEEVE_H2, sleeve_b=SLEEVE_SHORT),
        )
    )
    rows.append(
        _print_block(
            "%s_short_unconstrained" % tag,
            prepare_blend(long_df.iloc[0:0], short_df, sleeve_a=SLEEVE_H2, sleeve_b=SLEEVE_SHORT),
        )
    )
    rows.append(_print_block("%s_overlay_concat_no_occ" % tag, blend))

    uniq, skip_u = occupancy_unified_fcfs(
        blend, max_open=0, unique_symbol=True, sleeve_a=SLEEVE_H2, sleeve_b=SLEEVE_SHORT
    )
    rec = _print_block("%s_overlay_unique_symbol_nocap" % tag, uniq, skip_u)
    _winner_vs_long(long_df, uniq, rec)
    rows.append(rec)

    for cap in UNIFIED_CAPS:
        kept, skip = occupancy_unified_fcfs(
            blend, max_open=cap, unique_symbol=True, sleeve_a=SLEEVE_H2, sleeve_b=SLEEVE_SHORT
        )
        rec = _print_block("%s_UNIFIED_FCFS_%d" % (tag, cap), kept, skip)
        _winner_vs_long(long_df, kept, rec)
        rows.append(rec)

    for long_cap, short_cap in DEDICATED:
        kept, skip = occupancy_dedicated_sleeves(
            blend,
            a_cap=long_cap,
            b_cap=short_cap,
            unique_across=True,
            sleeve_a=SLEEVE_H2,
            sleeve_b=SLEEVE_SHORT,
        )
        rec = _print_block(
            "%s_DEDICATED_%d_%d_unique" % (tag, long_cap, short_cap), kept, skip
        )
        _winner_vs_long(long_df, kept, rec)
        rows.append(rec)
        kept_i, skip_i = occupancy_dedicated_sleeves(
            blend,
            a_cap=long_cap,
            b_cap=short_cap,
            unique_across=False,
            sleeve_a=SLEEVE_H2,
            sleeve_b=SLEEVE_SHORT,
        )
        rec_i = _print_block(
            "%s_DEDICATED_%d_%d_indep" % (tag, long_cap, short_cap), kept_i, skip_i
        )
        _winner_vs_long(long_df, kept_i, rec_i)
        rows.append(rec_i)
    return rows


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--long", type=Path, default=DEFAULT_LONG)
    ap.add_argument("--short", type=Path, default=DEFAULT_SHORT)
    ap.add_argument("--long-friction", type=float, default=0.0, help="Round-trip %% on long (default 0=CSV as-is)")
    ap.add_argument("--short-friction", type=float, default=0.10, help="Round-trip %% on short (research default)")
    ap.add_argument("--equal-friction", type=float, default=0.25, help="Equalized stress friction for both")
    ap.add_argument("--outdir", type=Path, default=None)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not args.long.exists():
        LOG.error("Long trades missing: %s", args.long)
        return 1
    if not args.short.exists():
        LOG.error("Short trades missing: %s", args.short)
        return 1

    long_raw = pd.read_csv(args.long)
    short_raw = pd.read_csv(args.short)
    long_raw["stock"] = long_raw["stock"].astype(str).str.upper()
    short_raw["stock"] = short_raw["stock"].astype(str).str.upper()
    LOG.info("Loaded long n=%d from %s", len(long_raw), args.long)
    LOG.info("Loaded short n=%d from %s", len(short_raw), args.short)

    long_res = _ensure_gain(long_raw, friction=float(args.long_friction), label="long")
    short_res = _ensure_gain(short_raw, friction=float(args.short_friction), label="short")
    print(
        "=== research frictions: long=%.2f%% short=%.2f%% ==="
        % (float(args.long_friction), float(args.short_friction))
    )
    rows = _run_suite(long_res, short_res, tag="res")

    long_eq = _ensure_gain(long_raw, friction=float(args.equal_friction), label="long")
    short_eq = _ensure_gain(short_raw, friction=float(args.equal_friction), label="short")
    print("=== equalized friction %.2f%% ===" % float(args.equal_friction))
    rows.extend(_run_suite(long_eq, short_eq, tag="eq025"))

    outdir = args.outdir or dated_outdir(ROOT / "reports" / "ascending_channels")
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "last15m_resist_short_blend_occupancy_summary.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    LOG.info("Wrote %s", path)
    print("Wrote %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
