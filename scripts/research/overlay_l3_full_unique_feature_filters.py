"""Feature-threshold A/B on full-IB unique L3 wait-12 CSV (no detector rescan).

Gates per variant vs baseline:
  - E/PF lift
  - geo-mean YEAR_BUCKETS PF (skip n<30; equal-weight, not n-weight)
  - win$keep >= 0.90 (winner-$ gate)
  - ADV top-300 overlap diagnostic (not a promote by itself)

Optional: --with-fundamentals joins SEC asof filed and adds fund filters.

Usage (Windows CMD):
  venv\\Scripts\\activate
  set PYTHONPATH=.
  python scripts\\research\\overlay_l3_full_unique_feature_filters.py
  python scripts\\research\\overlay_l3_full_unique_feature_filters.py --with-fundamentals
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.research.report_paths import dated_outdir  # noqa: E402

LOG = logging.getLogger("overlay_l3_full_unique_feature_filters")

DEFAULT_TRADES = (
    ROOT
    / "reports"
    / "ascending_channels"
    / "2026-10-10"
    / "channel_touch_15m_l3_wait12_full_unique.csv"
)
DEFAULT_KEEPER = (
    ROOT / "reports" / "ascending_channels" / "channel_touch_15m_l3_wait12.csv"
)

YEAR_BUCKETS: Sequence[Tuple[str, int, int]] = (
    ("2018-19", 0, 2019),
    ("2020-21", 2020, 2021),
    ("2022-23", 2022, 2023),
    ("2024-26", 2024, 2099),
)

WIN_KEEP_MIN = 0.90


def _stats(g: pd.Series) -> Dict[str, Any]:
    g = pd.to_numeric(g, errors="coerce").dropna()
    if g.empty:
        return {
            "n": 0,
            "E": None,
            "PF": None,
            "WR": None,
            "med": None,
            "win_sum": 0.0,
            "lose_sum": 0.0,
        }
    wins = float(g[g > 0].sum())
    losses = float((-g[g <= 0]).sum())
    pf = wins / losses if losses > 0 else float("inf")
    return {
        "n": int(len(g)),
        "E": round(float(g.mean()), 4),
        "PF": round(pf, 3) if np.isfinite(pf) else None,
        "WR": round(float((g > 0).mean() * 100.0), 2),
        "med": round(float(g.median()), 4),
        "win_sum": round(wins, 2),
        "lose_sum": round(losses, 2),
    }


def _drop_top3(g: pd.Series) -> Dict[str, Any]:
    g = pd.to_numeric(g, errors="coerce").dropna().to_numpy(dtype=float)
    if len(g) <= 3:
        return _stats(pd.Series(dtype=float))
    keep = np.ones(len(g), dtype=bool)
    keep[np.argsort(g)[::-1][:3]] = False
    return _stats(pd.Series(g[keep]))


def _year_buckets(df: pd.DataFrame, gain_col: str) -> List[Dict[str, Any]]:
    y = pd.to_datetime(df["buy_date"], errors="coerce").dt.year
    rows = []
    for label, lo, hi in YEAR_BUCKETS:
        m = (y >= lo) & (y <= hi)
        s = _stats(df.loc[m, gain_col])
        s["bucket"] = label
        rows.append(s)
    return rows


def _geo_mean_pf(years: List[Dict[str, Any]], *, min_n: int = 30) -> Optional[float]:
    pfs: List[float] = []
    for y in years:
        if int(y.get("n") or 0) < min_n:
            continue
        pf = y.get("PF")
        if pf is None or not np.isfinite(float(pf)) or float(pf) <= 0:
            return None
        pfs.append(float(pf))
    if not pfs:
        return None
    return float(math.exp(sum(math.log(p) for p in pfs) / len(pfs)))


def _keeper_symbols(path: Path) -> set:
    if not path.exists():
        return set()
    k = pd.read_csv(path, usecols=["stock"])
    return set(k["stock"].astype(str).str.upper())


def _adv_overlap(sub: pd.DataFrame, keeper_syms: set) -> Dict[str, Any]:
    if sub.empty:
        return {
            "pct_trades_in_keeper_syms": None,
            "pct_syms_in_keeper": None,
            "median_adv_20": None,
            "n_unique_syms": 0,
        }
    stocks = sub["stock"].astype(str).str.upper()
    in_k = stocks.isin(keeper_syms) if keeper_syms else pd.Series(False, index=sub.index)
    uniq = set(stocks.unique())
    med_adv = (
        float(pd.to_numeric(sub["adv_20"], errors="coerce").median())
        if "adv_20" in sub.columns
        else None
    )
    return {
        "pct_trades_in_keeper_syms": round(float(in_k.mean() * 100.0), 2),
        "pct_syms_in_keeper": (
            round(100.0 * len(uniq & keeper_syms) / max(1, len(uniq)), 2)
            if keeper_syms
            else None
        ),
        "median_adv_20": None if med_adv is None or not np.isfinite(med_adv) else round(med_adv, 1),
        "n_unique_syms": int(len(uniq)),
    }


def _mask_le(df: pd.DataFrame, col: str, thr: float) -> pd.Series:
    v = pd.to_numeric(df[col], errors="coerce")
    return v.notna() & (v <= float(thr))


def _mask_ge(df: pd.DataFrame, col: str, thr: float) -> pd.Series:
    v = pd.to_numeric(df[col], errors="coerce")
    return v.notna() & (v >= float(thr))


def _mask_eq(df: pd.DataFrame, col: str, thr: float) -> pd.Series:
    v = pd.to_numeric(df[col], errors="coerce")
    return v.notna() & (v == float(thr))


def build_price_variants(df: pd.DataFrame) -> List[Tuple[str, pd.Series]]:
    """Seed A/Bs from 2026-10-10 full-unique univariate quintiles."""
    variants: List[Tuple[str, pd.Series]] = [
        ("baseline", pd.Series(True, index=df.index)),
    ]
    specs: List[Tuple[str, Callable[[], pd.Series]]] = [
        ("atr_le_0.733", lambda: _mask_le(df, "atr_pct", 0.733)),
        ("atr_le_0.515", lambda: _mask_le(df, "atr_pct", 0.515)),
        ("range_le_0.766", lambda: _mask_le(df, "range_pct", 0.766)),
        ("range_le_0.489", lambda: _mask_le(df, "range_pct", 0.489)),
        ("width_le_5.45", lambda: _mask_le(df, "channel_width_pct", 5.45)),
        ("room_le_4.568", lambda: _mask_le(df, "room_to_resist_pct", 4.568)),
        ("volrel_le_3.443", lambda: _mask_le(df, "volume_rel_20", 3.443)),
        ("rsi_le_50.8", lambda: _mask_le(df, "rsi_14", 50.8)),
        ("bb_le_0.489", lambda: _mask_le(df, "bb_pctb", 0.489)),
        ("adv_ge_4.2e6", lambda: _mask_ge(df, "adv_20", 4_200_998.642)),
        ("adv_ge_1.3e6", lambda: _mask_ge(df, "adv_20", 1_326_158.092)),
    ]
    built: Dict[str, pd.Series] = {}
    for name, fn in specs:
        if name.split("_")[0] in ("atr", "range", "width", "room", "volrel", "rsi", "bb", "adv"):
            col_map = {
                "atr": "atr_pct",
                "range": "range_pct",
                "width": "channel_width_pct",
                "room": "room_to_resist_pct",
                "volrel": "volume_rel_20",
                "rsi": "rsi_14",
                "bb": "bb_pctb",
                "adv": "adv_20",
            }
            prefix = name.split("_")[0]
            if col_map[prefix] not in df.columns:
                continue
        m = fn()
        built[name] = m
        variants.append((name, m))

    # AND of two strongest non-ADV price filters (low-vol + tight range)
    if "atr_le_0.733" in built and "range_le_0.766" in built:
        variants.append(
            ("atr0.733+range0.766", built["atr_le_0.733"] & built["range_le_0.766"])
        )
    if "atr_le_0.515" in built and "range_le_0.489" in built:
        variants.append(
            ("atr0.515+range0.489", built["atr_le_0.515"] & built["range_le_0.489"])
        )
    if "atr_le_0.733" in built and "width_le_5.45" in built:
        variants.append(
            ("atr0.733+width5.45", built["atr_le_0.733"] & built["width_le_5.45"])
        )
    if "range_le_0.766" in built and "volrel_le_3.443" in built:
        variants.append(
            ("range0.766+volrel3.4", built["range_le_0.766"] & built["volrel_le_3.443"])
        )
    # Liquidity recovery diagnostic AND best price filter
    if "adv_ge_4.2e6" in built and "atr_le_0.733" in built:
        variants.append(
            ("adv4.2e6+atr0.733", built["adv_ge_4.2e6"] & built["atr_le_0.733"])
        )
    return variants


def build_fund_variants(df: pd.DataFrame) -> List[Tuple[str, pd.Series]]:
    if "has_fund" not in df.columns:
        return []
    variants: List[Tuple[str, pd.Series]] = []
    has = _mask_eq(df, "has_fund", 1)
    variants.append(("has_fund", has))
    if "fund_passes_screen" in df.columns:
        variants.append(("fund_screen", _mask_eq(df, "fund_passes_screen", 1)))
    if "fund_n_filters_pass" in df.columns:
        variants.append(("fund_npass_ge2", _mask_ge(df, "fund_n_filters_pass", 2)))
        variants.append(("fund_npass_ge3", _mask_ge(df, "fund_n_filters_pass", 3)))
    for col, thr, name in (
        ("fcf_roa", 0.12, "fcf_roa_ge_0.12"),
        ("ocf_to_debt", 0.30, "ocf_debt_ge_0.30"),
        ("capex_to_ocf", 0.35, "capex_ocf_le_0.35"),
        ("net_buyback_pct", 0.02, "buyback_ge_0.02"),
    ):
        if col not in df.columns:
            continue
        if "le" in name:
            variants.append((name, _mask_le(df, col, thr)))
        else:
            variants.append((name, _mask_ge(df, col, thr)))
    # Best price + fund AND (if columns present)
    if "atr_pct" in df.columns and "fund_passes_screen" in df.columns:
        variants.append(
            (
                "atr0.733+fund_screen",
                _mask_le(df, "atr_pct", 0.733) & _mask_eq(df, "fund_passes_screen", 1),
            )
        )
    if "adv_20" in df.columns and "fund_passes_screen" in df.columns:
        variants.append(
            (
                "adv4.2e6+fund_screen",
                _mask_ge(df, "adv_20", 4_200_998.642)
                & _mask_eq(df, "fund_passes_screen", 1),
            )
        )
    return variants


def evaluate_variant(
    name: str,
    mask: pd.Series,
    df: pd.DataFrame,
    gain_col: str,
    base_s: Dict[str, Any],
    keeper_syms: set,
) -> Dict[str, Any]:
    sub = df.loc[mask]
    g = pd.to_numeric(sub[gain_col], errors="coerce")
    s = _stats(g)
    d3 = _drop_top3(g)
    years = _year_buckets(sub, gain_col)
    geo = _geo_mean_pf(years)
    win_keep = (
        float(s["win_sum"]) / float(base_s["win_sum"])
        if base_s.get("win_sum") and s["win_sum"] is not None and base_s["win_sum"] > 0
        else None
    )
    lose_keep = (
        float(s["lose_sum"]) / float(base_s["lose_sum"])
        if base_s.get("lose_sum") and s["lose_sum"] is not None and base_s["lose_sum"] > 0
        else None
    )
    adv = _adv_overlap(sub, keeper_syms)
    base_geo = base_s.get("geo_PF")
    yr_all_pos = all(
        (y.get("E") is not None and float(y["E"]) > 0 and y.get("PF") is not None and float(y["PF"]) > 1)
        for y in years
        if int(y.get("n") or 0) >= 30
    )
    lift_e = (
        s["E"] is not None
        and base_s.get("E") is not None
        and float(s["E"]) >= float(base_s["E"])
    )
    lift_pf = (
        s["PF"] is not None
        and base_s.get("PF") is not None
        and float(s["PF"]) >= float(base_s["PF"])
    )
    geo_lift = (
        geo is not None
        and base_geo is not None
        and float(geo) > float(base_geo)
    )
    win_ok = win_keep is not None and float(win_keep) >= WIN_KEEP_MIN
    # Collapse to keeper names: >80% trades in keeper sym set => ADV recovery
    adv_recovery = (
        adv.get("pct_trades_in_keeper_syms") is not None
        and float(adv["pct_trades_in_keeper_syms"]) >= 80.0
    )
    promote = bool(
        s["n"]
        and lift_e
        and lift_pf
        and geo_lift
        and win_ok
        and yr_all_pos
        and d3.get("PF") is not None
        and float(d3["PF"]) > 1.0
        and not adv_recovery
        and name != "baseline"
    )
    if promote:
        verdict = "research_pass"
    elif (
        lift_e
        and lift_pf
        and geo_lift
        and (name.startswith("adv_") or adv_recovery or "adv" in name)
    ):
        verdict = "adv_recovery"
    else:
        verdict = "no_promote"

    row: Dict[str, Any] = {
        "variant": name,
        **{k: s[k] for k in ("n", "E", "PF", "WR", "med", "win_sum", "lose_sum")},
        "drop3_PF": d3.get("PF"),
        "drop3_E": d3.get("E"),
        "geo_PF": None if geo is None else round(geo, 4),
        "win_sum_keep_frac": None if win_keep is None else round(win_keep, 3),
        "lose_sum_keep_frac": None if lose_keep is None else round(lose_keep, 3),
        "years_all_positive": yr_all_pos,
        "adv_recovery_flag": adv_recovery,
        **adv,
        "verdict": verdict,
    }
    for y in years:
        row["E_%s" % y["bucket"]] = y.get("E")
        row["PF_%s" % y["bucket"]] = y.get("PF")
        row["n_%s" % y["bucket"]] = y.get("n")
    return row


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades", type=Path, default=DEFAULT_TRADES)
    ap.add_argument("--keeper", type=Path, default=DEFAULT_KEEPER)
    ap.add_argument("--with-fundamentals", action="store_true")
    ap.add_argument("--outdir", type=Path, default=None)
    args = ap.parse_args()

    t0 = time.perf_counter()
    df = pd.read_csv(args.trades)
    df["stock"] = df["stock"].astype(str).str.upper()
    gain_col = "gain_pct_net" if "gain_pct_net" in df.columns else "gain_pct"
    LOG.info("Loaded %d trades from %s (gain=%s)", len(df), args.trades, gain_col)

    keeper_syms = _keeper_symbols(args.keeper)
    LOG.info("Keeper symbols: %d from %s", len(keeper_syms), args.keeper)

    if args.with_fundamentals:
        from utils.research.channel_touch_fundamentals import attach_fundamentals_asof

        LOG.info("Joining fundamentals_quarterly asof filed...")
        df = attach_fundamentals_asof(df)
        n_has = int(pd.to_numeric(df["has_fund"], errors="coerce").fillna(0).sum())
        n_pass = int(
            pd.to_numeric(df["fund_passes_screen"], errors="coerce").fillna(0).sum()
        )
        LOG.info(
            "Fund join: has_fund=%d/%d (%.1f%%) screen_pass=%d",
            n_has,
            len(df),
            100.0 * n_has / max(1, len(df)),
            n_pass,
        )

    variants = build_price_variants(df)
    if args.with_fundamentals:
        variants.extend(build_fund_variants(df))

    base_mask = variants[0][1]
    base_s = _stats(pd.to_numeric(df.loc[base_mask, gain_col], errors="coerce"))
    base_years = _year_buckets(df.loc[base_mask], gain_col)
    base_s["geo_PF"] = _geo_mean_pf(base_years)

    rows: List[Dict[str, Any]] = []
    tag = "price+fund" if args.with_fundamentals else "price"
    print("\n=== L3 full-unique feature filters (%s) ===" % tag)
    print(
        "baseline: n=%s E=%s PF=%s WR=%s%% geo_PF=%s"
        % (base_s["n"], base_s["E"], base_s["PF"], base_s["WR"], base_s.get("geo_PF"))
    )

    for name, mask in variants:
        row = evaluate_variant(name, mask, df, gain_col, base_s, keeper_syms)
        rows.append(row)
        print(
            "%s: n=%s E=%s PF=%s WR=%s%% geo_PF=%s drop3_PF=%s win$keep=%s "
            "keeper_trade%%=%s years+=%s %s"
            % (
                name,
                row["n"],
                row["E"],
                row["PF"],
                row["WR"],
                row["geo_PF"],
                row["drop3_PF"],
                row["win_sum_keep_frac"],
                row["pct_trades_in_keeper_syms"],
                row["years_all_positive"],
                row["verdict"],
            )
        )

    outdir = dated_outdir(args.outdir or (ROOT / "reports" / "ascending_channels"))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "fund" if args.with_fundamentals else "price"
    summary = pd.DataFrame(rows)
    summary_path = outdir / ("l3_full_unique_feat_%s_summary_%s.csv" % (suffix, stamp))
    summary.to_csv(summary_path, index=False)
    if args.with_fundamentals:
        ann_path = outdir / ("l3_full_unique_feat_fund_annotated_%s.csv" % stamp)
        # keep fund cols + key ids for follow-up
        keep_cols = [
            c
            for c in df.columns
            if c
            in (
                "stock",
                "buy_date",
                "buy_time",
                "gain_pct_net",
                "gain_pct",
                "adv_20",
                "atr_pct",
                "range_pct",
                "has_fund",
                "fund_passes_screen",
                "fund_n_filters_pass",
                "f1_ocf_debt",
                "f2_capex_ocf",
                "f3_buybacks",
                "f4_fcf_roa",
                "ocf_to_debt",
                "capex_to_ocf",
                "net_buyback_pct",
                "fcf_roa",
            )
        ]
        df[keep_cols].to_csv(ann_path, index=False)
        LOG.info("Annotated fund -> %s", ann_path)
    LOG.info("Summary -> %s (%.1fs)", summary_path, time.perf_counter() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
