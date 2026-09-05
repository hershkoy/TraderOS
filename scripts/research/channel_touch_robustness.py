#!/usr/bin/env python3
"""
Robustness diagnostics for channel-touch (or any) trade CSV.

Tests:
  - Leave-one-out / drop top-N / drop top-p%% winners
  - Winsorize (cap) gains
  - Mean vs median vs trimmed mean; tail-dependency ratio
  - Bootstrap resampling of trade P&L %%
  - Odd/even year and random 50/50 splits
  - Concurrent open exposure + capacity-constrained trade count

Usage (Windows CMD):
  venv\\Scripts\\activate && set PYTHONPATH=. && python scripts\\research\\channel_touch_robustness.py --trades reports\\ascending_channels\\channel_touch_trades_20260825_014435.csv
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.research.report_paths import dated_outdir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("channel_touch_robustness")


def _pf(gains: np.ndarray) -> Optional[float]:
    g = np.asarray(gains, dtype=float)
    if g.size == 0:
        return None
    gp = float(g[g > 0].sum())
    gl = float((-g[g <= 0]).sum())
    if gl <= 0:
        return None if gp <= 0 else float("inf")
    return gp / gl


def summarize_gains(gains: np.ndarray, label: str) -> dict:
    g = np.asarray(gains, dtype=float)
    g = g[np.isfinite(g)]
    n = int(g.size)
    if n == 0:
        return {
            "scenario": label,
            "n": 0,
            "expectancy_pct": None,
            "median_pct": None,
            "trimmed_mean_5_pct": None,
            "profit_factor": None,
            "win_rate_pct": None,
            "sum_pct": None,
        }
    wins = g[g > 0]
    lo = int(np.floor(0.05 * n))
    hi = int(np.ceil(0.95 * n))
    if hi <= lo:
        trimmed = float(np.mean(g))
    else:
        trimmed = float(np.mean(np.sort(g)[lo:hi]))
    pf = _pf(g)
    return {
        "scenario": label,
        "n": n,
        "expectancy_pct": round(float(np.mean(g)), 4),
        "median_pct": round(float(np.median(g)), 4),
        "trimmed_mean_5_pct": round(trimmed, 4),
        "profit_factor": None if pf is None or not np.isfinite(pf) else round(float(pf), 4),
        "win_rate_pct": round(float((g > 0).mean() * 100.0), 2),
        "sum_pct": round(float(g.sum()), 4),
    }


def net_gains(df: pd.DataFrame, friction_pct: float) -> np.ndarray:
    if "gain_pct_net" in df.columns and friction_pct <= 0:
        return df["gain_pct_net"].astype(float).to_numpy()
    g = df["gain_pct"].astype(float).to_numpy()
    return g - float(friction_pct)


def drop_top_n_winners(gains: np.ndarray, n: int) -> np.ndarray:
    g = np.asarray(gains, dtype=float).copy()
    if n <= 0 or g.size == 0:
        return g
    idx = np.argsort(g)[::-1]
    drop = idx[: min(n, g.size)]
    mask = np.ones(g.size, dtype=bool)
    mask[drop] = False
    return g[mask]


def drop_top_winner_fraction(gains: np.ndarray, frac: float) -> np.ndarray:
    g = np.asarray(gains, dtype=float)
    n = max(1, int(np.ceil(frac * g.size)))
    return drop_top_n_winners(g, n)


def winsorize_gains(gains: np.ndarray, cap_pct: Optional[float] = None, mult_avg_win: Optional[float] = None) -> np.ndarray:
    g = np.asarray(gains, dtype=float).copy()
    cap = None
    if cap_pct is not None:
        cap = float(cap_pct)
    if mult_avg_win is not None:
        wins = g[g > 0]
        if wins.size:
            alt = float(mult_avg_win) * float(wins.mean())
            cap = alt if cap is None else min(cap, alt)
    if cap is None:
        return g
    return np.minimum(g, cap)


def tail_dependency_ratio(gains: np.ndarray, top_n: int = 3) -> dict:
    g = np.asarray(gains, dtype=float)
    wins = np.sort(g[g > 0])[::-1]
    gross = float(wins.sum()) if wins.size else 0.0
    top = float(wins[:top_n].sum()) if wins.size else 0.0
    return {
        "top_n": top_n,
        "top_sum_pct": round(top, 4),
        "gross_profit_pct": round(gross, 4),
        "tail_dependency_ratio": None if gross <= 0 else round(top / gross, 4),
        "top_labels_needed": True,
    }


def bootstrap_paths(
    gains: np.ndarray,
    *,
    n_iter: int = 2000,
    seed: int = 42,
) -> dict:
    rng = np.random.default_rng(seed)
    g = np.asarray(gains, dtype=float)
    n = g.size
    if n == 0:
        return {"n_iter": 0, "pct_negative_sum": None, "pct_pf_below_1": None}
    sums = np.empty(n_iter, dtype=float)
    pfs = np.empty(n_iter, dtype=float)
    means = np.empty(n_iter, dtype=float)
    for i in range(n_iter):
        sample = rng.choice(g, size=n, replace=True)
        sums[i] = sample.sum()
        means[i] = sample.mean()
        pf = _pf(sample)
        pfs[i] = pf if pf is not None and np.isfinite(pf) else (100.0 if pf == float("inf") else 0.0)
    return {
        "n_iter": n_iter,
        "n_trades": n,
        "pct_negative_sum": round(float((sums < 0).mean() * 100.0), 2),
        "pct_negative_mean": round(float((means < 0).mean() * 100.0), 2),
        "pct_pf_below_1": round(float((pfs < 1.0).mean() * 100.0), 2),
        "sum_p05": round(float(np.percentile(sums, 5)), 3),
        "sum_p50": round(float(np.percentile(sums, 50)), 3),
        "sum_p95": round(float(np.percentile(sums, 95)), 3),
        "mean_p05": round(float(np.percentile(means, 5)), 4),
        "mean_p50": round(float(np.percentile(means, 50)), 4),
        "mean_p95": round(float(np.percentile(means, 95)), 4),
    }


def year_split_stats(df: pd.DataFrame, gains: np.ndarray) -> List[dict]:
    years = pd.to_datetime(df["buy_date"]).dt.year.to_numpy()
    rows = []
    for y in sorted(set(years.tolist())):
        m = years == y
        rows.append(summarize_gains(gains[m], f"year_{y}"))
    odd = gains[years % 2 == 1]
    even = gains[years % 2 == 0]
    rows.append(summarize_gains(odd, "odd_years"))
    rows.append(summarize_gains(even, "even_years"))
    return rows


def random_half_splits(
    gains: np.ndarray,
    *,
    n_splits: int = 20,
    seed: int = 42,
) -> List[dict]:
    rng = np.random.default_rng(seed)
    g = np.asarray(gains, dtype=float)
    n = g.size
    rows = []
    for i in range(n_splits):
        idx = rng.permutation(n)
        a = g[idx[: n // 2]]
        b = g[idx[n // 2 :]]
        sa = summarize_gains(a, f"split_{i}_A")
        sb = summarize_gains(b, f"split_{i}_B")
        rows.append(
            {
                "split": i,
                "E_A": sa["expectancy_pct"],
                "PF_A": sa["profit_factor"],
                "E_B": sb["expectancy_pct"],
                "PF_B": sb["profit_factor"],
                "both_positive_E": bool(
                    sa["expectancy_pct"] is not None
                    and sb["expectancy_pct"] is not None
                    and sa["expectancy_pct"] > 0
                    and sb["expectancy_pct"] > 0
                ),
            }
        )
    return rows


def concurrent_open_stats(df: pd.DataFrame) -> dict:
    buy = pd.to_datetime(df["buy_date"])
    sell = pd.to_datetime(df["sell_date"])
    if buy.empty:
        return {"max": 0, "p95": 0, "median": 0}
    days = pd.date_range(buy.min(), sell.max(), freq="B")
    open_n = np.array([((buy <= d) & (sell >= d)).sum() for d in days], dtype=int)
    return {
        "max": int(open_n.max()) if open_n.size else 0,
        "p95": int(np.percentile(open_n, 95)) if open_n.size else 0,
        "median": int(np.median(open_n)) if open_n.size else 0,
        "mean": round(float(open_n.mean()), 2) if open_n.size else 0.0,
    }


def _tie_sort_frame(df: pd.DataFrame, *, tie_break: str) -> pd.DataFrame:
    """Stable same-day order. rs = higher RS first (HTML default); wait = longer wait first; fifo = symbol."""
    t = df.copy()
    t["_buy"] = pd.to_datetime(t["buy_date"], errors="coerce")
    kind = (tie_break or "rs").lower()
    if kind == "wait" and "wait_bars" in t.columns:
        t["_rk"] = -pd.to_numeric(t["wait_bars"], errors="coerce").fillna(0.0)
    elif kind == "rs" and "rs_spy_126d" in t.columns:
        t["_rk"] = -pd.to_numeric(t["rs_spy_126d"], errors="coerce").fillna(1e18)
    else:
        t["_rk"] = 0.0
    t["_sym"] = t["stock"].astype(str) if "stock" in t.columns else ""
    return t.sort_values(["_buy", "_rk", "_sym"], kind="mergesort")


def apply_max_open(
    df: pd.DataFrame,
    max_open: int,
    *,
    tie_break: str = "rs",
) -> pd.DataFrame:
    """Greedy keep trades in buy-date order while open count < max_open.

    ``tie_break``: ``rs`` (HTML / prior robustness default), ``wait`` (longer
    wait first — not RS), or ``fifo`` (symbol). A slot still counts as occupied
    on the sell date (sell >= buy).
    """
    if max_open <= 0 or df.empty:
        return df.copy()
    t = _tie_sort_frame(df, tie_break=tie_break)
    kept_idx: List[int] = []
    open_exits: List[pd.Timestamp] = []
    for idx, row in t.iterrows():
        b = pd.Timestamp(row["buy_date"])
        open_exits = [e for e in open_exits if e >= b]
        if len(open_exits) >= max_open:
            continue
        kept_idx.append(idx)
        open_exits.append(pd.Timestamp(row["sell_date"]))
    extra = [c for c in ("_buy", "_rk", "_sym") if c in t.columns]
    return t.loc[kept_idx].drop(columns=extra).copy()


def n_open_at_entry(df: pd.DataFrame) -> pd.Series:
    """Count other trades with buy < this buy and sell >= this buy (causal)."""
    if df.empty:
        return pd.Series(dtype=int)
    buy = pd.to_datetime(df["buy_date"], errors="coerce")
    sell = pd.to_datetime(df["sell_date"], errors="coerce")
    buys = buy.to_numpy()
    sells = sell.to_numpy()
    out = np.zeros(len(df), dtype=int)
    for i in range(len(df)):
        b = buys[i]
        if pd.isna(b):
            continue
        out[i] = int(((buys < b) & (sells >= b)).sum())
    return pd.Series(out, index=df.index, dtype=int)


def same_day_fill_count(df: pd.DataFrame) -> pd.Series:
    """How many unique-symbol fills share this buy_date (known at EOD, not intra-day)."""
    if df.empty or "buy_date" not in df.columns:
        return pd.Series(dtype=int)
    day = pd.to_datetime(df["buy_date"], errors="coerce").dt.normalize()
    counts = day.value_counts()
    return day.map(counts).astype(int)


def skip_crowded_days(df: pd.DataFrame, max_names: int) -> pd.DataFrame:
    """Stand aside when that calendar day has more than ``max_names`` fills."""
    if df.empty or int(max_names) <= 0:
        return df.iloc[0:0].copy()
    n = same_day_fill_count(df)
    return df.loc[n <= int(max_names)].copy()


def cap_same_day(
    df: pd.DataFrame,
    k: int,
    *,
    tie_break: str = "wait",
) -> pd.DataFrame:
    """Keep at most ``k`` fills per calendar day (FIFO by ``tie_break``)."""
    if df.empty or int(k) <= 0:
        return df.iloc[0:0].copy()
    t = _tie_sort_frame(df, tie_break=tie_break)
    t["_day"] = t["_buy"].dt.normalize()
    kept = t.groupby("_day", sort=False, as_index=False).head(int(k))
    extra = [c for c in ("_buy", "_rk", "_sym", "_day") if c in kept.columns]
    return kept.drop(columns=extra).copy()


def top_winner_labels(df: pd.DataFrame, gains: np.ndarray, n: int = 5) -> List[dict]:
    order = np.argsort(gains)[::-1][:n]
    out = []
    for i in order:
        out.append(
            {
                "stock": str(df.iloc[int(i)]["stock"]),
                "buy_date": str(df.iloc[int(i)]["buy_date"])[:10],
                "gain_pct_net": round(float(gains[int(i)]), 3),
            }
        )
    return out


def run_suite(
    df: pd.DataFrame,
    *,
    friction_pct: float = 0.25,
    bootstrap_iter: int = 2000,
    seed: int = 42,
) -> Dict[str, object]:
    g = net_gains(df, friction_pct)
    base = summarize_gains(g, "baseline")
    scenarios = [
        base,
        summarize_gains(drop_top_n_winners(g, 1), "drop_top_1"),
        summarize_gains(drop_top_n_winners(g, 3), "drop_top_3"),
        summarize_gains(drop_top_n_winners(g, 5), "drop_top_5"),
        summarize_gains(drop_top_winner_fraction(g, 0.05), "drop_top_5pct_winners"),
        summarize_gains(winsorize_gains(g, cap_pct=50.0), "winsor_cap_50"),
        summarize_gains(winsorize_gains(g, cap_pct=30.0), "winsor_cap_30"),
        summarize_gains(winsorize_gains(g, cap_pct=20.0), "winsor_cap_20"),
        summarize_gains(winsorize_gains(g, mult_avg_win=3.0), "winsor_3x_avg_win"),
    ]
    # Exclude BETR if present
    if (df["stock"].astype(str).str.upper() == "BETR").any():
        m = df["stock"].astype(str).str.upper() != "BETR"
        scenarios.append(summarize_gains(g[m.to_numpy()], "exclude_BETR"))

    tail = tail_dependency_ratio(g, top_n=3)
    boot = bootstrap_paths(g, n_iter=bootstrap_iter, seed=seed)
    years = year_split_stats(df, g)
    halves = random_half_splits(g, n_splits=20, seed=seed)
    both_pos = sum(1 for r in halves if r["both_positive_E"])
    exposure = concurrent_open_stats(df)
    capacity_rows = []
    for cap in (5, 10, 15, 20):
        sub = apply_max_open(df, cap)
        sg = net_gains(sub, friction_pct)
        capacity_rows.append({**summarize_gains(sg, f"max_open_{cap}"), "max_open": cap})

    verdict_bits = []
    d1 = next(s for s in scenarios if s["scenario"] == "drop_top_1")
    d3 = next(s for s in scenarios if s["scenario"] == "drop_top_3")
    w20 = next(s for s in scenarios if s["scenario"] == "winsor_cap_20")
    if d1["profit_factor"] is not None and d1["profit_factor"] < 1.0:
        verdict_bits.append("FAIL: PF < 1 after dropping top-1 winner")
    elif d1["expectancy_pct"] is not None and d1["expectancy_pct"] <= 0:
        verdict_bits.append("FAIL: E <= 0 after dropping top-1 winner")
    else:
        verdict_bits.append(
            f"PASS soft: after drop top-1, E={d1['expectancy_pct']} PF={d1['profit_factor']}"
        )
    if d3["expectancy_pct"] is not None and d3["expectancy_pct"] > 0:
        verdict_bits.append(f"PASS soft: after drop top-3, E still {d3['expectancy_pct']}")
    else:
        verdict_bits.append("WARN: E not clearly positive after drop top-3")
    if w20["expectancy_pct"] is not None and w20["expectancy_pct"] > 0:
        verdict_bits.append(f"PASS soft: winsor 20% still E={w20['expectancy_pct']}")
    else:
        verdict_bits.append("WARN: winsor 20% collapses expectancy")
    tdr = tail.get("tail_dependency_ratio")
    if tdr is not None and tdr >= 0.40:
        verdict_bits.append(f"WARN: top-3 tail dependency {tdr:.1%} (>=40%)")
    else:
        verdict_bits.append(f"OK: top-3 tail dependency {tdr}")
    if boot.get("pct_negative_mean") is not None and boot["pct_negative_mean"] > 20:
        verdict_bits.append(f"WARN: bootstrap P(mean<0)={boot['pct_negative_mean']}%")
    else:
        verdict_bits.append(f"OK: bootstrap P(mean<0)={boot.get('pct_negative_mean')}%")

    return {
        "n_trades": int(len(df)),
        "friction_pct": friction_pct,
        "baseline": base,
        "scenarios": scenarios,
        "tail_dependency": {**tail, "top_trades": top_winner_labels(df, g, 5)},
        "bootstrap": boot,
        "year_splits": years,
        "half_splits_summary": {
            "n_splits": len(halves),
            "both_halves_positive_E": both_pos,
            "pct_both_positive": round(100.0 * both_pos / max(len(halves), 1), 1),
        },
        "half_splits": halves,
        "exposure": exposure,
        "capacity": capacity_rows,
        "verdict": verdict_bits,
    }


def to_markdown(report: Dict[str, object]) -> str:
    lines = [
        f"# Channel-touch robustness ({datetime.now().strftime('%Y-%m-%d')})",
        "",
        f"n={report['n_trades']}, friction={report['friction_pct']}%",
        "",
        "## Verdict",
        "",
    ]
    for v in report["verdict"]:  # type: ignore[attr-defined]
        lines.append(f"- {v}")
    lines.extend(["", "## Scenarios", "", "| scenario | n | E% | median% | trimmed5% | PF | win% |", "|---|---|---|---|---|---|---|"])
    for s in report["scenarios"]:  # type: ignore[attr-defined]
        lines.append(
            f"| {s['scenario']} | {s['n']} | {s['expectancy_pct']} | {s['median_pct']} | "
            f"{s['trimmed_mean_5_pct']} | {s['profit_factor']} | {s['win_rate_pct']} |"
        )
    tail = report["tail_dependency"]  # type: ignore[index]
    lines.extend(
        [
            "",
            "## Tail dependency",
            "",
            f"- Top-3 / gross profit = **{tail.get('tail_dependency_ratio')}**",
            f"- Top trades: {tail.get('top_trades')}",
            "",
            "## Bootstrap",
            "",
            "```",
            json.dumps(report["bootstrap"], indent=2),
            "```",
            "",
            "## Exposure / capacity",
            "",
            f"- Concurrent open: {report['exposure']}",
            "",
            "| capacity | n | E% | PF |",
            "|---|---|---|---|",
        ]
    )
    for c in report["capacity"]:  # type: ignore[attr-defined]
        lines.append(f"| {c['max_open']} | {c['n']} | {c['expectancy_pct']} | {c['profit_factor']} |")
    lines.extend(
        [
            "",
            f"## Half-splits: {report['half_splits_summary']}",
            "",
            "## Year splits",
            "",
            "| bucket | n | E% | PF |",
            "|---|---|---|---|",
        ]
    )
    for y in report["year_splits"]:  # type: ignore[attr-defined]
        lines.append(f"| {y['scenario']} | {y['n']} | {y['expectancy_pct']} | {y['profit_factor']} |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Outlier / bootstrap robustness for trade CSV")
    ap.add_argument("--trades", type=Path, required=True)
    ap.add_argument("--friction-pct", type=float, default=0.25)
    ap.add_argument("--bootstrap-iter", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=Path, default=ROOT / "reports/ascending_channels")
    ap.add_argument(
        "--status-md",
        type=Path,
        default=ROOT / "docs/status_log/edge_hunt/channel_touch/2026-08-25_channel_touch_robustness.md",
    )
    args = ap.parse_args()
    args.outdir = dated_outdir(args.outdir)

    df = pd.read_csv(args.trades)
    df["stock"] = df["stock"].astype(str).str.strip().str.upper()
    report = run_suite(
        df,
        friction_pct=args.friction_pct,
        bootstrap_iter=args.bootstrap_iter,
        seed=args.seed,
    )
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.outdir.mkdir(parents=True, exist_ok=True)
    out_json = args.outdir / f"channel_touch_robustness_{stamp}.json"
    out_csv = args.outdir / f"channel_touch_robustness_scenarios_{stamp}.csv"
    out_json.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(report["scenarios"]).to_csv(out_csv, index=False)  # type: ignore[arg-type]
    md = to_markdown(report)
    args.status_md.parent.mkdir(parents=True, exist_ok=True)
    args.status_md.write_text(md + f"\nArtifacts: `{out_json.as_posix()}`, `{out_csv.as_posix()}`\n", encoding="utf-8")
    logger.info("Wrote %s", args.status_md)
    logger.info("Wrote %s", out_json)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
