#!/usr/bin/env python3
"""Stack leak-safe gates on the 15m H5 volume book to lift WR and PF.

Starts from unique-symbol/day + prior-bar volume_rel_20 >= 1 (H5).
Extra cutoffs are fit on past years only. Drop-top-N on the stitched OOS book.

Usage (Windows CMD):
  venv\\Scripts\\activate && set PYTHONPATH=. && python scripts\\research\\refine_15m_h5_filters.py
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Tuple
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

from backtest_channel_touch_trades import (  # noqa: E402
    YEAR_BUCKETS,
    _summarize,
    apply_friction,
    keep_one_per_symbol_day,
    summarize_by_year,
)
from channel_touch_robustness import (  # noqa: E402
    drop_top_n_winners,
    drop_top_winner_fraction,
    summarize_gains,
)
from mine_15m_unique_filters import (  # noqa: E402
    DEFAULT_TRADES,
    FRICTION,
    GAIN_COL,
    REG_FEATURES,
    add_derived,
    expanding_year_apply,
)
from utils.research.channel_touch_entry_model import (  # noqa: E402
    LogisticScorer,
    feature_matrix,
)
from utils.research.report_paths import dated_outdir, latest_matching  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("refine_15m_h5_filters")

MaskFn = Callable[[pd.DataFrame, pd.DataFrame], pd.Series]

# Books shown in the interactive HTML comparison table (user-facing labels).
REPORT_BOOKS: List[Tuple[str, str]] = [
    ("H5_only", "H5 only (volume_rel \u2265 1)"),
    ("H5_over_p50", "H5 + overshoot \u2265 train p50"),
    ("H5_logistic", "H5 then logistic"),
    ("H5_over_p80", "H5 + overshoot \u2265 train p80"),
    ("H5_over_p80_vol2", "H5 + overshoot p80 + vol \u2265 2"),
]
WINNER_BOOK = "H5_over_p80_vol2"


def _and(a: MaskFn, b: MaskFn) -> MaskFn:
    def fn(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
        return a(train, test).fillna(False) & b(train, test).fillna(False)

    return fn


def h5(_train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(test["volume_rel_20"], errors="coerce") >= 1.0


def stacks() -> List[Tuple[str, str, MaskFn]]:
    def vol_p60(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
        cap = float(pd.to_numeric(train["volume_rel_20"], errors="coerce").quantile(0.60))
        return pd.to_numeric(test["volume_rel_20"], errors="coerce") >= max(1.0, cap)

    def vol_ge2(_train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
        return pd.to_numeric(test["volume_rel_20"], errors="coerce") >= 2.0

    def over_p50(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
        cap = float(pd.to_numeric(train["overshoot"], errors="coerce").quantile(0.50))
        return pd.to_numeric(test["overshoot"], errors="coerce") >= cap

    def over_p80(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
        cap = float(pd.to_numeric(train["overshoot"], errors="coerce").quantile(0.80))
        return pd.to_numeric(test["overshoot"], errors="coerce") >= cap

    def width_p50(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
        cap = float(pd.to_numeric(train["channel_width_pct"], errors="coerce").quantile(0.50))
        return pd.to_numeric(test["channel_width_pct"], errors="coerce") <= cap

    def wait_p40(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
        cap = float(pd.to_numeric(train["wait_bars"], errors="coerce").quantile(0.40))
        return pd.to_numeric(test["wait_bars"], errors="coerce") <= cap

    def atr_band(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
        atr = pd.to_numeric(train["atr_pct"], errors="coerce")
        lo = float(atr.quantile(0.40))
        hi = float(atr.quantile(0.80))
        x = pd.to_numeric(test["atr_pct"], errors="coerce")
        return (x >= lo) & (x <= hi)

    def squeeze_pos(_train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
        return pd.to_numeric(test["squeeze_mom"], errors="coerce") > 0

    return [
        ("H5_only", "volume_rel_20 >= 1", h5),
        ("H5_vol_p60", "H5 and vol >= max(1, train p60)", _and(h5, vol_p60)),
        ("H5_vol_ge2", "H5 and volume_rel_20 >= 2", _and(h5, vol_ge2)),
        ("H5_over_p50", "H5 and overshoot >= train p50", _and(h5, over_p50)),
        ("H5_over_p80", "H5 and overshoot >= train p80", _and(h5, over_p80)),
        ("H5_narrow", "H5 and width <= train p50", _and(h5, width_p50)),
        ("H5_fast_wait", "H5 and wait_bars <= train p40", _and(h5, wait_p40)),
        ("H5_atr_band", "H5 and atr in train p40-p80", _and(h5, atr_band)),
        ("H5_squeeze_pos", "H5 and squeeze_mom > 0", _and(h5, squeeze_pos)),
        (
            "H5_over_p50_narrow",
            "H5 and overshoot p50 and width p50",
            _and(_and(h5, over_p50), width_p50),
        ),
        (
            "H5_over_p80_vol2",
            "H5 and overshoot p80 and vol >= 2",
            _and(_and(h5, over_p80), vol_ge2),
        ),
    ]


def comparison_from_summary(summary: pd.DataFrame, *, friction_pct: float) -> dict:
    """Static comparison rows for the TV report (expanding-year OOS, already net of friction)."""
    by_name = {str(r["book"]): r for _, r in summary.iterrows()}
    rows = []
    for key, label in REPORT_BOOKS:
        r = by_name.get(key)
        if r is None:
            continue
        rows.append(
            {
                "book": label,
                "key": key,
                "n": int(pd.to_numeric(r["n_trades"], errors="coerce") or 0),
                "wr": float(pd.to_numeric(r["win_rate_pct"], errors="coerce") or 0.0),
                "e": float(pd.to_numeric(r["expectancy_pct"], errors="coerce") or 0.0),
                "pf": float(pd.to_numeric(r["profit_factor"], errors="coerce") or 0.0),
                "highlight": key == WINNER_BOOK,
            }
        )
    return {
        "title": "H5 stack (expanding-year OOS, friction %.2f)" % friction_pct,
        "note": (
            "Unique-symbol/day 15m H2 span<=10. Cutoffs fit on prior years only. "
            "Embedded trades are the highlighted book. Research only; not nightly."
        ),
        "highlight": WINNER_BOOK,
        "friction_pct": float(friction_pct),
        "rows": rows,
    }


def latest_refine_summary(outdir: Path) -> Path:
    found = latest_matching(
        outdir,
        "channel_touch_15m_h5_refine_20*.csv",
        exclude_substr=("_stress_",),
    )
    if found is None:
        raise FileNotFoundError("No channel_touch_15m_h5_refine_*.csv in %s" % outdir)
    return found


def _row(name: str, desc: str, df: pd.DataFrame) -> dict:
    s = _summarize(df, gain_col=GAIN_COL)
    out = {"book": name, "desc": desc, **s}
    return out


def stress(name: str, df: pd.DataFrame) -> List[dict]:
    g = pd.to_numeric(df[GAIN_COL], errors="coerce").to_numpy(dtype=float)
    g = g[np.isfinite(g)]
    rows = []
    for label, gg in (
        ("full", g),
        ("drop_top1", drop_top_n_winners(g, 1)),
        ("drop_top5", drop_top_n_winners(g, 5)),
        ("drop_top1pct", drop_top_winner_fraction(g, 0.01)),
    ):
        m = summarize_gains(gg, label)
        rows.append(
            {
                "book": name,
                "stress": label,
                "n": m["n"],
                "expectancy_pct": m["expectancy_pct"],
                "profit_factor": m["profit_factor"],
                "win_rate_pct": m["win_rate_pct"],
            }
        )
    return rows


def _export_report(args: argparse.Namespace) -> int:
    """Winner OOS trades + comparison JSON for generate_channel_touch_tv_report."""
    args.outdir.mkdir(parents=True, exist_ok=True)
    summary_path = latest_refine_summary(args.outdir)
    summary = pd.read_csv(summary_path)
    payload = comparison_from_summary(summary, friction_pct=float(args.friction_pct))
    cmp_path = args.outdir / "channel_touch_15m_h5_refine_comparison.json"
    cmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote %s (%d books)", cmp_path, len(payload["rows"]))

    raw = pd.read_csv(args.trades)
    book = add_derived(apply_friction(keep_one_per_symbol_day(raw), float(args.friction_pct)))
    mask = dict((name, fn) for name, _desc, fn in stacks())[WINNER_BOOK]
    oos = expanding_year_apply(book, mask)
    drop_cols = [c for c in ("_day", "_t", "buy_dt") if c in oos.columns]
    trades_path = args.outdir / "channel_touch_trades_15m_h5_over_p80_vol2_oos.csv"
    oos.drop(columns=drop_cols, errors="ignore").to_csv(trades_path, index=False)
    s = _summarize(oos, gain_col=GAIN_COL)
    summary_txt = args.outdir / "channel_touch_trades_summary_15m_h5_over_p80_vol2_oos.txt"
    summary_txt.write_text(
        "\n".join(
            [
                "15m H5 stack expanding-year OOS (research, not nightly)",
                "preset=15m",
                "timeframe=15m",
                "provider=IB",
                "friction_pct=%.2f" % float(args.friction_pct),
                "entry_mode=h2_resist_break",
                "min_l3_wait_bars=12",
                "max_channel_span_days=10",
                "max_entries_per_day=0",
                "filter=H5_over_p80_vol2",
                "volume_rel_20>=1 then train-year overshoot p80 and volume_rel_20>=2",
                "feature_asof=prior-bar",
                "n_trades=%s" % s.get("n_trades"),
                "win_rate_pct=%s" % s.get("win_rate_pct"),
                "expectancy_pct=%s" % s.get("expectancy_pct"),
                "profit_factor=%s" % s.get("profit_factor"),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    logger.info(
        "winner %s n=%s E=%s PF=%s WR=%s -> %s",
        WINNER_BOOK,
        s.get("n_trades"),
        s.get("expectancy_pct"),
        s.get("profit_factor"),
        s.get("win_rate_pct"),
        trades_path,
    )
    print("Wrote %s" % cmp_path)
    print("Wrote %s" % trades_path)
    print("Wrote %s" % summary_txt)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Refine 15m H5 volume book")
    ap.add_argument("--trades", type=Path, default=DEFAULT_TRADES)
    ap.add_argument("--friction-pct", type=float, default=FRICTION)
    ap.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "reports" / "ascending_channels",
    )
    ap.add_argument(
        "--export-report",
        action="store_true",
        help="Write winner OOS trades + comparison JSON from the latest refine CSV (no logistic re-fit)",
    )
    args = ap.parse_args()
    args.outdir = dated_outdir(args.outdir)
    if args.export_report:
        return _export_report(args)
    raw = pd.read_csv(args.trades)
    book = add_derived(apply_friction(keep_one_per_symbol_day(raw), float(args.friction_pct)))
    logger.info("unique-symbol n=%d", len(book))

    rows: List[dict] = []
    oos_books: Dict[str, pd.DataFrame] = {}
    print("=== expanding-year OOS stacks on H5 ===")
    for name, desc, fn in stacks():
        oos = expanding_year_apply(book, fn)
        oos_books[name] = oos
        s = _summarize(oos, gain_col=GAIN_COL)
        print("%s | %s | n=%s E=%s PF=%s WR=%s med=%s" % (
            name,
            desc,
            s.get("n_trades"),
            s.get("expectancy_pct"),
            s.get("profit_factor"),
            s.get("win_rate_pct"),
            s.get("median_gain_pct"),
        ))
        rows.append(_row(name, desc, oos))

    print("\n=== H5 then expanding logistic (fit on H5 train only) ===")
    h5_oos = oos_books["H5_only"]
    # Fit logistic year-by-year on H5-filtered history, score H5 test.
    def h5_then_log(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
        tr = train.loc[h5(train, train).fillna(False)]
        te_m = h5(train, test).fillna(False)
        if len(tr) < 200 or not te_m.any():
            return pd.Series(False, index=test.index)
        x_tr, med = feature_matrix(tr, REG_FEATURES)
        te = test.loc[te_m]
        x_te, _ = feature_matrix(te, REG_FEATURES, medians=med)
        mu = np.nanmean(x_tr, axis=0)
        sd = np.nanstd(x_tr, axis=0)
        sd = np.where(sd < 1e-8, 1.0, sd)
        z_tr = (x_tr - mu) / sd
        z_te = (x_te - mu) / sd
        y_gain = pd.to_numeric(tr[GAIN_COL], errors="coerce").to_numpy(dtype=float)
        y_gain = np.where(np.isfinite(y_gain), y_gain, 0.0)
        model = LogisticScorer(l2=0.5, lr=0.05, epochs=350)
        model.fit(z_tr, (y_gain > 0).astype(float))
        thresh = float(np.median(model.score(z_tr)))
        p = model.score(z_te)
        keep = pd.Series(False, index=test.index)
        keep.loc[te.index] = p >= thresh
        return keep

    log_oos = expanding_year_apply(book, h5_then_log)
    oos_books["H5_logistic"] = log_oos
    s = _summarize(log_oos, gain_col=GAIN_COL)
    print("H5_logistic | n=%s E=%s PF=%s WR=%s med=%s" % (
        s.get("n_trades"),
        s.get("expectancy_pct"),
        s.get("profit_factor"),
        s.get("win_rate_pct"),
        s.get("median_gain_pct"),
    ))
    rows.append(_row("H5_logistic", "H5 then logistic p>=train median", log_oos))

    # Rank by PF then E among books with n>=3000
    summary = pd.DataFrame(rows)
    ranked = summary.copy()
    ranked["pf_num"] = pd.to_numeric(ranked["profit_factor"], errors="coerce")
    ranked["e_num"] = pd.to_numeric(ranked["expectancy_pct"], errors="coerce")
    ranked["n_num"] = pd.to_numeric(ranked["n_trades"], errors="coerce")
    usable = ranked.loc[ranked["n_num"] >= 3000].sort_values(
        ["pf_num", "e_num"], ascending=False
    )
    print("\n=== rank n>=3000 by PF then E ===")
    print(usable[["book", "n_trades", "win_rate_pct", "expectancy_pct", "profit_factor"]].to_string(index=False))

    print("\n=== drop-top-N on H5_only, best PF, H5_over_p80, H5_logistic ===")
    stress_rows: List[dict] = []
    for key in (
        "H5_only",
        "H5_over_p80",
        "H5_over_p50",
        "H5_vol_ge2",
        "H5_logistic",
        "H5_over_p50_narrow",
        "H5_over_p80_vol2",
        "H5_narrow",
        "H5_fast_wait",
    ):
        if key not in oos_books or oos_books[key].empty:
            continue
        part = stress(key, oos_books[key])
        stress_rows.extend(part)
        print(key)
        print(pd.DataFrame(part).to_string(index=False))

    winner_name = str(usable.iloc[0]["book"]) if not usable.empty else "H5_only"
    winner = oos_books.get(winner_name, h5_oos)
    print("\n=== winner year buckets: %s ===" % winner_name)
    if not winner.empty:
        print(
            summarize_by_year(winner, gain_col=GAIN_COL, buckets=YEAR_BUCKETS).to_string(
                index=False
            )
        )

    args.outdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    p1 = args.outdir / ("channel_touch_15m_h5_refine_%s.csv" % stamp)
    p2 = args.outdir / ("channel_touch_15m_h5_refine_stress_%s.csv" % stamp)
    summary.to_csv(p1, index=False)
    pd.DataFrame(stress_rows).to_csv(p2, index=False)
    print("Wrote %s" % p1)
    print("Wrote %s" % p2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
