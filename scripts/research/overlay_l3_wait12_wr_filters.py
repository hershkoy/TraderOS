"""Causal WR filters on 15m L3 wait-12 keeper CSV (no detector rescan).

A) Skip if signal-bar range_pct > threshold (default 1.0%% of close).
B) Require next N bars close above support (support from channel_pos at fill,
   advanced by slope_pct_per_bar).

Gate: WR/E/PF vs baseline; year buckets; drop-top-3 PF; winner-$ blunt check.

Usage (Windows CMD):
  venv\\Scripts\\activate
  set PYTHONPATH=.
  python scripts\\research\\overlay_l3_wait12_wr_filters.py
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.data.ohlcv_loader import load_ohlcv_many  # noqa: E402
from utils.research.report_paths import dated_outdir  # noqa: E402

LOG = logging.getLogger("overlay_l3_wait12_wr_filters")

DEFAULT_TRADES = (
    ROOT / "reports" / "ascending_channels" / "channel_touch_15m_l3_wait12.csv"
)


def _naive(ts: object) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    return t


def _stats(g: pd.Series) -> Dict[str, Any]:
    g = pd.to_numeric(g, errors="coerce").dropna()
    if g.empty:
        return {"n": 0, "E": None, "PF": None, "WR": None}
    wins = float(g[g > 0].sum())
    losses = float((-g[g <= 0]).sum())
    pf = wins / losses if losses > 0 else float("inf")
    return {
        "n": int(len(g)),
        "E": round(float(g.mean()), 4),
        "PF": round(pf, 3),
        "WR": round(float((g > 0).mean() * 100.0), 2),
        "med": round(float(g.median()), 4),
        "win_sum": round(wins, 2),
        "lose_sum": round(losses, 2),
    }


def _year_buckets(df: pd.DataFrame, gain_col: str) -> List[Dict[str, Any]]:
    y = pd.to_datetime(df["buy_date"], errors="coerce").dt.year
    rows = []
    for label, lo, hi in (
        ("2018-19", 0, 2019),
        ("2020-21", 2020, 2021),
        ("2022-23", 2022, 2023),
        ("2024-26", 2024, 2099),
    ):
        m = (y >= lo) & (y <= hi)
        s = _stats(df.loc[m, gain_col])
        s["bucket"] = label
        rows.append(s)
    return rows


def _drop_top3(g: pd.Series) -> Dict[str, Any]:
    g = pd.to_numeric(g, errors="coerce").dropna().to_numpy(dtype=float)
    if len(g) <= 3:
        return _stats(pd.Series(dtype=float))
    keep = np.ones(len(g), dtype=bool)
    keep[np.argsort(g)[::-1][:3]] = False
    return _stats(pd.Series(g[keep]))


def _support_at_fill(row: pd.Series) -> Optional[float]:
    px = float(row["buy_price"])
    pos = float(row["channel_pos"]) if pd.notna(row.get("channel_pos")) else float("nan")
    width = float(row["channel_width"]) if pd.notna(row.get("channel_width")) else float("nan")
    if px > 0 and np.isfinite(pos) and np.isfinite(width) and width > 0:
        return px - pos * width
    return None


def annotate_filters(
    trades: pd.DataFrame,
    panels: Dict[str, pd.DataFrame],
    *,
    max_range_pct: float,
    hold_bars: int,
) -> pd.DataFrame:
    out = trades.copy()
    out["sig_range_pct"] = np.nan
    out["sig_ok_range"] = False
    out["hold_ok"] = False
    out["support_fill"] = np.nan
    out["next_min_close_vs_sup"] = np.nan

    for i, row in out.iterrows():
        sym = str(row["stock"]).upper()
        df = panels.get(sym)
        if df is None or df.empty or "buy_time" not in row or pd.isna(row["buy_time"]):
            continue
        bar = _naive(row["buy_time"])
        idx = df.index
        # exact or nearest prior bar
        if bar in idx:
            loc = int(idx.get_loc(bar))
        else:
            pos = idx.searchsorted(bar)
            if pos <= 0:
                continue
            loc = int(pos - 1) if pos >= len(idx) or idx[pos] != bar else int(pos)
            if loc < 0 or loc >= len(idx):
                continue
        hi = float(df["high"].iloc[loc])
        lo = float(df["low"].iloc[loc])
        cl = float(df["close"].iloc[loc])
        if cl <= 0:
            continue
        rng = (hi - lo) / cl * 100.0
        out.at[i, "sig_range_pct"] = rng
        out.at[i, "sig_ok_range"] = bool(rng <= float(max_range_pct))

        sup0 = _support_at_fill(row)
        if sup0 is None or not np.isfinite(sup0):
            continue
        out.at[i, "support_fill"] = sup0
        slope_pct = float(row["slope_pct_per_bar"]) if pd.notna(row.get("slope_pct_per_bar")) else 0.0
        n_hold = max(1, int(hold_bars))
        ok = True
        min_vs = float("inf")
        for k in range(1, n_hold + 1):
            j = loc + k
            if j >= len(df):
                ok = False
                break
            # support advances ~slope_pct_per_bar of support level per bar
            sup_k = float(sup0) * (1.0 + slope_pct / 100.0 * k)
            cj = float(df["close"].iloc[j])
            min_vs = min(min_vs, cj - sup_k)
            if not np.isfinite(cj) or cj < sup_k:
                ok = False
                break
        out.at[i, "hold_ok"] = bool(ok)
        if np.isfinite(min_vs):
            out.at[i, "next_min_close_vs_sup"] = min_vs
    return out


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades", type=Path, default=DEFAULT_TRADES)
    ap.add_argument("--max-range-pct", type=float, default=1.0)
    ap.add_argument("--hold-bars", type=int, default=1, help="Next bars that must close >= support")
    ap.add_argument("--outdir", type=Path, default=None)
    args = ap.parse_args()

    t0 = time.perf_counter()
    df = pd.read_csv(args.trades)
    df["stock"] = df["stock"].astype(str).str.upper()
    df["buy_time"] = pd.to_datetime(df["buy_time"], errors="coerce")
    gain_col = "gain_pct_net" if "gain_pct_net" in df.columns else "gain_pct"
    LOG.info("Loaded %d trades from %s (gain=%s)", len(df), args.trades, gain_col)

    syms = sorted(df["stock"].unique().tolist())
    start = pd.to_datetime(df["buy_date"], errors="coerce").min() - pd.Timedelta(days=5)
    end = pd.to_datetime(df["sell_date"], errors="coerce").max() + pd.Timedelta(days=5)
    LOG.info("Loading IB 15m for %d symbols %s -> %s", len(syms), start.date(), end.date())
    panels = load_ohlcv_many(
        syms,
        start=start.date() if hasattr(start, "date") else date(2018, 11, 1),
        end=end.date() if hasattr(end, "date") else date.today(),
        timeframe="15m",
        provider="IB",
        use_cache=True,
    )
    # normalize indexes
    norm: Dict[str, pd.DataFrame] = {}
    for sym, px in panels.items():
        if px is None or px.empty:
            continue
        p = px.copy()
        p.index = pd.DatetimeIndex(pd.to_datetime(p.index)).tz_localize(None)
        p = p[~p.index.duplicated(keep="last")].sort_index()
        norm[str(sym).upper()] = p
    LOG.info("Panels ready: %d/%d", len(norm), len(syms))

    ann = annotate_filters(
        df,
        norm,
        max_range_pct=float(args.max_range_pct),
        hold_bars=int(args.hold_bars),
    )
    n_range = int(ann["sig_ok_range"].fillna(False).sum())
    n_hold = int(ann["hold_ok"].fillna(False).sum())
    both = ann["sig_ok_range"].fillna(False) & ann["hold_ok"].fillna(False)
    LOG.info(
        "Annotate: range_ok=%d/%d hold_ok=%d/%d both=%d (max_range=%.2f hold_bars=%d)",
        n_range,
        len(ann),
        n_hold,
        len(ann),
        int(both.sum()),
        float(args.max_range_pct),
        int(args.hold_bars),
    )

    variants: List[Tuple[str, pd.Series]] = [
        ("baseline", pd.Series(True, index=ann.index)),
        ("range_le_%.2f" % float(args.max_range_pct), ann["sig_ok_range"].fillna(False)),
        ("hold_%db" % int(args.hold_bars), ann["hold_ok"].fillna(False)),
        (
            "range+hold",
            ann["sig_ok_range"].fillna(False) & ann["hold_ok"].fillna(False),
        ),
    ]
    # Also hold 2 bars A/B if default was 1
    if int(args.hold_bars) == 1:
        ann2 = annotate_filters(df, norm, max_range_pct=float(args.max_range_pct), hold_bars=2)
        variants.append(("hold_2b", ann2["hold_ok"].fillna(False)))
        variants.append(
            (
                "range+hold_2b",
                ann["sig_ok_range"].fillna(False) & ann2["hold_ok"].fillna(False),
            )
        )

    base_g = pd.to_numeric(ann[gain_col], errors="coerce")
    base_s = _stats(base_g)
    rows = []
    print("\n=== L3 wait-12 WR filter A/B ===")
    print(
        "baseline: n=%s E=%s PF=%s WR=%s%% win$=%s lose$=%s"
        % (
            base_s["n"],
            base_s["E"],
            base_s["PF"],
            base_s["WR"],
            base_s["win_sum"],
            base_s["lose_sum"],
        )
    )
    for name, mask in variants:
        sub = ann.loc[mask]
        g = pd.to_numeric(sub[gain_col], errors="coerce")
        s = _stats(g)
        d3 = _drop_top3(g)
        years = _year_buckets(sub, gain_col)
        # winner-$ blunt: kept win_sum / baseline win_sum
        win_keep = (
            float(s["win_sum"]) / float(base_s["win_sum"])
            if base_s["win_sum"] and s["win_sum"] is not None
            else None
        )
        lose_keep = (
            float(s["lose_sum"]) / float(base_s["lose_sum"])
            if base_s["lose_sum"] and s["lose_sum"] is not None
            else None
        )
        yr_ok = all(
            (y.get("E") is not None and y["E"] > 0 and y.get("PF") is not None and y["PF"] > 1)
            for y in years
            if y.get("n", 0) >= 30
        )
        promote = (
            s["n"]
            and s["E"] is not None
            and s["PF"] is not None
            and float(s["E"]) >= float(base_s["E"])
            and float(s["PF"]) >= float(base_s["PF"])
            and float(s["WR"]) > float(base_s["WR"])
            and d3.get("PF") is not None
            and float(d3["PF"]) > 1.0
            and (win_keep is None or win_keep >= 0.85)
            and yr_ok
        )
        print(
            "%s: n=%s E=%s PF=%s WR=%s%% drop3_PF=%s win$keep=%.2f lose$keep=%.2f years+=%s %s"
            % (
                name,
                s["n"],
                s["E"],
                s["PF"],
                s["WR"],
                d3.get("PF"),
                win_keep or 0.0,
                lose_keep or 0.0,
                yr_ok,
                "PASS" if promote else "no_promote",
            )
        )
        for y in years:
            print(
                "  %s n=%s E=%s PF=%s WR=%s"
                % (y["bucket"], y["n"], y["E"], y["PF"], y["WR"])
            )
        rows.append(
            {
                "variant": name,
                **{k: s[k] for k in ("n", "E", "PF", "WR", "med", "win_sum", "lose_sum")},
                "drop3_PF": d3.get("PF"),
                "drop3_E": d3.get("E"),
                "win_sum_keep_frac": None if win_keep is None else round(win_keep, 3),
                "lose_sum_keep_frac": None if lose_keep is None else round(lose_keep, 3),
                "years_all_positive": yr_ok,
                "verdict": "research_pass" if promote else "no_promote",
            }
        )

    outdir = dated_outdir(args.outdir or (ROOT / "reports" / "ascending_channels"))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = pd.DataFrame(rows)
    summary_path = outdir / ("l3_wait12_wr_filters_summary_%s.csv" % stamp)
    ann_path = outdir / ("l3_wait12_wr_filters_annotated_%s.csv" % stamp)
    summary.to_csv(summary_path, index=False)
    ann.to_csv(ann_path, index=False)
    LOG.info("Summary -> %s (%.1fs)", summary_path, time.perf_counter() - t0)
    LOG.info("Annotated -> %s", ann_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
