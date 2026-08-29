"""Loop 5 combos on 15m L3/L4 raw CSVs: filters, logistic, bounce-close honesty."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

from backtest_channel_touch_trades import (  # noqa: E402
    apply_friction,
    filter_trades,
    select_same_day_rs,
    summarize_by_year,
    _summarize,
)
from utils.research.channel_touch_entry_model import attach_scores, time_split

FRIC = 0.10


def pack(df: pd.DataFrame, extra: dict | None = None) -> pd.DataFrame:
    extra = extra or {}
    out = filter_trades(
        df,
        require_in_channel=True,
        max_channel_span_days=float(extra.get("span", 10.0)),
        max_beyond_width=extra.get("beyond"),
        max_rsi=extra.get("max_rsi"),
    )
    m = pd.Series(True, index=out.index)
    if extra.get("min_close_loc") is not None and "close_loc" in out.columns:
        m &= out["close_loc"].fillna(-1) >= float(extra["min_close_loc"])
    if extra.get("min_wait") is not None and "wait_bars" in out.columns:
        m &= out["wait_bars"].fillna(-1) >= int(extra["min_wait"])
    if extra.get("squeeze_rising"):
        m &= out["squeeze_mom_rising"].fillna(0).astype(float) >= 1
    if extra.get("entry_pass"):
        m &= out["entry_pass"].fillna(False).astype(bool)
    out = out.loc[m].copy()
    out = select_same_day_rs(out, rs_col="rs_spy_126d", max_per_day=1)
    return apply_friction(out, FRIC)


def bounce_close_gain(df: pd.DataFrame) -> pd.Series:
    """Reprice fill at the tag-bar close (same-bar confirmation, no wick fill)."""
    buy = pd.to_numeric(df["buy_price"], errors="coerce")
    sell = pd.to_numeric(df["sell_price"], errors="coerce")
    cloc = pd.to_numeric(df["close_loc"], errors="coerce")
    rp = pd.to_numeric(df["range_pct"], errors="coerce") / 100.0
    denom = 1.0 - cloc * rp
    close_px = np.where((denom > 0.05) & np.isfinite(denom), buy / denom, np.nan)
    gain = (sell / close_px - 1.0) * 100.0 - FRIC
    return pd.Series(gain, index=df.index)


def row(name: str, df: pd.DataFrame, gain_col: str = "gain_pct_net") -> dict:
    s = _summarize(df, gain_col=gain_col)
    y = summarize_by_year(df, gain_col=gain_col)
    years = " ".join(
        f"{r['bucket']}:{r['expectancy_pct']}"
        for _, r in y.iterrows()
        if str(r["bucket"]) != "FULL"
    )
    return {"name": name, **s, "years": years}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--l3-raw", required=True, type=Path)
    ap.add_argument("--l4-raw", required=True, type=Path)
    ap.add_argument("--outdir", type=Path, default=ROOT / "reports" / "ascending_channels")
    args = ap.parse_args()

    l3 = pd.read_csv(args.l3_raw)
    l4 = pd.read_csv(args.l4_raw)
    for df in (l3, l4):
        df["buy_date"] = pd.to_datetime(df["buy_date"])
        if "gain_pct_net" not in df.columns:
            df["gain_pct_net"] = df["gain_pct"].astype(float) - FRIC

    rows = []
    rows.append(row("L3 BASE beyond0.25", pack(l3, {"beyond": 0.25})))
    rows.append(row("L3 beyond OFF", pack(l3, {"beyond": None})))
    rows.append(row("L3 beyondOFF+wait12", pack(l3, {"beyond": None, "min_wait": 12})))
    rows.append(row("L3 beyondOFF+rsi50", pack(l3, {"beyond": None, "max_rsi": 50})))
    rows.append(row("L3 beyondOFF+closeLoc0.6", pack(l3, {"beyond": None, "min_close_loc": 0.6})))
    rows.append(row("L3 beyondOFF+closeLoc0.5", pack(l3, {"beyond": None, "min_close_loc": 0.5})))
    rows.append(row("L3 beyondOFF+squeezeRise", pack(l3, {"beyond": None, "squeeze_rising": True})))
    rows.append(row("L3 b025+closeLoc0.6", pack(l3, {"beyond": 0.25, "min_close_loc": 0.6})))
    rows.append(row("L3 beyondOFF+wait12+close0.6", pack(l3, {"beyond": None, "min_wait": 12, "min_close_loc": 0.6})))

    scored, _ = attach_scores(l3, cutoff="2023-01-01", model_kind="logistic")
    tr, te = time_split(scored, "2023-01-01")
    rows.append(row("L3 logistic TRAIN+keeper beyondOFF", pack(tr, {"beyond": None, "entry_pass": True})))
    rows.append(row("L3 logistic TEST+keeper beyondOFF", pack(te, {"beyond": None, "entry_pass": True})))
    rows.append(row("L3 logistic TEST+keeper b025", pack(te, {"beyond": 0.25, "entry_pass": True})))
    rows.append(row("L3 logistic TEST+close0.6 beyondOFF", pack(te, {"beyond": None, "entry_pass": True, "min_close_loc": 0.6})))

    rows.append(row("L4 BASE beyond0.25", pack(l4, {"beyond": 0.25})))
    rows.append(row("L4 beyond OFF", pack(l4, {"beyond": None})))
    rows.append(row("L4 beyondOFF+wait12", pack(l4, {"beyond": None, "min_wait": 12})))
    rows.append(row("L4 beyondOFF+closeLoc0.6", pack(l4, {"beyond": None, "min_close_loc": 0.6})))
    rows.append(row("L4 beyondOFF+squeezeRise", pack(l4, {"beyond": None, "squeeze_rising": True})))
    scored4, _ = attach_scores(l4, cutoff="2023-01-01", model_kind="logistic")
    tr4, te4 = time_split(scored4, "2023-01-01")
    rows.append(row("L4 logistic TEST+keeper beyondOFF", pack(te4, {"beyond": None, "entry_pass": True})))
    rows.append(row("L4 beyondOFF+wait12+close0.6", pack(l4, {"beyond": None, "min_wait": 12, "min_close_loc": 0.6})))

    # Honesty: fill at close for bounce-close filter
    bounce = pack(l3, {"beyond": None, "min_close_loc": 0.6})
    bounce = bounce.copy()
    bounce["gain_close"] = bounce_close_gain(bounce)
    rows.append(row("L3 closeLoc0.6 FILL-AT-CLOSE honesty", bounce, gain_col="gain_close"))

    bounce4 = pack(l4, {"beyond": None, "min_close_loc": 0.6})
    bounce4 = bounce4.copy()
    bounce4["gain_close"] = bounce_close_gain(bounce4)
    rows.append(row("L4 closeLoc0.6 FILL-AT-CLOSE honesty", bounce4, gain_col="gain_close"))

    out = pd.DataFrame(rows)
    keep = [
        "name",
        "n_trades",
        "n_symbols",
        "win_rate_pct",
        "expectancy_pct",
        "profit_factor",
        "median_gain_pct",
        "avg_hold_days",
        "years",
    ]
    cols = [c for c in keep if c in out.columns]
    print(out[cols].to_string(index=False))
    args.outdir.mkdir(parents=True, exist_ok=True)
    dest = args.outdir / "channel_touch_15m_loop5_combos.csv"
    out.to_csv(dest, index=False)
    print("wrote", dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
