"""Filter-then-RS A/B on an existing l3_touch raw trades CSV (no rescan)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

from backtest_channel_touch_trades import (  # noqa: E402
    apply_friction,
    filter_trades,
    select_same_day_rs,
    _summarize,
)

RAW = ROOT / "reports" / "ascending_channels" / "channel_touch_trades_raw_20260828_192557.csv"
FRIC = 0.25


def keeper(df: pd.DataFrame, extra: dict | None = None) -> pd.DataFrame:
    extra = extra or {}
    out = filter_trades(
        df,
        require_in_channel=True,
        max_channel_span_days=365,
        max_beyond_width=0.25,
        max_channel_age_days=extra.get("max_age"),
    )
    extra_m = pd.Series(True, index=out.index)
    if extra.get("max_rsi") is not None and "rsi_14" in out.columns:
        extra_m &= out["rsi_14"].fillna(999) <= float(extra["max_rsi"])
    if extra.get("max_wait_days") is not None:
        extra_m &= out["_wait_days"].fillna(1e9) <= float(extra["max_wait_days"])
    if extra.get("min_wait_days") is not None:
        extra_m &= out["_wait_days"].fillna(-1) >= float(extra["min_wait_days"])
    if extra.get("squeeze_rising"):
        extra_m &= out["squeeze_mom_rising"].fillna(0).astype(float) >= 1
    if extra.get("squeeze_pos"):
        extra_m &= out["squeeze_mom"].fillna(-1e9) > 0
    if extra.get("max_bb") is not None:
        extra_m &= out["bb_pctb"].fillna(999) <= float(extra["max_bb"])
    out = out.loc[extra_m].copy()
    out = select_same_day_rs(out, rs_col="rs_spy_126d", max_per_day=1)
    out = apply_friction(out, FRIC)
    return out


def row(name: str, df: pd.DataFrame) -> dict:
    s = _summarize(df, gain_col="gain_pct_net")
    return {"name": name, **s}


def main() -> None:
    raw = pd.read_csv(RAW)
    raw["buy_date"] = pd.to_datetime(raw["buy_date"])
    raw["channel_end"] = pd.to_datetime(raw["channel_end"])
    raw["_wait_days"] = (raw["buy_date"] - raw["channel_end"]).dt.days
    print("raw", len(raw), "wait days describe")
    print(raw["_wait_days"].describe())

    base = keeper(raw, {})
    rows = [row("BASE in+span365+b025+RStop1", base)]

    for w in (15, 21, 30, 45, 60, 90, 120, 180):
        rows.append(row(f"max_wait_days {w}", keeper(raw, {"max_wait_days": w})))
    for w in (3, 5, 8, 10):
        rows.append(row(f"min_wait_days {w}", keeper(raw, {"min_wait_days": w})))
    for r in (50, 55, 60, 65, 70, 75):
        rows.append(row(f"max_rsi {r}", keeper(raw, {"max_rsi": r})))
    for a in (90, 120, 150, 180, 240, 300):
        rows.append(row(f"max_age {a}", keeper(raw, {"max_age": a})))
    rows.append(row("squeeze_rising", keeper(raw, {"squeeze_rising": True})))
    rows.append(row("squeeze_mom>0", keeper(raw, {"squeeze_pos": True})))
    for b in (0.2, 0.3, 0.4, 0.5, 0.6, 0.8):
        rows.append(row(f"max_bb {b}", keeper(raw, {"max_bb": b})))

    # combos after seeing singles - always include rsi60 + wait60
    rows.append(row("rsi60+wait60", keeper(raw, {"max_rsi": 60, "max_wait_days": 60})))
    rows.append(row("rsi60+age180", keeper(raw, {"max_rsi": 60, "max_age": 180})))
    rows.append(row("wait60+age180", keeper(raw, {"max_wait_days": 60, "max_age": 180})))
    rows.append(row("rsi60+wait60+age180", keeper(raw, {"max_rsi": 60, "max_wait_days": 60, "max_age": 180})))

    out = pd.DataFrame(rows)
    keep = [
        "name",
        "n_trades",
        "n_symbols",
        "win_rate_pct",
        "expectancy_pct",
        "profit_factor",
        "median_gain_pct",
        "avg_win_pct",
        "avg_loss_pct",
        "avg_hold_days",
    ]
    cols = [c for c in keep if c in out.columns]
    print(out[cols].to_string(index=False))
    dest = ROOT / "reports" / "ascending_channels" / "channel_touch_l3_opt_filter_ab_20260828.csv"
    out.to_csv(dest, index=False)
    print("wrote", dest)


if __name__ == "__main__":
    main()
