"""Filter-then-RS A/B on a 15m channel-touch raw trades CSV (no rescan)."""
from __future__ import annotations

import argparse
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
    summarize_by_year,
    _summarize,
)

FRIC = 0.10
SPAN = 10.0
BEYOND = 0.25


def keeper(df: pd.DataFrame, extra: dict | None = None) -> pd.DataFrame:
    extra = extra or {}
    out = filter_trades(
        df,
        require_in_channel=True,
        max_channel_span_days=float(extra.get("span", SPAN)),
        max_beyond_width=extra.get("beyond", BEYOND),
        max_rsi=extra.get("max_rsi"),
        max_channel_pos=extra.get("max_pos"),
    )
    extra_m = pd.Series(True, index=out.index)
    if extra.get("min_wait_bars") is not None and "wait_bars" in out.columns:
        extra_m &= out["wait_bars"].fillna(-1) >= int(extra["min_wait_bars"])
    if extra.get("touch_ge") is not None and "touch_num" in out.columns:
        extra_m &= out["touch_num"].fillna(0) >= int(extra["touch_ge"])
    if extra.get("touch_eq") is not None and "touch_num" in out.columns:
        extra_m &= out["touch_num"].fillna(0) == int(extra["touch_eq"])
    if extra.get("squeeze_rising"):
        extra_m &= out["squeeze_mom_rising"].fillna(0).astype(float) >= 1
    if extra.get("max_vol_rel") is not None and "volume_rel_20" in out.columns:
        extra_m &= out["volume_rel_20"].fillna(999) <= float(extra["max_vol_rel"])
    if extra.get("min_vol_rel") is not None and "volume_rel_20" in out.columns:
        extra_m &= out["volume_rel_20"].fillna(0) >= float(extra["min_vol_rel"])
    if extra.get("max_close_loc") is not None and "close_loc" in out.columns:
        extra_m &= out["close_loc"].fillna(999) <= float(extra["max_close_loc"])
    if extra.get("min_close_loc") is not None and "close_loc" in out.columns:
        extra_m &= out["close_loc"].fillna(-1) >= float(extra["min_close_loc"])
    out = out.loc[extra_m].copy()
    out = select_same_day_rs(out, rs_col="rs_spy_126d", max_per_day=1)
    return apply_friction(out, float(extra.get("friction", FRIC)))


def row(name: str, df: pd.DataFrame) -> dict:
    s = _summarize(df, gain_col="gain_pct_net")
    return {"name": name, **s}


def year_ok(df: pd.DataFrame) -> str:
    y = summarize_by_year(df, gain_col="gain_pct_net")
    if y.empty:
        return ""
    parts = []
    for _, r in y.iterrows():
        if str(r["bucket"]) == "FULL":
            continue
        e = r["expectancy_pct"]
        parts.append(f"{r['bucket']}:{e}")
    return " ".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, type=Path)
    ap.add_argument("--outdir", type=Path, default=ROOT / "reports" / "ascending_channels")
    args = ap.parse_args()

    raw = pd.read_csv(args.raw)
    raw["buy_date"] = pd.to_datetime(raw["buy_date"])
    if "channel_end" in raw.columns:
        raw["channel_end"] = pd.to_datetime(raw["channel_end"])

    rows = []
    base = keeper(raw, {})
    r = row("BASE in+span10+b025+RStop1", base)
    r["years"] = year_ok(base)
    rows.append(r)

    for rsi in (40, 45, 50, 55, 60, 70):
        d = keeper(raw, {"max_rsi": rsi})
        rec = row(f"max_rsi {rsi}", d)
        rec["years"] = year_ok(d)
        rows.append(rec)
    for pos in (0.15, 0.25, 0.40, 0.60):
        d = keeper(raw, {"max_pos": pos})
        rec = row(f"max_pos {pos}", d)
        rec["years"] = year_ok(d)
        rows.append(rec)
    for w in (6, 8, 12, 16, 26):
        d = keeper(raw, {"min_wait_bars": w})
        rec = row(f"min_wait_bars {w}", d)
        rec["years"] = year_ok(d)
        rows.append(rec)
    for b in (None, 0.0, 0.25, 0.5):
        d = keeper(raw, {"beyond": b})
        rec = row(f"beyond {b}", d)
        rec["years"] = year_ok(d)
        rows.append(rec)
    d = keeper(raw, {"squeeze_rising": True})
    rec = row("squeeze_rising", d)
    rec["years"] = year_ok(d)
    rows.append(rec)
    for loc in (0.3, 0.4, 0.5):
        d = keeper(raw, {"max_close_loc": loc})
        rec = row(f"max_close_loc {loc}", d)
        rec["years"] = year_ok(d)
        rows.append(rec)
    for loc in (0.5, 0.6):
        d = keeper(raw, {"min_close_loc": loc})
        rec = row(f"min_close_loc {loc}", d)
        rec["years"] = year_ok(d)
        rows.append(rec)
    d = keeper(raw, {"max_rsi": 50, "max_pos": 0.25})
    rec = row("rsi50+pos025", d)
    rec["years"] = year_ok(d)
    rows.append(rec)
    d = keeper(raw, {"max_rsi": 55, "min_wait_bars": 8})
    rec = row("rsi55+wait8", d)
    rec["years"] = year_ok(d)
    rows.append(rec)

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
    dest = args.outdir / f"{args.raw.stem}_filter_ab.csv"
    out.to_csv(dest, index=False)
    print("wrote", dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
