"""Verify l3 fills on the optimized keeper CSV; match setup by support nearest buy price."""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

from backtest_channel_touch_trades import (  # noqa: E402
    _line_at,
    _l3_rail_touch,
    summarize_by_year,
)
from find_ascending_channels import find_h2_l3_setups_windowed
from utils.data.ohlcv_loader import load_ohlcv_many

KEEPER = ROOT / "reports" / "ascending_channels" / "channel_touch_trades_20260828_194314.csv"


def main() -> None:
    k = pd.read_csv(KEEPER)
    k["buy_date"] = pd.to_datetime(k["buy_date"])
    print("keeper n", len(k), "max rsi", k["rsi_14"].max(), "median pos", k["channel_pos"].median())
    print("pos>0.2", int((k["channel_pos"] > 0.2).sum()), "pos<0", int((k["channel_pos"] < 0).sum()))
    y = summarize_by_year(k, gain_col="gain_pct_net")
    print(y.to_string(index=False))
    print("WTFC 2019-07", k[(k["stock"] == "WTFC") & (k["buy_date"] == "2019-07-16")])
    print("EYE 2025", k[(k["stock"] == "EYE") & (k["buy_date"] >= "2025-01-01") & (k["buy_date"] <= "2025-06-01")])

    sample = k.sample(n=min(8, len(k)), random_state=7)
    symbols = sorted(set(sample["stock"].astype(str).str.upper().tolist()))
    panels = load_ohlcv_many(
        symbols,
        timeframe="1d",
        provider="ALPACA",
        start=datetime(2018, 11, 1),
        end=datetime(2026, 8, 27),
        use_cache=True,
        fallback_provider="IB",
        merge_mode="prefix",
    )
    bad = 0
    for _, row in sample.iterrows():
        sym = str(row["stock"]).upper()
        df = panels.get(sym)
        buy = pd.Timestamp(row["buy_date"])
        h2 = pd.Timestamp(row["channel_end"])
        i = df.index.get_loc(df.index[df.index >= buy][0])
        setups = find_h2_l3_setups_windowed(df, window_bars=504, step_bars=252, pivot_len=15)
        cands = [c for c in setups if c["h2_date"] == h2.strftime("%Y-%m-%d")]
        if not cands:
            cands = [c for c in setups if c["start_date"] == str(row["channel_start"])]
        best = None
        best_err = 1e9
        for c in cands:
            sup = _line_at(float(c["support_y0"]), int(c["support_x0"]), float(c["support_slope"]), i)
            err = abs(sup - float(row["buy_price"]))
            if err < best_err:
                best_err, best = err, c
        if best is None:
            print(sym, buy.date(), "NO SETUP")
            bad += 1
            continue
        sup = _line_at(float(best["support_y0"]), int(best["support_x0"]), float(best["support_slope"]), i)
        hi, lo, cl = float(df["high"].iloc[i]), float(df["low"].iloc[i]), float(df["close"].iloc[i])
        ok = _l3_rail_touch(hi, lo, cl, sup, 1.2)
        wait_bars = i - int(best["h2_idx"])
        print(
            f"{sym} buy {buy.date()} H2 {h2.date()} wait_bars={wait_bars} pos={row['channel_pos']} "
            f"rsi={row['rsi_14']:.1f} buy={row['buy_price']} sup={sup:.3f} rail_ok={ok}"
        )
        if (not ok) or float(row["channel_pos"]) > 0.35 or float(row["rsi_14"]) > 50.01 or wait_bars < 6:
            print("  FAIL")
            bad += 1
    print("verify_failures", bad, "of", len(sample))


if __name__ == "__main__":
    main()
