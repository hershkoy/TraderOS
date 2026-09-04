"""Year splits + OHLCV verification that l3 fills tag support from above."""
from __future__ import annotations

import sys
from datetime import datetime
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
    _line_at,
    _l3_rail_touch,
)
from find_ascending_channels import find_h2_l3_setups_windowed
from utils.data.ohlcv_loader import load_ohlcv_many
from utils.research.report_paths import resolve_artifact

RAW = resolve_artifact("channel_touch_trades_raw_20260828_192557.csv")
KEEPER = resolve_artifact("channel_touch_trades_20260828_192557.csv")


def pack(raw: pd.DataFrame, extra_m: pd.Series) -> pd.DataFrame:
    out = raw.loc[extra_m].copy()
    out = filter_trades(
        out,
        require_in_channel=True,
        max_channel_span_days=365,
        max_beyond_width=0.25,
    )
    out = select_same_day_rs(out, rs_col="rs_spy_126d", max_per_day=1)
    return apply_friction(out, 0.25)


def main() -> None:
    raw = pd.read_csv(RAW)
    raw["buy_date"] = pd.to_datetime(raw["buy_date"])
    base_m = pd.Series(True, index=raw.index)
    variants = {
        "base": base_m,
        "rsi50": raw["rsi_14"].fillna(999) <= 50,
        "rsi60": raw["rsi_14"].fillna(999) <= 60,
        "squeeze_rising": raw["squeeze_mom_rising"].fillna(0).astype(float) >= 1,
        "min_wait8": (raw["buy_date"] - pd.to_datetime(raw["channel_end"])).dt.days >= 8,
        "min_wait8_rsi50": ((raw["buy_date"] - pd.to_datetime(raw["channel_end"])).dt.days >= 8)
        & (raw["rsi_14"].fillna(999) <= 50),
    }
    for name, m in variants.items():
        df = pack(raw, m)
        print("\n====", name, "n", len(df), "====")
        y = summarize_by_year(df, gain_col="gain_pct_net")
        print(y.to_string(index=False))
        # drop top 3 winners
        if not df.empty and "gain_pct_net" in df.columns:
            drop = df.sort_values("gain_pct_net", ascending=False).iloc[3:]
            from backtest_channel_touch_trades import _summarize

            s = _summarize(drop, gain_col="gain_pct_net")
            print("drop-top-3 n", s["n_trades"], "E", s["expectancy_pct"], "PF", s["profit_factor"])

    k = pd.read_csv(KEEPER)
    sample = k.sample(n=min(8, len(k)), random_state=28)
    symbols = sorted(set(sample["stock"].astype(str).str.upper()))
    print("\nVERIFY symbols", symbols)
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
        if df is None or df.empty:
            print(sym, buy.date(), "NO PANEL")
            bad += 1
            continue
        idx = df.index[df.index >= buy]
        if len(idx) == 0:
            print(sym, buy.date(), "NO BAR")
            bad += 1
            continue
        i = df.index.get_loc(idx[0])
        setups = find_h2_l3_setups_windowed(df, window_bars=504, step_bars=252, pivot_len=15)
        h2 = pd.Timestamp(row["channel_end"])
        match = [
            c
            for c in setups
            if c["start_date"] == str(row["channel_start"]) and c["h2_date"] == h2.strftime("%Y-%m-%d")
        ]
        if not match:
            print(sym, buy.date(), "NO SETUP MATCH L1", row["channel_start"], "H2", h2.date())
            bad += 1
            continue
        ch = match[0]
        sx0, sy0, sslope = int(ch["support_x0"]), float(ch["support_y0"]), float(ch["support_slope"])
        sup = _line_at(sy0, sx0, sslope, i)
        hi, lo, cl = float(df["high"].iloc[i]), float(df["low"].iloc[i]), float(df["close"].iloc[i])
        ok = _l3_rail_touch(hi, lo, cl, sup, 1.2)
        pos = float(row["channel_pos"])
        print(
            f"{sym} {buy.date()} H2 {h2.date()} pos={pos:.3f} buy={row['buy_price']} "
            f"OHLC={df['open'].iloc[i]:.3f}/{hi:.3f}/{lo:.3f}/{cl:.3f} sup={sup:.3f} rail_ok={ok}"
        )
        if not ok or pos > 0.35:
            bad += 1
            print("  FAIL strategy check")
    print("verify_failures", bad, "of", len(sample))


if __name__ == "__main__":
    main()
