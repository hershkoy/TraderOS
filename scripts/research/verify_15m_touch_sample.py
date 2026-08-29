"""Replay trades_for_symbol on a 15m keeper sample and check fills."""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

from backtest_channel_touch_trades import (  # noqa: E402
    _l3_rail_touch,
    summarize_by_year,
    trades_for_symbol,
)
from utils.data.ohlcv_loader import load_ohlcv_many
from utils.research.channel_touch_scale import PRESET_15M


def _bar_index(df: pd.DataFrame, ts: pd.Timestamp) -> int:
    idx = df.index
    if ts in idx:
        loc = idx.get_loc(ts)
        return int(loc.start if isinstance(loc, slice) else loc)
    later = idx[idx >= ts]
    if len(later) == 0:
        raise KeyError(ts)
    return int(idx.get_loc(later[0]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keeper", required=True, type=Path)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=29)
    ap.add_argument("--min-touch", type=int, default=3)
    args = ap.parse_args()

    k = pd.read_csv(args.keeper)
    k["buy_date"] = pd.to_datetime(k["buy_date"])
    print("keeper n", len(k))
    if "touch_num" in k.columns:
        print("touch_num", k["touch_num"].value_counts().to_dict())
    if "channel_pos" in k.columns:
        print("median pos", float(k["channel_pos"].median()), "pos>0.35", int((k["channel_pos"] > 0.35).sum()))
    if "wait_bars" in k.columns:
        print("wait_bars min/med", int(k["wait_bars"].min()), float(k["wait_bars"].median()))
    gain = "gain_pct_net" if "gain_pct_net" in k.columns else "gain_pct"
    print(summarize_by_year(k, gain_col=gain).to_string(index=False))

    sample = k.sample(n=min(int(args.n), len(k)), random_state=int(args.seed))
    symbols = sorted(set(sample["stock"].astype(str).str.upper().tolist()))
    panels = load_ohlcv_many(
        symbols,
        timeframe="15m",
        provider="IB",
        start=datetime(2018, 11, 1),
        end=datetime(2025, 12, 2),
        use_cache=True,
    )
    scan_kw = dict(
        entry_mode="l3_touch",
        entry_touch=int(args.min_touch),
        pivot_len=int(PRESET_15M["pivot_len"]),
        min_bars_apart=int(PRESET_15M["min_bars_apart"]),
        max_low_pivots=int(PRESET_15M["max_low_pivots"]),
        error_pct=float(PRESET_15M["error_pct"]),
        min_rally_pct=float(PRESET_15M["min_rally_pct"]),
        min_pullback_pct=float(PRESET_15M["min_pullback_pct"]),
        min_total_rise_pct=float(PRESET_15M["min_total_rise_pct"]),
        flat_pct=float(PRESET_15M["flat_pct"]),
        window_bars=int(PRESET_15M["window_bars"]),
        window_step_bars=int(PRESET_15M["window_step_bars"]),
        include_time=True,
        entry_features=False,
        squeeze_adaptive=False,
        stop_pct=float(PRESET_15M["stop_pct"]),
        trail_pct=float(PRESET_15M["trail_pct"]),
        atr_stop_mult=float(PRESET_15M["atr_stop_mult"]),
        stop_pct_floor=float(PRESET_15M["stop_pct_floor"]),
        stop_pct_ceil=float(PRESET_15M["stop_pct_ceil"]),
        entry_slip_pct=0.001,
        max_l3_wait_bars=252,
        min_l3_wait_bars=6,
    )
    replay: dict = {}
    for sym in symbols:
        df = panels.get(sym)
        if df is None or df.empty:
            replay[sym] = []
            continue
        replay[sym] = trades_for_symbol(sym, df, **scan_kw)

    bad = 0
    for _, row in sample.iterrows():
        sym = str(row["stock"]).upper()
        df = panels.get(sym)
        buy = pd.Timestamp(row["buy_time"]) if "buy_time" in row and pd.notna(row["buy_time"]) else pd.Timestamp(row["buy_date"])
        tnum = int(row["touch_num"]) if "touch_num" in row and pd.notna(row["touch_num"]) else -1
        wait = int(row["wait_bars"]) if "wait_bars" in row and pd.notna(row["wait_bars"]) else -1
        pos = float(row["channel_pos"]) if pd.notna(row.get("channel_pos")) else 999.0
        in_range = False
        rail_ok = False
        if df is not None and not df.empty:
            try:
                i = _bar_index(df, buy)
                hi, lo, cl = float(df["high"].iloc[i]), float(df["low"].iloc[i]), float(df["close"].iloc[i])
                px = float(row["buy_price"])
                in_range = lo <= px <= hi or abs(px - lo) / max(px, 1e-9) < 0.002
                slip = 0.001
                sup = px / (1.0 + slip)
                rail_ok = _l3_rail_touch(hi, lo, cl, sup, float(PRESET_15M["error_pct"]))
            except Exception:
                i = -1
        replayed = replay.get(sym) or []
        times = {pd.Timestamp(t.get("buy_time") or t["buy_date"]) for t in replayed}
        replay_hit = buy in times or any(abs((t - buy).total_seconds()) < 60 for t in times)
        print(
            f"{sym} buy={buy} touch={tnum} wait={wait} pos={pos:.3f} "
            f"in_range={in_range} rail_ok={rail_ok} replay_hit={replay_hit}"
        )
        fail = (
            (not in_range)
            or (not rail_ok)
            or (not replay_hit)
            or (tnum >= 0 and tnum < int(args.min_touch))
            or wait < 6
            or pos > 0.5
        )
        if fail:
            print("  FAIL")
            bad += 1
    print("verify_failures", bad, "of", len(sample))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
