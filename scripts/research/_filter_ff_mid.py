"""Filter unfiltered panel mid book to FF>=0.20 (no Theta re-scan)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from utils.research.forward_factor import FFTrade, book_stats, trades_to_frame

SRC = (
    ROOT
    / "reports"
    / "research"
    / "forward_factor"
    / "2026-10-02_panel"
    / "SPY_QQQ_AAPL_panel_60_90_mid_ff-999_trades.csv"
)
OUT = ROOT / "reports" / "research" / "forward_factor" / "2026-10-02_panel"


def main() -> int:
    df = pd.read_csv(SRC)
    print("n_unfiltered", len(df))
    print(df["ff"].describe())
    f = df[df["ff"] >= 0.20].copy()
    print("n_ff>=0.20", len(f))
    print(f[["symbol", "entry_date", "ff", "ret_pct", "pnl"]].to_string(index=False))
    trades = []
    for _, r in f.iterrows():
        trades.append(
            FFTrade(
                symbol=str(r["symbol"]),
                entry_date=pd.Timestamp(r["entry_date"]).date(),
                exit_date=pd.Timestamp(r["exit_date"]).date(),
                front=pd.Timestamp(r["front"]).date(),
                back=pd.Timestamp(r["back"]).date(),
                strike=float(r["strike"]),
                spot_entry=float(r["spot_entry"]),
                front_iv=float(r["front_iv"]),
                back_iv=float(r["back_iv"]),
                fwd_iv=float(r["fwd_iv"]),
                ff=float(r["ff"]),
                mark_mode="mid",
                entry_debit=float(r["entry_debit"]),
                exit_credit=float(r["exit_credit"]),
                pnl=float(r["pnl"]),
                ret_pct=float(r["ret_pct"]),
                hold_days=int(r["hold_days"]),
                commission=float(r["commission"]),
            )
        )
    stats = book_stats(trades)
    print("STATS", stats)
    trades_to_frame(trades).to_csv(OUT / "panel_mid_ff0.2_from_unfiltered.csv", index=False)
    with (OUT / "panel_mid_ff0.2_from_unfiltered_summary.json").open("w", encoding="utf-8") as fh:
        json.dump({"stats": stats, "n": len(trades)}, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
