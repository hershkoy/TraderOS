"""Smoke the Deepvue x last-15m overlay CLI on tiny temp CSVs."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.research.overlay_last_15m_deepvue_and import main


def test_overlay_writes_summary(tmp_path: Path) -> None:
    last15m = pd.DataFrame(
        [
            {
                "stock": "AAA",
                "buy_date": "2021-06-22",
                "buy_time": "2021-06-22 19:45:00",
                "gain_pct": -6.29,
                "exit_reason": "hard_stop",
            },
            {
                "stock": "BBB",
                "buy_date": "2024-03-04",
                "buy_time": "2024-03-04 19:45:00",
                "gain_pct": 2.0,
                "exit_reason": "trail_stop",
            },
        ]
    )
    signals = pd.DataFrame(
        [{"stock": "AAA", "signal_date": "2021-06-18", "focus_rank": 1}]
    )
    deepvue = pd.DataFrame(
        [
            {
                "stock": "AAA",
                "signal_date": "2021-06-18",
                "buy_date": "2021-06-21",
                "gain_pct": 86.2,
                "exit_reason": "ma_2close",
            }
        ]
    )
    trades = tmp_path / "l15.csv"
    sig = tmp_path / "sig.csv"
    dv = tmp_path / "dv.csv"
    last15m.to_csv(trades, index=False)
    signals.to_csv(sig, index=False)
    deepvue.to_csv(dv, index=False)
    outdir = tmp_path / "out"
    rc = main(
        [
            "--trades",
            str(trades),
            "--signals",
            str(sig),
            "--deepvue",
            str(dv),
            "--outdir",
            str(outdir),
        ]
    )
    assert rc == 0
    summary = pd.read_csv(outdir / "last_15m_deepvue_and_summary.csv")
    labels = set(summary["label"].astype(str))
    assert "L15_and_week" in labels
    assert "DV_and_l15_week" in labels
    week = summary.loc[summary["label"] == "L15_and_week"].iloc[0]
    assert int(week["n"]) == 1
    assert int(week["n_l15_and_week"]) == 1
