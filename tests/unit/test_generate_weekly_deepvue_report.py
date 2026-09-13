"""Unit tests for the weekly Deepvue HTML report wrapper."""
from pathlib import Path

import pandas as pd

from scripts.research.generate_weekly_deepvue_report import (
    DETECTOR_META,
    latest_weekly_trades,
    write_weekly_summary,
)


def test_latest_weekly_trades_picks_newest_date_folder(tmp_path: Path) -> None:
    older = tmp_path / "2026-09-11"
    older.mkdir()
    (older / "weekly_deepvue_trades.csv").write_text("stock\n", encoding="utf-8")
    newer = tmp_path / "2026-09-12"
    newer.mkdir()
    newest = newer / "weekly_deepvue_trades.csv"
    newest.write_text("stock\n", encoding="utf-8")
    assert latest_weekly_trades(tmp_path) == newest


def test_write_weekly_summary_marks_no_promote(tmp_path: Path) -> None:
    trades = tmp_path / "weekly_deepvue_trades.csv"
    trades.write_text("stock\n", encoding="utf-8")
    path = write_weekly_summary(trades)
    text = path.read_text(encoding="utf-8")
    assert "No promote" in text
    assert "next_session_open_mid" in text
    assert DETECTOR_META["script"].endswith("backtest_weekly_deepvue.py")


def test_weekly_deepvue_csv_shape_renders_without_sell_date() -> None:
    from scripts.research.generate_channel_touch_tv_report import trades_to_raw

    df = pd.DataFrame(
        {
            "stock": ["HOOD"],
            "sector": ["Financials"],
            "signal_date": ["2024-10-25"],
            "buy_date": ["2024-10-28"],
            "buy_time": ["2024-10-28 13:30"],
            "buy_price": [25.0],
            "sell_time": ["2024-10-28 18:00"],
            "sell_price": [22.1],
            "gain_pct": [-11.6],
            "exit_reason": ["hard_stop"],
            "rmv": [40.0],
            "rs_1m": [90.0],
            "rs_3m": [88.0],
            "focus_rank": [1],
        }
    )
    row = trades_to_raw(df)[0]
    assert row["symbol"] == "HOOD"
    assert row["buy"] == "2024-10-28"
    assert row["sell"] == "2024-10-28"
    assert row["buy_at"] == "2024-10-28 09:30"
    assert row["reason"] == "hard_stop"
    assert row.get("ch") is None
