"""Unit tests for channel-touch TV report metadata helpers."""
from pathlib import Path

from scripts.research.generate_channel_touch_tv_report import (
    _parse_summary_txt,
    _split_param_rows,
    _summary_path_for_trades,
    build_run_meta,
)


def test_summary_path_for_trades():
    p = Path("reports/ascending_channels/channel_touch_trades_20260827_223039.csv")
    assert _summary_path_for_trades(p).name == "channel_touch_trades_summary_20260827_223039.txt"


def test_parse_and_split_summary(tmp_path):
    summary = tmp_path / "channel_touch_trades_summary_test.txt"
    summary.write_text(
        "\n".join(
            [
                "Ascending channel bottom-touch long backtest",
                "entry_touch>=3",
                "error_pct=1.2",
                "min_rally_pct=4.0",
                "provider=ALPACA",
                "timeframe=1d",
                "fallback_provider=IB",
                "merge_mode=prefix",
                "n_trades=10",
                "expectancy_pct=2.5",
                "Exit: hard stop",
            ]
        ),
        encoding="utf-8",
    )
    kv, config_lines, notes = _parse_summary_txt(summary)
    assert kv["fallback_provider"] == "IB"
    assert kv["n_trades"] == "10"
    assert kv.get("entry_touch") == "3"
    assert "1.2" in kv.get("error_pct", "")
    assert notes[0].startswith("Exit:")

    det, bt, data = _split_param_rows(kv, config_lines)
    assert any(r["key"] == "fallback_provider" for r in data)
    assert kv["n_trades"] not in {r["key"] for r in bt}


def test_build_run_meta_has_git(tmp_path):
    trades = tmp_path / "channel_touch_trades_20260827_223039.csv"
    trades.write_text("stock,buy_date,sell_date,buy_price,sell_price,gain_pct\n", encoding="utf-8")
    meta = build_run_meta(trades)
    assert "branch" in meta["git"]
    assert meta["detector"]["script"].endswith("find_ascending_channels.py")
