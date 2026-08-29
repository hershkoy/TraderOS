"""Unit tests for channel-touch TV report metadata helpers."""
from pathlib import Path

import pandas as pd

from scripts.research.generate_channel_touch_tv_report import (
    _parse_summary_txt,
    _split_param_rows,
    _summary_path_for_trades,
    build_run_meta,
)


def test_summary_path_for_trades(tmp_path):
    trades = tmp_path / "channel_touch_trades_20260827_223039.csv"
    summary = tmp_path / "channel_touch_trades_summary_20260827_223039.txt"
    trades.write_text("stock\n", encoding="utf-8")
    summary.write_text("n_trades=1\n", encoding="utf-8")
    assert _summary_path_for_trades(trades) == summary


def test_summary_path_for_raw_trades(tmp_path):
    raw = tmp_path / "channel_touch_trades_raw_20260827_223039.csv"
    summary = tmp_path / "channel_touch_trades_summary_20260827_223039.txt"
    raw.write_text("stock\n", encoding="utf-8")
    summary.write_text("n_trades=1\n", encoding="utf-8")
    assert _summary_path_for_trades(raw) == summary


def test_raw_sibling_csv(tmp_path):
    filtered = tmp_path / "channel_touch_trades_20260828_194314.csv"
    raw = tmp_path / "channel_touch_trades_raw_20260828_194314.csv"
    filtered.write_text("stock\n", encoding="utf-8")
    raw.write_text("stock\n", encoding="utf-8")
    from scripts.research.generate_channel_touch_tv_report import _raw_sibling_csv

    assert _raw_sibling_csv(filtered) == raw
    assert _raw_sibling_csv(raw) == raw


def test_filter_kwargs_from_summary_multi_kv(tmp_path):
    summary = tmp_path / "channel_touch_trades_summary_test.txt"
    summary.write_text(
        "\n".join(
            [
                "require_in_channel=True max_channel_span_days=365.0",
                "max_beyond_width=0.25 max_rsi=50.0 min_l3_wait_bars=6",
                "min_adv=None",
            ]
        ),
        encoding="utf-8",
    )
    from scripts.research.generate_channel_touch_tv_report import filter_kwargs_from_summary

    kw = filter_kwargs_from_summary(summary)
    assert kw["require_in_channel"] is True
    assert kw["max_channel_span_days"] == 365.0
    assert kw["max_beyond_width"] == 0.25
    assert kw["max_rsi"] == 50.0
    assert kw["min_adv"] is None


def test_load_trades_for_report_skips_rs_cap(tmp_path):
    """UI max/day needs same-day extras; do not embed the already-capped CSV."""
    from scripts.research.generate_channel_touch_tv_report import load_trades_for_report

    cols = (
        "stock,buy_date,sell_date,buy_price,sell_price,gain_pct,hold_days,"
        "channel_pos,channel_span_days,max_beyond_width,rsi_14,rs_spy_126d\n"
    )
    # Two same-day fills; filtered CSV kept only the RS winner.
    raw = tmp_path / "channel_touch_trades_raw_20260828_000001.csv"
    filtered = tmp_path / "channel_touch_trades_20260828_000001.csv"
    summary = tmp_path / "channel_touch_trades_summary_20260828_000001.txt"
    raw.write_text(
        cols
        + "AAA,2020-01-02,2020-01-10,10,11,10,5,0.5,100,0.1,40,5.0\n"
        + "BBB,2020-01-02,2020-01-10,10,11,10,5,0.5,100,0.1,40,1.0\n"
        + "CCC,2020-01-03,2020-01-10,10,11,10,5,1.5,100,0.1,40,2.0\n",
        encoding="utf-8",
    )
    filtered.write_text(
        cols + "AAA,2020-01-02,2020-01-10,10,11,10,5,0.5,100,0.1,40,5.0\n",
        encoding="utf-8",
    )
    summary.write_text(
        "require_in_channel=True max_channel_span_days=365.0\nmax_beyond_width=0.25 max_rsi=50.0\n",
        encoding="utf-8",
    )
    df = load_trades_for_report(filtered)
    assert len(df) == 2
    assert set(df["stock"]) == {"AAA", "BBB"}


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


def test_spy_to_raw_keeps_window():
    from scripts.research.generate_channel_touch_tv_report import spy_to_raw

    idx = pd.date_range("2019-02-14", "2019-02-20", freq="B")
    s = pd.Series([100.0 + i for i in range(len(idx))], index=idx)
    rows = spy_to_raw(s, pd.Timestamp("2019-02-14"), pd.Timestamp("2019-02-20"))
    assert rows[0]["x"] == "2019-02-14"
    assert rows[-1]["x"] == "2019-02-20"


def test_build_run_meta_has_git(tmp_path):
    trades = tmp_path / "channel_touch_trades_20260827_223039.csv"
    trades.write_text("stock,buy_date,sell_date,buy_price,sell_price,gain_pct\n", encoding="utf-8")
    meta = build_run_meta(trades)
    assert "branch" in meta["git"]
    assert meta["detector"]["script"].endswith("find_ascending_channels.py")
