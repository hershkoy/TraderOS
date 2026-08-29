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


def test_trades_to_raw_emits_null_rs_and_ord():
    from scripts.research.generate_channel_touch_tv_report import trades_to_raw

    df = pd.DataFrame(
        {
            "stock": ["AAA", "BBB"],
            "buy_date": ["2020-01-02", "2020-01-02"],
            "sell_date": ["2020-01-10", "2020-01-11"],
            "buy_price": [10.0, 10.0],
            "sell_price": [11.0, 11.0],
            "gain_pct": [1.0, 2.0],
            "hold_days": [5, 6],
            "rs_spy_126d": [5.0, float("nan")],
        }
    )
    rows = trades_to_raw(df)
    by_sym = {r["symbol"]: r for r in rows}
    assert by_sym["AAA"]["rs"] == 5.0
    assert by_sym["BBB"]["rs"] is None
    assert by_sym["AAA"]["ord"] == 0
    assert by_sym["BBB"]["ord"] == 1


def test_trades_to_raw_resist_break_from_string_false():
    from scripts.research.generate_channel_touch_tv_report import trades_to_raw

    df = pd.DataFrame(
        {
            "stock": ["AAA", "BBB"],
            "buy_date": ["2020-01-02", "2020-01-03"],
            "sell_date": ["2020-01-10", "2020-01-11"],
            "buy_price": [10.0, 10.0],
            "sell_price": [11.0, 11.0],
            "gain_pct": [1.0, 2.0],
            "hold_days": [5, 6],
            "resist_break": ["False", "True"],
        }
    )
    rows = trades_to_raw(df)
    by_sym = {r["symbol"]: r for r in rows}
    assert by_sym["AAA"]["resist_break"] is False
    assert by_sym["BBB"]["resist_break"] is True


def test_filter_max_per_day_matches_python_rs_when_rs_missing():
    """NaN RS must not scramble same-day picks (old JS -Infinity comparator)."""
    from scripts.research.backtest_channel_touch_trades import select_same_day_rs
    from scripts.research.generate_channel_touch_tv_report import (
        filter_max_per_day_raw,
        trades_to_raw,
    )

    df = pd.DataFrame(
        {
            "stock": ["LOSE", "WIN", "ONLY", "NA1", "NA2"],
            "buy_date": pd.to_datetime(
                ["2020-01-02", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-04"]
            ),
            "sell_date": pd.to_datetime(
                ["2020-01-10", "2020-01-10", "2020-01-10", "2020-01-12", "2020-01-11"]
            ),
            "buy_price": [10.0] * 5,
            "sell_price": [11.0] * 5,
            "gain_pct": [1.0, 9.0, 2.0, 3.0, 4.0],
            "hold_days": [5] * 5,
            "rs_spy_126d": [1.0, 5.0, 2.0, float("nan"), float("nan")],
        }
    )
    py = select_same_day_rs(df, rs_col="rs_spy_126d", max_per_day=1)
    js = filter_max_per_day_raw(trades_to_raw(df), 1)
    py_keys = set(zip(py["stock"].astype(str), pd.to_datetime(py["buy_date"]).dt.strftime("%Y-%m-%d")))
    js_keys = set((t["symbol"], t["buy"]) for t in js)
    assert py_keys == js_keys
    assert {t["symbol"] for t in js} == {"WIN", "ONLY", "NA1"}
