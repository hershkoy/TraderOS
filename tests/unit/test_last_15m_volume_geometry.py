"""Same-list overlays: AMPL-like early exit and TARS-like channel_pos skip."""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from utils.research.last_15m_volume_geometry import (
    apply_early_exit,
    delayed_second_close,
    skip_mask,
    splice_calendar_year,
    winner_cut_early_exit,
    winner_cut_skip,
)
from utils.research.realistic_purchaser import bar_mid

ET = ZoneInfo("America/New_York")


def _bar(ts, o, h, l, c, v=100.0):
    return {"ts": ts, "open": o, "high": h, "low": l, "close": c, "volume": v}


def _et(y, m, d, hh, mm):
    return datetime(y, m, d, hh, mm, tzinfo=ET)


def test_skip_tars_like_channel_pos_and_vst_formation():
    df = pd.DataFrame(
        [
            {"stock": "AMPL", "formation_beyond_width": 0.04, "channel_pos": 1.03, "gain_pct": -21.3},
            {"stock": "VST", "formation_beyond_width": 1.82, "channel_pos": 1.007, "gain_pct": -19.6},
            {"stock": "TARS", "formation_beyond_width": 0.37, "channel_pos": 1.492, "gain_pct": -19.4},
            {"stock": "WIN", "formation_beyond_width": 0.10, "channel_pos": 1.10, "gain_pct": 8.0},
        ]
    )
    form = skip_mask(df, max_formation_beyond=0.25)
    assert list(df.loc[form, "stock"]) == ["AMPL", "WIN"]
    pos125 = skip_mask(df, max_channel_pos=1.25)
    assert "TARS" not in set(df.loc[pos125, "stock"])
    assert "WIN" in set(df.loc[pos125, "stock"])
    pos150 = skip_mask(df, max_channel_pos=1.50)
    assert "TARS" in set(df.loc[pos150, "stock"])


def test_winner_cut_skip_marks_blunt_when_mostly_winners():
    df = pd.DataFrame(
        {
            "gain_pct": [5.0, 4.0, 3.0, -1.0],
            "keep": [False, False, False, True],
        }
    )
    cut = winner_cut_skip(df["gain_pct"], df["keep"])
    assert cut["winners_dropped"] == 3
    assert cut["blunt"] is True


def test_ampl_like_evening_doji_exits_before_crash():
    fill = _et(2024, 2, 9, 15, 45)
    # Fill session: large bull. Next: gapped doji. Next: strong bear into body.
    # Crash two sessions later (original ATR sell).
    by_day = {
        fill.date(): [
            _bar(_et(2024, 2, 9, 9, 30), 13.00, 13.20, 12.95, 13.10, 80.0),
            _bar(fill, 13.10, 14.30, 13.05, 14.20, 120.0),
        ],
        _et(2024, 2, 12, 15, 45).date(): [
            _bar(_et(2024, 2, 12, 9, 30), 14.30, 14.35, 14.20, 14.28, 90.0),
            _bar(_et(2024, 2, 12, 15, 45), 14.28, 14.40, 14.15, 14.28, 90.0),
        ],
        _et(2024, 2, 13, 15, 45).date(): [
            _bar(_et(2024, 2, 13, 9, 30), 14.20, 14.25, 13.80, 13.90, 100.0),
            _bar(_et(2024, 2, 13, 15, 45), 13.90, 13.95, 13.40, 13.50, 110.0),
        ],
        _et(2024, 2, 14, 9, 30).date(): [
            _bar(_et(2024, 2, 14, 9, 30), 13.45, 13.55, 13.35, 13.40, 80.0),
            _bar(_et(2024, 2, 14, 15, 45), 13.40, 13.42, 13.20, 13.22, 80.0),
        ],
        _et(2024, 2, 21, 9, 45).date(): [
            _bar(_et(2024, 2, 21, 9, 30), 11.40, 11.50, 11.00, 11.10, 200.0),
            _bar(_et(2024, 2, 21, 9, 45), 11.10, 11.30, 11.00, 11.15, 200.0),
        ],
    }
    row = pd.Series(
        {
            "buy_price": 14.17,
            "buy_time": fill,
            "buy_date": "2024-02-09",
            "sell_price": 11.15,
            "sell_time": _et(2024, 2, 21, 9, 45),
            "sell_date": "2024-02-21",
            "exit_reason": "hard_stop",
        }
    )
    got = apply_early_exit(row, by_day, require_gap=True)
    assert got.doji_day.isoformat() == "2024-02-13"
    assert got.used_overlay is True
    assert got.exit_reason == "doji_star"
    assert got.sell_ts == _et(2024, 2, 14, 9, 30)
    assert got.sell_px == pytest.approx(bar_mid(13.55, 13.35))
    assert got.gain_pct > -21.0


def test_early_exit_keeps_atr_if_overlay_later():
    fill = _et(2024, 2, 9, 15, 45)
    by_day = {
        fill.date(): [_bar(fill, 13.0, 14.2, 13.0, 14.1, 100.0)],
        _et(2024, 2, 12, 9, 30).date(): [
            _bar(_et(2024, 2, 12, 9, 30), 13.5, 13.6, 13.4, 13.5, 50.0),
        ],
    }
    row = pd.Series(
        {
            "buy_price": 14.17,
            "buy_time": fill,
            "buy_date": "2024-02-09",
            "sell_price": 13.50,
            "sell_time": _et(2024, 2, 12, 9, 30),
            "sell_date": "2024-02-12",
            "exit_reason": "hard_stop",
            "gain_pct": -12.345,
        }
    )
    got = apply_early_exit(row, by_day, require_gap=True)
    assert got.used_overlay is False
    assert got.sell_px == 13.50
    assert got.gain_pct == -12.345


def test_winner_cut_early_exit_flip():
    b = pd.Series([5.0, -8.0])
    o = pd.Series([-1.0, -3.0])
    cut = winner_cut_early_exit(b, o)
    assert cut["winners_flip_to_loser"] == 1
    assert cut["losers_flip_to_winner"] == 0
    assert cut["winner_gain_delta_sum"] == pytest.approx(-6.0)


def test_delayed_second_close_after_pullback():
    fill = _et(2023, 7, 20, 15, 45)
    # Fill extended. Next day dumps back through rail (~20.3). Then close back above
    # with buyer volume.
    by_day = {
        fill.date(): [_bar(fill, 22.0, 23.5, 21.8, 23.2, 100.0)],
        _et(2023, 7, 21, 15, 45).date(): [
            _bar(_et(2023, 7, 21, 9, 30), 22.0, 22.1, 19.5, 19.8, 80.0),
            _bar(_et(2023, 7, 21, 15, 45), 19.8, 20.0, 19.4, 19.6, 80.0),
        ],
        _et(2023, 7, 24, 15, 45).date(): [
            _bar(_et(2023, 7, 24, 9, 30), 19.8, 21.0, 19.7, 20.8, 90.0),
            _bar(_et(2023, 7, 24, 15, 45), 20.8, 21.2, 20.6, 21.0, 90.0),
        ],
        _et(2023, 7, 25, 9, 30).date(): [
            _bar(_et(2023, 7, 25, 9, 30), 20.9, 21.1, 20.5, 20.7, 70.0),
            _bar(_et(2023, 7, 25, 15, 45), 20.7, 20.8, 18.5, 18.8, 70.0),
        ],
        _et(2023, 7, 26, 9, 30).date(): [
            _bar(_et(2023, 7, 26, 9, 30), 18.7, 18.9, 18.4, 18.6, 60.0),
        ],
    }
    row = pd.Series(
        {
            "buy_price": 23.24,
            "buy_time": fill,
            "buy_date": "2023-07-20",
            "channel_width": 4.92,
            "channel_pos": 1.492,
            "slope_pct_per_bar": 0.05,
            "l1_price": 12.27,
            "atr_1d_pct": 4.0,
        }
    )
    got = delayed_second_close(row, by_day, max_wait_sessions=10, buy_pct_min=50.0)
    assert got is not None
    assert got["buy_date"] == "2023-07-24"
    assert got["buy_price"] == pytest.approx(bar_mid(21.2, 20.6))


def test_doji_only_does_not_use_seller():
    fill = _et(2024, 2, 9, 15, 45)
    by_day = {
        fill.date(): [_bar(fill, 13.0, 14.2, 13.0, 14.1, 200.0)],
        _et(2024, 2, 12, 15, 45).date(): [
            _bar(_et(2024, 2, 12, 15, 45), 14.0, 14.1, 13.5, 13.55, 150.0)
        ],
        _et(2024, 2, 13, 15, 45).date(): [
            _bar(_et(2024, 2, 13, 15, 45), 13.5, 13.6, 13.0, 13.05, 150.0)
        ],
        _et(2024, 2, 14, 9, 30).date(): [
            _bar(_et(2024, 2, 14, 9, 30), 13.0, 13.2, 12.9, 13.05, 80.0)
        ],
        _et(2024, 2, 21, 9, 45).date(): [
            _bar(_et(2024, 2, 21, 9, 45), 11.1, 11.3, 11.0, 11.15, 200.0)
        ],
    }
    row = pd.Series(
        {
            "buy_price": 14.17,
            "buy_time": fill,
            "buy_date": "2024-02-09",
            "sell_price": 11.15,
            "sell_time": _et(2024, 2, 21, 9, 45),
            "sell_date": "2024-02-21",
            "exit_reason": "hard_stop",
            "gain_pct": -21.3,
        }
    )
    seller = apply_early_exit(
        row, by_day, enable_doji=False, enable_seller=True, enable_failed_breakout=False
    )
    doji_only = apply_early_exit(
        row, by_day, enable_doji=True, enable_seller=False, enable_failed_breakout=False
    )
    assert seller.used_overlay is True
    assert seller.exit_reason == "seller_sessions"
    assert seller.seller_day is not None
    assert doji_only.used_overlay is False
    assert doji_only.seller_day is None
    assert doji_only.gain_pct == pytest.approx(-21.3)


def test_failed_breakout_exits_on_first_close_at_or_below_resist():
    fill = _et(2023, 7, 20, 15, 45)
    by_day = {
        fill.date(): [_bar(fill, 22.0, 23.5, 21.8, 23.2, 100.0)],
        _et(2023, 7, 21, 15, 45).date(): [
            _bar(_et(2023, 7, 21, 9, 30), 22.0, 22.1, 21.4, 21.6, 80.0),
            _bar(_et(2023, 7, 21, 15, 45), 21.6, 21.8, 19.8, 20.50, 80.0),
        ],
        _et(2023, 7, 24, 9, 30).date(): [
            _bar(_et(2023, 7, 24, 9, 30), 20.9, 21.2, 20.6, 20.8, 70.0),
        ],
        _et(2023, 8, 1, 9, 45).date(): [
            _bar(_et(2023, 8, 1, 9, 45), 18.0, 18.2, 17.6, 17.8, 90.0),
        ],
    }
    row = pd.Series(
        {
            "buy_price": 23.24,
            "buy_time": fill,
            "buy_date": "2023-07-20",
            "sell_price": 17.80,
            "sell_time": _et(2023, 8, 1, 9, 45),
            "sell_date": "2023-08-01",
            "exit_reason": "hard_stop",
            "gain_pct": -23.4,
            "channel_width": 4.92,
            "channel_pos": 1.492,
            "slope_pct_per_bar": 0.0,
            "l1_price": 12.27,
        }
    )
    got = apply_early_exit(
        row,
        by_day,
        enable_doji=False,
        enable_seller=False,
        enable_failed_breakout=True,
    )
    assert got.failed_breakout_day.isoformat() == "2023-07-21"
    assert got.used_overlay is True
    assert got.exit_reason == "failed_breakout"
    assert got.sell_ts == _et(2023, 7, 24, 9, 30)
    assert got.sell_px == pytest.approx(bar_mid(21.2, 20.6))
    assert got.gain_pct > -23.0


def test_splice_calendar_year_replaces_only_that_year():
    base = pd.DataFrame(
        [
            {"stock": "AAA", "buy_date": "2021-06-01", "gain_pct": 1.0},
            {"stock": "BBB", "buy_date": "2022-03-01", "gain_pct": -5.0},
            {"stock": "CCC", "buy_date": "2023-01-01", "gain_pct": 2.0},
        ]
    )
    repl = pd.DataFrame(
        [
            {"stock": "L3A", "buy_date": "2022-02-01", "gain_pct": 0.4},
            {"stock": "L3B", "buy_date": "2022-11-01", "gain_pct": 0.2},
            {"stock": "L3C", "buy_date": "2021-12-01", "gain_pct": 9.0},
        ]
    )
    got = splice_calendar_year(base, repl, year=2022, src_base="1d", src_repl="l3")
    assert list(got["stock"]) == ["AAA", "L3A", "L3B", "CCC"]
    assert list(got["splice_src"]) == ["1d", "l3", "l3", "1d"]
    assert "BBB" not in set(got["stock"])
    assert "L3C" not in set(got["stock"])
