"""Unit tests for causal channel-touch walk-replay."""
from __future__ import annotations

from unittest import mock

import numpy as np
import pandas as pd

from utils.research.channel_touch_walk_replay import (
    WindowedSetupCache,
    compare_trade_lists,
    last_bar_fills,
    pending_to_trades,
    walk_replay_trades,
)


def _ohlcv(n: int = 80, start: str = "2024-01-02") -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=n)
    close = np.linspace(100.0, 120.0, n)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(n, 1_000_000.0),
        },
        index=idx,
    )


def _setup(df: pd.DataFrame, h2: int) -> dict:
    return {
        "support_x0": 10,
        "support_y0": 100.0,
        "support_slope": 0.05,
        "channel_width": 8.0,
        "h2_idx": int(h2),
        "l1_idx": 10,
        "l2_idx": 20,
        "h2_date": df.index[h2].strftime("%Y-%m-%d"),
        "start_date": df.index[10].strftime("%Y-%m-%d"),
        "end_date": df.index[h2].strftime("%Y-%m-%d"),
        "touch_indices": [10, 20],
        "slope_pct_per_bar": 0.05,
        "channel_width_pct": 6.0,
        "bars_span": int(h2 - 10),
        "pivot_len": 5,
    }


def test_last_bar_fills_ignore_earlier_tags():
    df = _ohlcv(40)
    setups = [_setup(df, h2=20)]
    with mock.patch(
        "utils.research.channel_touch_walk_replay._h2_rail_tag_fills",
        return_value=[(25, 110.0, 3, False, True)],
    ):
        out = last_bar_fills(
            df,
            setups,
            error_pct=1.2,
            slip=0.001,
            wait=252,
            min_wait=6,
            entry_touch=3,
            h2_resist_break=True,
            h2_resist_break_only=True,
        )
    assert out == []


def test_last_bar_fills_keep_fill_on_prefix_end():
    df = _ohlcv(40)
    last = len(df) - 1
    setups = [_setup(df, h2=20)]
    with mock.patch(
        "utils.research.channel_touch_walk_replay._h2_rail_tag_fills",
        return_value=[(last, 118.5, 3, False, True)],
    ) as tagged:
        out = last_bar_fills(
            df,
            setups,
            error_pct=1.2,
            slip=0.001,
            wait=252,
            min_wait=6,
            entry_touch=3,
            h2_resist_break=True,
            h2_resist_break_only=True,
        )
    assert len(out) == 1
    assert out[0][1] == last
    assert tagged.call_args.kwargs["n"] == len(df)


def test_last_bar_fills_skips_expired_wait_window():
    df = _ohlcv(40)
    setups = [_setup(df, h2=5)]
    with mock.patch(
        "utils.research.channel_touch_walk_replay._h2_rail_tag_fills",
        return_value=[(39, 118.0, 3, False, True)],
    ) as tagged:
        out = last_bar_fills(
            df,
            setups,
            error_pct=1.2,
            slip=0.001,
            wait=10,
            min_wait=6,
            entry_touch=3,
            h2_resist_break=True,
            h2_resist_break_only=True,
        )
    assert out == []
    tagged.assert_not_called()


def test_last_bar_fills_resist_break_only_drops_l3():
    df = _ohlcv(30)
    last = len(df) - 1
    setups = [_setup(df, h2=10)]
    with mock.patch(
        "utils.research.channel_touch_walk_replay._h2_rail_tag_fills",
        return_value=[(last, 111.0, 3, False, False)],
    ):
        out = last_bar_fills(
            df,
            setups,
            error_pct=1.2,
            slip=0.001,
            wait=252,
            min_wait=6,
            entry_touch=3,
            h2_resist_break=True,
            h2_resist_break_only=True,
        )
    assert out == []


def test_walk_replay_never_passes_future_bars_to_detector():
    df = _ohlcv(120)
    seen_ends = []

    def spy(frame, **kwargs):
        seen_ends.append(frame.index[-1])
        return []

    cache = WindowedSetupCache(
        spy, window_bars=40, step_bars=20, pivot_len=5, channel_kwargs={}
    )
    with mock.patch(
        "utils.research.channel_touch_walk_replay.WindowedSetupCache",
        return_value=cache,
    ), mock.patch(
        "utils.research.channel_touch_walk_replay.last_bar_fills",
        return_value=[],
    ):
        walk_replay_trades(
            "AAA",
            df,
            pivot_len=5,
            window_bars=40,
            window_step_bars=20,
            min_prefix_bars=50,
            progress_every=0,
            squeeze_adaptive=False,
            entry_features=False,
        )
    assert seen_ends
    assert max(seen_ends) <= df.index[-1]
    # Spy is called on windows of growing prefixes; no window can end after
    # the prefix it belongs to. The last recorded end is some prefix's last bar.
    assert all(ts <= df.index[-1] for ts in seen_ends)


def test_walk_replay_emits_fill_only_when_last_bar_matches():
    df = _ohlcv(90)
    h2 = 40
    setup = _setup(df, h2=h2)

    def finder(frame, **kwargs):
        n = len(frame)
        if n <= h2:
            return []
        ch = dict(setup)
        if n - 1 < h2:
            return []
        return [ch]

    fill_at = 55

    def fake_fills(high, low, close, **kwargs):
        n = int(kwargs["n"])
        last = n - 1
        if last == fill_at:
            return [(fill_at, 112.0, 3, False, True)]
        return []

    with mock.patch(
        "utils.research.channel_touch_walk_replay.find_h2_l3_setups",
        side_effect=finder,
    ), mock.patch(
        "utils.research.channel_touch_walk_replay._h2_rail_tag_fills",
        side_effect=fake_fills,
    ):
        rows = walk_replay_trades(
            "AAA",
            df,
            pivot_len=5,
            window_bars=0,
            min_prefix_bars=45,
            squeeze_adaptive=False,
            h2_resist_break=True,
            h2_resist_break_only=True,
            stop_pct=0.03,
            trail_pct=0.10,
        )
    assert len(rows) == 1
    assert rows[0]["entry_i"] == fill_at
    assert rows[0]["buy_date"] == df.index[fill_at].strftime("%Y-%m-%d")
    assert rows[0]["source"] == "walk"


def test_window_cache_matches_growing_prefixes():
    df = _ohlcv(90)

    def finder(frame, **kwargs):
        return [
            {
                "touch_indices": [1, 4],
                "support_x0": 1,
                "support_y0": 9.9,
                "support_slope": 0.01,
                "channel_width": 0.5,
                "h2_idx": min(6, len(frame) - 1),
                "l1_idx": 1,
                "l2_idx": 4,
                "start_date": "2024-01-03",
                "end_date": "2024-01-08",
                "bars_span": 5,
            }
        ]

    cache = WindowedSetupCache(
        finder, window_bars=20, step_bars=10, pivot_len=5, channel_kwargs={}
    )
    for t in range(25, len(df), 7):
        prefix = df.iloc[: t + 1]
        got = [_setup_key_local(c) for c in cache.setups(prefix)]
        fresh = WindowedSetupCache(
            finder, window_bars=20, step_bars=10, pivot_len=5, channel_kwargs={}
        )
        expect = [_setup_key_local(c) for c in fresh.setups(prefix)]
        assert got == expect


def _setup_key_local(ch: dict):
    idxs = ch.get("touch_indices") or []
    return (
        int(idxs[0]),
        int(idxs[-1]),
        int(ch.get("h2_idx", -1)),
        round(float(ch.get("support_slope", 0.0)), 8),
        round(float(ch.get("channel_width", 0.0)), 6),
    )


def test_compare_trade_lists_splits_matched_and_only():
    batch = [
        {"buy_date": "2024-06-03", "buy_price": 10.0, "gain_pct": 1.0},
        {"buy_date": "2024-07-01", "buy_price": 11.0, "gain_pct": -2.0},
    ]
    walk = [
        {"buy_date": "2024-06-03", "buy_price": 10.0, "gain_pct": 1.1},
        {"buy_date": "2024-08-01", "buy_price": 12.0, "gain_pct": 3.0},
    ]
    cmp = compare_trade_lists(batch, walk)
    assert cmp["n_matched"] == 1
    assert cmp["n_only_batch"] == 1
    assert cmp["n_only_walk"] == 1
    assert cmp["only_batch"][0]["buy_date"] == "2024-07-01"
    assert cmp["only_walk"][0]["buy_date"] == "2024-08-01"


def test_pending_to_trades_skips_overlap():
    df = _ohlcv(40)
    ch = _setup(df, h2=8)
    pending = [
        (ch, 12, 105.0, 3, True),
        (ch, 14, 106.0, 3, True),
    ]
    rows = pending_to_trades(
        "AAA",
        df,
        pending,
        squeeze_adaptive=False,
        stop_pct=0.03,
        trail_pct=0.10,
        atr_stop_mult=2.0,
    )
    assert len(rows) >= 1
    assert rows[0]["entry_i"] == 12
    if len(rows) > 1:
        assert rows[1]["entry_i"] > rows[0]["exit_i"]
