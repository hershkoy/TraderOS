"""Unit tests for H2+L3 blend occupancy (unified FCFS vs dedicated sleeves)."""
from __future__ import annotations

import pandas as pd

from utils.research.h2_l3_blend_occupancy import (
    SLEEVE_H2,
    SLEEVE_L3,
    occupancy_dedicated_sleeves,
    occupancy_unified_fcfs,
    prepare_blend,
    skip_counts,
    sleeve_counts,
)


def _book(rows):
    return pd.DataFrame(rows)


def test_prepare_blend_tags_sleeves_and_sorts():
    h2 = _book(
        [
            {
                "stock": "aaa",
                "buy_time": "2022-01-10 20:45:00",
                "sell_time": "2022-01-20 15:00:00",
                "gain_pct": 1.0,
                "buy_date": "2022-01-10",
            }
        ]
    )
    l3 = _book(
        [
            {
                "stock": "BBB",
                "buy_time": "2022-01-05 14:30:00",
                "sell_time": "2022-01-05 18:00:00",
                "gain_pct": 0.2,
                "buy_date": "2022-01-05",
            }
        ]
    )
    blend = prepare_blend(h2, l3)
    assert list(blend["_stock"]) == ["BBB", "AAA"]
    assert list(blend["sleeve"]) == [SLEEVE_L3, SLEEVE_H2]


def test_unified_cap1_h2_hold_starves_later_l3():
    h2 = _book(
        [
            {
                "stock": "AAA",
                "buy_time": "2022-01-03 20:45:00",
                "sell_time": "2022-01-20 15:00:00",
                "gain_pct": 5.0,
                "buy_date": "2022-01-03",
            }
        ]
    )
    l3 = _book(
        [
            {
                "stock": "BBB",
                "buy_time": "2022-01-04 14:30:00",
                "sell_time": "2022-01-04 18:00:00",
                "gain_pct": 1.0,
                "buy_date": "2022-01-04",
            }
        ]
    )
    blend = prepare_blend(h2, l3)
    kept, skip = occupancy_unified_fcfs(blend, max_open=1, unique_symbol=True)
    assert sleeve_counts(kept)[SLEEVE_H2] == 1
    assert sleeve_counts(kept)[SLEEVE_L3] == 0
    counts = skip_counts(skip)
    assert counts.get("cap_full_%s" % SLEEVE_L3) == 1


def test_dedicated_l3_fires_while_h2_at_cap():
    h2 = _book(
        [
            {
                "stock": "AAA",
                "buy_time": "2022-01-03 20:45:00",
                "sell_time": "2022-01-20 15:00:00",
                "gain_pct": 5.0,
                "buy_date": "2022-01-03",
            }
        ]
    )
    l3 = _book(
        [
            {
                "stock": "BBB",
                "buy_time": "2022-01-04 14:30:00",
                "sell_time": "2022-01-04 18:00:00",
                "gain_pct": 1.0,
                "buy_date": "2022-01-04",
            }
        ]
    )
    blend = prepare_blend(h2, l3)
    kept, skip = occupancy_dedicated_sleeves(
        blend, h2_cap=1, l3_cap=1, unique_across=True
    )
    assert sleeve_counts(kept)[SLEEVE_H2] == 1
    assert sleeve_counts(kept)[SLEEVE_L3] == 1
    assert skip.empty


def test_unique_across_blocks_same_name_l3():
    h2 = _book(
        [
            {
                "stock": "AAA",
                "buy_time": "2022-01-03 20:45:00",
                "sell_time": "2022-01-20 15:00:00",
                "gain_pct": 5.0,
                "buy_date": "2022-01-03",
            }
        ]
    )
    l3 = _book(
        [
            {
                "stock": "AAA",
                "buy_time": "2022-01-04 14:30:00",
                "sell_time": "2022-01-04 18:00:00",
                "gain_pct": 1.0,
                "buy_date": "2022-01-04",
            }
        ]
    )
    blend = prepare_blend(h2, l3)
    kept, skip = occupancy_dedicated_sleeves(
        blend, h2_cap=1, l3_cap=1, unique_across=True
    )
    assert sleeve_counts(kept)[SLEEVE_L3] == 0
    assert skip_counts(skip).get("same_symbol_%s" % SLEEVE_L3) == 1
    kept_i, skip_i = occupancy_dedicated_sleeves(
        blend, h2_cap=1, l3_cap=1, unique_across=False
    )
    assert sleeve_counts(kept_i)[SLEEVE_L3] == 1
    assert skip_i.empty


def test_sell_frees_unified_slot():
    h2 = _book(
        [
            {
                "stock": "AAA",
                "buy_time": "2022-01-03 20:45:00",
                "sell_time": "2022-01-04 14:00:00",
                "gain_pct": -2.0,
                "buy_date": "2022-01-03",
            }
        ]
    )
    l3 = _book(
        [
            {
                "stock": "BBB",
                "buy_time": "2022-01-04 14:00:00",
                "sell_time": "2022-01-04 18:00:00",
                "gain_pct": 1.0,
                "buy_date": "2022-01-04",
            }
        ]
    )
    blend = prepare_blend(h2, l3)
    kept, skip = occupancy_unified_fcfs(blend, max_open=1, unique_symbol=True)
    assert sleeve_counts(kept)[SLEEVE_L3] == 1
    assert skip.empty
