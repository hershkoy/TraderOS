"""TimescaleDB store helpers for 15m hot candidates (no live DB)."""
from __future__ import annotations

from utils.scanning.channel_touch_candidates_store import (
    CANDIDATE_COLUMNS,
    DEFAULT_SETTINGS,
    apply_settings_patch,
    merge_preserved_live_fields,
    normalize_settings,
    row_to_db_tuple,
    same_setup,
)


def test_normalize_settings_defaults_and_patch():
    s = normalize_settings(None)
    assert s["telegram_on_fill"] is True
    assert s["telegram_on_hot"] is False
    assert s["desktop_notify"] is True
    assert s["proximity_below_pct"] == 0.0
    assert s["max_abs_dist_pct"] is None
    assert s["sort_key"] == "abs_dist"
    patched = apply_settings_patch(
        s,
        {
            "telegram_on_hot": True,
            "max_abs_dist_pct": 3,
            "status_filter": "armed",
            "search": "nvda",
        },
    )
    assert patched["telegram_on_hot"] is True
    assert patched["telegram_on_fill"] is True
    assert patched["max_abs_dist_pct"] == 3.0
    assert patched["search"] == "nvda"


def test_merge_keeps_live_price_and_hot_notified_for_same_h2():
    existing = {
        "AAA": {
            "stock": "AAA",
            "h2_time": "2026-08-29 14:30:00",
            "last_price": 10.5,
            "last_price_ts": "2026-08-30 14:01:00",
            "dist_live_pct": 5.0,
            "hot": True,
            "hot_notified_on": "2026-08-30",
            "resist": 10.0,
        }
    }
    new_rows = [
        {
            "stock": "aaa",
            "h2_time": "2026-08-29 14:30:00",
            "resist": 10.0,
            "status": "armed",
        }
    ]
    out = merge_preserved_live_fields(new_rows, existing, below_pct=0.0)
    assert out[0]["last_price"] == 10.5
    assert out[0]["hot_notified_on"] == "2026-08-30"
    assert out[0]["hot"] is True
    assert out[0]["dist_live_pct"] == 5.0


def test_merge_clears_live_fields_when_h2_changes():
    existing = {
        "AAA": {
            "stock": "AAA",
            "h2_time": "2026-08-01 14:30:00",
            "last_price": 10.5,
            "hot_notified_on": "2026-08-30",
            "hot": True,
        }
    }
    new_rows = [{"stock": "AAA", "h2_time": "2026-08-29 15:45:00", "resist": 12.0, "status": "armed"}]
    out = merge_preserved_live_fields(new_rows, existing, below_pct=0.0)
    assert out[0]["last_price"] is None
    assert out[0]["hot_notified_on"] is None
    assert out[0]["hot"] is False


def test_same_setup_requires_stock_and_h2():
    assert same_setup({"stock": "A", "h2_time": "t1"}, {"stock": "A", "h2_time": "t1"})
    assert not same_setup({"stock": "A", "h2_time": "t1"}, {"stock": "A", "h2_time": "t2"})
    assert not same_setup({"stock": "A", "h2_time": "t1"}, {"stock": "B", "h2_time": "t1"})


def test_row_to_db_tuple_length_matches_columns():
    tup = row_to_db_tuple({"stock": "aaa", "status": "armed", "hot": 1, "wait_bars": "12"})
    assert len(tup) == len(CANDIDATE_COLUMNS)
    assert tup[0] == "AAA"
    assert tup[1] == "15m"
    assert tup[-2] is True  # hot
    assert DEFAULT_SETTINGS["sort_dir"] == "asc"
    assert DEFAULT_SETTINGS["timeframe_filter"] == "all"


def test_merge_keeps_15m_and_1d_separate():
    existing = {
        "AAA|15m": {
            "stock": "AAA",
            "timeframe": "15m",
            "h2_time": "t15",
            "last_price": 10.0,
            "hot_notified_on": "2026-08-30",
        },
        "AAA|1d": {
            "stock": "AAA",
            "timeframe": "1d",
            "h2_time": "t1d",
            "last_price": 20.0,
            "hot_notified_on": "2026-08-29",
        },
    }
    new_rows = [
        {"stock": "AAA", "timeframe": "1d", "h2_time": "t1d", "resist": 19.0, "status": "armed"}
    ]
    out = merge_preserved_live_fields(new_rows, existing, below_pct=0.0)
    assert out[0]["last_price"] == 20.0
    assert out[0]["hot_notified_on"] == "2026-08-29"
