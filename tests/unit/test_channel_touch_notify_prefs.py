"""Unit tests for per-detector /hot notify prefs."""
from __future__ import annotations

from utils.scanning.channel_touch_notify_prefs import (
    legacy_from_prefs,
    merge_notify_prefs,
    normalize_notify_prefs,
    prefs_from_legacy,
    want_notify,
)
from utils.scanning.channel_touch_candidates_store import apply_settings_patch, normalize_settings
from utils.scanning.channel_touch_15m import notify_payloads
import pandas as pd


def test_prefs_from_legacy_maps_old_checkboxes():
    prefs = prefs_from_legacy(
        telegram_on_fill=True,
        telegram_on_hot=True,
        telegram_on_sell=False,
        desktop_notify=False,
    )
    assert prefs["15m"]["fill"]["telegram"] is True
    assert prefs["dp"]["armed"]["telegram"] is True
    assert prefs["dp"]["approaching"]["telegram"] is True
    assert prefs["15m"]["sell"]["telegram"] is False
    assert prefs["15m"]["fill"]["browser"] is False
    assert prefs["15m"]["hot"]["telegram"] is False


def test_want_notify_reads_matrix_and_legacy():
    settings = normalize_settings(
        {
            "notify_prefs": {
                "15m": {"hot": {"telegram": True, "browser": False}},
                "dp": {"approaching": {"telegram": False, "browser": True}},
            }
        }
    )
    assert want_notify(settings, "15m", "hot", "telegram") is True
    assert want_notify(settings, "15m", "hot", "browser") is False
    assert want_notify(settings, "dp", "approaching", "browser") is True
    assert want_notify(settings, "15m", "approaching", "telegram") is False
    legacy_only = {"telegram_on_fill": False, "telegram_on_hot": True, "telegram_on_sell": True, "desktop_notify": True}
    assert want_notify(legacy_only, "15m", "fill", "telegram") is False
    assert want_notify(legacy_only, "dp", "approaching", "telegram") is True
    assert want_notify(legacy_only, "dp", "armed", "telegram") is False


def test_merge_prefs_preserves_untouched_cells():
    base = prefs_from_legacy()
    base["15m"]["hot"]["telegram"] = True
    merged = merge_notify_prefs(base, {"dp": {"approaching": {"telegram": True}}})
    assert merged["15m"]["hot"]["telegram"] is True
    assert merged["dp"]["approaching"]["telegram"] is True
    assert merged["15m"]["fill"]["telegram"] is True


def test_apply_settings_patch_syncs_legacy_flags():
    s = normalize_settings(None)
    patched = apply_settings_patch(
        s,
        {"notify_prefs": {"dp": {"approaching": {"telegram": True, "browser": True}}}},
    )
    assert patched["telegram_on_hot"] is True
    assert patched["notify_prefs"]["dp"]["approaching"]["browser"] is True
    assert patched["notify_applicable"]["dp"]["approaching"] is True
    legacy = legacy_from_prefs(patched["notify_prefs"])
    assert legacy["telegram_on_hot"] is True


def test_qs_prefs_round_trip_and_fill_missing_detector():
    s = normalize_settings({"notify_prefs": {"dp": {"approaching": {"telegram": True}}}})
    assert "qs" in s["notify_prefs"]
    assert s["notify_applicable"]["qs"]["approaching"] is True
    assert s["notify_prefs"]["qs"]["armed"]["telegram"] is True
    patched = apply_settings_patch(
        s,
        {"notify_prefs": {"qs": {"approaching": {"telegram": True, "browser": True}}}},
    )
    assert patched["notify_prefs"]["qs"]["approaching"]["telegram"] is True
    assert patched["notify_prefs"]["qs"]["approaching"]["browser"] is True
    assert patched["notify_prefs"]["dp"]["approaching"]["telegram"] is True
    assert patched["telegram_on_hot"] is True


def test_notify_payloads_15m_hot_when_prefs_on():
    fills = pd.DataFrame(
        [
            {
                "stock": "AAA",
                "fill_px": 10.0,
                "resist": 9.9,
                "wait_bars": 12,
                "volume_rel_20": 2.2,
                "overshoot": 0.1,
            }
        ]
    )
    hot = [
        {
            "stock": "BBB",
            "last_price": 11.0,
            "resist": 10.8,
            "dist_live_pct": 1.8,
            "wait_bars": 14,
            "volume_rel_20": 2.4,
        }
    ]
    settings = normalize_settings(
        {
            "notify_prefs": {
                "15m": {
                    "fill": {"telegram": False, "browser": False},
                    "hot": {"telegram": True, "browser": False},
                }
            }
        }
    )
    msgs = notify_payloads(
        fills=fills,
        newly_hot=hot,
        settings=settings,
        as_of="2026-08-30 15:45:00",
        n_armed=4,
        n_hot=1,
    )
    assert len(msgs) == 1
    assert "newly hot" in msgs[0].lower()
    assert "BBB" in msgs[0]
