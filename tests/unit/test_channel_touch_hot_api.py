"""Hot-candidate sort/filter, throttle, and payload (no live Alpaca/DB)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from utils.scanning.channel_touch_candidates_store import DEFAULT_SETTINGS, apply_settings_patch
from utils.scanning.channel_touch_hot_api import (
    candidates_payload,
    filter_candidates,
    hot_keys_from_rows,
    maybe_refresh_live_prices,
    prices_are_stale,
    reset_refresh_throttle,
    set_store,
    sort_candidates,
)


class MemoryStore:
    def __init__(self, rows=None, settings=None):
        self.rows = [dict(r) for r in (rows or [])]
        self.settings = dict(DEFAULT_SETTINGS)
        if settings:
            self.settings = apply_settings_patch(self.settings, settings)

    def load_settings(self):
        return dict(self.settings)

    def save_settings(self, patch=None, current=None):
        self.settings = apply_settings_patch(current or self.settings, patch)
        return dict(self.settings)

    def load_rows(self):
        return [dict(r) for r in self.rows]

    def update_live_prices(self, rows, *, price_ts=None):
        by = {str(r.get("stock")).upper(): r for r in rows}
        for item in self.rows:
            upd = by.get(str(item.get("stock")).upper())
            if not upd:
                continue
            item["last_price"] = upd.get("last_price")
            item["dist_live_pct"] = upd.get("dist_live_pct")
            item["hot"] = upd.get("hot")
            item["last_price_ts"] = str(price_ts) if price_ts is not None else upd.get("last_price_ts")

    def mark_hot_notified(self, stocks, day=None):
        day_s = str(day or "")[:10]
        want = {str(s).upper() for s in stocks}
        for item in self.rows:
            if str(item.get("stock")).upper() in want:
                item["hot_notified_on"] = day_s


def test_sort_closest_to_resist_first():
    rows = [
        {"stock": "FAR", "dist_live_pct": -4.0},
        {"stock": "NEAR", "dist_live_pct": 0.2},
        {"stock": "NONE", "dist_live_pct": None},
        {"stock": "THRU", "dist_live_pct": 1.5},
    ]
    ordered = sort_candidates(rows, sort_key="abs_dist", sort_dir="asc")
    assert [r["stock"] for r in ordered] == ["NEAR", "THRU", "FAR", "NONE"]


def test_filter_status_hot_and_max_abs_dist():
    rows = [
        {"stock": "A", "status": "armed", "hot": True, "dist_live_pct": 0.4},
        {"stock": "B", "status": "waiting", "hot": False, "dist_live_pct": -0.5},
        {"stock": "C", "status": "armed", "hot": False, "dist_live_pct": -5.0},
    ]
    hot = filter_candidates(rows, status_filter="hot")
    assert [r["stock"] for r in hot] == ["A"]
    close = filter_candidates(rows, max_abs_dist_pct=1.0)
    assert [r["stock"] for r in close] == ["A", "B"]
    search = filter_candidates(rows, search="b")
    assert [r["stock"] for r in search] == ["B"]
    mixed = [
        {"stock": "A", "timeframe": "15m", "status": "armed", "hot": True, "dist_live_pct": 0.4},
        {"stock": "D", "timeframe": "1d", "status": "armed", "hot": False, "dist_live_pct": -0.2},
    ]
    only_1d = filter_candidates(mixed, timeframe_filter="1d")
    assert [r["stock"] for r in only_1d] == ["D"]


def test_prices_are_stale_without_ts():
    now = datetime(2026, 8, 30, 14, 0, tzinfo=timezone.utc)
    assert prices_are_stale([{"last_price_ts": None}], now=now) is True
    fresh = now.strftime("%Y-%m-%d %H:%M:%S")
    assert prices_are_stale([{"last_price_ts": fresh}], now=now, max_age_sec=5) is False
    old = (now - timedelta(seconds=20)).strftime("%Y-%m-%d %H:%M:%S")
    assert prices_are_stale([{"last_price_ts": old}], now=now, max_age_sec=5) is True


def test_maybe_refresh_throttles_and_writes_prices():
    reset_refresh_throttle()
    now = datetime(2026, 8, 30, 14, 0, tzinfo=timezone.utc)
    store = MemoryStore(
        rows=[{"stock": "AAA", "resist": 100.0, "status": "armed", "last_price_ts": None}]
    )
    calls = {"n": 0}

    def fake_fetch(symbols, batch_size=200):
        calls["n"] += 1
        return {"AAA": 101.0}

    mono = {"t": 0.0}

    rows, did = maybe_refresh_live_prices(
        store,
        {"proximity_below_pct": 0.0},
        refresh=True,
        fetch_fn=fake_fetch,
        now=now,
        monotonic_fn=lambda: mono["t"],
    )
    assert did is True
    assert calls["n"] == 1
    assert rows[0]["hot"] is True
    assert rows[0]["dist_live_pct"] == 1.0

    rows2, did2 = maybe_refresh_live_prices(
        store,
        {"proximity_below_pct": 0.0},
        refresh=True,
        fetch_fn=fake_fetch,
        now=now,
        monotonic_fn=lambda: mono["t"],
    )
    assert did2 is False
    assert calls["n"] == 1
    assert rows2[0]["hot"] is True


def test_candidates_payload_uses_saved_filter():
    reset_refresh_throttle()
    store = MemoryStore(
        rows=[
            {"stock": "AAA", "status": "armed", "hot": True, "dist_live_pct": 0.1, "last_price_ts": "2026-08-30 14:00:00"},
            {"stock": "BBB", "status": "waiting", "hot": False, "dist_live_pct": -4.0, "last_price_ts": "2026-08-30 14:00:00"},
        ],
        settings={"status_filter": "hot", "as_of": "2026-08-30 13:45:00", "n_universe": 10},
    )
    set_store(store)
    try:
        payload = candidates_payload(store, refresh=False)
        assert payload["n_hot"] == 1
        assert payload["n_armed"] == 1
        assert [r["stock"] for r in payload["rows"]] == ["AAA"]
        assert payload["as_of"] == "2026-08-30 13:45:00"
        assert payload["refreshed"] is False
        assert payload["price_ts"] == "2026-08-30 14:00:00"
        assert payload["prices_stale"] is True
        assert payload["price_age_sec"] is not None
        assert payload["hot_keys"] == ["AAA|15m"]
    finally:
        set_store(None)


def test_candidates_payload_prices_stale_uses_latest_quote_age():
    reset_refresh_throttle()
    now = datetime(2026, 8, 31, 12, 0, tzinfo=timezone.utc)
    fresh = MemoryStore(
        rows=[
            {
                "stock": "AAA",
                "status": "armed",
                "hot": True,
                "dist_live_pct": 0.1,
                "last_price_ts": "2026-08-31 11:59:00",
            }
        ]
    )
    payload = candidates_payload(fresh, refresh=False, now=now)
    assert payload["prices_stale"] is False
    assert payload["price_age_sec"] == 60.0

    old = MemoryStore(
        rows=[
            {
                "stock": "AAA",
                "status": "armed",
                "hot": True,
                "dist_live_pct": 0.1,
                "last_price_ts": "2026-08-31 11:50:00",
            }
        ]
    )
    stale = candidates_payload(old, refresh=False, now=now)
    assert stale["prices_stale"] is True
    assert stale["price_age_sec"] == 600.0

    empty = candidates_payload(MemoryStore(rows=[]), refresh=False, now=now)
    assert empty["prices_stale"] is False
    assert empty["price_ts"] is None


def test_hot_keys_from_rows_skips_cold_and_blank():
    assert hot_keys_from_rows(
        [
            {"stock": "aaa", "timeframe": "15m", "hot": True},
            {"stock": "BBB", "hot": True},
            {"stock": "CCC", "timeframe": "1d", "hot": False},
            {"stock": "", "hot": True},
        ]
    ) == ["AAA|15m", "BBB|15m"]


def test_payload_hot_keys_ignore_ui_filter():
    reset_refresh_throttle()
    store = MemoryStore(
        rows=[
            {
                "stock": "AAA",
                "timeframe": "15m",
                "status": "armed",
                "hot": True,
                "dist_live_pct": 0.1,
                "last_price_ts": "2026-08-30 14:00:00",
            },
            {
                "stock": "BBB",
                "timeframe": "15m",
                "status": "armed",
                "hot": True,
                "dist_live_pct": 0.2,
                "last_price_ts": "2026-08-30 14:00:00",
            },
            {
                "stock": "CCC",
                "timeframe": "1d",
                "status": "waiting",
                "hot": False,
                "dist_live_pct": -4.0,
                "last_price_ts": "2026-08-30 14:00:00",
            },
        ],
        settings={"timeframe_filter": "1d"},
    )
    payload = candidates_payload(store, refresh=False)
    assert [r["stock"] for r in payload["rows"]] == ["CCC"]
    assert payload["n_hot"] == 2
    assert payload["hot_keys"] == ["AAA|15m", "BBB|15m"]
