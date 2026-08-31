"""Flask /hot dashboard routes with an in-memory store."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.scanning.channel_touch_candidates_store import DEFAULT_SETTINGS, apply_settings_patch
from utils.scanning.channel_touch_hot_api import reset_refresh_throttle, set_store


class MemoryStore:
    def __init__(self):
        self.rows = [
            {
                "stock": "AAA",
                "status": "armed",
                "resist": 50.0,
                "last_price": 50.2,
                "dist_live_pct": 0.4,
                "hot": True,
                "wait_bars": 14,
                "volume_rel_20": 2.1,
                "as_of": "2026-08-30 15:45:00",
                "last_price_ts": "2026-08-30 15:46:00",
            }
        ]
        self.settings = apply_settings_patch(DEFAULT_SETTINGS, {"as_of": "2026-08-30 15:45:00", "n_universe": 12})

    def load_settings(self):
        return dict(self.settings)

    def save_settings(self, patch=None, current=None):
        self.settings = apply_settings_patch(current or self.settings, patch)
        return dict(self.settings)

    def load_rows(self):
        return [dict(r) for r in self.rows]

    def update_live_prices(self, rows, *, price_ts=None):
        return None


def test_hot_page_and_api(monkeypatch):
    reset_refresh_throttle()
    store = MemoryStore()
    set_store(store)
    try:
        import charting_server as cs

        client = cs.app.test_client()
        page = client.get("/hot")
        assert page.status_code == 200
        assert b"Hot candidates" in page.data
        assert b"Detector" in page.data
        assert b'id="tf-filter"' in page.data
        assert b"Desktop + sound" in page.data

        got = client.get("/api/hot-candidates?refresh=0")
        assert got.status_code == 200
        body = got.get_json()
        assert body["n_hot"] == 1
        assert body["rows"][0]["stock"] == "AAA"
        assert body["as_of"] == "2026-08-30 15:45:00"

        posted = client.post(
            "/api/hot-candidates/settings",
            data=json.dumps({"telegram_on_hot": True, "status_filter": "armed"}),
            content_type="application/json",
        )
        assert posted.status_code == 200
        saved = posted.get_json()
        assert saved["telegram_on_hot"] is True
        assert saved["telegram_on_fill"] is True
        assert saved["status_filter"] == "armed"

        listed = client.get("/api/hot-candidates/settings")
        assert listed.get_json()["telegram_on_hot"] is True
    finally:
        set_store(None)
