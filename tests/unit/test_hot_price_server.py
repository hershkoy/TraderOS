"""Flask /health and /kick on the always-on price hub."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.scanning.channel_touch_hot_api import HotCandidatesHub, set_hot_hub


def test_health_and_kick():
    hub = HotCandidatesHub(
        always_run=True,
        payload_fn=lambda: {"price_ts": "2026-09-05 12:00:00", "rows": []},
    )
    set_hot_hub(hub)
    try:
        import hot_price_server as hps

        client = hps.app.test_client()
        got = client.get("/health")
        assert got.status_code == 200
        body = got.get_json()
        assert body["ok"] is True
        assert body["clients"] == 0
        assert body["always_run"] is True
        kicked = client.post("/kick")
        assert kicked.status_code == 200
        assert kicked.get_json()["ok"] is True
        via_get = client.get("/kick")
        assert via_get.status_code == 200
    finally:
        hub.stop()
        set_hot_hub(None)
