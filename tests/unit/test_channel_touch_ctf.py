"""CTF paste JSON for /hot candidates."""
from __future__ import annotations

from utils.scanning.channel_touch_ctf import (
    attach_channel_json,
    channel_json_for_candidate,
    channel_rails_fingerprint,
)


def _sample_row(**overrides):
    row = {
        "stock": "RDWR",
        "timeframe": "15m",
        "status": "armed",
        "as_of": "2025-06-13 13:30:00",
        "h2_time": "2025-06-05 14:45:00",
        "channel_start": "2025-05-20 15:00:00",
        "channel_end": "2025-06-05",
        "support_x0": 100,
        "support_y0": 20.0,
        "support_slope": 0.01,
        "channel_width": 1.5,
        "h2_idx": 200,
        "resist": 24.64,
        "fill_px": None,
    }
    row.update(overrides)
    return row


def test_channel_json_approx_from_support_geometry():
    ch = channel_json_for_candidate(_sample_row())
    assert ch is not None
    assert ch["sym"] == "RDWR"
    assert ch["src"] == "approx"
    assert ch["l1p"] == 20.0
    # L2 at H2-on-support: 20 + 0.01 * (200-100) = 21
    assert ch["l2p"] == 21.0
    assert ch["w"] == 1.5
    assert "h2t" in ch
    assert "fill_ms" in ch
    assert ch["fp"] == channel_rails_fingerprint(ch)


def test_channel_json_prefers_explicit_l1_l2():
    ch = channel_json_for_candidate(
        _sample_row(
            l1_time="2025-05-20 15:00:00",
            l1_price=20.5,
            l2_time="2025-05-28 18:00:00",
            l2_price=21.2,
        )
    )
    assert ch is not None
    assert ch["src"] == "rails"
    assert ch["l1p"] == 20.5
    assert ch["l2p"] == 21.2


def test_channel_json_includes_enp_when_filled():
    ch = channel_json_for_candidate(_sample_row(status="filled", fill_px=24.64))
    assert ch is not None
    assert ch["enp"] == 24.64


def test_fingerprint_ignores_fill_moves():
    a = channel_json_for_candidate(_sample_row(as_of="2025-06-13 13:30:00"))
    b = channel_json_for_candidate(_sample_row(as_of="2025-06-13 14:45:00"))
    assert a is not None and b is not None
    assert channel_rails_fingerprint(a) == channel_rails_fingerprint(b)


def test_fingerprint_changes_when_rails_change():
    a = channel_json_for_candidate(_sample_row())
    b = channel_json_for_candidate(_sample_row(channel_width=2.0))
    assert a is not None and b is not None
    assert channel_rails_fingerprint(a) != channel_rails_fingerprint(b)


def test_attach_channel_json_on_rows():
    rows = [_sample_row(), {"stock": "BAD"}]
    attach_channel_json(rows)
    assert "ch" in rows[0]
    assert rows[0]["ch"]["sym"] == "RDWR"
    assert "ch" not in rows[1]


def test_candidates_payload_includes_ch():
    from utils.scanning.channel_touch_candidates_store import DEFAULT_SETTINGS
    from utils.scanning.channel_touch_hot_api import candidates_payload, reset_refresh_throttle, set_store

    class MemoryStore:
        def __init__(self):
            self.rows = [_sample_row()]
            self.settings = dict(DEFAULT_SETTINGS)

        def load_settings(self):
            return dict(self.settings)

        def load_rows(self):
            return [dict(r) for r in self.rows]

        def update_live_prices(self, rows, *, price_ts=None):
            return None

        def load_bought(self, *, active_only=True):
            return []

        def mark_hot_notified(self, stocks, day=None):
            return None

    reset_refresh_throttle()
    set_store(MemoryStore())
    payload = candidates_payload(refresh=False)
    assert payload["rows"]
    assert payload["rows"][0].get("ch", {}).get("sym") == "RDWR"
