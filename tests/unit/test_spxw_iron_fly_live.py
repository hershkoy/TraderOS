"""Unit tests for SPXW iron fly live Friday auto-park helpers."""
from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from utils.scanning.spxw_iron_fly_live import (
    ET,
    friday_park_already_done,
    is_friday,
    is_wednesday,
    should_auto_park_friday,
    write_friday_park_marker,
)


def test_weekday_helpers():
    assert is_wednesday(date(2026, 10, 28)) is True  # Wed
    assert is_friday(date(2026, 10, 30)) is True  # Fri
    assert is_friday(date(2026, 10, 29)) is False  # Thu


def test_should_auto_park_friday_only_on_friday_buy(tmp_path: Path):
    marker = tmp_path / "park.json"
    fri = date(2026, 10, 30)
    thu = date(2026, 10, 29)

    assert should_auto_park_friday(day=fri, buy_ok=True, path=marker) is True
    assert should_auto_park_friday(day=fri, buy_ok=False, path=marker) is False
    assert should_auto_park_friday(day=thu, buy_ok=True, path=marker) is False


def test_friday_park_marker_success_blocks_retry(tmp_path: Path):
    marker = tmp_path / "park.json"
    fri = date(2026, 10, 30)
    write_friday_park_marker(fri, status="parked_no_transmit", path=marker)
    assert friday_park_already_done(fri, path=marker) is True
    assert should_auto_park_friday(day=fri, buy_ok=True, path=marker) is False


def test_friday_park_failed_is_retryable(tmp_path: Path):
    marker = tmp_path / "park.json"
    fri = date(2026, 10, 30)
    write_friday_park_marker(fri, status="place_failed", path=marker)
    assert friday_park_already_done(fri, path=marker) is False
    assert should_auto_park_friday(day=fri, buy_ok=True, path=marker) is True


def test_friday_park_in_progress_stale_retryable(tmp_path: Path, monkeypatch):
    marker = tmp_path / "park.json"
    fri = date(2026, 10, 30)
    write_friday_park_marker(fri, status="in_progress", path=marker)
    assert friday_park_already_done(fri, path=marker) is True

    stale = datetime.now(ET) - timedelta(minutes=11)

    def _stale_now():
        return stale + timedelta(minutes=11)

    # Rewrite marker with old ts
    import json

    payload = json.loads(marker.read_text(encoding="utf-8"))
    payload["ts"] = stale.isoformat()
    marker.write_text(json.dumps(payload), encoding="utf-8")
    assert friday_park_already_done(fri, path=marker) is False
