"""Unit tests for incremental IB 15m universe backfill."""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "data"))

from backfill_ib_15m_universe import (  # noqa: E402
    DEFAULT_CLIENT_ID,
    needs_backfill,
    parse_args,
    rewind_start,
)


def test_default_client_id_is_not_single_symbol_backfill():
    args = parse_args([])
    assert args.ib_client_id == DEFAULT_CLIENT_ID == 8822
    assert args.sleep == 1.0
    assert args.fresh_hours == 36.0
    assert args.inventory is False


def test_inventory_flag():
    args = parse_args(["--inventory", "--limit", "10"])
    assert args.inventory is True
    assert args.limit == 10


def test_needs_backfill_stale_vs_fresh():
    now = datetime(2026, 8, 30, 12, 0, tzinfo=timezone.utc)
    stale = pd.Timestamp("2025-12-02T20:45:00Z")
    fresh = pd.Timestamp("2026-08-29T20:45:00Z")
    assert needs_backfill(stale, now=now, fresh_hours=36.0) is True
    assert needs_backfill(fresh, now=now, fresh_hours=36.0) is False
    assert needs_backfill(None, now=now) is True


def test_needs_backfill_stale_before_cutoff():
    now = datetime(2026, 8, 30, tzinfo=timezone.utc)
    last = pd.Timestamp("2025-12-02T20:45:00Z")
    cut = datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert needs_backfill(last, now=now, stale_before=cut) is True
    recent = pd.Timestamp("2026-06-01T20:45:00Z")
    assert needs_backfill(recent, now=now, stale_before=cut) is False


def test_rewind_start_overlaps_last_bars():
    last = pd.Timestamp("2025-12-02T20:45:00Z")
    start = rewind_start(last, 2)
    assert start == datetime(2025, 12, 2, 20, 15, tzinfo=timezone.utc)
