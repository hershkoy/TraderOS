"""Unit tests for IB 5m single-symbol backfill CLI."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "data"))

from backfill_ib_5m_symbol import DEFAULT_CLIENT_ID, parse_args  # noqa: E402


def test_default_client_id_is_not_universe_or_live():
    args = parse_args(["--symbols", "SPY"])
    assert args.ib_client_id == DEFAULT_CLIENT_ID == 8824
    assert args.symbols == "SPY"
    assert args.since == "2018-01-01"
    assert args.batch_days == 7


def test_client_id_override():
    args = parse_args(["--ib-client-id", "9001", "--symbols", "QQQ"])
    assert args.ib_client_id == 9001
    assert args.symbols == "QQQ"
