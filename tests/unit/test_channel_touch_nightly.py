"""CLI defaults for the channel-touch nightly scanner."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "scanners"))

from channel_touch_nightly import build_arg_parser  # noqa: E402
from utils.scanning.channel_touch import LIVE_DEFAULTS  # noqa: E402
from utils.scanning.channel_touch_feed_status import as_of_behind_last_close  # noqa: E402


def test_nightly_cli_defaults_match_h2_resist_break():
    ns = build_arg_parser().parse_args(["--skip-update", "--dry-run"])
    assert ns.entry_mode == LIVE_DEFAULTS["entry_mode"] == "l3_touch"
    assert ns.h2_resist_break is True
    assert ns.h2_resist_break_only is True
    assert ns.min_l3_wait_bars == LIVE_DEFAULTS["min_l3_wait_bars"] == 6
    assert ns.max_rsi is None
    assert ns.require_in_channel is False
    assert ns.max_channel_span_days == 365.0
    assert ns.max_beyond_width is None
    assert ns.atr_stop_mult == 2.0
    assert ns.max_entries_per_day == 0
    assert LIVE_DEFAULTS["max_entries_per_day"] == 0
    assert ns.window_bars == 504
    assert ns.window_step_bars == 252
    assert ns.skip_hot_dashboard is False
    assert ns.shakeout_breakout is True
    assert ns.shakeout_breakout_min_inside == 1
    assert LIVE_DEFAULTS["shakeout_breakout"] is True


def test_nightly_treats_prior_session_as_of_as_behind():
    from datetime import datetime
    from zoneinfo import ZoneInfo

    et = ZoneInfo("America/New_York")
    friday_1700 = datetime(2026, 9, 18, 17, 0, tzinfo=et)
    behind, got, expected = as_of_behind_last_close("2026-09-17", now=friday_1700)
    assert behind is True
    assert got.isoformat() == "2026-09-17"
    assert expected.isoformat() == "2026-09-18"
