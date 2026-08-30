"""CLI defaults for the channel-touch nightly scanner."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "scanners"))

from channel_touch_nightly import build_arg_parser  # noqa: E402
from utils.scanning.channel_touch import LIVE_DEFAULTS  # noqa: E402


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
