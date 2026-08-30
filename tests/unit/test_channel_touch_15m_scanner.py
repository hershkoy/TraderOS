"""CLI defaults for the 15m channel-touch scanner."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "scanners"))

from channel_touch_15m import build_arg_parser  # noqa: E402
from utils.scanning.channel_touch_15m import FROZEN_OVERSHOOT_MIN, LIVE_15M_DEFAULTS  # noqa: E402


def test_15m_cli_defaults_match_h5_stack():
    ns = build_arg_parser().parse_args(["--dry-run"])
    assert ns.mode == "run"
    assert ns.provider == "IB"
    assert ns.timeframe == "15m"
    assert ns.min_l3_wait_bars == LIVE_15M_DEFAULTS["min_l3_wait_bars"] == 12
    assert ns.max_channel_span_days == 10.0
    assert ns.volume_rel_min == 2.0
    assert ns.overshoot_min == FROZEN_OVERSHOOT_MIN
    assert ns.proximity_below_pct == 0.0
    assert ns.lookback_sessions == 40
